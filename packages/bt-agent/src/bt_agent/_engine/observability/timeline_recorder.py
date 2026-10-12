"""TimelineRecorder — channel-filtered JSONL sink for run-level analytics.

A thin variant of :class:`JsonlRecorder` that only subscribes to a chosen
set of channels. Use this to write a compact, paper-analysis-grade record
of the run (event lifecycle, odom samples, etc.) without the per-tick
firehose that the live-viz path emits.

Two recorder instances are typically created per run:

- ``timeline.jsonl`` — event-driven channels (``/agent/event_log``,
  ``/agent/phase``, ``/agent/skill``, ``/agent/milestones``).
- ``odom_5hz.jsonl`` — sample-driven channel (``/sensor/odom``).

Both files land in ``LAM_CAPTURE_DIR`` and get aggregated into
``<run_dir>/captures/`` by the overnight harness alongside
``nav_debug.jsonl``.

Run fields (``run_id``, ``agent_tier``, ``task_id``) and the active
experiment's identity stamps are written on every line for trivially cheap
filtering / joining downstream.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
from pathlib import Path
from typing import Any, Callable, Dict, Iterable, Optional

from bt_agent._engine.observability import EventSink, SinkEvent
from bt_agent._engine.observability.jsonl_recorder import _safe_json

logger = logging.getLogger(__name__)


class TimelineRecorder:
    """Append events from a chosen subset of EventSink channels to a JSONL file.

    Parameters
    ----------
    path:
        Absolute or relative file path. Parent dirs are created.
    channels:
        Iterable of channel topics to record. Events on other topics are
        ignored. If empty / None, all events are recorded (firehose).
    on_change_only:
        Iterable of channel topics for which to dedupe — write only when
        the payload changes from the previous write on that channel.
        Useful for ``/agent/phase`` etc. that may emit unchanged values
        many times when the live-viz pipeline is also active.
    tag_provider:
        Optional callable returning a dict of fields to inject onto every
        line (e.g., ``{"run_id": ..., "agent_tier": ..., "task_id": ...}``).
        Called once per event.
    """

    def __init__(
        self,
        path: Path,
        channels: Optional[Iterable[str]] = None,
        on_change_only: Optional[Iterable[str]] = None,
        tag_provider: Optional[Callable[[], Dict[str, Any]]] = None,
    ) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._fh = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._unsub: Optional[Callable[[], None]] = None
        self._channels = set(channels) if channels else None
        self._on_change_only = set(on_change_only) if on_change_only else set()
        self._tag_provider = tag_provider
        # For dedupe: cache the last serialized payload per channel.
        self._last_payload_json: Dict[str, str] = {}
        logger.info(
            "TimelineRecorder writing to %s (channels=%s)",
            path,
            "all" if self._channels is None else sorted(self._channels),
        )

    def attach(self, sink: EventSink) -> None:
        self._unsub = sink.subscribe(self._on_event, name=f"timeline[{self._path.name}]")

    def _on_event(self, ev: SinkEvent) -> None:
        if self._channels is not None and ev.channel not in self._channels:
            return

        payload_safe = _safe_json(ev.payload)
        payload_json = json.dumps(payload_safe, ensure_ascii=False, sort_keys=True)

        if ev.channel in self._on_change_only:
            if self._last_payload_json.get(ev.channel) == payload_json:
                return
            self._last_payload_json[ev.channel] = payload_json

        line_obj: Dict[str, Any] = {"ts": ev.ts, "channel": ev.channel}
        if self._tag_provider is not None:
            try:
                tags = self._tag_provider() or {}
                if tags:
                    line_obj.update(tags)
            except Exception:  # noqa: BLE001 — never let tagging crash the writer
                logger.debug("tag_provider raised", exc_info=True)
        line_obj["payload"] = payload_safe

        line = json.dumps(line_obj, ensure_ascii=False, default=str)
        with self._lock:
            self._fh.write(line)
            self._fh.write("\n")
            self._fh.flush()

    def close(self) -> None:
        if self._unsub is not None:
            self._unsub()
            self._unsub = None
        with self._lock, contextlib.suppress(Exception):
            self._fh.close()
