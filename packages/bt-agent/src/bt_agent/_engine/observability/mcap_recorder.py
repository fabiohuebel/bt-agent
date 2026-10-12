"""MCAP recorder — write Foxglove-compatible MCAP files.

Foxglove Studio replays MCAP natively (``File > Open Local File``), so this
is the "recording" half of the live/replay split. Same channel set as the
WS server.

Optional dep: ``pip install mcap``. If the import fails, ``McapRecorder``
constructs but logs a warning and silently does nothing — the harness can
still run with WS + JSONL.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from bt_agent._engine.observability.channels import CHANNELS
from bt_agent._engine.observability.event_sink import EventSink, SinkEvent
from bt_agent._engine.observability.jsonl_recorder import _safe_json

logger = logging.getLogger(__name__)

try:
    from mcap.writer import Writer as _McapWriter

    _HAVE_MCAP = True
except ImportError:  # noqa: F401
    _McapWriter = None  # type: ignore[assignment]
    _HAVE_MCAP = False


class McapRecorder:
    """Subscribe to an EventSink and write a Foxglove-compatible MCAP file."""

    def __init__(self, path: Path) -> None:
        self._path = Path(path)
        self._enabled = _HAVE_MCAP
        self._lock = threading.Lock()
        self._unsub: Optional[Any] = None
        self._writer: Any = None
        self._fh: Any = None
        self._channel_ids: Dict[str, int] = {}

        if not self._enabled:
            logger.warning(
                "MCAP recording requested but `mcap` package is not installed. "
                "Install via `pip install mcap` to enable."
            )
            return

        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._fh = self._path.open("wb")
        self._writer = _McapWriter(self._fh)
        self._writer.start()
        # Pre-register all channels with their JSON schemas.
        for ch in CHANNELS:
            schema_id = self._writer.register_schema(
                name=ch["schema_name"],
                encoding="jsonschema",
                data=json.dumps(ch["schema"]).encode("utf-8"),
            )
            ch_id = self._writer.register_channel(
                topic=ch["topic"],
                message_encoding="json",
                schema_id=schema_id,
            )
            self._channel_ids[ch["topic"]] = ch_id
        logger.info("MCAP recording to %s", self._path)

    def attach(self, sink: EventSink) -> None:
        if not self._enabled:
            return
        self._unsub = sink.subscribe(self._on_event, name=f"mcap[{self._path.name}]")

    def _on_event(self, ev: SinkEvent) -> None:
        if not self._enabled:
            return
        if ev.channel not in self._channel_ids:
            return  # silently ignore non-/agent/ events (these go to JSONL only)
        ch_id = self._channel_ids[ev.channel]
        ts_ns = int(ev.ts * 1e9)
        try:
            data = json.dumps(_safe_json(ev.payload), ensure_ascii=False).encode("utf-8")
        except Exception:  # noqa: BLE001
            logger.exception("MCAP serialization failed for channel %s", ev.channel)
            return
        with self._lock:
            self._writer.add_message(
                channel_id=ch_id,
                log_time=ts_ns,
                publish_time=ts_ns,
                data=data,
            )

    def close(self) -> None:
        if not self._enabled:
            return
        if self._unsub is not None:
            self._unsub()
            self._unsub = None
        with self._lock:
            try:
                if self._writer is not None:
                    self._writer.finish()
            except Exception:  # noqa: BLE001
                logger.exception("MCAP finish failed")
            with contextlib.suppress(Exception):
                if self._fh is not None:
                    self._fh.close()
