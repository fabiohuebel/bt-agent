"""JSONL recorder — append every event as one JSON line.

Grep-friendly debug log; complementary to MCAP. No external deps.
"""

from __future__ import annotations

import contextlib
import json
import logging
import threading
from pathlib import Path
from typing import Any, Optional

from bt_agent._engine.observability import EventSink, SinkEvent

logger = logging.getLogger(__name__)


def _safe_json(obj: Any) -> Any:
    """Recursively coerce an object into something json.dumps can handle."""
    if isinstance(obj, (str, int, float, bool)) or obj is None:
        return obj
    if isinstance(obj, dict):
        return {str(k): _safe_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_safe_json(v) for v in obj]
    return str(obj)


class JsonlRecorder:
    """Append SinkEvent records to a JSONL file."""

    def __init__(self, path: Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self._path = path
        self._fh = path.open("a", encoding="utf-8")
        self._lock = threading.Lock()
        self._unsub: Optional[Any] = None
        logger.info("JSONL recording to %s", path)

    def attach(self, sink: EventSink) -> None:
        self._unsub = sink.subscribe(self._on_event, name=f"jsonl[{self._path.name}]")

    def _on_event(self, ev: SinkEvent) -> None:
        line = json.dumps(
            {
                "ts": ev.ts,
                "channel": ev.channel,
                "payload": _safe_json(ev.payload),
            },
            ensure_ascii=False,
        )
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
