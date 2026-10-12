"""Foxglove WebSocket server — bridge EventSink to Foxglove Studio.

The harness embeds this in a daemon thread. Foxglove Studio (or any
foxglove-websocket client) connects to ``ws://<host>:<port>``, sees the
pre-registered channels with JSON Schemas, and subscribes to whichever
channels the user's panels need.

Optional dep: ``pip install foxglove-websocket``. If missing, this module
prints an actionable error and the harness continues with stdout / JSONL
only.

Threading model
---------------
The Foxglove WS lib is asyncio. We run a private event loop in a daemon
thread. The synchronous tick thread emits via ``EventSink``; our subscriber
callback runs on the sink's own subscriber thread and hands each event off
to the WS loop with ``asyncio.run_coroutine_threadsafe``.
"""

from __future__ import annotations

import asyncio
import json
import logging
import threading
from typing import Any, Dict, Optional

from bt_agent._engine.observability.channels import CHANNELS
from bt_agent._engine.observability.event_sink import EventSink, SinkEvent
from bt_agent._engine.observability.jsonl_recorder import _safe_json

logger = logging.getLogger(__name__)

try:
    from foxglove_websocket.server import FoxgloveServer

    _HAVE_FOXGLOVE = True
except ImportError:
    FoxgloveServer = None  # type: ignore[assignment]
    _HAVE_FOXGLOVE = False


class FoxgloveWsServer:
    """Run a Foxglove WebSocket server in a background thread.

    Usage::

        server = FoxgloveWsServer(host="127.0.0.1", port=8765)
        server.start()
        server.attach(event_sink)
        # ... agent runs ...
        server.close()
    """

    def __init__(self, host: str = "127.0.0.1", port: int = 8765) -> None:
        self._host = host
        self._port = port
        self._enabled = _HAVE_FOXGLOVE
        self._loop: Optional[asyncio.AbstractEventLoop] = None
        self._thread: Optional[threading.Thread] = None
        self._server: Any = None
        self._channel_ids: Dict[str, int] = {}
        self._unsub: Optional[Any] = None
        self._ready = threading.Event()

        if not self._enabled:
            logger.warning(
                "Foxglove WS requested but `foxglove-websocket` is not installed. "
                "Install via `pip install foxglove-websocket` to enable."
            )

    @property
    def url(self) -> str:
        return f"ws://{self._host}:{self._port}"

    # -- Lifecycle ----------------------------------------------------------

    def start(self) -> None:
        if not self._enabled:
            return
        self._thread = threading.Thread(target=self._run_loop, name="fg-ws", daemon=True)
        self._thread.start()
        # Wait briefly for server readiness so attach() can publish without races.
        self._ready.wait(timeout=5.0)

    def attach(self, sink: EventSink) -> None:
        if not self._enabled:
            return
        # Replay buffered tail so a Studio connecting after the agent started
        # still gets the recent state.
        for ev in sink.tail():
            self._on_event(ev)
        self._unsub = sink.subscribe(self._on_event, name=f"fg-ws[{self._port}]")

    def close(self) -> None:
        if not self._enabled:
            return
        if self._unsub is not None:
            self._unsub()
            self._unsub = None
        loop = self._loop
        if loop is not None and loop.is_running():
            asyncio.run_coroutine_threadsafe(self._shutdown(), loop)
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    # -- Internal -----------------------------------------------------------

    def _run_loop(self) -> None:
        loop = asyncio.new_event_loop()
        self._loop = loop
        asyncio.set_event_loop(loop)
        try:
            loop.run_until_complete(self._serve())
        finally:
            loop.close()

    async def _serve(self) -> None:
        async with FoxgloveServer(self._host, self._port, "lam_v2") as server:
            self._server = server
            for ch in CHANNELS:
                ch_id = await server.add_channel({
                    "topic": ch["topic"],
                    "encoding": ch["encoding"],
                    "schemaName": ch["schema_name"],
                    "schema": json.dumps(ch["schema"]),
                })
                self._channel_ids[ch["topic"]] = ch_id
            logger.info("Foxglove WS listening on ws://%s:%d", self._host, self._port)
            self._ready.set()
            # Block forever until close() is signaled.
            self._stop_evt = asyncio.Event()
            await self._stop_evt.wait()

    async def _shutdown(self) -> None:
        if hasattr(self, "_stop_evt"):
            self._stop_evt.set()

    def _on_event(self, ev: SinkEvent) -> None:
        if not self._enabled or self._server is None:
            return
        if ev.channel not in self._channel_ids:
            return  # only /agent/* topics flow to Foxglove
        loop = self._loop
        if loop is None or not loop.is_running():
            return
        try:
            data = json.dumps(_safe_json(ev.payload), ensure_ascii=False).encode("utf-8")
        except Exception:  # noqa: BLE001
            logger.exception("FG serialization failed for channel %s", ev.channel)
            return
        ch_id = self._channel_ids[ev.channel]
        ts_ns = int(ev.ts * 1e9)
        asyncio.run_coroutine_threadsafe(
            self._server.send_message(ch_id, ts_ns, data),
            loop,
        )
