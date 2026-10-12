"""Tests for the EventSink broker."""

from __future__ import annotations

import threading
import time

from bt_agent._engine.observability import EventSink


def _wait_for(predicate, timeout: float = 1.0) -> bool:
    """Spin until predicate returns truthy or timeout. Useful for thread tests."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_emit_appends_to_ring() -> None:
    sink = EventSink(ring_size=5)
    try:
        for i in range(3):
            sink.emit("ch", {"i": i})
        events = sink.tail()
        assert [e.payload["i"] for e in events] == [0, 1, 2]
    finally:
        sink.close()


def test_ring_evicts_oldest() -> None:
    sink = EventSink(ring_size=3)
    try:
        for i in range(5):
            sink.emit("ch", {"i": i})
        events = sink.tail()
        assert [e.payload["i"] for e in events] == [2, 3, 4]
    finally:
        sink.close()


def test_subscriber_receives_events() -> None:
    sink = EventSink()
    received = []
    lock = threading.Lock()

    def cb(ev):
        with lock:
            received.append(ev.payload["i"])

    unsub = sink.subscribe(cb, name="t")
    try:
        for i in range(5):
            sink.emit("ch", {"i": i})
        assert _wait_for(lambda: len(received) == 5), f"only got {received}"
        assert received == [0, 1, 2, 3, 4]
    finally:
        unsub()
        sink.close()


def test_unsubscribe_stops_delivery() -> None:
    sink = EventSink()
    received = []

    def cb(ev):
        received.append(ev.payload["i"])

    unsub = sink.subscribe(cb, name="t")
    try:
        sink.emit("ch", {"i": 0})
        assert _wait_for(lambda: len(received) == 1)
        unsub()
        sink.emit("ch", {"i": 1})
        # Give the (now-stopped) thread time to NOT receive the second event.
        time.sleep(0.05)
        assert received == [0]
    finally:
        sink.close()


def test_slow_subscriber_does_not_block_emit() -> None:
    sink = EventSink(max_subscriber_queue=2)
    received = []

    started = threading.Event()
    finish = threading.Event()

    def slow_cb(ev):
        started.set()
        finish.wait(timeout=2.0)
        received.append(ev.payload["i"])

    unsub = sink.subscribe(slow_cb, name="slow")
    try:
        # Prime the slow worker so it's actively blocked in the callback.
        sink.emit("ch", {"i": 0})
        assert _wait_for(started.is_set, timeout=1.0), "slow callback never started"
        # Subsequent emits go into the bounded queue (max 2). After that,
        # drops should occur — but emit must not block.
        t0 = time.time()
        for i in range(1, 20):
            sink.emit("ch", {"i": i})
        elapsed = time.time() - t0
        assert elapsed < 0.5, f"emit was blocked for {elapsed}s"
    finally:
        finish.set()
        unsub()
        sink.close()


def test_multiple_subscribers_isolated() -> None:
    sink = EventSink()
    a, b = [], []
    unsub_a = sink.subscribe(lambda ev: a.append(ev.payload["i"]), name="a")
    unsub_b = sink.subscribe(lambda ev: b.append(ev.payload["i"]), name="b")
    try:
        for i in range(3):
            sink.emit("ch", {"i": i})
        assert _wait_for(lambda: len(a) == 3 and len(b) == 3)
        assert a == [0, 1, 2]
        assert b == [0, 1, 2]
    finally:
        unsub_a()
        unsub_b()
        sink.close()


def test_emit_after_close_is_silent() -> None:
    sink = EventSink()
    sink.close()
    sink.emit("ch", {"x": 1})  # must not raise
    assert len(sink.tail()) == 0


def test_subscriber_exception_does_not_crash_others() -> None:
    sink = EventSink()
    received = []

    def boom(ev):
        raise RuntimeError("kaboom")

    def good(ev):
        received.append(ev.payload["i"])

    unsub_bad = sink.subscribe(boom, name="bad")
    unsub_good = sink.subscribe(good, name="good")
    try:
        for i in range(3):
            sink.emit("ch", {"i": i})
        assert _wait_for(lambda: len(received) == 3)
        assert received == [0, 1, 2]
    finally:
        unsub_bad()
        unsub_good()
        sink.close()
