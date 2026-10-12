import json
import threading
import queue
from functools import partial

import pytest

from bt_agent.tracing import EventSink
from bt_agent._engine.observability.jsonl_recorder import JsonlRecorder
from bt_agent._engine.observability.timeline_recorder import TimelineRecorder


@pytest.mark.parametrize("close_sink", [False, True])
def test_shutdown_drains_all_accepted_callbacks_in_order(close_sink):
    sink = EventSink(max_subscriber_queue=3)
    started, release = threading.Event(), threading.Event()
    received, errors = [], []

    def callback(event):
        if event.payload["i"] == 0:
            started.set()
            assert release.wait(2)
        received.append(event.payload["i"])

    unsubscribe = sink.subscribe(callback)
    sink.emit("test", {"i": 0})
    assert started.wait(1)
    for i in range(1, 4):
        sink.emit("test", {"i": i})
    subscriber = sink._subscribers[0]

    def close():
        try:
            (sink.close if close_sink else unsubscribe)(timeout=2)
        except Exception as error:
            errors.append(error)

    closer = threading.Thread(target=close)
    closer.start()
    assert subscriber._stop.wait(1)
    release.set()
    closer.join(2)
    assert not closer.is_alive() and not errors
    assert received == list(range(4))
    sink.close()
    assert sink.statistics()[0]["pending"] == 0


@pytest.mark.parametrize("recorder_type", [JsonlRecorder, TimelineRecorder])
def test_immediate_recorder_close_flushes_every_accepted_record(tmp_path, recorder_type):
    sink = EventSink(max_subscriber_queue=200)
    path = tmp_path / "events.jsonl"
    recorder = recorder_type(path)
    recorder.attach(sink)
    for i in range(100):
        sink.emit("test", {"i": i})
    recorder.close()
    recorder.close()
    sink.close()
    assert [json.loads(line)["payload"]["i"] for line in path.read_text().splitlines()] == list(
        range(100)
    )


def test_close_timeout_is_explicit_and_retry_drains_remaining_events():
    sink = EventSink()
    started, release = threading.Event(), threading.Event()
    received = []

    def callback(event):
        started.set()
        release.wait(2)
        received.append(event.payload["i"])

    sink.subscribe(callback)
    sink.emit("test", {"i": 0})
    assert started.wait(1)
    sink.emit("test", {"i": 1})
    try:
        with pytest.raises(TimeoutError, match="still draining"):
            sink.close(timeout=0.01)
        assert sink.statistics()[0]["pending"] == 2
        with pytest.raises(RuntimeError, match="closed"):
            sink.subscribe(lambda _: None)
        sink.emit("test", {"i": 2})
    finally:
        release.set()
        sink.close(timeout=2)
    assert received == [0, 1]
    sink.close()


def test_overflow_and_callback_errors_remain_visible():
    sink = EventSink(max_subscriber_queue=1)
    started, release = threading.Event(), threading.Event()

    def callback(event):
        started.set()
        release.wait(2)
        raise ValueError("fixture failure")

    sink.subscribe(callback)
    sink.emit("test", {})
    assert started.wait(1)
    sink.emit("test", {})
    sink.emit("test", {})
    release.set()
    sink.close(timeout=2)
    stats = sink.statistics()[0]
    assert stats["accepted"] == 2 and stats["callback_errors"] == 2
    assert stats["overflow_dropped"] == 1 and stats["pending"] == 0


def test_self_close_fails_explicitly_without_stopping_other_subscribers():
    sink = EventSink()
    rejected = threading.Event()

    def callback(event):
        with pytest.raises(RuntimeError, match="outside"):
            sink.close()
        rejected.set()

    sink.subscribe(callback)
    sink.emit("test", {})
    assert rejected.wait(1)
    sink.close()


@pytest.mark.parametrize("timeout", [-1, float("inf"), float("nan")])
def test_invalid_shutdown_timeout_is_rejected(timeout):
    sink = EventSink()
    with pytest.raises(ValueError, match="finite"):
        sink.close(timeout=timeout)
    sink.close()


def test_empty_poll_racing_last_emit_and_close_does_not_drop_events(monkeypatch):
    from bt_agent import tracing

    empty_seen, resume = threading.Event(), threading.Event()

    class ControlledQueue(queue.Queue):
        first = True

        def get(self, *args, **kwargs):
            if self.first:
                self.first = False
                empty_seen.set()
                assert resume.wait(2)
                raise queue.Empty
            return super().get(*args, **kwargs)

    monkeypatch.setattr(tracing.queue, "Queue", ControlledQueue)
    sink, received, errors = EventSink(), [], []
    sink.subscribe(lambda event: received.append(event.payload))
    assert empty_seen.wait(1)
    subscriber = sink._subscribers[0]
    sink.emit("test", {"last": True})

    def close():
        try:
            sink.close(timeout=2)
        except Exception as error:
            errors.append(error)

    closer = threading.Thread(target=close)
    closer.start()
    assert subscriber._stop.wait(1)
    resume.set()
    closer.join(2)
    assert not closer.is_alive() and not errors
    assert received == [{"last": True}]


@pytest.mark.parametrize("recorder_type", [JsonlRecorder, TimelineRecorder])
def test_recorder_does_not_close_file_if_callback_has_not_stopped(tmp_path, recorder_type):
    started, release = threading.Event(), threading.Event()
    sink = EventSink()
    recorder = recorder_type(tmp_path / "events.jsonl")
    original_callback = recorder._on_event

    def blocked(event):
        started.set()
        assert release.wait(2)
        original_callback(event)

    recorder._on_event = blocked
    recorder.attach(sink)
    recorder._unsub = partial(recorder._unsub, timeout=0.01)
    sink.emit("test", {"saved": True})
    assert started.wait(1)
    try:
        with pytest.raises(TimeoutError):
            recorder.close()
        assert not recorder._fh.closed
        assert recorder._unsub is not None
    finally:
        release.set()
        sink.close(timeout=2)
        recorder.close()
    assert json.loads((tmp_path / "events.jsonl").read_text())["payload"] == {"saved": True}
