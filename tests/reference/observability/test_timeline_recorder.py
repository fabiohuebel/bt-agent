"""TimelineRecorder: line-atomic JSONL writes that survive close()
(physics-trace-recording, wave 1). Crash-safety of agent_traces/ depends on
every written line being complete and flushed before container teardown —
graceful drain (SIGINT/SIGTERM -> node.destroy_node()) gives ~30s, but a
`docker rm -f` after that must never find a torn line.
"""

from __future__ import annotations

import json
import threading
import time

from bt_agent._engine.observability import EventSink, TimelineRecorder


def _wait_for(predicate, timeout: float = 5.0) -> bool:
    """Spin until predicate returns truthy or timeout. Useful for thread tests."""
    deadline = time.time() + timeout
    while time.time() < deadline:
        if predicate():
            return True
        time.sleep(0.01)
    return False


def test_concurrent_emits_produce_complete_non_interleaved_lines(tmp_path) -> None:
    path = tmp_path / "timeline.jsonl"
    sink = EventSink()
    recorder = TimelineRecorder(path, channels={"/agent/event_log"})
    recorder.attach(sink)

    n_threads, n_each = 8, 25
    total = n_threads * n_each

    def _emit(worker_id: int) -> None:
        for i in range(n_each):
            sink.emit("/agent/event_log", {"worker": worker_id, "i": i})

    threads = [threading.Thread(target=_emit, args=(w,)) for w in range(n_threads)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    # The subscriber drains its queue on a background thread; poll instead of
    # relying solely on close()'s internal 1s join timeout, which can be
    # tight for 200 events under a loaded CI host.
    assert _wait_for(lambda: len(path.read_text().splitlines()) >= total)
    recorder.close()

    lines = path.read_text().splitlines()
    assert len(lines) == total
    seen = set()
    for line in lines:
        obj = json.loads(line)  # raises if a line is torn/partial/interleaved
        seen.add((obj["payload"]["worker"], obj["payload"]["i"]))
    assert len(seen) == n_threads * n_each


def test_data_is_flushed_and_readable_from_a_fresh_handle_after_close(tmp_path) -> None:
    path = tmp_path / "timeline.jsonl"
    sink = EventSink()
    recorder = TimelineRecorder(path, channels={"/agent/event_log"})
    recorder.attach(sink)

    sink.emit("/agent/event_log", {"kind": "task_received"})
    recorder.close()

    # A fresh file handle sees the write -- proves flush+close happened, not
    # just an internal buffer that a mid-drain kill would have discarded.
    with open(path) as f:
        record = json.loads(f.readline())
    assert record["payload"]["kind"] == "task_received"


def test_close_unsubscribes_so_late_emits_never_reach_the_closed_file(tmp_path) -> None:
    path = tmp_path / "timeline.jsonl"
    sink = EventSink()
    recorder = TimelineRecorder(path, channels={"/agent/event_log"})
    recorder.attach(sink)
    recorder.close()

    sink.emit("/agent/event_log", {"kind": "late"})  # should be a no-op post-close
    sink.close()

    assert path.read_text() == ""
