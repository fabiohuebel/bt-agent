from __future__ import annotations

import json

from bt_agent._engine.model import backend as backend_module
from bt_agent._engine.model.backend import ModelBackend
from bt_agent._engine.observability import EventSink, TimelineRecorder

IDENTITY_STAMPS = {
    "condition_id": "condition-123",
    "profile_id": "flat",
    "resolved_config_hash": "config-456",
    "git_sha": "c6f4c8939",
}


def _backend_config(model_name: str = "qwen35-vlm") -> dict:
    return {
        "endpoint": "http://model.test:8000/v1",
        "model_name": model_name,
        "agent_tier": "lam",
    }


def test_backend_run_start_and_all_runtime_tags_include_identity_stamps(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(backend_module, "_VLM_LOG_DIR", str(tmp_path))

    backend = ModelBackend(_backend_config(), identity_tags=IDENTITY_STAMPS)

    record = json.loads((tmp_path / "vlm_calls.jsonl").read_text().splitlines()[0])
    assert record["kind"] == "run_start"
    assert {key: record[key] for key in IDENTITY_STAMPS} == IDENTITY_STAMPS
    assert {key: backend._tags()[key] for key in IDENTITY_STAMPS} == IDENTITY_STAMPS


def test_timeline_and_odometry_records_include_identity_stamps(tmp_path, monkeypatch) -> None:
    monkeypatch.setattr(backend_module, "_VLM_LOG_DIR", str(tmp_path / "vlm"))
    backend = ModelBackend(_backend_config(), identity_tags=IDENTITY_STAMPS)
    sink = EventSink()
    timeline_path = tmp_path / "timeline.jsonl"
    odom_path = tmp_path / "odom_5hz.jsonl"
    timeline = TimelineRecorder(
        timeline_path,
        channels={"/agent/event_log"},
        tag_provider=backend._tags,
    )
    odometry = TimelineRecorder(
        odom_path,
        channels={"/sensor/odom"},
        tag_provider=backend._tags,
    )
    timeline.attach(sink)
    odometry.attach(sink)

    sink.emit("/agent/event_log", {"kind": "task_received"})
    sink.emit("/sensor/odom", {"x": 1.0, "y": 2.0})
    timeline.close()
    odometry.close()

    for path in (timeline_path, odom_path):
        record = json.loads(path.read_text().splitlines()[0])
        assert {key: record[key] for key in IDENTITY_STAMPS} == IDENTITY_STAMPS


def test_clean_restart_constructs_fresh_run_id_for_model_change(
    tmp_path,
    monkeypatch,
) -> None:
    monkeypatch.setattr(backend_module, "_VLM_LOG_DIR", str(tmp_path))

    first = ModelBackend(_backend_config("qwen35-vlm"), identity_tags=IDENTITY_STAMPS)
    second = ModelBackend(_backend_config("qwen36-vlm"), identity_tags=IDENTITY_STAMPS)

    assert first is not second
    assert first.model_name != second.model_name
    assert first._run_id != second._run_id
