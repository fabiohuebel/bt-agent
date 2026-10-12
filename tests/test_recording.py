import hashlib
import json
import math
from pathlib import Path
import threading

import pytest

from bt_agent.recording import RecordedModel
from bt_agent import create_agent
from bt_agent.episode import EpisodeHost
from bt_agent.skills import ReportSkill
from bt_agent.testing import InspectShelf, shelf_model


def test_exact_arguments_and_images_are_forwarded_and_recorded(tmp_path):
    calls = []

    class Model:
        model_name = "fixture"

        def query_structured(self, *args, **kwargs):
            calls.append((args, kwargs))
            return {"visible": True}

    messages, schema = [{"role": "user", "content": "fixture"}], {"type": "object"}
    pixels = b"exact supplied bytes"
    model = RecordedModel(Model(), tmp_path / "model")
    result = model.query_structured(
        messages, schema, images=[pixels], phase="capture", max_tokens=1024, temperature=0
    )
    assert result == {"visible": True}
    assert calls[0][0][0] is messages and calls[0][0][1] is schema
    assert calls[0][1]["images"][0] == pixels
    record = model.records()[0]
    assert (model.output / record["images"][0]["path"]).read_bytes() == pixels
    assert record["images"][0]["sha256"] == hashlib.sha256(pixels).hexdigest()
    assert model.model_name == "fixture"
    model.close()
    events = [json.loads(line) for line in (model.output / "calls.jsonl").read_text().splitlines()]
    assert [e["kind"] for e in events] == ["model_call_start", "model_call_end"]


@pytest.mark.parametrize(
    "error, expected",
    [(ValueError("malformed reply"), None), (OSError("private url"), "model_error")],
)
def test_response_errors_are_distinct_from_transport_errors(tmp_path, error, expected):
    class Model:
        def query(self, *args, **kwargs):
            raise error

    recorded = RecordedModel(Model(), tmp_path / "model")
    with pytest.raises(type(error)):
        recorded.query([])
    assert recorded.failure == expected
    recorded.close()
    assert "private url" not in (recorded.output / "calls.jsonl").read_text()


def test_pending_call_prevents_silent_close(tmp_path):
    entered, release = threading.Event(), threading.Event()

    class Model:
        def query(self, *args, **kwargs):
            entered.set()
            assert release.wait(3)
            return "done"

    recorded = RecordedModel(Model(), tmp_path / "model")
    thread = threading.Thread(target=lambda: recorded.query([]))
    thread.start()
    try:
        assert entered.wait(2)
        with pytest.raises(RuntimeError, match="pending"):
            recorded.close()
    finally:
        release.set()
        thread.join(2)
    recorded.close()
    assert recorded.records()[0]["status"] == "complete"


def test_refuses_existing_evidence(tmp_path):
    with pytest.raises(FileExistsError):
        RecordedModel(object(), tmp_path)


@pytest.mark.parametrize("exception", [KeyboardInterrupt, SystemExit])
def test_interrupted_model_call_keeps_a_terminal_record(tmp_path, exception):
    class Model:
        def query(self, *args, **kwargs):
            raise exception("private interrupt details")

    recorded = RecordedModel(Model(), tmp_path / "model")
    with pytest.raises(exception):
        recorded.query([])
    recorded.close()
    assert recorded.failure == "model_interrupted"
    assert recorded.records()[0]["status"] == "interrupted"
    contents = (recorded.output / "calls.jsonl").read_text()
    assert "private interrupt details" not in contents
    events = [json.loads(line) for line in contents.splitlines()]
    assert [row["kind"] for row in events] == ["model_call_start", "model_call_end"]
    assert events[-1]["call"]["error_type"] == f"builtins.{exception.__name__}"


def test_nonfinite_model_reply_is_recorded_but_not_rewritten(tmp_path):
    reply = {"bbox": [float("nan"), float("inf"), -float("inf"), 1.0]}

    class Model:
        def query_structured(self, *args, **kwargs):
            return reply

    recorded = RecordedModel(Model(), tmp_path / "model")
    assert recorded.query_structured([], {}) is reply
    assert math.isnan(reply["bbox"][0]) and recorded.failure is None
    recorded.close()
    events = [
        json.loads(line) for line in (recorded.output / "calls.jsonl").read_text().splitlines()
    ]
    assert events[-1]["call"]["response"]["bbox"][0] == {"nonfinite_float": "nan"}


def test_objects_snapshot_is_detached_and_resets():
    model = shelf_model()
    with create_agent(
        model=model,
        skills=[InspectShelf(), ReportSkill()],
        host=EpisodeHost(lambda _: None, lambda: None),
    ) as agent:
        assert agent.objects_snapshot() == ()
        assert agent.run("Inspect shelf").submitted
        snapshot = agent.objects_snapshot()
        assert len(snapshot) == 1
        snapshot[0].label = "caller changed its copy"
        assert agent.objects_snapshot()[0].label == "red mug on the shelf"
        agent.start("New task")
        assert agent.objects_snapshot() == ()
        agent.cancel()
    with pytest.raises(RuntimeError, match="closed"):
        agent.objects_snapshot()
