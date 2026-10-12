"""ModelBackend._post_chat wiring to the wave-2 exchange recorder.

Covers: phase threading from query()/query_structured()/query_tool_call(),
last_exchange_id exposure (the actor's skill_selected linkage), request-
before-network-I/O ordering, transport-error journaling, and the
LAM_VLM_LOG_DIR-unset byte-identity guarantee for the deployed agent.
"""

from __future__ import annotations

import json
from unittest.mock import Mock, patch

import pytest
import requests
from bt_agent._engine.model import backend as backend_module
from bt_agent._engine.model.backend import ModelBackend


def _backend(tmp_path=None, monkeypatch=None) -> ModelBackend:
    if tmp_path is not None and monkeypatch is not None:
        monkeypatch.setattr(backend_module, "_VLM_LOG_DIR", str(tmp_path))
    with patch("bt_agent._engine.model.backend._write_run_start"):
        return ModelBackend({"endpoint": "http://model.test:8000/v1", "model_name": "qwen35-vlm"})


def _mock_response(payload: dict) -> Mock:
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.json.return_value = payload
    return resp


def _chat_response(content: str = "ok", tool_calls=None) -> dict:
    message = {"content": content}
    if tool_calls is not None:
        message["tool_calls"] = tool_calls
    return {"choices": [{"message": message, "finish_reason": "stop"}]}


def _exchange_lines(log_dir) -> list[dict]:
    path = log_dir / "exchanges.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text().splitlines()]


# ---------------------------------------------------------------------------
# Phase threading
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "call",
    [
        lambda b: b.query([{"role": "user", "content": "hi"}], phase="planner"),
        lambda b: b.query_structured([{"role": "user", "content": "hi"}], {}, phase="evaluator"),
    ],
)
def test_query_methods_thread_phase_into_the_exchange_journal(tmp_path, monkeypatch, call):
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))
    backend = _backend()
    with patch("requests.post", return_value=_mock_response(_chat_response('{"a": 1}'))):
        call(backend)

    lines = _exchange_lines(tmp_path)
    request_lines = [line for line in lines if line["kind"] == "request"]
    assert len(request_lines) == 1
    assert request_lines[0]["phase"] in {"planner", "evaluator"}


def test_query_tool_call_threads_phase_capture(tmp_path, monkeypatch):
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))
    backend = _backend()
    tool_calls = [{"function": {"name": "capture", "arguments": "{}"}}]
    with patch("requests.post", return_value=_mock_response(_chat_response(tool_calls=tool_calls))):
        backend.query_tool_call(
            [{"role": "user", "content": "hi"}],
            [{"type": "function", "function": {"name": "capture"}}],
            phase="actor",
        )

    lines = _exchange_lines(tmp_path)
    assert lines[0]["phase"] == "actor"


def test_phase_defaults_to_other_when_not_passed(tmp_path, monkeypatch):
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))
    backend = _backend()
    with patch("requests.post", return_value=_mock_response(_chat_response())):
        backend.query([{"role": "user", "content": "hi"}])

    lines = _exchange_lines(tmp_path)
    assert lines[0]["phase"] == "other"


# ---------------------------------------------------------------------------
# last_exchange_id — the actor's SELECT_SKILL linkage seam
# ---------------------------------------------------------------------------


def test_last_exchange_id_is_none_before_any_call(tmp_path, monkeypatch):
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))
    backend = _backend()
    assert backend.last_exchange_id is None


def test_last_exchange_id_updates_after_each_call_and_matches_the_journal(tmp_path, monkeypatch):
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))
    backend = _backend()
    with patch("requests.post", return_value=_mock_response(_chat_response())):
        backend.query([{"role": "user", "content": "hi"}], phase="actor")
        first_id = backend.last_exchange_id
        backend.query([{"role": "user", "content": "hi again"}], phase="actor")
        second_id = backend.last_exchange_id

    assert first_id != second_id
    lines = _exchange_lines(tmp_path)
    request_ids = [line["exchange_id"] for line in lines if line["kind"] == "request"]
    assert request_ids == [first_id, second_id]


def test_last_exchange_id_is_set_even_when_recording_is_disabled(tmp_path, monkeypatch):
    """The actor must be able to link decisions to exchanges regardless of
    whether LAM_VLM_LOG_DIR happens to be set — recording is a separate
    concern from exchange identity."""
    monkeypatch.delenv("LAM_VLM_LOG_DIR", raising=False)
    backend = _backend(tmp_path, monkeypatch)
    with patch("requests.post", return_value=_mock_response(_chat_response())):
        backend.query([{"role": "user", "content": "hi"}])
    assert backend.last_exchange_id is not None


# ---------------------------------------------------------------------------
# Request journaled BEFORE network I/O; response after, success or not
# ---------------------------------------------------------------------------


def test_request_is_journaled_before_the_network_call(tmp_path, monkeypatch):
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))
    backend = _backend()
    seen_before_post = {}

    def _fake_post(*args, **kwargs):
        seen_before_post["lines"] = _exchange_lines(tmp_path)
        return _mock_response(_chat_response())

    with patch("requests.post", side_effect=_fake_post):
        backend.query([{"role": "user", "content": "hi"}], phase="actor")

    assert len(seen_before_post["lines"]) == 1
    assert seen_before_post["lines"][0]["kind"] == "request"


def test_transport_error_is_journaled_and_still_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))
    backend = _backend()
    with (
        patch("requests.post", side_effect=requests.exceptions.ConnectTimeout("no route")),
        pytest.raises(requests.exceptions.ConnectTimeout),
    ):
        backend.query([{"role": "user", "content": "hi"}], phase="actor")

    lines = _exchange_lines(tmp_path)
    assert lines[-1]["kind"] == "response"
    assert lines[-1]["completeness"] == "error"
    assert lines[-1]["error"]["type"] == "ConnectTimeout"


def test_malformed_json_body_is_journaled_and_still_raises(tmp_path, monkeypatch):
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))
    backend = _backend()
    resp = Mock()
    resp.raise_for_status = Mock()
    resp.json.side_effect = ValueError("not json")
    with patch("requests.post", return_value=resp), pytest.raises(ValueError):
        backend.query([{"role": "user", "content": "hi"}], phase="actor")

    lines = _exchange_lines(tmp_path)
    assert lines[-1]["completeness"] == "error"


# ---------------------------------------------------------------------------
# Never record Authorization headers or API keys
# ---------------------------------------------------------------------------


def test_authorization_header_never_reaches_the_journal(tmp_path, monkeypatch):
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))
    with patch("bt_agent._engine.model.backend._write_run_start"):
        backend = ModelBackend({
            "endpoint": "http://model.test:8000/v1",
            "model_name": "qwen35-vlm",
            "api_key": "sk-do-not-leak",
        })
    with patch("requests.post", return_value=_mock_response(_chat_response())):
        backend.query([{"role": "user", "content": "hi"}], phase="actor")

    raw_text = (tmp_path / "exchanges.jsonl").read_text()
    assert "sk-do-not-leak" not in raw_text


# ---------------------------------------------------------------------------
# HARD RULE: deployed behavior byte-identical when LAM_VLM_LOG_DIR is unset
# ---------------------------------------------------------------------------


def test_env_unset_produces_no_exchanges_jsonl_or_blobs(tmp_path, monkeypatch):
    monkeypatch.delenv("LAM_VLM_LOG_DIR", raising=False)
    monkeypatch.chdir(tmp_path)
    backend = _backend(tmp_path / "legacy_vlm_dir", monkeypatch)  # legacy vlm_calls.jsonl dir only
    with patch("requests.post", return_value=_mock_response(_chat_response())):
        backend.query([{"role": "user", "content": "hi"}], phase="actor")

    assert not (tmp_path / "exchanges.jsonl").exists()
    assert not (tmp_path / "blobs").exists()


def test_env_unset_leaves_vlm_calls_jsonl_and_tags_unchanged(tmp_path, monkeypatch):
    """The legacy vlm_calls.jsonl writer + its tag shape are wave-2-invisible."""
    monkeypatch.delenv("LAM_VLM_LOG_DIR", raising=False)
    backend = _backend(tmp_path, monkeypatch)
    with patch("requests.post", return_value=_mock_response(_chat_response())):
        backend.query([{"role": "user", "content": "hi"}], phase="actor")

    calls_path = tmp_path / "vlm_calls.jsonl"
    lines = calls_path.read_text().splitlines()
    call_record = json.loads(lines[-1])
    assert "episode_context" not in call_record
    assert set(call_record) == {"call_id", "ts", "run_id", "agent_tier", "task_id", "request", "response"}
