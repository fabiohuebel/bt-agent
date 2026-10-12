import json
from unittest.mock import Mock

import pytest
import requests

from bt_agent.models import ChatModel
from bt_agent._engine.model import backend


def response():
    reply = Mock()
    reply.json.return_value = {"choices": [{"message": {"content": "ok"}}]}
    return reply


def test_public_client_does_not_snapshot_environment_or_write_legacy_log(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "_VLM_LOG_DIR", str(tmp_path / "legacy"))
    monkeypatch.delenv("LAM_VLM_LOG_DIR", raising=False)
    monkeypatch.setenv("VLLM_API_KEY", "dummy-environment-credential")
    monkeypatch.setenv("LAM_OTHER_SETTING", "private-environment-value")
    post = Mock(return_value=response())
    monkeypatch.setattr(requests, "post", post)
    model = ChatModel(base_url="https://offline.invalid/v1", model="fixture", api_key="dummy-key")
    assert model.query([{"role": "user", "content": "hello"}], max_tokens=1024) == "ok"
    assert not (tmp_path / "legacy").exists()
    assert not list(tmp_path.iterdir())
    assert post.call_args.kwargs["headers"]["Authorization"] == "Bearer dummy-key"


def test_explicit_exchange_recording_does_not_copy_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(backend, "_VLM_LOG_DIR", str(tmp_path / "legacy"))
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path / "explicit"))
    monkeypatch.setenv("VLLM_API_KEY", "dummy-environment-credential")
    monkeypatch.setattr(requests, "post", Mock(return_value=response()))
    model = ChatModel(base_url="https://offline.invalid/v1", model="fixture")
    assert model.query([{"role": "user", "content": "hello"}]) == "ok"
    assert not (tmp_path / "legacy").exists()
    contents = (tmp_path / "explicit/exchanges.jsonl").read_text()
    events = [json.loads(row) for row in contents.splitlines()]
    assert [row["kind"] for row in events] == ["request", "response"]
    assert "dummy-environment-credential" not in contents


@pytest.mark.parametrize(
    "exception", [requests.Timeout, requests.HTTPError, requests.exceptions.InvalidURL]
)
def test_transport_error_does_not_copy_private_details_to_logs_or_caller(
    tmp_path, monkeypatch, caplog, exception
):
    monkeypatch.setattr(backend, "_VLM_LOG_DIR", str(tmp_path / "legacy"))
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path / "explicit"))
    error_response = requests.Response()
    error_response.status_code = 429
    error = exception("https://dummy-user:dummy-password@offline.invalid", response=error_response)
    monkeypatch.setattr(requests, "post", Mock(side_effect=error))
    model = ChatModel(base_url="https://offline.invalid/v1", model="fixture")
    with pytest.raises(requests.RequestException) as failure:
        model.query([{"role": "user", "content": "hello"}])
    assert "dummy-password" not in str(failure.value)
    assert failure.value.error_type == exception.__name__
    assert failure.value.status_code == 429
    assert failure.value.__context__ is None
    assert failure.value.request is None and failure.value.response is None
    assert "dummy-password" not in caplog.text
    contents = (tmp_path / "explicit/exchanges.jsonl").read_text()
    assert "dummy-password" not in contents
    assert json.loads(contents.splitlines()[-1])["error"]["status"] == 429


@pytest.mark.parametrize("method", ["query", "query_structured", "query_tool_call"])
def test_public_and_preserved_clients_send_identical_requests(tmp_path, monkeypatch, method):
    monkeypatch.setattr(backend, "_VLM_LOG_DIR", str(tmp_path / "legacy"))
    monkeypatch.delenv("LAM_VLM_LOG_DIR", raising=False)
    post = Mock(return_value=response())
    post.return_value.json.return_value = {
        "choices": [
            {
                "message": {
                    "content": '{"visible":true}',
                    "tool_calls": [
                        {"function": {"name": "report", "arguments": '{"text":"seen"}'}}
                    ],
                }
            }
        ]
    }
    monkeypatch.setattr(requests, "post", post)
    public = ChatModel(base_url="https://offline.invalid/v1", model="fixture")
    preserved = backend.ModelBackend(
        {
            "endpoint": "https://offline.invalid/v1",
            "model_name": "fixture",
            "api_key": "EMPTY",
            "seed": 0,
            "temperature": 0.0,
            "enable_thinking": False,
            "max_tokens": 4096,
            "timeout": 120,
        }
    )
    messages = [{"role": "user", "content": "hello"}]
    extra = (
        []
        if method == "query"
        else (
            [{"type": "object"}]
            if method == "query_structured"
            else [[{"type": "function", "function": {"name": "report"}}]]
        )
    )
    options = dict(max_tokens=1024, temperature=0, images=[b"fixture image bytes"])
    expected = getattr(preserved, method)(messages, *extra, **options)
    assert getattr(public, method)(messages, *extra, **options) == expected
    assert post.call_args_list[0] == post.call_args_list[1]
