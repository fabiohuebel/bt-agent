"""Full model-exchange recorder (physics-trace-recording wave 2).

Covers: request/response pairing (success, error, timeout, retry-attempt
numbering), image blob dedupe + ordered refs, credential scrubbing, and the
env-unset byte-identity guarantee (no exchanges.jsonl, no blobs/).
"""

from __future__ import annotations

import base64
import json
import os
from pathlib import Path

import requests
from bt_agent._engine.model import exchange_recorder as rec

_JPEG_A = b"\xff\xd8\xff\xe0fake-jpeg-bytes-a"
_JPEG_B = b"\xff\xd8\xff\xe0fake-jpeg-bytes-b"


def _data_url(raw: bytes, mime: str = "image/jpeg") -> str:
    return f"data:{mime};base64,{base64.b64encode(raw).decode('ascii')}"


def _payload(image: bytes | None = None) -> dict:
    messages = [
        {"role": "system", "content": "you are a robot"},
        {"role": "user", "content": "what do you see"},
    ]
    if image is not None:
        messages[-1] = {
            "role": "user",
            "content": [
                {"type": "text", "text": "what do you see"},
                {"type": "image_url", "image_url": {"url": _data_url(image)}},
            ],
        }
    return {
        "model": "qwen35-vlm",
        "messages": messages,
        "max_tokens": 512,
        "temperature": 0.0,
        "tools": [{"type": "function", "function": {"name": "navigate"}}],
        "tool_choice": "auto",
    }


def _read_lines(log_dir: Path) -> list[dict]:
    path = log_dir / "exchanges.jsonl"
    return [json.loads(line) for line in path.read_text().splitlines()]


# ---------------------------------------------------------------------------
# Request/response pairing
# ---------------------------------------------------------------------------


def test_request_written_before_response_and_pair_by_exchange_id(tmp_path, monkeypatch):
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))

    handle = rec.begin_exchange(phase="actor", payload=_payload())
    # The request line must exist BEFORE the response is journaled.
    lines_after_request = _read_lines(tmp_path)
    assert len(lines_after_request) == 1
    assert lines_after_request[0]["kind"] == "request"
    assert lines_after_request[0]["exchange_id"] == handle.exchange_id
    assert lines_after_request[0]["phase"] == "actor"
    assert lines_after_request[0]["seq"] == handle.seq

    rec.end_exchange(handle, response={"choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}]})
    lines = _read_lines(tmp_path)
    assert len(lines) == 2
    assert lines[1]["kind"] == "response"
    assert lines[1]["exchange_id"] == handle.exchange_id
    assert lines[1]["seq"] == handle.seq
    assert lines[1]["completeness"] == "complete"
    assert lines[1]["content"] == "ok"
    assert lines[1]["finish_reason"] == "stop"
    assert lines[1]["t_end"] >= lines[1]["t_start"]


def test_seq_is_monotonic_across_calls(tmp_path, monkeypatch):
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))
    h1 = rec.begin_exchange(phase="planner", payload=_payload())
    h2 = rec.begin_exchange(phase="planner", payload=_payload())
    assert h2.seq > h1.seq
    assert h1.exchange_id != h2.exchange_id


def test_episode_context_attached_when_given(tmp_path, monkeypatch):
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))
    ctx = {"run_id": "r1", "task_id": "t1"}
    rec.begin_exchange(phase="actor", payload=_payload(), episode_context=ctx)
    lines = _read_lines(tmp_path)
    assert lines[0]["episode_context"] == ctx


# ---------------------------------------------------------------------------
# Error / timeout / retry paths — ALL journaled
# ---------------------------------------------------------------------------


def test_transport_error_is_journaled_with_completeness_error(tmp_path, monkeypatch):
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))
    handle = rec.begin_exchange(phase="actor", payload=_payload())

    error = requests.exceptions.ConnectionError("connection refused")
    rec.end_exchange(handle, error=rec.describe_error(error))

    lines = _read_lines(tmp_path)
    assert lines[1]["completeness"] == "error"
    assert lines[1]["error"]["type"] == "ConnectionError"
    assert "connection refused" in lines[1]["error"]["message"]
    assert lines[1]["error"]["status"] is None


def test_timeout_is_journaled_as_a_distinct_error_type(tmp_path, monkeypatch):
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))
    handle = rec.begin_exchange(phase="scene_graph", payload=_payload())

    rec.end_exchange(handle, error=rec.describe_error(requests.exceptions.Timeout("read timed out")))

    lines = _read_lines(tmp_path)
    assert lines[1]["error"]["type"] == "Timeout"


def test_http_error_status_code_is_captured(tmp_path, monkeypatch):
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))
    handle = rec.begin_exchange(phase="actor", payload=_payload())

    response = requests.Response()
    response.status_code = 500
    http_error = requests.exceptions.HTTPError("server error", response=response)
    rec.end_exchange(handle, error=rec.describe_error(http_error))

    lines = _read_lines(tmp_path)
    assert lines[1]["error"]["status"] == 500


def test_retry_attempt_numbers_are_journaled_per_call(tmp_path, monkeypatch):
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))
    payload = _payload()

    h0 = rec.begin_exchange(phase="scene_graph", payload=payload, retry_attempt=0)
    rec.end_exchange(h0, error=rec.describe_error(requests.exceptions.Timeout("timed out")))
    h1 = rec.begin_exchange(phase="scene_graph", payload=payload, retry_attempt=1)
    rec.end_exchange(h1, response={"choices": [{"message": {"content": "ok"}}]})

    lines = _read_lines(tmp_path)
    requests_only = [line for line in lines if line["kind"] == "request"]
    assert [r["retry_attempt"] for r in requests_only] == [0, 1]
    # A pending call (request with no matching response) must still be a
    # legitimate, readable entry — never a partial/corrupt line.
    assert all(json.dumps(line) for line in lines)


def test_pending_request_with_no_response_is_still_readable(tmp_path, monkeypatch):
    """Simulated crash mid-call: request written, process dies before end_exchange."""
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))
    rec.begin_exchange(phase="actor", payload=_payload())

    lines = _read_lines(tmp_path)
    assert len(lines) == 1
    assert lines[0]["kind"] == "request"


# ---------------------------------------------------------------------------
# Image blobs: content-addressed, deduped, ordered refs
# ---------------------------------------------------------------------------


def test_image_is_extracted_to_a_sha256_blob_and_referenced(tmp_path, monkeypatch):
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))
    rec.begin_exchange(phase="capture", payload=_payload(image=_JPEG_A))

    lines = _read_lines(tmp_path)
    refs = lines[0]["image_refs"]
    assert len(refs) == 1
    sha = refs[0]["sha256"]
    assert refs[0]["mime"] == "image/jpeg"
    assert refs[0]["bytes"] == len(_JPEG_A)
    blob_path = Path(refs[0]["blob_path"])
    assert blob_path.exists()
    assert blob_path.read_bytes() == _JPEG_A
    assert blob_path.parent == tmp_path / "blobs"
    assert sha == blob_path.stem
    # The message part itself no longer carries the base64 payload.
    logged_content = lines[0]["messages"][-1]["content"]
    image_part = next(p for p in logged_content if p["type"] == "image_ref")
    assert image_part["ref"]["sha256"] == sha


def test_duplicate_image_across_calls_is_written_once(tmp_path, monkeypatch):
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))
    rec.begin_exchange(phase="capture", payload=_payload(image=_JPEG_A))
    rec.begin_exchange(phase="capture", payload=_payload(image=_JPEG_A))

    blobs = list((tmp_path / "blobs").iterdir())
    assert len(blobs) == 1

    lines = _read_lines(tmp_path)
    shas = {line["image_refs"][0]["sha256"] for line in lines}
    assert len(shas) == 1


def test_different_images_produce_distinct_blobs(tmp_path, monkeypatch):
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))
    rec.begin_exchange(phase="capture", payload=_payload(image=_JPEG_A))
    rec.begin_exchange(phase="capture", payload=_payload(image=_JPEG_B))

    blobs = list((tmp_path / "blobs").iterdir())
    assert len(blobs) == 2


def test_non_data_url_image_marks_request_incomplete_with_limitation(tmp_path, monkeypatch):
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))
    payload = _payload()
    payload["messages"][-1] = {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": "https://example.com/cat.jpg"}}],
    }

    rec.begin_exchange(phase="capture", payload=payload)

    lines = _read_lines(tmp_path)
    assert lines[0]["completeness"] == "incomplete"
    assert lines[0]["limitations"]
    assert not (tmp_path / "blobs").exists() or not list((tmp_path / "blobs").iterdir())


def test_malformed_base64_image_marks_incomplete_not_crashes(tmp_path, monkeypatch):
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))
    payload = _payload()
    payload["messages"][-1] = {
        "role": "user",
        "content": [{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,not-valid-base64!!!"}}],
    }

    handle = rec.begin_exchange(phase="capture", payload=payload)  # must not raise
    assert handle.exchange_id

    lines = _read_lines(tmp_path)
    assert lines[0]["completeness"] == "incomplete"
    assert lines[0]["limitations"]


# ---------------------------------------------------------------------------
# Credential scrubbing
# ---------------------------------------------------------------------------


def test_credential_looking_keys_are_scrubbed_from_response(tmp_path, monkeypatch):
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))
    handle = rec.begin_exchange(phase="actor", payload=_payload())

    rec.end_exchange(
        handle,
        response={
            "choices": [{"message": {"content": "ok"}}],
            "api_key": "sk-should-never-be-written",
            "server_meta": {"Authorization": "Bearer super-secret-token"},
        },
    )

    raw_text = (tmp_path / "exchanges.jsonl").read_text()
    assert "sk-should-never-be-written" not in raw_text
    assert "super-secret-token" not in raw_text
    lines = _read_lines(tmp_path)
    assert lines[1]["response"]["api_key"] == "<scrubbed>"
    assert lines[1]["response"]["server_meta"]["Authorization"] == "<scrubbed>"


def test_credential_looking_keys_are_scrubbed_from_error(tmp_path, monkeypatch):
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))
    handle = rec.begin_exchange(phase="actor", payload=_payload())

    rec.end_exchange(handle, error={"type": "HTTPError", "status": 401, "message": "bad", "api_key": "sk-leak"})

    raw_text = (tmp_path / "exchanges.jsonl").read_text()
    assert "sk-leak" not in raw_text


def test_token_count_fields_survive_scrubbing_verbatim(tmp_path, monkeypatch):
    """Regression: "token" is a substring of every usage/generation-setting
    field name below. A naive substring-based credential scrub redacted all
    of them to "<scrubbed>", destroying usage data in every journal line.
    """
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))
    payload = _payload()
    payload["max_completion_tokens"] = 256
    handle = rec.begin_exchange(phase="actor", payload=payload)

    rec.end_exchange(
        handle,
        response={
            "choices": [{"message": {"content": "ok"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15},
        },
    )

    lines = _read_lines(tmp_path)
    assert lines[0]["generation_settings"]["max_tokens"] == 512
    assert lines[1]["usage"] == {"prompt_tokens": 10, "completion_tokens": 5, "total_tokens": 15}


def test_credential_keys_scrubbed_by_exact_name_not_substring(tmp_path, monkeypatch):
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))
    handle = rec.begin_exchange(phase="actor", payload=_payload())

    rec.end_exchange(
        handle,
        response={
            "choices": [{"message": {"content": "ok"}}],
            "api_key": "sk-leak",
            "Authorization": "Bearer sk-leak-2",
            "x-api-key": "sk-leak-3",
            "access_token": "sk-leak-4",
            "secret_key": "sk-leak-5",
            "password": "hunter2",
        },
    )

    raw_text = (tmp_path / "exchanges.jsonl").read_text()
    for leaked in ("sk-leak", "sk-leak-2", "sk-leak-3", "sk-leak-4", "sk-leak-5", "hunter2"):
        assert leaked not in raw_text
    response = _read_lines(tmp_path)[1]["response"]
    for k in ("api_key", "Authorization", "x-api-key", "access_token", "secret_key", "password"):
        assert response[k] == "<scrubbed>"


# ---------------------------------------------------------------------------
# Env-unset byte identity — the hard rule
# ---------------------------------------------------------------------------


def test_env_unset_writes_nothing_and_stays_cheap(tmp_path, monkeypatch):
    monkeypatch.delenv("LAM_VLM_LOG_DIR", raising=False)
    monkeypatch.chdir(tmp_path)

    handle = rec.begin_exchange(phase="actor", payload=_payload(image=_JPEG_A))
    rec.end_exchange(handle, response={"choices": [{"message": {"content": "ok"}}]})

    # No new artifacts anywhere a deployed agent might look.
    assert not (tmp_path / "exchanges.jsonl").exists()
    assert not (tmp_path / "blobs").exists()
    assert not any(tmp_path.iterdir())
    # The handle is still real and usable (skill_call_id linkage must not
    # depend on recording being enabled).
    assert handle.exchange_id
    assert handle.seq > 0


def test_env_unset_is_read_fresh_not_cached(tmp_path, monkeypatch):
    """Toggling the env var mid-process must be honored on the next call."""
    monkeypatch.delenv("LAM_VLM_LOG_DIR", raising=False)
    h1 = rec.begin_exchange(phase="actor", payload=_payload())
    assert not h1._recording

    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))
    h2 = rec.begin_exchange(phase="actor", payload=_payload())
    assert h2._recording
    assert (tmp_path / "exchanges.jsonl").exists()


def test_env_unset_never_hashes_the_request(monkeypatch):
    """The no-op path must not pay a serialize+hash cost, and a bug in the
    hashing helper must not be able to break the deployed (unset) call path.
    """
    monkeypatch.delenv("LAM_VLM_LOG_DIR", raising=False)

    def _boom(payload):
        raise AssertionError("_hash_request must not be called when recording is off")

    monkeypatch.setattr(rec, "_hash_request", _boom)

    handle = rec.begin_exchange(phase="actor", payload=_payload())
    assert handle.exchange_id
    assert handle.request_hash is None


def test_hashing_failure_still_writes_the_request_with_a_limitation(tmp_path, monkeypatch):
    """A hash failure (non-serializable content, RecursionError, ...) must
    degrade to request_hash=None + a limitation entry — never propagate out
    of begin_exchange and fail a good model call.
    """
    monkeypatch.setenv("LAM_VLM_LOG_DIR", str(tmp_path))

    def _boom(payload):
        raise RecursionError("maximum recursion depth exceeded")

    monkeypatch.setattr(rec, "_hash_request", _boom)

    handle = rec.begin_exchange(phase="actor", payload=_payload())  # must not raise
    assert handle.exchange_id
    assert handle.request_hash is None

    lines = _read_lines(tmp_path)
    assert len(lines) == 1
    assert lines[0]["kind"] == "request"
    assert lines[0]["request_hash"] is None
    assert lines[0]["completeness"] == "incomplete"
    assert any("hash" in limitation for limitation in lines[0]["limitations"])


def test_recording_gate_is_independent_of_legacy_vlm_log_dir_default(tmp_path, monkeypatch):
    """LAM_VLM_LOG_DIR unset still leaves the legacy default constant set.

    ``backend._VLM_LOG_DIR`` always resolves to a directory (default
    ``/tmp/lam_vlm_log``) so vlm_calls.jsonl keeps writing there — but that
    must never leak into whether exchanges.jsonl/blobs get created.
    """
    monkeypatch.delenv("LAM_VLM_LOG_DIR", raising=False)
    assert os.environ.get("LAM_VLM_LOG_DIR") is None
    handle = rec.begin_exchange(phase="actor", payload=_payload())
    assert handle._log_dir is None
    assert handle._recording is False
