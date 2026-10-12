"""Tests for query_structured salvage + content-None fallback (backend.py).

Grounded in the 78-episode LAM sweep failure analysis:
  - 296x 'Unterminated string starting at line 3 column 16' (truncated JSON)
  - 51x content-is-None (qwen3 thinking mode routes text to reasoning_content)

FIX 1: query_structured must (a) fall back to reasoning_content when the
message content is None, and (b) salvage-parse truncated JSON before raising.
Genuinely broken JSON must still raise.
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pytest
from bt_agent._engine.model.backend import (
    ModelBackend,
    _loads_salvage_json,
    _repair_truncated_json,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _backend() -> ModelBackend:
    """Construct a backend without touching disk (run_start is stubbed)."""
    with patch("bt_agent._engine.model.backend._write_run_start"):
        return ModelBackend({"endpoint": "http://x/v1", "model_name": "m"})


def _resp(message: dict) -> dict:
    return {"choices": [{"message": message}]}


# ---------------------------------------------------------------------------
# (b) salvage: truncated JSON shapes
# ---------------------------------------------------------------------------


def test_salvage_unterminated_string():
    # The 296x sweep error: a string cut mid-value.
    broken = '{"verdict": "amend", "reasoning": "the mug is on the coun'
    out = _loads_salvage_json(broken)
    assert out["verdict"] == "amend"
    assert out["reasoning"].startswith("the mug is on the coun")


def test_salvage_unclosed_brace():
    broken = '{"skip_inspection": true, "inspect_handles": ["h1"]'
    out = _loads_salvage_json(broken)
    assert out["skip_inspection"] is True
    assert out["inspect_handles"] == ["h1"]


def test_salvage_trailing_comma():
    broken = '{"skip_inspection": true,'
    out = _loads_salvage_json(broken)
    assert out == {"skip_inspection": True}


def test_repair_is_idempotent_on_valid_json():
    good = '{"a": 1, "b": [2, 3]}'
    # Valid input is returned unchanged by the repair pass...
    assert json.loads(_repair_truncated_json(good)) == {"a": 1, "b": [2, 3]}
    # ...and salvage returns it via the fast path.
    assert _loads_salvage_json(good) == {"a": 1, "b": [2, 3]}


# ---------------------------------------------------------------------------
# (c) genuinely-broken JSON still raises
# ---------------------------------------------------------------------------


def test_genuinely_broken_json_still_raises():
    # Not truncation — structurally invalid; repair cannot rescue it.
    with pytest.raises(json.JSONDecodeError):
        _loads_salvage_json("this is not json at all {]")


def test_none_content_with_no_fallback_raises():
    # No string content anywhere → original TypeError propagates.
    with pytest.raises((TypeError, json.JSONDecodeError)):
        _loads_salvage_json(None)


# ---------------------------------------------------------------------------
# (a) content-None fallback to reasoning_content (via query_structured)
# ---------------------------------------------------------------------------


def test_query_structured_falls_back_to_reasoning_content():
    backend = _backend()
    payload_json = '{"skip_inspection": false, "inspect_handles": ["h2"]}'
    resp = _resp({"content": None, "reasoning_content": payload_json})
    with patch.object(backend, "_post_chat", return_value=resp):
        out = backend.query_structured([{"role": "user", "content": "x"}], {})
    assert out == {"skip_inspection": False, "inspect_handles": ["h2"]}


def test_query_structured_prefers_content_when_present():
    backend = _backend()
    resp = _resp({"content": '{"verdict": "pass"}', "reasoning_content": '{"verdict": "amend"}'})
    with patch.object(backend, "_post_chat", return_value=resp):
        out = backend.query_structured([{"role": "user", "content": "x"}], {})
    assert out == {"verdict": "pass"}


def test_query_structured_salvages_truncated_reasoning_content():
    # content=None AND the reasoning_content payload is truncated — both
    # fixes on the same call, which is the real verdict-path failure mode.
    backend = _backend()
    truncated = '{"verdict": "amend", "reasoning": "cut off here'
    resp = _resp({"content": None, "reasoning_content": truncated})
    with patch.object(backend, "_post_chat", return_value=resp):
        out = backend.query_structured([{"role": "user", "content": "x"}], {})
    assert out["verdict"] == "amend"


@pytest.mark.parametrize("payload", [["x"], "x", 7])
def test_query_structured_rejects_non_object_json(payload):
    backend = _backend()
    resp = _resp({"content": json.dumps(payload)})

    with patch.object(backend, "_post_chat", return_value=resp), pytest.raises(ValueError, match="JSON object"):
        backend.query_structured([{"role": "user", "content": "x"}], {})


# ---------------------------------------------------------------------------
# VLLM_MODE transport presets: timeout is a (connect, read) tuple
# ---------------------------------------------------------------------------
#
# Serving topology is a declaration (VLLM_MODE), not inferred from the URL —
# the ssh-tunnel fallback makes a remote server look like localhost. local:
# connect is instant-or-dead (2 s). remote: tunnel grace (10 s). Read stays
# the model-speed knob (VLLM_TIMEOUT_S / config "timeout").


def test_timeout_defaults_to_remote_preset(monkeypatch):
    monkeypatch.delenv("VLLM_MODE", raising=False)
    monkeypatch.delenv("VLLM_CONNECT_TIMEOUT_S", raising=False)
    assert _backend().timeout == (10, 120)


def test_timeout_local_mode_preset(monkeypatch):
    monkeypatch.setenv("VLLM_MODE", "local")
    monkeypatch.delenv("VLLM_CONNECT_TIMEOUT_S", raising=False)
    assert _backend().timeout == (2, 120)


def test_connect_timeout_env_overrides_mode_preset(monkeypatch):
    monkeypatch.setenv("VLLM_MODE", "local")
    monkeypatch.setenv("VLLM_CONNECT_TIMEOUT_S", "5")
    assert _backend().timeout == (5, 120)


def test_read_timeout_still_governed_by_config(monkeypatch):
    monkeypatch.delenv("VLLM_MODE", raising=False)
    monkeypatch.delenv("VLLM_CONNECT_TIMEOUT_S", raising=False)
    with patch("bt_agent._engine.model.backend._write_run_start"):
        backend = ModelBackend({"endpoint": "http://x/v1", "model_name": "m", "timeout": 60})
    assert backend.timeout == (10, 60)


def test_unknown_mode_falls_back_to_remote_preset(monkeypatch):
    monkeypatch.setenv("VLLM_MODE", "weird")
    monkeypatch.delenv("VLLM_CONNECT_TIMEOUT_S", raising=False)
    assert _backend().timeout == (10, 120)
