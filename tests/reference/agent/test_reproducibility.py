"""Tests for the reproducibility / variance-reduction patches.

Covers:
  * ``ModelBackend`` forwards ``seed`` into every payload when set, and
    omits the key otherwise.
  * Agent default temperatures are 0.0 (greedy) everywhere except
    on the replan / amend paths, which expose ``replan_temperature``
    and ``amend_temperature`` respectively.
  * Actor's amend-detection helper fires iff and only iff the most
    recent user message in history starts with the EVALUATOR CORRECTION
    prefix.
"""

from __future__ import annotations

import json
from typing import Any, Dict, List
from unittest.mock import Mock

import pytest
from bt_agent._engine.agent._internals_config import CORRECTION_PREFIX
from bt_agent._engine.agent.config import AgentConfig
from bt_agent._engine.agent.lam_agent.actor import _history_has_amend_correction
from bt_agent._engine.model.backend import ModelBackend


def _backend_with_recorded_post(seed: Any = None) -> ModelBackend:
    """Build a ModelBackend with ``_post_chat`` mocked out for inspection."""
    config: Dict[str, Any] = {
        "endpoint": "http://localhost:8000/v1",
        "model_name": "test-model",
        "api_format": "openai",
        "max_tokens": 100,
        "temperature": 0.0,
    }
    if seed is not None:
        config["seed"] = seed
    backend = ModelBackend(config)
    backend._post_chat = Mock(
        return_value={
            "choices": [{"message": {"content": "ok", "role": "assistant"}, "finish_reason": "stop"}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0},
        }
    )
    return backend


def test_backend_includes_seed_when_set():
    backend = _backend_with_recorded_post(seed=42)
    backend.query([{"role": "user", "content": "hi"}])
    assert backend._post_chat.call_args[0][0]["seed"] == 42


def test_backend_omits_seed_when_not_set():
    backend = _backend_with_recorded_post(seed=None)
    backend.query([{"role": "user", "content": "hi"}])
    assert "seed" not in backend._post_chat.call_args[0][0]


def test_backend_query_structured_includes_seed():
    backend = _backend_with_recorded_post(seed=7)
    backend._post_chat = Mock(
        return_value={"choices": [{"message": {"content": json.dumps({"x": 1}), "role": "assistant"}}]}
    )
    backend.query_structured(
        [{"role": "user", "content": "hi"}],
        {"type": "object", "properties": {"x": {"type": "integer"}}},
    )
    payload = backend._post_chat.call_args[0][0]
    assert payload["seed"] == 7
    assert payload["response_format"]["type"] == "json_schema"


def test_backend_query_tool_call_includes_seed():
    backend = _backend_with_recorded_post(seed=13)
    backend._post_chat = Mock(
        return_value={
            "choices": [{
                "message": {
                    "content": "",
                    "tool_calls": [{
                        "id": "x",
                        "type": "function",
                        "function": {"name": "f", "arguments": "{}"},
                    }],
                },
                "finish_reason": "tool_calls",
            }],
        }
    )
    backend.query_tool_call([{"role": "user", "content": "hi"}], tools=[])
    assert backend._post_chat.call_args[0][0]["seed"] == 13


def test_default_temperatures_are_greedy():
    cfg = AgentConfig()
    assert cfg.planner.synthesis_temperature == 0.0
    assert cfg.actor.temperature == 0.0
    assert cfg.evaluator.verdict_temperature == 0.0
    assert cfg.evaluator.evidence_temperature == 0.0
    assert cfg.model.temperature == 0.0


def test_default_replan_and_amend_temperatures_are_higher():
    cfg = AgentConfig()
    assert cfg.planner.replan_temperature == pytest.approx(0.7)
    assert cfg.actor.amend_temperature == pytest.approx(0.7)
    assert cfg.planner.replan_temperature > cfg.planner.synthesis_temperature
    assert cfg.actor.amend_temperature > cfg.actor.temperature


def test_default_seed_is_none():
    assert AgentConfig().model.seed is None


def test_config_overrides_via_from_dict():
    cfg = AgentConfig.from_dict({
        "planner": {"synthesis_temperature": 0.2, "replan_temperature": 0.9},
        "actor": {"temperature": 0.1, "amend_temperature": 0.5},
        "evaluator": {"verdict_temperature": 0.05},
        "model": {"temperature": 0.3, "seed": 99},
    })
    assert cfg.planner.synthesis_temperature == 0.2
    assert cfg.planner.replan_temperature == 0.9
    assert cfg.actor.temperature == 0.1
    assert cfg.actor.amend_temperature == 0.5
    assert cfg.evaluator.verdict_temperature == 0.05
    assert cfg.model.temperature == 0.3
    assert cfg.model.seed == 99


def test_amend_detection_empty_history():
    assert _history_has_amend_correction([]) is False


def test_amend_detection_no_correction_present():
    history: List[Dict[str, Any]] = [
        {"role": "assistant", "content": "thought"},
        {"role": "user", "content": "continue"},
        {"role": "tool", "content": "result"},
    ]
    assert _history_has_amend_correction(history) is False


def test_amend_detection_recent_correction_present():
    history: List[Dict[str, Any]] = [
        {"role": "user", "content": "first user msg"},
        {"role": "assistant", "content": "thought"},
        {"role": "user", "content": CORRECTION_PREFIX + "use the other handle"},
    ]
    assert _history_has_amend_correction(history) is True


def test_amend_detection_only_most_recent_user_counts():
    """Older correction superseded by a newer non-correction user msg → don't fire."""
    history: List[Dict[str, Any]] = [
        {"role": "user", "content": CORRECTION_PREFIX + "old correction"},
        {"role": "assistant", "content": "thought"},
        {"role": "tool", "content": "result"},
        {"role": "user", "content": "next thing"},
    ]
    assert _history_has_amend_correction(history) is False


def test_amend_detection_walks_past_assistant_and_tool():
    history: List[Dict[str, Any]] = [
        {"role": "user", "content": CORRECTION_PREFIX + "fix the handle"},
        {"role": "assistant", "content": "thought"},
        {"role": "tool", "content": "tool out"},
    ]
    assert _history_has_amend_correction(history) is True
