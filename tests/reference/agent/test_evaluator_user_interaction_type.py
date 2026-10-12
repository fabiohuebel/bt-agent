"""Fix 2 regression: the evaluator user template must surface the milestone's
interaction_type, and evaluate_completion() must interpolate it (falling back
to "unspecified" for milestones that predate the field)."""

from __future__ import annotations

from unittest.mock import MagicMock

from bt_agent._engine.agent.lam_agent.evaluator import evaluate_completion
from bt_agent._engine.agent.system_prompts import get_prompts


def test_released_evaluator_user_has_interaction_type_placeholder() -> None:
    assert "{interaction_type}" in get_prompts("tiamat_agent_submit_v5").evaluator_user


def _model_capturing_user_content() -> MagicMock:
    model = MagicMock()
    model.query_structured.return_value = {
        "skip_inspection": True,
        "verdict": "pass",
        "reasoning": "ok",
        "correction": None,
    }
    return model


def test_evaluate_completion_interpolates_milestone_interaction_type() -> None:
    model = _model_capturing_user_content()
    evaluate_completion(
        model,
        milestone={"id": "m1", "description": "x", "verify_condition": "y", "interaction_type": "press"},
        skill_observation="",
    )
    # Every model.query_structured call receives the same rendered user_content.
    user_content = model.query_structured.call_args_list[0][0][0][1]["content"]
    assert "press" in user_content


def test_evaluate_completion_defaults_interaction_type_to_unspecified_when_missing() -> None:
    model = _model_capturing_user_content()
    evaluate_completion(
        model,
        milestone={"id": "m1", "description": "x", "verify_condition": "y"},
        skill_observation="",
    )
    user_content = model.query_structured.call_args_list[0][0][0][1]["content"]
    assert "unspecified" in user_content
