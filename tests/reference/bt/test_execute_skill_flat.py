"""Failure-presentation tests at the Actor ingress seam."""

from __future__ import annotations

import pytest
from bt_agent._engine.agent.blackboard_keys import BB
from bt_agent._engine.bt.nodes.execute_skill import ExecuteSkill
from bt_agent._engine.bt.nodes.select_skill import actor_history_observation
from bt_agent._engine.skills.protocol import SkillHandle, SkillStatus


def _handle(
    status: SkillStatus,
    observation: str,
    result_data: dict | None = None,
) -> SkillHandle:
    handle = SkillHandle()
    handle.status = status
    handle.observation = observation
    handle.result_data = result_data or {}
    return handle


def _execute_node(handle: SkillHandle) -> ExecuteSkill:
    node = ExecuteSkill()
    node._skill_name = "navigate"
    node._handle = handle
    return node


@pytest.mark.parametrize("legacy_value", [None, "typed", "flat", "bogus"])
def test_execute_skill_is_always_typed_below_actor_seam(
    monkeypatch: pytest.MonkeyPatch,
    legacy_value: str | None,
) -> None:
    if legacy_value is None:
        monkeypatch.delenv("LAM_RESULT_STYLE", raising=False)
    else:
        monkeypatch.setenv("LAM_RESULT_STYLE", legacy_value)
    handle = _handle(
        SkillStatus.FAILED,
        "Navigation blocked.",
        {"arrival_status": "failed_blocked", "failure_reason": "stale_pose"},
    )
    node = _execute_node(handle)

    node._emit_completion("failed")

    assert handle.observation.startswith("Navigation blocked.")
    assert "odometry" in handle.observation
    assert handle.result_data == {
        "arrival_status": "failed_blocked",
        "failure_reason": "stale_pose",
    }
    event = node.blackboard.get(BB.PENDING_EVENTS)[-1]
    assert event["data"]["observation"] == handle.observation
    assert event["data"]["arrival_status"] == "failed_blocked"
    assert event["data"]["failure_reason"] == "stale_pose"
    assert "raw_observation" not in event["data"]


def test_flat_changes_only_registered_failed_skill_actor_text() -> None:
    registry = {"navigate": object()}
    handle = _handle(
        SkillStatus.FAILED,
        "Navigation blocked. Try another route.",
        {"arrival_status": "failed_blocked", "diagnostic": "edge-7"},
    )
    original_result_data = handle.result_data

    typed = actor_history_observation("navigate", handle, registry, "typed")
    flat = actor_history_observation("navigate", handle, registry, "flat")

    assert typed == "Navigation blocked. Try another route."
    assert flat == "failure of skill navigate"
    assert handle.observation == typed
    assert handle.status == SkillStatus.FAILED
    assert handle.result_data == original_result_data


def test_success_actor_text_is_byte_identical_between_presentations() -> None:
    registry = {"navigate": object()}
    observation = "Arrived short of 'mug' (0.90m from goal)."
    handle = _handle(SkillStatus.SUCCEEDED, observation, {"arrival_status": "arrived_short"})

    typed = actor_history_observation("navigate", handle, registry, "typed")
    flat = actor_history_observation("navigate", handle, registry, "flat")

    assert flat == typed == observation


def test_unknown_or_unparseable_tool_call_is_never_flattened() -> None:
    handle = _handle(SkillStatus.FAILED, "Unknown skill: hallucinated_tool")

    assert actor_history_observation("hallucinated_tool", handle, {"navigate": object()}, "flat") == handle.observation


def test_invalid_failure_presentation_refuses_ingress() -> None:
    handle = _handle(SkillStatus.FAILED, "Navigation blocked.")

    with pytest.raises(ValueError, match="Invalid failure presentation"):
        actor_history_observation("navigate", handle, {"navigate": object()}, "verbose")
