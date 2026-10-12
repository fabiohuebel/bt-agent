"""Tests for typed ``press_status`` propagation at the ExecuteSkill/actor boundary.

Mirror of ``test_execute_skill_arrival_status.py`` for PressSkill's own outcome
key (skills_contract_v0 press section, D8). Before this slice press failures
only reached the actor because the shared follow-path leaf *leaked*
``arrival_status`` into press's result_data — the three press_status codes
written by press's own leaves (``failed_invalid``, ``failed_no_movement``,
``failed_no_path``) were never read by ``_emit_completion``.
"""

from __future__ import annotations

from bt_agent._engine.agent.blackboard_keys import BB
from bt_agent._engine.bt.nodes.execute_skill import ExecuteSkill
from bt_agent._engine.skills.protocol import SkillHandle, SkillStatus


def _make_node() -> ExecuteSkill:
    node = ExecuteSkill()
    node._skill_name = "press"
    return node


def test_emit_completion_carries_press_status_in_event() -> None:
    node = _make_node()
    node._handle = SkillHandle()
    node._handle.status = SkillStatus.FAILED
    node._handle.observation = "Press blocked."
    node._handle.result_data = {"press_status": "failed_blocked"}

    node._emit_completion("failed")

    events = node.blackboard.get(BB.PENDING_EVENTS)
    assert events[-1]["kind"] == "skill_completed"
    assert events[-1]["data"]["press_status"] == "failed_blocked"
    assert events[-1]["data"]["arrival_status"] == ""


def test_emit_completion_appends_guidance_for_press_only_codes() -> None:
    """The regression this slice fixes: a press leaf that writes press_status
    with no arrival_status alongside it must still reach the agent's guidance."""
    node = _make_node()
    node._handle = SkillHandle()
    node._handle.status = SkillStatus.FAILED
    node._handle.observation = "[leaf=DriveOntoTarget] Press on 'panel' failed: robot never started moving."
    node._handle.result_data = {"press_status": "failed_no_movement"}

    node._emit_completion("failed")

    obs = node._handle.observation
    assert obs.startswith("[leaf=DriveOntoTarget] Press on 'panel' failed: robot never started moving.")
    assert "never started moving toward the goal" in obs


def test_emit_completion_appends_guidance_for_press_failed_invalid() -> None:
    node = _make_node()
    node._handle = SkillHandle()
    node._handle.status = SkillStatus.FAILED
    node._handle.observation = "Invalid goal coordinates for 'panel'."
    node._handle.result_data = {"press_status": "failed_invalid"}

    node._emit_completion("failed")

    assert "query the target again" in node._handle.observation


def test_emit_completion_pressed_success_leaves_observation_untouched() -> None:
    node = _make_node()
    node._handle = SkillHandle()
    node._handle.status = SkillStatus.SUCCEEDED
    node._handle.observation = "Stepped on 'panel'."
    node._handle.result_data = {"press_status": "pressed"}

    node._emit_completion("success")

    events = node.blackboard.get(BB.PENDING_EVENTS)
    assert node._handle.observation == "Stepped on 'panel'."
    assert events[-1]["data"]["press_status"] == "pressed"


def test_emit_completion_without_press_status_reports_empty_code() -> None:
    node = _make_node()
    node._handle = SkillHandle()
    node._handle.status = SkillStatus.SUCCEEDED
    node._handle.observation = "Found 2 objects."
    node._handle.result_data = {}

    node._emit_completion("success")

    events = node.blackboard.get(BB.PENDING_EVENTS)
    assert node._handle.observation == "Found 2 objects."
    assert events[-1]["data"]["press_status"] == ""


def test_emit_completion_appends_both_press_status_and_failure_reason_guidance() -> None:
    node = _make_node()
    node._handle = SkillHandle()
    node._handle.status = SkillStatus.FAILED
    node._handle.observation = "Press blocked."
    node._handle.result_data = {"press_status": "failed_blocked", "failure_reason": "wall_time_exceeded"}

    node._emit_completion("failed")

    obs = node._handle.observation
    assert "recovery attempts made no progress" in obs  # failed_blocked guidance
    assert "time budget" in obs.lower()  # wall_time_exceeded guidance


def test_emit_completion_does_not_duplicate_identical_guidance_sentences() -> None:
    """Shared codes resolve to byte-identical sentences across the arrival and
    press routers, so a result_data carrying both keys with the same code must
    append the sentence once, not twice."""
    node = _make_node()
    node._handle = SkillHandle()
    node._handle.status = SkillStatus.FAILED
    node._handle.observation = "Press blocked."
    node._handle.result_data = {"arrival_status": "failed_blocked", "press_status": "failed_blocked"}

    node._emit_completion("failed")

    assert node._handle.observation.count("recovery attempts made no progress") == 1
