"""Tests for typed arrival_status propagation at the ExecuteSkill/actor boundary.

Covers docs/spec/bt-nav-fixes.md Checkpoint 2:

- ``skill_completed`` events carry ``arrival_status`` from result_data.
- The actor's tool-result observation (``handle.observation``, later read
  by ``SelectSkill``/``build_history_entry`` as the "tool" message content)
  gets the deterministic guidance suffix appended — never replacing the
  original observation text.
- Skills without an ``arrival_status`` (e.g. most skills) are unaffected.
"""

from __future__ import annotations

from bt_agent._engine.agent.blackboard_keys import BB
from bt_agent._engine.bt.nodes.execute_skill import ExecuteSkill
from bt_agent._engine.skills.protocol import SkillHandle, SkillStatus


def _make_node() -> ExecuteSkill:
    node = ExecuteSkill()
    node._skill_name = "navigate"
    return node


def test_emit_completion_carries_arrival_status_in_event() -> None:
    node = _make_node()
    node._handle = SkillHandle()
    node._handle.status = SkillStatus.FAILED
    node._handle.observation = "Navigation blocked."
    node._handle.result_data = {"arrival_status": "failed_blocked"}

    node._emit_completion("failed")

    events = node.blackboard.get(BB.PENDING_EVENTS)
    assert events[-1]["kind"] == "skill_completed"
    assert events[-1]["data"]["arrival_status"] == "failed_blocked"


def test_emit_completion_appends_guidance_suffix_to_observation() -> None:
    node = _make_node()
    node._handle = SkillHandle()
    node._handle.status = SkillStatus.FAILED
    node._handle.observation = "Navigation blocked."
    node._handle.result_data = {"arrival_status": "failed_blocked"}

    node._emit_completion("failed")

    # Appended, not replacing — original text is still present.
    assert node._handle.observation.startswith("Navigation blocked.")
    assert len(node._handle.observation) > len("Navigation blocked.")
    events = node.blackboard.get(BB.PENDING_EVENTS)
    assert events[-1]["data"]["observation"] == node._handle.observation


def test_emit_completion_arrived_short_appends_distinct_guidance() -> None:
    node = _make_node()
    node._handle = SkillHandle()
    node._handle.status = SkillStatus.SUCCEEDED
    node._handle.observation = "Arrived short of 'mug' (0.90m from goal)."
    node._handle.result_data = {"arrival_status": "arrived_short", "residual_distance_m": 0.9}

    node._emit_completion("success")

    events = node.blackboard.get(BB.PENDING_EVENTS)
    assert events[-1]["data"]["arrival_status"] == "arrived_short"
    assert node._handle.observation.startswith("Arrived short of 'mug' (0.90m from goal).")


def test_emit_completion_without_arrival_status_leaves_observation_untouched() -> None:
    node = _make_node()
    node._handle = SkillHandle()
    node._handle.status = SkillStatus.SUCCEEDED
    node._handle.observation = "Found 2 objects."
    node._handle.result_data = {}

    node._emit_completion("success")

    events = node.blackboard.get(BB.PENDING_EVENTS)
    assert node._handle.observation == "Found 2 objects."
    assert events[-1]["data"]["arrival_status"] == ""


# ---------------------------------------------------------------------------
# Typed wrap-level failure_reason (bt-nav-fixes CP5)
# ---------------------------------------------------------------------------


def test_emit_completion_carries_failure_reason_in_event() -> None:
    node = _make_node()
    node._handle = SkillHandle()
    node._handle.status = SkillStatus.FAILED
    node._handle.observation = "wall-time budget exceeded (130.0s > 120.0s)"
    node._handle.result_data = {"failure_reason": "wall_time_exceeded"}

    node._emit_completion("failed")

    events = node.blackboard.get(BB.PENDING_EVENTS)
    assert events[-1]["data"]["failure_reason"] == "wall_time_exceeded"
    assert events[-1]["data"]["arrival_status"] == ""


def test_emit_completion_appends_failure_reason_guidance_suffix() -> None:
    node = _make_node()
    node._handle = SkillHandle()
    node._handle.status = SkillStatus.FAILED
    node._handle.observation = "safeguard_invariant failed: stale_pose"
    node._handle.result_data = {"failure_reason": "stale_pose"}

    node._emit_completion("failed")

    assert node._handle.observation.startswith("safeguard_invariant failed: stale_pose")
    assert len(node._handle.observation) > len("safeguard_invariant failed: stale_pose")


def test_emit_completion_without_failure_reason_leaves_observation_untouched() -> None:
    node = _make_node()
    node._handle = SkillHandle()
    node._handle.status = SkillStatus.FAILED
    node._handle.observation = "Some other failure."
    node._handle.result_data = {}

    node._emit_completion("failed")

    events = node.blackboard.get(BB.PENDING_EVENTS)
    assert node._handle.observation == "Some other failure."
    assert events[-1]["data"]["failure_reason"] == ""


def test_emit_completion_appends_both_arrival_and_failure_guidance_when_both_present() -> None:
    """Edge case: a skill body could theoretically set arrival_status on a
    prior tick and the wrap could fail on a later tick without the body
    re-running — both guidance sentences should be appended, not just one."""
    node = _make_node()
    node._handle = SkillHandle()
    node._handle.status = SkillStatus.FAILED
    node._handle.observation = "Navigation blocked."
    node._handle.result_data = {"arrival_status": "failed_blocked", "failure_reason": "wall_time_exceeded"}

    node._emit_completion("failed")

    obs = node._handle.observation
    assert obs.startswith("Navigation blocked.")
    assert "stuck" in obs.lower() or "recovery" in obs.lower()  # failed_blocked guidance
    assert "time budget" in obs.lower()  # wall_time_exceeded guidance
