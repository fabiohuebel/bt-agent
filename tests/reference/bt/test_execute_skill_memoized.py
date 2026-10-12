"""Tests for the ``memoized`` stamp at the ExecuteSkill/actor boundary.

Query memoization (tiamat-agent tuning variant #1,
lam_v2_textrung/bt_skills/query.py) marks a memoized reply via
``handle.result_data["memoized"] = True``. ``_emit_completion`` mirrors that
into the ``skill_completed`` event ONLY when present, since (unlike
arrival_status/press_status/failure_reason, which are always-present typed
outcome keys) this key must be absent by default so a query_memo-off episode's
trace stays byte-identical to today -- on ANY skill, not just query.
"""

from __future__ import annotations

from bt_agent._engine.agent.blackboard_keys import BB
from bt_agent._engine.bt.nodes.execute_skill import ExecuteSkill
from bt_agent._engine.skills.protocol import SkillHandle, SkillStatus


def _make_node(skill_name: str = "query") -> ExecuteSkill:
    node = ExecuteSkill()
    node._skill_name = skill_name
    return node


def test_emit_completion_stamps_memoized_true_when_present() -> None:
    node = _make_node()
    node._handle = SkillHandle()
    node._handle.status = SkillStatus.SUCCEEDED
    node._handle.observation = "You recall 1 thing(s) matching 'umbrella': umbrella 1 (last seen: hall)."
    node._handle.result_data = {"memoized": True}

    node._emit_completion("success")

    events = node.blackboard.get(BB.PENDING_EVENTS)
    assert events[-1]["data"]["memoized"] is True


def test_emit_completion_omits_memoized_key_by_default() -> None:
    """No result_data at all: the event must carry no 'memoized' key -- not
    even a False -- so an off/unaffected episode's trace shape is unchanged."""
    node = _make_node()
    node._handle = SkillHandle()
    node._handle.status = SkillStatus.SUCCEEDED
    node._handle.observation = "You recall 1 thing(s) matching 'umbrella': umbrella 1 (last seen: hall)."
    node._handle.result_data = {}

    node._emit_completion("success")

    events = node.blackboard.get(BB.PENDING_EVENTS)
    assert "memoized" not in events[-1]["data"]


def test_emit_completion_omits_memoized_key_for_unrelated_skills() -> None:
    """A skill that never sets result_data["memoized"] (every skill besides a
    memo-hit query) must never see the key appear."""
    node = _make_node(skill_name="press")
    node._handle = SkillHandle()
    node._handle.status = SkillStatus.SUCCEEDED
    node._handle.observation = "Stepped on 'panel'."
    node._handle.result_data = {"press_status": "pressed"}

    node._emit_completion("success")

    events = node.blackboard.get(BB.PENDING_EVENTS)
    assert "memoized" not in events[-1]["data"]
    assert events[-1]["data"]["press_status"] == "pressed"
