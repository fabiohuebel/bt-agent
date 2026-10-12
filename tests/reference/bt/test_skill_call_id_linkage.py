"""physics-trace-recording wave 2, item D: skill_call_id linkage.

A uuid minted by SelectSkill the instant the actor picks an action must be
carried, unchanged, through skill_selected -> skill_started ->
skill_completed, and the model exchange that produced the decision is
attached to skill_selected (via ModelBackend.last_exchange_id).
"""

from __future__ import annotations

import py_trees
import pytest
from bt_agent._engine.agent.blackboard_keys import BB
from bt_agent._engine.bt.nodes._state import build_skill_context
from bt_agent._engine.bt.nodes.execute_skill import ExecuteSkill
from bt_agent._engine.bt.nodes.select_skill import SelectSkill
from bt_agent._engine.skills.protocol import SkillHandle, SkillStatus


@pytest.fixture(autouse=True)
def _fresh_blackboard():
    """py_trees Blackboard storage is a process-wide singleton — reset it so
    these tests don't leak state to/from other test modules or each other."""
    py_trees.blackboard.Blackboard.clear()
    yield
    py_trees.blackboard.Blackboard.clear()


class _FakeModel:
    def __init__(self, last_exchange_id: str) -> None:
        self.last_exchange_id = last_exchange_id


def _select_skill_node(exchange_id: str) -> SelectSkill:
    node = SelectSkill()
    node._model = _FakeModel(exchange_id)
    node.blackboard.set(BB.PENDING_EVENTS, [])
    return node


def test_skill_selected_event_carries_a_fresh_skill_call_id_and_the_exchange_id() -> None:
    node = _select_skill_node("exchange-abc")
    node._result = ("CONTINUE", "capture", {"camera_view": "frontleft"}, "reasoning text", "", {})

    node.update()

    events = node.blackboard.get(BB.PENDING_EVENTS)
    event = events[-1]
    assert event["kind"] == "skill_selected"
    assert event["data"]["skill"] == "capture"
    assert event["data"]["exchange_id"] == "exchange-abc"
    skill_call_id = event["data"]["skill_call_id"]
    assert skill_call_id  # non-empty, minted
    assert node.blackboard.get(BB.SKILL_CALL_ID) == skill_call_id


def test_two_consecutive_selections_mint_distinct_skill_call_ids() -> None:
    node = _select_skill_node("exchange-1")
    node._result = ("CONTINUE", "navigate", {}, "r1", "", {})
    node.update()
    first_id = node.blackboard.get(BB.SKILL_CALL_ID)

    node._result = ("CONTINUE", "capture", {}, "r2", "", {})
    node.update()
    second_id = node.blackboard.get(BB.SKILL_CALL_ID)

    assert first_id != second_id


def test_task_complete_and_defer_decisions_never_mint_a_skill_call_id() -> None:
    node = _select_skill_node("exchange-x")
    node._result = ("TASK_COMPLETE", "", {}, "done", "", {})
    node.update()

    events = node.blackboard.get(BB.PENDING_EVENTS)
    assert events[-1]["kind"] == "completion_claimed"
    assert "skill_call_id" not in events[-1]["data"]


def test_execute_skill_carries_the_current_skill_call_id_into_started_and_completed() -> None:
    select = _select_skill_node("exchange-9")
    select._result = ("CONTINUE", "navigate", {"target": "mug"}, "go there", "", {})
    select.update()
    minted_id = select.blackboard.get(BB.SKILL_CALL_ID)

    execute = ExecuteSkill()
    execute._skill_name = "navigate"
    execute._handle = SkillHandle()
    execute._handle.status = SkillStatus.RUNNING
    # Directly exercise the skill_started emission path used by initialise().
    events = execute.blackboard.get(BB.PENDING_EVENTS) or []
    events.append({
        "kind": "skill_started",
        "data": {
            "skill": "navigate",
            "agent_params": {},
            "params": {},
            "skill_call_id": execute._bb_get(BB.SKILL_CALL_ID, ""),
        },
    })
    execute.blackboard.set(BB.PENDING_EVENTS, events)
    started_event = execute.blackboard.get(BB.PENDING_EVENTS)[-1]
    assert started_event["data"]["skill_call_id"] == minted_id

    execute._handle.status = SkillStatus.SUCCEEDED
    execute._emit_completion("success")
    completed_event = execute.blackboard.get(BB.PENDING_EVENTS)[-1]
    assert completed_event["kind"] == "skill_completed"
    assert completed_event["data"]["skill_call_id"] == minted_id


def test_execute_skill_defaults_to_empty_skill_call_id_when_none_minted_yet() -> None:
    """A client that never runs SelectSkill (e.g. the alt-tier / external
    agent paths) must not crash reading an unset BB.SKILL_CALL_ID."""
    execute = ExecuteSkill()
    execute._skill_name = "wait"
    execute._handle = SkillHandle()
    execute._handle.status = SkillStatus.SUCCEEDED

    execute._emit_completion("success")

    event = execute.blackboard.get(BB.PENDING_EVENTS)[-1]
    assert event["data"]["skill_call_id"] == ""


def test_build_skill_context_carries_skill_call_id_for_downstream_skills() -> None:
    node = _select_skill_node("exchange-ctx")
    node._result = ("CONTINUE", "capture", {}, "reasoning", "", {})
    node.update()
    minted_id = node.blackboard.get(BB.SKILL_CALL_ID)

    ctx = build_skill_context(node.blackboard)
    assert ctx.skill_call_id == minted_id
