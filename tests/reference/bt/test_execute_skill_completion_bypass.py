"""Regression tests for completion claims crossing the execute Sequence.

``SelectSkill`` (or ``DecisionMakerAgent``) returns SUCCESS when completion is
claimed, so the memoryful execute Sequence ticks ``ExecuteSkill`` once in the
same traversal.  That traversal must succeed without executing, fabricating, or
reporting a skill.
"""

from __future__ import annotations

import py_trees
import pytest
from bt_agent._engine.agent.blackboard_keys import BB
from bt_agent._engine.bt.nodes.execute_skill import ExecuteSkill


@pytest.fixture(autouse=True)
def _fresh_blackboard():
    """The py_trees blackboard is process-global; isolate every case."""
    py_trees.blackboard.Blackboard.clear()
    yield
    py_trees.blackboard.Blackboard.clear()


def _writer() -> py_trees.blackboard.Client:
    bb = py_trees.blackboard.Client(name=f"completion_bypass_{id(object())}")
    for key in (
        BB.COMPLETION_CLAIMED,
        BB.TASK_COMPLETE,
        BB.SKILL_HANDLE,
        BB.SELECTED_SKILL,
        BB.SKILL_PARAMS,
        BB.SKILL_REGISTRY,
        BB.PENDING_EVENTS,
    ):
        bb.register_key(key=key, access=py_trees.common.Access.WRITE)
    bb.set(BB.COMPLETION_CLAIMED, False)
    bb.set(BB.TASK_COMPLETE, False)
    bb.set(BB.SELECTED_SKILL, "missing_skill")
    bb.set(BB.SKILL_PARAMS, {})
    bb.set(BB.SKILL_REGISTRY, {})
    bb.set(BB.PENDING_EVENTS, [{"kind": "completion_claimed", "data": {"reasoning": "done"}}])
    return bb


@pytest.mark.parametrize("completion_key", [BB.COMPLETION_CLAIMED, BB.TASK_COMPLETE])
def test_completion_bypass_succeeds_without_a_fake_handle_or_skill_event(
    completion_key: str,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    bb = _writer()
    bb.set(completion_key, True)
    node = ExecuteSkill()
    execute_sequence = py_trees.composites.Sequence(
        name="Execute",
        memory=True,
        children=[py_trees.behaviours.Success(name="SelectSkill"), node],
    )
    monkeypatch.setattr(node, "_apply_results", lambda: pytest.fail("bypass published skill results"))
    monkeypatch.setattr(node, "_emit_completion", lambda _outcome: pytest.fail("bypass emitted a skill event"))

    execute_sequence.tick_once()

    assert execute_sequence.status == py_trees.common.Status.SUCCESS
    assert node.status == py_trees.common.Status.SUCCESS
    assert node.feedback_message == ": succeeded"
    assert node._handle is None
    assert bb.get(BB.SKILL_HANDLE) is None
    assert bb.get(BB.PENDING_EVENTS) == [{"kind": "completion_claimed", "data": {"reasoning": "done"}}]


def test_completion_bypass_flag_resets_on_the_next_initialisation() -> None:
    bb = _writer()
    bb.set(BB.COMPLETION_CLAIMED, True)
    node = ExecuteSkill()
    node.tick_once()
    assert node.status == py_trees.common.Status.SUCCESS

    bb.set(BB.COMPLETION_CLAIMED, False)
    node.tick_once()

    assert node._completion_bypass is False
    assert node.status == py_trees.common.Status.SUCCESS
    assert bb.get(BB.PENDING_EVENTS)[-1]["data"] == {
        "skill": "missing_skill",
        "outcome": "failed",
        "observation": "Unknown skill: missing_skill",
        "arrival_status": "",
        "press_status": "",
        "failure_reason": "",
        "skill_call_id": "",
    }
