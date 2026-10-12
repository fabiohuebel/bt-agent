"""Tests for the three-way Evaluator verdict (paper §5 Invariant I7).

Covers Step 3 of the paper-alignment refactor:

- Evaluator schema accepts {pass, amend, replan} and parses into ``EvaluatorResult``.
- Defensive escalation: AMEND without correction → REPLAN.
- ``EvaluateCompletion`` BT node writes ``BB.LAST_VERDICT`` and applies the
  amend-cap promotion (amend → replan when AMEND_COUNT >= AMEND_CAP).
- ``ApplyAmend`` increments the counter, injects ``δ`` into ``BB.CONVERSATION``
  as a user-role message, and clears ``BB.COMPLETION_CLAIMED``.
- ``ApplyReplan`` clears ``BB.MILESTONES`` (forcing PlanTask re-fire), builds
  ``BB.REPLAN_CONTEXT``, and resets per-milestone state.
- ``AdvanceMilestone`` resets the amend counter on milestone advance.
"""

from __future__ import annotations

from typing import Any
from unittest.mock import MagicMock

import py_trees
import pytest
from bt_agent._engine.agent.blackboard_keys import BB
from bt_agent._engine.agent.lam_agent.evaluator import (
    EvaluatorResult,
    Verdict,
    _parse_result,
    evaluate_completion,
)
from bt_agent._engine.bt.nodes.advance_milestone import AdvanceMilestone
from bt_agent._engine.bt.nodes.evaluate_completion import EvaluateCompletion
from bt_agent._engine.bt.nodes.route_verdict import (
    ApplyAmend,
    ApplyReplan,
    CheckVerdictAmend,
    CheckVerdictPass,
    CheckVerdictReplan,
)

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _bb() -> py_trees.blackboard.Client:
    """Create a fresh blackboard client with all Step 3 keys writable."""
    bb = py_trees.blackboard.Client(name=f"test_{id(object())}")
    for key in (
        BB.LAST_VERDICT,
        BB.AMEND_COUNT,
        BB.AMEND_CAP,
        BB.AMEND_NOTE,
        BB.REPLAN_CONTEXT,
        BB.MILESTONES,
        BB.CURRENT_MILESTONE_IDX,
        BB.COMPLETED_MILESTONES,
        BB.COMPLETION_CLAIMED,
        BB.SKILL_HANDLE,
        BB.CONVERSATION,
        BB.PENDING_EVENTS,
    ):
        bb.register_key(key=key, access=py_trees.common.Access.WRITE)
    return bb


def _tick_once(node: py_trees.behaviour.Behaviour) -> py_trees.common.Status:
    """Tick a node once and return its status."""
    for _ in node.tick():
        pass
    return node.status


# ---------------------------------------------------------------------------
# 1–2. evaluator parsing + defensive fallback
# ---------------------------------------------------------------------------


def test_parse_result_handles_three_verdicts() -> None:
    pass_r = _parse_result({"verdict": "pass", "reasoning": "done", "correction": None})
    assert pass_r.verdict == Verdict.PASS
    assert pass_r.correction is None

    amend_r = _parse_result({"verdict": "amend", "reasoning": "almost", "correction": "try the other camera"})
    assert amend_r.verdict == Verdict.AMEND
    assert amend_r.correction == "try the other camera"

    replan_r = _parse_result({"verdict": "replan", "reasoning": "infeasible"})
    assert replan_r.verdict == Verdict.REPLAN
    assert replan_r.correction is None


def test_parse_result_amend_without_correction_falls_back_to_replan() -> None:
    """AMEND with empty/null correction is defensively promoted to REPLAN."""
    r = _parse_result({"verdict": "amend", "reasoning": "vague", "correction": None})
    assert r.verdict == Verdict.REPLAN
    assert "AMEND verdict without corrective note" in r.reasoning

    r = _parse_result({"verdict": "amend", "reasoning": "vague", "correction": ""})
    assert r.verdict == Verdict.REPLAN


def test_parse_result_unknown_verdict_falls_back_to_replan() -> None:
    r = _parse_result({"verdict": "skibidi", "reasoning": "model went off-script"})
    assert r.verdict == Verdict.REPLAN
    assert "unknown verdict" in r.reasoning.lower()


def test_evaluate_completion_escalates_to_replan_on_model_exception() -> None:
    """Structured query throws → REPLAN with error reasoning."""
    model = MagicMock()
    model.query_structured.side_effect = RuntimeError("connection refused")

    result = evaluate_completion(
        model,
        milestone={"id": "m1", "description": "x", "verify_condition": "y"},
        skill_observation="",
    )
    assert result.verdict == Verdict.REPLAN
    assert "connection refused" in result.reasoning


# ---------------------------------------------------------------------------
# 3–4. EvaluateCompletion BT node — verdict write + cap promotion
# ---------------------------------------------------------------------------


def _stub_evaluate_completion_returning(result: EvaluatorResult) -> Any:
    """Patch the evaluate_completion entry point to return a fixed result."""
    import bt_agent._engine.agent.lam_agent.evaluator as evaluator_mod

    return MagicMock(spec=evaluator_mod.evaluate_completion, return_value=result)


def _setup_evaluate_node_with_result(
    result: EvaluatorResult,
    *,
    amend_count: int = 0,
    amend_cap: int = 2,
    monkeypatch: pytest.MonkeyPatch,
) -> tuple[EvaluateCompletion, py_trees.blackboard.Client]:
    """Build an EvaluateCompletion node with stubbed evaluator + initialized BB."""
    bb = _bb()
    for key in (
        BB.MODEL_BACKEND,
        BB.MILESTONES,
        BB.CURRENT_MILESTONE_IDX,
        BB.SKILL_HANDLE,
        BB.SKILL_REASONING,
        BB.SCENARIO,
        BB.SKILL_REGISTRY,
        BB.OBJECT_STORE,
        BB.EVENT_SINK,
        BB.PROMPT_VARIANT,
    ):
        bb.register_key(key=key, access=py_trees.common.Access.WRITE)

    bb.set(BB.MODEL_BACKEND, MagicMock())
    bb.set(BB.MILESTONES, [{"id": "m1", "description": "x", "verify_condition": "y"}])
    bb.set(BB.CURRENT_MILESTONE_IDX, 0)
    bb.set(BB.SKILL_HANDLE, None)
    bb.set(BB.SKILL_REASONING, "")
    bb.set(BB.SCENARIO, "test scenario")
    bb.set(BB.SKILL_REGISTRY, None)
    bb.set(BB.OBJECT_STORE, None)
    bb.set(BB.EVENT_SINK, None)
    bb.set(BB.PROMPT_VARIANT, "tiamat_agent_submit_v5")
    bb.set(BB.AMEND_COUNT, amend_count)
    bb.set(BB.AMEND_CAP, amend_cap)
    bb.set(BB.AMEND_NOTE, None)
    bb.set(BB.LAST_VERDICT, None)
    bb.set(BB.PENDING_EVENTS, [])

    # Patch the function the BT node imports inside the background thread.
    monkeypatch.setattr(
        "bt_agent._engine.agent.lam_agent.evaluator.evaluate_completion",
        lambda *a, **kw: result,
    )

    node = EvaluateCompletion()
    return node, bb


def test_evaluate_completion_pass_writes_pass_to_bb(monkeypatch: pytest.MonkeyPatch) -> None:
    result = EvaluatorResult(verdict=Verdict.PASS, reasoning="done")
    node, bb = _setup_evaluate_node_with_result(result, monkeypatch=monkeypatch)

    node.initialise()
    # The background thread is daemon — wait briefly until result is set.
    if node._thread is not None:
        node._thread.join(timeout=2.0)
    status = node.update()

    assert status == py_trees.common.Status.SUCCESS
    assert bb.get(BB.LAST_VERDICT) == Verdict.PASS
    assert bb.get(BB.AMEND_NOTE) is None


def test_evaluate_completion_amend_writes_correction(monkeypatch: pytest.MonkeyPatch) -> None:
    result = EvaluatorResult(verdict=Verdict.AMEND, reasoning="close", correction="use back camera")
    node, bb = _setup_evaluate_node_with_result(result, monkeypatch=monkeypatch)

    node.initialise()
    if node._thread is not None:
        node._thread.join(timeout=2.0)
    status = node.update()

    assert status == py_trees.common.Status.SUCCESS
    assert bb.get(BB.LAST_VERDICT) == Verdict.AMEND
    assert bb.get(BB.AMEND_NOTE) == "use back camera"


def test_evaluate_completion_promotes_amend_at_cap(monkeypatch: pytest.MonkeyPatch) -> None:
    """AMEND_COUNT == AMEND_CAP and evaluator returns AMEND → cap promotion to REPLAN."""
    result = EvaluatorResult(verdict=Verdict.AMEND, reasoning="still not right", correction="try X")
    node, bb = _setup_evaluate_node_with_result(result, amend_count=2, amend_cap=2, monkeypatch=monkeypatch)

    node.initialise()
    if node._thread is not None:
        node._thread.join(timeout=2.0)
    status = node.update()

    assert status == py_trees.common.Status.SUCCESS
    assert bb.get(BB.LAST_VERDICT) == Verdict.REPLAN  # promoted
    assert bb.get(BB.AMEND_NOTE) is None  # not an amend anymore

    events = bb.get(BB.PENDING_EVENTS)
    assert events[-1]["data"]["cap_promoted"] is True


# ---------------------------------------------------------------------------
# 5. ApplyAmend — counter, conversation injection, claim reset
# ---------------------------------------------------------------------------


def test_apply_amend_increments_count_and_injects_correction() -> None:
    bb = _bb()
    bb.set(BB.AMEND_COUNT, 0)
    bb.set(BB.AMEND_NOTE, "use the back camera and re-verify")
    bb.set(BB.CONVERSATION, [{"role": "assistant", "content": "I'm done"}])
    bb.set(BB.COMPLETION_CLAIMED, True)
    bb.set(BB.PENDING_EVENTS, [])

    node = ApplyAmend()
    status = _tick_once(node)
    assert status == py_trees.common.Status.SUCCESS

    assert bb.get(BB.AMEND_COUNT) == 1
    assert bb.get(BB.AMEND_NOTE) is None  # consumed
    assert bb.get(BB.COMPLETION_CLAIMED) is False

    convo = bb.get(BB.CONVERSATION)
    assert len(convo) == 2
    assert convo[-1]["role"] == "user"
    assert "use the back camera" in convo[-1]["content"]
    assert "EVALUATOR CORRECTION" in convo[-1]["content"]

    events = bb.get(BB.PENDING_EVENTS)
    assert events[-1]["kind"] == "amend_applied"
    assert events[-1]["data"]["count"] == 1


# ---------------------------------------------------------------------------
# 6. ApplyReplan — plan clear, REPLAN_CONTEXT build, state reset
# ---------------------------------------------------------------------------


def test_apply_replan_clears_milestones_and_builds_context() -> None:
    bb = _bb()
    bb.set(
        BB.MILESTONES,
        [
            {"id": "m0", "description": "open door"},
            {"id": "m1", "description": "find mug"},
            {"id": "m2", "description": "submit"},
        ],
    )
    bb.set(BB.CURRENT_MILESTONE_IDX, 1)
    bb.set(BB.COMPLETED_MILESTONES, {0})
    bb.set(BB.AMEND_COUNT, 1)
    bb.set(BB.AMEND_NOTE, "stale note that should be cleared")
    bb.set(BB.COMPLETION_CLAIMED, True)
    bb.set(BB.SKILL_HANDLE, "fake_handle_object")
    bb.set(BB.CONVERSATION, [{"role": "assistant", "content": "..."}])
    bb.set(
        BB.PENDING_EVENTS,
        [{"kind": "completion_evaluated", "data": {"reasoning": "mug not in scene"}}],
    )

    node = ApplyReplan()
    status = _tick_once(node)
    assert status == py_trees.common.Status.SUCCESS

    # Plan and per-milestone state cleared
    assert bb.get(BB.MILESTONES) == []
    assert bb.get(BB.CURRENT_MILESTONE_IDX) == 0
    assert bb.get(BB.COMPLETED_MILESTONES) == set()
    assert bb.get(BB.AMEND_COUNT) == 0
    assert bb.get(BB.AMEND_NOTE) is None
    assert bb.get(BB.COMPLETION_CLAIMED) is False
    assert bb.get(BB.SKILL_HANDLE) is None
    assert bb.get(BB.CONVERSATION) == []

    # Replan context populated
    rc = bb.get(BB.REPLAN_CONTEXT)
    assert rc is not None
    assert rc["failed_milestone_id"] == "m1"
    assert rc["failed_description"] == "find mug"
    assert rc["completed_descriptions"] == ["open door"]
    assert "mug not in scene" in rc["reason"]

    events = bb.get(BB.PENDING_EVENTS)
    assert events[-1]["kind"] == "replan_triggered"


# ---------------------------------------------------------------------------
# 7. AdvanceMilestone resets amend state
# ---------------------------------------------------------------------------


def test_advance_milestone_resets_amend_state() -> None:
    bb = py_trees.blackboard.Client(name="test_advance")
    for key in (
        BB.MILESTONES,
        BB.CURRENT_MILESTONE_IDX,
        BB.COMPLETED_MILESTONES,
        BB.COMPLETION_CLAIMED,
        BB.SKILL_HANDLE,
        BB.CONVERSATION,
        BB.AMEND_COUNT,
        BB.AMEND_NOTE,
        BB.PENDING_EVENTS,
    ):
        bb.register_key(key=key, access=py_trees.common.Access.WRITE)

    bb.set(BB.MILESTONES, [{"id": "m0"}, {"id": "m1"}])
    bb.set(BB.CURRENT_MILESTONE_IDX, 0)
    bb.set(BB.COMPLETED_MILESTONES, set())
    bb.set(BB.COMPLETION_CLAIMED, True)
    bb.set(BB.SKILL_HANDLE, "fake")
    bb.set(BB.CONVERSATION, [{"role": "user", "content": "x"}])
    bb.set(BB.AMEND_COUNT, 1)
    bb.set(BB.AMEND_NOTE, "stale note")
    bb.set(BB.PENDING_EVENTS, [])

    node = AdvanceMilestone()
    _tick_once(node)

    assert bb.get(BB.CURRENT_MILESTONE_IDX) == 1
    assert bb.get(BB.AMEND_COUNT) == 0
    assert bb.get(BB.AMEND_NOTE) is None


# ---------------------------------------------------------------------------
# 8. Verdict-route condition nodes — single-verdict matching
# ---------------------------------------------------------------------------


def test_verdict_check_nodes_each_match_only_their_verdict() -> None:
    bb = _bb()

    bb.set(BB.LAST_VERDICT, Verdict.PASS)
    assert _tick_once(CheckVerdictPass()) == py_trees.common.Status.SUCCESS
    assert _tick_once(CheckVerdictAmend()) == py_trees.common.Status.FAILURE
    assert _tick_once(CheckVerdictReplan()) == py_trees.common.Status.FAILURE

    bb.set(BB.LAST_VERDICT, Verdict.AMEND)
    assert _tick_once(CheckVerdictPass()) == py_trees.common.Status.FAILURE
    assert _tick_once(CheckVerdictAmend()) == py_trees.common.Status.SUCCESS
    assert _tick_once(CheckVerdictReplan()) == py_trees.common.Status.FAILURE

    bb.set(BB.LAST_VERDICT, Verdict.REPLAN)
    assert _tick_once(CheckVerdictPass()) == py_trees.common.Status.FAILURE
    assert _tick_once(CheckVerdictAmend()) == py_trees.common.Status.FAILURE
    assert _tick_once(CheckVerdictReplan()) == py_trees.common.Status.SUCCESS

    bb.set(BB.LAST_VERDICT, None)
    assert _tick_once(CheckVerdictPass()) == py_trees.common.Status.FAILURE


# ---------------------------------------------------------------------------
# 9. LAMAgent wires AMEND_CAP at construction; reset_for_task clears state
# ---------------------------------------------------------------------------


def test_lam_agent_initializes_amend_cap_from_constructor() -> None:
    from bt_agent._engine.agent.lam_agent import LAMAgent

    model = MagicMock()
    agent = LAMAgent(model, amend_cap=5)
    assert agent.blackboard.get(BB.AMEND_CAP) == 5
    assert agent.blackboard.get(BB.AMEND_COUNT) == 0
    assert agent.blackboard.get(BB.AMEND_NOTE) is None
    assert agent.blackboard.get(BB.LAST_VERDICT) is None
    assert agent.blackboard.get(BB.REPLAN_CONTEXT) is None


def test_lam_agent_reset_for_task_clears_amend_state_but_keeps_cap() -> None:
    from bt_agent._engine.agent.lam_agent import LAMAgent

    model = MagicMock()
    agent = LAMAgent(model, amend_cap=3)

    # Pretend an amend cycle ran
    agent.blackboard.set(BB.AMEND_COUNT, 2)
    agent.blackboard.set(BB.AMEND_NOTE, "stale")
    agent.blackboard.set(BB.LAST_VERDICT, Verdict.AMEND)
    agent.blackboard.set(BB.REPLAN_CONTEXT, {"failed_milestone_id": "x"})

    agent.reset_for_task("new task", "new scenario")

    assert agent.blackboard.get(BB.AMEND_CAP) == 3  # cap survives reset
    assert agent.blackboard.get(BB.AMEND_COUNT) == 0
    assert agent.blackboard.get(BB.AMEND_NOTE) is None
    assert agent.blackboard.get(BB.LAST_VERDICT) is None
    assert agent.blackboard.get(BB.REPLAN_CONTEXT) is None


# ---------------------------------------------------------------------------
# 10. Tree builder constructs the verdict-route Selector
# ---------------------------------------------------------------------------


def test_tree_builder_includes_verdict_route_selector() -> None:
    from bt_agent._engine.bt.tree_builder import build_agent_tree

    root = build_agent_tree()

    def find(node: py_trees.behaviour.Behaviour, name: str):
        if node.name == name:
            return node
        for child in node.children:
            result = find(child, name)
            if result is not None:
                return result
        return None

    vr = find(root, "verdict_route")
    assert vr is not None
    child_names = [c.name for c in vr.children]
    assert child_names == ["pass_route", "amend_route", "replan_route"]

    # Default ablation (empty, docs/spec/tiamat-ablation-flags.md CP2) must be
    # byte-identical: replan_route still uses ApplyReplan, not GiveUpOnReplan.
    replan_route = find(root, "replan_route")
    assert [type(c).__name__ for c in replan_route.children] == ["CheckVerdictReplan", "ApplyReplan"]
