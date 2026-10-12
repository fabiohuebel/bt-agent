"""CP2 — empty-plan react-fallback executor.

docs/spec/lam-improvement.md CP2 / Success criterion 2: after N=2 failed
syntheses the agent must enter a flat observe→reason→act loop (a single
catch-all milestone the execute branch works) INSTEAD of re-planning, and the
frozen path must be byte-identical (no counter, no fallback, empty stays empty).

Unit-level, no model endpoint: PlanTask.update() is driven directly by seeding
its blackboard result.
"""

from __future__ import annotations

# Import the LAMAgent package first so lam_agent fully initializes before
# tree_builder — importing tree_builder first hits the tree_builder ↔ lam_agent
# circular import. build_agent_tree is then imported function-locally in the
# tree tests (matching test_three_way_verdict.py's workaround).
import bt_agent._engine.agent.lam_agent  # noqa: F401  (init-order side effect)
import py_trees
from bt_agent._engine.agent.blackboard_keys import BB
from bt_agent._engine.agent.config import AgentConfig
from bt_agent._engine.bt.nodes.plan_task import _REACT_FALLBACK_ID, PlanTask


def _writer() -> py_trees.blackboard.Client:
    bb = py_trees.blackboard.Client(name=f"cp2_writer_{id(object())}")
    for key in (
        BB.TASK,
        BB.AGENT_CONFIG,
        BB.EMPTY_PLAN_COUNT,
        BB.REACT_FALLBACK_ACTIVE,
        BB.MILESTONES,
        BB.PENDING_EVENTS,
        BB.CURRENT_MILESTONE_IDX,
        BB.COMPLETED_MILESTONES,
        BB.COMPLETION_CLAIMED,
    ):
        bb.register_key(key=key, access=py_trees.common.Access.WRITE)
    return bb


def _drive_plan_task(config: AgentConfig, empties: int) -> py_trees.blackboard.Client:
    """Feed PlanTask `empties` consecutive empty syntheses; return the blackboard."""
    bb = _writer()
    bb.set(BB.TASK, "bring the red mug to the table")
    bb.set(BB.AGENT_CONFIG, config)
    bb.set(BB.EMPTY_PLAN_COUNT, 0)
    bb.set(BB.REACT_FALLBACK_ACTIVE, False)
    bb.set(BB.PENDING_EVENTS, [])

    node = PlanTask()
    for _ in range(empties):
        node._result = ([], "empty", {})
        node.update()
    return bb


def test_react_fallback_installs_after_threshold() -> None:
    """Structural: 2 empty syntheses → catch-all milestone + REACT_FALLBACK_ACTIVE."""
    bb = _drive_plan_task(AgentConfig(structural_fixes=True), empties=2)

    assert bb.get(BB.EMPTY_PLAN_COUNT) == 2
    assert bb.get(BB.REACT_FALLBACK_ACTIVE) is True
    milestones = bb.get(BB.MILESTONES)
    assert len(milestones) == 1, "flat react-fallback installs exactly one catch-all milestone"
    assert milestones[0]["id"] == _REACT_FALLBACK_ID
    assert milestones[0]["description"] == "bring the red mug to the table"


def test_no_fallback_before_threshold() -> None:
    """Structural: a single empty synthesis does not yet trip the fallback."""
    bb = _drive_plan_task(AgentConfig(structural_fixes=True), empties=1)

    assert bb.get(BB.EMPTY_PLAN_COUNT) == 1
    assert bb.get(BB.REACT_FALLBACK_ACTIVE) is False
    assert bb.get(BB.MILESTONES) == [], "below threshold the plan stays empty"


def test_frozen_path_never_falls_back() -> None:
    """Byte-identical: frozen variants never count empties nor install a fallback."""
    bb = _drive_plan_task(AgentConfig(structural_fixes=False), empties=5)

    # The counter is never written on the frozen path (stays at the seeded 0).
    assert bb.get(BB.EMPTY_PLAN_COUNT) == 0
    assert bb.get(BB.REACT_FALLBACK_ACTIVE) is False
    assert bb.get(BB.MILESTONES) == [], "frozen: empty plan returned verbatim, no fallback"


# ---------------------------------------------------------------------------
# Tree gating
# ---------------------------------------------------------------------------


def _find(node: py_trees.behaviour.Behaviour, name: str):
    if node.name == name:
        return node
    for child in getattr(node, "children", []):
        hit = _find(child, name)
        if hit is not None:
            return hit
    return None


def test_frozen_tree_has_no_react_fallback_branch() -> None:
    from bt_agent._engine.bt.tree_builder import build_agent_tree

    root = build_agent_tree()  # default structural_fixes=False
    assert _find(root, "react_fallback_pass") is None
    skill_or_eval = _find(root, "skill_or_eval")
    assert [c.name for c in skill_or_eval.children] == ["evaluate", "execute"]


def test_structural_tree_adds_react_fallback_branch_first() -> None:
    from bt_agent._engine.bt.tree_builder import build_agent_tree

    root = build_agent_tree(structural_fixes=True)
    skill_or_eval = _find(root, "skill_or_eval")
    # react_fallback_pass is highest priority so a claim short-circuits the evaluator.
    assert [c.name for c in skill_or_eval.children] == ["react_fallback_pass", "evaluate", "execute"]


def test_react_fallback_pass_falls_through_when_not_claimed() -> None:
    """Fallback active but nothing claimed → pass route FAILS so execute (act) runs."""
    from bt_agent._engine.bt.nodes.conditions import (
        CheckCompletionClaimed,
        CheckReactFallbackActive,
    )

    bb = py_trees.blackboard.Client(name=f"cp2_route_{id(object())}")
    for key in (BB.REACT_FALLBACK_ACTIVE, BB.COMPLETION_CLAIMED):
        bb.register_key(key=key, access=py_trees.common.Access.WRITE)
    bb.set(BB.REACT_FALLBACK_ACTIVE, True)
    bb.set(BB.COMPLETION_CLAIMED, False)

    assert CheckReactFallbackActive().update() == py_trees.common.Status.SUCCESS
    # Not claimed → the pass route stops here and control falls to execute.
    assert CheckCompletionClaimed().update() == py_trees.common.Status.FAILURE
