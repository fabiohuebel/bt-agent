"""CP3 — all_done goal-check node (replan-once-or-submit).

docs/spec/lam-improvement.md CP3 / Success criterion 3: when every milestone is
marked done but the goal may be unmet (the wrong-object-capture case), the tree
must trigger EXACTLY ONE corrective re-plan against the literal task, else submit
cleanly — never idle to timeout. Frozen tree keeps the bare CheckAllMilestonesDone.

Unit-level, no model endpoint: HandleAllDone.update() is driven directly.
"""

from __future__ import annotations

import bt_agent._engine.agent.lam_agent  # noqa: F401  (init-order side effect — see CP2 test)
import py_trees
from bt_agent._engine.agent.blackboard_keys import BB
from bt_agent._engine.bt.nodes.handle_all_done import HandleAllDone

_SEED_KEYS = (
    BB.TASK,
    BB.MILESTONES,
    BB.CURRENT_MILESTONE_IDX,
    BB.COMPLETED_MILESTONES,
    BB.REPLAN_CONTEXT,
    BB.CORRECTIVE_REPLAN_DONE,
    BB.REACT_FALLBACK_ACTIVE,
    BB.LAST_VERDICT,
    BB.PLATFORM_STATE,
    BB.TASK_COMPLETE,
    BB.TASK_SUBMITTED,
    BB.COMPLETION_CLAIMED,
    BB.SKILL_HANDLE,
    BB.CONVERSATION,
    BB.AMEND_COUNT,
    BB.AMEND_NOTE,
    BB.PENDING_EVENTS,
)


class _FakeSubmitBridge:
    def __init__(self, *, terminal: bool = True, action_ok: bool = True) -> None:
        self.terminal = terminal
        self.action_ok = action_ok
        self.actions = []

    def step(self, action: str):
        self.actions.append(action)
        info = {
            "action_ok": self.action_ok,
            "reason": None if self.action_ok else "empty_submit",
            "outcome": "premature" if self.terminal and self.action_ok else None,
        }
        return "submit response", 0.0, self.terminal, info


def _seed(**overrides) -> py_trees.blackboard.Client:
    bb = py_trees.blackboard.Client(name=f"cp3_{id(object())}")
    for key in _SEED_KEYS:
        bb.register_key(key=key, access=py_trees.common.Access.WRITE)
    defaults = {
        BB.TASK: "photograph the phone",
        BB.MILESTONES: [{"id": "m1", "description": "capture the map"}],
        BB.CURRENT_MILESTONE_IDX: 1,
        BB.COMPLETED_MILESTONES: {0},
        BB.REPLAN_CONTEXT: None,
        BB.CORRECTIVE_REPLAN_DONE: False,
        BB.REACT_FALLBACK_ACTIVE: False,
        BB.LAST_VERDICT: "pass",
        BB.PLATFORM_STATE: {"_bridge": _FakeSubmitBridge()},
        BB.TASK_COMPLETE: False,
        BB.TASK_SUBMITTED: False,
        BB.PENDING_EVENTS: [],
    }
    defaults.update(overrides)
    for key, val in defaults.items():
        bb.set(key, val)
    return bb


def test_all_done_without_evaluator_pass_triggers_one_corrective_replan() -> None:
    """Exhaustion without the normal PASS proof gets exactly one correction."""
    bb = _seed(**{BB.LAST_VERDICT: "amend"})
    HandleAllDone().update()

    assert bb.get(BB.CORRECTIVE_REPLAN_DONE) is True, "the one corrective replan is latched"
    assert bb.get(BB.MILESTONES) == [], "plan cleared so PlanTask re-fires"
    ctx = bb.get(BB.REPLAN_CONTEXT)
    assert ctx is not None
    assert ctx["failed_description"] == "photograph the phone", "replan against literal task"
    assert ctx["completed_descriptions"] == ["capture the map"]
    # Not yet submitted — the agent gets its one corrective pass first.
    assert bb.get(BB.TASK_COMPLETE) is False
    assert bb.get(BB.TASK_SUBMITTED) is False


def test_all_done_with_evaluator_pass_steps_submit_immediately() -> None:
    """Normal completion is plan exhaustion + evaluator PASS + real submit."""
    bb = _seed()

    status = HandleAllDone().update()

    assert status == py_trees.common.Status.SUCCESS
    assert bb.get(BB.CORRECTIVE_REPLAN_DONE) is False
    assert bb.get(BB.TASK_COMPLETE) is True
    assert bb.get(BB.TASK_SUBMITTED) is True
    actions = bb.get(BB.PLATFORM_STATE)["_bridge"].actions
    assert actions == ["submit All evaluator-approved milestones are complete; submitting the final task state."]


def test_all_done_after_corrective_spent_submits() -> None:
    """Second all_done (latch already set), still exhausted → clean submit."""
    bb = _seed(**{BB.CORRECTIVE_REPLAN_DONE: True})
    HandleAllDone().update()

    assert bb.get(BB.TASK_COMPLETE) is True
    assert bb.get(BB.TASK_SUBMITTED) is True
    actions = bb.get(BB.PLATFORM_STATE)["_bridge"].actions
    assert len(actions) == 1
    assert actions[0].startswith("submit ")
    assert actions[0].removeprefix("submit ").strip()
    # No second replan: the plan is left intact, latch unchanged.
    assert bb.get(BB.MILESTONES) != []
    assert bb.get(BB.REPLAN_CONTEXT) is None


def test_full_state_machine_replan_then_submit() -> None:
    """End-to-end: wrong-object capture → 1 replan; still unmet → submit."""
    bb = _seed(**{BB.LAST_VERDICT: "amend"})
    node = HandleAllDone()

    node.update()  # first all_done → corrective replan
    assert bb.get(BB.CORRECTIVE_REPLAN_DONE) is True
    assert bb.get(BB.TASK_COMPLETE) is False

    # Corrective plan ran, its milestones also completed → all_done fires again.
    bb.set(BB.MILESTONES, [{"id": "m2", "description": "still wrong"}])
    bb.set(BB.COMPLETED_MILESTONES, {0})
    bb.set(BB.LAST_VERDICT, "pass")
    node.update()  # second all_done → submit
    assert bb.get(BB.TASK_COMPLETE) is True
    assert bb.get(BB.TASK_SUBMITTED) is True
    assert bb.get(BB.PLATFORM_STATE)["_bridge"].actions[0].startswith("submit ")


def test_react_fallback_all_done_submits_without_replan() -> None:
    """react-fallback active → submit directly, never a corrective replan."""
    bb = _seed(**{BB.REACT_FALLBACK_ACTIVE: True})
    HandleAllDone().update()

    assert bb.get(BB.TASK_COMPLETE) is True
    assert bb.get(BB.TASK_SUBMITTED) is True
    assert bb.get(BB.CORRECTIVE_REPLAN_DONE) is False, "fallback path never spends the corrective replan"
    assert bb.get(BB.MILESTONES) != [], "plan untouched — no replan"
    assert bb.get(BB.PLATFORM_STATE)["_bridge"].actions[0].startswith("submit ")


def test_submit_rejection_does_not_set_completion() -> None:
    """An env response is terminal proof; a synthetic flag is not enough."""
    bridge = _FakeSubmitBridge(terminal=False, action_ok=False)
    bb = _seed(**{
        BB.CORRECTIVE_REPLAN_DONE: True,
        BB.PLATFORM_STATE: {"_bridge": bridge},
    })

    status = HandleAllDone().update()

    assert status == py_trees.common.Status.RUNNING
    assert bb.get(BB.TASK_COMPLETE) is False
    assert bb.get(BB.TASK_SUBMITTED) is False
    assert bridge.actions[0].startswith("submit ")
    events = bb.get(BB.PENDING_EVENTS)
    assert events[-1]["kind"] == "all_done_submit_rejected"


def test_premature_submit_guard_requires_exact_plan_exhaustion_and_pass() -> None:
    """Direct invocation cannot bypass the sequence's all-done/evaluator proof."""
    bridge = _FakeSubmitBridge()
    bb = _seed(**{
        BB.COMPLETED_MILESTONES: set(),
        BB.CORRECTIVE_REPLAN_DONE: True,
        BB.PLATFORM_STATE: {"_bridge": bridge},
    })
    assert HandleAllDone().update() == py_trees.common.Status.RUNNING
    assert bridge.actions == []
    assert bb.get(BB.TASK_COMPLETE) is False

    bb.set(BB.COMPLETED_MILESTONES, {0})
    bb.set(BB.CORRECTIVE_REPLAN_DONE, False)
    bb.set(BB.LAST_VERDICT, "amend")
    assert HandleAllDone().update() == py_trees.common.Status.SUCCESS
    assert bridge.actions == []
    assert bb.get(BB.TASK_COMPLETE) is False


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


def test_frozen_tree_all_done_is_bare_check() -> None:
    from bt_agent._engine.bt.tree_builder import build_agent_tree

    all_done = _find(build_agent_tree(), "all_done")
    assert [c.name for c in all_done.children] == ["CheckAllMilestonesDone"]


def test_structural_tree_all_done_has_handler() -> None:
    from bt_agent._engine.bt.tree_builder import build_agent_tree

    all_done = _find(build_agent_tree(structural_fixes=True), "all_done")
    assert [c.name for c in all_done.children] == ["CheckAllMilestonesDone", "HandleAllDone"]
