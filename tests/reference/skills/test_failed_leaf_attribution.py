"""Regression coverage for `_find_failed_leaf` blaming the wrong leaf.

RCA: a Selector-shaped subtree (e.g. `check_already_at_goal` fallback then
`_PlanPath`) reports overall FAILURE once every child has failed. The old
implementation returned the FIRST child with status==FAILURE, which is the
early fall-through guard leaf — not the leaf whose failure actually ended
the attempt. `_find_failed_leaf` must attribute failure to the LAST FAILURE
child instead.
"""

from __future__ import annotations

import py_trees
from bt_agent._engine.skills.protocol import _find_failed_leaf


class _FixedStatusLeaf(py_trees.behaviour.Behaviour):
    """A leaf whose status is set directly (never ticked)."""

    def __init__(self, name: str, status: py_trees.common.Status) -> None:
        super().__init__(name)
        self.status = status

    def update(self) -> py_trees.common.Status:
        return self.status


def test_selector_blames_last_failure_not_first_fallthrough() -> None:
    """Mirrors Navigate: Selector(check_already_at_goal, _PlanPath).

    check_already_at_goal fails fast (fall-through guard); _PlanPath fails
    later for the real reason. The Selector as a whole reports FAILURE only
    once both children have failed.
    """
    check_already_at_goal = _FixedStatusLeaf("check_already_at_goal", py_trees.common.Status.FAILURE)
    plan_path = _FixedStatusLeaf("_PlanPath", py_trees.common.Status.FAILURE)
    selector = py_trees.composites.Selector(name="navigate_selector", memory=False)
    selector.add_children([check_already_at_goal, plan_path])
    selector.status = py_trees.common.Status.FAILURE

    assert _find_failed_leaf(selector) == "_PlanPath"


def test_sequence_still_blames_its_single_failure_child() -> None:
    """Sequences only ever have one FAILURE child (later siblings stay
    INVALID) — last-vs-first must not change Sequence behavior."""
    ok_step = _FixedStatusLeaf("precondition_check", py_trees.common.Status.SUCCESS)
    failing_step = _FixedStatusLeaf("body", py_trees.common.Status.FAILURE)
    never_run = _FixedStatusLeaf("cleanup", py_trees.common.Status.INVALID)
    sequence = py_trees.composites.Sequence(name="action_sequence", memory=False)
    sequence.add_children([ok_step, failing_step, never_run])
    sequence.status = py_trees.common.Status.FAILURE

    assert _find_failed_leaf(sequence) == "body"
