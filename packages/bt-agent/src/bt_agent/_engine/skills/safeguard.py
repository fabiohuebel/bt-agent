"""Framework-supplied cross-cutting safeguards (paper §5 Invariant I5).

Three generic guards plus the wrap helper. The framework ships these as
the *mechanism* for safe-set lifting (Colledanchise & Ögren Lemma 5.5);
domain-specific safeguards stay in skill-author code.

Wrap shape (composed by ``_wrap_safeguarding`` below):

    ActionSkill  -> Sequence(memory=False, [Cancel, WallTime, Invariant, body])
    ContextSkill -> Sequence(memory=False, [Cancel, body])

``memory=False`` is required so guards re-evaluate every tick (the
hypothesis of the Safe-Set Lifting Theorem).
"""

from __future__ import annotations

import contextlib
import time
from typing import TYPE_CHECKING, Callable, Optional

import py_trees
from bt_agent._engine.agent.blackboard_keys import BB

if TYPE_CHECKING:
    from bt_agent._engine.skills.protocol import Skill, SkillContext, SkillHandle


# ---------------------------------------------------------------------------
# Typed wrap-level failure reasons (bt-nav-fixes CP5)
# ---------------------------------------------------------------------------


def _set_failure_reason(handle: "SkillHandle", reason: str) -> None:
    """Merge ``failure_reason`` into ``handle.result_data`` without clobbering
    whatever the skill body already wrote there (e.g. ``planned_path``)."""
    data = handle.result_data
    data["failure_reason"] = reason
    handle.result_data = data


# ---------------------------------------------------------------------------
# CancellationGuard — universal (both ContextSkill and ActionSkill)
# ---------------------------------------------------------------------------


class CancellationGuard(py_trees.behaviour.Behaviour):
    """Returns FAILURE if cancellation was requested (handle or blackboard).

    Two channels:
      - ``handle.is_cancel_requested`` — set by ``ExecuteSkill.terminate()``
        when the BT preempts the in-flight skill (tree-internal cancellation).
      - ``BB.CANCEL_REQUESTED`` — set by the agent / human (out-of-band).

    Tree-internal cancellation must propagate within one tick for Lemma 5.5
    to hold; that is structurally guaranteed by ``memory=False`` on the
    wrap Sequence.
    """

    def __init__(self, handle: "SkillHandle") -> None:
        super().__init__(name="CancellationGuard")
        self._handle = handle
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=BB.CANCEL_REQUESTED, access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        if self._handle.is_cancel_requested:
            self.feedback_message = "skill-handle cancellation"
            _set_failure_reason(self._handle, "cancelled")
            return py_trees.common.Status.FAILURE
        with contextlib.suppress(KeyError):
            if self.blackboard.get(BB.CANCEL_REQUESTED):
                self.feedback_message = "blackboard cancellation"
                _set_failure_reason(self._handle, "cancelled")
                return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


# ---------------------------------------------------------------------------
# WallTimeBudgetGuard — ActionSkill only
# ---------------------------------------------------------------------------


class WallTimeBudgetGuard(py_trees.behaviour.Behaviour):
    """Returns FAILURE when elapsed wall time since first tick exceeds budget.

    Budget source: ``handle.wall_time_budget_seconds`` when set (per-execution
    override — e.g. NavigateSkill scales it to the planned route, bt-nav-fixes
    CP4), else the static ``budget_seconds`` this guard was constructed with
    (``ActionSkill.wall_time_budget_seconds``, default 120s). Reading the
    override fresh every tick — rather than snapshotting it once — is what
    lets a skill update its own budget mid-run (e.g. on replan).
    """

    def __init__(self, budget_seconds: float, handle: "SkillHandle") -> None:
        super().__init__(name="WallTimeBudgetGuard")
        self._budget = budget_seconds
        self._handle = handle
        self._start_time: Optional[float] = None

    def initialise(self) -> None:
        # Snapshot start time on first tick of this skill execution
        if self._start_time is None:
            self._start_time = time.monotonic()

    def update(self) -> py_trees.common.Status:
        start = self._start_time if self._start_time is not None else time.monotonic()
        elapsed = time.monotonic() - start
        override = self._handle.wall_time_budget_seconds
        budget = override if override is not None else self._budget
        if elapsed > budget:
            msg = f"wall-time budget exceeded ({elapsed:.1f}s > {budget:.1f}s)"
            self._handle.observation = msg
            self.feedback_message = msg
            _set_failure_reason(self._handle, "wall_time_exceeded")
            return py_trees.common.Status.FAILURE
        return py_trees.common.Status.SUCCESS


# ---------------------------------------------------------------------------
# InvariantGuard — ActionSkill only
# ---------------------------------------------------------------------------


class InvariantGuard(py_trees.behaviour.Behaviour):
    """Calls ``skill.safeguard_invariant(state, handle)`` each tick.

    Returns FAILURE when the predicate returns ``False``/a reason string, or
    raises. Defensive: any exception is caught and treated as ``False``; the
    exception type+message is written to ``handle.observation`` so the
    actor can see what broke.

    Return-value contract (bt-nav-fixes CP5): the predicate may return
    ``True`` (success), plain ``False`` (failure, no typed reason —
    backward-compatible default most skills use), or a non-empty ``str``
    (failure with a typed ``failure_reason`` code, e.g. ``"stale_pose"``,
    written into ``handle.result_data``). A plain-truthiness check (``if
    not ok``) would misclassify the string case as success, so the three
    cases are distinguished explicitly below.

    State source: a ``state_provider`` callable, invoked each tick to
    return a fresh ``SkillContext``. Required for Lemma 5.5's
    "re-evaluated on every tick" hypothesis.
    """

    def __init__(
        self,
        skill: "Skill",
        handle: "SkillHandle",
        state_provider: Callable[[], "SkillContext"],
    ) -> None:
        super().__init__(name="InvariantGuard")
        self._skill = skill
        self._handle = handle
        self._state_provider = state_provider

    def update(self) -> py_trees.common.Status:
        try:
            state = self._state_provider()
            result = self._skill.safeguard_invariant(state, self._handle)
        except Exception as exc:  # noqa: BLE001 — defensive default per paper §5 I5
            self._handle.observation = f"safeguard_invariant raised {type(exc).__name__}: {exc}"
            self.feedback_message = "predicate exception"
            # TODO Step 8: emit safeguard_invariant_exception audit event
            return py_trees.common.Status.FAILURE

        if result is True:
            return py_trees.common.Status.SUCCESS

        if result is False:
            if not self._handle.observation:
                self._handle.observation = "safeguard_invariant returned False"
            self.feedback_message = "predicate False"
            # TODO Step 8: emit safeguard_invariant_failure audit event
            return py_trees.common.Status.FAILURE

        # Typed failure reason (str) — CP5.
        reason = str(result)
        if not self._handle.observation:
            self._handle.observation = f"safeguard_invariant failed: {reason}"
        self.feedback_message = f"predicate failed: {reason}"
        _set_failure_reason(self._handle, reason)
        # TODO Step 8: emit safeguard_invariant_failure audit event
        return py_trees.common.Status.FAILURE


# ---------------------------------------------------------------------------
# Wrap helper
# ---------------------------------------------------------------------------


def _wrap_safeguarding(
    skill: "Skill",
    body: py_trees.behaviour.Behaviour,
    handle: "SkillHandle",
    state_provider: Callable[[], "SkillContext"],
) -> py_trees.behaviour.Behaviour:
    """Compose the safeguarding Sequence around a skill body.

    ActionSkill  -> Sequence(memory=False, [Cancel, WallTime, Invariant, body])
    ContextSkill -> Sequence(memory=False, [Cancel, body])
    """
    # Local import to avoid module-load-time cycle with protocol.py
    from bt_agent._engine.skills.protocol import ActionSkill

    children: list = [CancellationGuard(handle)]
    if isinstance(skill, ActionSkill):
        children.append(WallTimeBudgetGuard(skill.wall_time_budget_seconds, handle))
        children.append(InvariantGuard(skill, handle, state_provider))
    children.append(body)
    return py_trees.composites.Sequence(
        name=f"safeguarded_{skill.name}",
        memory=False,
        children=children,
    )
