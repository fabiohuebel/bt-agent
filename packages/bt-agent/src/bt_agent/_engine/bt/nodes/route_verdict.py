"""Verdict routing — translate ``BB.LAST_VERDICT`` into a control action.

Three condition nodes (one per verdict) and two action nodes (``ApplyAmend``,
``ApplyReplan``). The pass route is just ``CheckVerdictPass → AdvanceMilestone``
which is already implemented; only amend and replan need new actions.

Composed by :func:`bt.tree_builder.build_agent_tree` as a Selector under the
evaluate Sequence; see :mod:`bt.nodes.evaluate_completion` for what writes
``BB.LAST_VERDICT``.
"""

from __future__ import annotations

from typing import Any

import py_trees
from bt_agent._engine.agent.blackboard_keys import BB
from bt_agent._engine.agent.lam_agent.evaluator import Verdict

# ---------------------------------------------------------------------------
# Condition nodes — SUCCESS iff LAST_VERDICT matches
# ---------------------------------------------------------------------------


class _CheckVerdictBase(py_trees.behaviour.Behaviour):
    """Common machinery for the three verdict-check nodes."""

    _expected: Verdict

    def __init__(self, name: str) -> None:
        super().__init__(name=name)
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=BB.LAST_VERDICT, access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        verdict = self._bb_get(BB.LAST_VERDICT)
        if verdict == self._expected:
            return py_trees.common.Status.SUCCESS
        return py_trees.common.Status.FAILURE

    def _bb_get(self, key: str, default: Any = None) -> Any:
        try:
            return self.blackboard.get(key)
        except KeyError:
            return default


class CheckVerdictPass(_CheckVerdictBase):
    """SUCCESS iff ``BB.LAST_VERDICT == Verdict.PASS``."""

    _expected = Verdict.PASS

    def __init__(self) -> None:
        super().__init__(name="CheckVerdictPass")


class CheckVerdictAmend(_CheckVerdictBase):
    """SUCCESS iff ``BB.LAST_VERDICT == Verdict.AMEND``."""

    _expected = Verdict.AMEND

    def __init__(self) -> None:
        super().__init__(name="CheckVerdictAmend")


class CheckVerdictReplan(_CheckVerdictBase):
    """SUCCESS iff ``BB.LAST_VERDICT == Verdict.REPLAN``."""

    _expected = Verdict.REPLAN

    def __init__(self) -> None:
        super().__init__(name="CheckVerdictReplan")


# ---------------------------------------------------------------------------
# ApplyAmend — increment counter, inject δ into conversation, reset claim
# ---------------------------------------------------------------------------


_CORRECTION_PREFIX = "⚠️ EVALUATOR CORRECTION: "


class ApplyAmend(py_trees.behaviour.Behaviour):
    """Apply an AMEND verdict.

    - Increments ``BB.AMEND_COUNT``.
    - Reads ``BB.AMEND_NOTE`` (δ) and appends a ``{"role": "user", ...}``
      entry to ``BB.CONVERSATION`` so the actor sees the correction on its
      next call (paper §5: "append δ as a new entry in h_act_mi").
    - Clears ``BB.AMEND_NOTE`` (consumed) and resets ``BB.COMPLETION_CLAIMED``
      so the next tick re-enters the execute branch via ``SelectSkill``.
    """

    def __init__(self) -> None:
        super().__init__(name="ApplyAmend")
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=BB.AMEND_COUNT, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.AMEND_NOTE, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.CONVERSATION, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.COMPLETION_CLAIMED, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.PENDING_EVENTS, access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        count = self._bb_get(BB.AMEND_COUNT, 0) + 1
        self.blackboard.set(BB.AMEND_COUNT, count)

        correction = self._bb_get(BB.AMEND_NOTE) or ""
        if correction:
            history = list(self._bb_get(BB.CONVERSATION, []) or [])
            history.append({
                "role": "user",
                "content": _CORRECTION_PREFIX + correction,
            })
            self.blackboard.set(BB.CONVERSATION, history)

        # Consume the note and reset the completion claim.
        self.blackboard.set(BB.AMEND_NOTE, None)
        self.blackboard.set(BB.COMPLETION_CLAIMED, False)

        events = self._bb_get(BB.PENDING_EVENTS, []) or []
        events.append({
            "kind": "amend_applied",
            "data": {"count": count, "correction": correction},
        })
        self.blackboard.set(BB.PENDING_EVENTS, events)

        self.feedback_message = f"amend #{count}"
        return py_trees.common.Status.SUCCESS

    def _bb_get(self, key: str, default: Any = None) -> Any:
        try:
            return self.blackboard.get(key)
        except KeyError:
            return default


# ---------------------------------------------------------------------------
# GiveUpOnReplan: "replan" ablation, terminate instead of re-planning
# ---------------------------------------------------------------------------


_GIVE_UP_TEXT = (
    "Replanning is disabled for this run; giving up on the current milestone and submitting the final task state."
)


class GiveUpOnReplan(py_trees.behaviour.Behaviour):
    """Apply a REPLAN verdict when the "replan" ablation is active.

    docs/spec/tiamat-ablation-flags.md: with replanning ablated there is no
    re-plan to hand a REPLAN verdict to (raw from the evaluator, or cap-promoted
    inside ``EvaluateCompletion.update()``, both land here the same way, since
    both just write ``BB.LAST_VERDICT``). Wiping ``BB.MILESTONES`` and re-firing
    ``PlanTask`` (``ApplyReplan``'s job) is not an option, so instead this node
    ends the episode as an honest give-up: the same real bridge "submit" step
    ``HandleAllDone`` uses on normal completion (bt/nodes/handle_all_done.py),
    confirmed terminal by the environment before ``TASK_COMPLETE`` is set, so a
    disabled replan looks like a normal agent outcome in the record rather than
    a crash or a silent hang. Unlike ``HandleAllDone`` this fires mid-plan, so
    it does not require, or touch, milestone completion state.
    """

    def __init__(self) -> None:
        super().__init__(name="GiveUpOnReplan")
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=BB.PLATFORM_STATE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.TASK_COMPLETE, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.TASK_SUBMITTED, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.PENDING_EVENTS, access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        action = f"submit {_GIVE_UP_TEXT}"
        platform_state = self._bb_get(BB.PLATFORM_STATE, {}) or {}
        bridge = platform_state.get("_bridge") if isinstance(platform_state, dict) else None
        if bridge is None or not callable(getattr(bridge, "step", None)):
            self._emit(
                "replan_ablated_submit_rejected",
                {"action": action, "observation": "platform bridge is unavailable"},
            )
            self.feedback_message = "replan ablated: give-up submit blocked, no bridge"
            return py_trees.common.Status.RUNNING

        observation, _reward, done, info = bridge.step(action)
        info = info if isinstance(info, dict) else {}
        action_ok = info.get("action_ok") is True
        if not (bool(done) and action_ok):
            self._emit(
                "replan_ablated_submit_rejected",
                {"action": action, "observation": str(observation), "terminal": bool(done), "action_ok": action_ok},
            )
            self.feedback_message = "replan ablated: give-up submit not terminal"
            return py_trees.common.Status.RUNNING

        self.blackboard.set(BB.TASK_COMPLETE, True)
        self.blackboard.set(BB.TASK_SUBMITTED, True)
        self._emit(
            "replan_ablated_give_up",
            {"action": action, "observation": str(observation), "outcome": info.get("outcome")},
        )
        self.feedback_message = "replan ablated -> give-up submit"
        return py_trees.common.Status.SUCCESS

    def _emit(self, kind: str, data: dict) -> None:
        events = self._bb_get(BB.PENDING_EVENTS, []) or []
        events.append({"kind": kind, "data": data})
        self.blackboard.set(BB.PENDING_EVENTS, events)

    def _bb_get(self, key: str, default: Any = None) -> Any:
        try:
            return self.blackboard.get(key)
        except KeyError:
            return default


# ---------------------------------------------------------------------------
# ApplyReplan — clear plan, build replan context for next PlanTask
# ---------------------------------------------------------------------------


class ApplyReplan(py_trees.behaviour.Behaviour):
    """Apply a REPLAN verdict.

    - Captures the failed milestone + already-completed milestones into
      ``BB.REPLAN_CONTEXT`` for the next ``PlanTask`` invocation.
    - Clears ``BB.MILESTONES`` (forcing ``CheckHasPlan`` to fail next tick →
      ``PlanTask`` re-fires).
    - Resets per-milestone state: ``CURRENT_MILESTONE_IDX``, ``COMPLETED_MILESTONES``,
      ``AMEND_COUNT``, ``AMEND_NOTE``, ``COMPLETION_CLAIMED``, ``SKILL_HANDLE``,
      ``CONVERSATION``.

    Suffix-only replan: completed milestones are preserved in ``REPLAN_CONTEXT``
    (as descriptions) so the planner knows what's already done; the new
    ``MILESTONES`` list is the fresh suffix and starts at index 0.
    """

    def __init__(self) -> None:
        super().__init__(name="ApplyReplan")
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=BB.MILESTONES, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.CURRENT_MILESTONE_IDX, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.COMPLETED_MILESTONES, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.REPLAN_CONTEXT, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.AMEND_COUNT, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.AMEND_NOTE, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.COMPLETION_CLAIMED, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.SKILL_HANDLE, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.CONVERSATION, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.PENDING_EVENTS, access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        milestones = self._bb_get(BB.MILESTONES, []) or []
        idx = self._bb_get(BB.CURRENT_MILESTONE_IDX, 0)
        completed = self._bb_get(BB.COMPLETED_MILESTONES, set()) or set()

        failed_milestone = milestones[idx] if idx < len(milestones) else {}
        completed_descriptions = [
            milestones[i].get("description", "") for i in sorted(completed) if i < len(milestones)
        ]

        # Pull the most recent evaluator event to recover the reasoning that
        # triggered the replan. Best-effort; default to "unspecified".
        reason = "unspecified"
        events = self._bb_get(BB.PENDING_EVENTS, []) or []
        for event in reversed(events):
            if event.get("kind") == "completion_evaluated":
                reason = event.get("data", {}).get("reasoning", "unspecified")
                break

        replan_context = {
            "failed_milestone_id": failed_milestone.get("id", ""),
            "failed_description": failed_milestone.get("description", ""),
            "reason": reason,
            "completed_descriptions": completed_descriptions,
        }
        self.blackboard.set(BB.REPLAN_CONTEXT, replan_context)

        # Clear plan and per-milestone state so PlanTask re-fires next tick.
        self.blackboard.set(BB.MILESTONES, [])
        self.blackboard.set(BB.CURRENT_MILESTONE_IDX, 0)
        self.blackboard.set(BB.COMPLETED_MILESTONES, set())
        self.blackboard.set(BB.AMEND_COUNT, 0)
        self.blackboard.set(BB.AMEND_NOTE, None)
        self.blackboard.set(BB.COMPLETION_CLAIMED, False)
        self.blackboard.set(BB.SKILL_HANDLE, None)
        self.blackboard.set(BB.CONVERSATION, [])

        events = self._bb_get(BB.PENDING_EVENTS, []) or []
        events.append({
            "kind": "replan_triggered",
            "data": replan_context,
        })
        self.blackboard.set(BB.PENDING_EVENTS, events)

        self.feedback_message = f"replan triggered for milestone {failed_milestone.get('id', '?')}"
        return py_trees.common.Status.SUCCESS

    def _bb_get(self, key: str, default: Any = None) -> Any:
        try:
            return self.blackboard.get(key)
        except KeyError:
            return default
