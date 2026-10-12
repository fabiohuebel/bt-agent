"""HandleAllDone — the all_done terminal node (replan-once-or-submit).

docs/spec/lam-improvement.md CP3. Replaces the Phase-2 ``HandleDeferredOrSubmit``
stub (tree_builder.py) that left the tree idling to timeout once every milestone
was marked done. Structural-fix only: added to the ``all_done`` Sequence solely
when ``structural_fixes`` is on, so the frozen tree still ends ``all_done`` at
``CheckAllMilestonesDone`` (byte-identical).

Behavior when every milestone is complete:

  * final evaluator verdict PASS → step the canonical terminal submit action.
  * react-fallback active → submit directly. The flat react loop has no final
    evaluator verdict, so this is an explicit give-up completion path.
  * corrective replan already spent, still exhausted → step the
    substrate's canonical ``submit <non-empty text>`` action.  Only a successful
    terminal response may set TASK_COMPLETE + TASK_SUBMITTED.
  * otherwise → run exactly ONE corrective re-plan against the literal task.

Why no ground-truth goal check here: the agent must not read the environment's
``won`` predicate to decide its own termination (RL-readiness invariant — the
oracle is not an agent input). In particular, reaching this node no longer
implies that the real goal is unmet: that inference was only true when a harness
exited early on ``bridge.won``. The evaluator's PASS is the normal completion
proof. The ``CORRECTIVE_REPLAN_DONE`` latch still caps the fallback correction at
one pass.
"""

from __future__ import annotations

from typing import Any

import py_trees
from bt_agent._engine.agent.blackboard_keys import BB


class HandleAllDone(py_trees.behaviour.Behaviour):
    """Terminal handler: evaluator submit, or one corrective fallback.

    ``evaluator_ablated`` (docs/spec/tiamat-ablation-flags.md, threaded in from
    ``build_agent_tree``): with the "evaluator" ablation on there is no
    evaluator model anywhere in the tree, so ``BB.LAST_VERDICT`` never becomes
    "pass" -- ``_evaluator_passed()`` would be permanently False. Since
    ``board_runner._build_agent`` hard-requires ``structural_fixes=True`` for
    every real run, that would mean every evaluator-ablated episode burns its
    one corrective replan (a milestone wipe + an extra planner call) on the
    very first plan exhaustion, before it could ever submit -- exactly the
    live-lock-shaped bug the ablation switch must not have. Evaluator ablation
    already means "any completion claim is proof enough" one milestone at a
    time (``bt/tree_builder.py``'s ``auto_accept`` route); this flag applies
    the same idea at the all-done terminal so plan exhaustion is terminal on
    the FIRST pass, with no corrective replan and no wipe.
    """

    def __init__(self, *, evaluator_ablated: bool = False) -> None:
        super().__init__(name="HandleAllDone")
        self._evaluator_ablated = evaluator_ablated
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=BB.TASK, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.MILESTONES, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.CURRENT_MILESTONE_IDX, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.COMPLETED_MILESTONES, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.REPLAN_CONTEXT, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.CORRECTIVE_REPLAN_DONE, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.REACT_FALLBACK_ACTIVE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.LAST_VERDICT, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.PLATFORM_STATE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.TASK_COMPLETE, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.TASK_SUBMITTED, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.COMPLETION_CLAIMED, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.SKILL_HANDLE, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.CONVERSATION, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.AMEND_COUNT, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.AMEND_NOTE, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.PENDING_EVENTS, access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        react_fallback = self._bb_get(BB.REACT_FALLBACK_ACTIVE, False)
        corrective_done = self._bb_get(BB.CORRECTIVE_REPLAN_DONE, False)

        # Defense in depth: CheckAllMilestonesDone is immediately before this
        # node in the tree, but the terminal action is load-bearing enough to
        # re-check exact plan exhaustion locally.
        if not self._plan_exhausted():
            self.feedback_message = "all_done submit blocked: completion proof missing"
            return py_trees.common.Status.RUNNING

        if react_fallback:
            return self._submit_status("react_fallback")
        if corrective_done:
            return self._submit_status("corrective_replan_spent")
        if self._evaluator_ablated:
            return self._submit_status("evaluator_ablated")
        if self._evaluator_passed():
            return self._submit_status("evaluator_passed")

        # An exhausted plan without a final evaluator PASS is not enough for
        # normal submission. Spend the one corrective pass instead.
        self._corrective_replan()
        return py_trees.common.Status.SUCCESS

    def _submit_status(self, reason: str) -> py_trees.common.Status:
        submitted = self._submit(reason=reason)
        return py_trees.common.Status.SUCCESS if submitted else py_trees.common.Status.RUNNING

    def _plan_exhausted(self) -> bool:
        """Require a non-empty plan with every and only valid index complete."""
        milestones = self._bb_get(BB.MILESTONES, []) or []
        completed = set(self._bb_get(BB.COMPLETED_MILESTONES, set()) or set())
        return bool(milestones) and completed == set(range(len(milestones)))

    def _evaluator_passed(self) -> bool:
        verdict = self._bb_get(BB.LAST_VERDICT)
        return getattr(verdict, "value", verdict) == "pass"

    # -- the one corrective replan against the literal task ------------------

    def _corrective_replan(self) -> None:
        task = self._bb_get(BB.TASK, "")
        milestones = self._bb_get(BB.MILESTONES, []) or []
        completed = self._bb_get(BB.COMPLETED_MILESTONES, set()) or set()
        completed_descriptions = [
            milestones[i].get("description", "") for i in sorted(completed) if i < len(milestones)
        ]

        # Suffix replan against the literal task: the planner is told everything
        # the plan believed it finished, and re-grounds the task to find whatever
        # the milestones missed (the wrong-object-capture case).
        self.blackboard.set(
            BB.REPLAN_CONTEXT,
            {
                "failed_milestone_id": "",
                "failed_description": task,
                "reason": "all_done_goal_check: milestones exhausted but task goal may be unmet",
                "completed_descriptions": completed_descriptions,
            },
        )
        self.blackboard.set(BB.CORRECTIVE_REPLAN_DONE, True)

        # Clear the plan + per-milestone state so PlanTask re-fires next tick
        # (mirrors ApplyReplan). CORRECTIVE_REPLAN_DONE is NOT reset — it caps us
        # at exactly one corrective pass.
        self.blackboard.set(BB.MILESTONES, [])
        self.blackboard.set(BB.CURRENT_MILESTONE_IDX, 0)
        self.blackboard.set(BB.COMPLETED_MILESTONES, set())
        self.blackboard.set(BB.AMEND_COUNT, 0)
        self.blackboard.set(BB.AMEND_NOTE, None)
        self.blackboard.set(BB.COMPLETION_CLAIMED, False)
        self.blackboard.set(BB.SKILL_HANDLE, None)
        self.blackboard.set(BB.CONVERSATION, [])

        events = self._bb_get(BB.PENDING_EVENTS, []) or []
        events.append({"kind": "all_done_corrective_replan", "data": {"task": task}})
        self.blackboard.set(BB.PENDING_EVENTS, events)
        self.feedback_message = "all_done → corrective replan (1/1)"

    # -- clean submit --------------------------------------------------------

    def _submit(self, *, reason: str) -> bool:
        closing_text = {
            "evaluator_passed": "All evaluator-approved milestones are complete; submitting the final task state.",
            "evaluator_ablated": (
                "Evaluator is ablated for this run; the plan is exhausted, submitting the final task state."
            ),
            "react_fallback": "React fallback is exhausted; submitting the final task state.",
            "corrective_replan_spent": "Corrective replanning is exhausted; submitting the final task state.",
        }[reason]
        action = f"submit {closing_text}"

        platform_state = self._bb_get(BB.PLATFORM_STATE, {}) or {}
        bridge = platform_state.get("_bridge") if isinstance(platform_state, dict) else None
        if bridge is None or not callable(getattr(bridge, "step", None)):
            self._submission_rejected(
                reason=reason,
                action=action,
                observation="platform bridge is unavailable",
                terminal=False,
                action_ok=False,
            )
            return False

        observation, _reward, done, info = bridge.step(action)
        info = info if isinstance(info, dict) else {}
        action_ok = info.get("action_ok") is True
        terminal_confirmed = bool(done) and action_ok
        if not terminal_confirmed:
            self._submission_rejected(
                reason=reason,
                action=action,
                observation=str(observation),
                terminal=bool(done),
                action_ok=action_ok,
            )
            return False

        # Completion is downstream of the environment-confirmed terminal step.
        # A premature submit is still a completed (losing) episode; runlog grades
        # success from the final environment state rather than this node.
        self.blackboard.set(BB.TASK_COMPLETE, True)
        self.blackboard.set(BB.TASK_SUBMITTED, True)
        events = self._bb_get(BB.PENDING_EVENTS, []) or []
        events.append({
            "kind": "all_done_submit",
            "data": {
                "reason": reason,
                "action": action,
                "observation": str(observation),
                "outcome": info.get("outcome"),
            },
        })
        self.blackboard.set(BB.PENDING_EVENTS, events)
        self.feedback_message = f"all_done → submit ({reason})"
        return True

    def _submission_rejected(
        self,
        *,
        reason: str,
        action: str,
        observation: str,
        terminal: bool,
        action_ok: bool,
    ) -> None:
        events = self._bb_get(BB.PENDING_EVENTS, []) or []
        events.append({
            "kind": "all_done_submit_rejected",
            "data": {
                "reason": reason,
                "action": action,
                "observation": observation,
                "terminal": terminal,
                "action_ok": action_ok,
            },
        })
        self.blackboard.set(BB.PENDING_EVENTS, events)
        self.feedback_message = f"all_done submit not terminal ({reason})"

    def _bb_get(self, key: str, default: Any = None) -> Any:
        try:
            return self.blackboard.get(key)
        except KeyError:
            return default
