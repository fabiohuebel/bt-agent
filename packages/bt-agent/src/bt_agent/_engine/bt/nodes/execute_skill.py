"""ExecuteSkill — start a skill and poll its handle until completion.

This is the core "send; monitor; cancel" node:
    initialise() — prepare + start the skill, get a SkillHandle
    update()     — poll handle.status each tick (RUNNING/SUCCESS/FAILURE)
    terminate()  — cancel the skill if the tree preempts us
"""

from __future__ import annotations

from typing import Any, List, Optional

import py_trees
from bt_agent._engine.agent.blackboard_keys import BB
from bt_agent._engine.bt.nodes._state import build_skill_context
from bt_agent._engine.skills.protocol import Skill, SkillContext, SkillHandle, SkillStatus


class ExecuteSkill(py_trees.behaviour.Behaviour):
    """Start or poll the selected skill.

    - For instant skills: start() returns a terminal handle, so update()
      immediately returns SUCCESS or FAILURE on the same tick.
    - For physical skills: start() returns a RUNNING handle. update()
      returns RUNNING each tick until the skill completes.
    - On preemption: terminate() calls handle.cancel().
    """

    def __init__(self) -> None:
        super().__init__(name="ExecuteSkill")
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=BB.SELECTED_SKILL, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.SKILL_PARAMS, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.SKILL_HANDLE, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.SKILL_REGISTRY, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.PENDING_EVENTS, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.SKILL_CALL_ID, access=py_trees.common.Access.READ)

        # World model keys for SkillContext
        self.blackboard.register_key(key=BB.PLATFORM_STATE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.MILESTONES, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.CURRENT_MILESTONE_IDX, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.TASK, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.SCENARIO, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.OBJECT_STORE, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.COMPLETION_CLAIMED, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.TASK_COMPLETE, access=py_trees.common.Access.READ)

        self._handle: Optional[SkillHandle] = None
        self._skill: Optional[Skill] = None  # Stored at initialise() for _apply_results promote hook
        self._skill_name: str = ""
        self._completion_bypass: bool = False

    def initialise(self) -> None:
        self._completion_bypass = False

        # Guard: if completion was claimed, there is no skill to execute.
        # Without this, the execute Sequence would run the stale SELECTED_SKILL.
        # LAMAgent sets COMPLETION_CLAIMED; DecisionMakerAgent sets TASK_COMPLETE.
        if self._bb_get(BB.COMPLETION_CLAIMED) or self._bb_get(BB.TASK_COMPLETE):
            # This is a control-flow bypass, not a successful skill execution.
            # SelectSkill returned SUCCESS inside the memoryful execute Sequence,
            # so ExecuteSkill must also return SUCCESS on this tick.  Do that
            # explicitly instead of manufacturing a terminal handle: a fake
            # handle is otherwise emitted as an empty ``skill_completed`` event
            # and, after an evaluator AMEND, re-enters actor history as a stale
            # duplicate of the preceding real skill.
            self._completion_bypass = True
            self._handle = None
            self.blackboard.set(BB.SKILL_HANDLE, None)
            self._skill = None
            self._skill_name = ""
            return

        skill_name = self.blackboard.get(BB.SELECTED_SKILL)
        params = self.blackboard.get(BB.SKILL_PARAMS)
        registry = self.blackboard.get(BB.SKILL_REGISTRY)

        self._skill_name = skill_name
        skill: Optional[Skill] = registry.get(skill_name)
        self._skill = skill

        if skill is None:
            # Unknown skill — create a failed handle
            self._handle = SkillHandle()
            self._handle.status = SkillStatus.FAILED
            self._handle.observation = f"Unknown skill: {skill_name}"
            self.blackboard.set(BB.SKILL_HANDLE, self._handle)
            return

        context = self._build_context()

        # Prepare: transform LLM params into execution-ready params
        prepared_params, error = skill.prepare(params, context)
        if error is not None:
            self._handle = SkillHandle()
            self._handle.status = SkillStatus.FAILED
            self._handle.observation = error
            self.blackboard.set(BB.SKILL_HANDLE, self._handle)
            return

        # Start the skill — pass a state_provider closure so the framework's
        # InvariantGuard re-evaluates the safeguard predicate against fresh
        # world state each tick (paper §5 Invariant I5; Lemma 5.5 hypothesis).
        state_provider = lambda: build_skill_context(self.blackboard)  # noqa: E731
        self._handle = skill.start(prepared_params, context, state_provider=state_provider)
        self.blackboard.set(BB.SKILL_HANDLE, self._handle)

        # Emit event — include raw LLM params (what the agent chose)
        # separately from prepared_params (what the skill received).
        events = self.blackboard.get(BB.PENDING_EVENTS) or []
        events.append({
            "kind": "skill_started",
            "data": {
                "skill": skill_name,
                "agent_params": params,
                "params": prepared_params,
                "skill_call_id": self._bb_get(BB.SKILL_CALL_ID, ""),
            },
        })
        self.blackboard.set(BB.PENDING_EVENTS, events)

    def update(self) -> py_trees.common.Status:
        if self._completion_bypass:
            # Preserve the execute Sequence's pre-existing SUCCESS transition
            # without publishing a nonexistent skill lifecycle.
            self.feedback_message = f"{self._skill_name}: succeeded"
            return py_trees.common.Status.SUCCESS

        if self._handle is None:
            return py_trees.common.Status.FAILURE

        status = self._handle.status

        if status == SkillStatus.RUNNING:
            self.feedback_message = f"{self._skill_name}: {self._handle.progress:.0%}"
            return py_trees.common.Status.RUNNING

        if status == SkillStatus.SUCCEEDED:
            self._apply_results()
            self._emit_completion("success")
            self.feedback_message = f"{self._skill_name}: succeeded"
            return py_trees.common.Status.SUCCESS

        if status == SkillStatus.FAILED:
            self._emit_completion("failed")
            self.feedback_message = f"{self._skill_name}: failed"
            # Return SUCCESS so the agent loop continues (goes back to SelectSkill).
            # Only FAILURE from ExecuteSkill means "abort the tree branch" (cancellation).
            return py_trees.common.Status.SUCCESS

        if status == SkillStatus.CANCELLED:
            self.feedback_message = f"{self._skill_name}: cancelled"
            return py_trees.common.Status.FAILURE

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        """Cancel the running skill when the tree preempts us."""
        if self._handle is not None and not self._handle.is_terminal:
            self._handle.cancel()

    def _build_context(self) -> SkillContext:
        return build_skill_context(self.blackboard)

    def _bb_get(self, key: str, default: Any = None) -> Any:
        try:
            return self.blackboard.get(key)
        except KeyError:
            return default

    def _apply_results(self) -> None:
        """Apply skill results to the ObjectStore via the skill's publish hook.

        Per paper §5 Invariant I6, the framework only calls the hook; the
        skill decides what (if anything) to write to K. Default behavior on
        the ``Skill`` ABC is a no-op, so skills that don't override
        ``publish_results`` silently keep ``R_K = ∅``.
        """
        if self._handle is None or self._skill is None:
            return
        store = self._bb_get(BB.OBJECT_STORE)
        if store is None:
            return
        self._skill.publish_results(self._handle, store)

    def _emit_completion(self, outcome: str) -> None:
        # Typed-status propagation: a skill may set its own outcome status —
        # result_data["arrival_status"] (NavigateSkill) or
        # result_data["press_status"] (PressSkill) — and/or
        # result_data["failure_reason"] (any ActionSkill, set generically by the
        # safeguard wrap — CancellationGuard/WallTimeBudgetGuard/InvariantGuard —
        # on cancellation, budget exhaustion, or a typed invariant clause;
        # bt-nav-fixes CP5). Append a deterministic one-sentence guidance suffix
        # per code to the observation the actor sees — code-routed, never an LLM
        # call — before the event (and any later history read of
        # handle.observation) captures it.
        result_data = self._handle.result_data if self._handle else None
        arrival_status = (result_data or {}).get("arrival_status", "")
        press_status = (result_data or {}).get("press_status", "")
        failure_reason = (result_data or {}).get("failure_reason", "")
        # Query memoization (tiamat-agent tuning variant #1,
        # lam_v2_textrung/bt_skills/query.py): a memoized reply never touched
        # the bridge, so it never consumed an action -- stamped here for
        # saved-action analysis. Unlike arrival_status/press_status/
        # failure_reason (always present, default ""), this key is only added
        # when a memo hit actually occurred, so a query_memo-off episode's
        # skill_completed events stay byte-identical to today (no new key at
        # all, on ANY skill, ever).
        memoized = bool((result_data or {}).get("memoized"))
        if self._handle is not None and (arrival_status or press_status or failure_reason):
            from bt_agent._engine.agent.lam_agent.actor import (
                route_arrival_guidance,
                route_failure_reason_guidance,
                route_press_status_guidance,
            )

            # Dedup, order-preserving: the arrival and press routers return
            # byte-identical sentences for shared codes, so a result carrying
            # both keys must not append the same sentence twice.
            guidance_parts: List[str] = []
            for guidance in (
                route_arrival_guidance(arrival_status),
                route_press_status_guidance(press_status),
                route_failure_reason_guidance(failure_reason),
            ):
                if guidance and guidance not in guidance_parts:
                    guidance_parts.append(guidance)
            if guidance_parts:
                self._handle.observation = f"{self._handle.observation} {' '.join(guidance_parts)}".strip()

        events = self._bb_get(BB.PENDING_EVENTS, [])
        event_data = {
            "skill": self._skill_name,
            "outcome": outcome,
            "observation": self._handle.observation if self._handle else "",
            "arrival_status": arrival_status,
            "press_status": press_status,
            "failure_reason": failure_reason,
            "skill_call_id": self._bb_get(BB.SKILL_CALL_ID, ""),
        }
        if memoized:
            event_data["memoized"] = True
        events.append({
            "kind": "skill_completed",
            "data": event_data,
        })
        self.blackboard.set(BB.PENDING_EVENTS, events)
