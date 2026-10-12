"""SelectSkill — LLM tool-calling to choose the next skill.

Runs the LLM call in a background thread. Returns RUNNING until the
thread completes, then writes the selected skill + params to the blackboard.

Conversation history is maintained across ReAct steps within a milestone
via BB.CONVERSATION. Before each LLM call, if the previous skill has
completed, a history entry is built from its name, args, and observation
and appended to the conversation.
"""

from __future__ import annotations

import os
import threading
import uuid
from typing import Any, Dict, List, Optional, Tuple

import py_trees
from bt_agent._engine.agent.blackboard_keys import BB
from bt_agent._engine.agent.role import Role
from bt_agent._engine.bt.nodes._state import build_skill_context
from bt_agent._engine.skills.protocol import SkillStatus

# "context" ablation (docs/spec/tiamat-ablation-flags.md CP3): a token budget
# this large is unreachable for any real conversation, so actor.py's
# _split_history (actor.py:810-838) always takes its early-return branch --
# full verbatim history, every tick, no summary block, no compaction call
# (extractive or LLM) ever fires.
_UNBOUNDED_TOKEN_BUDGET = 10**12

_CONTEXT_OVERFLOW_GIVE_UP_TEXT = (
    "This run's history is unbounded and the backend rejected the request as "
    "over its context window; giving up and submitting the final task state."
)

_CONTEXT_OVERFLOW_MARKERS = (
    "context length",
    "context_length_exceeded",
    "maximum context",
    "context window",
)


def _is_context_overflow_error(exc: BaseException) -> bool:
    """Best-effort classifier for a backend HTTP 400 context-length error.

    Only consulted when the "context" ablation is active: without it, the
    actor's own token-budget compaction keeps every real backend well clear
    of its context window, so this classifier is unreachable on the
    byte-identical default tree -- any model exception there still falls
    through to the pre-existing generic "wait" recovery below.
    """
    from bt_agent._engine.model.exchange_recorder import describe_error

    info = describe_error(exc)
    if info.get("status") != 400:
        return False
    message = (info.get("message") or "").lower()
    return any(marker in message for marker in _CONTEXT_OVERFLOW_MARKERS)


def actor_history_observation(
    skill_name: str,
    handle: Any,
    registry: Any,
    failure_presentation: str,
) -> str:
    """Return only the observation text exposed at the Actor ingress.

    Flat presentation applies to terminal failures of registered skills. The
    handle itself is never mutated, so typed status, result data, diagnostics,
    events, and safety records retain the complete failure evidence. Unknown
    or unparseable calls are not registered skills and therefore stay typed.
    """
    if failure_presentation not in {"typed", "flat"}:
        raise ValueError(f"Invalid failure presentation: {failure_presentation!r}")
    if failure_presentation == "flat" and handle.status == SkillStatus.FAILED and registry.get(skill_name) is not None:
        return f"failure of skill {skill_name}"
    return handle.observation or ""


class SelectSkill(py_trees.behaviour.Behaviour):
    """Call the LLM with available tools to select the next skill.

    Lifecycle:
        initialise() — gather context, append previous skill to history,
                        compute available skills, launch LLM thread
        update()     — poll thread; RUNNING until done, write selection to BB
        terminate()  — abandon result if preempted
    """

    def __init__(self) -> None:
        super().__init__(name="SelectSkill")
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=BB.MILESTONES, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.CURRENT_MILESTONE_IDX, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.SKILL_REGISTRY, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.MODEL_BACKEND, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.OBJECT_STORE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.SELECTED_SKILL, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.SKILL_PARAMS, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.SKILL_REASONING, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.SKILL_HANDLE, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.COMPLETION_CLAIMED, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.SKILL_CALL_ID, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.PENDING_EVENTS, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.CONVERSATION, access=py_trees.common.Access.WRITE)

        # World-model keys for the action-mask SkillContext (paper §5 I3).
        self.blackboard.register_key(key=BB.PLATFORM_STATE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.TASK, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.SCENARIO, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.EVENT_SINK, access=py_trees.common.Access.READ)  # OBS-DROP
        # Step 5: prompt variant
        self.blackboard.register_key(key=BB.PROMPT_VARIANT, access=py_trees.common.Access.READ)
        # Agent config
        self.blackboard.register_key(key=BB.AGENT_CONFIG, access=py_trees.common.Access.READ)
        # "context" ablation (CP3): the real-bridge give-up submit on a backend
        # context-overflow error, same shape as GiveUpOnReplan (route_verdict.py).
        self.blackboard.register_key(key=BB.TASK_COMPLETE, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.TASK_SUBMITTED, access=py_trees.common.Access.WRITE)
        # CP3 review round: drive_episode reads this to stamp the literal
        # termination_reason="context_overflow" (board_runner.py).
        self.blackboard.register_key(key=BB.CONTEXT_OVERFLOW_GIVE_UP, access=py_trees.common.Access.WRITE)

        self._thread: Optional[threading.Thread] = None
        self._result: Optional[Tuple[str, Optional[str], Dict, str, str, Dict]] = None
        # Race-free abort handles — see plan_task.py for the rationale.
        self._epoch: int = 0
        self._cancelled: bool = False
        # physics-trace-recording wave 2: kept so update() can read
        # ``model.last_exchange_id`` right after the background thread
        # finishes — the exchange that produced this tick's decision.
        self._model: Optional[Any] = None

    def initialise(self) -> None:
        model = self.blackboard.get(BB.MODEL_BACKEND)
        self._model = model
        registry = self.blackboard.get(BB.SKILL_REGISTRY)
        milestones = self.blackboard.get(BB.MILESTONES)
        idx = self.blackboard.get(BB.CURRENT_MILESTONE_IDX)
        milestone = milestones[idx]

        # -- Gather context from blackboard -----------------------------------

        # Object summaries
        object_summaries = ""
        object_store = self._bb_get(BB.OBJECT_STORE)
        if object_store:
            summaries = object_store.all_llm_summaries()
            if summaries:
                object_summaries = "\n".join(summaries)

        # Conversation history — append previous skill result if available
        history: List[Dict[str, Any]] = self._bb_get(BB.CONVERSATION, [])

        prev_handle = self._bb_get(BB.SKILL_HANDLE)
        prev_skill = self._bb_get(BB.SELECTED_SKILL)
        prev_params = self._bb_get(BB.SKILL_PARAMS)

        if prev_handle is not None and prev_handle.is_terminal and prev_skill:
            from bt_agent._engine.agent.lam_agent.actor import (
                build_history_entry,
                history_call_id,
            )

            tool_args = dict(prev_params or {})
            prev_reasoning = self._bb_get(BB.SKILL_REASONING, "")
            typed_observation = prev_handle.observation or ""
            config = self._bb_get(BB.AGENT_CONFIG)
            failure_presentation = (
                getattr(getattr(config, "actor", None), "failure_presentation", "typed")
                if config is not None
                else "typed"
            )
            observation = actor_history_observation(
                prev_skill,
                prev_handle,
                registry,
                failure_presentation,
            )

            entry = build_history_entry(
                prev_skill,
                tool_args,
                observation,
                call_id=history_call_id(prev_skill, typed_observation),
                reasoning=prev_reasoning,
            )
            history = list(history) + entry
            self.blackboard.set(BB.CONVERSATION, history)

        # Extract images from the previous skill's result_data.
        # Substrate-agnostic: any skill can provide images by putting
        # bytes values in result_data. The framework scans for bytes
        # (duck-typed) — no magic key names.
        injected_images: Optional[List[bytes]] = None
        if prev_handle is not None and prev_handle.is_terminal:
            found: List[bytes] = []
            for v in (prev_handle.result_data or {}).values():
                if isinstance(v, bytes):
                    found.append(v)
                elif isinstance(v, dict):
                    found.extend(b for b in v.values() if isinstance(b, bytes))
                elif isinstance(v, (list, tuple)):
                    found.extend(b for b in v if isinstance(b, bytes))
            if found:
                injected_images = found
                # Instruct the VLM to describe what it sees. The
                # description lands in the reasoning field of the next
                # tool call, which the compressed actor trace captures.
                history = list(history) + [{
                    "role": "user",
                    "content": (
                        "[Image(s) attached from previous skill. Begin "
                        "your reasoning with a 1-sentence description "
                        "of what you observe.]"
                    ),
                }]

        # -- Build available tools (paper §5 I3 declarative action mask) ------

        # Extract a substrate-agnostic tool-call log from h^act_{m_i}.
        # Skills inspect this in is_available() for ordering constraints.
        from bt_agent._engine.bt.nodes._conversation import extract_tool_calls

        history_state = {"tool_calls": extract_tool_calls(history)}
        gating_context: Dict[str, Any] = {
            "current_idx": idx,
            "total_milestones": len(milestones),
        }

        # Σ^avail_t = Σ_ctx ∪ {σ ∈ Σ_act : is_available(history) ∧ precondition(state)}
        # Step 6: layered visibility (paper §5 I2) is dispatched through
        # bt_agent._engine.agent.role. The Role.ACTOR list adds read_skill_docs and
        # the optional defer meta-tool — single source of truth.
        skill_context = build_skill_context(self.blackboard)
        tools = Role.ACTOR.available_tools(
            registry=registry,
            state=skill_context,
            history_state=history_state,
            gating_context=gating_context,
        )

        # -- Launch LLM thread ------------------------------------------------

        event_sink = self._bb_get(BB.EVENT_SINK)  # OBS-DROP
        prompt_variant = self._bb_get(BB.PROMPT_VARIANT) or "tiamat_agent_submit_v5"  # Step 5
        config = self._bb_get(BB.AGENT_CONFIG)
        scenario = self._bb_get(BB.SCENARIO, "") or ""

        # Symmetric action-budget disclosure (tiamat_world docs/spec/
        # budget-disclosure.md, CP4): board_runner.drive_episode pushes
        # "action_budget" onto BB.PLATFORM_STATE only when the episode's
        # scaffold_config configured disclosure; absent here (None) means
        # actor.select_skill's render helper returns "", byte-identical to
        # today. Read fresh every tick so the counter is always live.
        platform_state = self._bb_get(BB.PLATFORM_STATE, {}) or {}
        action_budget = platform_state.get("action_budget") if isinstance(platform_state, dict) else None

        self._cancelled = False
        self._epoch += 1
        my_epoch = self._epoch
        self._result = None
        self._thread = threading.Thread(
            target=self._select_in_background,
            args=(
                my_epoch,
                model,
                tools,
                milestone,
                idx,
                len(milestones),
                object_summaries,
                history,
                injected_images,
                event_sink,  # OBS-DROP
                prompt_variant,
                config,
                scenario,
                action_budget,
            ),
            daemon=True,
        )
        self._thread.start()

    def _select_in_background(
        self,
        my_epoch: int,
        model: Any,
        tools: List[Dict],
        milestone: Dict[str, Any],
        milestone_index: int,
        total_milestones: int,
        object_summaries: str,
        history: List[Dict[str, Any]],
        injected_images: Optional[List[bytes]],
        event_sink: Any = None,  # OBS-DROP
        prompt_variant: str = "tiamat_agent_submit_v5",
        config: Any = None,
        scenario: str = "",
        action_budget: Optional[Dict[str, Any]] = None,
    ) -> None:
        # "context" ablation (docs/spec/tiamat-ablation-flags.md CP3): computed
        # up front so the except clause below can gate context-overflow
        # classification on it too -- a real overflow is only reachable once
        # the budget below is unbounded; the default byte-identical tree keeps
        # the old blanket "wait" recovery for every model error.
        context_ablated = bool(config is not None and "context" in getattr(config, "ablation", frozenset()))
        try:
            from bt_agent._engine.agent.lam_agent.actor import select_skill

            # Thread config values to actor function params
            actor_kwargs: Dict[str, Any] = {}
            if config is not None:
                ac = config.actor
                actor_kwargs["max_history_tokens"] = _UNBOUNDED_TOKEN_BUDGET if context_ablated else ac.token_budget
                actor_kwargs["compaction"] = ac.compaction
                # ABLATION HOOK (LAM_REASONING_MODE): override the actor's per-turn
                # reasoning mode ("act" | "act_thinking" | "react_single" | ...).
                # Inert unless the env var is set — certified config is react_single.
                actor_kwargs["reasoning_mode"] = os.environ.get("LAM_REASONING_MODE") or ac.reasoning_mode
                actor_kwargs["temperature"] = ac.temperature
                actor_kwargs["amend_temperature"] = ac.amend_temperature

            result = select_skill(
                model,
                tools,
                milestone=milestone,
                milestone_index=milestone_index,
                total_milestones=total_milestones,
                object_summaries=object_summaries,
                history=history,
                injected_images=injected_images,
                event_sink=event_sink,  # OBS-DROP
                prompt_variant=prompt_variant,
                scenario=scenario,
                action_budget=action_budget,
                **actor_kwargs,
            )
        except Exception as e:
            if context_ablated and _is_context_overflow_error(e):
                # Caught at this actor call site, per docs/spec/tiamat-ablation-flags.md
                # CP3: update() below ends the episode with a real give-up submit
                # instead of letting this repeat and eventually trip
                # board_runner.ModelTransportFailureTracker, which would discard the
                # whole run as an infra_error envelope with no termination_reason at all.
                result = ("CONTEXT_OVERFLOW", "", {}, str(e), "", {})
            else:
                result = ("CONTINUE", "wait", {}, f"LLM error: {e}", "", {})
        # Only commit if we are still the live generation (see plan_task.py).
        if my_epoch == self._epoch and not self._cancelled:
            self._result = result

    def update(self) -> py_trees.common.Status:
        if self._result is None:
            return py_trees.common.Status.RUNNING

        decision, skill_name, params, reasoning, memory_note, raw = self._result

        if decision == "CONTEXT_OVERFLOW":
            return self._give_up_context_overflow(reasoning)

        if decision == "TASK_COMPLETE":
            self.blackboard.set(BB.COMPLETION_CLAIMED, True)
            self.blackboard.set(BB.SKILL_HANDLE, None)
            # Store the actor's claim reasoning for the evaluator to consider
            self.blackboard.set(BB.SKILL_REASONING, reasoning)

            # Emit observability event
            events = self.blackboard.get(BB.PENDING_EVENTS) or []
            events.append({
                "kind": "completion_claimed",
                "data": {"reasoning": reasoning},
            })
            self.blackboard.set(BB.PENDING_EVENTS, events)

            self.feedback_message = "LLM claims milestone complete"
            return py_trees.common.Status.SUCCESS

        if decision == "DEFER":
            # TODO: validate deferral, handle rejection
            self.feedback_message = f"LLM defers: {reasoning}"
            return py_trees.common.Status.SUCCESS

        # CONTINUE — skill selected
        self.blackboard.set(BB.SELECTED_SKILL, skill_name)
        self.blackboard.set(BB.SKILL_PARAMS, params)
        self.blackboard.set(BB.SKILL_REASONING, reasoning)
        self.blackboard.set(BB.SKILL_HANDLE, None)
        self.blackboard.set(BB.COMPLETION_CLAIMED, False)

        # physics-trace-recording wave 2: mint the id that links this
        # decision through skill_selected -> skill_started -> skill_completed
        # -> capture/submission, and attach the model exchange that produced
        # it. ``last_exchange_id`` is read from the model AFTER the
        # background thread has already returned its final query_tool_call —
        # every reasoning_mode path (including retries) ends on the tool-call
        # that decided ``skill_name``.
        skill_call_id = uuid.uuid4().hex
        exchange_id = getattr(self._model, "last_exchange_id", None)
        self.blackboard.set(BB.SKILL_CALL_ID, skill_call_id)

        # Emit event
        events = self.blackboard.get(BB.PENDING_EVENTS) or []
        events.append({
            "kind": "skill_selected",
            "data": {
                "skill": skill_name,
                "reasoning": reasoning,
                "skill_call_id": skill_call_id,
                "exchange_id": exchange_id,
            },
        })
        self.blackboard.set(BB.PENDING_EVENTS, events)

        self.feedback_message = f"Selected: {skill_name}"
        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status: py_trees.common.Status) -> None:
        # Drop in-flight thread's result on the floor (see plan_task.py).
        self._cancelled = True
        self._result = None

    def _give_up_context_overflow(self, detail: str) -> py_trees.common.Status:
        """ "context" ablation give-up: unbounded history eventually exceeds the
        backend's context window. Mirrors GiveUpOnReplan's real-bridge give-up
        submit (bt/nodes/route_verdict.py); on a confirmed terminal submit this
        also sets ``BB.CONTEXT_OVERFLOW_GIVE_UP``, which drive_episode
        (lam_v2_textrung/board_runner.py, CP3 review round) checks BEFORE its
        generic ``runner.is_complete`` -> "agent_submit" branch, so the record's
        ``termination_reason`` reads the literal "context_overflow" instead.
        The specific cause is also stamped into the pending-events data.

        Always returns RUNNING, win or lose: this node is the FIRST child of the
        ``execute`` Sequence (via ``select_or_resume``) -- SUCCESS here would
        cascade into ``ExecuteSkill`` this same tick on a stale/empty skill
        selection, and FAILURE would cascade all the way up to ``main`` and
        re-fire ``PlanTask``. RUNNING halts both cascades; BB.TASK_COMPLETE
        (once set) is read directly by AgentRunner.is_complete, independent of
        this node's py_trees status.
        """
        if self._bb_get(BB.TASK_COMPLETE, False):
            return py_trees.common.Status.RUNNING

        action = f"submit {_CONTEXT_OVERFLOW_GIVE_UP_TEXT}"
        platform_state = self._bb_get(BB.PLATFORM_STATE, {}) or {}
        bridge = platform_state.get("_bridge") if isinstance(platform_state, dict) else None
        if bridge is None or not callable(getattr(bridge, "step", None)):
            self._emit_context_overflow_event(
                "context_overflow_submit_rejected",
                {"action": action, "detail": detail, "observation": "platform bridge is unavailable"},
            )
            self.feedback_message = "context overflow: give-up submit blocked, no bridge"
            return py_trees.common.Status.RUNNING

        observation, _reward, done, info = bridge.step(action)
        info = info if isinstance(info, dict) else {}
        action_ok = info.get("action_ok") is True
        if not (bool(done) and action_ok):
            self._emit_context_overflow_event(
                "context_overflow_submit_rejected",
                {
                    "action": action,
                    "detail": detail,
                    "observation": str(observation),
                    "terminal": bool(done),
                    "action_ok": action_ok,
                },
            )
            self.feedback_message = "context overflow: give-up submit not terminal"
            return py_trees.common.Status.RUNNING

        self.blackboard.set(BB.TASK_COMPLETE, True)
        self.blackboard.set(BB.TASK_SUBMITTED, True)
        self.blackboard.set(BB.CONTEXT_OVERFLOW_GIVE_UP, True)
        self._emit_context_overflow_event(
            "context_overflow_give_up",
            {"action": action, "detail": detail, "observation": str(observation), "outcome": info.get("outcome")},
        )
        self.feedback_message = "context overflow -> give-up submit"
        return py_trees.common.Status.RUNNING

    def _emit_context_overflow_event(self, kind: str, data: Dict[str, Any]) -> None:
        events = self.blackboard.get(BB.PENDING_EVENTS) or []
        events.append({"kind": kind, "data": data})
        self.blackboard.set(BB.PENDING_EVENTS, events)

    def _bb_get(self, key: str, default: Any = None) -> Any:
        """Safe blackboard read with default for missing keys."""
        try:
            return self.blackboard.get(key)
        except KeyError:
            return default
