"""PlanTask — generate milestones from a high-level task via agentic LLM loop.

Runs the planner in a background thread. The planner queries the scene
graph, reads skill docs, and synthesizes a milestone plan with a
dependency DAG. Returns RUNNING until the thread completes, then writes
milestones and pre-queried objects to the blackboard.
"""

from __future__ import annotations

import threading
from typing import Any, Dict, List, Optional, Tuple

import py_trees
from bt_agent._engine.agent.blackboard_keys import BB

# CP2: a single catch-all milestone that turns the execute branch into a flat
# observe->reason->act loop (the react executor) over the literal task. Its id is
# stable so downstream nodes / traces can recognize the react-fallback state.
_REACT_FALLBACK_ID = "react_fallback"


def _react_fallback_milestone(task: str) -> Dict[str, Any]:
    """Build the single catch-all milestone the react-fallback runs.

    No dependencies, ``type=direct``: the actor works the whole task in one
    milestone's ReAct loop instead of the planner re-emitting empty plans.
    """
    return {
        "id": _REACT_FALLBACK_ID,
        "description": task,
        "detailed_instructions": (
            "Work the whole task directly, one action at a time: observe the latest "
            "result, reason about the single next step, and act. Call claim_complete "
            "when the task is done."
        ),
        "verify_condition": "Is the task complete?",
        "type": "direct",
        "dependencies": [],
    }


def _static_task_milestone(task: str) -> Dict[str, Any]:
    """The single static milestone shared by two callers: the planner's own
    exception fallback below, and the "planner" ablation (CP3,
    docs/spec/tiamat-ablation-flags.md) which always installs this exact
    shape instead of ever calling ``generate_plan``.
    """
    return {
        "id": "fallback_task",
        "description": task,
        "detailed_instructions": task,
        "verify_condition": "Is the task complete?",
        "type": "direct",
        "dependencies": [],
    }


class PlanTask(py_trees.behaviour.Behaviour):
    """Generate milestones from the task description.

    Lifecycle:
        initialise() — read task + scenario from BB, launch LLM thread
        update()     — poll thread; RUNNING until done, then write milestones
        terminate()  — abandon result if preempted
    """

    def __init__(self) -> None:
        super().__init__(name="PlanTask")
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=BB.TASK, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.SCENARIO, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.MILESTONES, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.CURRENT_MILESTONE_IDX, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.COMPLETED_MILESTONES, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.DEFERRED_MILESTONES, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.COMPLETION_CLAIMED, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.MODEL_BACKEND, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.PENDING_EVENTS, access=py_trees.common.Access.WRITE)

        # Context for agentic planning
        self.blackboard.register_key(key=BB.SKILL_REGISTRY, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.PLATFORM_STATE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.OBJECT_STORE, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.PLAN_REASONING, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.EVENT_SINK, access=py_trees.common.Access.READ)  # OBS-DROP
        # Step 3: replan re-entry — read+clear the failed/completed context.
        self.blackboard.register_key(key=BB.REPLAN_CONTEXT, access=py_trees.common.Access.WRITE)
        # Step 5: read the active prompt variant (constant for the agent's lifetime).
        self.blackboard.register_key(key=BB.PROMPT_VARIANT, access=py_trees.common.Access.READ)
        # Extensible milestone schema
        self.blackboard.register_key(key=BB.MILESTONE_SCHEMA_EXTENSION, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.MILESTONE_FIELDS_GUIDANCE, access=py_trees.common.Access.READ)
        # Agent config
        self.blackboard.register_key(key=BB.AGENT_CONFIG, access=py_trees.common.Access.READ)
        # CP2: empty-plan react-fallback bookkeeping (only touched when gated on).
        self.blackboard.register_key(key=BB.EMPTY_PLAN_COUNT, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.REACT_FALLBACK_ACTIVE, access=py_trees.common.Access.WRITE)

        self._thread: Optional[threading.Thread] = None
        self._result: Optional[Tuple[List[Dict[str, Any]], str, Dict[str, Any]]] = None
        self._is_replan: bool = False
        # Race-free abort handles. ``_epoch`` is bumped each time we spawn a
        # new background thread; workers capture their epoch and only commit
        # to ``_result`` if it still matches. ``_cancelled`` is set when
        # py_trees calls terminate() on us (e.g. because AgentRunner.reset()
        # is stopping the tree mid-LLM-call at a sim auto-advance). See the
        # T5→T6 boundary investigation in 2026-05-22 — without these, the
        # planner's old-task synthesis result clobbers the new task's BB.
        self._epoch: int = 0
        self._cancelled: bool = False

    def initialise(self) -> None:
        task = self.blackboard.get(BB.TASK)
        scenario = self.blackboard.get(BB.SCENARIO)
        model = self.blackboard.get(BB.MODEL_BACKEND)
        skill_registry = self.blackboard.get(BB.SKILL_REGISTRY)
        object_store = self._bb_get(BB.OBJECT_STORE)
        platform_state = self._bb_get(BB.PLATFORM_STATE, {})
        event_sink = self._bb_get(BB.EVENT_SINK)  # OBS-DROP: visualizer broker, may be None

        # Step 3: pick up replan context if this is a re-plan re-entry.
        replan_context = self._bb_get(BB.REPLAN_CONTEXT)
        self._is_replan = replan_context is not None

        # Step 5: prompt variant (defaults to "tiamat_agent_submit_v5" if not set on the BB).
        prompt_variant = self._bb_get(BB.PROMPT_VARIANT) or "tiamat_agent_submit_v5"
        # Extensible milestone schema
        milestone_schema_extension = self._bb_get(BB.MILESTONE_SCHEMA_EXTENSION)
        milestone_fields_guidance = self._bb_get(BB.MILESTONE_FIELDS_GUIDANCE, "")

        # Agent config — planner reads gather limits from here
        config = self._bb_get(BB.AGENT_CONFIG)

        self._cancelled = False
        self._epoch += 1
        my_epoch = self._epoch
        self._result = None
        self._thread = threading.Thread(
            target=self._plan_in_background,
            args=(
                my_epoch,
                model,
                task,
                scenario,
                skill_registry,
                object_store,
                platform_state,
                milestone_schema_extension,
                milestone_fields_guidance,
                event_sink,
                replan_context,
                prompt_variant,
                config,
            ),
            daemon=True,
        )
        self._thread.start()

    def _plan_in_background(
        self,
        my_epoch: int,
        model: Any,
        task: str,
        scenario: str,
        skill_registry: Any,
        object_store: Any,
        platform_state: Dict[str, Any],
        milestone_schema_extension: Any,
        milestone_fields_guidance: str,
        event_sink: Any = None,  # OBS-DROP
        replan_context: Optional[Dict[str, Any]] = None,
        prompt_variant: str = "tiamat_agent_submit_v5",
        config: Any = None,
    ) -> None:
        """Run planning LLM call. Result stored for next update() poll.

        Result is only committed if (a) the worker's captured ``my_epoch``
        still matches ``self._epoch`` (no new initialise() has replaced us)
        AND (b) ``self._cancelled`` is False (terminate() hasn't fired).
        Otherwise the result is dropped — the LLM call is wasted tokens but
        cannot pollute a successor task's blackboard.
        """
        ablation = getattr(config, "ablation", frozenset()) if config is not None else frozenset()
        if "planner" in ablation:
            # "planner" ablation (docs/spec/tiamat-ablation-flags.md CP3): zero
            # planner model calls, ever -- ``generate_plan`` is never imported
            # nor called, on the initial plan or any replan re-entry alike (the
            # ``replan_context`` argument is intentionally unused here; the
            # static milestone never adapts to it). A REPLAN verdict would
            # otherwise cycle here forever (ApplyReplan wipes BB.MILESTONES,
            # PlanTask re-fires, lands right back on the same static milestone,
            # claims again, REPLAN again, ...) bounded only by the tick cap --
            # CP3 review round closed that hole at the routing layer instead:
            # build_agent_tree() routes REPLAN to GiveUpOnReplan whenever
            # "planner" is ablated, exactly as the "replan" ablation does, so
            # this branch is never re-entered via a replan under a live tree.
            result: Tuple[List[Dict[str, Any]], str, Dict[str, Any]] = (
                [_static_task_milestone(task)],
                "Planner is ablated for this run; installing the static single-milestone template.",
                {},
            )
        else:
            try:
                from bt_agent._engine.agent.lam_agent.planner import generate_plan

                # Thread config values to planner function params
                planner_kwargs: Dict[str, Any] = {}
                if config is not None:
                    pc = config.planner
                    planner_kwargs["max_iterations"] = pc.max_gather_iterations
                    planner_kwargs["wall_time_budget"] = pc.gather_wall_time_budget
                    planner_kwargs["synthesis_temperature"] = pc.synthesis_temperature
                    planner_kwargs["replan_temperature"] = pc.replan_temperature
                    # Structural fix (CP1): only the structural variant threads an
                    # explicit synthesis token budget + slim schema. Gated so every
                    # frozen variant keeps omitting max_tokens (backend default) and
                    # the full schema — byte-identical plans.
                    if getattr(config, "structural_fixes", False):
                        planner_kwargs["synthesis_max_tokens"] = pc.synthesis_max_tokens
                        planner_kwargs["slim_schema"] = True

                milestones, reasoning, plan_data = generate_plan(
                    model,
                    task,
                    scenario,
                    skill_registry=skill_registry,
                    object_store=object_store,
                    platform_state=platform_state,
                    milestone_schema_extension=milestone_schema_extension,
                    milestone_fields_guidance=milestone_fields_guidance or "",
                    event_sink=event_sink,  # OBS-DROP
                    replan_context=replan_context,
                    prompt_variant=prompt_variant,
                    **planner_kwargs,
                )
                result = (milestones, reasoning, plan_data)
            except Exception as e:
                # Fallback: single milestone with the full task
                result = (
                    [_static_task_milestone(task)],
                    f"Planning failed ({e}), using fallback.",
                    {},
                )
        if my_epoch == self._epoch and not self._cancelled:
            self._result = result

    def update(self) -> py_trees.common.Status:
        if self._result is None:
            return py_trees.common.Status.RUNNING

        milestones, reasoning, plan_data = self._result

        # CP2: empty-plan react-fallback. On the structural variant, a synthesis
        # that comes back with zero milestones (truncation salvage failed) would
        # otherwise fail CheckHasPlan → re-fire PlanTask forever (the frozen
        # re-plan loop). Instead, count consecutive empties and, at the threshold,
        # install a single catch-all milestone so the execute branch runs the
        # actor as a flat observe→reason→act loop. Gated: frozen variants keep the
        # exact old behavior (empty list returned, no counter, no fallback).
        config = self._bb_get(BB.AGENT_CONFIG)
        if config is not None and getattr(config, "structural_fixes", False) and not milestones:
            count = (self._bb_get(BB.EMPTY_PLAN_COUNT, 0) or 0) + 1
            self.blackboard.set(BB.EMPTY_PLAN_COUNT, count)
            threshold = config.planner.empty_plan_fallback_threshold
            if count >= threshold:
                task = self.blackboard.get(BB.TASK)
                milestones = [_react_fallback_milestone(task)]
                reasoning = f"Empty-plan react-fallback after {count} empty syntheses."
                self.blackboard.set(BB.REACT_FALLBACK_ACTIVE, True)
                self._result = (milestones, reasoning, plan_data)

        # Write milestones to blackboard
        self.blackboard.set(BB.MILESTONES, milestones)
        self.blackboard.set(BB.CURRENT_MILESTONE_IDX, 0)
        self.blackboard.set(BB.COMPLETED_MILESTONES, set())
        self.blackboard.set(BB.DEFERRED_MILESTONES, [])
        self.blackboard.set(BB.COMPLETION_CLAIMED, False)

        # Write plan metadata
        self.blackboard.set(BB.PLAN_REASONING, reasoning)

        # Step 6.5: skill-side data writes to the ObjectStore happen inside
        # each skill's `publish_results` hook (Step 10) during the planner's
        # gather loop. The framework no longer reaches into plan_data for
        # substrate-specific fields.

        # Step 3: replan context is consumed once the new plan lands.
        self.blackboard.set(BB.REPLAN_CONTEXT, None)

        # Emit event — distinguish initial plan from replan for observability.
        events = self.blackboard.get(BB.PENDING_EVENTS) or []
        events.append({
            "kind": "plan_replanned" if self._is_replan else "plan_generated",
            "data": {
                "milestones": milestones,
                "reasoning": reasoning,
            },
        })
        self.blackboard.set(BB.PENDING_EVENTS, events)

        prefix = "Replan" if self._is_replan else "Plan"
        self.feedback_message = f"{prefix}: {len(milestones)} milestones"
        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status: py_trees.common.Status) -> None:
        # Signal the background thread to drop its in-flight result. Combined
        # with the epoch check on the worker, this guarantees no plan from a
        # previous task can land on the blackboard after the runner has reset
        # us for a new task. The HTTP call itself still completes (we don't
        # cancel that at the transport layer); we just refuse to publish it.
        self._cancelled = True
        self._result = None

    def _bb_get(self, key: str, default: Any = None) -> Any:
        try:
            return self.blackboard.get(key)
        except KeyError:
            return default
