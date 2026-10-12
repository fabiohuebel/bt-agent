"""Maintained task-centric controller with the frozen paper model policy.

The supervisor repeats select/execute until a completion claim, then waits for
host acknowledgment. It never constructs a plan or advances a milestone.
Skill BT execution and actor serialization come from the preserved engine.
"""

from concurrent.futures import Future
from copy import deepcopy
from dataclasses import replace
import hashlib
import inspect
import math
import os
from pathlib import Path
import threading
import time
import uuid

import py_trees

from ._engine.agent.interface import AgentEvent
from ._engine.agent.blackboard_keys import BB
from ._engine.agent.lam_agent import actor
from ._engine.agent.role import Role
from ._engine.bt.nodes._conversation import extract_tool_calls
from ._engine.skills.registry import SkillRegistry
from .profiles import core_config
from .runtime import (
    EpisodeOutcome,
    _acquire_runtime,
    _release_runtime,
    _validate_services,
    _describe_runtime,
)
from .skills import ObjectStore, SkillHandle, SkillStatus
from .task_state import TaskContext, TaskState, TaskStatus, _paper_task_view
from ._blackboard_lifecycle import track_tree

Status = py_trees.common.Status

# Preserve the host protocol independently of the new task-state lifecycle.
# This legacy sentence is not a description of the new runtime's data model.
_PAPER_SUBMIT_ACTION = (
    "submit Evaluator is ablated for this run; the plan is exhausted, "
    "submitting the final task state."
)


class TaskRuntime:
    """Default planner-free runtime, also selected with ``profile='task-core-v1'``.

    Use lifecycle methods on one controller thread. Model work and skill BTs
    run asynchronously. One runtime per process remains the supported contract.
    ``task`` is an immutable snapshot; changing it requires a lifecycle action.
    """

    def __init__(self, *, model, skills, host, internal_skills=()):
        registered, internal = _validate_services(host, skills, internal_skills)
        self._config = core_config("task-core-v1")
        self._model, self._host = model, host
        self._registry = SkillRegistry()
        for skill in registered:
            self._registry.register(skill)
        self._internal = {skill.name: skill for skill in internal}
        self._store = ObjectStore()
        self._task = None
        self._platform = {}
        self._history = []
        self._handle = None
        self._skill = None
        self._selection = None
        self._call_id = ""
        self._ticks = 0
        self._closed = False
        self._stopping = False
        self.events, self.outputs = [], []
        # Do not overlap backend calls after cancellation/reset. A transport
        # request cannot be force-killed; its isolated Future is just retired.
        self._model_lock = threading.Lock()
        self._tree = py_trees.trees.BehaviourTree(
            py_trees.composites.Sequence(
                "Task",
                memory=True,
                children=[_Select(self), _Execute(self), _Submit(self)],
            )
        )
        # Frozen skill safeguards still read one process-global cancellation
        # flag. Scope that compatibility channel to this runtime's lease; task,
        # history and completion state never live on the legacy blackboard.
        try:
            _acquire_runtime(self)
            self._cancel_key = py_trees.blackboard.Blackboard.absolute_name(
                "/", BB.CANCEL_REQUESTED
            )
            self._cancel_existed = self._cancel_key in py_trees.blackboard.Blackboard.storage
            self._old_cancel = py_trees.blackboard.Blackboard.storage.get(self._cancel_key)
        except BaseException:
            _release_runtime(self)
            raise

    @property
    def task(self):
        return self._task

    @property
    def done(self):
        return self._task is not None and self._task.status in {
            TaskStatus.SUBMITTED,
            TaskStatus.CANCELLED,
        }

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Runtime is closed")
        if os.environ.get("LAM_REASONING_MODE") not in (None, "", "react_single"):
            raise ValueError("LAM_REASONING_MODE conflicts with the explicit task-core-v1 profile")

    def start(self, task, *, scenario="", platform_state=None):
        self._check_open()
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Task must be non-empty text")
        if self._task is not None and not self.done:
            raise RuntimeError("Finish or cancel the current episode before starting another")
        state = dict(platform_state or {})
        if "_bridge" in state:
            raise ValueError(
                "Pass the episode host through create_agent, not platform_state['_bridge']"
            )
        self._stop_tree()
        self._host.reset()
        py_trees.blackboard.Blackboard.set(self._cancel_key, False)
        self._task = TaskState(task, scenario)
        self._platform = {**state, "_bridge": self._host}
        self._store.reset()
        self._history = []
        self._handle, self._skill, self._selection = None, None, None
        self._call_id, self._ticks = "", 0
        self._stopping = False
        self.events.clear()
        self.outputs.clear()

    def push_state(self, state):
        self._check_open()
        if "_bridge" in state:
            raise ValueError("Cannot replace the explicit episode host")
        self._platform = {**state, "_bridge": self._host}

    def _context(self):
        return TaskContext(self._task, self._store, self._call_id, dict(self._platform))

    def _emit(self, kind, **data):
        self.events.append(AgentEvent(kind, data))

    def tick(self):
        self._check_open()
        if self._task is None or self.done:
            raise RuntimeError("No active episode")
        if self._stopping:
            raise RuntimeError("Cancellation is pending; retry cancel/close before continuing")
        offset = len(self.events)
        track_tree(self._tree.root)
        track_tree(getattr(self._handle, "subtree", None))
        try:
            self._tree.tick()
        finally:
            track_tree(self._tree.root)
            track_tree(getattr(self._handle, "subtree", None))
        self._ticks += 1
        return self.events[offset:], []

    def objects_snapshot(self):
        self._check_open()
        return tuple(deepcopy(self._store.all_objects())) if self._task is not None else ()

    def _stop_tree(self, *, settle_timeout=2.0):
        self._stopping = True
        self._tree.root.stop(Status.INVALID)
        handle = self._handle
        if handle is not None and not handle.is_terminal:
            handle.cancel()
            deadline = time.monotonic() + settle_timeout
            while not handle.is_terminal and time.monotonic() < deadline:
                time.sleep(0.005)
            if not handle.is_terminal:
                raise RuntimeError(
                    "Skill has not acknowledged cancellation; runtime ownership is retained"
                )

    def cancel(self, reason=""):
        # Stop remains available even if an environment change blocks new work.
        if self._closed:
            raise RuntimeError("Runtime is closed")
        if self._task is not None and not self.done:
            self._stop_tree()
            self._task = replace(
                self._task, status=TaskStatus.CANCELLED, cancellation_reason=reason
            )
            self._emit("cancelled", source="operator", reason=reason)

    def run(self, task, *, scenario="", max_ticks=10000, tick_interval=0.01, platform_state=None):
        if isinstance(max_ticks, bool) or not isinstance(max_ticks, int) or max_ticks < 1:
            raise ValueError("max_ticks must be a positive integer")
        if not math.isfinite(tick_interval) or tick_interval < 0:
            raise ValueError("tick_interval must be finite and non-negative")
        self.start(task, scenario=scenario, platform_state=platform_state)
        try:
            for _ in range(max_ticks):
                self.tick()
                if self.done:
                    submitted = self._task.status is TaskStatus.SUBMITTED
                    return EpisodeOutcome(
                        "submitted" if submitted else "cancelled", submitted, self._ticks
                    )
                time.sleep(tick_interval)
            self.cancel("tick limit")
            return EpisodeOutcome("tick_limit", False, self._ticks)
        except BaseException:
            self.cancel("runtime exception")
            raise

    def describe(self):
        data = _describe_runtime(
            model=self._model,
            config=self._config,
            registry=self._registry,
            internal=self._internal,
            profile="task-core-v1",
            completion=_Submit,
        )
        data["state_model"] = "task/v1"
        data["planner_supported"] = False
        data["policy_compatibility"] = "paper-core-v1 prompts, tools, budgets and terminal action"
        data["qualification"] = "maintained task-state runtime; no live robot qualification"
        data["loaded_source_sha256"]["task_state"] = hashlib.sha256(
            Path(inspect.getfile(TaskState)).read_bytes()
        ).hexdigest()
        return data

    def close(self):
        if not self._closed:
            self._stop_tree()
            if self._task is not None and not self.done:
                self._task = replace(
                    self._task, status=TaskStatus.CANCELLED, cancellation_reason="closed"
                )
            self._closed = True
            if self._cancel_existed:
                py_trees.blackboard.Blackboard.set(self._cancel_key, self._old_cancel)
            else:
                py_trees.blackboard.Blackboard.unset(self._cancel_key)
            _release_runtime(self)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


class _Select(py_trees.behaviour.Behaviour):
    def __init__(self, runtime):
        super().__init__("SelectTaskSkill")
        self.runtime = runtime
        self.future = None

    def initialise(self):
        r = self.runtime
        previous = r._handle
        # A selection without a tool name (text-only replies) is retried
        # without a history entry, matching the paper SelectSkill gate.
        if previous is not None and previous.is_terminal and r._selection and r._selection[0]:
            name, params, reasoning = r._selection
            r._history += actor.build_history_entry(
                name,
                dict(params),
                previous.observation or "",
                call_id=actor.history_call_id(name, previous.observation or ""),
                reasoning=reasoning,
            )
        history = list(r._history)
        images = []
        if previous is not None and previous.is_terminal:
            for value in (previous.result_data or {}).values():
                if isinstance(value, bytes):
                    images.append(value)
                elif isinstance(value, dict):
                    images.extend(item for item in value.values() if isinstance(item, bytes))
                elif isinstance(value, (list, tuple)):
                    images.extend(item for item in value if isinstance(item, bytes))
        if images:
            history.append(
                {
                    "role": "user",
                    "content": (
                        "[Image(s) attached from previous skill. Begin your reasoning "
                        "with a 1-sentence description of what you observe.]"
                    ),
                }
            )
        tools = Role.ACTOR.available_tools(
            registry=r._registry,
            state=r._context(),
            history_state={"tool_calls": extract_tool_calls(history)},
            gating_context={"current_idx": 0, "total_milestones": 1},
        )
        ac = r._config.actor
        kwargs = dict(
            milestone=_paper_task_view(r._task),
            milestone_index=0,
            total_milestones=1,
            object_summaries="\n".join(r._store.all_llm_summaries()),
            history=history,
            injected_images=images or None,
            event_sink=None,
            prompt_variant=r._config.prompt_variant,
            scenario=r._task.scenario,
            action_budget=deepcopy(r._platform.get("action_budget")),
            max_history_tokens=ac.token_budget,
            compaction=ac.compaction,
            reasoning_mode=ac.reasoning_mode,
            temperature=ac.temperature,
            amend_temperature=ac.amend_temperature,
        )
        future = self.future = Future()

        def select():
            # Capture this generation's Future/inputs, never the node's mutable
            # future slot. Cancellation drops late results without state writes.
            with r._model_lock:
                if not future.set_running_or_notify_cancel():
                    return
                try:
                    result = actor.select_skill(r._model, tools, **kwargs)
                except Exception as exc:
                    result = ("CONTINUE", "wait", {}, f"LLM error: {exc}", "", {})
                future.set_result((result, getattr(r._model, "last_exchange_id", None)))

        threading.Thread(target=select, daemon=True).start()

    def update(self):
        if not self.future.done():
            return Status.RUNNING
        r = self.runtime
        result, exchange_id = self.future.result()
        decision, name, params, reasoning, _memory, _raw = result
        r._handle, r._skill = None, None
        if decision == "TASK_COMPLETE":
            r._selection = None
            r._task = replace(r._task, status=TaskStatus.SUBMITTING, completion_reason=reasoning)
            r._emit("completion_claimed", reasoning=reasoning)
        elif decision in ("CONTINUE", "DEFER"):
            # The frozen actor maps any ``defer`` call to DEFER, but this
            # profile never offers that milestone meta-tool. Execute rejects
            # it as an unknown skill and the model sees that observation.
            if decision == "DEFER":
                name = "defer"
            r._selection = (name, params, reasoning)
            r._call_id = uuid.uuid4().hex
            r._emit(
                "skill_selected",
                skill=name,
                reasoning=reasoning,
                skill_call_id=r._call_id,
                exchange_id=exchange_id,
            )
        else:
            raise RuntimeError(
                f"Unsupported task decision {decision!r}; this profile has no planner"
            )
        return Status.SUCCESS

    def terminate(self, new_status):
        if self.future is not None:
            self.future.cancel()
            self.future = None


class _Execute(py_trees.behaviour.Behaviour):
    def __init__(self, runtime):
        super().__init__("ExecuteTaskSkill")
        self.runtime = runtime

    def initialise(self):
        r = self.runtime
        if r._task.status is TaskStatus.SUBMITTING:
            return
        name, params, _reasoning = r._selection
        r._skill = r._internal.get(name) or r._registry.get(name)
        error = f"Unknown skill: {name}" if r._skill is None else None
        context = r._context()
        if r._skill is not None:
            prepared, error = r._skill.prepare(params, context)
        if error is not None:
            r._handle = SkillHandle()
            r._handle.observation = error
            r._handle.status = SkillStatus.FAILED
            return
        r._handle = r._skill.start(prepared, context, state_provider=r._context)
        r._emit(
            "skill_started",
            skill=name,
            agent_params=params,
            params=prepared,
            skill_call_id=r._call_id,
        )

    def update(self):
        r = self.runtime
        if r._task.status is TaskStatus.SUBMITTING:
            return Status.SUCCESS
        handle = r._handle
        if not handle.is_terminal:
            return Status.RUNNING
        if handle.status is SkillStatus.CANCELLED:
            return Status.FAILURE
        if handle.status is SkillStatus.SUCCEEDED and r._skill is not None:
            r._skill.publish_results(handle, r._store)
        data = handle.result_data or {}
        codes = {
            key: data.get(key, "") for key in ("arrival_status", "press_status", "failure_reason")
        }
        guidance = []
        for text in (
            actor.route_arrival_guidance(codes["arrival_status"]),
            actor.route_press_status_guidance(codes["press_status"]),
            actor.route_failure_reason_guidance(codes["failure_reason"]),
        ):
            if text and text not in guidance:
                guidance.append(text)
        if guidance:
            handle.observation = f"{handle.observation} {' '.join(guidance)}".strip()
        r._emit(
            "skill_completed",
            skill=r._selection[0],
            outcome="success" if handle.status is SkillStatus.SUCCEEDED else "failed",
            observation=handle.observation,
            skill_call_id=r._call_id,
            **codes,
            **({"memoized": True} if data.get("memoized") else {}),
        )
        return Status.SUCCESS

    def terminate(self, new_status):
        handle = self.runtime._handle
        if handle is not None and not handle.is_terminal:
            handle.cancel()


class _Submit(py_trees.behaviour.Behaviour):
    def __init__(self, runtime):
        super().__init__("SubmitTask")
        self.runtime = runtime

    def update(self):
        r = self.runtime
        if r._task.status is not TaskStatus.SUBMITTING:
            return Status.SUCCESS
        observation, _reward, done, info = r._host.step(_PAPER_SUBMIT_ACTION)
        info = info if isinstance(info, dict) else {}
        data = dict(
            reason="evaluator_ablated", action=_PAPER_SUBMIT_ACTION, observation=str(observation)
        )
        if not (bool(done) and info.get("action_ok") is True):
            r._emit(
                "all_done_submit_rejected",
                **data,
                terminal=bool(done),
                action_ok=info.get("action_ok") is True,
            )
            return Status.RUNNING
        r._task = replace(r._task, status=TaskStatus.SUBMITTED)
        r._emit("all_done_submit", **data, outcome=info.get("outcome"))
        return Status.SUCCESS
