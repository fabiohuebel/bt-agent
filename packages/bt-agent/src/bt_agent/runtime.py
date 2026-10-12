"""Small public lifecycle facade over the preserved controller."""

from dataclasses import asdict, dataclass
from copy import deepcopy
from collections.abc import Iterable, Mapping
import math
import hashlib
import inspect
import json
import os
import threading
import time
from typing import Any

import py_trees

from ._engine.agent.blackboard_keys import BB
from ._engine.agent.lam_agent import LAMAgent
from ._engine.agent.runner import AgentRunner
from ._engine.agent.role import ACTOR_META_TOOL_NAMES
from .profiles import DEFAULT_RUNTIME_PROFILE, core_config
from .skills import Skill, ContextSkill, EffectType
from ._engine.agent.system_prompts import get_prompts
from ._blackboard_lifecycle import track_client, track_tree

_owner_lock = threading.Lock()
_owner = None


def _validate_services(host, skills, internal_skills):
    if os.environ.get("LAM_REASONING_MODE") not in (None, "", "react_single"):
        raise ValueError("LAM_REASONING_MODE conflicts with the explicit runtime profile")
    if not callable(getattr(host, "step", None)) or not callable(getattr(host, "reset", None)):
        raise TypeError("Episode host must implement step(action) and reset()")
    registered, internal = list(skills), list(internal_skills)
    if any(
        getattr(skill, "effect_type", None) not in (EffectType.CONTEXT, EffectType.ACTION)
        for skill in registered
    ):
        raise TypeError("Skill effect_type must be EffectType.CONTEXT or EffectType.ACTION")
    if any(
        not isinstance(skill, ContextSkill) or skill.effect_type != EffectType.CONTEXT
        for skill in internal
    ):
        raise TypeError("Internal handlers must be context skills, never physical actions")
    names = [getattr(s, "name", None) for s in registered + internal]
    if any(not isinstance(n, str) or not n.strip() for n in names) or len(names) != len(set(names)):
        raise ValueError("Skill names must be non-empty and unique")
    if any(n in ACTOR_META_TOOL_NAMES | {"submit"} for n in names):
        raise ValueError("Skill name collides with controller-owned completion/meta tools")
    return registered, internal


def _acquire_runtime(runtime):
    global _owner
    with _owner_lock:
        if _owner is not None:
            raise RuntimeError(
                "One active runtime per process; close the previous runtime or use another process"
            )
        _owner = runtime


def _release_runtime(runtime):
    global _owner
    with _owner_lock:
        if _owner is runtime:
            _owner = None


def _describe_runtime(*, model, config, registry, internal, profile, completion):
    from pathlib import Path
    from ._engine.agent.lam_agent import actor
    from ._engine.skills import protocol
    from . import __version__, tracing

    loaded_config = asdict(config)
    loaded_config["ablation"] = sorted(loaded_config["ablation"])
    prompt = asdict(get_prompts(config.prompt_variant))
    return {
        "schema": "bt-agent-loaded-runtime/v1",
        "package_version": __version__,
        "profile": profile,
        "reference_commit": "62056af6fcb9ac3ec611760415c0c4f2f91a1fb6",
        "agent_config": loaded_config,
        "prompt_bundle_sha256": hashlib.sha256(
            json.dumps(prompt, sort_keys=True).encode()
        ).hexdigest(),
        "skill_tools": [s.tool_def() for s in registry.all_skills().values()],
        "internal_handlers": sorted(internal),
        "loaded_source_sha256": {
            name: hashlib.sha256(Path(inspect.getfile(obj)).read_bytes()).hexdigest()
            for name, obj in {
                "actor": actor,
                "completion": completion,
                "skills": protocol,
                "tracing": tracing,
                "runtime_support": AgentRuntime,
                "blackboard_lifecycle": track_client,
            }.items()
        },
        "controller_request_caps": {"actor_decision": 1024, "history_compaction": 512},
        "model": {
            "name": getattr(model, "model_name", None),
            "backend_default_output_cap": getattr(model, "max_tokens", None),
            "seed": getattr(model, "seed", None),
            "thinking": getattr(model, "_enable_thinking", None),
        },
        "qualification": "core extraction only; no robot/simulator qualification",
    }


@dataclass(frozen=True)
class EpisodeOutcome:
    """Termination, not task success. Grading belongs to the episode host's caller."""

    termination: str
    submitted: bool
    ticks: int


class AgentRuntime:
    def __init__(
        self,
        *,
        model: Any,
        skills: Iterable[Skill],
        host: Any,
        profile: str,
        internal_skills: Iterable[ContextSkill] = (),
    ):
        if profile != "paper-core-v1":
            raise ValueError("Use create_agent() to select a non-paper runtime profile")
        config = core_config(profile)
        registered, internal = _validate_services(host, skills, internal_skills)
        try:
            _acquire_runtime(self)
            self._agent = LAMAgent(model, config=config)
            track_client(self._agent.blackboard)
            for skill in registered:
                self._agent.register_skill(skill.name, skill)
            # The frozen text runner resolves its private `wait` fallback here,
            # without adding it to the model's tool menu. Keep this support in
            # the public facade instead of requiring adapters to patch _engine.
            self._internal_skills = {s.name: s for s in internal}
            if internal:
                ordinary_get = self._agent._registry.get
                self._agent._registry.get = lambda name: self._internal_skills.get(
                    name
                ) or ordinary_get(name)
            self._runner = AgentRunner(self._agent)
            self._track_clients()
        except BaseException:
            _release_runtime(self)
            raise
        self._host = host
        self._profile = profile
        self._closed = False
        self._started = False
        self._ended = False
        self.events = []
        self.outputs = []

    def _track_clients(self):
        track_client(self._agent.blackboard)
        track_client(self._runner._bb)
        track_tree(self._runner.tree.root)

    def _check_open(self):
        if self._closed:
            raise RuntimeError("Runtime is closed")
        if os.environ.get("LAM_REASONING_MODE") not in (None, "", "react_single"):
            raise ValueError("LAM_REASONING_MODE conflicts with the explicit paper-core-v1 profile")

    def describe(self):
        """Loaded configuration and code/tool identities; never serialize secrets."""
        from ._engine.bt.nodes.handle_all_done import HandleAllDone

        return _describe_runtime(
            model=self._agent._model,
            config=self._agent._config,
            registry=self._agent._registry,
            internal=self._internal_skills,
            profile=self._profile,
            completion=HandleAllDone,
        )

    def start(self, task: str, *, scenario: str = "", platform_state: Mapping | None = None):
        self._check_open()
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Task must be non-empty text")
        if self._started and not self._ended:
            raise RuntimeError("Finish or cancel the current episode before starting another")
        state = dict(platform_state or {})
        if "_bridge" in state:
            raise ValueError(
                "Pass the episode host through create_agent, not platform_state['_bridge']"
            )
        self._runner.reset(task, scenario)
        self._host.reset()
        self._runner.push_state({**state, "_bridge": self._host})
        self.events.clear()
        self.outputs.clear()
        self._started, self._ended = True, False

    def tick(self):
        self._check_open()
        if not self._started or self._ended:
            raise RuntimeError("No active episode")
        self._track_clients()
        try:
            events, outputs = self._runner.tick()
        finally:
            self._track_clients()
        self.events.extend(events)
        self.outputs.extend(outputs)
        self._ended = self._runner.is_complete or self._runner.is_cancelled
        return events, outputs

    def push_state(self, state: Mapping):
        self._check_open()
        if "_bridge" in state:
            raise ValueError("Cannot replace the explicit episode host")
        self._runner.push_state({**state, "_bridge": self._host})

    @property
    def done(self):
        return self._ended

    def objects_snapshot(self):
        """Detached skill objects for the application's artifact/report adapter.

        Call on the controller thread, after a tick or from its terminal flush
        callback. This does not expose a mutable blackboard or a success grade.
        Images/geometry may be present; do not forward this wholesale to a model.
        """
        self._check_open()
        if not self._started:
            return ()
        try:
            store = self._agent.blackboard.get(BB.OBJECT_STORE)
        except KeyError:
            return ()
        return tuple(deepcopy(store.all_objects())) if store is not None else ()

    def _stop_tree(self, *, settle_timeout: float = 2.0):
        self._runner.tree.root.stop(py_trees.common.Status.INVALID)
        try:
            handle = self._agent.blackboard.get(BB.SKILL_HANDLE)
        except KeyError:
            handle = None
        if handle is not None and not handle.is_terminal:
            handle.cancel()
            deadline = time.monotonic() + settle_timeout
            while not handle.is_terminal and time.monotonic() < deadline:
                time.sleep(0.005)
            if not handle.is_terminal:
                raise RuntimeError(
                    "Skill has not acknowledged cancellation; runtime ownership is retained"
                )

    def cancel(self, reason: str = ""):
        # Stop remains available even if an environment change blocks new work.
        if self._closed:
            raise RuntimeError("Runtime is closed")
        if self._started and not self._ended:
            self._agent.request_cancel("operator", reason)
            self._stop_tree()
            self._ended = True

    def run(
        self,
        task: str,
        *,
        scenario: str = "",
        max_ticks: int = 10000,
        tick_interval: float = 0.01,
        platform_state: Mapping | None = None,
    ) -> EpisodeOutcome:
        if isinstance(max_ticks, bool) or not isinstance(max_ticks, int) or max_ticks < 1:
            raise ValueError("max_ticks must be a positive integer")
        if not math.isfinite(tick_interval) or tick_interval < 0:
            raise ValueError("tick_interval must be finite and non-negative")
        self.start(task, scenario=scenario, platform_state=platform_state)
        try:
            for _ in range(max_ticks):
                self.tick()
                if self._ended:
                    submitted = bool(self._agent.blackboard.get(BB.TASK_SUBMITTED))
                    reason = (
                        "submitted"
                        if submitted
                        else "cancelled" if self._runner.is_cancelled else "completed_unsubmitted"
                    )
                    return EpisodeOutcome(reason, submitted, self._runner.tick_count)
                time.sleep(tick_interval)
            self.cancel("tick limit")
            return EpisodeOutcome("tick_limit", False, self._runner.tick_count)
        except BaseException:
            self.cancel("runtime exception")
            raise

    def close(self):
        if not self._closed:
            self._stop_tree()
            self._closed = True
            _release_runtime(self)

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()


def create_agent(*, model, skills, host, profile=DEFAULT_RUNTIME_PROFILE, internal_skills=()):
    """Create the task runtime.

    Select ``paper-core-v1`` explicitly to reproduce the paper controller. That
    profile is an unpolished research snapshot; see ``bt_agent.profiles``.
    """
    core_config(profile)  # Fail closed on typos before choosing an implementation.
    if profile == "task-core-v1":
        from .task_runtime import TaskRuntime

        return TaskRuntime(model=model, skills=skills, host=host, internal_skills=internal_skills)
    return AgentRuntime(
        model=model, skills=skills, host=host, profile=profile, internal_skills=internal_skills
    )
