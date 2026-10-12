"""Explicit hardware-profile composition; no ROS connection or motion on import.

This is an offline-qualified adapter candidate, not a robot bringup program.
The application owns sensor freshness, report durability, mux/lease authority,
and actual physical stopping. Frozen simulator profiles are not modified.
"""

from dataclasses import asdict, dataclass
from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path

from bt_agent import create_agent
from bt_agent.skills import ReportSkill, register_doc_root

from ._hardware.battery_guard import BatteryGuard, BatteryGuardPolicy
from ._hardware.battery_posture import BatteryPostureHandoff, PostureTelemetry
from ._hardware.bt_controls import Controller as InterventionController, instrument_skill


@dataclass(frozen=True)
class HardwareProfile:
    survey_description: str
    local_inspection: bool
    exploration_fallback: bool
    task_contract: str = "general"

    def __post_init__(self):
        if self.survey_description not in ("original", "visual_question_v1"):
            raise ValueError("Explicit supported survey description required")
        if type(self.local_inspection) is not bool or type(self.exploration_fallback) is not bool:
            raise ValueError("Hardware feature choices must be booleans")
        if self.task_contract not in ("general", "capture"):
            raise ValueError("Explicit general or capture task contract required")

    def check_environment(self):
        # Retained deployment switches are checked, never silently set globally.
        expected = {
            "LAM_NAV_CONFIRM": "1",
            "LAM_CAPTURE_LOCAL_INSPECTION": str(int(self.local_inspection)),
            "LAM_FARM_EXPLORATION_FALLBACK": str(int(self.exploration_fallback)),
        }
        for name, value in expected.items():
            if os.environ.get(name) != value:
                raise ValueError(f"Hardware profile requires explicit {name}={value}")
        if os.environ.get("LAM_QUERY_TOPK", "0") != "0":
            raise ValueError("Top-k query is outside this hardware profile")
        if os.environ.get("LAM_SIM_EXPERIMENT_ARM"):
            raise ValueError("Simulator experiment profiles must not bind to hardware")
        timeout = float(os.environ.get("LAM_NAV_CONFIRM_TIMEOUT_S", "300"))
        if not math.isfinite(timeout) or timeout <= 0:
            raise ValueError("Confirmation timeout must be positive and finite")
        return {**expected, "LAM_QUERY_TOPK": "0", "LAM_NAV_CONFIRM_TIMEOUT_S": timeout}


def verify_sources():
    root = Path(__file__).parent
    receipt = json.loads((root / "hardware-source.json").read_text())
    prefix = "packages/bt-agent-robotics/src/bt_agent_robotics/"
    for row in receipt["files"]:
        path = root / row["destination"].removeprefix(prefix)
        if hashlib.sha256(path.read_bytes()).hexdigest() != row["sha256"]:
            raise ValueError("Changed hardware profile source: " + row["destination"])
    return receipt


def _register_docs():
    from bt_agent._engine.skills import _discover_doc_files

    root = Path(__file__).parent / "_hardware/docs"
    present = _discover_doc_files()
    for path in root.glob("*.md"):
        prior = present.get(path.stem)
        if prior and prior.read_text().strip() != path.read_text().strip():
            raise RuntimeError("Conflicting skill docs registered; use a fresh process per profile")
    register_doc_root(root)


class HardwareAgent:
    """Own a frozen task controller with explicit hardware skill/guard bindings.

    Controls and battery adapters outlive individual tasks. Resetting the task
    does not clear their persistent motion holds or claim any physical stop.
    """

    def __init__(
        self,
        *,
        model,
        host,
        navigation,
        path_planning,
        scene_memory,
        perception,
        confirmation_gate,
        profile,
        interventions=None,
        battery=None,
        capture_debug_writer=None,
        survey_debug_writer=None,
    ):
        if not isinstance(profile, HardwareProfile):
            raise TypeError("An explicit HardwareProfile is required")
        self.profile = profile
        self._profile_snapshot = profile
        self.environment = profile.check_environment()
        self.sources = verify_sources()
        if confirmation_gate is None:
            raise ValueError("Supply an explicit operator-confirmation adapter")
        for method in ("publish_plan", "poll", "clear"):
            if not callable(getattr(confirmation_gate, method, None)):
                raise TypeError("Incomplete confirmation adapter")
        if not callable(getattr(navigation, "get_pose_age", None)):
            raise TypeError("Hardware navigation must expose pose freshness")
        if (
            getattr(model, "temperature", None) != 0
            or getattr(model, "_enable_thinking", None) is not False
        ):
            raise ValueError("Hardware paper-Core requires temperature zero and thinking disabled")
        self.confirmation_gate = confirmation_gate
        if interventions is not None and interventions.backend is not navigation:
            raise ValueError("Interventions must fence the exact navigation adapter")
        if battery is not None and battery.planner is not navigation:
            raise ValueError("Use the battery-fenced planner for all motion skills")
        _register_docs()
        from ._hardware.skills.navigate import NavigateSkill
        from ._hardware.skills.press import PressSkill
        from ._hardware.skills.capture import CaptureSkill
        from ._hardware.skills.query import QuerySkill
        from ._hardware.hardware_survey_description import make_survey
        from ._hardware.hardware_execution_guard import install_execution_guard

        motion = [
            NavigateSkill(navigation, path_planning, confirmation_gate),
            PressSkill(navigation, path_planning, confirmation_gate),
        ]
        for skill in motion:
            if interventions is not None:
                instrument_skill(skill, interventions)
            if battery is not None:
                battery.instrument_skill(skill)
        survey = make_survey(perception, profile.survey_description)
        survey._debug_writer = survey_debug_writer
        skills = [
            QuerySkill(scene_memory),
            *motion,
            ReportSkill(),
            survey,
            CaptureSkill(perception, None, debug_writer=capture_debug_writer),
        ]
        # This binding installs a guard on the frozen tree. Never inherit a
        # maintained-core default change as an implicit hardware migration.
        self.runtime = create_agent(model=model, host=host, skills=skills, profile="paper-core-v1")
        try:
            self._expected_tools = deepcopy(self.runtime.describe()["skill_tools"])
            # Official, version-pinned adapter boundary; not a public arbitrary
            # tree-editing hook. Applied before the first task starts.
            self.execution_guard = install_execution_guard(self.runtime._runner.tree.root)
        except BaseException:
            self.runtime.close()
            raise
        self.interventions, self.battery = interventions, battery
        self.perception = perception

    def _check(self):
        try:
            if self.profile != self._profile_snapshot:
                raise ValueError("Declared hardware profile changed after skill construction")
            if (
                self.profile.check_environment() != self.environment
                or not self.confirmation_gate.enabled
            ):
                raise ValueError("Hardware configuration changed during the runtime")
        except Exception:
            self.cancel("hardware configuration changed")
            raise

    def describe(self):
        self._check()
        policy = self.runtime.describe()
        if policy["skill_tools"] != self._expected_tools:
            raise ValueError("Loaded hardware tools changed after profile construction")
        return dict(
            schema="bt-agent-hardware-profile/v1",
            profile=asdict(self.profile),
            policy=policy,
            hardware_reference=self.sources["commit"],
            source_bundle_sha256=hashlib.sha256(
                json.dumps(self.sources, sort_keys=True).encode()
            ).hexdigest(),
            adapter_source_sha256={
                name: hashlib.sha256(Path(__file__).with_name(name).read_bytes()).hexdigest()
                for name in ("hardware.py", "hardware_ros.py", "hardware_inputs.py")
            },
            execution_guard=self.execution_guard,
            settings=self.environment,
            perception_model=getattr(self.perception, "model_name", None),
            interventions=self.interventions is not None,
            battery_guard=self.battery is not None,
            qualification="offline adapter candidate; not revalidated on hardware",
            whole_physical_stack_parity=False,
        )

    def start(self, task, *, scenario="", platform_state=None):
        self._check()
        self._retire_terminal_scope()
        if not isinstance(task, str) or not task.strip():
            raise ValueError("Task must be non-empty text")
        if self.profile.task_contract == "capture":
            from ._hardware.capture_contract import HEADER, TEXT

            task = task + "\n\n" + HEADER + "\n" + TEXT
        return self.runtime.start(task, scenario=scenario, platform_state=platform_state)

    def tick(self):
        self._check()
        self._retire_terminal_scope()
        try:
            return self.runtime.tick()
        finally:
            self._retire_terminal_scope()

    def _retire_terminal_scope(self):
        # A framework safeguard can reject before the instrumented body's first
        # tick. py_trees need not terminate that still-INVALID child, so its
        # build-time scope can remain registered despite a terminal handle.
        # Retire only terminal executions; never clear a live scope/hold here.
        if self.interventions is not None:
            with self.interventions.lock:
                active = self.interventions.active
                if active is not None and active["handle"].is_terminal:
                    self.interventions.finish(active["key"], active["handle"].status)

    def push_state(self, state):
        return self.runtime.push_state(state)

    @property
    def done(self):
        return self.runtime.done

    def cancel(self, reason=""):
        try:
            return self.runtime.cancel(reason)
        finally:
            self._retire_terminal_scope()

    def close(self):
        try:
            return self.runtime.close()
        finally:
            self._retire_terminal_scope()

    def __enter__(self):
        return self

    def __exit__(self, *_):
        self.close()
