"""Red-phase coverage for exceptions escaping a skill subtree tick."""

from __future__ import annotations

from typing import Any, Dict

import py_trees
from bt_agent._engine.skills.protocol import ContextSkill, SkillContext, SkillHandle, SkillStatus


class _CrashingLeaf(py_trees.behaviour.Behaviour):
    def update(self) -> py_trees.common.Status:
        raise RuntimeError("boom")


class _CrashingSkill(ContextSkill):
    name = "crashing_skill"
    description = "Raises from its subtree leaf."
    params: Dict[str, str] = {}

    def build_subtree(
        self, params: Dict[str, Any], context: SkillContext, handle: SkillHandle
    ) -> py_trees.behaviour.Behaviour:
        return _CrashingLeaf(name="crashing_leaf")

    def tool_def(self) -> Dict[str, Any]:
        return super().tool_def()


def test_worker_converts_subtree_exception_to_terminal_failure() -> None:
    handle = _CrashingSkill().start({}, SkillContext())

    completed = handle._completion_event.wait(timeout=2.0)

    assert completed, "skill completion event was not set after its subtree raised"
    assert handle.status == SkillStatus.FAILED
    assert handle.result_data["failure_reason"] == "subtree_exception"
    assert handle._completion_event.is_set()
    assert "RuntimeError" in handle.observation or "boom" in handle.observation


class _TerminateRecorder(py_trees.decorators.Decorator):
    """Stands in for _TerminalDiscipline: records terminate() invocations."""

    def __init__(self, child: py_trees.behaviour.Behaviour) -> None:
        super().__init__(name="terminate_recorder", child=child)
        self.terminations: list = []

    def update(self) -> py_trees.common.Status:
        return self.decorated.status

    def terminate(self, new_status: py_trees.common.Status) -> None:
        self.terminations.append(new_status)


class _SecondTickCrashLeaf(py_trees.behaviour.Behaviour):
    """RUNNING on the first tick, raises on the second — a mid-run crash.

    (A FIRST-tick crash never assigns the decorator a status, and
    Composite.stop(INVALID) skips INVALID children by design — that edge is
    owned and tested by _TerminalDiscipline's own crash guard in
    lam_v2_tiamat. Here we verify the protocol-level stop cascade for a
    subtree that has actually started.)"""

    def __init__(self, name: str = "second_tick_crash") -> None:
        super().__init__(name)
        self._ticks = 0

    def update(self) -> py_trees.common.Status:
        self._ticks += 1
        if self._ticks == 1:
            return py_trees.common.Status.RUNNING
        raise RuntimeError("boom")


class _CrashingSkillWithTerminal(ContextSkill):
    """Mid-run-crashing leaf wrapped in a terminal decorator — the navigate/press shape."""

    name = "crashing_skill_with_terminal"
    description = "Raises from a decorated leaf."
    params: Dict[str, str] = {}

    def __init__(self) -> None:
        super().__init__()
        self.recorder: _TerminateRecorder | None = None

    def build_subtree(
        self, params: Dict[str, Any], context: SkillContext, handle: SkillHandle
    ) -> py_trees.behaviour.Behaviour:
        self.recorder = _TerminateRecorder(_SecondTickCrashLeaf())
        return self.recorder

    def tool_def(self) -> Dict[str, Any]:
        return super().tool_def()


def test_crash_guard_stops_subtree_so_terminal_decorators_fire() -> None:
    """A crashed leaf must still trigger terminate() on enclosing decorators.

    _TerminalDiscipline (navigate/press) does park-at-self + brake in
    terminate(); if the crash guard marks FAILED without stopping the
    subtree, the robot is left un-braked with a latched goal.
    """
    skill = _CrashingSkillWithTerminal()
    handle = skill.start({}, SkillContext())

    assert handle._completion_event.wait(timeout=2.0)
    assert handle.status == SkillStatus.FAILED
    assert skill.recorder is not None
    assert (
        py_trees.common.Status.INVALID in skill.recorder.terminations
    ), "subtree was not stopped: terminal decorators (park+brake) never ran"
