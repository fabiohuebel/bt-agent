"""ROS-free tests for the operator confirmation gate in NavigateSkill."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import py_trees
import pytest
from bt_agent._engine.skills.protocol import SkillContext, SkillHandle, SkillStatus
from bt_agent_robotics.skills.navigate import ConfirmGate, NavigateSkill, _ConfirmPath
from bt_agent_robotics.skills.utils.path_planner import PathPlanner


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, seconds: float) -> None:
        self.now += seconds


class StubConfirmGate:
    def __init__(self, responses: Optional[List[Optional[str]]] = None, enabled: bool = True) -> None:
        self.enabled = enabled
        self.responses = list(responses or [])
        self.publish_calls: List[Tuple[List[Tuple[float, float]], str, Tuple[float, float], Optional[int]]] = []
        self.target_image_calls: List[bytes] = []
        self.clear_calls = 0

    def publish_plan(
        self,
        waypoints: List[Tuple[float, float]],
        label: str,
        robot_xy: Tuple[float, float],
        object_id: Optional[int] = None,
    ) -> None:
        if not self.enabled:
            return
        self.publish_calls.append((list(waypoints), label, robot_xy, object_id))
        if object_id is None:
            return
        crops_dir = os.environ.get("LAM_OBJECT_CROPS_DIR")
        if not crops_dir:
            return
        crop_path = Path(crops_dir) / f"object_{object_id:06d}.jpg"
        if crop_path.is_file():
            self.target_image_calls.append(crop_path.read_bytes())

    def poll(self) -> Optional[str]:
        return self.responses.pop(0) if self.responses else None

    def clear(self) -> None:
        if not self.enabled:
            return
        self.clear_calls += 1
        self.target_image_calls.append(b"")


class _LocalPlanner:
    def __init__(
        self,
        pose: Tuple[float, float, float] = (0.0, 0.0, 0.0),
        call_order: Optional[List[Tuple[Any, ...]]] = None,
    ) -> None:
        self.pose = pose
        self.sent_waypoints: List[Tuple[float, float, float]] = []
        self.brake_calls: List[bool] = []
        self.cancelled = False
        self.call_order = call_order

    def send_waypoint(self, x: float, y: float, theta: float) -> None:
        self.sent_waypoints.append((x, y, theta))
        if self.call_order is not None:
            self.call_order.append(("send_waypoint", x, y, theta))

    def send_velocity(self, vx_body: float, vy_body: float, wz: float) -> None:
        pass

    def get_robot_pose(self) -> Tuple[float, float, float]:
        return self.pose

    def get_robot_velocity(self) -> Tuple[float, float]:
        return (0.0, 0.0)

    def get_pose_age(self) -> Optional[float]:
        return None

    def cancel(self) -> None:
        self.cancelled = True

    def set_brake(self, engaged: bool) -> None:
        self.brake_calls.append(engaged)
        if self.call_order is not None:
            self.call_order.append(("set_brake", engaged))


class _FixedPlanner(PathPlanner):
    def __init__(self, waypoints: List[Tuple[float, float]]) -> None:
        self.waypoints = list(waypoints)

    def plan(self, start_xy, goal_xy) -> Optional[List[Tuple[float, float]]]:
        return list(self.waypoints)

    def replan(self, robot_xy, goal_xy) -> Optional[List[Tuple[float, float]]]:
        return list(self.waypoints)

    def update_map(self, map_data: Any) -> None:
        pass

    @property
    def waypoint_threshold(self) -> float:
        return 0.3


def _params() -> Dict[str, Any]:
    return {
        "goal_x": 2.0,
        "goal_y": 1.0,
        "goal_theta": 0.25,
        "target_handle": "obj_1",
        "target_label": "green chair",
        "target_object_id": 42,
        "viewpoint_index": 0,
        "total_viewpoints": 1,
        "is_exploration": False,
    }


def _handle(waypoints: Optional[List[Tuple[float, float]]] = None) -> SkillHandle:
    handle = SkillHandle()
    handle.status = SkillStatus.RUNNING
    if waypoints is not None:
        handle.result_data = {"_waypoints": list(waypoints), "planned_path": list(waypoints)}
    return handle


def _tick(node: py_trees.behaviour.Behaviour) -> py_trees.common.Status:
    for _ in node.tick():
        pass
    return node.status


def test_disabled_gate_is_noop_and_navigate_dispatches_on_first_tick(monkeypatch) -> None:
    monkeypatch.delenv("LAM_NAV_CONFIRM", raising=False)
    waypoints = [(1.0, 0.5), (2.0, 1.0)]
    gate = StubConfirmGate(enabled=True)
    local_planner = _LocalPlanner()
    handle = _handle()
    skill = NavigateSkill(local_planner, _FixedPlanner(waypoints), confirm_gate=gate)
    tree = skill.build_subtree(_params(), SkillContext(platform_state={}), handle)

    assert _tick(tree) == py_trees.common.Status.RUNNING
    assert local_planner.brake_calls == [False]
    assert local_planner.sent_waypoints == [(1.0, 0.5, pytest.approx(0.463647609))]
    assert handle.result_data == {"_waypoints": waypoints, "planned_path": waypoints}
    assert gate.publish_calls == []
    assert gate.target_image_calls == []
    assert gate.clear_calls == 0


def test_enabled_first_tick_publishes_once_and_blocks_waypoint_dispatch(monkeypatch) -> None:
    monkeypatch.setenv("LAM_NAV_CONFIRM", "true")
    waypoints = [(1.0, 0.5), (2.0, 1.0)]
    gate = StubConfirmGate()
    local_planner = _LocalPlanner(pose=(0.25, -0.5, 0.0))
    handle = _handle()
    skill = NavigateSkill(local_planner, _FixedPlanner(waypoints), confirm_gate=gate)
    tree = skill.build_subtree(_params(), SkillContext(platform_state={}), handle)

    assert _tick(tree) == py_trees.common.Status.RUNNING
    assert gate.publish_calls == [(waypoints, "green chair", (0.25, -0.5), 42)]
    assert local_planner.brake_calls == []
    assert local_planner.sent_waypoints == []
    assert "Awaiting operator confirmation" in handle.observation

    assert _tick(tree) == py_trees.common.Status.RUNNING
    assert len(gate.publish_calls) == 1
    assert local_planner.sent_waypoints == []


def test_enabled_publish_plan_publishes_target_crop_bytes(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LAM_NAV_CONFIRM", "true")
    monkeypatch.setenv("LAM_OBJECT_CROPS_DIR", str(tmp_path))
    crop_bytes = b"\xff\xd8farm-target\xff\xd9"
    (tmp_path / "object_000042.jpg").write_bytes(crop_bytes)
    gate = StubConfirmGate()
    tree = NavigateSkill(
        _LocalPlanner(),
        _FixedPlanner([(2.0, 1.0)]),
        confirm_gate=gate,
    ).build_subtree(_params(), SkillContext(platform_state={}), _handle())

    assert _tick(tree) == py_trees.common.Status.RUNNING
    assert gate.target_image_calls == [crop_bytes]


def test_missing_target_crop_does_not_interrupt_confirmation(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("LAM_NAV_CONFIRM", "true")
    monkeypatch.setenv("LAM_OBJECT_CROPS_DIR", str(tmp_path))
    gate = StubConfirmGate()
    tree = NavigateSkill(
        _LocalPlanner(),
        _FixedPlanner([(2.0, 1.0)]),
        confirm_gate=gate,
    ).build_subtree(_params(), SkillContext(platform_state={}), _handle())

    assert _tick(tree) == py_trees.common.Status.RUNNING
    assert gate.publish_calls[-1][-1] == 42
    assert gate.target_image_calls == []


def test_publish_target_image_reads_expected_file(monkeypatch, tmp_path) -> None:
    crop_bytes = b"\xff\xd8production-gate\xff\xd9"
    (tmp_path / "object_000042.jpg").write_bytes(crop_bytes)
    monkeypatch.setenv("LAM_OBJECT_CROPS_DIR", str(tmp_path))

    class _Message:
        format = ""
        data = b""

    class _Publisher:
        def __init__(self) -> None:
            self.messages = []

        def publish(self, message) -> None:
            self.messages.append(message)

    gate = ConfirmGate.__new__(ConfirmGate)
    gate.enabled = True
    gate._compressed_image_type = _Message
    gate._target_image_pub = _Publisher()

    gate.publish_target_image(42)

    assert len(gate._target_image_pub.messages) == 1
    assert gate._target_image_pub.messages[0].format == "jpeg"
    assert gate._target_image_pub.messages[0].data == crop_bytes


def test_clear_blanks_target_image() -> None:
    gate = StubConfirmGate()

    gate.clear()

    assert gate.target_image_calls == [b""]


def test_go_on_third_tick_extends_budget_and_clears(monkeypatch) -> None:
    clock = _FakeClock(10.0)
    monkeypatch.setattr("bt_agent_robotics.skills.navigate.time.monotonic", clock)
    waypoints = [(1.0, 0.5), (2.0, 1.0)]
    gate = StubConfirmGate([None, "go"])
    local_planner = _LocalPlanner()
    handle = _handle(waypoints)
    handle.wall_time_budget_seconds = 60.0
    node = _ConfirmPath(_params(), handle, gate, local_planner)

    assert _tick(node) == py_trees.common.Status.RUNNING
    post_plan_budget = handle.wall_time_budget_seconds
    clock.advance(1.25)
    assert _tick(node) == py_trees.common.Status.RUNNING
    clock.advance(1.0)
    assert _tick(node) == py_trees.common.Status.SUCCESS

    assert handle.result_data["confirm_wait_s"] == 2.25
    assert handle.wall_time_budget_seconds > post_plan_budget
    assert gate.clear_calls == 1
    assert "Operator confirmed path" in handle.observation


def test_go_dispatches_before_release_then_follow_resends_waypoint_zero(monkeypatch) -> None:
    monkeypatch.setenv("LAM_NAV_CONFIRM", "1")
    waypoints = [(1.0, 0.5), (2.0, 1.0)]
    gate = StubConfirmGate(["go"])
    call_order: List[Tuple[Any, ...]] = []
    local_planner = _LocalPlanner(call_order=call_order)
    skill = NavigateSkill(local_planner, _FixedPlanner(waypoints), confirm_gate=gate)
    tree = skill.build_subtree(_params(), SkillContext(platform_state={}), _handle())

    assert _tick(tree) == py_trees.common.Status.RUNNING
    assert call_order == []
    assert _tick(tree) == py_trees.common.Status.RUNNING

    assert call_order[0][:3] == ("send_waypoint", 1.0, 0.5)
    assert call_order[0][3] == pytest.approx(0.463647609)
    assert call_order[1] == ("set_brake", False)
    assert call_order[2][:3] == ("send_waypoint", 1.0, 0.5)
    assert call_order[2][3] == pytest.approx(0.463647609)
    assert len(call_order) == 3


def test_abort_fails_before_follow_and_terminal_discipline_parks_and_brakes(monkeypatch) -> None:
    monkeypatch.setenv("LAM_NAV_CONFIRM", "1")
    waypoints = [(1.0, 0.5), (2.0, 1.0)]
    gate = StubConfirmGate(["abort"])
    local_planner = _LocalPlanner(pose=(0.25, -0.5, 0.1))
    handle = _handle()
    skill = NavigateSkill(local_planner, _FixedPlanner(waypoints), confirm_gate=gate)
    tree = skill.build_subtree(_params(), SkillContext(platform_state={}), handle)

    assert _tick(tree) == py_trees.common.Status.RUNNING
    assert local_planner.sent_waypoints == []
    assert _tick(tree) == py_trees.common.Status.FAILURE

    assert handle.result_data["arrival_status"] == "failed_operator_abort"
    assert local_planner.sent_waypoints == [(0.25, -0.5, 0.1)]
    assert local_planner.brake_calls == [True]
    assert gate.clear_calls == 1


def test_confirmation_timeout_never_releases_and_terminal_engages_brake(monkeypatch) -> None:
    clock = _FakeClock()
    monkeypatch.setattr("bt_agent_robotics.skills.navigate.time.monotonic", clock)
    monkeypatch.setenv("LAM_NAV_CONFIRM", "1")
    monkeypatch.setenv("LAM_NAV_CONFIRM_TIMEOUT_S", "0.1")
    gate = StubConfirmGate()
    local_planner = _LocalPlanner(pose=(0.25, -0.5, 0.1))
    handle = _handle()
    skill = NavigateSkill(local_planner, _FixedPlanner([(2.0, 1.0)]), confirm_gate=gate)
    tree = skill.build_subtree(_params(), SkillContext(platform_state={}), handle)

    assert _tick(tree) == py_trees.common.Status.RUNNING
    assert local_planner.brake_calls == []
    clock.advance(0.11)
    assert _tick(tree) == py_trees.common.Status.FAILURE

    assert handle.result_data["arrival_status"] == "failed_operator_abort"
    assert handle.observation == "Operator confirmation for path to 'green chair' timed out after 0.1s."
    assert local_planner.sent_waypoints == [(0.25, -0.5, 0.1)]
    assert local_planner.brake_calls == [True]
    assert gate.clear_calls == 1


def test_cancellation_during_confirmation_never_releases_and_terminal_engages_brake(monkeypatch) -> None:
    monkeypatch.setenv("LAM_NAV_CONFIRM", "1")
    gate = StubConfirmGate()
    local_planner = _LocalPlanner(pose=(0.25, -0.5, 0.1))
    handle = _handle()
    skill = NavigateSkill(local_planner, _FixedPlanner([(2.0, 1.0)]), confirm_gate=gate)
    tree = skill.build_subtree(_params(), SkillContext(platform_state={}), handle)

    assert _tick(tree) == py_trees.common.Status.RUNNING
    assert local_planner.brake_calls == []
    handle.cancel()
    tree.stop(py_trees.common.Status.INVALID)

    assert local_planner.sent_waypoints == [(0.25, -0.5, 0.1)]
    assert local_planner.brake_calls == [True]
    assert gate.clear_calls == 1


def test_disabled_gate_preserves_waypoint_and_brake_lifecycle_sequence(monkeypatch) -> None:
    monkeypatch.delenv("LAM_NAV_CONFIRM", raising=False)
    call_order: List[Tuple[Any, ...]] = []
    local_planner = _LocalPlanner(pose=(0.25, -0.5, 0.1), call_order=call_order)
    skill = NavigateSkill(local_planner, _FixedPlanner([(1.0, 0.5)]), confirm_gate=StubConfirmGate())
    tree = skill.build_subtree(_params(), SkillContext(platform_state={}), _handle())

    assert _tick(tree) == py_trees.common.Status.RUNNING
    tree.stop(py_trees.common.Status.INVALID)

    assert call_order == [
        ("set_brake", False),
        ("send_waypoint", 1.0, 0.5, 0.25),
        ("send_waypoint", 0.25, -0.5, 0.1),
        ("set_brake", True),
    ]


@pytest.mark.parametrize("value", ["1", "true", "TRUE "])
def test_truthy_environment_enables_gate(monkeypatch, value: str) -> None:
    monkeypatch.setenv("LAM_NAV_CONFIRM", value)
    gate = StubConfirmGate(enabled=False)
    NavigateSkill(_LocalPlanner(), _FixedPlanner([(1.0, 1.0)]), confirm_gate=gate)
    assert gate.enabled is True


@pytest.mark.parametrize("value", [None, "", "0", "false"])
def test_falsey_environment_disables_gate(monkeypatch, value: Optional[str]) -> None:
    if value is None:
        monkeypatch.delenv("LAM_NAV_CONFIRM", raising=False)
    else:
        monkeypatch.setenv("LAM_NAV_CONFIRM", value)
    gate = StubConfirmGate(enabled=True)
    NavigateSkill(_LocalPlanner(), _FixedPlanner([(1.0, 1.0)]), confirm_gate=gate)
    assert gate.enabled is False
