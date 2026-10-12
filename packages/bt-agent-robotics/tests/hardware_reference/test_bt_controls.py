"""No hardware: real skill BT worker and ExecuteSkill continuation boundary."""

import json
import time

import py_trees
import pytest
from bt_agent._engine.agent.blackboard_keys import BB
from bt_agent._engine.bt.nodes.execute_skill import ExecuteSkill
from bt_agent.skills import SkillContext, SkillStatus
from bt_agent_robotics._hardware.bt_controls import (
    Controller,
    MotionHeld,
    ScopedPlanner,
    instrument_skill,
)
from bt_agent_robotics._hardware.skills.navigate import NavigateSkill
from bt_agent_robotics._hardware.skills.press import PressSkill


class Backend:
    def __init__(self):
        self.pose = (0.0, 0.0, 0.0)
        self.velocity = (0.1, 0.0)
        self.events = []

    def get_robot_pose(self):
        return self.pose

    def get_robot_velocity(self):
        return self.velocity

    def get_pose_age(self):
        return 0.01

    def send_waypoint(self, *point):
        self.events.append(("waypoint", point))

    def send_velocity(self, *velocity):
        self.events.append(("velocity", velocity))

    def set_brake(self, engaged):
        self.events.append(("brake", engaged))

    def cancel(self):
        self.events.append(("cancel",))


class Planner:
    waypoint_threshold = 0.2

    def plan(self, *_):
        return [(3.0, 0.0)]

    def replan(self, *_):
        return self.plan()

    def update_map(self, _):
        pass


class Gate:
    enabled = True

    def __init__(self):
        self.decision = None
        self.plans = []

    def publish_plan(self, *args):
        self.plans.append(args)
        self.decision = None

    def poll(self):
        result, self.decision = self.decision, None
        return result  # noqa: R504 - saved before clearing

    def clear(self):
        pass


def wait(predicate):
    end = time.monotonic() + 2
    while not predicate() and time.monotonic() < end:
        time.sleep(0.005)
    assert predicate()


def setup_skill(monkeypatch, skill_type=NavigateSkill):
    monkeypatch.setenv("LAM_NAV_CONFIRM", "1")
    backend, gate, events = Backend(), Gate(), []
    controller = Controller(backend, emit=events.append)
    skill = instrument_skill(skill_type(backend, Planner(), gate), controller)
    return skill, controller, backend, gate, events


def start(skill, call_id="call1"):
    context = SkillContext(
        skill_call_id=call_id, main_task="test task", platform_state={"robot_position": (0.0, 0.0, 0.0)}
    )
    params = {
        "goal_x": 3.0,
        "goal_y": 0.0,
        "goal_theta": 0.0,
        "target_label": "test object",
        "target_handle": "test1",
        "viewpoint_index": 0,
        "total_viewpoints": 1,
    }
    return skill.start(params, context)


@pytest.mark.parametrize("kind", ["interrupt", "timeout", "failure"])
@pytest.mark.parametrize("skill_type", [NavigateSkill, PressSkill])
def test_interrupt_real_bt_then_agent_continues_and_next_navigation_asks_go(monkeypatch, kind, skill_type):
    skill, controller, backend, gate, events = setup_skill(monkeypatch, skill_type)
    handle = start(skill)
    wait(lambda: bool(gate.plans))
    gate.decision = "go"
    wait(lambda: ("brake", False) in backend.events)
    key = controller.snapshot()["active"]["key"]
    old_scope = ScopedPlanner(controller, key)
    before = len(backend.events)
    assert controller.request(key, kind, "request1")
    assert not controller.request(key, kind, "request1")
    assert handle._completion_event.wait(2)
    assert handle.status == SkillStatus.FAILED
    assert not handle.is_cancel_requested
    if kind == "interrupt":
        assert "operator_interrupt" in handle.observation
    else:
        public = json.dumps({"observation": handle.observation, "result": handle.result_data}).lower()
        for forbidden in ("injected", "intervention", "operator", "choose the next", "task remains active"):
            assert forbidden not in public
        assert handle.result_data["failure_reason"] == (
            "wall_time_exceeded" if kind == "timeout" else "execution_failed"
        )
        diagnostic = next(e for e in events if e["kind"] == "intervention_feedback")
        assert diagnostic["injected"] and diagnostic["actual_leaf"] == "OperatorInterventionGuard"
    assert backend.events[before][0:2] == ("brake", True)
    assert ("brake", False) not in backend.events[before:]
    assert len([e for e in events if e["kind"] == "intervention_consumed_by_bt"]) == 1
    assert controller.snapshot()["active"] is None

    # Actual supervisor boundary: FAILED is fed back and returns SUCCESS to
    # continue SelectSkill. CANCELLED would abort that branch instead.
    node = ExecuteSkill()
    node._completion_bypass = False
    node._skill_name = skill.name
    node._handle = handle
    assert node.update() == py_trees.common.Status.SUCCESS
    assert node.blackboard.get(BB.PENDING_EVENTS)[-1]["data"]["outcome"] == "failed"
    if kind != "interrupt":
        completion = json.dumps(node.blackboard.get(BB.PENDING_EVENTS)[-1]["data"]).lower()
        for forbidden in ("injected", "intervention", "operator", "choose the next", "task remains active"):
            assert forbidden not in completion

    next_handle = start(skill, "call2")
    try:
        wait(lambda: len(gate.plans) == 2)
        after = len(backend.events)
        time.sleep(0.03)
        assert len(backend.events) == after  # no automatic movement or release
        with pytest.raises(MotionHeld):
            old_scope.send_waypoint(9.0, 9.0, 0.0)
        with pytest.raises(MotionHeld):
            old_scope.set_brake(False)
        with pytest.raises(ValueError):
            controller.request(key, "interrupt", "stale_request")
        gate.decision = "go"
        wait(lambda: ("brake", False) in backend.events[after:])
    finally:
        next_handle.cancel()
        assert next_handle._completion_event.wait(2)
    assert [event for event in backend.events if event[0] == "brake"][-1] == ("brake", True)


def test_idle_and_mismatched_requests_do_nothing(monkeypatch):
    skill, controller, backend, gate, _ = setup_skill(monkeypatch)
    with pytest.raises(ValueError):
        controller.request("none", "interrupt", "r")
    assert not backend.events
    handle = start(skill)
    try:
        wait(lambda: bool(gate.plans))
        with pytest.raises(ValueError):
            controller.request("wrong", "interrupt", "r")
        assert ("brake", False) not in backend.events
    finally:
        handle.cancel()
        assert handle._completion_event.wait(2)


def test_stationary_measurement_requires_continuous_low_velocity(monkeypatch):
    skill, controller, _, gate, events = setup_skill(monkeypatch)
    now = [10.0]
    controller.clock = lambda: now[0]
    handle = start(skill)
    wait(lambda: bool(gate.plans))
    key = controller.snapshot()["active"]["key"]
    controller.request(key, "interrupt", "r")
    assert handle._completion_event.wait(2)
    controller.observe_motion(0, 0)
    now[0] += 0.1
    controller.observe_motion(0.1, 0)
    controller.observe_motion(0, 0)
    now[0] += 0.21
    controller.observe_motion(0, 0)
    assert len([e for e in events if e["kind"] == "stationary_confirmed"]) == 1
