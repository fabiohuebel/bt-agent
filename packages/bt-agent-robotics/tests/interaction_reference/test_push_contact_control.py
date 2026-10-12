import math

import pytest
from bt_agent_robotics._push.skill import _PushMotion


class Planner:
    def __init__(self, pose=(0, 0, 0), age=0):
        self.pose = pose
        self.age = age
        self.ack = False
        self.commands = []

    def cancel(self):
        self.commands.append(("cancel",))

    def set_brake(self, value):
        self.commands.append(("brake", value))

    def request_push_speed(self, *args):
        self.commands.append(("request", *args))

    def push_speed_acknowledged(self, *args):
        return self.ack

    def release_push_speed(self, *args):
        self.commands.append(("release", *args))

    def get_pose_age(self):
        return self.age

    def get_robot_pose(self):
        return self.pose

    def send_waypoint(self, *args):
        self.commands.append(("waypoint", *args))

    def send_velocity(self, *args):
        self.commands.append(("velocity", *args))


def motion(planner, goal):
    m = _PushMotion(planner, contact_control=True)
    m.waypoint(goal, max_speed=0.15)
    return m


def test_no_contact_velocity_before_speed_ack():
    p = Planner()
    m = motion(p, (1, 0, 0))
    m.service(m.acquiring)
    assert not any(x[0] == "velocity" for x in p.commands)
    assert p.commands[-1] == ("brake", True)


def test_world_goal_is_transformed_to_body_velocity():
    p = Planner((0, 0, math.pi / 2))
    m = motion(p, (1, 0, math.pi / 2))
    p.ack = True
    m.service(m.acquiring)
    _, vx, vy, w = p.commands[-1]
    assert abs(vx) < 1e-12 and math.isclose(vy, -0.15) and abs(w) < 1e-12
    assert not any(x[0] == "waypoint" for x in p.commands)


@pytest.mark.parametrize(
    "pose,goal", [((0, 0, 0), (2, 2, math.pi)), ((4, -2, 1.3), (-9, 8, -3)), ((0, 0, 0), (0.01, 0, 0))]
)
def test_contact_commands_respect_speed_and_yaw_limits(pose, goal):
    p = Planner(pose)
    m = motion(p, goal)
    p.ack = True
    m.service(m.acquiring)
    _, vx, vy, w = p.commands[-1]
    assert math.hypot(vx, vy) <= 0.150000000001 and abs(w) <= 0.3


@pytest.mark.parametrize("age", [None, -1, 2])
def test_stale_pose_cannot_produce_contact_motion(age):
    p = Planner(age=age)
    m = motion(p, (1, 0, 0))
    p.ack = True
    with pytest.raises(RuntimeError):
        m.service(m.acquiring)
    assert not any(x[0] == "velocity" for x in p.commands)


def test_nonfinite_pose_cannot_produce_contact_motion():
    p = Planner((float("nan"), 0, 0))
    m = motion(p, (1, 0, 0))
    p.ack = True
    with pytest.raises(RuntimeError):
        m.service(m.acquiring)
    assert not any(x[0] == "velocity" for x in p.commands)


def test_stop_keeps_existing_brake_and_lease_release():
    p = Planner()
    m = motion(p, (1, 0, 0))
    p.ack = True
    m.service(m.acquiring)
    m.stop()
    assert p.commands[-3][0] == "cancel" and p.commands[-2] == ("brake", True) and p.commands[-1][0] == "release"


def test_default_profile_keeps_the_qualified_waypoint_path():
    p = Planner()
    m = _PushMotion(p)
    m.waypoint((1, 0, 0), max_speed=0.15)
    p.ack = True
    m.service(m.acquiring)
    assert p.commands[-1] == ("waypoint", 1.25, 0, 0)
