"""Fault injection at the ROS object boundary; no ROS processes or robot."""

import sys
from types import ModuleType, SimpleNamespace as NS

import pytest

from bt_agent_robotics.hardware_ros import RosHardwareIO
from bt_agent_robotics.platform import CAMERAS


class Node:
    def __init__(self, fail_at=None):
        self.calls, self.live, self.fail_at, self.destroy_failure = 0, [], fail_at, None

    def _create(self):
        self.calls += 1
        if self.calls == self.fail_at:
            raise RuntimeError("injected creation failure")
        resource = object()
        self.live.append(resource)
        return resource

    def create_publisher(self, *args, **kwargs):
        return self._create()

    create_subscription = create_publisher

    def destroy_subscription(self, item):
        if item is self.destroy_failure:
            raise RuntimeError("injected destroy failure")
        self.live.remove(item)

    destroy_publisher = destroy_subscription


@pytest.fixture
def ros_types(monkeypatch):
    for name, attrs in {
        "nav_msgs.msg": {"Odometry": object},
        "sensor_msgs.msg": {"CompressedImage": object},
        "std_msgs.msg": {"String": object},
        "tf2_msgs.msg": {"TFMessage": object},
        "rclpy.qos": {
            "QoSProfile": lambda **kw: NS(**kw),
            "DurabilityPolicy": NS(TRANSIENT_LOCAL=1),
            "ReliabilityPolicy": NS(RELIABLE=1, BEST_EFFORT=2),
        },
    }.items():
        monkeypatch.setitem(sys.modules, name.split(".")[0], ModuleType(name.split(".")[0]))
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)

    class Planner:
        def __init__(self, node, lock, pose, velocity, stamp):
            self.pose, self.events = pose, []
            self._pub = node.create_publisher()
            self._cmd_vel_pub = node.create_publisher()
            self._stop_pub = node.create_publisher()
            self._press_precision_pub = None

        def get_robot_pose(self):
            return tuple(self.pose)

        def send_waypoint(self, *args):
            self.events.append(("waypoint", args))

        def send_velocity(self, *args):
            self.events.append(("velocity", args))

        def set_brake(self, value):
            self.events.append(("brake", value))

        def cancel(self):
            self.send_waypoint(*self.get_robot_pose())
            self.send_velocity(0, 0, 0)

    module = ModuleType("bt_agent_robotics._hardware.ros.ros_local_planner")
    module.RosLocalPlanner = Planner
    monkeypatch.setitem(sys.modules, module.__name__, module)


def make(node, tmp_path):
    return RosHardwareIO(
        node,
        epoch="test",
        evidence=tmp_path / "io",
        pose_frame="map",
        body_frame="body",
        camera_frames={c: c + "_optical" for c in CAMERAS},
    )


@pytest.mark.parametrize("fail_at", range(1, 13))
def test_partial_constructor_failure_cleans_up_owned_ros_objects(ros_types, tmp_path, fail_at):
    node = Node(fail_at)
    with pytest.raises(RuntimeError, match="creation failure"):
        make(node, tmp_path)
    assert not node.live


def test_close_retries_failed_cleanup_and_closes_journal(ros_types, tmp_path):
    node = Node()
    io = make(node, tmp_path)
    node.destroy_failure = node.live[0]
    with pytest.raises(RuntimeError):
        io.close()
    assert node.live == [node.destroy_failure]
    assert io.journal.closed and io._closed
    node.destroy_failure = None
    io.close()
    io.close()
    assert not node.live


def test_cancel_without_pose_still_sends_stop(ros_types, tmp_path):
    node = Node()
    io = make(node, tmp_path)
    try:
        io.navigation.cancel()
        assert ("brake", True) in io.navigation.events
        assert ("velocity", (0, 0, 0)) in io.navigation.events
        assert not any(event[0] == "waypoint" for event in io.navigation.events)
    finally:
        io.close()


def test_stale_pose_is_not_available_for_parking(ros_types, tmp_path):
    io = make(Node(), tmp_path)
    try:
        io.platform.clock = lambda: 10.0
        io.platform.odometry([1, 2], [0, 0, 0, 1], [0, 0], 0, epoch="test")
        io.platform.clock = lambda: 20.0
        with pytest.raises(RuntimeError, match="pose"):
            io.navigation.get_robot_pose()
    finally:
        io.close()


def test_cleanup_does_not_destroy_caller_resources(ros_types, tmp_path):
    node = Node()
    unrelated = node.create_publisher()
    io = make(node, tmp_path)
    io.close()
    assert node.live == [unrelated]


def test_cancel_attempts_zero_even_if_brake_publisher_fails(ros_types, tmp_path, monkeypatch):
    io = make(Node(), tmp_path)

    def fail(*args):
        raise OSError("brake publisher unavailable")

    monkeypatch.setattr(io.navigation, "set_brake", fail)
    try:
        with pytest.raises(RuntimeError, match="cancellation output failed"):
            io.navigation.cancel()
        assert ("velocity", (0, 0, 0)) in io.navigation.events
    finally:
        io.close()


def test_failed_parking_attempt_engages_brake_and_sends_zero(ros_types, tmp_path, monkeypatch):
    io = make(Node(), tmp_path)
    io.platform.clock = lambda: 10.0
    io.platform.odometry([1, 2], [0, 0, 0, 1], [0, 0], 0, epoch="test")

    def fail(*args):
        raise OSError("waypoint publisher unavailable")

    monkeypatch.setattr(io.navigation, "send_waypoint", fail)
    try:
        with pytest.raises(RuntimeError):
            io.navigation.cancel()
        assert ("brake", True) in io.navigation.events
        assert ("velocity", (0, 0, 0)) in io.navigation.events
    finally:
        io.close()


def test_nonfinite_pose_clock_is_not_fresh(ros_types, tmp_path):
    io = make(Node(), tmp_path)
    try:
        io.platform.clock = lambda: 10.0
        io.platform.odometry([1, 2], [0, 0, 0, 1], [0, 0], 0, epoch="test")
        io.platform.clock = lambda: float("nan")
        with pytest.raises(RuntimeError):
            io.navigation.get_robot_pose()
    finally:
        io.close()
