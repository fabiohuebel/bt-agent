"""The preserved navigation transport against fake ROS publishers (no ROS running)."""

import importlib
import math
import sys
import threading
from types import ModuleType, SimpleNamespace as NS

import pytest


class PoseStamped:
    def __init__(self):
        self.header = NS(frame_id=None, stamp=None)
        self.pose = NS(position=NS(x=0.0, y=0.0, z=0.0), orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))


class Twist:
    def __init__(self):
        self.linear = NS(x=0.0, y=0.0, z=0.0)
        self.angular = NS(x=0.0, y=0.0, z=0.0)


class Int8:
    data = 0


class Node:
    def __init__(self):
        self.messages, self.depths = {}, {}

    def create_publisher(self, message_type, topic, depth):
        self.messages[topic], self.depths[topic] = [], depth

        def publish(message):
            assert isinstance(message, message_type)
            self.messages[topic].append(message)

        return NS(publish=publish)

    def get_clock(self):
        return NS(now=lambda: NS(to_msg=lambda: "fake-ros-stamp"))


@pytest.fixture
def transport(monkeypatch):
    for package, attrs in {
        "geometry_msgs.msg": {"PoseStamped": PoseStamped, "Twist": Twist},
        "std_msgs.msg": {"Int8": Int8},
        "rclpy.node": {"Node": Node},
    }.items():
        parent = package.split(".")[0]
        monkeypatch.setitem(sys.modules, parent, ModuleType(parent))
        module = ModuleType(package)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, package, module)
    name = "bt_agent_robotics.ros_navigation"
    previous = sys.modules.pop(name, None)
    try:
        module = importlib.import_module(name)
        monkeypatch.setattr(module.time, "monotonic", lambda: 101.0)
        node = Node()
        pose, velocity, stamp = [1.0, 2.0, 0.7], [0.3, 0.1], [100.0]
        planner = module.RosLocalPlanner(node, threading.RLock(), pose, velocity, stamp)
        yield planner, node, pose, velocity, stamp
    finally:
        sys.modules.pop(name, None)
        if previous is not None:
            sys.modules[name] = previous


def test_waypoint_frame_quaternion_and_timestamp(transport):
    planner, node, *_ = transport
    planner.send_waypoint(2.0, 3.0, math.pi / 2)
    msg = node.messages["/goal_waypoint"][-1]
    assert msg.header.frame_id == "map" and msg.header.stamp == "fake-ros-stamp"
    assert (msg.pose.position.x, msg.pose.position.y, msg.pose.position.z) == (2.0, 3.0, 0.0)
    assert msg.pose.orientation.z == pytest.approx(math.sqrt(0.5))
    assert msg.pose.orientation.w == pytest.approx(math.sqrt(0.5))


def test_velocity_remains_an_input_to_the_mux(transport):
    planner, node, *_ = transport
    planner.send_velocity(0.2, -0.1, 0.4)
    assert "/spot/cmd_vel" not in node.messages
    msg = node.messages["/spot/cmd_vel_primitive"][-1]
    assert (msg.linear.x, msg.linear.y, msg.angular.z) == (0.2, -0.1, 0.4)


def test_cancel_replaces_latched_goal_and_publishes_zero(transport):
    planner, node, pose, *_ = transport
    planner.cancel()
    msg = node.messages["/goal_waypoint"][-1]
    assert (msg.pose.position.x, msg.pose.position.y) == tuple(pose[:2])
    cmd = node.messages["/spot/cmd_vel_primitive"][-1]
    assert (cmd.linear.x, cmd.linear.y, cmd.angular.z) == (0.0, 0.0, 0.0)


def test_brake_is_the_path_follower_bitmask(transport):
    planner, node, *_ = transport
    planner.set_brake(True)
    planner.set_brake(False)
    assert [msg.data for msg in node.messages["/stop"]] == [16, 0]
    assert node.depths["/stop"] == 1


def test_shared_pose_velocity_and_monotonic_freshness(transport):
    planner, _, pose, velocity, stamp = transport
    assert planner.get_robot_pose() == tuple(pose)
    assert planner.get_robot_velocity() == tuple(velocity)
    assert planner.get_pose_age() == 1.0
    pose[0] = 5.0
    assert planner.get_robot_pose()[0] == 5.0
    stamp[0] = 0.0
    assert planner.get_pose_age() is None
