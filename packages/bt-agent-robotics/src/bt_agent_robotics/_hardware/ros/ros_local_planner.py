"""ROS2 implementation of LocalPlannerInterface.

Publishes PoseStamped waypoints to ``/goal_waypoint`` (goal-directed nav),
Twist commands to ``/spot/cmd_vel`` (direct velocity control for the
relative-move skills), Int8 brake commands to ``/stop`` (the operator stop
the localPlanner / pathFollower already honor), and reads odometry from a
shared lock-protected reference updated by the bridge node's
``/spot/platform/odom`` subscriber.
"""

from __future__ import annotations

import math
import os
import threading
import time
from typing import List, Optional, Tuple

from geometry_msgs.msg import PoseStamped, Twist
from bt_agent_robotics._hardware.skills.navigate import LocalPlannerInterface
from rclpy.node import Node
from std_msgs.msg import Bool, Int8

# /stop payloads (CMU-stack convention, read by localPlanner + pathFollower):
# any value >= 2 is a hard stop, 0 clears it.
_BRAKE_ENGAGE = 16  # pathFollower /stop is a BITMASK: bit4(16) suppresses cmd publish entirely; 2 only blocks reverse
_BRAKE_RELEASE = 0


class RosLocalPlanner(LocalPlannerInterface):
    """Thin ROS2 wrapper around the Spot local planner."""

    def __init__(
        self,
        node: Node,
        odom_lock: threading.Lock,
        odom_ref: List[float],
        velocity_ref: List[float],
        pose_stamp_ref: Optional[List[float]] = None,
    ) -> None:
        self._node = node
        self._pub = node.create_publisher(PoseStamped, "/goal_waypoint", 10)
        # Move primitives publish on /spot/cmd_vel_primitive, which is a
        # labeled input to cmd_vel_mux. The mux owns /spot/cmd_vel and
        # arbitrates among {navigate, primitive} by priority + freshness
        # (see lam_v2_tiamat/ros/cmd_vel_mux.py). One owner of the actuator
        # output, no race conditions, production-ready architecture.
        self._cmd_vel_pub = node.create_publisher(Twist, "/spot/cmd_vel_primitive", 10)
        # Absolute topic name on purpose: localPlanner/pathFollower run in
        # the /spot namespace but subscribe to "/stop" absolutely, and the
        # launch files never remap it (bringup_local_planner/launch/
        # local_planner_bringup.launch.py). Depth 1 matches the existing
        # in-repo /stop publisher (navigation_graph mobility_nav_node.py).
        self._stop_pub = node.create_publisher(Int8, "/stop", 1)
        self._press_precision_pub = None
        if os.environ.get("LAM_PRESS_PRECISION", "0") == "1":
            from rclpy.duration import Duration
            from rclpy.qos import QoSProfile

            qos = QoSProfile(depth=1, lifespan=Duration(seconds=0.25))
            self._press_precision_pub = node.create_publisher(Bool, "/spot/path_follower/press_precision", qos)
        self._lock = odom_lock
        self._odom = odom_ref  # mutable [x, y, yaw], updated by bridge
        self._vel = velocity_ref  # mutable [linear_speed, angular_speed], updated by bridge
        # mutable [monotonic_seconds] stamped by the bridge's odom callback,
        # or None when the caller doesn't plumb one — freshness checks then
        # degrade gracefully (see navigate.py's _pose_age).
        self._pose_stamp = pose_stamp_ref

    def send_waypoint(self, x: float, y: float, theta: float) -> None:
        msg = PoseStamped()
        msg.header.frame_id = "map"
        msg.header.stamp = self._node.get_clock().now().to_msg()
        msg.pose.position.x = x
        msg.pose.position.y = y
        msg.pose.position.z = 0.0
        # Yaw → quaternion (rotation about Z)
        msg.pose.orientation.z = math.sin(theta / 2.0)
        msg.pose.orientation.w = math.cos(theta / 2.0)
        self._pub.publish(msg)

    def send_velocity(self, vx_body: float, vy_body: float, wz: float) -> None:
        msg = Twist()
        msg.linear.x = float(vx_body)
        msg.linear.y = float(vy_body)
        msg.angular.z = float(wz)
        self._cmd_vel_pub.publish(msg)

    def get_robot_pose(self) -> Tuple[float, float, float]:
        with self._lock:
            return (self._odom[0], self._odom[1], self._odom[2])

    def get_robot_velocity(self) -> Tuple[float, float]:
        with self._lock:
            return (self._vel[0], self._vel[1])

    def cancel(self) -> None:
        pose = self.get_robot_pose()
        self.send_waypoint(*pose)
        # Also emit a zero Twist so any in-flight velocity command from a
        # prior move skill is overridden immediately rather than racing.
        self.send_velocity(0.0, 0.0, 0.0)

    def set_brake(self, engaged: bool) -> None:
        """Publish the operator brake on ``/stop``.

        Engaging it is what actually forbids the chain from producing
        cmd_vel once a skill is done — parking the goal at the robot's own
        pose only removes the *reason* to drive, this removes the ability.
        """
        msg = Int8()
        msg.data = _BRAKE_ENGAGE if engaged else _BRAKE_RELEASE
        self._stop_pub.publish(msg)

    def get_pose_age(self) -> Optional[float]:
        """Seconds since the odom bridge last stamped the shared pose ref.

        Returns ``None`` when no ``pose_stamp_ref`` was plumbed in (or it
        hasn't been stamped yet) — callers degrade gracefully in that case.
        """
        if self._pose_stamp is None:
            return None
        with self._lock:
            stamp = self._pose_stamp[0]
        if stamp <= 0.0:
            return None
        return time.monotonic() - stamp

    def set_press_precision(self, enabled: bool) -> bool:
        """Refresh a 500 ms controller lease for final press only.

        Requires the patched follower. It tightens stopping tolerance, never
        overrides braking, collision checking or the success criterion.
        """
        if self._press_precision_pub is None:
            return True  # Feature disabled: preserve the original backend.
        if enabled and self._press_precision_pub.get_subscription_count() == 0:
            return False
        message = Bool()
        message.data = enabled
        self._press_precision_pub.publish(message)
        return True
