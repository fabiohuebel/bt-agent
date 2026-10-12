"""cmd_vel_mux — priority-based arbitration for ``/spot/cmd_vel``.

Production pattern: exactly ONE node owns the physical-actuator cmd_vel
output. Every controller with cmd_vel intent (path_follower for goal-
directed nav, our PD for move primitives, future teleop, e-stop, etc.)
publishes on a labeled input topic. This node arbitrates among them
each tick and forwards the winner.

Algorithm per tick (50 Hz default):
    0. If the mux brake is engaged, publish a zero Twist — regardless of
       inputs. The mux is BORN braked and releases only on an explicit
       /stop==0 received after its own startup; every nonzero /stop
       re-engages. Latched readback on /lam/mux_brake_state. (2026-08-14
       incident: /stop gates only pathFollower, so stale planner goals and
       primitive publishers could move the robot on mux-up with the
       operator brake engaged. The mux is the final gate, so it brakes.)
    1. Collect inputs whose last message arrived within their per-topic
       timeout window ("fresh").
    2. If none are fresh, publish a zero Twist (safe failsafe).
    3. Otherwise, publish the latest message of the fresh input with the
       highest priority.

Configuration via ROS parameter ``inputs`` — a list of strings in
``name:topic:priority:timeout_s`` form. The default config matches the
lam_v2_tiamat usage:

    navigate:  /spot/cmd_vel_navigate, priority 10, timeout 0.5s
        Long timeout because path_follower can fall silent between
        target waypoints; we want it to remain "active" for the mux
        across short gaps.
    primitive: /spot/cmd_vel_primitive, priority 20, timeout 0.25s
        Short-ish timeout because move skills run to completion (~1–2 s)
        and we want fall-through to navigate once they end. PD ticks at
        up to ~100 Hz (TICK_LOOP_SLEEP=0.01s in framework config), so
        0.25 s = ~25 expected PD periods of margin — tolerates GIL
        contention / sim-time stalls without falling through prematurely
        (which would chase the stale T0 waypoint set by _SnapshotAndCancel).

Extending: append more entries (teleop, e-stop, manual override) to the
``inputs`` parameter. Higher priority wins. Add as many as you need —
the topology is the same.
"""

from __future__ import annotations

import contextlib
import json
import os
import time
from pathlib import Path
from typing import List, Optional

import rclpy
from geometry_msgs.msg import Twist
from bt_agent_robotics._hardware.battery_hold_state import MuxBatteryHold
from bt_agent_robotics._hardware.ros.cmd_vel_mux_logic import MuxInput as _Input
from bt_agent_robotics._hardware.ros.cmd_vel_mux_logic import (
    brake_released_after_stop,
    select_command,
)
from rclpy.node import Node
from rclpy.qos import DurabilityPolicy, QoSProfile
from std_msgs.msg import Int8, String

_DEFAULT_INPUTS = [
    "navigate:/spot/cmd_vel_navigate:10:0.5",
    # Primitive timeout 0.25 s — chosen for safe handover margin over the
    # PD's ~10 ms tick rate (TICK_LOOP_SLEEP=0.01s gives ~100 Hz ceiling).
    # 250 ms tolerates GIL contention / sim-time stalls up to ~15 missed
    # ticks without falling through to navigate (which would chase the
    # stale T0 waypoint set by _SnapshotAndCancel and creep backward).
    "primitive:/spot/cmd_vel_primitive:20:0.25",
]


def _parse(spec: str) -> _Input:
    parts = spec.split(":")
    if len(parts) != 4:
        raise ValueError(f"Bad cmd_vel_mux input spec {spec!r}; expected 'name:topic:priority:timeout_s'.")
    name, topic, pri_s, to_s = parts
    return _Input(
        name=name,
        topic=topic,
        priority=int(pri_s),
        timeout_s=float(to_s),
    )


class CmdVelMux(Node):
    """Priority + freshness arbitration node. Single owner of the output topic."""

    def __init__(self) -> None:
        super().__init__("cmd_vel_mux")
        # Acquire the same-path exclusive owner lock BEFORE creating any output
        # publisher. No file IO or changed behavior when the feature is disabled.
        self._battery_hold = MuxBatteryHold()
        self.declare_parameter("inputs", _DEFAULT_INPUTS)
        self.declare_parameter("output_topic", "/spot/cmd_vel")
        self.declare_parameter("output_rate_hz", 50.0)
        self.declare_parameter("log_transitions", True)

        specs: List[str] = list(self.get_parameter("inputs").value)
        if not specs:
            raise ValueError("cmd_vel_mux: 'inputs' parameter must be non-empty.")
        self._inputs: List[_Input] = [_parse(s) for s in specs]
        for inp in self._inputs:
            # Default-fn capture: bind inp at lambda-definition time.
            self.create_subscription(
                Twist,
                inp.topic,
                lambda msg, bound=inp: self._on_msg(bound, msg),
                10,
            )
            self.get_logger().info(
                f"input '{inp.name}': topic={inp.topic} priority={inp.priority} timeout={inp.timeout_s:.3f}s"
            )

        out_topic: str = self.get_parameter("output_topic").value
        rate_hz: float = float(self.get_parameter("output_rate_hz").value)
        self._log_transitions: bool = bool(self.get_parameter("log_transitions").value)
        self._pub = self.create_publisher(Twist, out_topic, 10)
        self._out_topic = self._pub.topic_name  # resolved/remapped topic for publisher-count checks
        self._battery_status_pub = None
        if self._battery_hold.enabled:
            self._battery_status_pub = self.create_publisher(String, "/lam/battery_hold/mux_status", 1)
            self._owner_pid = os.getpid()
            self._owner_start = Path("/proc/self/stat").read_text().rsplit(")", 1)[1].split()[19]
            self._boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()

        # Fail-closed brake authority (2026-08-14 incident: robot moved on
        # mux-up with the operator brake engaged — /stop gates only
        # pathFollower, and primitive/direct publishers bypass it). The mux
        # is born braked and releases only on /stop==0 received AFTER its
        # own startup; a message published before wiring cannot count.
        self._brake_released = False
        self.create_subscription(Int8, "/stop", self._on_stop, 1)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self._brake_state_pub = self.create_publisher(String, "/lam/mux_brake_state", latched)
        self._publish_brake_state()

        self._timer = self.create_timer(1.0 / max(1.0, rate_hz), self._tick)
        self._zero = Twist()
        self._last_active: Optional[str] = None
        self.get_logger().info(
            f"output: {out_topic} @ {rate_hz:.1f} Hz — BORN BRAKED; zero Twist until /stop==0 arrives post-startup"
        )

    def _on_msg(self, inp: _Input, msg: Twist) -> None:
        inp.last_msg = msg
        inp.last_ts = self.get_clock().now().nanoseconds * 1e-9

    def _on_stop(self, msg: Int8) -> None:
        released = brake_released_after_stop(int(msg.data))
        if self._battery_hold.snapshot()["active"]:
            released = False
        if released != self._brake_released:
            self._brake_released = released
            self.get_logger().info(f"mux brake → {'RELEASED' if released else 'ENGAGED'} (/stop={int(msg.data)})")
            self._publish_brake_state()

    def _publish_brake_state(self) -> None:
        message = String()
        message.data = "RELEASED" if self._brake_released else "ENGAGED"
        self._brake_state_pub.publish(message)

    def _tick(self) -> None:
        hold = self._battery_hold.snapshot()
        if hold["active"]:
            self._brake_released = False
            self._publish_brake_state()
            if hold["phase"] == "suspend":
                # No Twist whatsoever: even a zero velocity command can fight
                # Spot's Sit. The already-braked handoff owns posture now.
                self._publish_battery_status(hold)
                self._note_transition("battery-suspended")
                return
            self._pub.publish(self._zero)
            self._publish_battery_status(hold)
            self._note_transition("battery-stop")
            return
        self._publish_battery_status(hold)
        now = self.get_clock().now().nanoseconds * 1e-9
        winner, publish_zero = select_command(self._inputs, now, self._brake_released)
        if publish_zero:
            self._pub.publish(self._zero)
            self._note_transition("braked" if not self._brake_released else None)
            return
        self._pub.publish(winner.last_msg)
        self._note_transition(winner.name)

    def _publish_battery_status(self, hold):
        if self._battery_status_pub is None:
            return
        count = self.count_publishers(self._out_topic)
        message = String()
        message.data = json.dumps(
            {
                **hold,
                "output_suspended": hold["active"] and hold["phase"] == "suspend" and hold["valid"],
                "monotonic_s": time.monotonic(),
                "owner_pid": self._owner_pid,
                "owner_start": self._owner_start,
                "boot_id": self._boot_id,
                "owner_lock_acquired": self._battery_hold.owns_lock,
                "output_topic": self._out_topic,
                "output_publisher_count": count,
                "exclusive_output_owner": self._battery_hold.owns_lock and count == 1,
            },
            sort_keys=True,
        )
        self._battery_status_pub.publish(message)

    def destroy_node(self):
        result = super().destroy_node()
        self._battery_hold.close()
        return result  # noqa: R504 - release owner lock only after ROS outputs are destroyed

    def _note_transition(self, name: Optional[str]) -> None:
        if not self._log_transitions or name == self._last_active:
            return
        if name is None:
            self.get_logger().info("no fresh input → zero Twist")
        elif name == "braked":
            self.get_logger().info("mux braked → zero Twist (all inputs gated)")
        else:
            self.get_logger().info(f"active input → {name}")
        self._last_active = name


def main(args=None) -> None:
    rclpy.init(args=args)
    node = CmdVelMux()
    with contextlib.suppress(KeyboardInterrupt):
        rclpy.spin(node)
    node.destroy_node()
    with contextlib.suppress(Exception):
        rclpy.shutdown()


if __name__ == "__main__":
    main()
