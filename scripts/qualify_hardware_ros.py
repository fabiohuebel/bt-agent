"""Network-isolated zero-motion ROS probe for the preserved intervention wire.

No driver, robot SDK, simulator, model or final actuator publisher is started.
Only brake, zero velocity and park-at-origin messages may leave this process.
"""

import argparse
import json
import os
from pathlib import Path
import threading
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--domain", type=int, default=97)
    args = parser.parse_args()
    if not Path("/.dockerenv").exists() or set(os.listdir("/sys/class/net")) != {"lo"}:
        raise RuntimeError("Requires Docker --network none; never run in a robot network")
    if (
        not 1 <= args.domain <= 101
        or args.domain == 42
        or os.environ.get("ROS_LOCALHOST_ONLY") != "1"
    ):
        raise RuntimeError("Isolated localhost-only test domain required")
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ["ROS_DOMAIN_ID"] = str(args.domain)
    for key in (
        "FASTRTPS_DEFAULT_PROFILES_FILE",
        "FASTDDS_DEFAULT_PROFILES_FILE",
        "ROS_DISCOVERY_SERVER",
        "CYCLONEDDS_URI",
    ):
        os.environ.pop(key, None)
    import rclpy
    from geometry_msgs.msg import PoseStamped, Twist
    from std_msgs.msg import Int8, String
    from bt_agent.skills import SkillContext, SkillHandle, SkillStatus
    from bt_agent_robotics._hardware.ros.ros_local_planner import RosLocalPlanner
    from bt_agent_robotics._hardware.ros.bt_controls_bridge import BTControlsBridge
    from bt_agent_robotics._hardware.bt_controls import InterventionGuard

    rclpy.init()
    node = rclpy.create_node("release_hardware_wire_probe")
    bridge = None
    result = dict(passed=False, domain=args.domain, physical_motion=False)
    messages = dict(brake=[], velocity=[], goal=[], ack=[])
    try:
        # Fresh test-only pose; never relabel a cached physical sample as fresh.
        stamp = [time.monotonic()]
        planner = RosLocalPlanner(node, threading.RLock(), [0.0, 0.0, 0.0], [0.0, 0.0], stamp)
        node.create_subscription(Int8, "/stop", lambda m: messages["brake"].append(m.data), 10)
        node.create_subscription(
            Twist,
            "/spot/cmd_vel_primitive",
            lambda m: messages["velocity"].append([m.linear.x, m.linear.y, m.angular.z]),
            10,
        )
        node.create_subscription(
            PoseStamped,
            "/goal_waypoint",
            lambda m: messages["goal"].append([m.pose.position.x, m.pose.position.y]),
            10,
        )
        node.create_subscription(
            String, "/lam/bt_controls/ack", lambda m: messages["ack"].append(json.loads(m.data)), 10
        )
        requests = node.create_publisher(String, "/lam/bt_controls/request", 10)
        bridge = BTControlsBridge(node, planner, str(args.out))

        def spin_until(predicate):
            deadline = time.monotonic() + 5
            while not predicate() and time.monotonic() < deadline:
                stamp[0] = time.monotonic()
                rclpy.spin_once(node, timeout_sec=0.02)
            if not predicate():
                raise TimeoutError("ROS probe condition not met")

        spin_until(
            lambda: requests.get_subscription_count() == 1
            and planner._stop_pub.get_subscription_count() == 1
        )
        outcomes = []
        for index, kind in enumerate(("interrupt", "timeout", "failure")):
            handle = SkillHandle()
            handle.status = SkillStatus.RUNNING
            key = bridge.controller.begin(
                "navigate", SkillContext(skill_call_id=str(index), main_task="offline test"), handle
            )
            message = String()
            message.data = json.dumps(dict(key=key, kind=kind, request_id=str(index)))
            requests.publish(message)
            spin_until(lambda: any(a.get("request_id") == str(index) for a in messages["ack"]))
            assert messages["ack"][-1]["ok"]
            guard = InterventionGuard(bridge.controller, key, handle)
            for _ in guard.tick():
                pass
            outcomes.append(
                dict(kind=kind, observation=handle.observation, result=handle.result_data)
            )
            bridge.controller.finish(key, "offline_failure")
        spin_until(
            lambda: len(messages["brake"]) >= 3
            and len(messages["velocity"]) >= 3
            and len(messages["goal"]) >= 3
        )
        if any(value != 16 for value in messages["brake"]) or any(
            any(v != 0 for v in row) for row in messages["velocity"] + messages["goal"]
        ):
            raise AssertionError("Unexpected non-stop output")
        if node.count_publishers("/spot/cmd_vel") or node.count_publishers("/cmd_vel"):
            raise AssertionError("An actuator-output publisher was unexpectedly created")
        result.update(
            passed=True,
            messages=messages,
            outcomes=outcomes,
            journal=str(bridge.journal_path),
            scope="Intervention request/ack and brake/zero/park wire only; battery/mux/Spot sensor wiring not covered",
        )
    finally:
        if bridge is not None:
            node.destroy_timer(bridge.timer)
            bridge.journal.close()
        node.destroy_node()
        rclpy.shutdown()
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(passed=result["passed"], physical_motion=False, domain=args.domain)))


if __name__ == "__main__":
    main()
