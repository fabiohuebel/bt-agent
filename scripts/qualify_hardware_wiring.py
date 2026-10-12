"""Typed ROS/mux/battery qualification with synthetic telemetry and a fake Sit.

Only runs inside Docker --network none. The real mux publishes to a test topic;
no robot driver, SDK, GPU, model server or simulator is involved.
"""

import argparse
import json
import os
from pathlib import Path
import signal
import subprocess
import sys
import time
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--case",
        choices=(
            "critical",
            "lease-denied",
            "stale-battery",
            "invalid-battery",
            "duplicate-output",
            "sit-denied",
            "sit-unavailable",
        ),
        required=True,
    )
    args = parser.parse_args()
    if not Path("/.dockerenv").exists() or set(os.listdir("/sys/class/net")) != {"lo"}:
        raise RuntimeError("Requires isolated Docker --network none")
    if os.environ.get("ROS_DOMAIN_ID") != "97" or os.environ.get("ROS_LOCALHOST_ONLY") != "1":
        raise RuntimeError("Requires test domain 97 and localhost-only discovery")
    args.out.mkdir(parents=True, exist_ok=False)
    for key in (
        "FASTRTPS_DEFAULT_PROFILES_FILE",
        "FASTDDS_DEFAULT_PROFILES_FILE",
        "ROS_DISCOVERY_SERVER",
        "CYCLONEDDS_URI",
    ):
        os.environ.pop(key, None)
    topic = "/release/test/cmd_vel"
    hold_path = (args.out / "hold.json").absolute()
    os.environ.update(
        LAM_BATTERY_GUARD="1",
        LAM_BATTERY_HOLD_FILE=str(hold_path),
        LAM_BATTERY_ACTUATOR_TOPIC=topic,
        LAM_PRESS_PRECISION="1",
    )
    import py_trees
    import rclpy
    from geometry_msgs.msg import Twist, TransformStamped
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import Int8, String, Bool
    from std_srvs.srv import Trigger
    from tf2_msgs.msg import TFMessage
    from spot_msgs.msg import (
        BatteryState,
        BatteryStateArray,
        FootState,
        FootStateArray,
        LeaseResource,
        LeaseArray,
    )
    from rclpy.qos import QoSProfile, DurabilityPolicy
    from bt_agent_robotics._hardware.battery_hold_state import atomic_write_hold, read_hold
    from bt_agent_robotics._hardware.ros.battery_guard_bridge import BatteryGuardBridge
    from bt_agent_robotics.hardware_ros import RosHardwareIO

    atomic_write_hold(hold_path, {"active": False})
    rclpy.init(args=["--ros-args", "-r", "/sit:=/release/test/sit"])
    node = rclpy.create_node("release_hardware_wiring")
    result = dict(passed=False, case=args.case, physical_motion=False, spot_message_shims=False)
    mux = bridge = io = report_episode = None
    logs, outputs, sit_calls, received_precision = [], [], [], []
    frames = {
        name: name + "_optical" for name in ("frontleft", "frontright", "left", "right", "back")
    }
    feed_battery, guard_running = [True], [False]
    invalid_battery = [False]
    feed_velocity = [0.3]
    handle = SimpleNamespace(result_data={}, observation="")
    tree = None
    try:
        io = RosHardwareIO(
            node,
            epoch="offline-" + args.case,
            evidence=args.out / "io",
            pose_frame="map",
            body_frame="body",
            camera_frames=frames,
        )
        bridge = BatteryGuardBridge(node, io.navigation, str(args.out / "battery"))
        pubs = {
            "battery": node.create_publisher(BatteryStateArray, "/status/battery_states", 10),
            "feet": node.create_publisher(FootStateArray, "/status/feet", 10),
            "lease": node.create_publisher(LeaseArray, "/status/leases", 10),
            "odom": node.create_publisher(Odometry, "/spot/platform/odom", 10),
            "input": node.create_publisher(Twist, "/spot/cmd_vel_primitive", 10),
            "stop": node.create_publisher(Int8, "/stop", 10),
        }
        cameras = {
            name: node.create_publisher(
                CompressedImage, f"/spot/camera/{name}/image/compressed", 10
            )
            for name in frames
        }
        transforms = node.create_publisher(
            TFMessage,
            "/tf_static",
            QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL),
        )
        node.create_subscription(
            Twist,
            topic,
            lambda m: outputs.append((time.monotonic(), m.linear.x, m.linear.y, m.angular.z)),
            100,
        )
        node.create_subscription(
            Bool,
            "/spot/path_follower/press_precision",
            lambda m: received_precision.append((time.monotonic(), m.data)),
            10,
        )

        def fake_sit(_request, response):
            sit_calls.append(time.monotonic())
            response.success = args.case != "sit-denied"
            response.message = "OFFLINE FAKE SERVICE ONLY"
            return response

        if args.case != "sit-unavailable":
            node.create_service(Trigger, "/release/test/sit", fake_sit)

        class Body(py_trees.behaviour.Behaviour):
            def update(self):
                return py_trees.common.Status.RUNNING

        class FakeSkill:
            name = "navigate"

            def build_subtree(self, *unused):
                return Body("offline_running_body")

        skill = FakeSkill()
        bridge.instrument_skill(skill)

        import cv2
        import numpy as np

        ok, encoded = cv2.imencode(".jpg", np.full((24, 32, 3), 100, dtype=np.uint8))
        assert ok
        jpeg = encoded.tobytes()
        tf = TFMessage()
        for frame in frames.values():
            t = TransformStamped()
            t.header.frame_id, t.child_frame_id = "body", frame
            t.transform.rotation.w = 1.0
            tf.transforms.append(t)

        def feed():
            stamp = node.get_clock().now().to_msg()
            if feed_battery[0]:
                b = BatteryState()
                b.header.stamp, b.charge_percentage, b.status = (
                    stamp,
                    90.0,
                    BatteryState.STATUS_DISCHARGING,
                )
                if invalid_battery[0]:
                    b.charge_percentage, b.status = 0.0, BatteryState.STATUS_UNKNOWN
                msg = BatteryStateArray()
                msg.battery_states = [b]
                pubs["battery"].publish(msg)
            lease = LeaseResource()
            lease.lease.resource = "body"
            lease.lease_owner.client_name = (
                "other-controller" if args.case == "lease-denied" else "ros_spot"
            )
            leases = LeaseArray()
            leases.resources = [lease]
            pubs["lease"].publish(leases)
            foot = FootState()
            foot.contact = FootState.CONTACT_MADE
            foot.foot_position_rt_body.z = 0.1 if sit_calls and args.case != "sit-denied" else 0.5
            feet = FootStateArray()
            feet.states = [foot] * 4
            pubs["feet"].publish(feet)
            odom = Odometry()
            odom.header.stamp, odom.header.frame_id = stamp, "map"
            odom.pose.pose.orientation.w = 1.0
            pubs["odom"].publish(odom)
            for name, pub in cameras.items():
                image = CompressedImage()
                image.header.stamp, image.header.frame_id = stamp, frames[name]
                image.format, image.data = "jpeg", jpeg
                pub.publish(image)
            transforms.publish(tf)
            intent = Twist()
            intent.linear.x = feed_velocity[0]
            pubs["input"].publish(intent)
            if guard_running[0]:
                tree.tick_once()
                if tree.status == py_trees.common.Status.FAILURE:
                    guard_running[0] = False

        timer = node.create_timer(0.05, feed)

        def wait(predicate, seconds=5):
            deadline = time.monotonic() + seconds
            while not predicate() and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.01)
            if not predicate():
                raise TimeoutError(
                    f"condition timeout: {bridge.handoff.state}; {bridge.telemetry()}; {io.readiness()}"
                )

        def spin(seconds):
            end = time.monotonic() + seconds
            wait(lambda: time.monotonic() >= end, seconds + 1)

        def start_mux(suffix):
            log = (args.out / ("mux-" + suffix + ".log")).open("xb")
            logs.append(log)
            return subprocess.Popen(
                [
                    sys.executable,
                    "-m",
                    "bt_agent_robotics._hardware.ros.cmd_vel_mux",
                    "--ros-args",
                    "-p",
                    "output_topic:=" + topic,
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
                start_new_session=True,
            )

        def stop_mux(process):
            process.send_signal(signal.SIGINT)
            process.wait(timeout=5)
            if process.returncode not in (0, -signal.SIGINT):
                raise RuntimeError("Unclean mux exit: " + str(process.returncode))

        mux = start_mux("initial")
        wait(
            lambda: io.readiness()["ready"]
            and bridge.policy.can_start_motion()
            and bridge.mux is not None
            and len(outputs) >= 3
        )
        assert outputs and all(
            all(v == 0 for v in row[1:]) for row in outputs
        ), "Mux must start braked"
        # Typed messages must really reach their original callbacks, not shims.
        assert bridge.policy.snapshot()["real_percent"] == 90.0 and bridge.feet[0] == "standing"
        assert bridge.lease[0] is (args.case != "lease-denied")
        # Precision wire has bounded ROS lifespan; geometry/controller behavior
        # is not inferred from acknowledgment of this Bool heartbeat.
        wait(lambda: io.navigation._press_precision_pub.get_subscription_count() == 1)
        assert io.navigation.set_press_precision(True)
        wait(lambda: received_precision and received_precision[-1][1])
        assert io.navigation._press_precision_pub.qos_profile.lifespan.nanoseconds == 250_000_000
        io.navigation.set_press_precision(False)
        wait(lambda: received_precision[-1][1] is False)

        bridge.policy.set_override(9.0)
        rejected = skill.build_subtree({}, SimpleNamespace(skill_call_id="admission"), handle)
        rejected.tick_once()
        assert (
            rejected.status == py_trees.common.Status.FAILURE
            and "was not started" in handle.observation
        )
        assert not sit_calls
        bridge.policy.clear_override()
        tree = skill.build_subtree({}, SimpleNamespace(skill_call_id="running"), handle)
        guard_running[0] = True
        release = Int8()
        release.data = 0
        pubs["stop"].publish(release)
        wait(lambda: any(row[1] != 0 for row in outputs))  # test topic only

        duplicate = (
            node.create_publisher(Twist, topic, 10) if args.case == "duplicate-output" else None
        )
        if duplicate is not None:
            wait(lambda: bridge.mux is None)
        if args.case == "stale-battery":
            feed_battery[0] = False
        elif args.case == "invalid-battery":
            invalid_battery[0] = True
        else:
            bridge.policy.set_override(5.0)
        wait(lambda: bridge.policy.snapshot()["held"], 5)
        held_at = time.monotonic()
        wait(lambda: not guard_running[0])
        assert "BatteryReserveGuard" in handle.observation
        for method, params in (
            ("set_brake", (False,)),
            ("send_waypoint", (1.0, 0.0, 0.0)),
            ("send_velocity", (0.2, 0.0, 0.0)),
        ):
            try:
                getattr(bridge.planner, method)(*params)
            except RuntimeError:
                pass
            else:
                raise AssertionError("Held planner accepted motion: " + method)
        pubs["stop"].publish(release)  # racing GO must not release the hold
        if args.case == "critical":
            wait(lambda: bridge.handoff.state == "seated_held", 8)
            assert len(sit_calls) == 1
            bridge.policy.clear_override()
            spin(0.3)
            count = len(outputs)
            spin(0.4)
            assert len(outputs) == count, "Mux must emit no Twist during posture handoff"
            stop_mux(mux)
            mux = start_mux("restart")
            # Clearing the coordinator file cannot clear the persisted mux latch.
            atomic_write_hold(hold_path, {"active": False})
            spin(0.8)
            assert mux.poll() is None and len(outputs) == count
            assert read_hold(hold_path)["active"] and read_hold(hold_path)["phase"] == "suspend"
            assert len(sit_calls) == 1, "Restart must not retry Sit"
        elif args.case in ("sit-denied", "sit-unavailable"):
            wait(lambda: bridge.handoff.state == "held_error")
            spin(0.4)
            assert len(sit_calls) == (1 if args.case == "sit-denied" else 0)
        else:
            spin(0.6)
            assert not sit_calls, "No Sit without valid battery and exclusive ownership"
        # Exercise the real task controller while the independent motion hold
        # remains latched. The model is a named offline fixture, not inference.
        from bt_agent.testing import ScriptedModel
        from bt_agent_robotics.hardware import HardwareProfile
        from bt_agent_robotics.hardware_ros import HardwareROSEpisode

        os.environ.update(
            LAM_NAV_CONFIRM="1",
            LAM_CAPTURE_LOCAL_INSPECTION="1",
            LAM_FARM_EXPLORATION_FALLBACK="0",
            LAM_QUERY_TOPK="0",
        )
        actor = ScriptedModel(
            [
                (
                    "report",
                    {"text": "Motion is blocked pending operator recovery."},
                    "Report the constraint.",
                ),
                ("claim_complete", {}, "The constraint has been reported."),
            ]
        )
        actor.temperature, actor._enable_thinking = 0, False
        gate = SimpleNamespace(
            enabled=True, publish_plan=lambda *_: None, poll=lambda: None, clear=lambda: None
        )
        report_episode = HardwareROSEpisode(
            io,
            model=actor,
            path_planning=None,
            scene_memory=None,
            perception=None,
            confirmation_gate=gate,
            profile=HardwareProfile("original", True, False),
            battery=bridge,
        )
        report_episode.start("Report the current motion constraint.")

        def reported():
            if not report_episode.agent.done or not io.publication.terminal_delivered:
                report_episode.tick()
            return report_episode.host.submitted and io.publication.terminal_delivered

        wait(reported)
        assert io.publication.reports == ["Motion is blocked pending operator recovery."]
        assert read_hold(hold_path)["active"]
        assert all(all(v == 0 for v in row[1:]) for row in outputs if row[0] > held_at + 0.1)
        assert not node.count_publishers("/spot/cmd_vel") and not node.count_publishers("/cmd_vel")
        result.update(
            passed=True,
            typed_messages=True,
            io_ready=True,
            agent_reported_while_held=True,
            phase=bridge.handoff.state,
            sit_calls=len(sit_calls),
            persistent_hold=read_hold(hold_path),
            precision_messages=received_precision,
            outputs=outputs,
            bt_failure=handle.observation,
            journal=str(bridge.journal_path),
            scope="Synthetic telemetry, real ROS serialization/mux and fake Sit; not physical stopping or posture evidence",
        )
    except BaseException as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if report_episode is not None:
            report_episode.agent.close()
        if mux is not None and mux.poll() is None:
            mux.send_signal(signal.SIGINT)
            try:
                mux.wait(timeout=5)
            except subprocess.TimeoutExpired:
                mux.kill()
                mux.wait(timeout=2)
                result["passed"] = False
                result["forced_mux_stop"] = True
        if mux is not None:
            result["mux_exit"] = mux.returncode
            if mux.returncode not in (0, -signal.SIGINT):
                result["passed"] = False
        if bridge is not None:
            node.destroy_timer(bridge.timer)
            bridge.journal.close()
        if io is not None:
            io.close()
        node.destroy_node()
        rclpy.shutdown()
        for log in logs:
            log.close()
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(
        json.dumps(
            {key: result[key] for key in ("passed", "case", "typed_messages", "sit_calls", "phase")}
        )
    )
    if not result["passed"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
