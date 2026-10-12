"""Two synthetic inspection tasks through the real controller and hardware ROS IO.

Network-isolated transport/lifecycle fixture, NOT perception or task performance.
No robot, mux, local controller, simulator or model server is started.
"""

import argparse
import json
import os
from pathlib import Path
import time


class Perception:
    def query_structured(self, messages, schema, **kwargs):
        if "chosen_view" in schema["properties"]:
            return dict(chosen_view="B", reasoning="Synthetic transport fixture.")
        if kwargs.get("phase") == "surround_survey":
            return dict(
                visible="frontleft camera" in str(messages),
                reasoning="Fixture view.",
                evaluation="Fixture shelf.",
            )
        if "acceptable" in schema["properties"]:
            return dict(acceptable=True, bbox_1000=[100, 100, 800, 800], reasoning="Fixture box.")
        return dict(visible=True, bbox_1000=[100, 100, 800, 800], reasoning="Fixture detection.")


class Gate:
    enabled = True

    def publish_plan(self, *args):
        raise AssertionError("Fixture target is at the observed pose; no navigation authorized")

    def poll(self):
        return None

    def clear(self):
        pass


class Planner:
    waypoint_threshold = 0.2

    def plan(self, *_):
        return [(0.0, 0.0)]

    replan = plan

    def update_map(self, _):
        pass


class Memory:
    def query(self, *args, **kwargs):
        from bt_agent_robotics._hardware.skills.query import SceneGraphResult

        return SceneGraphResult(
            found=[
                dict(
                    object_id="1",
                    object_label="shelf",
                    viewpoint_position=dict(x=0.0, y=0.0, z=0.0),
                    viewpoint_orientation=[0.0, 0.0, 0.0, 1.0],
                    all_viewpoints=[[0.0, 0.0, 0.0, 0.0]],
                )
            ]
        )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument(
        "--driver-inputs", action="store_true", help="Exercise driver topics and chained static TF"
    )
    args = parser.parse_args()
    if (
        not Path("/.dockerenv").exists()
        or set(os.listdir("/sys/class/net")) != {"lo"}
        or os.environ.get("ROS_DOMAIN_ID") != "97"
        or os.environ.get("ROS_LOCALHOST_ONLY") != "1"
    ):
        raise RuntimeError("Requires Docker --network none, domain97, localhost only")
    args.out.mkdir(parents=True, exist_ok=False)
    os.environ.update(
        LAM_NAV_CONFIRM="1",
        LAM_CAPTURE_LOCAL_INSPECTION="1",
        LAM_FARM_EXPLORATION_FALLBACK="0",
        LAM_QUERY_TOPK="0",
    )
    import rclpy
    from geometry_msgs.msg import TransformStamped, Twist
    from nav_msgs.msg import Odometry
    from sensor_msgs.msg import CompressedImage
    from std_msgs.msg import String
    from tf2_msgs.msg import TFMessage
    from rclpy.qos import QoSProfile, DurabilityPolicy
    from bt_agent.testing import ScriptedModel
    from bt_agent_robotics.hardware import HardwareProfile
    from bt_agent_robotics.hardware_ros import RosHardwareIO, HardwareROSEpisode
    from bt_agent_robotics.hardware_inputs import HardwareInputProfile
    import cv2
    import numpy as np

    rclpy.init()
    node = rclpy.create_node("hardware_episode_fixture")
    submissions, terminals, velocities = [], [], []
    node.create_subscription(String, "/task_submission", lambda m: submissions.append(m.data), 50)
    node.create_subscription(String, "/sim_control", lambda m: terminals.append(m.data), 50)
    node.create_subscription(
        Twist,
        "/spot/cmd_vel_primitive",
        lambda m: velocities.append([m.linear.x, m.linear.y, m.angular.z]),
        50,
    )
    input_profile = HardwareInputProfile.spot_driver() if args.driver_inputs else None
    frames = (
        dict(input_profile.camera_frames)
        if input_profile
        else {
            name: name + "_optical" for name in ("frontleft", "frontright", "left", "right", "back")
        }
    )
    cameras = {
        name: node.create_publisher(
            CompressedImage,
            (
                input_profile.camera_topics[name]
                if input_profile
                else f"/spot/camera/{name}/image/compressed"
            ),
            10,
        )
        for name in frames
    }
    odometry = node.create_publisher(Odometry, "/spot/platform/odom", 10)
    transforms = node.create_publisher(
        TFMessage, "/tf_static", QoSProfile(depth=10, durability=DurabilityPolicy.TRANSIENT_LOCAL)
    )
    tf = TFMessage()
    if input_profile:
        world = TransformStamped()
        world.header.frame_id, world.child_frame_id = "map", "vision"
        world.transform.translation.x, world.transform.rotation.w = -1.0, 1.0
        head = TransformStamped()
        head.header.frame_id, head.child_frame_id = "body", "head"
        head.transform.rotation.w = 1.0
        tf.transforms.extend([world, head])
    for name, frame in frames.items():
        t = TransformStamped()
        t.header.frame_id, t.child_frame_id = ("head", name) if input_profile else ("body", frame)
        t.transform.rotation.w = 1.0
        tf.transforms.append(t)
        if input_profile:
            optical = TransformStamped()
            optical.header.frame_id, optical.child_frame_id = name, frame
            optical.transform.rotation.w = 1.0
            tf.transforms.append(optical)
    _, encoded = cv2.imencode(".jpg", np.full((24, 32, 3), 100, dtype=np.uint8))
    jpeg = encoded.tobytes()

    def feed():
        stamp = node.get_clock().now().to_msg()
        odom = Odometry()
        odom.header.stamp, odom.header.frame_id = stamp, "vision" if input_profile else "map"
        odom.child_frame_id = "body"
        odom.pose.pose.position.x = 1.0 if input_profile else 0.0
        odom.pose.pose.orientation.w = 1.0
        odometry.publish(odom)
        transforms.publish(tf)
        for name, pub in cameras.items():
            image = CompressedImage()
            image.header.stamp, image.header.frame_id = stamp, frames[name]
            image.data, image.format = jpeg, "jpeg"
            pub.publish(image)

    node.create_timer(0.04, feed)
    result, episode = dict(passed=False, fixture=True, episodes=[]), None
    try:
        for index in (1, 2):
            io = RosHardwareIO(
                node,
                epoch=f"fixture-{index}",
                evidence=args.out / f"episode-{index}",
                pose_frame="map",
                body_frame="body",
                camera_frames=frames,
                input_profile=input_profile,
            )
            model = ScriptedModel(
                [
                    ("query", {"target_description": "shelf"}, "Find fixture shelf."),
                    ("navigate", {"target_object": "shelf_1"}, "Check current pose."),
                    ("surround_survey", {"target_description": "shelf"}, "Inspect fixture image."),
                    (
                        "capture",
                        {"target_object": "shelf_1", "camera_view": "frontleft"},
                        "Record fixture image.",
                    ),
                    ("report", {"text": f"Fixture report {index}."}, "Report."),
                    ("claim_complete", {}, "Submit."),
                ]
            )
            model.temperature, model._enable_thinking = 0, False
            episode = HardwareROSEpisode(
                io,
                model=model,
                path_planning=Planner(),
                scene_memory=Memory(),
                perception=Perception(),
                confirmation_gate=Gate(),
                profile=HardwareProfile("visual_question_v1", True, False),
            )
            deadline = time.monotonic() + 10
            while not io.readiness()["ready"] and time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.01)
            episode.start(f"Offline fixture inspection {index}")
            while time.monotonic() < deadline:
                rclpy.spin_once(node, timeout_sec=0.005)
                if not episode.agent.done or not io.publication.terminal_delivered:
                    episode.tick()
                if len(terminals) == index and len(submissions) == 2 * index:
                    break
            assert episode.host.submitted and io.publication.terminal_delivered
            assert len(io.publication.captures) == 1 and io.publication.reports == [
                f"Fixture report {index}."
            ]
            assert len(terminals) == index and len(submissions) == 2 * index
            assert all(m == "task complete" for m in terminals)
            assert len(list(io.out.glob("*.jpg"))) == 1
            assert all(request["max_tokens"] == 1024 for request in model.requests)
            result["episodes"].append(
                dict(
                    epoch=io.epoch,
                    capture=io.publication.captures[0],
                    report=io.publication.reports[0],
                    local_terminal=True,
                    readiness=io.readiness(),
                    model_calls=len(model.requests),
                    profile=episode.agent.describe(),
                )
            )
            episode.close()
            episode = None
        assert all(all(value == 0 for value in row) for row in velocities)
        assert (
            node.count_publishers("/spot/cmd_vel") == 0 and node.count_publishers("/cmd_vel") == 0
        )
        result.update(
            passed=True,
            submissions=submissions,
            terminals=terminals,
            physical_motion=False,
            scope="Canned model/vision and synthetic sensors; protocol/reset/persistence only, not task success",
        )
    except BaseException as exc:
        result["error"] = {"type": type(exc).__name__, "message": str(exc)}
        raise
    finally:
        if episode is not None:
            episode.close()
        node.destroy_node()
        rclpy.shutdown()
        (args.out / "result.json").write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps(dict(passed=True, episodes=2, physical_motion=False, fixture=True)))


if __name__ == "__main__":
    main()
