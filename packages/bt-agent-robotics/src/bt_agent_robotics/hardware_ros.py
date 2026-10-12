"""Explicit one-episode ROS binding for the hardware profile.

Importing starts nothing. The caller owns the ROS context/executor, robot driver,
FARM/model services, map alignment, confirmation UI and motion mux. Publication
acknowledgment is local evidence persistence plus ROS publish, not a remote grade.
"""

import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time

from bt_agent.episode import EpisodeHost
from .hardware import HardwareAgent
from .platform import PlatformState, CAMERAS
from .publication import PhysicalPublication


class _OwnedRosNode:
    """Track only this binding's resources, including partially completed setup."""

    def __init__(self, node):
        self.node, self.publishers, self.subscriptions = node, [], []

    def __getattr__(self, name):
        return getattr(self.node, name)

    def create_publisher(self, *args, **kwargs):
        publisher = self.node.create_publisher(*args, **kwargs)
        self.publishers.append(publisher)
        return publisher

    def create_subscription(self, *args, **kwargs):
        subscription = self.node.create_subscription(*args, **kwargs)
        self.subscriptions.append(subscription)
        return subscription

    def close(self):
        failures = []
        for owned, destroy in (
            (self.subscriptions, self.node.destroy_subscription),
            (self.publishers, self.node.destroy_publisher),
        ):
            for item in list(owned):
                try:
                    destroy(item)
                except Exception as exc:
                    failures.append(exc)
                else:
                    owned.remove(item)
        if failures:
            raise RuntimeError(
                f"Failed to destroy {len(failures)} owned ROS resources; close may retry"
            ) from failures[0]


class RosHardwareIO:
    def __init__(
        self,
        node,
        *,
        epoch,
        evidence,
        pose_frame,
        body_frame,
        camera_frames,
        source_age_s=0.75,
        frame_age_s=2.0,
        input_profile=None,
    ):
        # The preserved waypoint sender labels every goal as "map". Accepting
        # odometry in another frame would silently reinterpret its coordinates.
        # Reject before importing ROS or creating publishers/evidence files.
        if pose_frame != "map":
            raise ValueError(
                "Hardware waypoints use map; supply map-frame odometry and aligned graph coordinates"
            )
        if not isinstance(body_frame, str) or not body_frame.strip() or body_frame == pose_frame:
            raise ValueError("Supply a distinct explicit body frame")
        from .hardware_inputs import HardwareInputNormalizer, HardwareInputProfile

        if input_profile is not None and (
            not isinstance(input_profile, HardwareInputProfile)
            or input_profile.body_frame != body_frame
            or dict(input_profile.camera_frames) != dict(camera_frames)
        ):
            raise ValueError("Input profile must match the configured body and camera frames")
        self._input = HardwareInputNormalizer(input_profile) if input_profile else None
        from nav_msgs.msg import Odometry
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String
        from tf2_msgs.msg import TFMessage
        from rclpy.qos import QoSProfile, DurabilityPolicy, ReliabilityPolicy
        from ._hardware.ros.ros_local_planner import RosLocalPlanner

        if set(camera_frames) != set(CAMERAS) or len(set(camera_frames.values())) != 5:
            raise ValueError("Explicit five-camera frame mapping required")
        if (
            not pose_frame
            or not body_frame
            or any(not isinstance(v, str) or not v for v in camera_frames.values())
        ):
            raise ValueError("Explicit coordinate frames required")
        if any(not math.isfinite(v) or v <= 0 for v in (source_age_s, frame_age_s)):
            raise ValueError("Positive finite freshness limits required")
        self.node, self.epoch = node, epoch
        self.pose_frame, self.body_frame = pose_frame, body_frame
        self.camera_frames = dict(camera_frames)
        self.source_age_s, self.frame_age_s = source_age_s, frame_age_s
        self.platform = PlatformState(epoch)
        self.out = Path(evidence)
        self.out.mkdir(parents=True, exist_ok=False)
        self.journal = (self.out / "publication.jsonl").open("x")
        self._lock = threading.RLock()
        self._state_lock = threading.RLock()
        self._errors = {}
        self._source_stamps = {}
        self._closed = False
        self._owned = _OwnedRosNode(node)
        self._String = String
        io = self

        class FreshPosePlanner(RosLocalPlanner):
            def get_pose_age(self):
                age = io.platform.pose_age()
                return (
                    math.inf
                    if io._closed
                    or age is None
                    or not math.isfinite(age)
                    or age < 0
                    or "odom" in io._errors
                    else age
                )

            def get_robot_pose(self):
                if self.get_pose_age() > io.source_age_s:
                    raise RuntimeError("No valid observed hardware pose")
                return super().get_robot_pose()

            def cancel(self):
                failures = []

                def attempt(function, *args):
                    try:
                        function(*args)
                    except Exception as exc:
                        failures.append(exc)

                try:
                    pose = self.get_robot_pose()
                except RuntimeError:
                    pose = None
                if pose is not None:
                    attempt(self.send_waypoint, *pose)
                if pose is None or failures:
                    attempt(self.set_brake, True)
                # Never let pose/parking failure suppress the independent zero command.
                attempt(self.send_velocity, 0.0, 0.0, 0.0)
                if failures:
                    raise RuntimeError("Hardware cancellation output failed") from failures[0]

        try:
            node = self._owned
            self.navigation = FreshPosePlanner(
                node,
                self.platform.lock,
                self.platform.pose_ref,
                self.platform.velocity_ref,
                self.platform.pose_stamp_ref,
            )
            latched = QoSProfile(
                depth=10,
                durability=DurabilityPolicy.TRANSIENT_LOCAL,
                reliability=ReliabilityPolicy.RELIABLE,
            )
            self._submission = node.create_publisher(String, "/task_submission", latched)
            self._control = node.create_publisher(String, "/sim_control", latched)
            sensors = QoSProfile(depth=1, reliability=ReliabilityPolicy.BEST_EFFORT)
            self._subscriptions = [
                node.create_subscription(
                    Odometry,
                    input_profile.odometry_topic if input_profile else "/spot/platform/odom",
                    self._odom,
                    sensors,
                )
            ]
            for camera in CAMERAS:
                self._subscriptions.append(
                    node.create_subscription(
                        CompressedImage,
                        (
                            input_profile.camera_topics[camera]
                            if input_profile
                            else f"/spot/camera/{camera}/image/compressed"
                        ),
                        lambda msg, name=camera: self._camera(name, msg),
                        sensors,
                    )
                )
            self._subscriptions.append(
                node.create_subscription(TFMessage, "/tf_static", self._transforms, latched)
            )
            self.publication = PhysicalPublication(
                lambda wire: self._publish("submission", self._submission, wire),
                lambda wire: self._publish("control", self._control, wire),
                capture_evidence=self._capture_evidence,
            )
        except BaseException:
            self.close()
            raise

    def _fresh_source(self, key, header):
        stamp = int(header.stamp.sec) * 1_000_000_000 + int(header.stamp.nanosec)
        age = (self.node.get_clock().now().nanoseconds - stamp) / 1e9
        if stamp <= 0 or not 0 <= age <= self.source_age_s:
            raise ValueError("Stale or future source timestamp")
        if stamp <= self._source_stamps.get(key, 0):
            raise ValueError("Repeated or out-of-order source timestamp")
        return stamp

    def _odom(self, message):
        with self._state_lock:
            self._odom_locked(message)

    def _odom_locked(self, message):
        if self._closed:
            return
        try:
            normalizer = getattr(self, "_input", None)
            if normalizer is None and message.header.frame_id != self.pose_frame:
                raise ValueError("Odometry frame differs from the configured planning frame")
            stamp = self._fresh_source("odom", message.header)
            p, q, v = message.pose.pose.position, message.pose.pose.orientation, message.twist.twist
            position, quaternion = (
                normalizer.odometry(message)
                if normalizer
                else ([p.x, p.y, 0.0], [q.x, q.y, q.z, q.w])
            )
            if not self.platform.odometry(
                position[:2],
                quaternion,
                (v.linear.x, v.linear.y),
                v.angular.z,
                epoch=self.epoch,
            ):
                raise ValueError("Odometry sample not accepted")
            self._source_stamps["odom"] = stamp
            self._errors.pop("odom", None)
        except Exception as exc:
            self._errors["odom"] = str(exc)

    def _camera(self, name, message):
        with self._state_lock:
            self._camera_locked(name, message)

    def _camera_locked(self, name, message):
        if self._closed:
            return
        try:
            if message.header.frame_id != self.camera_frames[name]:
                raise ValueError("Camera optical frame differs from configured frame")
            stamp = self._fresh_source(name, message.header)
            if not self.platform.camera(name, bytes(message.data), epoch=self.epoch):
                raise ValueError("Camera sample not accepted")
            self._source_stamps[name] = stamp
            self._errors.pop(name, None)
        except Exception as exc:
            self._errors[name] = str(exc)

    def _transforms(self, message):
        with self._state_lock:
            self._transforms_locked(message)

    def _transforms_locked(self, message):
        if self._closed:
            return
        normalizer = getattr(self, "_input", None)
        if normalizer is not None:
            try:
                normalizer.transforms.update(message.transforms)
                normalizer.transforms.lookup("map", normalizer.profile.source_pose_frame)
            except (ValueError, TypeError, AttributeError) as exc:
                self._errors["odom"] = str(exc)
            for camera in CAMERAS:
                try:
                    self.platform.camera_transform(
                        camera, normalizer.camera_transform(camera), epoch=self.epoch
                    )
                    self._errors.pop("tf:" + camera, None)
                except (ValueError, TypeError, AttributeError) as exc:
                    self._errors["tf:" + camera] = str(exc)
            return
        from scipy.spatial.transform import Rotation
        import numpy as np

        by_frame = {value: key for key, value in self.camera_frames.items()}
        for transform in message.transforms:
            if (
                transform.header.frame_id != self.body_frame
                or transform.child_frame_id not in by_frame
            ):
                continue
            camera = by_frame[transform.child_frame_id]
            try:
                t, q = transform.transform.translation, transform.transform.rotation
                if abs(sum(v * v for v in (q.x, q.y, q.z, q.w)) - 1) > 1e-3:
                    raise ValueError("Camera rotation must be a unit quaternion")
                matrix = np.eye(4)
                matrix[:3, :3] = Rotation.from_quat([q.x, q.y, q.z, q.w]).as_matrix()
                matrix[:3, 3] = [t.x, t.y, t.z]
                # Retain the deployed inverse(body-to-camera) convention.
                self.platform.camera_transform(camera, np.linalg.inv(matrix), epoch=self.epoch)
                self._errors.pop("tf:" + camera, None)
            except Exception as exc:
                self._errors["tf:" + camera] = str(exc)

    def readiness(self):
        with self._state_lock, self.platform.lock:
            return self._readiness_locked()

    def _readiness_locked(self):
        state, frames = self.platform.snapshot(), self.platform.frame_records()
        pose_age = self.platform.pose_age()
        now = time.monotonic()
        return dict(
            ready=not self._errors
            and pose_age is not None
            and pose_age <= self.source_age_s
            and set(frames) == set(CAMERAS)
            and all(0 <= now - row["received_at"] <= self.frame_age_s for row in frames.values())
            and set(state["camera_to_body_transforms"]) == set(CAMERAS),
            errors=dict(self._errors),
            pose_age_s=pose_age,
            cameras=sorted(frames),
            pose_frame=self.pose_frame,
            body_frame=self.body_frame,
            camera_frames=self.camera_frames,
            input_profile=(
                self._input.profile.describe() if getattr(self, "_input", None) else None
            ),
        )

    def agent_state(self):
        with self._state_lock, self.platform.lock:
            return self._agent_state_locked()

    def _agent_state_locked(self):
        state, frames = self.platform.snapshot(), self.platform.frame_records()
        now = time.monotonic()
        allowed = {
            name
            for name, row in frames.items()
            if name not in self._errors
            and "tf:" + name not in self._errors
            and name in state["camera_to_body_transforms"]
            and 0 <= now - row["received_at"] <= self.frame_age_s
        }
        state["camera_images"] = {
            name: value for name, value in state["camera_images"].items() if name in allowed
        }
        state["camera_to_body_transforms"] = {
            name: value
            for name, value in state["camera_to_body_transforms"].items()
            if name in allowed and "tf:" + name not in self._errors
        }
        age = self.platform.pose_age()
        if "odom" in self._errors or age is None or not 0 <= age <= self.source_age_s:
            state["robot_position"] = None
        return state

    def _record(self, event):
        self.journal.write(
            json.dumps(dict(epoch=self.epoch, monotonic_s=time.monotonic(), **event)) + "\n"
        )
        self.journal.flush()
        os.fsync(self.journal.fileno())

    def _publish(self, kind, publisher, wire):
        with self._lock:
            if self._closed:
                raise RuntimeError("Hardware ROS output is closed")
            self._record(dict(kind=kind, phase="intent", wire=wire))
            message = self._String()
            message.data = wire
            publisher.publish(message)
            self._record(dict(kind=kind, phase="accepted_locally", wire=wire))

    def _capture_evidence(self, record, image):
        with self._lock:
            if self._closed:
                raise RuntimeError("Hardware ROS output is closed")
            if not isinstance(image, bytes) or not image:
                raise ValueError("Exact capture model image required for hardware evidence")
            digest = hashlib.sha256(image).hexdigest()
            target = self.out / (digest + ".jpg")
            if not target.exists():
                with target.open("xb") as stream:
                    stream.write(image)
                    stream.flush()
                    os.fsync(stream.fileno())
                directory_fd = os.open(self.out, os.O_RDONLY | os.O_DIRECTORY)
                try:
                    os.fsync(directory_fd)
                finally:
                    os.close(directory_fd)
            if hashlib.sha256(target.read_bytes()).hexdigest() != digest:
                raise ValueError("Existing capture evidence is incomplete or corrupted")
            self._record(dict(kind="capture_image", image_file=target.name, record=record))

    def close(self):
        with self._lock, self._state_lock:
            self._close_locked()

    def _close_locked(self):
        self._closed = True
        failures = []
        for close in (self._owned.close, self.platform.close, self.journal.close):
            try:
                close()
            except Exception as exc:
                failures.append(exc)
        if failures:
            raise RuntimeError("Hardware I/O cleanup incomplete; close may retry") from failures[0]


class HardwareROSEpisode:
    """One task; close/recreate for another. External persistent holds survive."""

    def __init__(
        self,
        io,
        *,
        model,
        path_planning,
        scene_memory,
        perception,
        confirmation_gate,
        profile,
        battery=None,
        interventions=None,
    ):
        self.io, self.started = io, False
        self.host = EpisodeHost(io.publication.report, self._flush)
        navigation = battery.planner if battery is not None else io.navigation
        self.agent = HardwareAgent(
            model=model,
            host=self.host,
            navigation=navigation,
            path_planning=path_planning,
            scene_memory=scene_memory,
            perception=perception,
            confirmation_gate=confirmation_gate,
            profile=profile,
            battery=battery,
            interventions=interventions,
        )

    def _flush(self):
        self.io.publication.flush(self.agent.runtime.objects_snapshot())

    def start(self, task, *, scenario=""):
        if self.started:
            raise RuntimeError("Use a fresh hardware ROS episode for each task")
        if not self.io.readiness()["ready"]:
            raise RuntimeError("Hardware sensor/frame readiness is incomplete")
        self.agent.start(task, scenario=scenario, platform_state=self.io.agent_state())
        self.started = True

    def tick(self):
        if not self.started:
            raise RuntimeError("Start the hardware episode first")
        if self.agent.done:
            if self.host.submitted:
                self.io.publication.task_complete(submitted=True)
            return [], []
        self.agent.push_state(self.io.agent_state())
        events, outputs = self.agent.tick()
        if any(event.kind == "skill_completed" for event in events):
            self._flush()
        if self.agent.done and self.host.submitted:
            self.io.publication.task_complete(submitted=True)
        return events, outputs

    def close(self):
        self.agent.close()
        self.io.close()
