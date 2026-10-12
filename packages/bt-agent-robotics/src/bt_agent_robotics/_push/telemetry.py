"""ROS receiver for controller-only native push telemetry and exact bindings."""

import json
import os
import threading
import time
from copy import deepcopy

from bt_agent_robotics._push.feedback import PushFeedbackBuffer


def episode(value):
    return tuple(value.get(k) for k in ("run_id", "task_id", "task_index"))


class RosPushTelemetry:
    def __init__(self, node):
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from std_msgs.msg import String

        latched = QoSProfile(
            depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL, reliability=ReliabilityPolicy.RELIABLE
        )
        self._lock = threading.RLock()
        self._context = None
        self._bindings = None
        self._selected = None
        self._snapshot = None
        self._error = None
        self._observed = False
        self.buffer = PushFeedbackBuffer()
        self._transform = None
        self._subscriptions = [
            node.create_subscription(String, "/episode_context", self._on_context, latched),
            node.create_subscription(String, "/sim/push_controller_bindings", self._on_bindings, latched),
            node.create_subscription(String, "/sim/push_controller_samples", self._on_sample, 4096),
            node.create_subscription(String, "/sim/interaction_events", self._on_snapshot, latched),
        ]

    @staticmethod
    def parse(msg):
        try:
            value = json.loads(msg.data)
        except (ValueError, TypeError):
            return None
        return value if isinstance(value, dict) else None

    def _on_context(self, msg):
        value = self.parse(msg)
        if not value:
            return
        key = episode(value)
        if not key[0] or not key[1] or type(key[2]) is not int or key[2] < 1:
            return
        with self._lock:
            if key != self._context:
                self.buffer.reset()
                self._selected = None
                self._error = None
                self._snapshot = None
                self._transform = None
            self._context = key

    def _on_bindings(self, msg):
        value = self.parse(msg)
        if (
            not value
            or value.get("schema") != "native-push-bindings/v1"
            or value.get("profile") != "measured_kinematic_rail_prototype_v1"
        ):
            return
        with self._lock:
            self._bindings = value

    def _on_snapshot(self, msg):
        value = self.parse(msg)
        if not value or value.get("schema") != "public-interaction-snapshot-v1":
            return
        with self._lock:
            if episode(value) == self._context:
                self._snapshot = value

    def _on_sample(self, msg):
        value = self.parse(msg)
        if not value:
            return
        now = time.monotonic()
        with self._lock:
            if self._selected is None or episode(value) != self._context:
                return
            if value.get("mechanism_id") != "push:" + self._selected["asset"]["key"]:
                return
            source_time = value.get("native_monotonic")
            if type(source_time) not in (int, float) or not 0 <= now - source_time <= 1.0:
                self._error = "native_sample_stale"
                return
            if self.buffer.observe(value, now):
                self._observed = True

    def begin(self, handle):
        from bt_agent_robotics._push.transform import build_transform

        with self._lock:
            if self._context is None or self._bindings is None or episode(self._bindings) != self._context:
                raise RuntimeError("Native push binding is missing or from a different episode")
            candidates = [b for b in self._bindings.get("bindings", []) if b.get("canonical_handle") == handle]
            if len(candidates) != 1:
                raise RuntimeError("Target has no unique authored native push binding")
            binding = deepcopy(candidates[0])
            key = binding["asset"]["key"]
            native_id = binding["native_instance_id"]
            if (
                binding["frame"] != "ros_map_spawn_relative"
                or not binding["native_handle"]
                or not binding["native_template"]
            ):
                raise RuntimeError("Native push identity/frame is incomplete")
            self.buffer.bind(self._context, key, native_id)
            self._selected = binding
            self._error = None
            self._observed = False
            self._transform = build_transform(binding["spawn_position"], binding["spawn_rotation_y"])
            binding["episode_token"] = json.dumps(self._context, separators=(",", ":"))
            return binding

    def drain(self, now):
        from bt_agent_robotics._push.controller import Observation

        with self._lock:
            if self._error:
                raise RuntimeError(self._error)
            if self._selected is None:
                raise RuntimeError("Push invocation is not bound")
            if not self._observed:
                return []
            result = []
            for event, received in self.buffer.drain(now):
                data = event["evidence"]
                if data.get("frame") != "habitat_y_up" or data.get("robot_pose_correction") is not False:
                    raise RuntimeError("Unqualified native push sample")
                obj = tuple(self._transform.point(data["actual_pose"])[:2])
                robot = tuple(self._transform.point(data["robot_pose"])[:2])
                result.append(
                    Observation(
                        json.dumps(self._context, separators=(",", ":")),
                        data["native_instance_id"],
                        data["canonical_key"],
                        data["sample"],
                        received,
                        obj,
                        robot,
                        data["activation"],
                        (
                            tuple(self._transform.point(data["episode_initial_pose"])[:2])
                            if os.environ.get("LAM_HABITAT_PUSH_EPISODE_RESUME") == "1"
                            else None
                        ),
                        (
                            data.get("episode_robot_travel_m")
                            if os.environ.get("LAM_HABITAT_PUSH_EPISODE_RESUME") == "1"
                            else None
                        ),
                        data.get("episode_motion_observed") is True,
                    )
                )
            return result

    def completed_snapshot_received(self, key):
        gate = {"step_stool": "stool", "dining_chair": "chair"}[key]
        with self._lock:
            return bool(
                self._snapshot
                and any(
                    e == dict(mechanism_id=gate, status="gate_open_completed")
                    for e in self._snapshot.get("completed", [])
                )
            )

    def end(self):
        with self._lock:
            self.buffer.reset()
            self._selected = None
            self._transform = None
