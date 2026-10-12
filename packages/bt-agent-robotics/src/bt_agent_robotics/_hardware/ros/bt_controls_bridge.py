"""Opt-in ROS transport for the staged BT intervention experiments."""

import json
import math
import threading
import time
import uuid
from pathlib import Path

from bt_agent_robotics._hardware.bt_controls import Controller


class BTControlsBridge:
    def __init__(self, node, backend, capture_dir):
        from nav_msgs.msg import Odometry
        from rclpy.qos import DurabilityPolicy, QoSProfile
        from std_msgs.msg import String

        self.node, self.message_type = node, String
        self.mux_brake = "UNKNOWN"
        self.log_errors = 0
        self.log_lock = threading.Lock()
        path = Path(capture_dir)
        path.mkdir(parents=True, exist_ok=True)
        self.journal_path = path / f"bt_controls_{uuid.uuid4().hex}.jsonl"
        self.journal = self.journal_path.open("x", buffering=1)
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = node.create_publisher(String, "/lam/bt_controls/status", latched)
        self.ack_pub = node.create_publisher(String, "/lam/bt_controls/ack", 10)
        self.event_pub = node.create_publisher(String, "/lam/bt_controls/events", 50)
        self.controller = Controller(backend, emit=self.emit)
        node.create_subscription(String, "/lam/mux_brake_state", self.on_brake, latched)
        node.create_subscription(String, "/lam/bt_controls/request", self.on_request, 10)
        node.create_subscription(Odometry, "/spot/platform/odom", self.on_odom, 10)
        self.timer = node.create_timer(0.1, self.publish_status)
        self.emit({
            "kind": "bt_controls_ready",
            "wall_ns": time.time_ns(),
            "monotonic_s": time.monotonic(),
            "experimental": True,
            "journal": str(self.journal_path),
            "agent_continues": True,
            "timing_schema": "receipt-v2",
        })

    def publish(self, publisher, data):
        message = self.message_type()
        message.data = json.dumps(data)
        publisher.publish(message)

    def emit(self, event):
        try:
            with self.log_lock:
                self.journal.write(json.dumps(event) + "\n")
            self.publish(self.event_pub, event)
        except Exception as error:
            self.log_errors += 1
            self.node.get_logger().error(f"BT intervention recording failed: {error}")

    def on_brake(self, message):
        self.mux_brake = message.data.strip()

    def on_odom(self, message):
        velocity = message.twist.twist
        self.controller.observe_motion(math.hypot(velocity.linear.x, velocity.linear.y), velocity.angular.z)

    def publish_status(self):
        self.publish(
            self.status_pub,
            {
                **self.controller.snapshot(),
                "journal": str(self.journal_path),
                "recording_errors": self.log_errors,
                "timing_schema": "receipt-v2",
                "bridge_node": self.node.get_fully_qualified_name(),
            },
        )

    def on_request(self, message):
        received_wall_ns = time.time_ns()
        received_monotonic_s = time.monotonic()
        request = {}
        try:
            if len(message.data) > 2048:
                raise ValueError("Oversized request")
            request = json.loads(message.data)
            if not isinstance(request, dict):
                request = {}
                raise ValueError("Expected an object")
            if request.get("kind") == "probe":
                if request.get("bridge_journal") != str(self.journal_path):
                    raise ValueError("Probe targets another bridge instance")
                self.publish(
                    self.ack_pub,
                    {
                        "request_id": request.get("request_id"),
                        "ok": True,
                        "probe": True,
                        "journal": str(self.journal_path),
                    },
                )
                return  # No controller request, cancellation, brake, or task change.
            # Timestamp callback entry, not controller acceptance after its lock.
            # Use the persistent journal/event publisher: a short-lived CLI
            # publisher can disappear before rosbag discovers the request topic.
            self.emit({
                "kind": "intervention_received",
                "wall_ns": received_wall_ns,
                "monotonic_s": received_monotonic_s,
                "request_id": request.get("request_id"),
                "key": request.get("key"),
                "intervention": request.get("kind"),
                "cli_publish_wall_ns": request.get("cli_publish_wall_ns"),
                "cli_publish_monotonic_s": request.get("cli_publish_monotonic_s"),
            })
            accepted = self.controller.request(request["key"], request["kind"], request["request_id"])
            self.publish(
                self.ack_pub,
                {"request_id": request["request_id"], "ok": True, "new": accepted},
            )
        except Exception as error:
            self.publish(
                self.ack_pub,
                {
                    "request_id": request.get("request_id"),
                    "ok": False,
                    "error": str(error),
                },
            )
        self.publish_status()
