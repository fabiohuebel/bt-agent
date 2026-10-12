"""Opt-in battery demo integration. No automatic release, stand, power or claim."""

import json
import math
import os
import threading
import time
import uuid
from pathlib import Path

from bt_agent_robotics._hardware.battery_guard import BatteryGuard, BatteryGuardPolicy
from bt_agent_robotics._hardware.battery_posture import BatteryPostureHandoff, PostureTelemetry


def battery_percent(message):
    states = getattr(message, "battery_states", ())
    # Native Spot BatteryState: unknown/missing/booting must not be interpreted
    # as a genuine zero-percent reading that could trigger a posture command.
    if any(getattr(state, "status", 0) not in (2, 3) for state in states):
        return None
    values = [getattr(state, "charge_percentage", None) for state in states]
    if not values or not all(BatteryGuardPolicy._valid_percent(value) for value in values):
        return None
    return min(values)


def seated_geometry(message):
    heights = [
        abs(float(foot.foot_position_rt_body.z))
        for foot in getattr(message, "states", ())
        if int(getattr(foot, "contact", 0)) == 1
    ]
    if len(heights) < 3 or not all(math.isfinite(height) for height in heights):
        return "unknown"
    mean = sum(heights) / len(heights)
    return "sitting" if mean < 0.20 else "standing" if mean > 0.35 else "transitioning"


def mux_process_live(value):
    try:
        pid = int(value["owner_pid"])
        if pid <= 0:
            return False
        fields = Path(f"/proc/{pid}/stat").read_text().rsplit(")", 1)[1].split()
        return fields[0] != "Z" and fields[19] == str(value["owner_start"])
    except (OSError, KeyError, ValueError, IndexError):
        return False


class BatteryFencedPlanner:
    def __init__(self, bridge, backend):
        self.bridge, self.backend = bridge, backend

    def __getattr__(self, name):
        return getattr(self.backend, name)

    def _call(self, name, *args):
        safe = (
            name == "cancel"
            or name == "set_brake"
            and args == (True,)
            or name == "send_velocity"
            and all(value == 0 for value in args)
            or name == "set_press_precision"
            and args == (False,)
        )
        with self.bridge.lock:
            state = self.bridge.policy.snapshot()
            allowed = state["can_continue"] if state["motion_active"] else state["can_start"]
            if not safe and (self.bridge.persisted_hold()["active"] or not allowed):
                raise RuntimeError("[leaf=BatteryReserveGuard] Motion blocked by battery reserve condition.")
            return getattr(self.backend, name)(*args)

    def send_waypoint(self, *args):
        return self._call("send_waypoint", *args)

    def send_velocity(self, *args):
        return self._call("send_velocity", *args)

    def set_brake(self, engaged):
        return self._call("set_brake", engaged)

    def set_press_precision(self, enabled):
        if hasattr(self.backend, "set_press_precision"):
            return self._call("set_press_precision", enabled)
        return None

    def cancel(self):
        return self._call("cancel")


class BatteryGuardBridge:
    def __init__(self, node, backend, capture_dir):
        from nav_msgs.msg import Odometry
        from rclpy.qos import DurabilityPolicy, QoSProfile, qos_profile_sensor_data
        from spot_msgs.msg import BatteryStateArray, FootStateArray, LeaseArray
        from std_msgs.msg import String
        from std_srvs.srv import Trigger

        self.node, self.backend, self.message_type = node, backend, String
        self.lock = threading.RLock()
        self.log_lock = threading.Lock()
        self.instance = uuid.uuid4().hex
        self.hold_path = Path(os.environ["LAM_BATTERY_HOLD_FILE"])
        if not self.hold_path.is_absolute():
            raise ValueError("LAM_BATTERY_HOLD_FILE must be absolute")
        self.boot_id = Path("/proc/sys/kernel/random/boot_id").read_text().strip()
        Path(capture_dir).mkdir(parents=True, exist_ok=True)
        self.journal_path = Path(capture_dir) / f"battery_guard_{self.instance}.jsonl"
        self.journal = self.journal_path.open("x", buffering=1)
        self.log_errors = 0
        self.odom = self.brake = self.lease = self.feet = self.mux = None
        self.mux_identity = None
        self.actuator_topic = os.environ.get("LAM_BATTERY_ACTUATOR_TOPIC", "/cmd_vel")
        self.seen = {}
        self.posture_event_cursor = 0
        latched = QoSProfile(depth=1, durability=DurabilityPolicy.TRANSIENT_LOCAL)
        self.status_pub = node.create_publisher(String, "/lam/battery_guard/status", latched)
        self.event_pub = node.create_publisher(String, "/lam/battery_guard/events", 100)
        self.ack_pub = node.create_publisher(String, "/lam/battery_guard/ack", 10)
        self.policy = BatteryGuardPolicy(emit=self.emit)
        self.planner = BatteryFencedPlanner(self, backend)
        # Lease telemetry is ~1 Hz (measured max gap 1.104 s); keep motion and
        # mux freshness tight while allowing two seconds for the lease sample.
        self.handoff = BatteryPostureHandoff(self, telemetry_age=0.75, owner_age=2.0)
        self.restored_hold = self.persisted_hold()["active"]
        self.sit_client = node.create_client(Trigger, "/sit")
        self.trigger = Trigger
        self.subscriptions = [
            node.create_subscription(
                BatteryStateArray,
                "/status/battery_states",
                self.on_battery,
                qos_profile_sensor_data,
            ),
            node.create_subscription(Odometry, "/spot/platform/odom", self.on_odom, qos_profile_sensor_data),
            node.create_subscription(FootStateArray, "/status/feet", self.on_feet, qos_profile_sensor_data),
            node.create_subscription(LeaseArray, "/status/leases", self.on_lease, qos_profile_sensor_data),
            node.create_subscription(String, "/lam/mux_brake_state", self.on_brake, 10),
            node.create_subscription(String, "/lam/battery_hold/mux_status", self.on_mux, 10),
            node.create_subscription(String, "/lam/battery_guard/request", self.on_request, 10),
        ]
        self.timer = node.create_timer(0.05, self.tick)
        self.emit({
            "kind": "battery_guard_ready",
            "journal": str(self.journal_path),
            "restart_hold": self.persisted_hold(),
            "start_percent": 10,
            "critical_percent": 5,
        })

    def emit(self, event):
        event = {"wall_ns": time.time_ns(), "monotonic_s": time.monotonic(), **event}
        try:
            with self.log_lock:
                self.journal.write(json.dumps(event) + "\n")
            self.publish(self.event_pub, event)
        except Exception as error:
            self.log_errors += 1
            self.node.get_logger().error(f"Battery guard journal failed: {error}")

    def publish(self, publisher, value):
        message = self.message_type()
        message.data = json.dumps(value)
        publisher.publish(message)

    def persisted_hold(self):
        from bt_agent_robotics._hardware.battery_hold_state import read_hold

        return read_hold(self.hold_path)

    def _write_hold(self, event_id, phase, reason="battery_critical"):
        from bt_agent_robotics._hardware.battery_hold_state import atomic_write_hold

        current = self.persisted_hold()
        if current.get("active") and current.get("hold_id") not in (None, event_id):
            raise RuntimeError("Existing battery hold requires manual reset")
        if current.get("active") and current.get("phase") == "suspend":
            if current.get("hold_id") != event_id:
                raise RuntimeError("Existing suspended hold requires manual reset")
            phase = "suspend"
        atomic_write_hold(
            self.hold_path,
            {"active": True, "hold_id": event_id, "phase": phase, "reason": reason},
        )

    def on_battery(self, message):
        with self.lock:
            self.policy.update_real(battery_percent(message), time.monotonic())

    def on_odom(self, message):
        stamp = message.header.stamp
        source_ns = int(stamp.sec) * 1_000_000_000 + int(stamp.nanosec)
        if source_ns <= 0 or not 0 <= self.node.get_clock().now().nanoseconds - source_ns <= 750_000_000:
            self.odom = None
            return
        twist = message.twist.twist
        self.odom = (
            math.hypot(twist.linear.x, twist.linear.y),
            twist.angular.z,
            time.monotonic(),
        )

    def on_brake(self, message):
        self.brake = (message.data.strip() == "ENGAGED", time.monotonic())

    def on_feet(self, message):
        self.feet = (seated_geometry(message), time.monotonic())

    def on_lease(self, message):
        markers = tuple(
            x.strip()
            for x in os.environ.get("LAM_LEASE_OWNERS", "ros_spot,spot_wrapper,spot_ros2").split(",")
            if x.strip()
        )
        resources = [
            resource
            for resource in getattr(message, "resources", ())
            if getattr(getattr(resource, "lease", None), "resource", "") == "body"
        ]
        owned = False
        if len(resources) == 1:
            owner = getattr(resources[0], "lease_owner", None)
            names = (getattr(owner, "client_name", ""), getattr(owner, "user_name", ""))
            owned = any(marker in name for marker in markers for name in names)
        self.lease = (owned, time.monotonic())

    def on_mux(self, message):
        try:
            value = json.loads(message.data)
            if not isinstance(value, dict):
                raise ValueError("Mux status must be an object")
            now = time.monotonic()
            if value.get("boot_id") != self.boot_id or not 0 <= now - float(value["monotonic_s"]) <= 0.75:
                raise ValueError("Stale or foreign-host mux status")
            identity = (value.get("owner_pid"), value.get("owner_start"), value.get("boot_id"))
            if (
                not all(identity)
                or value.get("owner_lock_acquired") is not True
                or value.get("output_topic") != self.actuator_topic
                or value.get("output_publisher_count") != 1
                or not mux_process_live(value)
            ):
                raise ValueError("Unqualified mux owner/output")
            if self.mux_identity is None:
                self.mux_identity = identity
            if self.mux_identity != identity:
                raise ValueError("Mux process changed; manual requalification required")
            self.mux = (value, now)
        except (ValueError, TypeError, KeyError, OverflowError):
            self.mux = None

    def telemetry(self):
        now = time.monotonic()
        hold = self.persisted_hold()
        m, received = self.mux if self.mux else ({}, None)
        same = m.get("hold_id") == self.handoff.event_id and hold.get("hold_id") == self.handoff.event_id
        lease_owned, lease_at = self.lease or (False, None)
        source_at = m.get("monotonic_s")
        mux_at = min(received, source_at) if received is not None and source_at is not None else None
        owner_ok = (
            lease_owned
            and same
            and m.get("exclusive_output_owner") is True
            and m.get("valid") is True
            and mux_at is not None
            and math.isfinite(mux_at)
            and 0 <= now - mux_at <= 0.75
            and math.isfinite(source_at)
            and 0 <= now - source_at <= 0.75
            and math.isfinite(received)
            and 0 <= now - received <= 0.75
            and mux_process_live(m)
        )
        owner_at = lease_at  # mux freshness is checked independently above
        linear, angular, odom_at = self.odom or (None, None, None)
        brake, brake_at = self.brake or (False, None)
        posture, posture_at = self.feet or ("unknown", None)
        return PostureTelemetry(
            linear_speed=linear,
            angular_speed=angular,
            odom_at=odom_at,
            brake_engaged=brake,
            brake_at=brake_at,
            exclusive_owner=owner_ok,
            owner_at=owner_at,
            hold_active=hold.get("active") is True and same,
            hold_at=now,
            mux_suspended=same
            and m.get("valid") is True
            and m.get("output_suspended") is True
            and m.get("phase") == "suspend",
            mux_at=mux_at,
            posture=posture,
            posture_at=posture_at,
        )

    def enter_motion_hold(self, event_id):
        self._write_hold(event_id, "stop")

    def request_stop(self, event_id):
        errors = []
        for name, operation in (
            ("brake", lambda: self.backend.set_brake(True)),
            ("zero", lambda: self.backend.send_velocity(0.0, 0.0, 0.0)),
        ):
            try:
                operation()
                self.emit({
                    "kind": "battery_stop_command",
                    "command": name,
                    "hold_id": event_id,
                })
            except Exception as error:
                errors.append(str(error))
        if errors:
            raise RuntimeError("; ".join(errors))

    def request_mux_suspend(self, event_id):
        if not self.handoff._ready(self.telemetry(), time.monotonic()):
            raise RuntimeError("Stationary/ownership evidence lost before mux suspension")
        self._write_hold(event_id, "suspend")

    def request_sit(self, event_id):
        telemetry = self.telemetry()
        now = time.monotonic()
        state = self.policy.snapshot()
        if (
            event_id != self.handoff.event_id
            or not state["sit_requested"]
            or state["effective_percent"] is None
            or not self.handoff._ready(telemetry, now)
            or not telemetry.mux_suspended
            or not self.handoff._fresh(telemetry.mux_at, now)
        ):
            raise RuntimeError("Sit refused: missing fresh battery/hold/ownership/stationary/mux proof")
        if not self.sit_client.service_is_ready():
            raise RuntimeError("Sit service unavailable; no retry")
        future = self.sit_client.call_async(self.trigger.Request())
        future.add_done_callback(self.on_sit_response)

    def on_sit_response(self, future):
        with self.lock:
            try:
                accepted = future.result().success is True
            except Exception:
                accepted = False
            self.handoff.sit_acknowledged(accepted=accepted, now=time.monotonic())

    def stop_for_guard(self, state):
        with self.lock:
            event_id = f"{self.instance}:{state['hold_id']}" if state["held"] else "start_precondition"
            try:
                if state["held"]:
                    self._write_hold(event_id, "stop", state["reason"])
            finally:
                self.request_stop(event_id)  # Still brake if durable storage fails.

    def instrument_skill(self, skill):
        import py_trees

        original = skill.build_subtree

        def build(params, context, handle):
            body = original(params, context, handle)

            def stop(state):
                self.emit({
                    "kind": "battery_condition_failed_in_bt",
                    "leaf": "BatteryReserveGuard",
                    "skill": skill.name,
                    "call_id": context.skill_call_id,
                    "state": state,
                })
                handle.result_data = {
                    **handle.result_data,
                    "failure_reason": state["reason"],
                    "failed_leaf": "BatteryReserveGuard",
                }
                action = "Navigation" if skill.name == "navigate" else "The skill"
                if not state.get("motion_active"):
                    reason = (
                        "battery is below the required threshold"
                        if state["reason"] == "battery_below_start_threshold"
                        else state["reason"]
                    )
                    handle.observation = f"[leaf=BatteryReserveGuard] {action} was not started: {reason}."
                else:
                    handle.observation = (
                        f"[leaf=BatteryReserveGuard] {action} stopped before completion: {state['reason']}."
                    )
                if state.get("sit_requested"):
                    handle.observation += (
                        " A stop-and-sit request was issued due to critical battery."
                        " Further motion is blocked pending operator reset."
                    )
                try:
                    self.stop_for_guard(state)
                finally:
                    # BT-controls allocates its scope during build, before its
                    # first tick. Release that scope even on a start rejection.
                    body.stop(py_trees.common.Status.INVALID)

            return BatteryGuard(
                body,
                self.policy,
                stop,
                name="BatteryReserveGuard",
            )

        skill.build_subtree = build

    def tick(self):
        with self.lock:
            state = self.policy.snapshot()
            # The command fence applies continuously; authored BT emits the
            # ordinary skill failure on its next tick. No agent cancellation.
            if state["held"] and not self.persisted_hold()["active"]:
                try:
                    self.stop_for_guard(state)
                except Exception as error:
                    self.emit({"kind": "battery_fence_error", "error": str(error)})
            if state["sit_requested"] and self.handoff.state == "idle" and not self.restored_hold:
                self.handoff.start(f"{self.instance}:{state['hold_id']}", time.monotonic())
            telemetry = self.telemetry()
            self.handoff.tick(time.monotonic(), telemetry)
            for event in self.handoff.events[self.posture_event_cursor :]:
                self.emit(event)
            self.posture_event_cursor = len(self.handoff.events)
            self.publish(
                self.status_pub,
                {
                    **state,
                    "journal": str(self.journal_path),
                    "instance": self.instance,
                    "posture_phase": self.handoff.state,
                    "hold": self.persisted_hold(),
                    "log_errors": self.log_errors,
                    "bridge_node": self.node.get_fully_qualified_name(),
                    "readiness": {
                        "body_lease_owned": bool(self.lease and self.lease[0]),
                        "lease_age_s": time.monotonic() - self.lease[1] if self.lease else None,
                        "mux_qualified_owner": telemetry.exclusive_owner,
                        "mux_suspended": telemetry.mux_suspended,
                        "posture_geometry": telemetry.posture,
                        "odom_age_s": time.monotonic() - telemetry.odom_at if telemetry.odom_at else None,
                    },
                },
            )

    def on_request(self, message):
        request = {}
        with self.lock:
            try:
                if len(message.data) > 2048:
                    raise ValueError("Oversized request")
                request = json.loads(message.data)
                if not isinstance(request, dict):
                    raise ValueError("Expected object")
                request_id = request["request_id"]
                if not isinstance(request_id, str) or not 1 <= len(request_id) <= 128:
                    raise ValueError("Invalid request ID")
                if request.get("instance") != self.instance:
                    raise ValueError("Bridge instance changed")
                if request_id in self.seen:
                    if self.seen[request_id][0] != request:
                        raise ValueError("Request ID reused")
                    self.publish(self.ack_pub, self.seen[request_id][1])
                    return
                kind = request["kind"]
                if kind == "probe":
                    pass
                elif kind == "set-simulated":
                    state = self.policy.snapshot()
                    if state["effective_percent"] is None or state["real_percent"] < 50:
                        raise ValueError("Simulation requires fresh real battery >=50%")
                    self.policy.set_override(request["percent"])
                elif kind == "clear-simulation":
                    self.policy.clear_override()  # Never clears the persisted stop or posture hold.
                else:
                    raise ValueError("Unknown command")
                self.emit({
                    "kind": "battery_operator_request",
                    "request": request,
                    "state": self.policy.snapshot(),
                })
                ack = {"request_id": request_id, "ok": True, "instance": self.instance}
                self.seen[request_id] = (request, ack)
                if len(self.seen) > 256:
                    self.seen.pop(next(iter(self.seen)))
            except Exception as error:
                ack = {
                    "request_id": request.get("request_id") if isinstance(request, dict) else None,
                    "ok": False,
                    "error": str(error),
                }
            self.publish(self.ack_pub, ack)
