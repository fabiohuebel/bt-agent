"""Staged, ROS-free, one-shot stop-to-Sit handoff. NOT a hardware adapter.

All timestamps are monotonic in one clock domain. Adapters must validate source
timestamps, not stamp cached dashboard state as fresh. A single serialized owner
must call this state machine. No path releases a hold, stands, powers, or claims.
"""

import math
from dataclasses import dataclass
from typing import Protocol


@dataclass(frozen=True)
class PostureTelemetry:
    linear_speed: float | None = None
    angular_speed: float | None = None
    odom_at: float | None = None
    brake_engaged: bool = False
    brake_at: float | None = None
    exclusive_owner: bool = False
    owner_at: float | None = None
    hold_active: bool = False
    hold_at: float | None = None
    mux_suspended: bool = False
    mux_at: float | None = None
    posture: str = "unknown"
    posture_at: float | None = None


class PostureBackend(Protocol):
    """Methods are nonblocking, raise on refusal, and never retry internally.

    enter_motion_hold must synchronously fence ALL new motion for this task,
    including GO/Home/press, while preserving the task. request_stop must revoke
    current execution and issue brake/zero. request_mux_suspend requires an
    exclusive task-preserving ownership handoff; existing mux_off.py is NOT one.
    request_sit must atomically revalidate authority/hold/mux suspension against
    competing commands before sending once. It must not claim or power motors.
    The ownership fence remains retained throughout sitting and after errors.
    """

    def enter_motion_hold(self, event_id: str) -> None: ...
    def request_stop(self, event_id: str) -> None: ...
    def request_mux_suspend(self, event_id: str) -> None: ...
    def request_sit(self, event_id: str) -> None: ...


class BatteryPostureHandoff:
    """Latched until explicitly replaced during separate operator requalification.

    `held_error` means human attention, NOT confirmation that physical stopping
    succeeded. The hardware adapter is responsible for its persistent stop hold.
    """

    def __init__(
        self,
        backend: PostureBackend,
        *,
        telemetry_age=0.5,
        owner_age=None,
        stationary_duration=0.3,
        seated_duration=0.3,
        stage_timeout=10.0,
        sit_timeout=10.0,
    ):
        owner_age = telemetry_age if owner_age is None else owner_age
        for value in (telemetry_age, owner_age, stationary_duration, seated_duration, stage_timeout, sit_timeout):
            if not math.isfinite(value) or value <= 0:
                raise ValueError("positive finite timing parameters required")
        self.backend = backend
        self.telemetry_age = telemetry_age
        self.owner_age = owner_age
        self.stationary_duration = stationary_duration
        self.seated_duration = seated_duration
        self.stage_timeout = stage_timeout
        self.sit_timeout = sit_timeout
        self.state = "idle"
        self.event_id = None
        self.events = []
        self._since = None
        self._stable_since = None
        self._stable_last_at = None
        self._last_now = None
        self._sit_at = None
        self._seated_since = None
        self._seated_last_at = None

    @property
    def motion_held(self):
        """Policy intent, not hardware acknowledgement."""
        return self.state != "idle"

    def _record(self, now, kind, **details):
        self.events.append(dict(at=now, kind=kind, event_id=self.event_id, **details))

    def _fail(self, now, reason):
        self.state = "held_error"
        self._record(now, "handoff_error", reason=reason)

    def _clock(self, now):
        if not math.isfinite(now) or (self._last_now is not None and now < self._last_now):
            raise ValueError("monotonic finite time required")
        self._last_now = now

    def _command(self, now, name):
        self._record(now, name + "_requested")
        try:
            getattr(self.backend, name)(self.event_id)
        except Exception as exc:
            self._fail(now, f"{name}: {type(exc).__name__}: {exc}")
            return False
        return True

    def start(self, event_id: str, now: float):
        self._clock(now)
        if self.state != "idle":
            self._record(now, "repeat_ignored", received_event_id=event_id)
            return False
        if not event_id:
            raise ValueError("event_id required")
        self.event_id = event_id
        self.state = "waiting_stop"
        self._since = now
        # Even a failed hold installation must attempt the fail-safe stop.
        held = self._command(now, "enter_motion_hold")
        stopped = self._command(now, "request_stop")
        if not held or not stopped:
            self.state = "held_error"
        return held and stopped

    def _fresh(self, stamp, now, max_age=None):
        limit = self.telemetry_age if max_age is None else max_age
        return stamp is not None and math.isfinite(stamp) and 0 <= now - stamp <= limit

    def _authority_ready(self, t, now):
        speeds = (t.linear_speed, t.angular_speed)
        return (
            all(v is not None and math.isfinite(v) for v in speeds)
            and self._fresh(t.odom_at, now)
            and t.brake_engaged
            and self._fresh(t.brake_at, now)
            and t.exclusive_owner
            and self._fresh(t.owner_at, now, self.owner_age)
            and t.hold_active
            and self._fresh(t.hold_at, now)
        )

    def _ready(self, t, now):
        return self._authority_ready(t, now) and abs(t.linear_speed) < 0.03 and abs(t.angular_speed) < 0.03

    def tick(self, now: float, t: PostureTelemetry):
        self._clock(now)
        if self.state in ("idle", "held_error", "seated_held"):
            return self.state
        # Sitting is a physical posture transition, not stationary execution.
        # Require fresh authority throughout but allow expected lowering motion.
        ready = self._authority_ready(t, now) if self.state == "waiting_sit" else self._ready(t, now)
        if not ready:
            self._stable_since = None
            if self.state in ("waiting_mux", "waiting_sit"):
                self._fail(now, "Lost stationary/brake/ownership/hold evidence during handoff")
            elif now - self._since >= self.stage_timeout:
                self._fail(now, "Fresh stationary/brake/ownership/hold evidence unavailable")
            return self.state
        if self.state == "waiting_stop":
            if (
                self._stable_since is None
                or self._stable_last_at is None
                or not 0 <= t.odom_at - self._stable_last_at <= self.telemetry_age
            ):
                self._stable_since = max(t.odom_at, self._since)
            self._stable_last_at = t.odom_at
            # Require distinct fresh odometry samples across the stable window.
            if t.odom_at - self._stable_since >= self.stationary_duration:
                self._record(now, "stationary_confirmed")
                self.state, self._since = "waiting_mux", now
                self._command(now, "request_mux_suspend")
            elif now - self._since >= self.stage_timeout:
                self._fail(now, "Stationary window not confirmed")
        elif self.state == "waiting_mux":
            if t.mux_suspended and self._fresh(t.mux_at, now) and t.mux_at >= self._since:
                self._record(now, "mux_suspended_confirmed")
                self.state, self._since, self._sit_at = "waiting_sit", now, now
                self._command(now, "request_sit")
            elif now - self._since >= self.stage_timeout:
                self._fail(now, "Mux suspension not confirmed; no Sit sent")
        elif self.state == "waiting_sit":
            if not t.mux_suspended or not self._fresh(t.mux_at, now):
                self._fail(now, "Mux suspension lost after Sit")
            else:
                seated = (
                    t.posture == "sitting"
                    and self._fresh(t.posture_at, now)
                    and t.posture_at > self._sit_at
                    and self._ready(t, now)
                )
                if seated:
                    sample_at = min(t.posture_at, t.odom_at)
                    if (
                        self._seated_since is None
                        or self._seated_last_at is None
                        or not 0 <= sample_at - self._seated_last_at <= self.telemetry_age
                    ):
                        self._seated_since = max(t.posture_at, t.odom_at)
                    self._seated_last_at = sample_at
                    if sample_at - self._seated_since >= self.seated_duration:
                        self.state = "seated_held"
                        self._record(now, "seated_confirmed")
                else:
                    self._seated_since = None
                    self._seated_last_at = None
                if self.state == "waiting_sit" and now - self._since >= self.sit_timeout:
                    self._fail(now, "Sit posture not confirmed; no retry")
        return self.state

    def sit_acknowledged(self, *, accepted: bool, now: float):
        self._clock(now)
        if self.state != "waiting_sit":
            self._record(now, "sit_ack_ignored", accepted=accepted)
            return
        self._record(now, "sit_acknowledgement", accepted=accepted)
        if not accepted:
            self._fail(now, "Sit denied; no retry")
