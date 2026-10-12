"""Invocation-bound output limit for optional measured push; no path changes.

Applied after ordinary mux arbitration/braking. The skill must acquire and
observe an acknowledgement while braked, then renew every tick. Expiry stops
motion until the same invocation releases while braked. This module never
releases the robot brake or supplies a motion command.
"""

import math


class PushSpeedLimit:
    def __init__(self, timeout_s=0.5):
        if not math.isfinite(timeout_s) or timeout_s <= 0:
            raise ValueError("Positive heartbeat timeout required")
        self.timeout_s = timeout_s
        self.token = None
        self.max_speed = None
        self.received_at = None
        self.fault = None
        self.release_requested = False

    def observe(self, request, now, *, braked):
        if set(request) != {"token", "active", "max_speed"}:
            return False
        token = request["token"]
        active = request["active"]
        speed = request["max_speed"]
        if (
            not isinstance(token, str)
            or not 16 <= len(token) <= 128
            or type(active) is not bool
            or not math.isfinite(now)
        ):
            return False
        if not active:
            if token != self.token:
                return False
            self.release_requested = True
            self.advance(now, braked=braked)
            return True
        if type(speed) not in (int, float) or not math.isfinite(speed) or not 0 < speed <= 0.2:
            return False
        if self.token is None:
            if not braked:
                return False
            self.token = token
            self.max_speed = speed
            self.fault = None
        elif token != self.token or speed != self.max_speed or self.release_requested:
            return False
        if self.received_at is not None and (now < self.received_at or now - self.received_at > self.timeout_s):
            self.fault = "speed_lease_expired"
        if self.fault:
            return False
        self.received_at = now
        return True

    def advance(self, now, *, braked):
        if self.release_requested and braked:
            self.token = None
            self.max_speed = None
            self.received_at = None
            self.fault = None
            self.release_requested = False
        if self.token is not None and (self.received_at is None or not 0 <= now - self.received_at <= self.timeout_s):
            self.fault = "speed_lease_expired"

    def limit(self, velocity, now, *, braked):
        self.advance(now, braked=braked)
        if self.token is None:
            return velocity
        if self.fault or self.release_requested or braked:
            return (0.0, 0.0, 0.0)
        if len(velocity) != 3 or not all(math.isfinite(v) for v in velocity):
            self.fault = "invalid_velocity"
            return (0.0, 0.0, 0.0)
        x, y, yaw_rate = velocity
        norm = math.hypot(x, y)
        scale = min(1.0, self.max_speed / norm) if norm else 1.0
        return (x * scale, y * scale, max(-0.3, min(0.3, yaw_rate)))

    def snapshot(self):
        return dict(
            schema="push-speed-limit/v1",
            token=self.token,
            max_speed=self.max_speed,
            status=self.fault or ("releasing" if self.release_requested else "active" if self.token else "inactive"),
        )
