"""Opt-in battery condition for authored motion BTs; not a safety controller.

Percentages are on a 0..100 scale, timestamps use the receiver's monotonic
clock. A synthetic reading can only lower a valid real reading. The caller
must fence motion commands while held and perform any posture handoff; this
module never releases brakes, commands sitting, or cancels the task agent.
"""

from __future__ import annotations

import contextlib
import math
import threading
import time

import py_trees


class BatteryGuardPolicy:
    def __init__(
        self,
        *,
        clock=time.monotonic,
        stale_after_s=3.0,
        start_percent=10.0,
        critical_percent=5.0,
        emit=lambda event: None,
    ):
        if not (0 <= critical_percent < start_percent <= 100):
            raise ValueError("Require 0 <= critical < start <= 100")
        if not math.isfinite(stale_after_s) or stale_after_s <= 0:
            raise ValueError("stale_after_s must be positive and finite")
        self.clock, self.stale_after_s = clock, stale_after_s
        self.start_percent, self.critical_percent = start_percent, critical_percent
        self.emit = emit
        self.lock = threading.RLock()
        self.real_percent = self.received_monotonic = self.override = None
        self.motion_active = False
        self.held = self.critical_latched = False
        self.hold_reason = None
        self.hold_id = 0

    @staticmethod
    def _valid_percent(value):
        return (
            not isinstance(value, bool)
            and isinstance(value, (float, int))
            and math.isfinite(value)
            and 0 <= value <= 100
        )

    def _emit(self, kind, **fields):
        with contextlib.suppress(Exception):
            self.emit({"kind": kind, "monotonic_s": self.clock(), **fields})

    def update_real(self, percent, received_monotonic=None):
        with self.lock:
            stamp = self.clock() if received_monotonic is None else received_monotonic
            # Invalid data invalidates the prior sample instead of refreshing it.
            valid_stamp = (
                isinstance(stamp, (float, int))
                and not isinstance(stamp, bool)
                and math.isfinite(stamp)
                and stamp <= self.clock()
            )
            if valid_stamp and self.received_monotonic is not None and stamp < self.received_monotonic:
                return False  # Never let an out-of-order callback replace newer data.
            self.real_percent = float(percent) if self._valid_percent(percent) else None
            self.received_monotonic = float(stamp) if valid_stamp else None
            self._evaluate()
            return True

    def set_override(self, percent):
        if not self._valid_percent(percent):
            raise ValueError("Simulation percentage must be finite and within 0..100")
        with self.lock:
            self.override = float(percent)
            self._emit("battery_simulation_set", simulated=True, override_percent=self.override)
            self._evaluate()

    def clear_override(self):
        with self.lock:
            self.override = None
            self._emit("battery_simulation_cleared", simulated=False)
            self._evaluate()  # Clearing a simulation never clears a hold.

    def set_motion_active(self, active):
        with self.lock:
            self.motion_active = bool(active)
            self._evaluate()

    def _reading(self):
        if self.real_percent is None or self.received_monotonic is None:
            return None, "battery_unavailable"
        age = self.clock() - self.received_monotonic
        if not 0 <= age <= self.stale_after_s:
            return None, "battery_stale"
        value = self.real_percent
        if self.override is not None:
            value = min(value, self.override)
        return value, None

    def _evaluate(self):
        value, invalid = self._reading()
        critical = value is not None and value <= self.critical_percent
        if critical or (invalid and self.motion_active):
            if not self.held:
                self.hold_id += 1
                self.held = True
                self.hold_reason = "battery_critical" if critical else invalid
                self._emit(
                    "battery_hold_latched",
                    hold_id=self.hold_id,
                    reason=self.hold_reason,
                    simulated=self.override is not None,
                    effective_percent=value,
                )
            if critical and not self.critical_latched:
                self.critical_latched = True
                self.hold_reason = "battery_critical"
                self._emit(
                    "battery_sit_requested",
                    hold_id=self.hold_id,
                    simulated=self.override is not None,
                    effective_percent=value,
                )
        return value, invalid

    def snapshot(self):
        with self.lock:
            value, invalid = self._evaluate()
            reason = self.hold_reason if self.held else invalid
            if reason is None and value < self.start_percent:
                reason = "battery_below_start_threshold"
            return {
                "can_start": not self.held and value is not None and value >= self.start_percent,
                "can_continue": not self.held and value is not None and value > self.critical_percent,
                "held": self.held,
                "critical_latched": self.critical_latched,
                "sit_requested": self.critical_latched,
                "reason": reason,
                "hold_id": self.hold_id,
                "effective_percent": value,
                "real_percent": self.real_percent,
                "simulated": self.override is not None,
                "override_percent": self.override,
                "received_monotonic": self.received_monotonic,
                "motion_active": self.motion_active,
            }

    def can_start_motion(self):
        return self.snapshot()["can_start"]

    def can_continue_motion(self):
        return self.snapshot()["can_continue"]

    def explicit_release(self):
        """Operator-only API. Does not release brakes, stand, or resume motion."""
        with self.lock:
            value, invalid = self._evaluate()
            if invalid or value < self.start_percent or self.motion_active:
                raise RuntimeError("Release requires fresh recovered battery and inactive motion")
            was_held = self.held
            self.held = self.critical_latched = False
            self.hold_reason = None
            if was_held:
                self._emit(
                    "battery_hold_released",
                    hold_id=self.hold_id,
                    simulated=self.override is not None,
                    effective_percent=value,
                )
            return was_held

    @property
    def sit_requested(self):
        return self.snapshot()["sit_requested"]

    @property
    def state(self):
        return self.snapshot()

    @property
    def critical(self):
        return self.snapshot()["critical_latched"]


class BatteryGuard(py_trees.decorators.Decorator):
    """Checks before every child tick; stops the child before yielding FAILURE.

    stop(snapshot) must synchronously latch a command fence and request braking.
    It must not unblock motion if child invalidation/cancellation subsequently
    emits stale commands. Posture control must separately verify stationary
    odometry and mux ownership. A blocking child still delays the next BT tick.
    """

    def __init__(self, child, policy, stop, *, name="BatteryGuard"):
        super().__init__(name=name, child=child)
        self.policy, self.stop_callback = policy, stop
        self._started = False
        self._stop_called = False
        self.failure = None

    def initialise(self):
        self._started = self._stop_called = False
        self.failure = None

    def tick(self):
        if self.status != py_trees.common.Status.RUNNING:
            self.initialise()
        state = self.policy.snapshot()
        allowed = state["can_continue"] if self._started else state["can_start"]
        if not allowed:
            self.failure = state
            self.feedback_message = "Motion unavailable: " + (state["reason"] or "battery condition")
            if not self._stop_called:
                self._stop_called = True
                try:
                    self.stop_callback(state)
                except Exception as error:
                    # The caller's latch must precede I/O. Never tick motion
                    # after a stop transport error; expose error for monitoring.
                    self.policy._emit("battery_stop_error", error=str(error), **state)
            self.stop(py_trees.common.Status.FAILURE)
            yield self
            return
        if not self._started:
            self._started = True
            self.policy.set_motion_active(True)
        yield from self.decorated.tick()
        self.status = self.decorated.status
        if self.status != py_trees.common.Status.RUNNING:
            self.stop(self.status)
        yield self

    def update(self):
        return self.decorated.status

    def terminate(self, new_status):
        self.policy.set_motion_active(False)
