"""Opt-in persistent battery motion hold shared by mux and operator console.

The coordinator atomically writes LAM_BATTERY_HOLD_FILE. The mux additionally
persists '<file>.mux-latch'. Neither a cleared main file nor a restart releases
that latch. Manual recovery requires stopped, fenced requalification before
explicitly removing the latch and restarting the mux. No reset API is provided.
"""

from __future__ import annotations

import contextlib
import fcntl
import json
import os
import tempfile
from pathlib import Path

BLOCKED_ACTIONS = frozenset({
    "confirm_go",
    "brake_off",
    "come_home",
    "come_home_plan",
    "stand",
    "power_on",
    "mux_up",
    "sit",
})


def hold_error(reason, *, phase="stop"):
    return dict(active=True, hold_id="invalid", phase=phase, reason=reason, valid=False)


def parse_hold(value):
    if not isinstance(value, dict) or type(value.get("active")) is not bool:
        return hold_error("Invalid battery hold: active boolean required")
    if value.get("valid") is False:
        return hold_error(
            "Persisted invalid battery hold",
            phase="suspend" if value.get("phase") == "suspend" else "stop",
        )
    if not value["active"]:
        return dict(active=False, hold_id=None, phase="clear", reason="", valid=True)
    if (
        isinstance(value.get("hold_id"), bool)
        or not isinstance(value.get("hold_id"), (str, int))
        or not str(value["hold_id"]).strip()
        or value.get("phase") not in {"stop", "suspend"}
        or not isinstance(value.get("reason"), str)
    ):
        return hold_error("Invalid battery hold payload")
    return {key: value[key] for key in ("active", "hold_id", "phase", "reason")} | {"valid": True}


def _read(path, *, missing_clear=False):
    try:
        with Path(path).open() as stream:
            value = stream.read(16385)
        if len(value) > 16384:
            raise ValueError("oversized state")
        return parse_hold(json.loads(value))
    except FileNotFoundError:
        if missing_clear:
            return parse_hold({"active": False})
        return hold_error("Battery hold state missing")
    except (OSError, ValueError, TypeError):
        return hold_error("Battery hold state unreadable")


def atomic_write_hold(path, state):
    """Atomic, fsynced file and directory update; caller owns coordination."""
    path = Path(path)
    fd, temporary = tempfile.mkstemp(prefix=path.name + ".", dir=path.parent)
    try:
        with os.fdopen(fd, "w") as stream:
            json.dump(state, stream, sort_keys=True)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
        directory = os.open(path.parent, os.O_RDONLY | os.O_DIRECTORY)
        try:
            os.fsync(directory)
        finally:
            os.close(directory)
    finally:
        with contextlib.suppress(FileNotFoundError):
            os.unlink(temporary)


def read_hold(path):
    """Read-only effective state; malformed persisted latch means no output."""
    if not path or not Path(path).is_absolute():
        return hold_error("Absolute LAM_BATTERY_HOLD_FILE required")
    main = _read(path)
    latch_path = str(path) + ".mux-latch"
    latch = _read(latch_path, missing_clear=True)
    if not latch["valid"]:
        return hold_error("Invalid persisted mux latch; output must remain suspended", phase="suspend")
    if latch["active"]:
        if latch["phase"] == "suspend":
            return latch
        if main["active"] and main["valid"] and main["phase"] == "suspend":
            if main["hold_id"] == latch["hold_id"]:
                return main
            return hold_error("Hold identity changed during handoff")
        return latch
    return main


def env_hold(environ=None):
    environ = os.environ if environ is None else environ
    if environ.get("LAM_BATTERY_GUARD") != "1":
        return dict(active=False, hold_id=None, phase="clear", reason="", valid=True)
    return read_hold(environ.get("LAM_BATTERY_HOLD_FILE"))


def action_refusal(action, environ=None):
    state = env_hold(environ)
    if action in BLOCKED_ACTIONS and state["active"]:
        return "Battery motion hold: " + state["reason"] + "; qualified operator recovery required"
    return None


class MuxBatteryHold:
    """Owns persistent mux latch and same-path singleton process lock.

    All physical output publishers must use the SAME absolute hold path. ROS
    publisher counts are separately checked by the mux's live status before Sit.
    Flag-off behavior deliberately does not touch files or locks.
    """

    def __init__(self, environ=None):
        environ = os.environ if environ is None else environ
        self.enabled = environ.get("LAM_BATTERY_GUARD") == "1"
        self.path = environ.get("LAM_BATTERY_HOLD_FILE")
        self._lock_fd = None
        self._latched = None
        if not self.enabled:
            return
        if not self.path or not Path(self.path).is_absolute():
            raise ValueError("Absolute LAM_BATTERY_HOLD_FILE required")
        self._lock_fd = os.open(self.path + ".mux-owner.lock", os.O_CREAT | os.O_RDWR | os.O_CLOEXEC, 0o600)
        try:
            fcntl.flock(self._lock_fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BaseException:
            os.close(self._lock_fd)
            self._lock_fd = None
            raise RuntimeError("Another battery-aware mux owns this hold path") from None

    def snapshot(self):
        if not self.enabled:
            return parse_hold({"active": False})
        state = read_hold(self.path)
        if self._latched is not None:
            if self._latched["phase"] == "suspend":
                return self._latched.copy()
            if not state["active"]:
                state = self._latched.copy()
        if state["active"] and state != self._latched:
            try:
                atomic_write_hold(self.path + ".mux-latch", state)
            except OSError:
                # Without durable suspension proof the adapter must not Sit.
                state = hold_error("Cannot persist mux battery hold")
            self._latched = state.copy()
        return state

    @property
    def owns_lock(self):
        return self._lock_fd is not None

    def close(self):
        if self._lock_fd is not None:
            os.close(self._lock_fd)
            self._lock_fd = None
