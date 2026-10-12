"""Fail-closed brake gate in cmd_vel_mux (2026-08-14 incident).

The robot moved on mux-up with the operator brake engaged: /stop gates only
pathFollower, so a stale localPlanner goal regenerated fresh navigate
commands, and primitive commands never had a brake at all. The mux is the
final software output before /spot/cmd_vel — it must therefore be its own
brake authority: born braked, gating EVERY input, released only by an
explicit /stop==0 received after its own startup.

Pure-logic tests, no rclpy (host test env has no ROS).
"""

from __future__ import annotations

from bt_agent_robotics._hardware.ros.cmd_vel_mux_logic import (
    MuxInput,
    brake_released_after_stop,
    select_command,
)


def _inputs(now: float) -> list:
    navigate = MuxInput(name="navigate", topic="/spot/cmd_vel_navigate", priority=10, timeout_s=0.5)
    primitive = MuxInput(name="primitive", topic="/spot/cmd_vel_primitive", priority=20, timeout_s=0.25)
    navigate.last_msg = "nav-twist"
    navigate.last_ts = now - 0.1
    primitive.last_msg = "prim-twist"
    primitive.last_ts = now - 0.1
    return [navigate, primitive]


def test_incident_regression_braked_mux_outputs_zero_despite_fresh_inputs():
    """The exact incident shape: fresh commands on BOTH inputs, brake never
    released since mux startup -> output must be zero."""
    now = 100.0
    winner, publish_zero = select_command(_inputs(now), now, brake_released=False)
    assert winner is None
    assert publish_zero is True


def test_startup_state_is_braked_until_explicit_post_startup_release():
    # No /stop observed yet -> stays braked.
    assert brake_released_after_stop(None) is False
    # Engage value (16) -> braked.
    assert brake_released_after_stop(16) is False
    # Any nonzero value is engage-like; only exactly 0 releases.
    assert brake_released_after_stop(2) is False
    assert brake_released_after_stop(0) is True


def test_released_mux_arbitrates_by_priority_and_freshness():
    now = 100.0
    inputs = _inputs(now)
    winner, publish_zero = select_command(inputs, now, brake_released=True)
    assert publish_zero is False
    assert winner.name == "primitive"  # higher priority wins

    inputs[1].last_ts = now - 0.3  # primitive stale (timeout 0.25)
    winner, _ = select_command(inputs, now, brake_released=True)
    assert winner.name == "navigate"

    inputs[0].last_ts = now - 0.6  # navigate stale too
    winner, publish_zero = select_command(inputs, now, brake_released=True)
    assert winner is None
    assert publish_zero is True


def test_reengaging_the_brake_regates_a_released_mux():
    now = 100.0
    released = brake_released_after_stop(0)
    assert released is True
    released = brake_released_after_stop(16)
    assert released is False
    winner, publish_zero = select_command(_inputs(now), now, brake_released=released)
    assert winner is None
    assert publish_zero is True
