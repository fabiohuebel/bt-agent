"""Pure fake-backend qualification: no ROS, process control or robot access."""

from dataclasses import replace

import pytest
from bt_agent_robotics._hardware.battery_posture import BatteryPostureHandoff, PostureTelemetry


class Backend:
    def __init__(self, fail=None):
        self.calls = []
        self.fail = fail

    def _call(self, name, key):
        self.calls.append((name, key))
        if name == self.fail:
            raise RuntimeError("refused")

    def enter_motion_hold(self, key):
        self._call("hold", key)

    def request_stop(self, key):
        self._call("stop", key)

    def request_mux_suspend(self, key):
        self._call("mux", key)

    def request_sit(self, key):
        self._call("sit", key)


def telemetry(now, **changes):
    value = PostureTelemetry(
        linear_speed=0,
        angular_speed=0,
        odom_at=now,
        brake_engaged=True,
        brake_at=now,
        exclusive_owner=True,
        owner_at=now,
        hold_active=True,
        hold_at=now,
        mux_suspended=True,
        mux_at=now,
        posture="standing",
        posture_at=now,
    )
    return replace(value, **changes)


def started(**kwargs):
    backend = Backend()
    machine = BatteryPostureHandoff(backend, **kwargs)
    assert machine.start("critical-1", 0)
    return backend, machine


def through_sit(backend, machine):
    machine.tick(0, telemetry(0))
    machine.tick(0.31, telemetry(0.31))
    machine.tick(0.32, telemetry(0.32))
    assert [c[0] for c in backend.calls] == ["hold", "stop", "mux", "sit"]
    assert machine.state == "waiting_sit"


def test_exact_order_ack_is_not_posture_confirmation_and_hold_persists():
    backend, machine = started()
    through_sit(backend, machine)
    machine.sit_acknowledged(accepted=True, now=0.33)
    assert machine.state == "waiting_sit"
    machine.tick(0.34, telemetry(0.34, posture="sitting", posture_at=0.32))
    assert machine.state == "waiting_sit"  # no pre-command/cached success
    machine.tick(0.35, telemetry(0.35, posture="sitting"))
    assert machine.state == "waiting_sit"
    machine.tick(0.66, telemetry(0.66, posture="sitting"))
    assert machine.state == "seated_held"
    assert machine.motion_held
    assert machine.events[-1]["kind"] == "seated_confirmed"
    assert all(c[1] == "critical-1" for c in backend.calls)


@pytest.mark.parametrize(
    "changes",
    [
        {"odom_at": None},
        {"odom_at": -10},
        {"odom_at": 10},
        {"linear_speed": None},
        {"linear_speed": float("nan")},
        {"linear_speed": 0.1},
        {"angular_speed": -0.1},
        {"brake_engaged": False},
        {"brake_at": -10},
        {"exclusive_owner": False},
        {"owner_at": -10},
        {"hold_active": False},
        {"hold_at": -10},
    ],
)
def test_missing_stale_moving_no_authority_fail_closed(changes):
    backend, machine = started(stage_timeout=1)
    machine.tick(0, telemetry(0, **changes))
    machine.tick(1, telemetry(1, **changes))
    assert machine.state == "held_error"
    assert machine.motion_held
    assert [c[0] for c in backend.calls] == ["hold", "stop"]


def test_one_odom_sample_does_not_establish_sustained_stop():
    backend, machine = started()
    machine.tick(0, telemetry(0))
    machine.tick(0.4, telemetry(0.4, odom_at=0))
    assert machine.state == "waiting_stop"
    assert [c[0] for c in backend.calls] == ["hold", "stop"]


@pytest.mark.parametrize("changes", [{"mux_suspended": False}, {"mux_at": -10}])
def test_mux_not_off_or_stale_never_sends_sit(changes):
    backend, machine = started(stage_timeout=1)
    machine.tick(0, telemetry(0))
    machine.tick(0.31, telemetry(0.31))
    machine.tick(0.32, telemetry(0.32, **changes))
    machine.tick(1.32, telemetry(1.32, **changes))
    assert machine.state == "held_error"
    assert [c[0] for c in backend.calls] == ["hold", "stop", "mux"]


def test_ownership_lost_between_mux_and_sit():
    backend, machine = started(stage_timeout=1)
    machine.tick(0, telemetry(0))
    machine.tick(0.31, telemetry(0.31))
    machine.tick(1.32, telemetry(1.32, exclusive_owner=False))
    assert machine.state == "held_error"
    assert [c[0] for c in backend.calls] == ["hold", "stop", "mux"]


@pytest.mark.parametrize("failure", ["hold", "stop", "mux", "sit"])
def test_command_failure_latches_without_retry(failure):
    backend = Backend(fail=failure)
    machine = BatteryPostureHandoff(backend)
    machine.start("critical-1", 0)
    for now in (0, 0.31, 0.32, 1, 2):
        machine.tick(now, telemetry(now))
    assert machine.state == "held_error"
    assert machine.motion_held
    assert not machine.start("critical-2", 3)
    assert [c[0] for c in backend.calls].count(failure) == 1
    assert ("stop", "critical-1") in backend.calls  # also if hold failed


@pytest.mark.parametrize("denied", [True, False])
def test_denied_or_timed_out_sit_never_retries(denied):
    backend, machine = started(sit_timeout=1)
    through_sit(backend, machine)
    if denied:
        machine.sit_acknowledged(accepted=False, now=0.33)
    else:
        machine.sit_acknowledged(accepted=True, now=0.33)
        machine.tick(1.33, telemetry(1.33))
    assert machine.state == "held_error"
    assert not machine.start("critical-1", 2)
    assert not machine.start("critical-2", 3)
    assert [c[0] for c in backend.calls].count("sit") == 1


def test_repeated_critical_events_after_success_never_resend():
    backend, machine = started()
    through_sit(backend, machine)
    machine.tick(0.5, telemetry(0.5, posture="sitting"))
    machine.tick(0.81, telemetry(0.81, posture="sitting"))
    assert not machine.start("critical-2", 1)
    assert machine.state == "seated_held"
    assert len(backend.calls) == 4


def test_mux_restarts_after_sit_is_error_not_success():
    backend, machine = started()
    through_sit(backend, machine)
    machine.tick(0.5, telemetry(0.5, posture="sitting", mux_suspended=False))
    assert machine.state == "held_error"


def test_backward_clock_is_rejected():
    _, machine = started()
    machine.tick(1, telemetry(1))
    with pytest.raises(ValueError):
        machine.tick(0.5, telemetry(0.5))


def test_separate_owner_freshness_accepts_one_hz_lease_without_loosening_odom_or_mux():
    _, machine = started(telemetry_age=0.75, owner_age=2.0)
    assert machine._authority_ready(telemetry(2, owner_at=0.8), 2)
    assert not machine._authority_ready(telemetry(2, owner_at=0.8, odom_at=1), 2)
    assert not machine._authority_ready(telemetry(2, owner_at=-0.01), 2)
    backend, machine = started(telemetry_age=0.75, owner_age=2.0)
    through_sit(backend, machine)
    machine.tick(0.5, telemetry(0.5, owner_at=-0.7, mux_at=-0.3))
    assert machine.state == "held_error"  # stale mux is still rejected at .75 s


def test_owner_freshness_defaults_to_telemetry_freshness():
    _, machine = started(telemetry_age=0.75)
    assert not machine._authority_ready(telemetry(2, owner_at=0.8), 2)


def test_normal_posture_motion_during_sit_then_sustained_seated_confirmation():
    backend, machine = started()
    through_sit(backend, machine)
    machine.tick(0.5, telemetry(0.5, posture="transitioning", linear_speed=0.08, angular_speed=0.05))
    assert machine.state == "waiting_sit"
    machine.tick(0.8, telemetry(0.8, posture="sitting"))
    assert machine.state == "waiting_sit"
    machine.tick(1.11, telemetry(1.11, posture="sitting"))
    assert machine.state == "seated_held"
    assert len(backend.calls) == 4


def test_one_seated_sample_and_a_seated_blip_never_confirm():
    backend, machine = started()
    through_sit(backend, machine)
    machine.tick(0.4, telemetry(0.4, posture="sitting"))
    machine.tick(0.71, telemetry(0.71, posture="sitting", posture_at=0.4))
    assert machine.state == "waiting_sit"
    machine.tick(0.72, telemetry(0.72, posture="transitioning"))
    machine.tick(0.8, telemetry(0.8, posture="sitting"))
    machine.tick(1, telemetry(1, posture="sitting"))
    assert machine.state == "waiting_sit"
    machine.tick(1.11, telemetry(1.11, posture="sitting"))
    assert machine.state == "seated_held"


@pytest.mark.parametrize(
    "changes",
    [
        {"exclusive_owner": False},
        {"hold_active": False},
        {"brake_engaged": False},
        {"mux_suspended": False},
        {"odom_at": -10},
    ],
)
def test_posture_motion_does_not_excuse_lost_authority_or_freshness(changes):
    backend, machine = started()
    through_sit(backend, machine)
    machine.tick(0.5, telemetry(0.5, posture="transitioning", linear_speed=0.1, **changes))
    assert machine.state == "held_error"
    assert len(backend.calls) == 4
