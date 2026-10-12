"""Off-robot adapter tests, including a complete mocked stop-to-Sit handoff."""

import io
import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace as NS

import pytest
from bt_agent_robotics._hardware.battery_guard import BatteryGuardPolicy
from bt_agent_robotics._hardware.battery_hold_state import atomic_write_hold
from bt_agent_robotics._hardware.battery_posture import BatteryPostureHandoff
from bt_agent_robotics._hardware.ros.battery_guard_bridge import (
    BatteryFencedPlanner,
    BatteryGuardBridge,
    battery_percent,
    seated_geometry,
)


class Backend:
    def __init__(self):
        self.calls = []

    def __getattr__(self, name):
        return lambda *args: self.calls.append((name, args))


def bridge(tmp_path):
    value = BatteryGuardBridge.__new__(BatteryGuardBridge)
    value.lock = threading.RLock()
    value.backend = Backend()
    value.node = NS(get_fully_qualified_name=lambda: "/battery_test")
    value.instance = "instance"
    value.hold_path = tmp_path / "hold.json"
    atomic_write_hold(value.hold_path, {"active": False})
    value.events, value.sent = [], []
    value.emit = value.events.append
    value.publish = lambda publisher, data: value.sent.append((publisher, data))
    value.policy = BatteryGuardPolicy(emit=value.emit)
    value.policy.update_real(90)
    value.planner = BatteryFencedPlanner(value, value.backend)
    value.handoff = BatteryPostureHandoff(value, telemetry_age=0.75, owner_age=2.0)
    value.seen = {}
    value.ack_pub, value.status_pub = "ack", "status"
    value.journal_path = tmp_path / "journal.jsonl"
    value.journal = io.StringIO()
    value.log_errors = value.posture_event_cursor = 0
    value.odom = value.brake = value.lease = value.feet = value.mux = None
    value.boot_id = "boot"
    value.restored_hold = False
    return value


def request(value, kind, request_id="r", **extra):
    value.on_request(NS(data=json.dumps(dict(kind=kind, instance=value.instance, request_id=request_id, **extra))))
    return value.sent[-1][1]


def test_battery_array_invalid_and_minimum():
    assert battery_percent(NS(battery_states=[])) is None
    assert (
        battery_percent(NS(battery_states=[NS(charge_percentage=80, status=2), NS(charge_percentage=40, status=3)]))
        == 40
    )
    assert battery_percent(NS(battery_states=[NS(charge_percentage=float("nan"), status=3)])) is None
    for status in (0, 1, 4):
        assert battery_percent(NS(battery_states=[NS(charge_percentage=0, status=status)])) is None


def test_simulation_guard_and_idempotent_requests(tmp_path):
    value = bridge(tmp_path)
    assert request(value, "set-simulated", percent=10)["ok"]
    assert request(value, "set-simulated", percent=10)["ok"]
    assert not request(value, "set-simulated", percent=5)["ok"]  # Same ID, different payload.
    assert value.policy.snapshot()["effective_percent"] == 10
    assert request(value, "set-simulated", request_id="r2", percent=5)["ok"]
    assert value.policy.sit_requested
    assert request(value, "clear-simulation", request_id="r3")["ok"]
    assert value.policy.snapshot()["held"]


@pytest.mark.parametrize("real", [49, None, float("nan")])
def test_simulation_denied_unhealthy_real(tmp_path, real):
    value = bridge(tmp_path)
    value.policy.update_real(real)
    assert not request(value, "set-simulated", percent=10)["ok"]


def test_simulation_denied_stale_real(tmp_path):
    value = bridge(tmp_path)
    value.policy.received_monotonic = time.monotonic() - 4
    assert not request(value, "set-simulated", percent=10)["ok"]


def test_fence_blocks_motion_allows_stopping(tmp_path):
    value = bridge(tmp_path)
    value.planner.send_waypoint(1, 2, 0)
    value.policy.set_override(5)
    with pytest.raises(RuntimeError):
        value.planner.send_waypoint(3, 4, 0)
    with pytest.raises(RuntimeError):
        value.planner.set_brake(False)
    value.planner.set_brake(True)
    value.planner.send_velocity(0, 0, 0)
    value.planner.cancel()
    assert len(value.backend.calls) == 4


def test_persistent_hold_blocks_restart_with_healthy_battery(tmp_path):
    value = bridge(tmp_path)
    atomic_write_hold(
        value.hold_path,
        {"active": True, "hold_id": "old", "phase": "stop", "reason": "test"},
    )
    assert value.policy.can_start_motion()
    with pytest.raises(RuntimeError):
        value.planner.send_velocity(1, 0, 0)
    with pytest.raises(RuntimeError):
        value._write_hold("new", "stop")


def test_storage_error_still_brakes(tmp_path, monkeypatch):
    value = bridge(tmp_path)
    value.policy.set_override(5)
    monkeypatch.setattr(value, "_write_hold", lambda *args: (_ for _ in ()).throw(OSError("disk full")))
    with pytest.raises(OSError):
        value.stop_for_guard(value.policy.snapshot())
    assert [name for name, _ in value.backend.calls] == ["set_brake", "send_velocity"]


def test_lease_body_only_and_posture_geometry(tmp_path):
    value = bridge(tmp_path)
    owner = NS(client_name="ros_spot", user_name="driver")
    value.on_lease(NS(resources=[NS(lease=NS(resource="body"), lease_owner=owner)]))
    assert value.lease[0]
    value.on_lease(NS(resources=[NS(lease=NS(resource="other"), lease_owner=owner)]))
    assert not value.lease[0]

    def foot(z):
        return NS(contact=1, foot_position_rt_body=NS(z=z))

    assert seated_geometry(NS(states=[foot(0.1)] * 4)) == "sitting"
    assert seated_geometry(NS(states=[foot(0.1)])) == "unknown"


def test_no_sit_without_mux_and_ownership_proof(tmp_path):
    value = bridge(tmp_path)
    value.policy.set_override(5)
    value.handoff.start("instance:1", time.monotonic())
    with pytest.raises(RuntimeError, match="Sit refused"):
        value.request_sit("instance:1")


def test_lease_age_is_separate_from_tight_mux_source_freshness(tmp_path, monkeypatch):
    value = bridge(tmp_path)
    monkeypatch.setattr("bt_agent_robotics._hardware.ros.battery_guard_bridge.time.monotonic", lambda: 100.0)
    monkeypatch.setattr("bt_agent_robotics._hardware.ros.battery_guard_bridge.mux_process_live", lambda _: True)
    value.handoff.start("instance:1", 99)
    value.lease, value.odom, value.brake = (True, 98.8), (0, 0, 100), (True, 100)
    status = {
        "hold_id": "instance:1",
        "exclusive_output_owner": True,
        "valid": True,
        "monotonic_s": 100,
        "phase": "stop",
    }
    value.mux = status, 100
    t = value.telemetry()
    assert t.owner_at == 98.8
    assert t.exclusive_owner and value.handoff._authority_ready(t, 100)
    for source_at in (99, 100.1):
        value.mux = {**status, "monotonic_s": source_at}, 100
        assert not value.telemetry().exclusive_owner
    value.mux = status, 99
    assert not value.telemetry().exclusive_owner  # stale receipt also fails


def test_mocked_complete_handoff_is_one_shot(tmp_path, monkeypatch):
    value = bridge(tmp_path)
    now = [100.0]
    monkeypatch.setattr("bt_agent_robotics._hardware.ros.battery_guard_bridge.time.monotonic", lambda: now[0])
    value.policy.clock = lambda: now[0]
    value.policy.received_monotonic = now[0]
    value.policy.set_override(5)
    calls = []
    future = NS(add_done_callback=lambda cb: calls.append("future_registered"))
    value.sit_client = NS(
        service_is_ready=lambda: True,
        call_async=lambda req: calls.append("sit") or future,
    )
    value.trigger = NS(Request=lambda: NS())

    def fresh(phase="stop", sitting=False):
        value.odom = (0, 0, now[0])
        value.brake = (True, now[0])
        value.lease = (True, now[0])
        value.feet = ("sitting" if sitting else "standing", now[0])
        value.mux = (
            {
                "hold_id": "instance:1",
                "valid": True,
                "phase": phase,
                "exclusive_output_owner": True,
                "output_suspended": phase == "suspend",
                "monotonic_s": now[0],
                "owner_pid": os.getpid(),
                "owner_start": Path(f"/proc/{os.getpid()}/stat").read_text().rsplit(")", 1)[1].split()[19],
            },
            now[0],
        )

    value.tick()
    assert value.handoff.state == "waiting_stop"
    fresh()
    value.tick()
    now[0] += 0.4
    fresh()
    value.tick()
    assert value.persisted_hold()["phase"] == "suspend"
    now[0] += 0.05
    fresh("suspend")
    value.tick()
    assert calls.count("sit") == 1
    now[0] += 0.1
    fresh("suspend", sitting=True)
    value.tick()
    assert value.handoff.state == "waiting_sit"
    now[0] += 0.31
    fresh("suspend", sitting=True)
    value.tick()
    assert value.handoff.state == "seated_held"
    value.tick()
    assert calls.count("sit") == 1
    assert value.persisted_hold()["active"]


def test_restart_never_repeats_sit(tmp_path):
    value = bridge(tmp_path)
    value.restored_hold = True
    atomic_write_hold(
        value.hold_path,
        {
            "active": True,
            "hold_id": "previous",
            "phase": "suspend",
            "reason": "critical",
        },
    )
    value.policy.set_override(5)
    value.tick()
    assert value.handoff.state == "idle"


@pytest.mark.parametrize("start_rejected", [True, False])
def test_actual_navigate_with_both_wrappers_no_leaked_scope(tmp_path, monkeypatch, start_rejected):
    from bt_agent.skills import SkillStatus
    from bt_agent_robotics._hardware.bt_controls import Controller, instrument_skill
    from bt_agent_robotics._hardware.skills.navigate import NavigateSkill
    from .test_bt_controls import Backend as RobotBackend
    from .test_bt_controls import Gate, Planner, start, wait

    monkeypatch.setenv("LAM_NAV_CONFIRM", "1")
    value = bridge(tmp_path)
    value.backend = RobotBackend()
    value.planner = BatteryFencedPlanner(value, value.backend)
    controller = Controller(value.planner)
    gate = Gate()
    skill = instrument_skill(NavigateSkill(value.planner, Planner(), gate), controller)
    value.instrument_skill(skill)
    value.policy.set_override(9 if start_rejected else 10)
    handle = start(skill)
    if not start_rejected:
        wait(lambda: bool(gate.plans))
        gate.decision = "go"
        wait(lambda: ("brake", False) in value.backend.events)
        value.policy.set_override(5)
    assert handle._completion_event.wait(2)
    assert handle.status == SkillStatus.FAILED
    assert "BatteryReserveGuard" in handle.observation
    if start_rejected:
        assert (
            handle.observation
            == "[leaf=BatteryReserveGuard] Navigation was not started: battery is below the required threshold."
        )
    else:
        assert "Navigation stopped before completion" in handle.observation
        assert "A stop-and-sit request was issued due to critical battery." in handle.observation
        assert "Sitting is not confirmed" not in handle.observation
        assert "Further motion is blocked pending operator reset" in handle.observation
    assert controller.snapshot()["active"] is None
    before = len(value.backend.events)
    retry = start(skill, "call2")
    assert retry._completion_event.wait(2)
    assert retry.status == SkillStatus.FAILED
    assert "Navigation was not started" in retry.observation
    assert ("stop-and-sit request was issued" in retry.observation) is not start_rejected
    assert controller.snapshot()["active"] is None
    assert ("brake", False) not in value.backend.events[before:]
    assert not any(event[0] == "waypoint" for event in value.backend.events[before:])
