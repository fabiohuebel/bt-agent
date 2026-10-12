"""Persistent hold and command interlock tests, entirely local files/fakes."""

import json

import pytest
from bt_agent_robotics._hardware.battery_hold_state import (
    BLOCKED_ACTIONS,
    MuxBatteryHold,
    action_refusal,
    atomic_write_hold,
    env_hold,
    read_hold,
)


def environment(tmp_path):
    path = tmp_path / "battery.json"
    atomic_write_hold(path, {"active": False})
    return {"LAM_BATTERY_GUARD": "1", "LAM_BATTERY_HOLD_FILE": str(path)}, path


def hold(phase="stop", hold_id="battery-1"):
    return dict(active=True, hold_id=hold_id, phase=phase, reason="Battery critical")


def test_flag_off_ignores_files_and_leaves_them_untouched(tmp_path):
    env = {"LAM_BATTERY_HOLD_FILE": str(tmp_path / "missing")}
    mux = MuxBatteryHold(env)
    assert not mux.snapshot()["active"]
    assert not env_hold(env)["active"]
    assert not mux.owns_lock
    assert list(tmp_path.iterdir()) == []


def test_stop_upgrade_suspend_clear_delete_and_restart_never_resume(tmp_path):
    env, path = environment(tmp_path)
    mux = MuxBatteryHold(env)
    assert not mux.snapshot()["active"]
    atomic_write_hold(path, hold())
    assert mux.snapshot()["phase"] == "stop"
    atomic_write_hold(path, hold("suspend"))
    assert mux.snapshot()["phase"] == "suspend"
    atomic_write_hold(path, {"active": False})
    assert mux.snapshot()["phase"] == "suspend"
    assert read_hold(path)["phase"] == "suspend"
    path.unlink()
    mux.close()
    restarted = MuxBatteryHold(env)
    assert restarted.snapshot()["phase"] == "suspend"
    restarted.close()


def test_stop_latch_survives_restart_and_main_clear(tmp_path):
    env, path = environment(tmp_path)
    mux = MuxBatteryHold(env)
    atomic_write_hold(path, hold())
    mux.snapshot()
    mux.close()
    atomic_write_hold(path, {"active": False})
    restarted = MuxBatteryHold(env)
    assert restarted.snapshot()["phase"] == "stop"
    restarted.close()


@pytest.mark.parametrize(
    "raw",
    ["garbage", "{}", "[]", '{"active": "false"}', '{"active":true,"phase":"release"}'],
)
def test_invalid_file_fails_closed(tmp_path, raw):
    _, path = environment(tmp_path)
    path.write_text(raw)
    state = read_hold(path)
    assert state["active"] and not state["valid"]


def test_corrupt_persistent_latch_suspends_but_not_valid_sit_evidence(tmp_path):
    env, path = environment(tmp_path)
    path.with_name(path.name + ".mux-latch").write_text("invalid")
    state = read_hold(path)
    assert state["active"] and state["phase"] == "suspend" and not state["valid"]
    mux = MuxBatteryHold(env)
    assert not mux.snapshot()["valid"]
    assert not mux.snapshot()["valid"]
    mux.close()
    restarted = MuxBatteryHold(env)
    assert not restarted.snapshot()["valid"]
    restarted.close()


def test_duplicate_mux_cannot_acquire_output_ownership(tmp_path):
    env, _ = environment(tmp_path)
    first = MuxBatteryHold(env)
    with pytest.raises(RuntimeError):
        MuxBatteryHold(env)
    first.close()
    second = MuxBatteryHold(env)
    second.close()


def test_missing_enabled_file_fails_closed(tmp_path):
    env, path = environment(tmp_path)
    path.unlink()
    mux = MuxBatteryHold(env)
    assert mux.snapshot()["active"]
    mux.close()


def test_hold_id_cannot_change_during_stop_to_suspend(tmp_path):
    env, path = environment(tmp_path)
    mux = MuxBatteryHold(env)
    atomic_write_hold(path, hold())
    mux.snapshot()
    atomic_write_hold(path, hold("suspend", "another"))
    state = mux.snapshot()
    assert state["phase"] == "stop" and not state["valid"]
    mux.close()


def test_persistence_failure_no_sit_ack(monkeypatch, tmp_path):
    env, path = environment(tmp_path)
    mux = MuxBatteryHold(env)
    atomic_write_hold(path, hold("suspend"))

    def fail(*_):
        raise OSError("disk full")

    monkeypatch.setattr("bt_agent_robotics._hardware.battery_hold_state.atomic_write_hold", fail)
    state = mux.snapshot()
    assert state["phase"] == "stop" and not state["valid"]
    mux.close()


def test_actions_fenced_but_stop_and_abort_remain(tmp_path):
    env, path = environment(tmp_path)
    atomic_write_hold(path, hold())
    for action in BLOCKED_ACTIONS:
        assert action_refusal(action, env)
    for action in (
        "brake_on",
        "confirm_abort",
        "agent_stop",
        "power_off",
        "shutdown_all",
    ):
        assert action_refusal(action, env) is None
    assert set(json.loads(path.read_text())) == {"active", "hold_id", "phase", "reason"}
