"""Exercise actual mux callback methods with fake ROS imports, no publishers."""

import importlib.util
import json
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import pytest
from bt_agent_robotics._hardware.battery_hold_state import MuxBatteryHold, atomic_write_hold


class Message:
    def __init__(self):
        self.data = ""


class Publisher:
    def __init__(self):
        self.sent = []

    def publish(self, msg):
        self.sent.append(msg)


@pytest.fixture
def mux_class(monkeypatch):
    modules = {
        "rclpy": {},
        "rclpy.node": {"Node": type("Node", (), {})},
        "rclpy.qos": {"DurabilityPolicy": object, "QoSProfile": object},
        "geometry_msgs": {},
        "geometry_msgs.msg": {"Twist": Message},
        "std_msgs": {},
        "std_msgs.msg": {"String": Message, "Int8": Message},
    }
    for name, attrs in modules.items():
        module = ModuleType(name)
        module.__dict__.update(attrs)
        monkeypatch.setitem(sys.modules, name, module)
    path = Path(__import__("bt_agent_robotics").__file__).parent / "_hardware/ros/cmd_vel_mux.py"
    spec = importlib.util.spec_from_file_location("battery_fake_mux_test", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.CmdVelMux


def make_mux(cls, tmp_path, enabled=True):
    path = tmp_path / "battery.json"
    atomic_write_hold(path, {"active": False})
    mux = object.__new__(cls)
    mux._battery_hold = MuxBatteryHold({
        "LAM_BATTERY_GUARD": "1" if enabled else "0",
        "LAM_BATTERY_HOLD_FILE": str(path),
    })
    mux._pub = Publisher()
    mux._battery_status_pub = Publisher() if enabled else None
    mux._brake_state_pub = Publisher()
    mux._zero = "ZERO"
    mux._brake_released = True
    mux._inputs = []
    mux._log_transitions = False
    mux._owner_pid, mux._owner_start, mux._boot_id = 123, "start", "boot"
    mux._out_topic = "/cmd_vel"
    mux.count_publishers = lambda _: 1
    mux.get_clock = lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=1000000000))
    mux.get_logger = lambda: SimpleNamespace(info=lambda _: None)
    return mux, path


def test_real_tick_stop_then_suspend_emits_no_twist_even_on_stop_command(mux_class, tmp_path):
    mux, path = make_mux(mux_class, tmp_path)
    state = dict(active=True, hold_id="1", phase="stop", reason="critical")
    atomic_write_hold(path, state)
    mux._tick()
    assert mux._pub.sent == ["ZERO"]
    assert not mux._brake_released
    atomic_write_hold(path, {**state, "phase": "suspend"})
    mux._tick()
    for value in (0, 16, 0):
        mux._on_stop(SimpleNamespace(data=value))
        mux._tick()
    assert mux._pub.sent == ["ZERO"]  # no zero-Twist posture fight
    status = json.loads(mux._battery_status_pub.sent[-1].data)
    assert status["output_suspended"] and status["exclusive_output_owner"]
    assert status["hold_id"] == "1" and status["monotonic_s"] > 0
    assert mux._brake_state_pub.sent[-1].data == "ENGAGED"
    mux._battery_hold.close()


def test_duplicate_ros_publisher_not_exclusive_proof(mux_class, tmp_path):
    mux, path = make_mux(mux_class, tmp_path)
    mux.count_publishers = lambda _: 2
    atomic_write_hold(path, dict(active=True, hold_id="1", phase="suspend", reason="critical"))
    mux._tick()
    status = json.loads(mux._battery_status_pub.sent[-1].data)
    assert status["output_suspended"] and not status["exclusive_output_owner"]
    mux._battery_hold.close()


def test_flag_off_tick_and_brake_release_unchanged(mux_class, tmp_path):
    mux, path = make_mux(mux_class, tmp_path, enabled=False)
    atomic_write_hold(path, dict(active=True, hold_id="1", phase="suspend", reason="critical"))
    mux._tick()
    assert mux._pub.sent == ["ZERO"]  # normal no-fresh-input behavior
    mux._on_stop(SimpleNamespace(data=0))
    assert mux._brake_released
    mux._battery_hold.close()
