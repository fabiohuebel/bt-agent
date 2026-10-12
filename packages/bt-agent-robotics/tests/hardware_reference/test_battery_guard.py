"""Pure policy and real BT lifecycle tests; never connect to a robot."""

import math

import py_trees
import pytest
from bt_agent_robotics._hardware.battery_guard import BatteryGuard, BatteryGuardPolicy


class Clock:
    now = 100.0

    def __call__(self):
        return self.now


def make_policy(percent=100):
    clock, events = Clock(), []
    policy = BatteryGuardPolicy(clock=clock, emit=events.append)
    policy.update_real(percent)
    return policy, clock, events


@pytest.mark.parametrize(
    "value,allowed", [(100, True), (10, True), (9.99, False), (5.01, False), (5, False), (0, False)]
)
def test_start_threshold(value, allowed):
    policy, _, _ = make_policy(value)
    assert policy.can_start_motion() is allowed


def test_ongoing_between_thresholds_and_critical_latch():
    policy, _, events = make_policy(10)
    policy.set_motion_active(True)
    policy.update_real(8)
    assert policy.snapshot()["can_continue"]
    assert not policy.can_start_motion()
    policy.update_real(5)
    state = policy.snapshot()
    assert state["held"] and state["sit_requested"]
    assert not state["can_continue"]
    policy.update_real(100)
    assert not policy.can_start_motion()
    with pytest.raises(RuntimeError):
        policy.explicit_release()
    policy.set_motion_active(False)
    assert policy.explicit_release()
    assert policy.can_start_motion()
    assert not policy.explicit_release()
    assert sum(e["kind"] == "battery_sit_requested" for e in events) == 1


@pytest.mark.parametrize("bad", [None, math.nan, math.inf, -1, 101, "10", True])
def test_invalid_sample_blocks_and_cannot_be_overridden(bad):
    policy, _, _ = make_policy()
    policy.update_real(bad)
    policy.set_override(100)
    assert not policy.can_start_motion()
    assert not policy.sit_requested
    policy.set_motion_active(True)
    assert policy.snapshot()["held"]
    assert not policy.sit_requested


def test_stale_idle_and_active_behavior():
    policy, clock, _ = make_policy()
    clock.now += 3.01
    assert not policy.can_start_motion()
    assert not policy.snapshot()["held"]
    policy.set_motion_active(True)
    assert policy.snapshot()["held"]
    assert not policy.sit_requested
    policy.update_real(100)
    policy.set_motion_active(False)
    assert not policy.can_start_motion()  # Recovery requires operator release.
    policy.explicit_release()
    assert policy.can_start_motion()


def test_override_can_only_lower_and_cannot_rescue_stale():
    policy, clock, _ = make_policy(7)
    policy.set_override(10)
    assert policy.snapshot()["effective_percent"] == 7
    assert not policy.can_start_motion()
    clock.now += 4
    assert policy.snapshot()["effective_percent"] is None
    policy.set_override(5)
    assert not policy.sit_requested  # Unknown real data does not authorize sitting.


def test_demo_10_to_5_duplicate_and_latch():
    policy, _, events = make_policy(100)
    policy.set_override(10)
    assert policy.can_start_motion()
    policy.set_motion_active(True)
    policy.set_override(5)
    policy.set_override(5)
    assert policy.snapshot()["real_percent"] == 100
    assert policy.sit_requested
    assert sum(e["kind"] == "battery_sit_requested" for e in events) == 1
    event = next(e for e in events if e["kind"] == "battery_sit_requested")
    assert event["simulated"] is True
    policy.clear_override()
    policy.set_motion_active(False)
    assert policy.snapshot()["held"]
    policy.explicit_release()
    policy.set_override(5)
    assert policy.snapshot()["hold_id"] == 2


def test_stale_hold_can_upgrade_to_critical_without_new_hold():
    policy, clock, events = make_policy()
    policy.set_motion_active(True)
    clock.now += 4
    assert policy.snapshot()["held"]
    policy.update_real(5)
    assert policy.sit_requested
    assert policy.snapshot()["hold_id"] == 1
    assert sum(e["kind"] == "battery_hold_latched" for e in events) == 1


def test_future_and_out_of_order_timestamps():
    policy, clock, _ = make_policy(80)
    assert not policy.update_real(5, clock.now - 1)
    assert policy.can_start_motion()
    policy.update_real(80, clock.now + 1)
    assert not policy.can_start_motion()
    policy.update_real(80, math.nan)
    assert not policy.can_start_motion()


@pytest.mark.parametrize("bad", [None, math.nan, -1, 101, True])
def test_bad_override_rejected(bad):
    policy, _, _ = make_policy()
    with pytest.raises(ValueError):
        policy.set_override(bad)


def test_release_rejects_low_or_stale():
    policy, clock, _ = make_policy(5)
    with pytest.raises(RuntimeError):
        policy.explicit_release()
    policy.update_real(10)
    clock.now += 4
    with pytest.raises(RuntimeError):
        policy.explicit_release()


class MovingChild(py_trees.behaviour.Behaviour):
    def __init__(self, events):
        super().__init__("FollowPath")
        self.events = events
        self.result = py_trees.common.Status.RUNNING

    def update(self):
        self.events.append("motion_tick")
        return self.result

    def terminate(self, status):
        self.events.append(("child_terminated", status))


def test_real_bt_stops_before_child_invalidation_and_agent_can_continue():
    policy, _, _ = make_policy()
    events = []
    child = MovingChild(events)
    guard = BatteryGuard(child, policy, lambda state: events.append("fence_and_brake"))
    guard.tick_once()
    assert events == ["motion_tick"]
    policy.set_override(5)
    guard.tick_once()
    assert events == ["motion_tick", "fence_and_brake", ("child_terminated", py_trees.common.Status.INVALID)]
    assert guard.status == py_trees.common.Status.FAILURE
    assert "battery_critical" in guard.feedback_message
    assert not policy.snapshot()["motion_active"]
    # The caller consumes normal failure; no task-agent cancellation occurs.
    guard.tick_once()  # An actor retry cannot move while held.
    assert events.count("motion_tick") == 1


def test_real_bt_low_precondition_never_ticks_child():
    policy, _, _ = make_policy(9)
    events = []
    guard = BatteryGuard(MovingChild(events), policy, lambda state: events.append("stop"))
    guard.tick_once()
    assert events == ["stop"]
    assert guard.status == py_trees.common.Status.FAILURE


def test_real_bt_staleness_fails_without_sit():
    policy, clock, _ = make_policy()
    events = []
    guard = BatteryGuard(MovingChild(events), policy, lambda state: events.append("stop"))
    guard.tick_once()
    clock.now += 4
    guard.tick_once()
    assert guard.status == py_trees.common.Status.FAILURE
    assert policy.snapshot()["held"] and not policy.sit_requested


def test_success_clears_active_without_hold():
    policy, _, _ = make_policy()
    child = MovingChild([])
    child.result = py_trees.common.Status.SUCCESS
    guard = BatteryGuard(child, policy, lambda _: pytest.fail("unexpected stop"))
    guard.tick_once()
    assert guard.status == py_trees.common.Status.SUCCESS
    assert not policy.snapshot()["motion_active"]
    assert policy.can_start_motion()


def test_diagnostic_and_stop_errors_do_not_tick_child():
    policy, _, _ = make_policy()
    events = []
    guard = BatteryGuard(MovingChild(events), policy, lambda _: (_ for _ in ()).throw(RuntimeError("transport")))
    guard.tick_once()
    policy.emit = lambda _: (_ for _ in ()).throw(RuntimeError("journal"))
    policy.set_override(5)
    guard.tick_once()
    assert guard.status == py_trees.common.Status.FAILURE
    assert events.count("motion_tick") == 1
