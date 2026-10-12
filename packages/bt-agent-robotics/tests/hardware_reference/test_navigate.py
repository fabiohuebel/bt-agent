"""Tests for the B1 navigate-BT hardening fixes (bt-nav-fixes spec).

Covers, per docs/spec/bt-nav-fixes.md Checkpoint 1:

- Replan budget (N=3, backend-agnostic) in ``_FollowPath``: a stuck robot
  where replan keeps returning the identical waypoint sequence fails with
  ``arrival_status="failed_blocked"`` instead of looping forever.
- No-progress guard: distance-to-waypoint not improving over T seconds
  triggers the same recovery branch as the per-waypoint timeout, earlier.
- Pose freshness: stale pose fails the safeguard invariant and is refused
  by ``_CheckAlreadyAtGoal`` / ``_ConfirmArrival``; no stamp available
  degrades gracefully (today's behavior).
- ``arrived_short``: settle-timeout force-accept beyond the arrival
  distance sets ``arrival_status="arrived_short"`` + ``residual_distance_m``.

No ROS spin or sim — pure pycore/bt_skills unit tests with hand-rolled
test doubles (the shared ``MockLocalPlanner`` moves the robot toward the
target, which is the opposite of what "stuck" scenarios need here).
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

import py_trees
from bt_agent.skills import SkillContext, SkillHandle, SkillStatus
from bt_agent_robotics._hardware.skills.navigate import (
    _ARRIVAL_DISTANCE,
    _MAX_REPLAN_ATTEMPTS,
    _NO_PROGRESS_EPS_M,
    _NO_PROGRESS_WINDOW_S,
    _POSE_STALENESS_S,
    _WALL_TIME_BUDGET_CAP_S,
    _WALL_TIME_BUDGET_FLOOR_S,
    _WAYPOINT_TIMEOUT,
    NOMINAL_WALK_SPEED_MPS,
    NavigateSkill,
    _CheckAlreadyAtGoal,
    _ConfirmArrival,
    _FollowPath,
    _park_at_self,
    _path_length_m,
    _PlanPath,
    _scaled_wall_time_budget_s,
    _TerminalDiscipline,
)
from bt_agent_robotics.skills.utils.path_planner import PathPlanner

# ---------------------------------------------------------------------------
# Test doubles
# ---------------------------------------------------------------------------


class _FakeClock:
    """Controllable stand-in for ``time.monotonic`` — no real sleeping."""

    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


class _StuckLocalPlanner:
    """Local planner whose pose never moves — simulates a physically stuck robot.

    Implements ``LocalPlannerInterface`` by duck typing (matches
    ``MockLocalPlanner`` convention in ``lam_v2_testing.mocks``).
    """

    def __init__(self, pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        self._pose = pose
        self.sent_waypoints: List[Tuple[float, float, float]] = []
        self.cancelled = False
        self._pose_age: Optional[float] = None  # None => no freshness tracked
        # Terminal-discipline observability: brake edges plus a single
        # ordered event log, so tests can assert *sequence* (release before
        # the first goal, park before the brake) and not just membership.
        self.brake_calls: List[bool] = []
        self.events: List[str] = []

    def send_waypoint(self, x: float, y: float, theta: float) -> None:
        self.sent_waypoints.append((x, y, theta))
        self.events.append("waypoint")

    def send_velocity(self, vx_body: float, vy_body: float, wz: float) -> None:
        pass

    def get_robot_pose(self) -> Tuple[float, float, float]:
        return self._pose

    def get_robot_velocity(self) -> Tuple[float, float]:
        return (0.0, 0.0)

    def cancel(self) -> None:
        self.cancelled = True

    def get_pose_age(self) -> Optional[float]:
        return self._pose_age

    def set_brake(self, engaged: bool) -> None:
        self.brake_calls.append(engaged)
        self.events.append("brake_on" if engaged else "brake_off")


class _SettledLocalPlanner(_StuckLocalPlanner):
    """Pose fixed, velocity reports "stopped" — for _ConfirmArrival tests."""


class _FixedPathPlanner(PathPlanner):
    """Always returns the same waypoint sequence — "stuck" per the spec's evidence."""

    def __init__(self, waypoints: List[Tuple[float, float]], threshold: float = 0.3) -> None:
        self._waypoints = list(waypoints)
        self._threshold = threshold
        self.replan_calls = 0

    def plan(self, start_xy, goal_xy) -> Optional[List[Tuple[float, float]]]:
        return list(self._waypoints)

    def replan(self, robot_xy, goal_xy) -> Optional[List[Tuple[float, float]]]:
        self.replan_calls += 1
        return list(self._waypoints)  # identical sequence every time — stuck

    def update_map(self, map_data: Any) -> None:
        pass

    @property
    def waypoint_threshold(self) -> float:
        return self._threshold


class _RecoveringPathPlanner(_FixedPathPlanner):
    """First replan() call returns a genuinely different path."""

    def __init__(self, waypoints, recovered_waypoints, threshold: float = 0.3) -> None:
        super().__init__(waypoints, threshold)
        self._recovered = list(recovered_waypoints)

    def replan(self, robot_xy, goal_xy) -> Optional[List[Tuple[float, float]]]:
        self.replan_calls += 1
        return list(self._recovered)


class _NoPathReplanner(_FixedPathPlanner):
    """replan() always returns no path — target became unreachable mid-route."""

    def replan(self, robot_xy, goal_xy) -> Optional[List[Tuple[float, float]]]:
        self.replan_calls += 1
        return


class _PingPongPathPlanner(PathPlanner):
    """Alternates between two distinct, non-improving waypoint sequences.

    Simulates odometry noise flipping the nearest graph node: the path
    genuinely differs tick to tick, but the robot (frozen in these tests)
    never gets any closer to the goal.
    """

    def __init__(self, path_a: List[Tuple[float, float]], path_b: List[Tuple[float, float]], threshold=0.3):
        self._path_a = list(path_a)
        self._path_b = list(path_b)
        self._threshold = threshold
        self.replan_calls = 0

    def plan(self, start_xy, goal_xy) -> Optional[List[Tuple[float, float]]]:
        return list(self._path_a)

    def replan(self, robot_xy, goal_xy) -> Optional[List[Tuple[float, float]]]:
        self.replan_calls += 1
        return list(self._path_b) if self.replan_calls % 2 else list(self._path_a)

    def update_map(self, map_data: Any) -> None:
        pass

    @property
    def waypoint_threshold(self) -> float:
        return self._threshold


def _make_params(goal_x: float = 5.0, goal_y: float = 5.0, goal_theta: float = 0.0) -> Dict[str, Any]:
    return {
        "goal_x": goal_x,
        "goal_y": goal_y,
        "goal_theta": goal_theta,
        "target_handle": "obj_1",
        "target_label": "target",
        "viewpoint_index": 0,
        "total_viewpoints": 1,
        "is_exploration": False,
    }


def _make_handle(waypoints: Optional[List[Tuple[float, float]]] = None) -> SkillHandle:
    h = SkillHandle()
    h.status = SkillStatus.RUNNING
    if waypoints is not None:
        h.result_data = {"_waypoints": waypoints, "planned_path": waypoints}
    return h


def _tick(node: py_trees.behaviour.Behaviour) -> py_trees.common.Status:
    for _ in node.tick():
        pass
    return node.status


# ---------------------------------------------------------------------------
# 1. Replan budget (N=3, backend-agnostic, same-path detection)
# ---------------------------------------------------------------------------


def test_stuck_robot_fails_blocked_after_n_replan_attempts(monkeypatch) -> None:
    """PrecomputedGraphPlanner-style stuck robot: replan always returns the
    identical waypoint sequence. Fails with arrival_status=failed_blocked
    after exactly _MAX_REPLAN_ATTEMPTS replan() calls — not an unbounded
    loop toward the 120s wall guard.
    """
    clock = _FakeClock(0.0)
    monkeypatch.setattr("bt_agent_robotics._hardware.skills.navigate.time.monotonic", clock)

    waypoints = [(5.0, 5.0)]
    planner = _FixedPathPlanner(waypoints, threshold=0.3)
    local_planner = _StuckLocalPlanner(pose=(0.0, 0.0, 0.0))
    params = _make_params()
    handle = _make_handle(waypoints)

    node = _FollowPath(params, handle, planner, local_planner)

    # First tick: sends the waypoint.
    assert _tick(node) == py_trees.common.Status.RUNNING

    status = py_trees.common.Status.RUNNING
    for _ in range(_MAX_REPLAN_ATTEMPTS):
        clock.advance(_WAYPOINT_TIMEOUT + 0.1)
        status = _tick(node)

    assert status == py_trees.common.Status.FAILURE
    assert planner.replan_calls == _MAX_REPLAN_ATTEMPTS
    assert handle.result_data["arrival_status"] == "failed_blocked"
    # ~30s at default timeouts (N=3 * _WAYPOINT_TIMEOUT=10s), never near 120s.
    assert clock.now < 60.0


def test_replan_returning_new_path_resets_budget_and_recovers(monkeypatch) -> None:
    """A replan that reflects genuine distance-to-goal progress (the robot
    actually got closer) resets the budget — path adoption alone does not.
    """
    clock = _FakeClock(0.0)
    monkeypatch.setattr("bt_agent_robotics._hardware.skills.navigate.time.monotonic", clock)

    waypoints = [(5.0, 5.0)]
    recovered = [(1.0, 1.0)]
    planner = _RecoveringPathPlanner(waypoints, recovered, threshold=0.3)
    local_planner = _StuckLocalPlanner(pose=(0.0, 0.0, 0.0))
    params = _make_params()
    handle = _make_handle(waypoints)

    node = _FollowPath(params, handle, planner, local_planner)
    _tick(node)  # send first waypoint

    # Robot actually advances toward the goal before the stuck check fires
    # — real progress, not merely a differently-shaped path.
    local_planner._pose = (3.0, 3.0, 0.0)
    clock.advance(_WAYPOINT_TIMEOUT + 0.1)
    status = _tick(node)

    assert status == py_trees.common.Status.RUNNING
    assert node._waypoints == recovered
    assert node._replan_attempts == 0  # budget reset on genuine distance progress


def test_path_change_alone_without_distance_progress_does_not_reset_budget(monkeypatch) -> None:
    """A replan that returns a genuinely different path is NOT, by itself,
    progress — only real distance-to-goal improvement resets the budget.
    Guards the exact gap that let A->B->A ping-pong reset the budget
    forever (the 120s-spin defect under noisy localization).
    """
    clock = _FakeClock(0.0)
    monkeypatch.setattr("bt_agent_robotics._hardware.skills.navigate.time.monotonic", clock)

    waypoints = [(5.0, 5.0)]
    recovered = [(1.0, 1.0)]
    planner = _RecoveringPathPlanner(waypoints, recovered, threshold=0.3)
    local_planner = _StuckLocalPlanner(pose=(0.0, 0.0, 0.0))  # never moves
    params = _make_params()
    handle = _make_handle(waypoints)

    node = _FollowPath(params, handle, planner, local_planner)
    _tick(node)  # send first waypoint

    clock.advance(_WAYPOINT_TIMEOUT + 0.1)
    status = _tick(node)

    assert status == py_trees.common.Status.RUNNING
    assert node._waypoints == recovered  # new path is still adopted
    assert node._replan_attempts == 1  # but the budget is NOT reset


def test_ping_pong_paths_do_not_reset_budget_and_eventually_fail(monkeypatch) -> None:
    """A->B->A path oscillation (e.g. odometry noise flipping the nearest
    graph node) must not reset the recovery budget just because the path
    differs from the previous tick — the budget must still exhaust and
    fail, never spinning to the 120s wall guard.
    """
    clock = _FakeClock(0.0)
    monkeypatch.setattr("bt_agent_robotics._hardware.skills.navigate.time.monotonic", clock)

    path_a = [(5.0, 5.0)]
    path_b = [(4.0, 6.0)]
    planner = _PingPongPathPlanner(path_a, path_b, threshold=0.3)
    local_planner = _StuckLocalPlanner(pose=(0.0, 0.0, 0.0))  # never moves
    params = _make_params()
    handle = _make_handle(path_a)

    node = _FollowPath(params, handle, planner, local_planner)
    assert _tick(node) == py_trees.common.Status.RUNNING  # sends first waypoint

    status = py_trees.common.Status.RUNNING
    for _ in range(_MAX_REPLAN_ATTEMPTS):
        clock.advance(_WAYPOINT_TIMEOUT + 0.1)
        status = _tick(node)
        if status == py_trees.common.Status.RUNNING and not node._sent:
            # Adopting the (different) replanned path consumes one tick to
            # resend the waypoint before the next stuck check is evaluated.
            status = _tick(node)

    assert status == py_trees.common.Status.FAILURE
    assert planner.replan_calls == _MAX_REPLAN_ATTEMPTS
    assert handle.result_data["arrival_status"] == "failed_blocked"
    assert clock.now < 60.0  # never near the 120s wall guard


# ---------------------------------------------------------------------------
# 1b. failed_no_path — exhausting event is a replan that returned no path
# ---------------------------------------------------------------------------


def test_stuck_robot_exhausting_on_no_path_replan_fails_no_path(monkeypatch) -> None:
    """When the budget-exhausting replan returns no path at all, the typed
    status is failed_no_path (actionable: try a different viewpoint) —
    distinct from failed_blocked (identical/oscillating paths).
    """
    clock = _FakeClock(0.0)
    monkeypatch.setattr("bt_agent_robotics._hardware.skills.navigate.time.monotonic", clock)

    waypoints = [(5.0, 5.0)]
    planner = _NoPathReplanner(waypoints, threshold=0.3)
    local_planner = _StuckLocalPlanner(pose=(0.0, 0.0, 0.0))
    params = _make_params()
    handle = _make_handle(waypoints)

    node = _FollowPath(params, handle, planner, local_planner)
    assert _tick(node) == py_trees.common.Status.RUNNING

    status = py_trees.common.Status.RUNNING
    for _ in range(_MAX_REPLAN_ATTEMPTS):
        clock.advance(_WAYPOINT_TIMEOUT + 0.1)
        status = _tick(node)

    assert status == py_trees.common.Status.FAILURE
    assert handle.result_data["arrival_status"] == "failed_no_path"


def test_same_path_genuine_progress_resets_timer_no_burst_within_timeout_window(monkeypatch) -> None:
    """Regression: a stuck-branch entry that returns the identical path
    (same_path, i.e. got_new_path=False) but where the robot has genuinely
    advanced must still reset the waypoint timer + no-progress tracker on
    entry — not only when a differently-shaped path was adopted.
    Otherwise the very next tick immediately re-enters the stuck branch
    (elapsed still > _WAYPOINT_TIMEOUT) and replans/increments the budget
    at tick rate instead of ~_WAYPOINT_TIMEOUT cadence.
    """
    clock = _FakeClock(0.0)
    monkeypatch.setattr("bt_agent_robotics._hardware.skills.navigate.time.monotonic", clock)

    waypoints = [(5.0, 5.0)]
    planner = _FixedPathPlanner(waypoints, threshold=0.3)  # always identical path
    local_planner = _StuckLocalPlanner(pose=(0.0, 0.0, 0.0))
    params = _make_params()
    handle = _make_handle(waypoints)

    node = _FollowPath(params, handle, planner, local_planner)
    _tick(node)  # sends first waypoint

    # Robot genuinely advances toward the goal before the timeout fires.
    local_planner._pose = (3.0, 3.0, 0.0)
    clock.advance(_WAYPOINT_TIMEOUT + 0.1)
    status = _tick(node)

    assert status == py_trees.common.Status.RUNNING
    assert planner.replan_calls == 1
    assert node._replan_attempts == 0  # genuine progress reset the budget

    # Next tick immediately (no clock advance). If the timer/no-progress
    # tracker weren't reset on entry, elapsed would still exceed
    # _WAYPOINT_TIMEOUT and the stuck branch would fire again right away.
    status = _tick(node)

    assert status == py_trees.common.Status.RUNNING
    assert planner.replan_calls == 1  # no burst — did not re-enter the stuck branch
    assert node._replan_attempts == 0


def test_best_dist_to_goal_only_updates_on_supra_eps_improvement(monkeypatch) -> None:
    """Sub-epsilon creep must not ratchet the goal-distance baseline down
    every cycle — otherwise a slowly-but-truly advancing robot never
    accumulates enough gap-vs-baseline to trigger genuine_progress, and
    the budget exhausts even though real cumulative progress (> eps) was
    made across several sub-eps increments.
    """
    clock = _FakeClock(0.0)
    monkeypatch.setattr("bt_agent_robotics._hardware.skills.navigate.time.monotonic", clock)

    waypoints = [(5.0, 5.0)]
    planner = _FixedPathPlanner(waypoints, threshold=0.3)  # identical path every replan
    local_planner = _StuckLocalPlanner(pose=(0.0, 0.0, 0.0))
    params = _make_params(goal_x=5.0, goal_y=5.0)
    handle = _make_handle(waypoints)

    node = _FollowPath(params, handle, planner, local_planner)
    _tick(node)  # sends first waypoint; best_dist_to_goal = hypot(5,5) ~= 7.07

    # Move the robot along the diagonal toward the goal by a fixed amount
    # each stuck cycle — individually below eps, but cumulative (across
    # the _MAX_REPLAN_ATTEMPTS cycles) above eps.
    step = 0.02
    assert step < _NO_PROGRESS_EPS_M
    assert step * _MAX_REPLAN_ATTEMPTS > _NO_PROGRESS_EPS_M

    ux, uy = 1 / math.sqrt(2), 1 / math.sqrt(2)  # unit vector toward (5, 5)
    for _ in range(_MAX_REPLAN_ATTEMPTS):
        rx, ry, _ = local_planner.get_robot_pose()
        local_planner._pose = (rx + step * ux, ry + step * uy, 0.0)
        clock.advance(_WAYPOINT_TIMEOUT + 0.1)
        status = _tick(node)
        assert status == py_trees.common.Status.RUNNING  # never spuriously exhausts

    assert node._replan_attempts == 0  # cumulative > eps progress eventually resets it


# ---------------------------------------------------------------------------
# 2. No-progress guard fires before the fixed per-waypoint timeout
# ---------------------------------------------------------------------------


def test_no_progress_triggers_recovery_before_waypoint_timeout(monkeypatch) -> None:
    clock = _FakeClock(0.0)
    monkeypatch.setattr("bt_agent_robotics._hardware.skills.navigate.time.monotonic", clock)

    waypoints = [(5.0, 5.0)]
    recovered = [(1.0, 1.0)]
    planner = _RecoveringPathPlanner(waypoints, recovered, threshold=0.3)
    local_planner = _StuckLocalPlanner(pose=(0.0, 0.0, 0.0))
    params = _make_params()
    handle = _make_handle(waypoints)

    node = _FollowPath(params, handle, planner, local_planner)
    _tick(node)  # sends waypoint, resets progress tracking @ t=0

    # A tick with no distance improvement, establishing the best-distance baseline.
    status = _tick(node)
    assert status == py_trees.common.Status.RUNNING
    assert planner.replan_calls == 0

    # Advance past the no-progress window but well under the fixed timeout.
    assert _NO_PROGRESS_WINDOW_S < _WAYPOINT_TIMEOUT
    clock.advance(_NO_PROGRESS_WINDOW_S + 0.1)
    status = _tick(node)

    assert status == py_trees.common.Status.RUNNING
    assert planner.replan_calls == 1  # recovery branch fired
    assert node._waypoints == recovered
    assert clock.now < _WAYPOINT_TIMEOUT  # proves it fired before the 10s timeout


# ---------------------------------------------------------------------------
# 3. Pose freshness
# ---------------------------------------------------------------------------


def test_safeguard_invariant_fails_on_stale_pose() -> None:
    local_planner = _StuckLocalPlanner()
    local_planner._pose_age = _POSE_STALENESS_S + 0.5  # stale

    skill = NavigateSkill(local_planner=local_planner, path_planner=_FixedPathPlanner([(1.0, 1.0)]))
    handle = _make_handle()
    state = SkillContext(platform_state={"robot_position": (0.0, 0.0, 0.0)})

    # CP5: typed failure-reason string, not plain False.
    assert skill.safeguard_invariant(state, handle) == "stale_pose"


def test_safeguard_invariant_true_on_fresh_pose() -> None:
    local_planner = _StuckLocalPlanner()
    local_planner._pose_age = 0.01  # fresh

    skill = NavigateSkill(local_planner=local_planner, path_planner=_FixedPathPlanner([(1.0, 1.0)]))
    handle = _make_handle()
    state = SkillContext(platform_state={"robot_position": (0.0, 0.0, 0.0)})

    assert skill.safeguard_invariant(state, handle) is True


def test_safeguard_invariant_degrades_gracefully_without_stamp() -> None:
    """Backend without get_pose_age (duck-typed, no method at all) behaves
    exactly like before the freshness fix — no staleness gating.
    """

    class _NoFreshnessPlanner:
        def get_robot_pose(self):
            return (0.0, 0.0, 0.0)

    skill = NavigateSkill(local_planner=_NoFreshnessPlanner(), path_planner=_FixedPathPlanner([(1.0, 1.0)]))
    handle = _make_handle()
    state = SkillContext(platform_state={"robot_position": (0.0, 0.0, 0.0)})

    assert skill.safeguard_invariant(state, handle) is True


def test_check_already_at_goal_refuses_arrived_on_stale_pose() -> None:
    local_planner = _StuckLocalPlanner(pose=(5.0, 5.0, 0.0))  # exactly at goal
    local_planner._pose_age = _POSE_STALENESS_S + 1.0  # stale

    params = _make_params()
    handle = _make_handle()
    context = SkillContext(object_store=None)

    node = _CheckAlreadyAtGoal(params, handle, local_planner, context)
    assert _tick(node) == py_trees.common.Status.FAILURE  # refused, falls through


def test_check_already_at_goal_succeeds_on_fresh_pose() -> None:
    local_planner = _StuckLocalPlanner(pose=(5.0, 5.0, 0.0))
    local_planner._pose_age = 0.01

    params = _make_params()
    handle = _make_handle()
    context = SkillContext(object_store=None)

    node = _CheckAlreadyAtGoal(params, handle, local_planner, context)
    assert _tick(node) == py_trees.common.Status.SUCCESS


def test_confirm_arrival_refuses_arrived_on_stale_pose() -> None:
    """Even a forced (settle-timeout) accept must not report arrival on a
    stale pose — the safeguard invariant path is a second, independent
    line of defense against the same failure mode.
    """
    local_planner = _StuckLocalPlanner(pose=(5.0, 5.0, 0.0))  # at goal
    local_planner._pose_age = _POSE_STALENESS_S + 1.0  # stale

    params = _make_params()
    handle = _make_handle()
    node = _ConfirmArrival(params, handle, local_planner)
    node.initialise()

    status = node._accept("target", suffix="(settle timeout)", force=True)

    assert status == py_trees.common.Status.RUNNING
    assert handle.result_data.get("arrival_status") is None


# ---------------------------------------------------------------------------
# 4. arrived_short — settle-timeout force-accept beyond arrival distance
# ---------------------------------------------------------------------------


def test_confirm_arrival_force_accept_beyond_threshold_is_arrived_short() -> None:
    dist = _ARRIVAL_DISTANCE + 0.6
    local_planner = _StuckLocalPlanner(pose=(5.0 - dist, 5.0, 0.0))
    local_planner._pose_age = 0.01  # fresh

    params = _make_params(goal_x=5.0, goal_y=5.0)
    handle = _make_handle()
    node = _ConfirmArrival(params, handle, local_planner)
    node.initialise()

    status = node._accept("target", suffix="(settle timeout)", force=True)

    assert status == py_trees.common.Status.SUCCESS
    assert handle.result_data["arrival_status"] == "arrived_short"
    assert handle.result_data["residual_distance_m"] == round(dist, 3)
    assert "short" in handle.observation.lower()


def test_confirm_arrival_normal_accept_within_threshold_is_arrived() -> None:
    """Regression: force-accept that IS within threshold stays 'arrived',
    with no residual_distance_m field — arrived still implies within-threshold.
    """
    local_planner = _StuckLocalPlanner(pose=(5.0, 5.0, 0.0))  # exactly at goal
    local_planner._pose_age = 0.01

    params = _make_params(goal_x=5.0, goal_y=5.0)
    handle = _make_handle()
    node = _ConfirmArrival(params, handle, local_planner)
    node.initialise()

    status = node._accept("target")  # non-forced, settled path

    assert status == py_trees.common.Status.SUCCESS
    assert handle.result_data["arrival_status"] == "arrived"
    assert "residual_distance_m" not in handle.result_data


def test_confirm_arrival_non_forced_beyond_threshold_keeps_running() -> None:
    """Non-forced accept beyond threshold is unchanged: wait, don't accept."""
    dist = _ARRIVAL_DISTANCE + 0.5
    local_planner = _StuckLocalPlanner(pose=(5.0 - dist, 5.0, 0.0))
    local_planner._pose_age = 0.01

    params = _make_params(goal_x=5.0, goal_y=5.0)
    handle = _make_handle()
    node = _ConfirmArrival(params, handle, local_planner)
    node.initialise()

    status = node._accept("target")

    assert status == py_trees.common.Status.RUNNING


# ---------------------------------------------------------------------------
# 5. Scaled wall-time budget (bt-nav-fixes CP4)
# ---------------------------------------------------------------------------


def test_scaled_wall_time_budget_floors_short_routes() -> None:
    """Direction 1: a short route never gets less than the floor, even
    though the raw formula would compute far less wall time."""
    assert _scaled_wall_time_budget_s(1.0) == _WALL_TIME_BUDGET_FLOOR_S


def test_scaled_wall_time_budget_caps_long_routes() -> None:
    """Direction 2: a very long route never exceeds the cap, protecting
    against a single skill call monopolizing the agent loop forever."""
    assert _scaled_wall_time_budget_s(1000.0) == _WALL_TIME_BUDGET_CAP_S


def test_scaled_wall_time_budget_scales_linearly_between_floor_and_cap() -> None:
    path_length = 120.0
    expected = 2.0 * path_length / NOMINAL_WALK_SPEED_MPS
    assert _WALL_TIME_BUDGET_FLOOR_S < expected < _WALL_TIME_BUDGET_CAP_S
    assert _scaled_wall_time_budget_s(path_length) == expected


def test_plan_path_sets_wall_time_budget_from_route_length() -> None:
    """Long-but-legitimate route (CP4's motivating case): never killed by
    the generic 120s default."""
    planner = _FixedPathPlanner([(120.0, 0.0)], threshold=0.3)
    local_planner = _StuckLocalPlanner(pose=(0.0, 0.0, 0.0))
    params = _make_params(goal_x=120.0, goal_y=0.0)
    handle = _make_handle()
    context = SkillContext(platform_state={})

    node = _PlanPath(params, handle, planner, local_planner, context)
    assert _tick(node) == py_trees.common.Status.SUCCESS

    expected = _scaled_wall_time_budget_s(_path_length_m((0.0, 0.0), [(120.0, 0.0)]))
    assert handle.wall_time_budget_seconds == expected
    assert expected > 120.0  # exceeds the generic ActionSkill default


def test_plan_path_short_route_floors_wall_time_budget() -> None:
    planner = _FixedPathPlanner([(1.0, 0.0)], threshold=0.3)
    local_planner = _StuckLocalPlanner(pose=(0.0, 0.0, 0.0))
    params = _make_params(goal_x=1.0, goal_y=0.0)
    handle = _make_handle()
    context = SkillContext(platform_state={})

    node = _PlanPath(params, handle, planner, local_planner, context)
    assert _tick(node) == py_trees.common.Status.SUCCESS

    assert handle.wall_time_budget_seconds == _WALL_TIME_BUDGET_FLOOR_S


def test_plan_path_long_route_caps_wall_time_budget() -> None:
    planner = _FixedPathPlanner([(1000.0, 0.0)], threshold=0.3)
    local_planner = _StuckLocalPlanner(pose=(0.0, 0.0, 0.0))
    params = _make_params(goal_x=1000.0, goal_y=0.0)
    handle = _make_handle()
    context = SkillContext(platform_state={})

    node = _PlanPath(params, handle, planner, local_planner, context)
    assert _tick(node) == py_trees.common.Status.SUCCESS

    assert handle.wall_time_budget_seconds == _WALL_TIME_BUDGET_CAP_S


def test_follow_path_updates_wall_time_budget_on_successful_replan(monkeypatch) -> None:
    """Every successful (re)plan inside _FollowPath rescales the budget —
    not just the initial _PlanPath call."""
    clock = _FakeClock(0.0)
    monkeypatch.setattr("bt_agent_robotics._hardware.skills.navigate.time.monotonic", clock)

    waypoints = [(5.0, 5.0)]
    recovered = [(50.0, 50.0)]  # much longer remaining route than the original
    planner = _RecoveringPathPlanner(waypoints, recovered, threshold=0.3)
    local_planner = _StuckLocalPlanner(pose=(0.0, 0.0, 0.0))
    params = _make_params()
    handle = _make_handle(waypoints)
    handle.wall_time_budget_seconds = _WALL_TIME_BUDGET_FLOOR_S  # simulate the initial _PlanPath value

    node = _FollowPath(params, handle, planner, local_planner)
    _tick(node)  # send first waypoint

    clock.advance(_WAYPOINT_TIMEOUT + 0.1)
    _tick(node)  # stuck branch fires, replans to `recovered`

    expected = _scaled_wall_time_budget_s(_path_length_m((0.0, 0.0), recovered))
    assert handle.wall_time_budget_seconds == expected
    assert handle.wall_time_budget_seconds != _WALL_TIME_BUDGET_FLOOR_S


# ---------------------------------------------------------------------------
# 6. Typed safeguard_invariant failure reasons (bt-nav-fixes CP5)
# ---------------------------------------------------------------------------


def test_safeguard_invariant_returns_pose_lost_on_missing_localization() -> None:
    skill = NavigateSkill(local_planner=_StuckLocalPlanner(), path_planner=_FixedPathPlanner([(1.0, 1.0)]))
    handle = _make_handle()
    state = SkillContext(platform_state={})

    assert skill.safeguard_invariant(state, handle) == "pose_lost"


def test_safeguard_invariant_returns_path_invalidated_on_empty_planned_path() -> None:
    skill = NavigateSkill(local_planner=_StuckLocalPlanner(), path_planner=_FixedPathPlanner([(1.0, 1.0)]))
    handle = _make_handle()
    handle.result_data = {"planned_path": []}
    state = SkillContext(platform_state={"robot_position": (0.0, 0.0, 0.0)})

    assert skill.safeguard_invariant(state, handle) == "path_invalidated"


# ---------------------------------------------------------------------------
# 7. Terminal discipline — park at self + /stop brake (nav-terminal-park)
# ---------------------------------------------------------------------------
#
# Hardware defect (2026-08-08): the chain behind the skill (goal_adapter ->
# localPlanner -> pathFollower -> cmd_vel_adapter -> cmd_vel_mux) kept
# commanding after NavigateSkill reported terminal, because goal_adapter
# latches the last waypoint. Intent encoded below: once the skill leaves
# RUNNING — by ANY route — the latched goal is the robot's own pose and the
# operator brake is engaged, so nothing downstream can command motion.


class _NoPathPlanner(_FixedPathPlanner):
    """plan() finds nothing — drives _PlanPath's failed_no_path terminal path."""

    def plan(self, start_xy, goal_xy) -> Optional[List[Tuple[float, float]]]:
        return


def _navigate_tree(
    local_planner: Any,
    path_planner: PathPlanner,
    handle: SkillHandle,
    params: Optional[Dict[str, Any]] = None,
    context: Optional[SkillContext] = None,
) -> py_trees.behaviour.Behaviour:
    skill = NavigateSkill(local_planner=local_planner, path_planner=path_planner)
    return skill.build_subtree(
        params if params is not None else _make_params(),
        context if context is not None else SkillContext(platform_state={}),
        handle,
    )


def test_navigate_success_parks_at_self_then_engages_brake() -> None:
    """Terminal SUCCESS (already-at-goal shortcut): the last goal published
    is the robot's own pose, and the brake is engaged after it."""
    local_planner = _StuckLocalPlanner(pose=(5.0, 5.0, 0.25))
    local_planner._pose_age = 0.01  # fresh
    handle = _make_handle()

    tree = _navigate_tree(local_planner, _FixedPathPlanner([(5.0, 5.0)]), handle)
    assert _tick(tree) == py_trees.common.Status.SUCCESS

    assert local_planner.sent_waypoints[-1] == (5.0, 5.0, 0.25)  # parked at self
    assert local_planner.brake_calls == [False, True]
    assert local_planner.events[-2:] == ["waypoint", "brake_on"]  # park BEFORE brake
    # Outcome contract untouched by the discipline.
    assert handle.result_data["arrival_status"] == "already_at_goal"


def test_navigate_releases_brake_before_first_goal() -> None:
    """A brake left engaged by the previous skill would silently swallow this
    skill's motion — release must precede the first goal publication."""
    local_planner = _StuckLocalPlanner(pose=(0.0, 0.0, 0.0))
    handle = _make_handle()

    tree = _navigate_tree(local_planner, _FixedPathPlanner([(5.0, 5.0)]), handle)
    assert _tick(tree) == py_trees.common.Status.RUNNING  # body still driving

    assert local_planner.events[0] == "brake_off"
    assert "waypoint" in local_planner.events
    assert local_planner.events.index("brake_off") < local_planner.events.index("waypoint")


def test_navigate_failure_no_path_parks_at_self_and_engages_brake() -> None:
    """Typed failure (failed_no_path) still parks + brakes, and the brake
    does not mask the failure code."""
    local_planner = _StuckLocalPlanner(pose=(1.0, 2.0, 0.5))
    handle = _make_handle()

    tree = _navigate_tree(local_planner, _NoPathPlanner([(5.0, 5.0)]), handle)
    assert _tick(tree) == py_trees.common.Status.FAILURE

    assert local_planner.sent_waypoints[-1] == (1.0, 2.0, 0.5)
    assert local_planner.brake_calls[-1] is True
    assert handle.result_data["arrival_status"] == "failed_no_path"


def test_navigate_cancellation_parks_at_self_and_engages_brake() -> None:
    """Cancellation is a terminal path too — the robot must not keep driving
    toward the last waypoint after the operator cancels."""
    local_planner = _StuckLocalPlanner(pose=(1.0, 2.0, 0.5))
    handle = _make_handle()
    handle.cancel()

    tree = _navigate_tree(local_planner, _FixedPathPlanner([(5.0, 5.0)]), handle)
    assert _tick(tree) == py_trees.common.Status.FAILURE

    assert local_planner.sent_waypoints[-1] == (1.0, 2.0, 0.5)
    assert local_planner.brake_calls[-1] is True


def test_navigate_mid_run_invalidation_parks_at_self_and_engages_brake() -> None:
    """The tree being invalidated mid-run (what _tick_subtree_loop does on
    cancellation, and what the memory=False safeguard Sequence does to the
    body when a guard fails) is the dangerous case: the robot IS moving."""
    local_planner = _StuckLocalPlanner(pose=(1.0, 2.0, 0.5))
    handle = _make_handle()

    tree = _navigate_tree(local_planner, _FixedPathPlanner([(5.0, 5.0)]), handle)
    assert _tick(tree) == py_trees.common.Status.RUNNING

    tree.stop(py_trees.common.Status.INVALID)

    assert local_planner.sent_waypoints[-1] == (1.0, 2.0, 0.5)
    assert local_planner.brake_calls[-1] is True


def test_safeguard_abort_on_stale_pose_brakes_without_inventing_a_park_goal() -> None:
    """Both layers, exercised through the real safeguard wrap: an invariant
    abort brakes unconditionally, but a stale pose must NOT be republished as
    a goal — a fabricated park pose could pull the robot somewhere wrong.
    """
    from bt_agent._engine.skills.safeguard import _wrap_safeguarding

    local_planner = _StuckLocalPlanner(pose=(1.0, 2.0, 0.5))
    local_planner._pose_age = 0.01  # fresh to start
    handle = _make_handle()
    skill = NavigateSkill(local_planner=local_planner, path_planner=_FixedPathPlanner([(5.0, 5.0)]))
    context = SkillContext(platform_state={"robot_position": (1.0, 2.0, 0.5)})
    body = skill.build_subtree(_make_params(), context, handle)
    tree = _wrap_safeguarding(skill, body, handle, lambda: context)

    assert _tick(tree) == py_trees.common.Status.RUNNING
    waypoints_before = len(local_planner.sent_waypoints)

    local_planner._pose_age = _POSE_STALENESS_S + 1.0  # localization goes stale
    assert _tick(tree) == py_trees.common.Status.FAILURE

    assert len(local_planner.sent_waypoints) == waypoints_before  # no invented park goal
    assert local_planner.brake_calls[-1] is True
    assert handle.result_data["failure_reason"] == "stale_pose"


def test_park_at_self_refuses_stale_pose() -> None:
    local_planner = _StuckLocalPlanner(pose=(1.0, 2.0, 0.5))
    local_planner._pose_age = _POSE_STALENESS_S + 1.0

    assert _park_at_self(local_planner) is False
    assert local_planner.sent_waypoints == []


def test_park_at_self_refuses_non_finite_pose() -> None:
    """A NaN/inf odom read is a lost pose in disguise — never publish it."""
    local_planner = _StuckLocalPlanner(pose=(float("nan"), 2.0, 0.5))
    local_planner._pose_age = 0.01

    assert _park_at_self(local_planner) is False
    assert local_planner.sent_waypoints == []


def test_terminal_discipline_degrades_on_backend_without_brake() -> None:
    """Backends that don't wire /stop (mocks, older bridges) must still park
    and must not raise — brake is an enhancement, park is the floor."""

    class _NoBrakePlanner:
        def __init__(self) -> None:
            self.sent_waypoints: List[Tuple[float, float, float]] = []

        def send_waypoint(self, x: float, y: float, theta: float) -> None:
            self.sent_waypoints.append((x, y, theta))

        def get_robot_pose(self) -> Tuple[float, float, float]:
            return (3.0, 4.0, 1.0)

    local_planner = _NoBrakePlanner()
    child = py_trees.behaviours.Failure(name="body")
    node = _TerminalDiscipline(child=child, local_planner=local_planner)

    assert _tick(node) == py_trees.common.Status.FAILURE
    assert local_planner.sent_waypoints == [(3.0, 4.0, 1.0)]


def test_terminal_discipline_engages_brake_even_when_park_raises() -> None:
    """A terminal path must never leave the brake released: if parking (or the
    pose read feeding it) raises, the /stop brake still engages."""

    class _RaisingWaypointPlanner(_StuckLocalPlanner):
        def send_waypoint(self, x: float, y: float, theta: float) -> None:
            raise RuntimeError("publisher dead")

    local_planner = _RaisingWaypointPlanner(pose=(1.0, 2.0, 0.5))
    inner = py_trees.behaviours.Success(name="body")
    guard = _TerminalDiscipline(child=inner, local_planner=local_planner)

    guard.terminate(py_trees.common.Status.FAILURE)

    assert local_planner.brake_calls[-1] is True


def test_terminal_discipline_engages_brake_on_non_numeric_pose() -> None:
    """A garbage pose (None fields) must not escape the park path unbraked."""

    class _GarbagePosePlanner(_StuckLocalPlanner):
        def get_robot_pose(self):
            return (None, None, None)

    local_planner = _GarbagePosePlanner()
    local_planner._pose_age = 0.01
    inner = py_trees.behaviours.Success(name="body")
    guard = _TerminalDiscipline(child=inner, local_planner=local_planner)

    guard.terminate(py_trees.common.Status.FAILURE)

    assert local_planner.brake_calls[-1] is True


def test_terminal_discipline_first_tick_crash_still_brakes() -> None:
    """A leaf crashing on its FIRST tick never gets a py_trees status, so the
    stop(INVALID) cascade skips the decorator (INVALID->INVALID) — yet
    initialise() already released the brake. The decorator's own tick guard
    must park+brake before re-raising."""
    import pytest

    class _CrashLeaf(py_trees.behaviour.Behaviour):
        def update(self) -> py_trees.common.Status:
            raise RuntimeError("boom")

    local_planner = _StuckLocalPlanner(pose=(1.0, 2.0, 0.5))
    local_planner._pose_age = 0.01
    guard = _TerminalDiscipline(child=_CrashLeaf(name="crash"), local_planner=local_planner)

    with pytest.raises(RuntimeError):
        for _ in guard.tick():
            pass

    assert local_planner.brake_calls[0] is False  # initialise released it
    assert local_planner.brake_calls[-1] is True  # crash guard re-engaged it
    assert local_planner.sent_waypoints[-1] == (1.0, 2.0, 0.5)  # parked at self
