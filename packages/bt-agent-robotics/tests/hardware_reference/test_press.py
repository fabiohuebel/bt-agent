"""Tests for the B1 press-skill hardening (bt-nav-fixes CP6).

Applies the same pattern proportionately to press's actual structure:

- Pose-freshness refusal wherever press acts on pose (``safeguard_invariant``,
  ``_DriveOntoTarget._accept``) — mirrors NavigateSkill.
- Typed failure codes: ``safeguard_invariant`` returns a reason string
  (``pose_lost`` / ``stale_pose`` / ``path_invalidated``) instead of plain
  ``False``; the shared ``_FollowPath`` leaf's ``arrival_status`` failure is
  translated into press's own ``press_status`` key via ``_build_result``
  (feat/press-typed-bt) — the old mirror-both-keys leak is removed; the
  shared navigate.py leaf itself stays untouched.
- A wall-time budget scaled the same way as NavigateSkill (CP4) — press's
  nav-graph approach phase has the same distance dimension.

move_relative is explicitly out of scope (Fabio's ruling) and untouched.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import py_trees
from bt_agent.skills import SkillContext, SkillHandle, SkillStatus
from bt_agent_robotics._hardware.skills.navigate import (
    _WALL_TIME_BUDGET_CAP_S,
    _WALL_TIME_BUDGET_FLOOR_S,
    _path_length_m,
    _scaled_wall_time_budget_s,
)
from bt_agent_robotics._hardware.skills.press import (
    _POSE_STALENESS_S,
    PressSkill,
    _ConfirmPressPath,
    _DriveOntoTarget,
    _PlanPath,
    _PressFollowPath,
    _ValidateTarget,
)
from bt_agent_robotics.skills.utils.path_planner import PathPlanner

# ---------------------------------------------------------------------------
# Test doubles (mirror test_navigate.py conventions)
# ---------------------------------------------------------------------------


class _StubLocalPlanner:
    """Duck-typed ``LocalPlannerInterface`` double with controllable pose age."""

    def __init__(self, pose: Tuple[float, float, float] = (0.0, 0.0, 0.0)) -> None:
        self._pose = pose
        self._velocity: Tuple[float, float] = (0.0, 0.0)
        self.sent_waypoints: List[Tuple[float, float, float]] = []
        self.cancelled = False
        self._pose_age: Optional[float] = None  # None => no freshness tracked
        # Terminal-discipline observability (mirrors test_navigate.py).
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
        return self._velocity

    def cancel(self) -> None:
        self.cancelled = True

    def get_pose_age(self) -> Optional[float]:
        return self._pose_age

    def set_brake(self, engaged: bool) -> None:
        self.brake_calls.append(engaged)
        self.events.append("brake_on" if engaged else "brake_off")


class _FixedPathPlanner(PathPlanner):
    """Always returns the same waypoint sequence."""

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


class _NoPathPlanner(_FixedPathPlanner):
    def plan(self, start_xy, goal_xy) -> Optional[List[Tuple[float, float]]]:
        return None


class _NoReplanPlanner(_FixedPathPlanner):
    """Plans once, then never finds a path again — drives the follow-path
    leaf's no_path exhaustion branch rather than its blocked branch."""

    def replan(self, robot_xy, goal_xy) -> Optional[List[Tuple[float, float]]]:
        self.replan_calls += 1
        return


def _make_params(goal_x: float = 5.0, goal_y: float = 5.0, goal_theta: float = 0.0) -> Dict[str, Any]:
    return {
        "goal_x": goal_x,
        "goal_y": goal_y,
        "goal_theta": goal_theta,
        "target_handle": "obj_1",
        "target_label": "target",
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


class _ApprovalGate:
    enabled = True

    def __init__(self):
        self.plans = []
        self.decision = None
        self.clears = 0

    def publish_plan(self, waypoints, label, robot_xy, object_id=None):
        self.plans.append((list(waypoints), label, robot_xy))
        self.decision = None

    def poll(self):
        decision, self.decision = self.decision, None
        return decision  # noqa: R504 - saved value before clearing the input

    def clear(self):
        self.clears += 1


def test_press_requires_approval_for_both_motion_phases(monkeypatch):
    monkeypatch.setenv("LAM_NAV_CONFIRM", "1")
    planner = _StubLocalPlanner()
    gate = _ApprovalGate()
    handle = _make_handle()
    skill = PressSkill(planner, _FixedPathPlanner([(0.0, 0.0)]), gate)
    tree = skill.build_subtree(_make_params(), SkillContext(), handle)
    assert _tick(tree) == py_trees.common.Status.RUNNING
    assert len(gate.plans) == 1
    assert planner.sent_waypoints == []
    assert False not in planner.brake_calls
    gate.decision = "go"
    assert _tick(tree) == py_trees.common.Status.RUNNING
    assert planner.brake_calls[-1] is False
    assert _tick(tree) == py_trees.common.Status.RUNNING
    assert len(gate.plans) == 2
    assert gate.plans[-1][0] == [(5.0, 5.0)]
    assert planner.brake_calls[-1] is True
    assert all(w[:2] != (5.0, 5.0) for w in planner.sent_waypoints)
    gate.decision = "go"
    assert _tick(tree) == py_trees.common.Status.RUNNING
    assert planner.sent_waypoints[-1][:2] == (5.0, 5.0)
    assert planner.brake_calls[-1] is False
    tree.stop(py_trees.common.Status.INVALID)
    assert planner.brake_calls[-1] is True


def test_press_abort_keeps_motion_braked_and_reports_press_status(monkeypatch):
    monkeypatch.setenv("LAM_NAV_CONFIRM", "1")
    planner, gate, handle = _StubLocalPlanner(), _ApprovalGate(), _make_handle()
    tree = PressSkill(planner, _FixedPathPlanner([(1.0, 0.0)]), gate).build_subtree(
        _make_params(),
        SkillContext(),
        handle,
    )
    _tick(tree)
    gate.decision = "abort"
    assert _tick(tree) == py_trees.common.Status.FAILURE
    assert handle.result_data["press_status"] == "failed_operator_abort"
    assert "arrival_status" not in handle.result_data
    assert False not in planner.brake_calls
    assert gate.clears == 1


def test_press_cancel_wins_over_queued_go():
    planner, gate, handle = _StubLocalPlanner(), _ApprovalGate(), _make_handle([(1.0, 0.0)])
    leaf = _ConfirmPressPath(_make_params(), handle, gate, planner)
    _tick(leaf)
    gate.decision = "go"
    handle.cancel()
    assert _tick(leaf) == py_trees.common.Status.FAILURE
    assert planner.sent_waypoints == []
    assert False not in planner.brake_calls


def test_press_confirmation_timeout_never_releases_brake():
    planner, gate, handle = _StubLocalPlanner(), _ApprovalGate(), _make_handle([(1.0, 0.0)])
    leaf = _ConfirmPressPath(_make_params(), handle, gate, planner)
    _tick(leaf)
    leaf._timeout_s = -1
    assert _tick(leaf) == py_trees.common.Status.FAILURE
    assert handle.result_data["press_status"] == "failed_operator_abort"
    assert False not in planner.brake_calls


def test_press_precision_is_scoped_to_final_leaf_and_cleared_on_cancel():
    planner, handle = _StubLocalPlanner(), _make_handle()
    requests = []
    planner.set_press_precision = lambda enabled: requests.append(enabled) or True  # noqa: SIM222 - records the call
    leaf = _DriveOntoTarget(_make_params(), handle, planner)
    assert _tick(leaf) == py_trees.common.Status.RUNNING
    assert _tick(leaf) == py_trees.common.Status.RUNNING
    assert requests == [True, True]
    leaf.stop(py_trees.common.Status.INVALID)
    assert requests[-1] is False
    assert planner.cancelled


def test_press_missing_precision_controller_refuses_goal():
    planner, handle = _StubLocalPlanner(), _make_handle()
    planner.set_press_precision = lambda enabled: not enabled
    leaf = _DriveOntoTarget(_make_params(), handle, planner)
    assert _tick(leaf) == py_trees.common.Status.FAILURE
    assert planner.sent_waypoints == []
    assert handle.result_data["press_status"] == "failed_no_movement"


def test_press_success_clears_precision_without_changing_acceptance_distance():
    planner, handle = _StubLocalPlanner(pose=(4.9, 5.0, 0.0)), _make_handle()
    requests = []
    planner.set_press_precision = lambda enabled: requests.append(enabled) or True  # noqa: SIM222 - records the call
    leaf = _DriveOntoTarget(_make_params(), handle, planner)
    assert _tick(leaf) == py_trees.common.Status.SUCCESS
    assert requests == [True, False]


# ---------------------------------------------------------------------------
# 1. Pose-freshness refusal
# ---------------------------------------------------------------------------


def test_press_safeguard_invariant_returns_pose_lost_on_missing_localization() -> None:
    skill = PressSkill(local_planner=_StubLocalPlanner(), path_planner=_FixedPathPlanner([(1.0, 1.0)]))
    handle = _make_handle()
    state = SkillContext(platform_state={})

    assert skill.safeguard_invariant(state, handle) == "pose_lost"


def test_press_safeguard_invariant_returns_stale_pose() -> None:
    local_planner = _StubLocalPlanner()
    local_planner._pose_age = _POSE_STALENESS_S + 0.5  # stale

    skill = PressSkill(local_planner=local_planner, path_planner=_FixedPathPlanner([(1.0, 1.0)]))
    handle = _make_handle()
    state = SkillContext(platform_state={"robot_position": (0.0, 0.0, 0.0)})

    assert skill.safeguard_invariant(state, handle) == "stale_pose"


def test_press_safeguard_invariant_returns_path_invalidated_on_empty_planned_path() -> None:
    skill = PressSkill(local_planner=_StubLocalPlanner(), path_planner=_FixedPathPlanner([(1.0, 1.0)]))
    handle = _make_handle()
    handle.result_data = {"planned_path": []}
    state = SkillContext(platform_state={"robot_position": (0.0, 0.0, 0.0)})

    assert skill.safeguard_invariant(state, handle) == "path_invalidated"


def test_press_safeguard_invariant_true_when_healthy() -> None:
    local_planner = _StubLocalPlanner()
    local_planner._pose_age = 0.01  # fresh

    skill = PressSkill(local_planner=local_planner, path_planner=_FixedPathPlanner([(1.0, 1.0)]))
    handle = _make_handle()
    state = SkillContext(platform_state={"robot_position": (0.0, 0.0, 0.0)})

    assert skill.safeguard_invariant(state, handle) is True


def test_press_safeguard_invariant_degrades_gracefully_without_stamp() -> None:
    """Backend without get_pose_age behaves exactly like before the fix."""

    class _NoFreshnessPlanner:
        def get_robot_pose(self):
            return (0.0, 0.0, 0.0)

    skill = PressSkill(local_planner=_NoFreshnessPlanner(), path_planner=_FixedPathPlanner([(1.0, 1.0)]))
    handle = _make_handle()
    state = SkillContext(platform_state={"robot_position": (0.0, 0.0, 0.0)})

    assert skill.safeguard_invariant(state, handle) is True


def test_drive_onto_target_refuses_accept_on_stale_pose() -> None:
    local_planner = _StubLocalPlanner(pose=(5.0, 5.0, 0.0))  # exactly at goal
    local_planner._pose_age = _POSE_STALENESS_S + 1.0  # stale

    params = _make_params(goal_x=5.0, goal_y=5.0)
    handle = _make_handle()
    node = _DriveOntoTarget(params, handle, local_planner)
    node.initialise()

    status = node._accept("target", 0.0)

    assert status == py_trees.common.Status.RUNNING
    assert handle.result_data.get("press_status") is None


def test_drive_onto_target_accepts_normally_on_fresh_pose() -> None:
    local_planner = _StubLocalPlanner(pose=(5.0, 5.0, 0.0))
    local_planner._pose_age = 0.01  # fresh

    params = _make_params(goal_x=5.0, goal_y=5.0)
    handle = _make_handle()
    node = _DriveOntoTarget(params, handle, local_planner)
    node.initialise()

    status = node._accept("target", 0.0)

    assert status == py_trees.common.Status.SUCCESS
    assert handle.result_data["press_status"] == "pressed"


# ---------------------------------------------------------------------------
# 2. Typed failure codes: shared _FollowPath leaf translated to press_status
# ---------------------------------------------------------------------------


class _FakeClock:
    def __init__(self, start: float = 0.0) -> None:
        self.now = start

    def __call__(self) -> float:
        return self.now

    def advance(self, dt: float) -> None:
        self.now += dt


def _run_follow_path_to_exhaustion(monkeypatch, planner: _FixedPathPlanner, handle: SkillHandle):
    """Tick a _PressFollowPath until its replan budget is spent (FAILURE)."""
    from bt_agent_robotics._hardware.skills import navigate as navigate_mod

    clock = _FakeClock(0.0)
    monkeypatch.setattr(navigate_mod.time, "monotonic", clock)

    local_planner = _StubLocalPlanner(pose=(0.0, 0.0, 0.0))
    node = _PressFollowPath(_make_params(), handle, planner, local_planner)
    _tick(node)  # sends first waypoint

    status = py_trees.common.Status.RUNNING
    for _ in range(navigate_mod._MAX_REPLAN_ATTEMPTS):
        clock.advance(navigate_mod._WAYPOINT_TIMEOUT + 0.1)
        status = _tick(node)
    return status


def test_press_follow_path_translates_shared_leaf_failure_into_press_status(monkeypatch) -> None:
    """The shared navigate._FollowPath leaf writes arrival_status on failure
    (navigate's contract); press's canonical outcome key is press_status (per
    press.md / human_baseline.py / skills_contract_v0)."""
    waypoints = [(5.0, 5.0)]
    handle = _make_handle(waypoints)

    status = _run_follow_path_to_exhaustion(monkeypatch, _FixedPathPlanner(waypoints, threshold=0.3), handle)

    assert status == py_trees.common.Status.FAILURE
    assert handle.result_data["press_status"] == "failed_blocked"


def test_press_follow_path_failure_does_not_leak_arrival_status(monkeypatch) -> None:
    """Leak removal: press must not publish navigate's arrival_status key.
    Both routers map shared codes to byte-identical sentences, so leaving the
    leak in place would append the same guidance twice once press_status is
    routed."""
    waypoints = [(5.0, 5.0)]
    handle = _make_handle(waypoints)

    _run_follow_path_to_exhaustion(monkeypatch, _FixedPathPlanner(waypoints, threshold=0.3), handle)

    assert "arrival_status" not in handle.result_data


def test_press_follow_path_failure_uses_press_result_shape(monkeypatch) -> None:
    """The translated result carries press's result_data shape (press.md), not
    navigate's — no viewpoint_index/total_viewpoints/is_exploration."""
    waypoints = [(5.0, 5.0)]
    handle = _make_handle(waypoints)

    _run_follow_path_to_exhaustion(monkeypatch, _FixedPathPlanner(waypoints, threshold=0.3), handle)

    data = handle.result_data
    assert data["target_handle"] == "obj_1"
    assert data["target_label"] == "target"
    assert data["goal"] == {"x": 5.0, "y": 5.0}
    assert "final_distance" in data
    for navigate_only in ("viewpoint_index", "total_viewpoints", "is_exploration"):
        assert navigate_only not in data


def test_press_follow_path_no_path_exhaustion_reports_failed_no_path(monkeypatch) -> None:
    waypoints = [(5.0, 5.0)]
    handle = _make_handle(waypoints)
    planner = _NoReplanPlanner(waypoints, threshold=0.3)

    status = _run_follow_path_to_exhaustion(monkeypatch, planner, handle)

    assert status == py_trees.common.Status.FAILURE
    assert handle.result_data["press_status"] == "failed_no_path"
    assert "arrival_status" not in handle.result_data


def test_press_follow_path_success_does_not_touch_press_status(monkeypatch) -> None:
    waypoints = [(0.1, 0.1)]
    planner = _FixedPathPlanner(waypoints, threshold=1.0)  # threshold big enough for immediate arrival
    local_planner = _StubLocalPlanner(pose=(0.05, 0.05, 0.0))
    params = _make_params()
    handle = _make_handle(waypoints)

    node = _PressFollowPath(params, handle, planner, local_planner)
    _tick(node)  # sends first waypoint
    status = _tick(node)  # within threshold, advances past last waypoint

    assert status == py_trees.common.Status.SUCCESS
    assert "press_status" not in handle.result_data


# ---------------------------------------------------------------------------
# 3. Scaled wall-time budget (bt-nav-fixes CP4, reused for press)
# ---------------------------------------------------------------------------


def test_press_plan_path_sets_wall_time_budget_from_route_length() -> None:
    planner = _FixedPathPlanner([(120.0, 0.0)], threshold=0.3)
    local_planner = _StubLocalPlanner(pose=(0.0, 0.0, 0.0))
    params = _make_params(goal_x=120.0, goal_y=0.0)
    handle = _make_handle()
    context = SkillContext(platform_state={})

    node = _PlanPath(params, handle, planner, local_planner, context)
    assert _tick(node) == py_trees.common.Status.SUCCESS

    expected = _scaled_wall_time_budget_s(_path_length_m((0.0, 0.0), [(120.0, 0.0)]))
    assert handle.wall_time_budget_seconds == expected
    assert expected > 120.0  # exceeds the generic ActionSkill default


def test_press_plan_path_short_route_floors_wall_time_budget() -> None:
    planner = _FixedPathPlanner([(1.0, 0.0)], threshold=0.3)
    local_planner = _StubLocalPlanner(pose=(0.0, 0.0, 0.0))
    params = _make_params(goal_x=1.0, goal_y=0.0)
    handle = _make_handle()
    context = SkillContext(platform_state={})

    node = _PlanPath(params, handle, planner, local_planner, context)
    assert _tick(node) == py_trees.common.Status.SUCCESS

    assert handle.wall_time_budget_seconds == _WALL_TIME_BUDGET_FLOOR_S


def test_press_plan_path_long_route_caps_wall_time_budget() -> None:
    planner = _FixedPathPlanner([(1000.0, 0.0)], threshold=0.3)
    local_planner = _StubLocalPlanner(pose=(0.0, 0.0, 0.0))
    params = _make_params(goal_x=1000.0, goal_y=0.0)
    handle = _make_handle()
    context = SkillContext(platform_state={})

    node = _PlanPath(params, handle, planner, local_planner, context)
    assert _tick(node) == py_trees.common.Status.SUCCESS

    assert handle.wall_time_budget_seconds == _WALL_TIME_BUDGET_CAP_S


def test_press_plan_path_failure_unaffected_no_budget_set() -> None:
    """Regression: no-path failure still reports failed_no_path and never
    sets a wall-time budget override."""
    planner = _NoPathPlanner([])
    local_planner = _StubLocalPlanner(pose=(0.0, 0.0, 0.0))
    params = _make_params()
    handle = _make_handle()
    context = SkillContext(platform_state={})

    node = _PlanPath(params, handle, planner, local_planner, context)
    assert _tick(node) == py_trees.common.Status.FAILURE
    assert handle.result_data["press_status"] == "failed_no_path"
    assert handle.wall_time_budget_seconds is None


# ---------------------------------------------------------------------------
# 4. Terminal discipline — park at self + /stop brake (nav-terminal-park)
# ---------------------------------------------------------------------------
#
# Press drives the same goal_adapter -> localPlanner -> pathFollower chain
# (it reuses _FollowPath and sends its final approach through the same
# waypoint interface), so it inherits the same defect: the chain keeps
# commanding after the skill reports terminal.


def _press_tree(
    local_planner: Any,
    path_planner: PathPlanner,
    handle: SkillHandle,
    params: Optional[Dict[str, Any]] = None,
) -> py_trees.behaviour.Behaviour:
    skill = PressSkill(local_planner=local_planner, path_planner=path_planner)
    return skill.build_subtree(
        params if params is not None else _make_params(),
        SkillContext(platform_state={}),
        handle,
    )


def test_press_success_parks_at_self_then_engages_brake() -> None:
    local_planner = _StubLocalPlanner(pose=(5.0, 5.0, 0.0))  # already on the target
    local_planner._pose_age = 0.01  # fresh
    handle = _make_handle()

    tree = _press_tree(local_planner, _FixedPathPlanner([(5.0, 5.0)], threshold=1.0), handle)
    status = py_trees.common.Status.RUNNING
    for _ in range(4):
        status = _tick(tree)
        if status != py_trees.common.Status.RUNNING:
            break

    assert status == py_trees.common.Status.SUCCESS
    assert handle.result_data["press_status"] == "pressed"
    assert local_planner.sent_waypoints[-1] == (5.0, 5.0, 0.0)  # parked at self
    assert local_planner.brake_calls == [False, True]
    assert local_planner.events[-2:] == ["waypoint", "brake_on"]


def test_press_releases_brake_before_first_goal() -> None:
    local_planner = _StubLocalPlanner(pose=(0.0, 0.0, 0.0))
    handle = _make_handle()

    tree = _press_tree(local_planner, _FixedPathPlanner([(5.0, 5.0)]), handle)
    assert _tick(tree) == py_trees.common.Status.RUNNING

    assert local_planner.events[0] == "brake_off"
    assert local_planner.events.index("brake_off") < local_planner.events.index("waypoint")


def test_press_failure_no_path_parks_at_self_and_engages_brake() -> None:
    local_planner = _StubLocalPlanner(pose=(1.0, 2.0, 0.5))
    handle = _make_handle()

    tree = _press_tree(local_planner, _NoPathPlanner([]), handle)
    assert _tick(tree) == py_trees.common.Status.FAILURE

    assert local_planner.sent_waypoints[-1] == (1.0, 2.0, 0.5)
    assert local_planner.brake_calls[-1] is True
    assert handle.result_data["press_status"] == "failed_no_path"


def test_press_mid_run_invalidation_parks_at_self_and_engages_brake() -> None:
    local_planner = _StubLocalPlanner(pose=(1.0, 2.0, 0.5))
    handle = _make_handle()

    tree = _press_tree(local_planner, _FixedPathPlanner([(5.0, 5.0)]), handle)
    assert _tick(tree) == py_trees.common.Status.RUNNING

    tree.stop(py_trees.common.Status.INVALID)

    assert local_planner.sent_waypoints[-1] == (1.0, 2.0, 0.5)
    assert local_planner.brake_calls[-1] is True


# ---------------------------------------------------------------------------
# 5. press_status written at EVERY outcome, and every code routed
# ---------------------------------------------------------------------------


def test_validate_target_writes_failed_invalid() -> None:
    handle = _make_handle()
    params = _make_params(goal_x=float("nan"))

    node = _ValidateTarget(params, handle)

    assert _tick(node) == py_trees.common.Status.FAILURE
    assert handle.result_data["press_status"] == "failed_invalid"


def test_drive_onto_target_timeout_without_movement_writes_failed_no_movement(monkeypatch) -> None:
    from bt_agent_robotics._hardware.skills import press as press_mod

    clock = _FakeClock(0.0)
    monkeypatch.setattr(press_mod.time, "monotonic", clock)

    local_planner = _StubLocalPlanner(pose=(0.0, 0.0, 0.0))  # far from goal, never moves
    local_planner._pose_age = 0.01
    params = _make_params(goal_x=5.0, goal_y=5.0)
    handle = _make_handle()

    node = _DriveOntoTarget(params, handle, local_planner)
    assert _tick(node) == py_trees.common.Status.RUNNING

    clock.advance(press_mod._PRESS_TIMEOUT + 0.1)

    assert _tick(node) == py_trees.common.Status.FAILURE
    assert handle.result_data["press_status"] == "failed_no_movement"


def test_every_press_status_failure_code_has_routed_guidance() -> None:
    """Contract check across packages: every failure code press can emit is
    resolvable by the pycore router, so no press failure reaches the agent as
    a bare code with no recovery hint (skills_contract_v0 D8)."""
    from bt_agent._engine.agent.lam_agent.actor import route_press_status_guidance

    for code in ("failed_no_path", "failed_blocked", "failed_no_movement", "failed_invalid"):
        assert route_press_status_guidance(code), f"unrouted press_status code: {code!r}"
    assert route_press_status_guidance("pressed") == ""
