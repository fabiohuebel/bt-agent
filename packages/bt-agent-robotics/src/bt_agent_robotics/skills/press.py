"""PressSkill — step on or press down on a floor object.

Drives the robot onto a floor-level target (panel, button, platform).
Reuses the nav-graph path planning from NavigateSkill to get near the
target, then does a precise final approach to drive directly onto it.

Body subtree (framework auto-wraps with safeguards):

    Sequence("press", memory=True)
    ├── _ValidateTarget     — resolve handle, extract object position
    ├── _PlanPath           — Dijkstra → waypoint list to nearest node
    ├── _FollowPath         — waypoint sequencing (from navigate.py)
    └── _DriveOntoTarget    — precise approach directly onto object

Agent-facing docs: ``bt_skills/docs/press.md``.
"""

from __future__ import annotations

import logging
import math
import time
from typing import Any, Dict, Optional, Tuple, Union

import py_trees
from bt_agent.skills import ActionSkill, SkillContext, SkillHandle
from bt_agent_robotics.skills.navigate import (
    _POSE_STALENESS_S,
    LocalPlannerInterface,
    _FollowPath,
    _nav_debug_log,
    _path_length_m,
    _pose_age,
    _pose_is_stale,
    _scaled_wall_time_budget_s,
    _TerminalDiscipline,
)
from bt_agent_robotics.skills.utils.path_planner import PathPlanner

logger = logging.getLogger(__name__)

# Press-specific constants
_PRESS_DISTANCE = 0.2  # meters — on top of object
_PRESS_BLOCKED_DIST = 0.5  # meters — close enough if physically blocked
_PRESS_TIMEOUT = 30.0  # seconds for the final approach
_MOVEMENT_THRESHOLD = 0.05  # m/s — above this counts as "saw movement"
_SETTLE_DURATION = 1.5  # seconds robot must be stopped to detect "blocked"
_RETURN_DWELL_S = 0.5  # seconds on the trigger before stepping back
_RETURN_MIN_DIST = 0.3  # meters — already at the approach pose, nothing to do
_RETURN_ACCEPT_DIST = 0.35  # meters — back at the approach pose
_RETURN_TIMEOUT_S = 12.0  # seconds — give up stepping back (press still succeeded)


class PressSkill(ActionSkill):
    """Step on or press down on a floor object (panel, button, platform).

    Resolves the target handle from the ObjectStore, extracts the object's
    position (not viewpoint), plans a path to get near it, then drives
    precisely onto it.
    """

    name = "press"
    description = "Step on or press down on a floor object (panel, button)."
    usage = "Use after NAVIGATE and SURROUND_SURVEY. The robot drives directly onto the target."
    params = {
        "target_object": "Handle from QUERY (e.g. 'blue_panel_1').",
    }

    def __init__(
        self,
        local_planner: LocalPlannerInterface,
        path_planner: PathPlanner,
    ) -> None:
        self._local_planner = local_planner
        self._path_planner = path_planner

    # -- Gating ---------------------------------------------------------------

    def precondition(self, state: SkillContext) -> Tuple[bool, str]:
        if state.object_store is None:
            return False, "ObjectStore not available."
        if not state.object_store.all_objects():
            return False, "No object handles in ObjectStore (call QUERY first)."
        return True, ""

    def is_available(
        self,
        state: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Press requires NAVIGATE + SURROUND_SURVEY in the current milestone."""
        calls = state.get("tool_calls", [])
        # Find last navigate call
        last_nav_idx = -1
        for i, tc in enumerate(calls):
            if tc.name == "navigate":
                last_nav_idx = i
        if last_nav_idx < 0:
            return False, "NAVIGATE to the target first."
        # Require surround_survey after the last navigate
        has_survey_after = any(tc.name == "surround_survey" for tc in calls[last_nav_idx + 1 :])
        if not has_survey_after:
            return False, "SURROUND_SURVEY after NAVIGATE to confirm arrival."
        return True, f"{self.name} available"

    # -- Safeguard ------------------------------------------------------------

    def safeguard_invariant(self, state: SkillContext, handle: SkillHandle) -> Union[bool, str]:
        """Same pattern as NavigateSkill.safeguard_invariant (bt-nav-fixes
        CP6) — press reuses the nav-graph approach phase, so the same pose/
        path failure modes apply. Returns a typed failure-reason string
        instead of plain ``False`` so InvariantGuard can report exactly
        which clause failed.
        """
        if state.platform_state.get("robot_position") is None:
            return "pose_lost"
        if _pose_is_stale(self._local_planner):
            return "stale_pose"
        planned = handle.result_data.get("planned_path")
        if planned is not None and len(planned) == 0:
            return "path_invalidated"
        return True

    # -- Prepare --------------------------------------------------------------

    def prepare(
        self,
        params: Dict[str, Any],
        context: SkillContext,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        target_ref = params.get("target_object", "")
        if not target_ref:
            return {}, "No target_object provided."

        store = context.object_store
        if store is None:
            return {}, "ObjectStore not available."

        scene_obj = store.get(target_ref)
        if scene_obj is None:
            available = [o.handle for o in store.all_objects()]
            avail_str = ", ".join(available[:10]) if available else "none"
            return (
                {},
                f"'{target_ref}' not in ObjectStore. Available: {avail_str}.",
            )

        # Extract object position (not viewpoint)
        obj_pos = getattr(scene_obj, "position", None)
        if obj_pos is None:
            return {}, f"'{target_ref}' has no position field."

        goal_x = obj_pos.get("x", 0.0)
        goal_y = obj_pos.get("y", 0.0)

        if not (math.isfinite(goal_x) and math.isfinite(goal_y)):
            return {}, f"Invalid position for '{target_ref}'."

        # Theta: face the object
        robot_pos = context.platform_state.get("robot_position")
        if robot_pos is not None:
            goal_theta = math.atan2(goal_y - robot_pos[1], goal_x - robot_pos[0])
        else:
            goal_theta = 0.0

        prepared = {
            "goal_x": goal_x,
            "goal_y": goal_y,
            "goal_theta": goal_theta,
            "target_handle": scene_obj.handle,
            "target_label": scene_obj.label,
        }
        # Approach pose = where the robot stands BEFORE the press moves (the
        # viewpoint the preceding navigate reached). _ReturnToApproach steps
        # back here after the trigger fired. Recording it later (after the
        # press's own drive toward the panel) left the body straddling the
        # neighboring panel and re-firing it (GT smoke, 2026-09-04).
        if robot_pos is not None and len(robot_pos) >= 2:
            prepared["_press_approach_pose"] = (
                float(robot_pos[0]),
                float(robot_pos[1]),
                float(robot_pos[2]) if len(robot_pos) > 2 else 0.0,
            )
        return prepared, None

    # -- Subtree --------------------------------------------------------------

    def build_subtree(
        self,
        params: Dict[str, Any],
        context: SkillContext,
        handle: SkillHandle,
    ) -> py_trees.behaviour.Behaviour:
        body = py_trees.composites.Sequence(
            name="press",
            memory=True,
            children=[
                _ValidateTarget(params, handle),
                _PlanPath(params, handle, self._path_planner, self._local_planner, context),
                _PressFollowPath(params, handle, self._path_planner, self._local_planner),
                _DriveOntoTarget(params, handle, self._local_planner),
                _MarkPressed(params, self._path_planner),
                _ReturnToApproach(params, handle, self._local_planner),
            ],
        )
        # Press drives the same goal chain as navigate (shared _FollowPath +
        # its own send_waypoint approach), so it inherits the same terminal
        # defect and the same fix — see navigate.py's terminal-discipline
        # section.
        return _TerminalDiscipline(child=body, local_planner=self._local_planner)


# ---------------------------------------------------------------------------
# BT Nodes
# ---------------------------------------------------------------------------


class _ValidateTarget(py_trees.behaviour.Behaviour):
    """Check that goal coordinates are valid."""

    def __init__(self, params: Dict, handle: SkillHandle) -> None:
        super().__init__(name="validate_target")
        self._params = params
        self._handle = handle

    def update(self) -> py_trees.common.Status:
        if self._handle.is_cancel_requested:
            return py_trees.common.Status.FAILURE

        x = self._params.get("goal_x", 0.0)
        y = self._params.get("goal_y", 0.0)
        label = self._params.get("target_label", "target")

        if not (math.isfinite(x) and math.isfinite(y)):
            self._handle.observation = f"Invalid goal coordinates for '{label}'."
            self._handle.result_data = _build_result(self._params, "failed_invalid")
            return py_trees.common.Status.FAILURE

        self._handle.observation = f"Target validated for '{label}'."
        return py_trees.common.Status.SUCCESS


class _PlanPath(py_trees.behaviour.Behaviour):
    """Compute path via PathPlanner. Updates map data first."""

    def __init__(
        self,
        params: Dict,
        handle: SkillHandle,
        planner: PathPlanner,
        local_planner: LocalPlannerInterface,
        context: SkillContext,
    ) -> None:
        super().__init__(name="plan_path")
        self._params = params
        self._handle = handle
        self._planner = planner
        self._local_planner = local_planner
        self._context = context

    def update(self) -> py_trees.common.Status:
        if self._handle.is_cancel_requested:
            return py_trees.common.Status.FAILURE

        label = self._params.get("target_label", "target")
        goal_x = self._params["goal_x"]
        goal_y = self._params["goal_y"]

        nav_graph = self._context.platform_state.get("scene_graph")
        if nav_graph is not None:
            self._planner.update_map(nav_graph)

        rx, ry, _ = self._local_planner.get_robot_pose()

        self._handle.observation = f"Planning path to '{label}'..."
        # Interactable keep-out (NavGraphPlanner.set_keep_out): name this leg's
        # target so its own floor trigger is not routed around.
        set_target = getattr(self._planner, "set_active_target", None)
        if set_target is not None:
            set_target(self._params.get("target_handle"))
        waypoints = self._planner.plan((rx, ry), (goal_x, goal_y))

        if waypoints is None or len(waypoints) == 0:
            self._handle.observation = f"No path to '{label}'. Target may be unreachable."
            self._handle.result_data = _build_result(self._params, "failed_no_path")
            return py_trees.common.Status.FAILURE

        self._handle.result_data = {
            "_waypoints": waypoints,
            "planned_path": waypoints,
        }
        # CP6: same scaled wall-time budget as NavigateSkill (bt-nav-fixes
        # CP4) — press's nav-graph approach phase has the same distance
        # dimension, so a static number would either false-kill long
        # approaches or under-bound short ones. The final drive-onto-target
        # phase is short and already has its own leaf-level _PRESS_TIMEOUT.
        self._handle.wall_time_budget_seconds = _scaled_wall_time_budget_s(_path_length_m((rx, ry), waypoints))
        self._handle.observation = f"Path planned to '{label}': {len(waypoints)} waypoints."
        self._handle.progress = 0.05
        return py_trees.common.Status.SUCCESS


class _PressFollowPath(_FollowPath):
    """``_FollowPath`` reused from navigate.py (waypoint sequencing is
    identical), but press's own outcome-status key is ``press_status`` —
    per ``bt_skills/docs/press.md``, ``human_baseline.py`` and
    ``skills_contract_v0``, not the ``arrival_status`` key the shared leaf
    writes (that's navigate's contract).

    On failure the shared leaf's navigate-shaped result is *translated*, not
    mirrored: the code moves to ``press_status`` and the payload is rebuilt
    through press's own ``_build_result``. Two reasons the old mirror (which
    left ``arrival_status`` in place alongside) had to go:

    1. Both routers map shared codes to byte-identical guidance sentences, so
       publishing both keys would append the same sentence twice to the
       observation once ``press_status`` is routed.
    2. The leaked payload carried navigate-only fields (``viewpoint_index``,
       ``total_viewpoints``, ``is_exploration``) and lacked press's
       ``final_distance`` — a result shape press never documents.

    navigate.py itself stays untouched.
    """

    def update(self) -> py_trees.common.Status:
        status = super().update()
        if status != py_trees.common.Status.FAILURE:
            return status
        # Cancellation exits the shared leaf without an arrival_status; that
        # path is reported by the safeguard wrap's failure_reason, exactly as
        # it is for navigate.
        arrival_status = self._handle.result_data.get("arrival_status")
        if arrival_status is not None:
            self._handle.result_data = _build_result(self._params, arrival_status)
        return status


class _DriveOntoTarget(py_trees.behaviour.Behaviour):
    """Drive precisely onto the floor object.

    After _FollowPath gets the robot near the target via nav-graph
    waypoints, this node drives directly to the object position.

    Two phases:
    1. Wait for movement (local planner startup latency).
    2. Drive onto object — SUCCESS when dist < 0.2m, or if robot
       is physically blocked within 0.5m for 1.5s.
    """

    def __init__(
        self,
        params: Dict,
        handle: SkillHandle,
        local_planner: LocalPlannerInterface,
    ) -> None:
        super().__init__(name="drive_onto_target")
        self._params = params
        self._handle = handle
        self._local_planner = local_planner

    def initialise(self) -> None:
        self._start = time.monotonic()
        self._saw_movement = False
        self._stopped_since: Optional[float] = None
        self._sent_goal = False
        # Approach pose fallback (prepare() records it before any motion);
        # only set here when prepare() could not read the robot position.
        if "_press_approach_pose" not in self._params:
            rx, ry, ryaw = self._local_planner.get_robot_pose()
            self._params["_press_approach_pose"] = (rx, ry, ryaw)

    def update(self) -> py_trees.common.Status:
        label = self._params.get("target_label", "target")
        goal_x = self._params["goal_x"]
        goal_y = self._params["goal_y"]
        now = time.monotonic()
        elapsed = now - self._start

        if self._handle.is_cancel_requested:
            self._local_planner.cancel()
            return py_trees.common.Status.FAILURE

        # Send waypoint to exact object position
        if not self._sent_goal:
            theta = self._params.get("goal_theta", 0.0)
            self._local_planner.send_waypoint(goal_x, goal_y, theta)
            self._sent_goal = True

        linear, angular = self._local_planner.get_robot_velocity()
        rx, ry, _ = self._local_planner.get_robot_pose()
        dist = math.hypot(goal_x - rx, goal_y - ry)

        # >>> NAV-DEBUG
        _nav_debug_log({
            "stage": "drive_onto_target",
            "ts": now,
            "label": label,
            "linear_vel": round(linear, 3),
            "angular_vel": round(angular, 3),
            "dist_to_goal": round(dist, 3),
            "saw_movement": self._saw_movement,
            "elapsed_s": round(elapsed, 2),
        })
        # <<< NAV-DEBUG

        # Track movement
        if not self._saw_movement and (linear > _MOVEMENT_THRESHOLD or angular > _MOVEMENT_THRESHOLD):
            self._saw_movement = True
            logger.info(
                "Press movement detected for '%s' (v=%.2f, ω=%.2f)",
                label,
                linear,
                angular,
            )

        # Timeout
        if elapsed > _PRESS_TIMEOUT:
            if not self._saw_movement:
                logger.warning("Press timeout for '%s' — never saw movement", label)
                self._handle.observation = (
                    f"[leaf=DriveOntoTarget] Press on '{label}' failed: "
                    f"robot never started moving (waited {elapsed:.1f}s)."
                )
                self._handle.result_data = _build_result(
                    self._params,
                    "failed_no_movement",
                    final_distance=dist,
                )
                return py_trees.common.Status.FAILURE
            logger.warning("Press timeout for '%s', accepting at %.2fm", label, dist)
            return self._accept(label, dist, suffix="(timeout)")

        # Success: on top of object
        if dist < _PRESS_DISTANCE:
            return self._accept(label, dist)

        # Velocity-based: robot blocked near object
        if self._saw_movement:
            is_stopped = linear < _MOVEMENT_THRESHOLD and angular < _MOVEMENT_THRESHOLD
            if is_stopped:
                if self._stopped_since is None:
                    self._stopped_since = now
                elif now - self._stopped_since >= _SETTLE_DURATION and dist < _PRESS_BLOCKED_DIST:
                    logger.info(
                        "Press: robot blocked at %.2fm from '%s', accepting",
                        dist,
                        label,
                    )
                    return self._accept(label, dist)
            else:
                self._stopped_since = None

        # Re-send waypoint periodically
        self._local_planner.send_waypoint(goal_x, goal_y, self._params.get("goal_theta", 0.0))

        self._handle.progress = 0.9 + 0.1 * max(0, 1 - dist / _PRESS_BLOCKED_DIST)
        self._handle.observation = f"Pressing onto '{label}'... ({dist:.2f}m away)"
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        if new_status == py_trees.common.Status.INVALID:
            self._local_planner.cancel()

    def _accept(
        self,
        label: str,
        final_dist: float,
        suffix: str = "",
    ) -> py_trees.common.Status:
        # Stale pose ⇒ can't trust the distance-to-target reading used to
        # decide arrival — refuse and keep trying (bt-nav-fixes CP6; mirrors
        # NavigateSkill._ConfirmArrival._accept).
        pose_age = _pose_age(self._local_planner)
        if pose_age is not None and pose_age > _POSE_STALENESS_S:
            logger.warning("Refusing press-accept for '%s' — pose stale (age=%.2fs)", label, pose_age)
            self._stopped_since = None
            return py_trees.common.Status.RUNNING

        obs = f"Stepped on '{label}'."
        if suffix:
            obs += f" {suffix}"
        self._handle.observation = obs
        self._handle.progress = 1.0
        self._handle.result_data = _build_result(
            self._params,
            "pressed",
            final_distance=final_dist,
        )
        return py_trees.common.Status.SUCCESS


class _MarkPressed(py_trees.behaviour.Behaviour):
    """Tell the path planner which trigger was just pressed so the keep-out
    exempts it by identity on the next legs (NavGraphPlanner.set_last_pressed)."""

    def __init__(self, params: Dict, path_planner: Any) -> None:
        super().__init__(name="mark_pressed")
        self._params = params
        self._path_planner = path_planner

    def update(self) -> py_trees.common.Status:
        mark = getattr(self._path_planner, "set_last_pressed", None)
        if mark is not None:
            mark(self._params.get("target_handle"))
        return py_trees.common.Status.SUCCESS


class _ReturnToApproach(py_trees.behaviour.Behaviour):
    """After the press registered, step back off the trigger to the pose the
    robot pressed from (``_press_approach_pose``). A floor trigger commonly
    actuates something nearby (a safe door, a closet door): standing on the
    trigger while it swings leaves the robot pinned. Always SUCCESS — the
    press itself already succeeded; failing to step back only logs."""

    def __init__(self, params: Dict, handle: SkillHandle, local_planner: LocalPlannerInterface) -> None:
        super().__init__(name="return_to_approach")
        self._params = params
        self._handle = handle
        self._local_planner = local_planner

    def initialise(self) -> None:
        self._start = time.monotonic()
        self._sent = False
        self._pressed_observation = self._handle.observation
        self._pressed_result = self._handle.result_data

    def update(self) -> py_trees.common.Status:
        pose = self._params.get("_press_approach_pose")
        if not pose or self._handle.is_cancel_requested:
            return py_trees.common.Status.SUCCESS
        ax, ay, _ = pose
        rx, ry, _ = self._local_planner.get_robot_pose()
        now = time.monotonic()
        elapsed = now - self._start
        if math.hypot(ax - rx, ay - ry) < _RETURN_MIN_DIST:
            return py_trees.common.Status.SUCCESS
        if elapsed < _RETURN_DWELL_S:
            return py_trees.common.Status.RUNNING  # let the trigger register
        if not self._sent:
            theta = math.atan2(ay - ry, ax - rx)
            self._local_planner.send_waypoint(ax, ay, theta)
            self._sent = True
        dist = math.hypot(ax - rx, ay - ry)
        if dist < _RETURN_ACCEPT_DIST or elapsed > _RETURN_TIMEOUT_S:
            if elapsed > _RETURN_TIMEOUT_S:
                logger.warning("Press: could not step back to the approach pose (%.2fm away)", dist)
            self._local_planner.cancel()
            self._handle.observation = self._pressed_observation
            self._handle.result_data = self._pressed_result
            return py_trees.common.Status.SUCCESS
        self._local_planner.send_waypoint(ax, ay, math.atan2(ay - ry, ax - rx))
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        if new_status == py_trees.common.Status.INVALID:
            self._local_planner.cancel()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _build_result(
    params: Dict[str, Any],
    press_status: str,
    final_distance: float = -1.0,
) -> Dict[str, Any]:
    """Press's result payload, keyed by ``press_status`` at every outcome.

    The complete code set (skills_contract_v0 press section) and where each
    is written — every terminal outcome of the body subtree goes through
    here, mirroring NavigateSkill's ``arrival_status`` discipline:

    ===================  =========================================
    ``pressed``          ``_DriveOntoTarget._accept``
    ``failed_invalid``   ``_ValidateTarget`` (non-finite goal)
    ``failed_no_path``   ``_PlanPath`` (empty plan) and
                         ``_PressFollowPath`` (replan budget spent,
                         planner returning no path)
    ``failed_blocked``   ``_PressFollowPath`` (replan budget spent
                         with a path but no progress)
    ``failed_no_movement`` ``_DriveOntoTarget`` (press timeout,
                         robot never moved)
    ===================  =========================================

    Cancellation is deliberately absent: like navigate, it is reported by
    the safeguard wrap's ``failure_reason``, not by an outcome status.
    Every failure code above is routed to agent guidance by
    ``bt_agent._engine.agent.lam_agent.actor.route_press_status_guidance``.
    """
    return {
        "press_status": press_status,
        "target_handle": params.get("target_handle", ""),
        "target_label": params.get("target_label", ""),
        "goal": {
            "x": params.get("goal_x", 0.0),
            "y": params.get("goal_y", 0.0),
        },
        "final_distance": final_distance,
    }
