"""NavigateSkill — move the robot to an object identified by QUERY.

The skill resolves the target from the ObjectStore, computes an SE2 goal,
plans a path via the injected PathPlanner, and sequences waypoints to the
injected LocalPlannerInterface. ALL navigation logic lives here in pycore;
the ROS2 bridge just forwards waypoints and reads odometry.

Body subtree (framework auto-wraps with safeguards — see safeguard.py):

    Sequence("navigate_with_heading", memory=True)
    ├── Selector("navigate", memory=True)
    │   ├── _CheckAlreadyAtGoal — positional shortcut when near goal AND stationary
    │   └── Sequence("navigate_plan_and_execute", memory=True)
    │       ├── _ValidateGoal   — check coords valid, planner has map
    │       ├── _PlanPath       — Dijkstra/A* via PathPlanner → waypoint list
    │       ├── _ConfirmPath    — optional operator approval before motion
    │       ├── _FollowPath     — send waypoints, poll pose, replan on stuck
    │       └── _ConfirmArrival — wait for settling, retain positional verdict
    └── _AlignHeading — align after either arrival path; report residual/timeout

Agent-facing docs (leaf catalog, safeguard semantics, recovery hints):
``tiamat_skills/docs/navigate.md`` — surfaced at runtime via
``read_skill_docs("navigate")``.
"""

from __future__ import annotations

import json as _json
import logging
import math
import os
import time
from abc import ABC, abstractmethod
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import py_trees
from bt_agent.skills import ActionSkill, SkillContext, SkillHandle
from bt_agent_robotics.skills.utils.compute_pose import compute_pose
from bt_agent_robotics.skills.utils.path_planner import PathPlanner

logger = logging.getLogger(__name__)

# Arrival is only accepted when the robot is within this distance of the
# goal (meters).  Prevents "arrived" reports when the robot stopped short.
_ARRIVAL_DISTANCE = 0.4

# >>> NAV-DEBUG
_NAV_DEBUG_WRITER = None


def _nav_debug_log(entry):
    if _NAV_DEBUG_WRITER is not None:
        try:
            _NAV_DEBUG_WRITER(entry)
        except Exception:
            pass  # Optional diagnostics only.


# <<< NAV-DEBUG


# ---------------------------------------------------------------------------
# Pose freshness (backend-agnostic; degrades gracefully when unsupported)
# ---------------------------------------------------------------------------

# Pose is considered stale once it's older than this (seconds since the
# local planner's last odom update).
_POSE_STALENESS_S = 1.0

# Log-once guard: warn a single time per process when a local planner
# doesn't expose pose freshness at all (graceful degradation to today's
# behavior — staleness checks never trigger).
_stale_pose_unavailable_logged = False


def _pose_age(local_planner: "LocalPlannerInterface") -> Optional[float]:
    """Seconds since ``local_planner``'s pose was last updated.

    Returns ``None`` when the backend doesn't track freshness (no
    ``get_pose_age`` method, or it doesn't return a number — e.g. a bare
    mock in tests). Callers must treat ``None`` as "can't judge staleness"
    and degrade to pre-fix behavior, not as "stale".
    """
    getter = getattr(local_planner, "get_pose_age", None)
    if getter is None:
        return None
    try:
        age = getter()
    except Exception:
        return None
    if not isinstance(age, (int, float)):
        return None
    return float(age)


def _pose_is_stale(local_planner: "LocalPlannerInterface") -> bool:
    """True only if the pose is stamped AND older than ``_POSE_STALENESS_S``."""
    age = _pose_age(local_planner)
    if age is None:
        global _stale_pose_unavailable_logged
        if not _stale_pose_unavailable_logged:
            logger.info(
                "Local planner does not expose pose freshness (get_pose_age) — "
                "staleness checks degrade to pre-fix behavior."
            )
            _stale_pose_unavailable_logged = True
        return False
    return age > _POSE_STALENESS_S


# ---------------------------------------------------------------------------
# LocalPlannerInterface ABC
# ---------------------------------------------------------------------------


class LocalPlannerInterface(ABC):
    """Proxy to the local motion planner (MPPI/DWA).

    The real implementation publishes PoseStamped to /goal_waypoint
    (for goal-directed nav) and Twist on /spot/cmd_vel (for direct
    velocity control by the relative-move skills) and reads Odometry
    from /spot/platform/odom.

    Implementations:
        - lam_v2_roscore: thin ROS2 pub/sub wrapper
        - lam_v2_testing: MockLocalPlanner (simulates movement)
    """

    @abstractmethod
    def send_waypoint(self, x: float, y: float, theta: float) -> None:
        """Publish a waypoint for the local planner to drive to."""
        ...

    @abstractmethod
    def send_velocity(self, vx_body: float, vy_body: float, wz: float) -> None:
        """Publish a body-frame velocity command (Twist on /spot/cmd_vel).

        Used by the relative-move skills (Σ_act move_forward / move_backward /
        move_left / move_right) which run their own per-tick PD controller
        directly on the locomotion policy, bypassing the waypoint path
        follower. Strafe (``vy_body != 0``) requires the locomotion policy
        to accept 3-DOF Twist — Spot's RL locomotion policy does.

        Convention (REP-103, robot body frame):
            vx_body > 0  → forward
            vy_body > 0  → left
            wz     > 0   → yaw left (CCW)
        """
        ...

    @abstractmethod
    def get_robot_pose(self) -> Tuple[float, float, float]:
        """Read current robot pose (x, y, yaw) from odometry."""
        ...

    @abstractmethod
    def get_robot_velocity(self) -> Tuple[float, float]:
        """Read current robot velocity from odometry.

        Returns:
            (linear_speed_m_s, angular_speed_rad_s)
        """
        ...

    @abstractmethod
    def cancel(self) -> None:
        """Stop the local planner."""
        ...

    def get_pose_age(self) -> Optional[float]:
        """Seconds since the last pose update, or ``None`` if untracked.

        Concrete (not abstract) — implementations that don't plumb a
        timestamped odom ref inherit this default and staleness checks
        degrade gracefully rather than break. ``RosLocalPlanner`` overrides
        it with a real freshness reading.
        """
        return

    def set_brake(self, engaged: bool) -> None:
        """Engage (``True``) or release (``False``) the operator brake.

        The real implementation publishes ``std_msgs/Int8`` on ``/stop`` —
        the stop input the localPlanner / pathFollower already subscribe to
        (/stop is a pathFollower BITMASK; engage publishes bit4=16, publish-suppress; 0 clears). Concrete (not
        abstract) for the same reason as ``get_pose_age`` above: backends
        that don't wire a brake inherit a no-op and degrade to park-at-self
        alone instead of breaking.
        """
        return


# ---------------------------------------------------------------------------
# Terminal discipline — park at self + operator brake
# ---------------------------------------------------------------------------
#
# Observed on hardware (2026-08-08): after the skill reported a terminal
# status, the chain behind it (goal_adapter -> localPlanner -> pathFollower
# -> cmd_vel_adapter -> cmd_vel_mux) kept commanding — goal_adapter latches
# the last waypoint, so the follower station-keeps or creeps indefinitely.
# The skill must never leave the chain able to publish cmd_vel once it is
# done. Two independent layers, applied on EVERY terminal path by
# ``_TerminalDiscipline`` below:
#
#   1. Park at self — republish the robot's own pose as the goal, so the
#      latched goal can no longer pull it anywhere. Refused when the pose
#      can't be trusted: a fabricated park goal is worse than none.
#   2. Operator brake — /stop, which holds even when layer 1 was refused.
#
# Both run AFTER the leaves have written their typed outcome, so neither
# can mask a failure code.


def _park_at_self(local_planner: "LocalPlannerInterface") -> bool:
    """Publish the robot's current pose as the planner goal.

    Uses the same goal channel the skill drives during navigation
    (``send_waypoint`` -> ``/goal_waypoint``), so the latched goal becomes a
    no-op instead of a destination.

    Returns ``True`` when a park goal was published, ``False`` (without
    publishing) when the pose can't be trusted — stale, unreadable, or
    non-finite. Backends that don't track freshness at all degrade to
    pre-fix behavior (``cancel()`` already republished the current pose
    unconditionally), so this is never *more* trusting than the status quo.
    """
    if _pose_is_stale(local_planner):
        logger.warning("Not parking at self — pose is stale; relying on the brake alone.")
        return False
    try:
        x, y, yaw = local_planner.get_robot_pose()
        if not (math.isfinite(x) and math.isfinite(y) and math.isfinite(yaw)):
            logger.warning("Not parking at self — pose is not finite (%s, %s, %s).", x, y, yaw)
            return False
        local_planner.send_waypoint(x, y, yaw)
        return True
    except Exception as exc:  # noqa: BLE001 — a terminal path must never raise
        logger.warning("Not parking at self — park failed (%s: %s).", type(exc).__name__, exc)
        return False


def _set_brake(local_planner: "LocalPlannerInterface", engaged: bool) -> None:
    """Engage/release the operator brake, tolerating backends without one.

    Duck-typed (``getattr``) rather than a direct call because local-planner
    doubles in tests and older bridges implement this interface by
    convention, not inheritance — same graceful-degradation contract as
    ``get_pose_age``.
    """
    setter = getattr(local_planner, "set_brake", None)
    if setter is None:
        return
    try:
        setter(engaged)
    except Exception as exc:  # noqa: BLE001 — a terminal path must never raise
        logger.warning("Brake %s failed (%s: %s).", "engage" if engaged else "release", type(exc).__name__, exc)


class _TerminalDiscipline(py_trees.decorators.Decorator):
    """Wraps a motion skill's body so the chain is idle once it stops RUNNING.

    Status-transparent: forwards the child's status verbatim, so skill
    outcomes and their typed codes (the frozen contract) are untouched. It
    owns only *when* the motion chain is armed:

      - first tick, before the body publishes any goal: release the brake
        unless an operator confirmation gate owns that release;
      - every terminal transition — SUCCESS, FAILURE, or INVALID
        (cancellation, safeguard abort, priority interrupt): park at self,
        then engage the brake.

    py_trees guarantees the coverage: a decorator's ``terminate`` runs on
    its own SUCCESS/FAILURE, the ``memory=False`` safeguard Sequence
    invalidates the body (``stop(INVALID)``) whenever a guard fails
    mid-run, and ``_tick_subtree_loop`` does the same on cancellation.
    """

    def __init__(
        self,
        child: py_trees.behaviour.Behaviour,
        local_planner: "LocalPlannerInterface",
        release_brake_on_initialise: bool = True,
        name: str = "terminal_discipline",
    ) -> None:
        super().__init__(name=name, child=child)
        self._local_planner = local_planner
        self._release_brake_on_initialise = release_brake_on_initialise

    def initialise(self) -> None:
        if self._release_brake_on_initialise:
            _set_brake(self._local_planner, False)

    def tick(self):
        # Crash-safety: a leaf exception aborts the tick generator without
        # py_trees ever assigning this decorator a status — Composite.stop
        # then SKIPS it (INVALID->INVALID), so terminate() would never run
        # even though initialise() already released the brake. The
        # discipline owns "never exit un-braked", so enforce it here and
        # re-raise for the worker-thread net to convert into typed FAILURE.
        try:
            yield from super().tick()
        except Exception:
            try:
                _park_at_self(self._local_planner)
            finally:
                _set_brake(self._local_planner, True)
            raise

    def update(self) -> py_trees.common.Status:
        return self.decorated.status

    def terminate(self, new_status: py_trees.common.Status) -> None:
        if new_status not in (
            py_trees.common.Status.SUCCESS,
            py_trees.common.Status.FAILURE,
            py_trees.common.Status.INVALID,
        ):
            return
        # Brake engagement is unconditional: parking is best-effort, but a
        # terminal must NEVER exit with the brake released.
        try:
            _park_at_self(self._local_planner)
        finally:
            _set_brake(self._local_planner, True)


# ---------------------------------------------------------------------------
# NavigateSkill
# ---------------------------------------------------------------------------

# Timeout before declaring a waypoint stuck (seconds)
_WAYPOINT_TIMEOUT = 10.0

# Recovery budget: max replan attempts triggered by being stuck on one
# waypoint (timeout or no-progress). An attempt is spent whenever a replan
# does NOT yield genuine distance-to-goal progress (> _NO_PROGRESS_EPS_M)
# since the last budget reset — that covers a replan returning the
# identical waypoint sequence, no path at all, AND a genuinely *different*
# path that doesn't measurably close the distance to the goal (the
# A->B->A ping-pong trap from odometry noise). Exceeding the budget fails
# the skill instead of spinning to the 120s wall-time guard.
_MAX_REPLAN_ATTEMPTS = 3

# No-progress guard: distance-to-current-waypoint must improve by more than
# this (meters) within _NO_PROGRESS_WINDOW_S, or recovery triggers early —
# before the (longer) fixed per-waypoint timeout above. Also gates the
# goal-distance check that decides whether a stuck-branch replan counts as
# genuine progress (see ``_best_dist_to_goal`` in ``_FollowPath``): that
# baseline only ratchets down when it improves by more than this eps, so
# sub-eps creep can neither fake progress nor mask real (if slow)
# advancement toward the goal.
_NO_PROGRESS_EPS_M = 0.05
_NO_PROGRESS_WINDOW_S = 5.0


# ---------------------------------------------------------------------------
# Scaled wall-time budget (bt-nav-fixes CP4)
# ---------------------------------------------------------------------------
#
# The generic ActionSkill default (120s, safeguard.py) was never navigate-
# tuned: a long legitimate route can exceed it (false kill) while a short
# hop doesn't need anywhere near that much slack. Instead, ``_PlanPath``
# (initial plan) and ``_FollowPath`` (every successful replan) compute a
# budget scaled to the actual route length via ``handle.wall_time_budget_
# seconds`` (WallTimeBudgetGuard reads it fresh each tick — safeguard.py).

# Spot's comfortable walk speed (m/s) — used only to size the wall-time
# budget, not as a control-loop setpoint (the local planner owns actual
# velocity). No faster-authoritative number found in-repo; brief-suggested
# default.
NOMINAL_WALK_SPEED_MPS = 0.6

# Budget = clamp(2 * planned_path_length / NOMINAL_WALK_SPEED_MPS, floor, cap).
# The 2x factor covers waypoint re-sends, replan pauses, and settling time
# on top of straight-line travel time at nominal speed.
_WALL_TIME_BUDGET_FLOOR_S = 60.0
_WALL_TIME_BUDGET_CAP_S = 600.0


def _path_length_m(start_xy: Tuple[float, float], waypoints: List[Tuple[float, float]]) -> float:
    """Sum of Euclidean segment lengths from ``start_xy`` through ``waypoints``."""
    if not waypoints:
        return 0.0
    points = [start_xy] + list(waypoints)
    return sum(
        math.hypot(points[i + 1][0] - points[i][0], points[i + 1][1] - points[i][1]) for i in range(len(points) - 1)
    )


# Wall-budget scale for simulators slower than real time (Isaac Sim runs at
# ~0.3x with the follow and semantic cameras on): every wall-time budget the
# navigate skill sizes is multiplied by it. Set from the node's
# ``wall_budget_scale`` parameter (set_wall_budget_scale); 1.0 = real time.
WALL_BUDGET_SCALE = 1.0


def set_wall_budget_scale(scale: float) -> None:
    global WALL_BUDGET_SCALE
    WALL_BUDGET_SCALE = max(0.1, float(scale))


# Pinned-robot back-off (opt-in, node param ``nav_backoff``). When the stuck
# branch fires and the robot has not moved since its waypoint was sent, it
# is physically pinned (a cone, a door frame, a wall): the path follower
# keeps commanding full speed and nothing happens (Isaac oracle, umbrella
# spot: 14 min commanded at 1.2 m/s, 7 m travelled). Reverse briefly on the
# primitive cmd_vel input (mux priority over the follower), then let the
# follower resume. Twice per leg at most; then the normal replan budget.
NAV_BACKOFF_ENABLED = False
_BACKOFF_SPEED_MPS = 0.4
_BACKOFF_S = 3.0
_BACKOFF_TURN_RAD_S = 0.6
_MAX_BACKOFFS_PER_LEG = 2
# A wedged legged robot still jitters (Isaac pass 5: 0.2-0.5 m of shuffling
# at the same spot for 15 minutes), so "pinned" is judged on net displacement
# since the waypoint was sent, not on being perfectly still.
_PINNED_EPS_M = 0.5


def set_nav_backoff(enabled: bool) -> None:
    global NAV_BACKOFF_ENABLED
    NAV_BACKOFF_ENABLED = bool(enabled)


# Simulator rescue hook (node param ``sim_nudge_topic``): called only when the
# robot is still pinned after every back-off of the leg. None = no rescue
# (the replan budget runs as before). Isaac Sim: gg-sim's /sim_nudge.
NAV_NUDGE_FN: Optional[Callable[[], None]] = None
_MAX_NUDGES_PER_LEG = 2


def set_nav_nudge(fn: Optional[Callable[[], None]]) -> None:
    global NAV_NUDGE_FN
    NAV_NUDGE_FN = fn


def _scaled_wall_time_budget_s(path_length_m: float) -> float:
    """CP4 formula: clamp(2 * path_length / NOMINAL_WALK_SPEED_MPS, floor, cap),
    then times WALL_BUDGET_SCALE."""
    raw = 2.0 * path_length_m / NOMINAL_WALK_SPEED_MPS
    return max(_WALL_TIME_BUDGET_FLOOR_S, min(_WALL_TIME_BUDGET_CAP_S, raw)) * WALL_BUDGET_SCALE


def _nav_confirm_enabled(value: Optional[str] = None) -> bool:
    """Parse the deliberately narrow operator-gate truthy vocabulary."""
    raw = os.environ.get("LAM_NAV_CONFIRM", "") if value is None else value
    return raw.strip().lower() in ("1", "true")


def _nav_confirm_timeout_s() -> float:
    """Read the confirmation timeout, falling back safely on malformed input."""
    try:
        return float(os.environ.get("LAM_NAV_CONFIRM_TIMEOUT_S", "300"))
    except ValueError:
        return 300.0


def _numeric_object_id(value: Any) -> Optional[int]:
    """Convert the ObjectStore's scene-graph ID to the crop filename ID."""
    if value is None or isinstance(value, bool):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class ConfirmGate:
    """ROS-backed publication and input channel for operator path approval.

    ROS imports and entities are created only when enabled, preserving the
    ROS-free behavior of the skill and its unit tests when the gate is off.
    """

    def __init__(self, node: Any, enabled: bool) -> None:
        self.enabled = enabled
        self._node = node
        self._latest: Optional[str] = None
        if not enabled:
            return
        if node is None:
            raise RuntimeError("LAM_NAV_CONFIRM requires a ROS-backed local planner")

        from nav_msgs.msg import Path
        from rclpy.qos import DurabilityPolicy, QoSProfile, ReliabilityPolicy
        from sensor_msgs.msg import CompressedImage
        from std_msgs.msg import String
        from visualization_msgs.msg import MarkerArray

        qos = QoSProfile(
            depth=1,
            durability=DurabilityPolicy.TRANSIENT_LOCAL,
            reliability=ReliabilityPolicy.RELIABLE,
        )
        self._path_type = Path
        self._marker_array_type = MarkerArray
        self._compressed_image_type = CompressedImage
        self._target_image_skip_logged = False
        self._path_pub = node.create_publisher(Path, "/lam/planned_path", qos)
        self._markers_pub = node.create_publisher(MarkerArray, "/lam/planned_path_markers", qos)
        self._target_image_pub = node.create_publisher(
            CompressedImage,
            "/lam/target_object_image",
            qos,
        )
        self._subscription = node.create_subscription(String, "/lam/nav_confirm", self._on_confirm, 10)

    def _on_confirm(self, message: Any) -> None:
        self._latest = message.data.strip().lower()

    def publish_plan(
        self,
        waypoints: List[Tuple[float, float]],
        label: str,
        robot_xy: Tuple[float, float],
        object_id: Optional[int] = None,
    ) -> None:
        """Publish the candidate route as a latched Path and MarkerArray."""
        if not self.enabled:
            return

        from geometry_msgs.msg import Point, PoseStamped
        from visualization_msgs.msg import Marker

        self._latest = None
        stamp = self._node.get_clock().now().to_msg()
        points_xy = [robot_xy] + list(waypoints)

        path = self._path_type()
        path.header.frame_id = "vision"
        path.header.stamp = stamp
        for x, y in points_xy:
            pose = PoseStamped()
            pose.header.frame_id = "vision"
            pose.header.stamp = stamp
            pose.pose.position.x = float(x)
            pose.pose.position.y = float(y)
            pose.pose.position.z = 0.1
            pose.pose.orientation.w = 1.0
            path.poses.append(pose)
        self._path_pub.publish(path)

        def marker(marker_id: int, marker_type: int) -> Any:
            result = Marker()
            result.header.frame_id = "vision"
            result.header.stamp = stamp
            result.ns = "lam_planned_path"
            result.id = marker_id
            result.type = marker_type
            result.action = Marker.ADD
            result.pose.orientation.w = 1.0
            return result

        delete_all = marker(0, Marker.LINE_STRIP)
        delete_all.action = Marker.DELETEALL

        line = marker(1, Marker.LINE_STRIP)
        line.scale.x = 0.07
        line.color.r = 0.1
        line.color.g = 1.0
        line.color.b = 0.1
        line.color.a = 0.9
        line.points = [Point(x=float(x), y=float(y), z=0.1) for x, y in points_xy]

        spheres = marker(2, Marker.SPHERE_LIST)
        spheres.scale.x = spheres.scale.y = spheres.scale.z = 0.15
        spheres.color.r = 0.1
        spheres.color.g = 1.0
        spheres.color.b = 0.1
        spheres.color.a = 0.9
        spheres.points = [Point(x=float(x), y=float(y), z=0.1) for x, y in waypoints]

        goal_x, goal_y = waypoints[-1]
        text = marker(3, Marker.TEXT_VIEW_FACING)
        text.pose.position.x = float(goal_x)
        text.pose.position.y = float(goal_y)
        text.pose.position.z = 0.35
        text.scale.z = 0.25
        text.color.r = text.color.g = text.color.b = 1.0
        text.color.a = 0.9
        text.text = label

        markers = self._marker_array_type()
        markers.markers = [delete_all, line, spheres, text]
        self._markers_pub.publish(markers)
        self.publish_target_image(object_id)

    def _debug_target_image_once(self, message: str, *args: Any) -> None:
        if getattr(self, "_target_image_skip_logged", False):
            return
        logger.debug(message, *args)
        self._target_image_skip_logged = True

    def publish_target_image(self, object_id: Optional[int]) -> None:
        """Publish the target's pre-extracted JPEG without affecting safety flow."""
        if not self.enabled:
            return
        try:
            crops_dir = os.environ.get("LAM_OBJECT_CROPS_DIR")
            if not crops_dir or object_id is None:
                self._debug_target_image_once(
                    "Target image unavailable: LAM_OBJECT_CROPS_DIR=%r object_id=%r",
                    crops_dir,
                    object_id,
                )
                return
            crop_path = os.path.join(crops_dir, f"object_{object_id:06d}.jpg")
            with open(crop_path, "rb") as crop_file:
                crop_bytes = crop_file.read()
            message = self._compressed_image_type()
            message.format = "jpeg"
            message.data = crop_bytes
            self._target_image_pub.publish(message)
        except Exception as exc:
            self._debug_target_image_once(
                "Target image publish skipped for object_id=%r: %s",
                object_id,
                exc,
            )

    def _clear_target_image(self) -> None:
        try:
            message = self._compressed_image_type()
            message.format = "jpeg"
            message.data = b""
            self._target_image_pub.publish(message)
        except Exception as exc:
            self._debug_target_image_once("Target image clear skipped: %s", exc)

    def poll(self) -> Optional[str]:
        """Return the latest recognized operator decision without blocking."""
        return self._latest if self._latest in ("go", "abort") else None

    def clear(self) -> None:
        """Remove the latched path and all path markers."""
        if not self.enabled:
            return

        from visualization_msgs.msg import Marker

        stamp = self._node.get_clock().now().to_msg()
        path = self._path_type()
        path.header.frame_id = "vision"
        path.header.stamp = stamp
        self._path_pub.publish(path)

        delete_all = Marker()
        delete_all.header.frame_id = "vision"
        delete_all.header.stamp = stamp
        delete_all.ns = "lam_planned_path"
        delete_all.id = 0
        delete_all.action = Marker.DELETEALL
        markers = self._marker_array_type()
        markers.markers = [delete_all]
        self._markers_pub.publish(markers)
        self._clear_target_image()
        self._latest = None


class NavigateSkill(ActionSkill):
    """Move to an object identified by QUERY.

    Resolves the target handle from the ObjectStore, computes an SE2 goal
    from the object's viewpoint, plans a path, and drives waypoint-by-waypoint.
    """

    name = "navigate"
    description = "Move to an object identified by QUERY."
    usage = "Use after QUERY returned a handle. Pass the handle to navigate there."
    params = {
        "target_object": "Handle from QUERY (e.g. 'red_umbrella_1').",
        "viewpoint_index": (
            "(Optional) 0-based index into the target's viewpoint list. "
            "Default 0 (first viewpoint). No 'primary'; use the explicit "
            "distinct raw indices in the navigation result or observation. "
            "A distinct-view count is not an index range."
        ),
    }

    def __init__(
        self,
        local_planner: LocalPlannerInterface,
        path_planner: PathPlanner,
        confirm_gate: Optional[ConfirmGate] = None,
    ) -> None:
        self._local_planner = local_planner
        self._path_planner = path_planner
        confirm_enabled = _nav_confirm_enabled()
        if confirm_gate is None:
            confirm_gate = ConfirmGate(getattr(local_planner, "_node", None), enabled=confirm_enabled)
        else:
            confirm_gate.enabled = confirm_enabled
        self._confirm_gate = confirm_gate

    def tool_def(self) -> Dict[str, Any]:
        """Hand-written: the generic builder (``Skill.tool_def``) marks every
        declared param as a required string, so it cannot express an
        optional integer param. ``viewpoint_index`` needs exactly that
        shape (unified-skill-api spec: opt, int, default 0).

        Mirrors ``QuerySkill.tool_def`` (see ``bt_skills/query.py``):
        ``target_object`` renders byte-for-byte as the generic builder
        would (same name, same description, still required);
        ``viewpoint_index`` is the only schema diff.
        """
        desc = self.description
        if self.usage:
            desc += " " + self.usage

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_object": {
                            "type": "string",
                            "description": self.params["target_object"],
                        },
                        "viewpoint_index": {
                            "type": "integer",
                            "minimum": 0,
                            "description": self.params["viewpoint_index"],
                        },
                        "memory_note": {
                            "type": "string",
                            "description": "Brief note for your future self about key findings or state.",
                        },
                    },
                    "required": ["target_object"],
                },
            },
        }

    def precondition(self, state: SkillContext) -> Tuple[bool, str]:
        if state.object_store is None:
            return False, "ObjectStore not available."
        if not state.object_store.all_objects():
            return False, "No object handles in ObjectStore (call QUERY first)."
        return True, ""

    def safeguard_invariant(self, state: SkillContext, handle: SkillHandle) -> Union[bool, str]:
        """Localization fresh AND (if path has been planned) cached path non-empty.

        I5-compliant: reads only ``state.robot_position`` (s_robot) and
        ``handle.result_data["planned_path"]`` (skill execution state).
        Does not consult ``object_store``, ``milestone``, ``main_task``,
        or ``scenario``. Pose freshness is read from ``self._local_planner``
        (backend infra, not K-store data) — permitted under the same rule.

        Returns a typed failure-reason string (bt-nav-fixes CP5) instead of
        plain ``False`` so ``InvariantGuard`` can report exactly which
        clause failed: ``"pose_lost"``, ``"stale_pose"``, or
        ``"path_invalidated"``.
        """
        if state.platform_state.get("robot_position") is None:
            return "pose_lost"
        if _pose_is_stale(self._local_planner):
            return "stale_pose"
        planned = handle.result_data.get("planned_path")
        if planned is not None and len(planned) == 0:
            return "path_invalidated"
        return True

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
                f"'{target_ref}' not in ObjectStore. Available: {avail_str}. Use QUERY first to find new objects.",
            )
        if getattr(scene_obj, "navigation_unavailable", False):
            return {}, f"'{target_ref}' has no approved navigation viewpoint. Report it unreachable."

        # Duck-typed field access — QuerySkill (producer) sets these attrs
        vp_pos = getattr(scene_obj, "viewpoint_position", None)
        if vp_pos is None:
            return {}, f"'{target_ref}' has no viewpoint_position. Use QUERY first."

        # Parse viewpoint index
        vp_idx_raw = params.get("viewpoint_index", "0")
        try:
            vp_idx = int(vp_idx_raw)
        except (ValueError, TypeError):
            vp_idx = 0

        catalog = _viewpoint_catalog(
            scene_obj,
            context.platform_state.get("camera_to_body_transforms"),
        )
        total_vps = max(1, len(catalog["distinct_indices"]))

        # Compute SE2 goal
        goal = compute_pose(
            scene_obj,
            viewpoint_index=vp_idx,
            robot_position=context.platform_state.get("robot_position"),
            camera_to_body_transforms=context.platform_state.get("camera_to_body_transforms"),
        )

        prepared = {
            "goal_x": goal["x"],
            "goal_y": goal["y"],
            "goal_theta": goal["theta"],
            "target_handle": scene_obj.handle,
            "target_label": scene_obj.label,
            "target_object_id": _numeric_object_id(getattr(scene_obj, "object_id", None)),
            "viewpoint_index": vp_idx,
            "total_viewpoints": total_vps,
            "viewpoint_catalog": catalog,
            "navigation_version": "se2-viewpoint-alignment/v1",
            "is_exploration": getattr(scene_obj, "is_exploration", False),
        }
        return prepared, None

    def build_subtree(
        self,
        params: Dict[str, Any],
        context: SkillContext,
        handle: SkillHandle,
    ) -> py_trees.behaviour.Behaviour:
        plan_and_execute = py_trees.composites.Sequence(
            name="navigate_plan_and_execute",
            memory=True,
            children=[
                _ValidateGoal(params, handle, self._path_planner),
                _PlanPath(
                    params,
                    handle,
                    self._path_planner,
                    self._local_planner,
                    context,
                    record_robot_xy=self._confirm_gate.enabled,
                ),
                _ConfirmPath(params, handle, self._confirm_gate, self._local_planner, _set_brake),
                _FollowPath(params, handle, self._path_planner, self._local_planner),
                _ConfirmArrival(params, handle, self._local_planner),
            ],
        )
        # memory=True so the shortcut is a one-shot startup check.  Without
        # memory the Selector would re-tick _CheckAlreadyAtGoal every tick
        # and could preempt an active FollowPath if the robot momentarily
        # stops close to the goal.
        arrival = py_trees.composites.Selector(
            name="navigate",
            memory=True,
            children=[
                _CheckAlreadyAtGoal(params, handle, self._local_planner, context),
                plan_and_execute,
            ],
        )
        # A nearby alternate can require a large turn even when its XY is
        # already reached. Align after either arrival path, exactly once.
        body = py_trees.composites.Sequence(
            name="navigate_with_heading",
            memory=True,
            children=[arrival, _AlignHeading(params, handle, self._local_planner)],
        )
        # Terminal discipline wraps the WHOLE body: individual leaves can
        # only cancel the paths they know about, and several terminal paths
        # (already-at-goal, invalid goal, no path) never reached a cancel at
        # all. See the module section above.
        return _TerminalDiscipline(
            child=body,
            local_planner=self._local_planner,
            release_brake_on_initialise=not self._confirm_gate.enabled,
        )


# ---------------------------------------------------------------------------
# BT Nodes
# ---------------------------------------------------------------------------


# Reuse execution tolerances, not task-outcome-tuned diversity thresholds.
_ALT_VIEWPOINT_DISTANCE = _ARRIVAL_DISTANCE


class _CheckAlreadyAtGoal(py_trees.behaviour.Behaviour):
    """Shortcut when the robot is already at the goal AND stationary.

    Returning SUCCESS short-circuits the parent Selector, skipping plan/
    follow/confirm entirely.  Returning FAILURE lets the Selector fall
    through to the full plan-and-execute branch — i.e. FAILURE here just
    means "the shortcut doesn't apply, do the normal navigate path".

    Conditions for SUCCESS:
      - Distance to goal < ``_ARRIVAL_DISTANCE`` (same gate as ``_ConfirmArrival``).
      - Robot velocity below settling thresholds (so ``_ConfirmArrival``
        wouldn't have anything to wait for anyway).

    This establishes positional arrival only. The shared continuation aligns
    the heading before navigation terminates.
    """

    def __init__(
        self,
        params: Dict,
        handle: SkillHandle,
        local_planner: LocalPlannerInterface,
        context: SkillContext,
    ) -> None:
        super().__init__(name="check_already_at_goal")
        self._params = params
        self._handle = handle
        self._local_planner = local_planner
        self._context = context

    def update(self) -> py_trees.common.Status:
        if self._handle.is_cancel_requested:
            return py_trees.common.Status.FAILURE

        # Stale pose ⇒ can't trust "already at goal". Fall through to the
        # full plan-and-execute branch rather than report a shortcut arrival.
        if _pose_is_stale(self._local_planner):
            return py_trees.common.Status.FAILURE

        goal_x = self._params["goal_x"]
        goal_y = self._params["goal_y"]
        rx, ry, _ = self._local_planner.get_robot_pose()
        dist = math.hypot(goal_x - rx, goal_y - ry)

        # Not at goal → fall through to full navigation
        if dist >= _ARRIVAL_DISTANCE:
            return py_trees.common.Status.FAILURE

        # At goal but still moving → let _ConfirmArrival do the settling
        linear, angular = self._local_planner.get_robot_velocity()
        if linear > _SETTLE_LINEAR or angular > _SETTLE_ANGULAR:
            return py_trees.common.Status.FAILURE

        label = self._params.get("target_label", "target")
        vp_idx = self._params.get("viewpoint_index", 0)

        obs = f"Already at '{label}' ({dist:.2f}m from goal, raw viewpoint {vp_idx})."

        # >>> NAV-DEBUG
        _nav_debug_log({
            "stage": "check_already_at_goal",
            "ts": time.monotonic(),
            "label": label,
            "dist_to_goal": round(dist, 3),
            "linear_vel": round(linear, 3),
            "angular_vel": round(angular, 3),
            "vp_idx": vp_idx,
            "distinct_viewpoint_indices": self._params.get("viewpoint_catalog", {}).get("distinct_indices", []),
        })
        # <<< NAV-DEBUG

        self._handle.observation = obs
        self._handle.progress = 1.0
        self._handle.result_data = _build_result(self._params, "already_at_goal")
        return py_trees.common.Status.SUCCESS


class _ValidateGoal(py_trees.behaviour.Behaviour):
    """Check that goal coordinates are valid and the path planner has a map."""

    def __init__(self, params: Dict, handle: SkillHandle, planner: PathPlanner) -> None:
        super().__init__(name="validate_goal")
        self._params = params
        self._handle = handle
        self._planner = planner

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

        self._handle.observation = f"Goal validated for '{label}'."
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
        record_robot_xy: bool = False,
    ) -> None:
        super().__init__(name="plan_path")
        self._params = params
        self._handle = handle
        self._planner = planner
        self._local_planner = local_planner
        self._context = context
        self._record_robot_xy = record_robot_xy

    def update(self) -> py_trees.common.Status:
        if self._handle.is_cancel_requested:
            return py_trees.common.Status.FAILURE

        label = self._params.get("target_label", "target")

        # Update map data if available
        nav_graph = self._context.platform_state.get("scene_graph")  # TODO: use dedicated nav_graph field
        if nav_graph is not None:
            self._planner.update_map(nav_graph)

        # Get current robot position
        rx, ry, _ = self._local_planner.get_robot_pose()
        if self._record_robot_xy:
            self._params["_plan_robot_xy"] = (rx, ry)

        # Plan path
        self._handle.observation = f"Planning path to '{label}'..."
        # Interactable keep-out (NavGraphPlanner.set_keep_out): name this leg's
        # target so its own floor trigger is not routed around.
        set_target = getattr(self._planner, "set_active_target", None)
        if set_target is not None:
            set_target(self._params.get("target_handle"))
        waypoints = self._plan_preferring_keep_out_clean_viewpoint((rx, ry))

        if waypoints is None or len(waypoints) == 0:
            self._handle.observation = f"No path to '{label}'. Target may be unreachable from current position."
            self._handle.result_data = _build_result(self._params, "failed_no_path")
            return py_trees.common.Status.FAILURE

        # Store waypoints for _FollowPath. ``planned_path`` is the
        # safeguard-readable alias consulted by NavigateSkill.safeguard_invariant.
        self._handle.result_data = {"_waypoints": waypoints, "planned_path": waypoints}
        # CP4: scale the wall-time budget to the actual route length now
        # that it's known (WallTimeBudgetGuard reads this fresh each tick).
        self._handle.wall_time_budget_seconds = _scaled_wall_time_budget_s(_path_length_m((rx, ry), waypoints))
        self._handle.observation = f"Path planned to '{label}': {len(waypoints)} waypoints."
        self._handle.progress = 0.05
        return py_trees.common.Status.SUCCESS

    def _plan_preferring_keep_out_clean_viewpoint(
        self, robot_xy: Tuple[float, float]
    ) -> Optional[List[Tuple[float, float]]]:
        """Plan to the requested viewpoint; if the planner's interactable
        keep-out (NavGraphPlanner.set_keep_out) leaves no clean route to it,
        try the target's other viewpoints in index order and adopt the first
        one reachable without stepping on a trigger, updating goal_x/goal_y/
        goal_theta/viewpoint_index in the shared params. Only when every
        viewpoint is blocked does the plain (keep-out-free) route win."""
        goal = (self._params["goal_x"], self._params["goal_y"])
        plan_strict = getattr(self._planner, "plan_strict", None)
        if plan_strict is None or not getattr(self._planner, "has_keep_out", False):
            return self._planner.plan(robot_xy, goal)

        waypoints = plan_strict(robot_xy, goal)
        if waypoints:
            return waypoints

        store = getattr(self._context, "object_store", None)
        scene_obj = store.get(self._params.get("target_handle", "")) if store is not None else None
        requested = int(self._params.get("viewpoint_index", 0) or 0)
        label = self._params.get("target_label", "target")
        alternatives: List[Tuple[int, Dict[str, Any]]] = []
        if scene_obj is not None:
            platform_state = self._context.platform_state or {}
            catalog = self._params.get("viewpoint_catalog") or _viewpoint_catalog(
                scene_obj, platform_state.get("camera_to_body_transforms")
            )
            for idx in catalog["distinct_indices"]:
                if idx == requested:
                    continue
                alt = compute_pose(
                    scene_obj,
                    viewpoint_index=idx,
                    robot_position=platform_state.get("robot_position"),
                    camera_to_body_transforms=platform_state.get("camera_to_body_transforms"),
                )
                alternatives.append((idx, alt))
                alt_wps = plan_strict(robot_xy, (alt["x"], alt["y"]))
                if alt_wps:
                    logger.info(
                        "Keep-out: viewpoint %d of '%s' is only reachable over a trigger; using viewpoint %d",
                        requested,
                        label,
                        idx,
                    )
                    self._adopt_viewpoint(idx, alt, f"keep-out: viewpoint {requested} -> {idx}")
                    return alt_wps
        # Every viewpoint needs the keep-out relaxed (NavGraphPlanner.plan's
        # penalised fallback). Take the viewpoint whose relaxed route stays
        # farthest from a neighbouring trigger; the requested one wins ties.
        clearance = getattr(self._planner, "keep_out_clearance", None)
        best_wps = self._planner.plan(robot_xy, goal)
        if clearance is None or not alternatives:
            return best_wps
        best = (clearance(best_wps) if best_wps else -1.0, requested, None)
        for idx, alt in alternatives:
            wps = self._planner.plan(robot_xy, (alt["x"], alt["y"]))
            if not wps:
                continue
            score = clearance(wps)
            if score > best[0] + 1e-6:
                best = (score, idx, alt)
                best_wps = wps
        if best[2] is not None:
            logger.info(
                "Keep-out: no clean route to any viewpoint of '%s'; viewpoint %d keeps %.2f m from the nearest other"
                " trigger",
                label,
                best[1],
                best[0],
            )
            self._adopt_viewpoint(best[1], best[2], f"keep-out clearance: viewpoint {requested} -> {best[1]}")
        return best_wps

    def _adopt_viewpoint(self, idx: int, pose: Dict[str, Any], note: str) -> None:
        self._params["goal_x"] = pose["x"]
        self._params["goal_y"] = pose["y"]
        self._params["goal_theta"] = pose["theta"]
        self._params["viewpoint_index"] = idx
        self._params["viewpoint_fallback"] = note


def _waypoint_heading(
    waypoints: List[Tuple[float, float]],
    index: int,
    goal_theta: float,
) -> float:
    """Match FollowPath's heading for a waypoint at ``index``."""
    wx, wy = waypoints[index]
    if index >= len(waypoints) - 1:
        return goal_theta
    nx_wp, ny_wp = waypoints[index + 1]
    if nx_wp == wx and ny_wp == wy:
        return goal_theta
    return math.atan2(ny_wp - wy, nx_wp - wx)


class _ConfirmPath(py_trees.behaviour.Behaviour):
    """Wait for one operator decision after the initial plan and before motion.

    Version 1 intentionally confirms only the initial path. If ``_FollowPath``
    later replaces it during stuck-waypoint recovery, that replan does not
    return through this behaviour and therefore does not re-fire the gate.
    """

    def __init__(
        self,
        params: Dict,
        handle: SkillHandle,
        confirm_gate: ConfirmGate,
        local_planner: LocalPlannerInterface,
        brake_setter: Callable[[LocalPlannerInterface, bool], None] = _set_brake,
    ) -> None:
        super().__init__(name="confirm_path")
        self._params = params
        self._handle = handle
        self._gate = confirm_gate
        self._local_planner = local_planner
        self._brake_setter = brake_setter
        self._timeout_s = _nav_confirm_timeout_s()

    def initialise(self) -> None:
        self._published = False
        self._cleared = False
        self._start: Optional[float] = None
        self._budget_at_start = self._handle.wall_time_budget_seconds or 0.0

    def update(self) -> py_trees.common.Status:
        if not self._gate.enabled:
            return py_trees.common.Status.SUCCESS

        label = self._params.get("target_label", "target")
        if not self._published:
            waypoints = self._handle.result_data.get("_waypoints", [])
            robot_pose = self._params.get("_plan_robot_xy", (0.0, 0.0))
            robot_xy = (float(robot_pose[0]), float(robot_pose[1]))
            self._gate.publish_plan(
                waypoints,
                label,
                robot_xy,
                self._params.get("target_object_id"),
            )
            self._start = time.monotonic()
            self._published = True
            self._handle.observation = (
                f"Awaiting operator confirmation for path to '{label}' ({len(waypoints)} waypoints)."
            )
            return py_trees.common.Status.RUNNING

        decision = self._gate.poll()
        now = time.monotonic()
        wait = now - self._start
        if decision == "go":
            waypoints = self._handle.result_data["_waypoints"]
            wx, wy = waypoints[0]
            wp_theta = _waypoint_heading(waypoints, 0, self._params["goal_theta"])
            self._local_planner.send_waypoint(wx, wy, wp_theta)
            self._brake_setter(self._local_planner, False)
            self._handle.wall_time_budget_seconds = self._budget_at_start + wait
            result = self._handle.result_data
            confirm_wait = round(wait, 2)
            result["confirm_wait_s"] = confirm_wait
            self._handle.result_data = result
            self._params["_confirm_wait_s"] = confirm_wait
            self._clear()
            self._handle.observation = f"Operator confirmed path to '{label}' (waited {wait:.2f}s)."
            return py_trees.common.Status.SUCCESS

        if decision == "abort":
            return self._abort(f"Operator aborted the planned path to '{label}'.")

        if self._handle.is_cancel_requested:
            self._clear()
            return py_trees.common.Status.FAILURE

        if wait > self._timeout_s:
            timeout = f"{self._timeout_s:g}"
            return self._abort(f"Operator confirmation for path to '{label}' timed out after {timeout}s.")
        # Keep the outer WallTimeBudgetGuard from expiring while the human is
        # deciding. On confirmation the same base+wait value is finalized and
        # exposed as confirm_wait_s above (never double-count the wait).
        self._handle.wall_time_budget_seconds = self._budget_at_start + wait
        return py_trees.common.Status.RUNNING

    def _abort(self, observation: str) -> py_trees.common.Status:
        self._clear()
        self._handle.result_data = _build_result(self._params, "failed_operator_abort")
        self._handle.observation = observation
        return py_trees.common.Status.FAILURE

    def _clear(self) -> None:
        if self._published and not self._cleared:
            self._gate.clear()
            self._cleared = True

    def terminate(self, new_status: py_trees.common.Status) -> None:
        self._clear()


class _FollowPath(py_trees.behaviour.Behaviour):
    """Follow waypoints by sending them to the local planner one at a time.

    Multi-tick: returns RUNNING each tick while the robot is moving.

    For each waypoint: send it on the first tick, then poll distance on
    subsequent ticks until ``dist < threshold``.  This send-before-check
    pattern prevents skipping waypoints when the robot starts nearby.

    On per-waypoint timeout (``_WAYPOINT_TIMEOUT``), attempts to replan
    via the PathPlanner.  Returns FAILURE if replanning is exhausted.
    """

    def __init__(
        self,
        params: Dict,
        handle: SkillHandle,
        planner: PathPlanner,
        local_planner: LocalPlannerInterface,
    ) -> None:
        super().__init__(name="follow_path")
        self._params = params
        self._handle = handle
        self._planner = planner
        self._local_planner = local_planner

    def initialise(self) -> None:
        self._waypoints: List[Tuple[float, float]] = self._handle.result_data.get("_waypoints", [])
        self._idx = 0
        self._sent = False
        self._wp_start = time.monotonic()
        self._replan_attempts = 0
        self._backoffs = 0
        self._nudges = 0
        self._backoff_until: Optional[float] = None
        self._backoff_wz = 0.0
        self._sent_xy: Optional[Tuple[float, float]] = None
        self._reset_progress_tracking()
        # Best distance-to-GOAL seen across the episode — deliberately NOT
        # distance-to-current-waypoint, which changes identity on every
        # replan and would let a merely-different (e.g. ping-ponging) path
        # masquerade as progress. Used to decide whether a replan resets
        # the recovery budget below.
        goal_x, goal_y = self._params.get("goal_x", 0.0), self._params.get("goal_y", 0.0)
        rx, ry, _ = self._local_planner.get_robot_pose()
        self._best_dist_to_goal = math.hypot(goal_x - rx, goal_y - ry)

    def _reset_progress_tracking(self) -> None:
        """Reset the no-progress guard's best-distance tracking."""
        self._best_dist = math.inf
        self._best_dist_ts = time.monotonic()

    def update(self) -> py_trees.common.Status:
        if self._handle.is_cancel_requested:
            self._local_planner.cancel()
            return py_trees.common.Status.FAILURE

        if not self._waypoints:
            return py_trees.common.Status.SUCCESS

        if self._backoff_until is not None:
            if time.monotonic() < self._backoff_until:
                self._local_planner.send_velocity(-_BACKOFF_SPEED_MPS, 0.0, self._backoff_wz)
                return py_trees.common.Status.RUNNING
            # Done backing off: stop the primitive input (the mux falls back
            # to the follower) and re-send the current waypoint.
            self._local_planner.send_velocity(0.0, 0.0, 0.0)
            self._backoff_until = None
            self._sent = False

        label = self._params.get("target_label", "target")
        goal_theta = self._params["goal_theta"]

        wx, wy = self._waypoints[self._idx]
        # Tighter arrival threshold next to a floor trigger (NavGraphPlanner
        # .waypoint_threshold_at); planners without keep-out keep the default.
        threshold_at = getattr(self._planner, "waypoint_threshold_at", None)
        if threshold_at is None or self._idx == 0:
            threshold = self._planner.waypoint_threshold
        else:
            threshold = threshold_at((wx, wy))
        wp_theta = self._compute_theta(wx, wy, goal_theta)

        # Send waypoint before checking arrival (prevents first-tick skip)
        if not self._sent:
            self._local_planner.send_waypoint(wx, wy, wp_theta)
            self._sent = True
            self._wp_start = time.monotonic()
            self._reset_progress_tracking()
            sx, sy, _ = self._local_planner.get_robot_pose()
            self._sent_xy = (sx, sy)
            return py_trees.common.Status.RUNNING

        # Check arrival at current waypoint
        rx, ry, _ = self._local_planner.get_robot_pose()
        dist = math.hypot(wx - rx, wy - ry)

        if dist < threshold:
            self._idx += 1
            if self._idx >= len(self._waypoints):
                return py_trees.common.Status.SUCCESS
            self._sent = False  # trigger send on next tick
            return py_trees.common.Status.RUNNING

        # No-progress tracking: record the best (smallest) distance seen to
        # the current waypoint. If it hasn't improved by more than
        # _NO_PROGRESS_EPS_M within _NO_PROGRESS_WINDOW_S, treat it as stuck
        # even before the fixed per-waypoint timeout elapses.
        if dist < self._best_dist - _NO_PROGRESS_EPS_M:
            self._best_dist = dist
            self._best_dist_ts = time.monotonic()

        now = time.monotonic()
        elapsed = now - self._wp_start
        # Both windows are wall clock: scale them like every other wall
        # budget when the simulator runs slower than real time.
        timed_out = elapsed > _WAYPOINT_TIMEOUT * WALL_BUDGET_SCALE
        no_progress = (now - self._best_dist_ts) > _NO_PROGRESS_WINDOW_S * WALL_BUDGET_SCALE

        # Stuck (timeout OR no-progress) → replan, budgeted
        if timed_out or no_progress:
            logger.warning(
                "Waypoint %d/%d stuck (%s, %.1fs) for '%s'",
                self._idx + 1,
                len(self._waypoints),
                "timeout" if timed_out else "no-progress",
                elapsed if timed_out else (now - self._best_dist_ts),
                label,
            )
            if (
                NAV_BACKOFF_ENABLED
                and self._backoffs < _MAX_BACKOFFS_PER_LEG
                and self._sent_xy is not None
                and math.hypot(rx - self._sent_xy[0], ry - self._sent_xy[1]) < _PINNED_EPS_M
            ):
                self._backoffs += 1
                self._backoff_wz = 0.0 if self._backoffs % 2 else _BACKOFF_TURN_RAD_S
                self._backoff_until = time.monotonic() + _BACKOFF_S * WALL_BUDGET_SCALE
                logger.warning(
                    "Robot pinned at waypoint %d/%d for '%s' (moved < %.2f m since the send); "
                    "backing off %.1fs (attempt %d/%d)",
                    self._idx + 1,
                    len(self._waypoints),
                    label,
                    _PINNED_EPS_M,
                    _BACKOFF_S * WALL_BUDGET_SCALE,
                    self._backoffs,
                    _MAX_BACKOFFS_PER_LEG,
                )
                self._wp_start = time.monotonic()
                self._reset_progress_tracking()
                self._local_planner.send_velocity(-_BACKOFF_SPEED_MPS, 0.0, self._backoff_wz)
                return py_trees.common.Status.RUNNING

            if (
                NAV_BACKOFF_ENABLED
                and NAV_NUDGE_FN is not None
                and self._backoffs >= _MAX_BACKOFFS_PER_LEG
                and self._nudges < _MAX_NUDGES_PER_LEG
                and self._sent_xy is not None
                and math.hypot(rx - self._sent_xy[0], ry - self._sent_xy[1]) < _PINNED_EPS_M
            ):
                self._nudges += 1
                logger.warning(
                    "Robot still pinned at waypoint %d/%d for '%s' after %d back-offs; "
                    "requesting a simulator nudge (%d/%d)",
                    self._idx + 1,
                    len(self._waypoints),
                    label,
                    self._backoffs,
                    self._nudges,
                    _MAX_NUDGES_PER_LEG,
                )
                try:
                    NAV_NUDGE_FN()
                except Exception:  # noqa: BLE001 -- a failed rescue call must not kill the leg
                    logger.exception("nudge request failed")
                # The route itself ran into the obstacle (Isaac pass 4: the
                # nudged robot walked straight back into the wedge). Drop the
                # edge at the robot and re-route (replan() removes the nearest
                # edge); the replan budget is not charged for the rescue.
                goal_x, goal_y = self._params["goal_x"], self._params["goal_y"]
                rerouted = self._planner.replan((rx, ry), (goal_x, goal_y))
                if rerouted and list(rerouted) != list(self._waypoints):
                    self._waypoints = rerouted
                    self._idx = 0
                    logger.info("Re-routed after the nudge: %d waypoints for '%s'", len(rerouted), label)
                self._wp_start = time.monotonic()
                self._reset_progress_tracking()
                self._sent = False  # re-send the waypoint from the new pose
                return py_trees.common.Status.RUNNING

            # Reset the waypoint timer + no-progress tracker unconditionally
            # on entry to the stuck branch — whichever way it resolves below
            # (adopted path, genuine progress, or neither). A reset scoped to
            # only some exit paths (e.g. only when a new path is adopted)
            # leaves the same_path/no_path + genuine-progress combination
            # unreset: the next tick would find elapsed still > the timeout
            # and re-enter this branch immediately, replanning and spending
            # budget at tick rate instead of ~_WAYPOINT_TIMEOUT cadence.
            self._wp_start = time.monotonic()
            self._reset_progress_tracking()
            self._replan_attempts += 1
            goal_x, goal_y = self._params["goal_x"], self._params["goal_y"]
            new_waypoints = self._planner.replan((rx, ry), (goal_x, goal_y))
            same_path = new_waypoints is not None and list(new_waypoints) == list(self._waypoints)
            got_new_path = bool(new_waypoints) and not same_path

            # CP4: any successful replan (whether it returns a new path or
            # re-confirms the same one) knows a fresh route length from the
            # robot's current position — rescale the wall-time budget to it.
            if new_waypoints:
                self._handle.wall_time_budget_seconds = _scaled_wall_time_budget_s(
                    _path_length_m((rx, ry), new_waypoints)
                )

            if got_new_path:
                self._waypoints = new_waypoints
                self._idx = 0
                self._sent = False
                logger.info("Replanned: %d new waypoints for '%s'", len(new_waypoints), label)

            # Genuine progress only: has the robot's actual distance to the
            # GOAL improved by more than eps since the budget was last reset?
            # A merely-different path is NOT progress by itself — that's the
            # A->B->A ping-pong trap (odometry noise flipping the nearest
            # graph node), which would otherwise reset the N=3 budget forever
            # and reproduce the 120s-spin defect this budget exists to kill.
            # The baseline itself only ratchets down on a >eps improvement
            # (mirroring ``_best_dist`` above) — updating on ANY improvement
            # would let sub-eps creep lower the bar every cycle and mask a
            # slowly-but-truly advancing robot from ever registering progress.
            dist_to_goal = math.hypot(goal_x - rx, goal_y - ry)
            genuine_progress = dist_to_goal < self._best_dist_to_goal - _NO_PROGRESS_EPS_M
            if genuine_progress:
                self._best_dist_to_goal = dist_to_goal
                self._replan_attempts = 0
                return py_trees.common.Status.RUNNING

            # No genuine progress — counts against the recovery budget,
            # whether the replan returned an identical path, no path at all,
            # or even a different-but-unproven one.
            if self._replan_attempts >= _MAX_REPLAN_ATTEMPTS:
                no_path = not new_waypoints
                status_code = "failed_no_path" if no_path else "failed_blocked"
                if no_path:
                    self._handle.observation = (
                        f"[leaf=FollowPath] Navigation to '{label}' failed: no path found "
                        f"after {self._replan_attempts} replan attempts. Target may be "
                        f"unreachable from waypoint {self._idx + 1}/{len(self._waypoints)}."
                    )
                else:
                    self._handle.observation = (
                        f"[leaf=FollowPath] Navigation to '{label}' blocked after "
                        f"{self._replan_attempts} replan attempts. "
                        f"Stuck at waypoint {self._idx + 1}/{len(self._waypoints)}."
                    )
                self._handle.result_data = _build_result(self._params, status_code)
                return py_trees.common.Status.FAILURE

            if same_path:
                progress_desc = "identical path"
            elif not new_waypoints:
                progress_desc = "no path"
            else:
                progress_desc = "different path, no distance progress"
            logger.warning(
                "Replan attempt %d/%d made no progress for '%s' (%s)",
                self._replan_attempts,
                _MAX_REPLAN_ATTEMPTS,
                label,
                progress_desc,
            )
            return py_trees.common.Status.RUNNING

        # Re-send waypoint (local planner may need periodic updates)
        self._local_planner.send_waypoint(wx, wy, wp_theta)

        progress = self._idx / max(len(self._waypoints), 1)
        self._handle.progress = 0.1 + 0.85 * progress
        self._handle.observation = (
            f"Navigating to '{label}'... (waypoint {self._idx + 1}/{len(self._waypoints)}, {dist:.1f}m away)"
        )

        # >>> NAV-DEBUG
        _nav_debug_log({
            "stage": "follow_path",
            "ts": time.monotonic(),
            "label": label,
            "wp_idx": self._idx,
            "wp_total": len(self._waypoints),
            "wp_target": [wx, wy],
            "robot_xy": [rx, ry],
            "dist_to_wp": round(dist, 3),
            "threshold": threshold,
            "wp_elapsed_s": round(time.monotonic() - self._wp_start, 2),
        })
        # <<< NAV-DEBUG

        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        # Cancel on FAILURE too: the local planner latches the last waypoint
        # (which may be in a wall/obstacle) and would keep driving toward it
        # after we report FAILURE.  cancel() republishes current pose as the
        # goal, halting the robot.
        if new_status in (
            py_trees.common.Status.INVALID,
            py_trees.common.Status.FAILURE,
        ):
            self._local_planner.cancel()

    def _compute_theta(self, wx: float, wy: float, goal_theta: float) -> float:
        """Face direction of travel, or goal yaw for the last waypoint."""
        return _waypoint_heading(self._waypoints, self._idx, goal_theta)


# ---------------------------------------------------------------------------
# Arrival confirmation — velocity-based settling
# ---------------------------------------------------------------------------

_SETTLE_LINEAR = 0.1  # m/s — robot is "stopped" below this
_SETTLE_ANGULAR = 0.05  # rad/s — robot has finished rotating below this
_SETTLE_DURATION = 1.5  # seconds robot must be continuously stopped
_SETTLE_TIMEOUT = 15.0  # max wait before accepting current position
_MOVEMENT_THRESHOLD = 0.05  # m/s — above this counts as "saw movement"


_ALIGN_TOL_RAD = 0.15
_ALIGN_KP = 1.0
_ALIGN_MAX_W = 0.5  # rad/s
_ALIGN_TIMEOUT_S = 10.0


class _AlignHeading(py_trees.behaviour.Behaviour):
    """Rotate in place to the viewpoint's heading (``goal_theta``) after
    arrival. The path follower stops with whatever heading the last leg had;
    a viewpoint is a (position, heading) pair and small targets are only
    framed by the camera ring at that heading. Timeout remains best-effort
    success with explicit residual metadata; cancellation/stale pose fail safely.
    The positional arrival verdict is preserved separately."""

    def __init__(self, params: Dict, handle: SkillHandle, local_planner: LocalPlannerInterface) -> None:
        super().__init__(name="align_heading")
        self._params = params
        self._handle = handle
        self._local_planner = local_planner

    def initialise(self) -> None:
        self._start = time.monotonic()
        self._arrival_observation = self._handle.observation
        self._arrival_result = dict(self._handle.result_data or {})
        self._finished = False

    def _finish(self, reason="aligned", status=py_trees.common.Status.SUCCESS) -> py_trees.common.Status:
        self._local_planner.send_velocity(0.0, 0.0, 0.0)
        self._local_planner.cancel()
        self._finished = True
        try:
            x, y, yaw = (float(v) for v in self._local_planner.get_robot_pose())
            finite_pose = all(math.isfinite(v) for v in (x, y, yaw))
        except Exception:
            finite_pose = False
        goal = _finite_float(self._params.get("goal_theta"))
        residual = _yaw_difference(goal, yaw) if finite_pose and goal is not None else None
        metadata = {
            "status": reason,
            "timed_out": reason == "timeout",
            "elapsed_s": max(0.0, time.monotonic() - self._start),
            "goal_yaw_rad": goal,
            "achieved_yaw_rad": float(yaw) if finite_pose else None,
            "residual_yaw_rad": residual,
            "tolerance_rad": _ALIGN_TOL_RAD,
            "within_tolerance": residual is not None and abs(residual) <= _ALIGN_TOL_RAD,
            "pose_fresh": not _pose_is_stale(self._local_planner),
        }
        self._handle.observation = self._arrival_observation
        if "viewpoint_catalog" in self._params:
            indices = self._params["viewpoint_catalog"]["distinct_indices"]
            self._handle.observation += (
                f" Distinct recorded viewpoints (raw indices): {indices}."
                " Use viewpoint_index=<raw index> to select a recorded view."
                if indices
                else " No valid full-pose alternative list is available."
            )
        if reason != "aligned":
            self._handle.observation += f" Heading alignment: {reason}; residual={residual}."
        self._handle.result_data = dict(self._arrival_result, heading_alignment=metadata)
        _nav_debug_log(
            dict(
                stage="align_heading_terminal", ts=time.monotonic(), label=self._params.get("target_label"), **metadata
            )
        )
        return status

    def update(self) -> py_trees.common.Status:
        if self._handle.is_cancel_requested:
            return self._finish("cancelled", py_trees.common.Status.FAILURE)
        if _pose_is_stale(self._local_planner):
            return self._finish("stale_pose", py_trees.common.Status.FAILURE)
        goal_theta = _finite_float(self._params.get("goal_theta"))
        try:
            raw_pose = self._local_planner.get_robot_pose()
            valid_pose = len(raw_pose) == 3 and all(_finite_float(v) is not None for v in raw_pose)
            yaw = _finite_float(raw_pose[2]) if valid_pose else None
        except Exception:
            yaw = None
        if yaw is None or goal_theta is None:
            return self._finish("invalid_pose", py_trees.common.Status.FAILURE)
        err = (float(goal_theta) - yaw + math.pi) % (2 * math.pi) - math.pi
        if abs(err) <= _ALIGN_TOL_RAD:
            return self._finish()
        if time.monotonic() - self._start > _ALIGN_TIMEOUT_S:
            logger.warning("Heading alignment timed out (%.2f rad off)", err)
            return self._finish("timeout")
        wz = max(-_ALIGN_MAX_W, min(_ALIGN_MAX_W, _ALIGN_KP * err))
        self._local_planner.send_velocity(0.0, 0.0, wz)
        return py_trees.common.Status.RUNNING

    def terminate(self, new_status: py_trees.common.Status) -> None:
        if new_status == py_trees.common.Status.INVALID:
            if hasattr(self, "_start") and not self._finished:
                self._finish("interrupted", py_trees.common.Status.FAILURE)
            else:
                self._local_planner.send_velocity(0.0, 0.0, 0.0)
                self._local_planner.cancel()


class _ConfirmArrival(py_trees.behaviour.Behaviour):
    """Wait for the robot to physically stop, then confirm arrival.

    After ``_FollowPath`` completes (last waypoint reached by distance),
    the robot may still be decelerating or rotating to the final yaw.

    Two phases:

    1. **Wait for movement** — the local planner has startup latency
       (~0.5-1s). Don't start the settling clock until velocity exceeds
       ``_MOVEMENT_THRESHOLD``.
    2. **Wait for settling** — once movement was observed, wait for
       velocity to drop below thresholds for ``_SETTLE_DURATION``.

    If the timeout fires without movement → FAILURE (local planner
    didn't respond). If the timeout fires after movement → force accept.
    """

    def __init__(
        self,
        params: Dict,
        handle: SkillHandle,
        local_planner: LocalPlannerInterface,
    ) -> None:
        super().__init__(name="confirm_arrival")
        self._params = params
        self._handle = handle
        self._local_planner = local_planner

    def initialise(self) -> None:
        self._start = time.monotonic()
        self._stopped_since: Optional[float] = None
        self._saw_movement = False

    def update(self) -> py_trees.common.Status:
        label = self._params.get("target_label", "target")
        now = time.monotonic()
        elapsed = now - self._start

        linear, angular = self._local_planner.get_robot_velocity()
        rx, ry, _ = self._local_planner.get_robot_pose()
        dist_to_goal = math.hypot(self._params["goal_x"] - rx, self._params["goal_y"] - ry)
        is_settled = linear < _SETTLE_LINEAR and angular < _SETTLE_ANGULAR

        # >>> NAV-DEBUG
        _nav_debug_log({
            "stage": "confirm_arrival",
            "ts": time.monotonic(),
            "label": label,
            "linear_vel": round(linear, 3),
            "angular_vel": round(angular, 3),
            "dist_to_goal": round(dist_to_goal, 3),
            "is_settled": is_settled,
            "saw_movement": self._saw_movement,
            "elapsed_s": round(elapsed, 2),
        })
        # <<< NAV-DEBUG

        # Track whether the robot has started moving
        if not self._saw_movement and (linear > _MOVEMENT_THRESHOLD or angular > _MOVEMENT_THRESHOLD):
            self._saw_movement = True
            logger.info("Movement detected for '%s' (v=%.2f, ω=%.2f)", label, linear, angular)

        # Timeout — behavior depends on whether movement was seen
        if elapsed > _SETTLE_TIMEOUT and not self._saw_movement:
            logger.warning("Settle timeout for '%s' — never saw movement", label)
            self._handle.observation = (
                f"[leaf=ConfirmArrival] Navigation to '{label}' failed: "
                f"robot never started moving (waited {elapsed:.1f}s)."
            )
            self._handle.result_data = _build_result(self._params, "failed_no_movement")
            return py_trees.common.Status.FAILURE
        if elapsed > _SETTLE_TIMEOUT:
            logger.warning("Settle timeout for '%s', accepting current position", label)
            return self._accept(label, suffix="(settle timeout)", force=True)

        # Phase 1: wait for movement before starting settling clock
        if not self._saw_movement:
            self._handle.observation = f"Waiting for robot to start moving toward '{label}'... ({elapsed:.1f}s)"
            return py_trees.common.Status.RUNNING

        # Phase 2: wait for settling
        if is_settled:
            if self._stopped_since is None:
                self._stopped_since = now
            elif now - self._stopped_since >= _SETTLE_DURATION:
                return self._accept(label)
        else:
            self._stopped_since = None

        self._handle.observation = (
            f"Waiting for robot to settle at '{label}'... (v={linear:.2f} m/s, \u03c9={angular:.2f} rad/s)"
        )
        return py_trees.common.Status.RUNNING

    def _accept(self, label: str, suffix: str = "", force: bool = False) -> py_trees.common.Status:
        # Stale pose ⇒ can't trust any arrival judgement — never report
        # "arrived", even on forced (settle-timeout) accept.
        pose_age = _pose_age(self._local_planner)
        if pose_age is not None and pose_age > _POSE_STALENESS_S:
            logger.warning(
                "Refusing arrival for '%s' — pose stale (age=%.2fs)",
                label,
                pose_age,
            )
            self._stopped_since = None
            return py_trees.common.Status.RUNNING

        rx, ry, _ = self._local_planner.get_robot_pose()
        dist = math.hypot(self._params["goal_x"] - rx, self._params["goal_y"] - ry)

        # Gate: robot must be within _ARRIVAL_DISTANCE of the goal.
        # ``force=True`` bypasses the gate (used on settle timeout) — but
        # bypassing the gate is exactly the "dishonest short arrival" case,
        # so the resulting status below reflects the honest distance.
        if not force and dist > _ARRIVAL_DISTANCE:
            logger.info(
                "Settled but %.2fm from goal (need <%.2fm) for '%s', waiting",
                dist,
                _ARRIVAL_DISTANCE,
                label,
            )
            self._stopped_since = None  # reset settling clock
            return py_trees.common.Status.RUNNING

        # Halt the local planner at current pose before reporting SUCCESS.
        # The last waypoint may be in a wall/obstacle; without this cancel
        # the planner keeps pushing the robot toward it and it slides around
        # the goal after we've reported "arrived".
        self._local_planner.cancel()

        # ``arrived`` implies within-threshold. A forced accept that's still
        # beyond the threshold is an honest short arrival, not a silent one.
        short = dist > _ARRIVAL_DISTANCE
        status = "arrived_short" if short else "arrived"

        obs = f"Arrived short of '{label}' ({dist:.2f}m from goal)." if short else f"Arrived at '{label}'."
        if suffix:
            obs += f" {suffix}"
        self._handle.observation = obs
        self._handle.progress = 1.0
        result = _build_result(self._params, status)
        if short:
            result["residual_distance_m"] = round(dist, 3)
        self._handle.result_data = result
        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status: py_trees.common.Status) -> None:
        # Stop the local planner on failure paths so the robot doesn't keep
        # chasing the last waypoint after we report FAILURE.  SUCCESS already
        # calls cancel() inside _accept().
        if new_status in (
            py_trees.common.Status.INVALID,
            py_trees.common.Status.FAILURE,
        ):
            self._local_planner.cancel()


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _yaw_difference(a: float, b: float) -> float:
    return (a - b + math.pi) % (2 * math.pi) - math.pi


def _finite_float(value):
    try:
        number = float(value)
        return number if math.isfinite(number) else None
    except (TypeError, ValueError, OverflowError):
        return None


def _same_se2_view(a: Dict[str, float], b: Dict[str, float]) -> bool:
    return (
        math.hypot(a["x"] - b["x"], a["y"] - b["y"]) < _ARRIVAL_DISTANCE
        and abs(_yaw_difference(a["theta"], b["theta"])) <= _ALIGN_TOL_RAD
    )


def _viewpoint_catalog(scene_obj: Any, camera_transforms=None) -> Dict[str, Any]:
    """Stable raw indices, deduplicated in executable body SE2, never by camera name."""
    poses = []
    invalid = []
    representatives = []
    for index, vp in enumerate(getattr(scene_obj, "all_viewpoints", [])):
        try:
            if len(vp) < 7 or not all(math.isfinite(float(v)) for v in vp[:7]):
                raise ValueError("requires finite XYZ and quaternion")
            if sum(float(v) ** 2 for v in vp[3:7]) <= 0:
                raise ValueError("zero quaternion")
            goal = compute_pose(scene_obj, viewpoint_index=index, camera_to_body_transforms=camera_transforms)
            if not all(math.isfinite(float(goal[k])) for k in ("x", "y", "theta")):
                raise ValueError("nonfinite effective goal")
        except (TypeError, ValueError, IndexError, KeyError) as exc:
            invalid.append(dict(raw_index=index, reason=str(exc)))
            continue
        poses.append(dict(raw_index=index, **goal))
        if not any(_same_se2_view(goal, representative) for representative in representatives):
            representatives.append(poses[-1])
    if invalid:
        logger.warning("Ignored malformed recorded viewpoints for '%s': %s", getattr(scene_obj, "label", ""), invalid)
    return dict(
        distinct_indices=[p["raw_index"] for p in representatives],
        valid_indices=[p["raw_index"] for p in poses],
        effective_goals=poses,
        invalid_entries=invalid,
        translation_tolerance_m=_ARRIVAL_DISTANCE,
        heading_tolerance_rad=_ALIGN_TOL_RAD,
    )


def _build_result(params: Dict[str, Any], arrival_status: str) -> Dict[str, Any]:
    result = {
        "arrival_status": arrival_status,
        "target_handle": params.get("target_handle", ""),
        "target_label": params.get("target_label", ""),
        "viewpoint_index": params.get("viewpoint_index", 0),
        "total_viewpoints": params.get("total_viewpoints", 1),
        "is_exploration": params.get("is_exploration", False),
        "goal": {
            "x": params.get("goal_x", 0.0),
            "y": params.get("goal_y", 0.0),
            "theta": params.get("goal_theta", 0.0),
        },
    }
    if "_confirm_wait_s" in params:
        result["confirm_wait_s"] = params["_confirm_wait_s"]
    if "viewpoint_catalog" in params:
        result["viewpoint_catalog"] = params["viewpoint_catalog"]
        result["navigation_version"] = params.get("navigation_version")
    return result
