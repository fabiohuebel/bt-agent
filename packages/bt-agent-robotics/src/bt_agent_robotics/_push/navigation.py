from __future__ import annotations
import math, time, logging
import py_trees
from bt_agent_robotics.skills.navigate import (_ALIGN_TOL_RAD, _finite_float, _yaw_difference, _pose_is_stale, _nav_debug_log, _ALIGN_TIMEOUT_S, _ALIGN_MAX_W, _ALIGN_KP, _FollowPath, _TerminalDiscipline)
logger = logging.getLogger(__name__)

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

    def _tolerance(self):
        value = _finite_float(self._params.get("heading_tolerance_rad", _ALIGN_TOL_RAD))
        return value if value is not None and 0 < value <= _ALIGN_TOL_RAD else None

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
        tolerance = self._tolerance()
        metadata = {
            "status": reason,
            "timed_out": reason == "timeout",
            "elapsed_s": max(0.0, time.monotonic() - self._start),
            "goal_yaw_rad": goal,
            "achieved_yaw_rad": float(yaw) if finite_pose else None,
            "residual_yaw_rad": residual,
            "tolerance_rad": tolerance,
            "within_tolerance": tolerance is not None and residual is not None and abs(residual) <= tolerance,
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
        tolerance = self._tolerance()
        if tolerance is None:
            return self._finish("invalid_tolerance", py_trees.common.Status.FAILURE)
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
        if abs(err) <= tolerance:
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

