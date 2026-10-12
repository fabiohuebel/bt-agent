"""SE2 goal computation from store object viewpoints.

Ported from lam_core navigator.py. Pure Python, no ROS2.
Duck-typed: reads ``viewpoint_position``, ``viewpoint_orientation``,
``all_viewpoints``, ``position`` etc. via attribute access on whatever
SkillObject the caller provides.
"""

from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Tuple

from bt_agent.skills import SkillObject


def compute_pose(
    target: SkillObject,
    viewpoint_index: int = 0,
    robot_position: Optional[Tuple[float, float, float]] = None,
    camera_to_body_transforms: Optional[Dict[str, Any]] = None,
) -> Dict[str, float]:
    """Compute an SE2 navigation goal {x, y, theta} from a store object.

    Expects the target to have ``all_viewpoints``, ``position``, and
    optionally ``viewpoint_position`` / ``viewpoint_orientation`` /
    ``viewpoint_camera_id`` as legacy fallbacks — set by the producing
    skill (QuerySkill).  Fields are accessed via getattr with sensible
    defaults so this function works with any SkillObject subclass that
    provides the expected attributes.

    Args:
        target: SkillObject with position/viewpoint data (duck-typed).
        viewpoint_index: Index into ``all_viewpoints`` (0-based).  Each
            index identifies a distinct vantage; there is no "primary" /
            "best" — callers pick by position in the list.  Out-of-bounds
            falls back to the legacy ``viewpoint_position`` field.
        robot_position: Current robot pose (x, y, yaw). Used as fallback.
        camera_to_body_transforms: Dict of camera_name -> 4x4 SE3 ndarray.

    Returns:
        {"x": float, "y": float, "theta": float} in map frame.
    """
    all_viewpoints = getattr(target, "all_viewpoints", [])
    all_vp_camera_ids = getattr(target, "all_viewpoint_camera_ids", [])

    # ``viewpoint_index`` indexes directly into ``all_viewpoints``.  No
    # special case for index 0 — there is no notion of a "best" viewpoint;
    # callers pick the viewpoint they want by position in the list.
    # ``viewpoint_position`` / ``viewpoint_orientation`` / ``viewpoint_camera_id``
    # remain as legacy fallbacks for objects with an empty all_viewpoints list
    # or when the requested index is out of bounds.
    if 0 <= viewpoint_index < len(all_viewpoints):
        vp = all_viewpoints[viewpoint_index]
        vp_pos = {"x": vp[0], "y": vp[1], "z": vp[2] if len(vp) > 2 else 0.0}
        vp_orient = list(vp[3:7]) if len(vp) >= 7 else []
        camera_id = all_vp_camera_ids[viewpoint_index] if viewpoint_index < len(all_vp_camera_ids) else ""
    else:
        vp_pos = getattr(target, "viewpoint_position", {"x": 0.0, "y": 0.0, "z": 0.0})
        vp_orient = getattr(target, "viewpoint_orientation", [0.0, 0.0, 0.0, 1.0])
        camera_id = getattr(target, "viewpoint_camera_id", "")

    # Try viewpoint-based pose (preferred)
    if vp_pos and (vp_pos.get("x", 0) != 0 or vp_pos.get("y", 0) != 0):
        x = vp_pos["x"]
        y = vp_pos["y"]
        theta = 0.0

        if len(vp_orient) == 4:
            theta = _compute_yaw_from_viewpoint(
                vp_pos,
                vp_orient,
                camera_id,
                camera_to_body_transforms,
            )

        return {"x": x, "y": y, "theta": theta}

    # Fallback: aim at object center from robot position
    pos = getattr(target, "position", {"x": 0.0, "y": 0.0, "z": 0.0})
    tx = pos.get("x", 0.0)
    ty = pos.get("y", 0.0)

    if robot_position is not None:
        rx, ry = robot_position[0], robot_position[1]
        theta = math.atan2(ty - ry, tx - rx)
    else:
        theta = 0.0

    return {"x": tx, "y": ty, "theta": theta}


def _compute_yaw_from_viewpoint(
    vp_pos: Dict[str, float],
    quat: List[float],
    camera_id: str,
    camera_to_body_transforms: Optional[Dict[str, Any]],
) -> float:
    """Extract yaw from viewpoint quaternion, optionally transforming via camera-to-body."""
    if camera_to_body_transforms and camera_id in camera_to_body_transforms:
        try:
            import numpy as np
            from scipy.spatial.transform import Rotation

            T_camera_body = camera_to_body_transforms[camera_id]

            T_map_cam = np.eye(4)
            T_map_cam[:3, :3] = Rotation.from_quat(quat).as_matrix()
            T_map_cam[:3, 3] = [vp_pos["x"], vp_pos["y"], vp_pos.get("z", 0.0)]

            T_map_body = T_map_cam @ T_camera_body

            return math.atan2(T_map_body[1, 0], T_map_body[0, 0])
        except ImportError:
            pass  # Fall through to direct quaternion extraction

    # Direct yaw extraction from quaternion [qx, qy, qz, qw]
    qx, qy, qz, qw = quat
    siny_cosp = 2.0 * (qw * qz + qx * qy)
    cosy_cosp = 1.0 - 2.0 * (qy * qy + qz * qz)
    return math.atan2(siny_cosp, cosy_cosp)


def compute_push_target(
    obj_position: Dict[str, float],
    robot_position: Tuple[float, float, float],
    push_distance: float = 0.8,
) -> Dict[str, float]:
    """Compute a push-through target past the object.

    Returns an SE2 goal ``push_distance`` metres past the object
    from the robot's perspective.
    """
    ox = obj_position.get("x", 0.0)
    oy = obj_position.get("y", 0.0)
    rx, ry = robot_position[0], robot_position[1]

    dx = ox - rx
    dy = oy - ry
    dist = math.hypot(dx, dy)

    if dist > 0.05:
        dir_x = dx / dist
        dir_y = dy / dist
    else:
        dir_x, dir_y = 1.0, 0.0

    target_x = ox + dir_x * push_distance
    target_y = oy + dir_y * push_distance
    theta = math.atan2(dir_y, dir_x)

    return {"x": target_x, "y": target_y, "theta": theta}
