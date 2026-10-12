from __future__ import annotations
import math
from dataclasses import dataclass
from typing import List
import numpy as np

HAB_TO_ROS_BASIS = np.array([
    [1.0, 0.0, 0.0],
    [0.0, 0.0, -1.0],
    [0.0, 1.0, 0.0],
], dtype=float)


def _rotation_matrix_y(theta: float) -> np.ndarray:
    """Habitat active rotation about +Y by ``theta`` radians -- matches
    ``Magnum.Quaternion.rotation(Rad(theta), Vector3.y_axis()).to_matrix()``
    (verified against magnum directly; see test_matches_magnum_reference)."""
    c, s = math.cos(theta), math.sin(theta)
    return np.array([
        [c, 0.0, s],
        [0.0, 1.0, 0.0],
        [-s, 0.0, c],
    ], dtype=float)


@dataclass
class Transform:
    """Habitat -> ROS map (spawn-relative) transform, precomputed once."""
    p0: np.ndarray
    r0_inv: np.ndarray
    basis: np.ndarray = None  # type: ignore[assignment]

    def __post_init__(self):
        if self.basis is None:
            self.basis = HAB_TO_ROS_BASIS

    def point(self, p) -> List[float]:
        """Transform a habitat point -> ROS map point. Height (y) is not
        offset by the spawn, only rotated/rebased (rotation about Y leaves
        the Y-component unchanged, so height passes through untouched)."""
        rel = np.array([p[0] - self.p0[0], p[1], p[2] - self.p0[2]], dtype=float)
        rel = self.r0_inv @ rel
        ros = self.basis @ rel
        return [float(ros[0]), float(ros[1]), float(ros[2])]

    def direction(self, d) -> List[float]:
        """Transform a habitat direction vector (no spawn-position offset)."""
        v = self.r0_inv @ np.array(d, dtype=float)
        ros = self.basis @ v
        return [float(ros[0]), float(ros[1]), float(ros[2])]

    def yaw(self, yaw_hab: float) -> float:
        """Habitat ground-plane heading (``atan2(dx, dz)`` convention) ->
        ROS map yaw (radians about +Z)."""
        d = [math.sin(yaw_hab), 0.0, math.cos(yaw_hab)]
        dp = self.direction(d)
        return math.atan2(dp[1], dp[0])

    def pose(self, pos_hab, yaw_hab: float) -> List[float]:
        """Body pose ``[x, y, z, qx, qy, qz, qw]`` at a habitat position+yaw."""
        x, y, z = self.point(pos_hab)
        qx, qy, qz, qw = quat_from_yaw(self.yaw(yaw_hab))
        return [x, y, z, qx, qy, qz, qw]


def build_transform(spawn_pos, spawn_rot: float) -> Transform:
    p0 = np.array(spawn_pos, dtype=float)
    r0_inv = _rotation_matrix_y(spawn_rot).T  # rotation matrices are orthogonal
    return Transform(p0=p0, r0_inv=r0_inv)


def quat_from_yaw(yaw: float) -> List[float]:
    """Z-axis-only quaternion [qx, qy, qz, qw] for a ROS map yaw."""
    return [0.0, 0.0, math.sin(yaw / 2.0), math.cos(yaw / 2.0)]

