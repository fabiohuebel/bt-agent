"""Explicit passive driver input normalization; no ROS import or output.

Static transforms use the ROS parent<-child convention. They are immutable for
one session: a changed edge poisons that child until the application recreates
the input adapter. This is coordinate conversion, never map calibration.
"""

from dataclasses import dataclass
from types import MappingProxyType
import re

import numpy as np
from scipy.spatial.transform import Rotation

from .platform import CAMERAS


def _frame(value):
    if not isinstance(value, str) or not re.fullmatch(r"[A-Za-z][A-Za-z0-9_/]*", value):
        raise ValueError("Explicit nonempty relative frame name required")
    return value


@dataclass(frozen=True)
class HardwareInputProfile:
    source_pose_frame: str
    body_frame: str
    camera_frames: dict
    camera_topics: dict
    odometry_topic: str = "/spot/platform/odom"

    def __post_init__(self):
        _frame(self.source_pose_frame)
        _frame(self.body_frame)
        if self.body_frame in (self.source_pose_frame, "map"):
            raise ValueError("Body and world frames must differ")
        if set(self.camera_frames) != set(CAMERAS) or set(self.camera_topics) != set(CAMERAS):
            raise ValueError("All five cameras must be configured explicitly")
        frames = dict(self.camera_frames)
        topics = dict(self.camera_topics)
        for value in frames.values():
            _frame(value)
        if len(set(frames.values())) != 5 or set(frames.values()) & {
            self.body_frame,
            self.source_pose_frame,
            "map",
        }:
            raise ValueError("Distinct camera optical frames required")
        for value in [self.odometry_topic, *topics.values()]:
            if not isinstance(value, str) or not re.fullmatch(r"/[A-Za-z][A-Za-z0-9_/]*", value):
                raise ValueError("Absolute ROS topic names required")
        if len(set(topics.values())) != 5 or self.odometry_topic in topics.values():
            raise ValueError("Distinct sensor topics required")
        object.__setattr__(self, "camera_frames", MappingProxyType(frames))
        object.__setattr__(self, "camera_topics", MappingProxyType(topics))

    @classmethod
    def spot_driver(cls):
        """Explicit opt-in layout observed in the archived Stanford driver bag."""
        return cls(
            source_pose_frame="vision",
            body_frame="body",
            camera_frames={name: name + "_fisheye" for name in CAMERAS},
            camera_topics={name: f"/camera/{name}/compressed" for name in CAMERAS},
        )

    def describe(self):
        return dict(
            source_pose_frame=self.source_pose_frame,
            body_frame=self.body_frame,
            output_pose_frame="map",
            camera_frames=dict(self.camera_frames),
            camera_topics=dict(self.camera_topics),
            odometry_topic=self.odometry_topic,
            transforms="static session-pinned TF; not inferred alignment",
        )


def rigid_matrix(translation, quaternion):
    t, q = np.asarray(translation, dtype=float), np.asarray(quaternion, dtype=float)
    if t.shape != (3,) or q.shape != (4,) or not np.isfinite(t).all() or not np.isfinite(q).all():
        raise ValueError("Finite translation and quaternion required")
    if abs(float(q @ q) - 1) > 1e-3:
        raise ValueError("Transform orientation must be a unit quaternion")
    result = np.eye(4)
    result[:3, :3], result[:3, 3] = Rotation.from_quat(q).as_matrix(), t
    return result


class StaticTransformTree:
    def __init__(self):
        self._parents, self._invalid = {}, set()

    def update(self, transforms):
        for record in transforms:
            child = _frame(record.child_frame_id)
            try:
                parent = _frame(record.header.frame_id)
                if parent == child:
                    raise ValueError("Self-referential transform")
                t, q = record.transform.translation, record.transform.rotation
                matrix = rigid_matrix([t.x, t.y, t.z], [q.x, q.y, q.z, q.w])
                previous = self._parents.get(child)
                if previous is not None and (
                    previous[0] != parent or not np.allclose(previous[1], matrix, rtol=0, atol=1e-8)
                ):
                    raise ValueError("Static transform changed within the session")
                self._parents[child] = (parent, matrix)
            except (TypeError, ValueError, AttributeError):
                self._invalid.add(child)

    def _ancestors(self, frame):
        frame = _frame(frame)
        chain, value = {}, np.eye(4)
        while True:
            if frame in self._invalid or frame in chain:
                raise ValueError("Invalid, changed or cyclic static transform path")
            chain[frame] = value
            if frame not in self._parents:
                return chain
            frame, edge = self._parents[frame]
            value = edge @ value

    def lookup(self, target, source):
        targets, sources = self._ancestors(target), self._ancestors(source)
        for ancestor, target_matrix in targets.items():
            if ancestor in sources:
                return np.linalg.inv(target_matrix) @ sources[ancestor]
        raise ValueError(f"No static transform from {source} to {target}")


class HardwareInputNormalizer:
    def __init__(self, profile):
        if not isinstance(profile, HardwareInputProfile):
            raise TypeError("Explicit hardware input profile required")
        self.profile, self.transforms = profile, StaticTransformTree()

    def odometry(self, message):
        if (
            message.header.frame_id != self.profile.source_pose_frame
            or message.child_frame_id != self.profile.body_frame
        ):
            raise ValueError("Driver odometry frames differ from the input profile")
        p, q = message.pose.pose.position, message.pose.pose.orientation
        pose = self.transforms.lookup("map", self.profile.source_pose_frame) @ rigid_matrix(
            [p.x, p.y, p.z], [q.x, q.y, q.z, q.w]
        )
        # Twist is expressed in child/body coordinates and is not rotated into map.
        return pose[:3, 3].tolist(), Rotation.from_matrix(pose[:3, :3]).as_quat().tolist()

    def camera_transform(self, camera):
        # Match the preserved skill interface's inverse(body<-camera) convention.
        return np.linalg.inv(
            self.transforms.lookup(self.profile.body_frame, self.profile.camera_frames[camera])
        )
