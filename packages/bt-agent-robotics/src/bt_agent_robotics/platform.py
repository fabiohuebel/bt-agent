"""Passive robot-state intake. No ROS, model call, control or renderer starts here."""

from copy import deepcopy
import hashlib
import math
import threading
import time

CAMERAS = ("frontleft", "frontright", "left", "right", "back")


class PlatformState:
    """One episode's sensor snapshots, with explicit receive timestamps.

    Intake uses a caller-owned epoch. Legacy ROS frames carry no episode ID;
    the live adapter still must own subscriptions and isolate task transitions.
    Snapshots do not fabricate odometry or silently turn stale data into fresh.
    """

    def __init__(self, epoch, *, clock=time.monotonic, on_frame=None):
        if not isinstance(epoch, str) or not epoch:
            raise ValueError("An explicit episode epoch is required")
        self.epoch = epoch
        self.clock = clock
        self.on_frame = on_frame
        self.lock = threading.RLock()
        # These references can be passed to the preserved RosLocalPlanner.
        self.pose_ref = [0.0, 0.0, 0.0]
        self.velocity_ref = [0.0, 0.0]
        self.pose_stamp_ref = [0.0]
        self._pose_present = False
        self._images = {}
        self._frames = {}
        self._transforms = {}
        self._sequence = 0
        self._closed = False

    def _stamp(self, value):
        stamp = self.clock() if value is None else value
        if type(stamp) not in (int, float) or not math.isfinite(stamp) or stamp <= 0:
            raise ValueError("Receive stamp must be positive finite monotonic seconds")
        if stamp > self.clock():
            raise ValueError("Receive stamp cannot be in the future; do not use ROS/wall time here")
        return float(stamp)

    def odometry(
        self, position_xy, quaternion_xyzw, velocity_xy, angular_z, *, epoch, received_at=None
    ):
        with self.lock:
            if self._closed or epoch != self.epoch:
                return False
            stamp = self._stamp(received_at)
            if stamp <= self.pose_stamp_ref[0]:
                return False
            if len(position_xy) != 2 or len(quaternion_xyzw) != 4 or len(velocity_xy) != 2:
                raise ValueError("Invalid odometry dimensions")
            values = [*position_xy, *quaternion_xyzw, *velocity_xy, angular_z]
            if any(type(v) not in (int, float) or not math.isfinite(v) for v in values):
                raise ValueError("Nonfinite odometry")
            x, y, z, w = quaternion_xyzw
            if abs(x * x + y * y + z * z + w * w - 1) > 1e-3:
                raise ValueError("Odometry orientation must be a unit quaternion")
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            self.pose_ref[:] = [float(position_xy[0]), float(position_xy[1]), yaw]
            self.velocity_ref[:] = [math.hypot(*velocity_xy), abs(float(angular_z))]
            self.pose_stamp_ref[0] = stamp
            self._pose_present = True
            return True

    def camera(self, camera, raw_jpeg, *, epoch, received_at=None):
        with self.lock:
            if self._closed or epoch != self.epoch:
                return False
            if camera not in CAMERAS:
                raise ValueError("Unknown physical camera")
            stamp = self._stamp(received_at)
            if camera in self._frames and stamp <= self._frames[camera]["received_at"]:
                return False
            if not isinstance(raw_jpeg, bytes) or not raw_jpeg:
                raise ValueError("Camera input must be nonempty JPEG bytes")
            import cv2
            import numpy as np
            from ._image_rotation import correct_orientation

            decoded = cv2.imdecode(np.frombuffer(raw_jpeg, dtype=np.uint8), cv2.IMREAD_COLOR)
            if decoded is None:
                raise ValueError("Invalid camera image; no stale frame is replaced")
            corrected = correct_orientation(raw_jpeg, camera)
            self._sequence += 1
            record = dict(
                epoch=self.epoch,
                frame_id=self._sequence,
                camera=camera,
                received_at=stamp,
                raw_sha256=hashlib.sha256(raw_jpeg).hexdigest(),
                corrected_sha256=hashlib.sha256(corrected).hexdigest(),
                binding="receiver-assigned; legacy wire has no model frame ID",
            )
            # Evidence sink failure is visible; do not accept an unrecorded frame.
            if self.on_frame is not None:
                self.on_frame(deepcopy(record), raw_jpeg, corrected)
            self._images[camera] = corrected
            self._frames[camera] = record
            return True

    def camera_transform(self, camera, matrix, *, epoch):
        with self.lock:
            if self._closed or epoch != self.epoch:
                return False
            import numpy as np

            data = np.asarray(matrix, dtype=float)
            if camera not in CAMERAS or data.shape != (4, 4) or not np.isfinite(data).all():
                raise ValueError("Expected a finite 4x4 camera/body transform")
            if not np.allclose(data[3], [0, 0, 0, 1]):
                raise ValueError("Invalid transform homogeneous row")
            self._transforms[camera] = data.copy()
            return True

    def pose_age(self):
        with self.lock:
            return self.clock() - self.pose_stamp_ref[0] if self._pose_present else None

    def snapshot(self):
        with self.lock:
            if self._closed:
                raise RuntimeError("Platform state is closed")
            return dict(
                robot_position=tuple(self.pose_ref) if self._pose_present else None,
                scene_graph={},
                camera_images=dict(self._images),
                camera_to_body_transforms={k: v.copy() for k, v in self._transforms.items()},
                terrain_map=None,
            )

    def frame_records(self):
        with self.lock:
            return deepcopy(self._frames)

    def close(self):
        with self.lock:
            self._closed = True
            self._images.clear()
