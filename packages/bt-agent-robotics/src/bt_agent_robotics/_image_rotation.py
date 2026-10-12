"""Camera image rotation for Spot — correct head/side camera orientation.

Spot's head cameras (frontleft, frontright) produce images rotated 90°.
The right side camera is rotated 180°. This module corrects them so the
VLM sees upright images.

Also provides ``derotate_coords`` to transform VLM coordinates (in
rotated image space) back to the simulator's original camera frame
for capture submissions.

Matches the v1 logic in ``lam_core.perception.image_utils``.
"""

from __future__ import annotations

from typing import Tuple

import cv2
import numpy as np

# Camera name → cv2 rotation code.  None = no rotation.
_ROTATION_MAP = {
    "frontleft": cv2.ROTATE_90_CLOCKWISE,
    "frontright": cv2.ROTATE_90_CLOCKWISE,
    "left": None,
    "right": cv2.ROTATE_180,
    "back": None,
}


def correct_orientation(jpeg_bytes: bytes, camera_name: str) -> bytes:
    """Rotate a JPEG image to correct Spot camera orientation.

    Returns the corrected JPEG bytes (or the original if no rotation
    is needed or decoding fails).
    """
    rotation = _ROTATION_MAP.get(camera_name)
    if rotation is None:
        return jpeg_bytes

    try:
        buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
        img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
        if img is None:
            return jpeg_bytes
        img = cv2.rotate(img, rotation)
        _, encoded = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 90])
        return bytes(encoded)
    except Exception:
        return jpeg_bytes


def derotate_coords(
    pixel_coords: Tuple[int, int],
    bbox_1000: Tuple[int, int, int, int],
    camera_name: str,
    rotated_width: int,
    rotated_height: int,
) -> Tuple[Tuple[int, int], Tuple[int, int, int, int]]:
    """Transform coordinates from rotated image space back to original.

    The VLM produces ``pixel_coords`` and ``bbox_1000`` in the rotated
    (corrected) image.  The simulator expects pixel coordinates in the
    original (un-rotated) camera frame.  This function inverts the
    rotation applied by ``correct_orientation``.

    Returns:
        ``(orig_pixel, orig_bbox_pixels)`` — both in original camera
        pixel coordinates.
    """
    rotation = _ROTATION_MAP.get(camera_name)

    # Original image dimensions (before rotation was applied)
    if rotation == cv2.ROTATE_90_CLOCKWISE:
        orig_w, orig_h = rotated_height, rotated_width
    else:
        orig_w, orig_h = rotated_width, rotated_height

    def to_px(norm: int, limit: int) -> int:
        # Nearest-pixel conversion makes normalized coordinates produced from
        # integer mask extents round-trip, including odd camera dimensions.
        return int(round(norm * limit / 1000))

    px, py = pixel_coords
    x1, y1, x2, y2 = bbox_1000

    if rotation is None:
        return (px, py), (to_px(x1, orig_w), to_px(y1, orig_h), to_px(x2, orig_w), to_px(y2, orig_h))

    if rotation == cv2.ROTATE_90_CLOCKWISE:
        # Inverse of 90° CW is 90° CCW
        orig_px = py
        orig_py = rotated_width - 1 - px

        tx1 = to_px(y1, orig_w)
        ty1 = to_px(1000 - x1, orig_h)
        tx2 = to_px(y2, orig_w)
        ty2 = to_px(1000 - x2, orig_h)

        return (orig_px, orig_py), (min(tx1, tx2), min(ty1, ty2), max(tx1, tx2), max(ty1, ty2))

    if rotation == cv2.ROTATE_180:
        orig_px = rotated_width - 1 - px
        orig_py = rotated_height - 1 - py

        tx1 = to_px(1000 - x1, orig_w)
        ty1 = to_px(1000 - y1, orig_h)
        tx2 = to_px(1000 - x2, orig_w)
        ty2 = to_px(1000 - y2, orig_h)

        return (orig_px, orig_py), (min(tx1, tx2), min(ty1, ty2), max(tx1, tx2), max(ty1, ty2))

    return (px, py), (to_px(x1, orig_w), to_px(y1, orig_h), to_px(x2, orig_w), to_px(y2, orig_h))


def rotate_raw_to_corrected(
    pixel_xy: Tuple[int, int],
    bbox_px: Tuple[int, int, int, int],
    camera_name: str,
    orig_width: int,
    orig_height: int,
) -> Tuple[Tuple[int, int], Tuple[int, int, int, int], int, int]:
    """Map a raw-frame point and half-open bbox into corrected-image space.

    This is the geometric inverse of :func:`derotate_coords` before its
    normalized-bbox quantization.  ``bbox_px`` uses ``(x1, y1, x2, y2)`` with
    exclusive upper bounds, matching numpy mask extents.
    """
    rotation = _ROTATION_MAP.get(camera_name)
    x, y = pixel_xy
    x1, y1, x2, y2 = bbox_px
    if rotation == cv2.ROTATE_90_CLOCKWISE:
        return (orig_height - 1 - y, x), (orig_height - y2, x1, orig_height - y1, x2), orig_height, orig_width
    if rotation == cv2.ROTATE_180:
        return (
            (orig_width - 1 - x, orig_height - 1 - y),
            (orig_width - x2, orig_height - y2, orig_width - x1, orig_height - y1),
            orig_width,
            orig_height,
        )
    return (x, y), bbox_px, orig_width, orig_height


def build_capture_submission_message(
    camera: str,
    description: str,
    pixel_coords: Tuple[int, int],
    bbox_1000: Tuple[int, int, int, int],
    rotated_width: int,
    rotated_height: int,
) -> str:
    """Build the exact ``/task_submission`` wire string the sim expects for a
    capture: ``'topic, (xP, yP, xC1, yC1, xC2, yC2), description'``, with the
    target point and bbox derotated from the VLM's corrected-orientation view
    back to the simulator's original camera frame (``derotate_coords``,
    above).

    The SHARED submission-formatting helper for both agent kinds — the BT
    path (``TiamatBridgeNode._submit_capture_for``) and the flat/CaP-X path
    (``FlatPolicyNode._flat_submission_callback``) both call this, so they
    publish byte-identical wire strings for the same capture (CP2 parity
    fix — docs/spec/ros-baseline-node.md).
    """
    topic = f"/spot/camera/{camera}/image/compressed"
    (tx, ty), (x1, y1, x2, y2) = derotate_coords(pixel_coords, tuple(bbox_1000), camera, rotated_width, rotated_height)
    return f"{topic}, ({tx}, {ty}, {x1}, {y1}, {x2}, {y2}), {description}"
