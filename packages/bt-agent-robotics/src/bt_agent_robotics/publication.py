"""Exact physical capture/report wire formatting behind explicit callbacks."""

import hashlib
import json
import math
import threading

from .platform import CAMERAS


class PhysicalPublication:
    """One episode's output adapter. Callback completion is its delivery boundary.

    Use callbacks that raise on failure. Successful local ROS publish alone is
    not proof a remote host received it; network acknowledgement remains the
    transport's responsibility. Deduplication here is per capture handle, as in
    the evaluated agent, not a distributed exactly-once guarantee.
    """

    def __init__(self, publish_submission, publish_control, *, capture_evidence=None):
        self.publish_submission = publish_submission
        self.publish_control = publish_control
        self.capture_evidence = capture_evidence
        self._sent = set()
        self._terminal = False
        self._lock = threading.RLock()
        self.captures = []
        self.reports = []

    def report(self, text):
        if not isinstance(text, str) or not text.strip():
            raise ValueError("Report must be nonempty")
        with self._lock:
            if self._terminal:
                raise RuntimeError("Episode has terminated")
            payload = json.dumps({"kind": "report", "text": text})
            self.publish_submission(payload)
            self.reports.append(text)

    def flush(self, objects):
        from ._image_rotation import build_capture_submission_message

        with self._lock:
            if self._terminal:
                return
            for cap in objects:
                if not hasattr(cap, "bbox_1000") or not hasattr(cap, "target_coords"):
                    continue
                handle = getattr(cap, "handle", "")
                if not handle or handle in self._sent:
                    continue
                camera = getattr(cap, "camera", "frontleft")
                point = tuple(getattr(cap, "target_coords", (0, 0)))
                box = tuple(getattr(cap, "bbox_1000", (0, 0, 0, 0)))
                width = getattr(cap, "rotated_image_width", 1000)
                height = getattr(cap, "rotated_image_height", 1000)
                if camera not in CAMERAS or len(point) != 2 or len(box) != 4:
                    raise ValueError("Malformed capture artifact")
                if (
                    any(
                        not isinstance(v, (int, float)) or not math.isfinite(v)
                        for v in (*point, *box, width, height)
                    )
                    or min(width, height) <= 0
                ):
                    raise ValueError("Nonfinite/invalid capture geometry")
                # Preserve finite out-of-frame model predictions. Their quality
                # belongs to adjudication, not silent coordinate clipping here.
                wire = build_capture_submission_message(
                    camera,
                    getattr(cap, "description", "") or getattr(cap, "label", ""),
                    point,
                    box,
                    width,
                    height,
                )
                image = getattr(cap, "image", None)
                record = dict(
                    handle=handle,
                    camera=camera,
                    skill_call_id=getattr(cap, "skill_call_id", ""),
                    model_image_sha256=(
                        hashlib.sha256(image).hexdigest() if isinstance(image, bytes) else None
                    ),
                    wire=wire,
                    model_frame_id_in_legacy_wire=False,
                )
                if self.capture_evidence is not None:
                    self.capture_evidence(dict(record), image)
                self.publish_submission(wire)
                self._sent.add(handle)
                self.captures.append(record)

    def task_complete(self, *, submitted):
        with self._lock:
            if submitted is not True:
                raise RuntimeError("A milestone claim is not acknowledged terminal submission")
            if not self._terminal:
                self.publish_control("task complete")
                self._terminal = True

    @property
    def terminal_delivered(self):
        return self._terminal
