import hashlib
import json
import math
from types import SimpleNamespace

import cv2
import numpy as np
import pytest

from bt_agent_robotics.platform import PlatformState
from bt_agent_robotics.publication import PhysicalPublication
from bt_agent_robotics._image_rotation import correct_orientation, build_capture_submission_message


def jpeg():
    rgb = np.arange(8 * 12 * 3, dtype=np.uint8).reshape(8, 12, 3)
    ok, data = cv2.imencode(".jpg", rgb)
    assert ok
    return data.tobytes()


def test_no_fabricated_pose_and_epoch_isolation():
    state = PlatformState("one", clock=lambda: 10.0)
    assert state.snapshot()["robot_position"] is None and state.pose_age() is None
    assert not state.odometry(None, None, None, None, epoch="old")
    assert state.odometry(
        [1, 2], [0, 0, math.sin(0.25), math.cos(0.25)], [3, 4], -0.2, epoch="one", received_at=9.9
    )
    assert state.snapshot()["robot_position"] == pytest.approx((1, 2, 0.5))
    assert state.velocity_ref == pytest.approx([5, 0.2])
    assert state.pose_age() == pytest.approx(0.1)
    assert not state.odometry([9, 9], [0, 0, 0, 1], [0, 0], 0, epoch="one", received_at=9.8)


@pytest.mark.parametrize("camera", ["frontleft", "frontright", "left", "right", "back"])
def test_camera_matches_reference_codec_and_records_exact_bytes(camera):
    evidence = []
    state = PlatformState("one", clock=lambda: 10.0, on_frame=lambda *row: evidence.append(row))
    raw = jpeg()
    assert state.camera(camera, raw, epoch="one")
    assert state.snapshot()["camera_images"][camera] == correct_orientation(raw, camera)
    record, original, corrected = evidence[0]
    assert original == raw
    assert record["corrected_sha256"] == hashlib.sha256(corrected).hexdigest()
    assert state.frame_records()[camera] == record


def test_bad_frame_does_not_replace_a_good_one():
    state = PlatformState("one", clock=lambda: 10.0)
    state.camera("frontleft", jpeg(), epoch="one", received_at=9.5)
    before = state.snapshot()["camera_images"]
    with pytest.raises(ValueError, match="Invalid camera"):
        state.camera("frontleft", b"bad", epoch="one")
    assert state.snapshot()["camera_images"] == before
    assert not state.camera("frontleft", b"bad", epoch="old")


@pytest.mark.parametrize("stamp", [float("nan"), float("inf"), -1, 0, 11])
def test_bad_or_future_timestamp(stamp):
    state = PlatformState("one", clock=lambda: 10.0)
    with pytest.raises(ValueError):
        state.camera("frontleft", jpeg(), epoch="one", received_at=stamp)


def test_camera_observer_failure_is_not_accepted():
    def fail(*args):
        raise OSError("disk full")

    state = PlatformState("one", clock=lambda: 10.0, on_frame=fail)
    with pytest.raises(OSError):
        state.camera("left", jpeg(), epoch="one")
    assert state.snapshot()["camera_images"] == {}


def test_transform_copy_and_closed_intake():
    state = PlatformState("one")
    matrix = np.eye(4)
    state.camera_transform("left", matrix, epoch="one")
    matrix[0, 3] = 8
    assert state.snapshot()["camera_to_body_transforms"]["left"][0, 3] == 0
    state.close()
    assert not state.camera("left", jpeg(), epoch="one")
    with pytest.raises(RuntimeError):
        state.snapshot()


def artifact(camera="frontleft"):
    return SimpleNamespace(
        handle="cap_1",
        camera=camera,
        description="fixture",
        label="",
        target_coords=(3, 4),
        bbox_1000=(10, 20, 800, 900),
        rotated_image_width=8,
        rotated_image_height=12,
        image=jpeg(),
        skill_call_id="call-1",
    )


@pytest.mark.parametrize("camera", ["frontleft", "frontright", "left", "right", "back"])
def test_capture_wire_and_dedup_match_reference(camera):
    submissions, controls = [], []
    output = PhysicalPublication(submissions.append, controls.append)
    cap = artifact(camera)
    output.flush([cap])
    output.flush([cap])
    assert submissions == [
        build_capture_submission_message(camera, "fixture", (3, 4), cap.bbox_1000, 8, 12)
    ]
    assert output.captures[0]["model_image_sha256"] == hashlib.sha256(cap.image).hexdigest()
    output.report("A finding.")
    assert submissions[-1] == json.dumps({"kind": "report", "text": "A finding."})
    with pytest.raises(RuntimeError):
        output.task_complete(submitted=False)
    output.task_complete(submitted=True)
    output.task_complete(submitted=True)
    assert controls == ["task complete"]


def test_failed_publish_can_retry_but_does_not_claim_exactly_once():
    attempts = []

    def publish(value):
        attempts.append(value)
        if len(attempts) == 1:
            raise OSError("transport failed")

    output = PhysicalPublication(publish, lambda _: None)
    with pytest.raises(OSError):
        output.flush([artifact()])
    assert output.captures == []
    output.flush([artifact()])
    assert len(attempts) == 2 and len(output.captures) == 1


def test_no_clipping_of_finite_bad_model_boxes():
    captured = []
    output = PhysicalPublication(captured.append, lambda _: None)
    cap = artifact("left")
    cap.bbox_1000 = (-10, 0, 1200, 1400)
    output.flush([cap])
    assert captured == [
        build_capture_submission_message("left", "fixture", (3, 4), cap.bbox_1000, 8, 12)
    ]
