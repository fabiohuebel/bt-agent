import hashlib
import json
import threading
from types import SimpleNamespace as NS

import cv2
import numpy as np
import pytest

from bt_agent_robotics.hardware_ros import RosHardwareIO, HardwareROSEpisode
from bt_agent_robotics.platform import PlatformState, CAMERAS
from bt_agent_robotics.publication import PhysicalPublication


@pytest.mark.parametrize("pose_frame", ["odom", "body", "", None, 1])
def test_waypoint_frame_mismatch_rejected_before_ros_or_output(tmp_path, pose_frame):
    evidence = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="waypoints use map"):
        RosHardwareIO(
            None,
            epoch="frame-test",
            evidence=evidence,
            pose_frame=pose_frame,
            body_frame="body",
            camera_frames={},
        )
    assert not evidence.exists()


@pytest.mark.parametrize("body_frame", ["map", "", " ", None, 1])
def test_body_frame_must_be_explicit_and_distinct(tmp_path, body_frame):
    evidence = tmp_path / "must-not-exist"
    with pytest.raises(ValueError, match="body frame"):
        RosHardwareIO(
            None,
            epoch="frame-test",
            evidence=evidence,
            pose_frame="map",
            body_frame=body_frame,
            camera_frames={},
        )
    assert not evidence.exists()


@pytest.fixture
def io(tmp_path, monkeypatch):
    value = RosHardwareIO.__new__(RosHardwareIO)
    value.epoch = "test"
    value.node = NS(get_clock=lambda: NS(now=lambda: NS(nanoseconds=10_000_000_000)))
    value.source_age_s, value.frame_age_s = 0.75, 2.0
    value.pose_frame, value.body_frame = "map", "body"
    value.camera_frames = {name: name + "_optical" for name in CAMERAS}
    value.platform = PlatformState("test", clock=lambda: 10.0)
    value._errors, value._source_stamps = {}, {}
    value._closed = False
    value._lock = threading.RLock()
    value._state_lock = threading.RLock()
    value._String = lambda: NS(data="")
    value.out = tmp_path
    value.journal = (tmp_path / "journal.jsonl").open("x")
    monkeypatch.setattr("bt_agent_robotics.hardware_ros.time.monotonic", lambda: 10.0)
    yield value
    value.journal.close()


def header(stamp=9_900_000_000, frame="map"):
    return NS(stamp=NS(sec=stamp // 1_000_000_000, nanosec=stamp % 1_000_000_000), frame_id=frame)


def odom(stamp=9_900_000_000, frame="map", q=1.0):
    return NS(
        header=header(stamp, frame),
        pose=NS(pose=NS(position=NS(x=1.0, y=2.0), orientation=NS(x=0.0, y=0.0, z=0.0, w=q))),
        twist=NS(twist=NS(linear=NS(x=0.0, y=0.0), angular=NS(z=0.0))),
    )


def camera(stamp=9_900_000_000, frame="frontleft_optical", data=None):
    if data is None:
        _, image = cv2.imencode(".jpg", np.zeros((8, 12, 3), dtype=np.uint8))
        data = image.tobytes()
    return NS(header=header(stamp, frame), data=data)


@pytest.mark.parametrize("stamp", [0, 8_000_000_000, 10_000_000_001])
def test_reject_stale_future_or_missing_source_stamp(io, stamp):
    io._odom(odom(stamp))
    assert io.platform.pose_age() is None and "odom" in io._errors
    assert io.agent_state()["robot_position"] is None


@pytest.mark.parametrize("bad", [odom(frame="odom"), odom(q=2.0), odom(q=float("nan"))])
def test_invalid_pose_does_not_become_an_observed_origin(io, bad):
    io._odom(bad)
    assert io.platform.pose_age() is None


def test_good_pose_then_repeated_message_is_masked(io):
    io._odom(odom())
    assert io.agent_state()["robot_position"] == (1.0, 2.0, 0.0)
    io._odom(odom())
    assert "odom" in io._errors
    assert io.agent_state()["robot_position"] is None


@pytest.mark.parametrize("bad", [camera(frame="wrong"), camera(data=b"invalid"), camera(stamp=0)])
def test_invalid_camera_is_not_exposed(io, bad):
    io._camera("frontleft", bad)
    assert "frontleft" in io._errors and not io.agent_state()["camera_images"]


def test_sensor_snapshot_requires_valid_transform_and_not_stale_frames(io, monkeypatch):
    io._camera("frontleft", camera())
    assert not io.agent_state()["camera_images"]
    io.platform.camera_transform("frontleft", np.eye(4), epoch="test")
    assert "frontleft" in io.agent_state()["camera_images"]
    io._errors["tf:frontleft"] = "invalid transform"
    assert not io.agent_state()["camera_images"]
    io._errors.clear()
    monkeypatch.setattr("bt_agent_robotics.hardware_ros.time.monotonic", lambda: 13.0)
    assert not io.agent_state()["camera_images"]


def test_transform_preserves_deployed_inverse_and_rejects_invalid_rotation(io):
    transform = NS(
        header=NS(frame_id="body"),
        child_frame_id="frontleft_optical",
        transform=NS(translation=NS(x=1.0, y=2.0, z=3.0), rotation=NS(x=0.0, y=0.0, z=0.0, w=1.0)),
    )
    io._transforms(NS(transforms=[transform]))
    assert io.platform.snapshot()["camera_to_body_transforms"]["frontleft"][:3, 3] == pytest.approx(
        [-1, -2, -3]
    )
    transform.transform.rotation.w = 2.0
    io._transforms(NS(transforms=[transform]))
    assert "tf:frontleft" in io._errors


def test_capture_persists_exact_bytes_and_checks_partial_files(io):
    image = camera().data
    io._capture_evidence({"handle": "cap_1"}, image)
    path = io.out / (hashlib.sha256(image).hexdigest() + ".jpg")
    assert path.read_bytes() == image
    path.write_bytes(b"incomplete previous write")
    with pytest.raises(ValueError, match="incomplete or corrupted"):
        io._capture_evidence({"handle": "cap_1"}, image)


def test_failed_publication_not_acknowledged(io):
    class Publisher:
        def publish(self, message):
            raise OSError("offline failure")

    publication = PhysicalPublication(
        lambda wire: io._publish("submission", Publisher(), wire),
        lambda wire: io._publish("control", Publisher(), wire),
    )
    with pytest.raises(OSError):
        publication.report("finding")
    with pytest.raises(OSError):
        publication.task_complete(submitted=True)
    assert not publication.reports and not publication.terminal_delivered
    records = [json.loads(line) for line in (io.out / "journal.jsonl").read_text().splitlines()]
    assert all(row["phase"] == "intent" for row in records)


def test_terminal_publish_can_retry_without_reticking_completed_agent():
    calls = []

    def publish(wire):
        calls.append(wire)
        if len(calls) == 1:
            raise OSError("ambiguous transport failure")

    episode = HardwareROSEpisode.__new__(HardwareROSEpisode)
    episode.started = True
    episode.agent = NS(done=True)
    episode.host = NS(submitted=True)
    episode.io = NS(publication=PhysicalPublication(lambda _: None, publish))
    with pytest.raises(OSError):
        episode.tick()
    assert episode.tick() == ([], [])
    assert episode.io.publication.terminal_delivered
    episode.tick()
    assert len(calls) == 2  # no claim of exactly-once external delivery


def test_unsubmitted_cancellation_never_publishes_terminal():
    episode = HardwareROSEpisode.__new__(HardwareROSEpisode)
    episode.started = True
    episode.agent, episode.host = NS(done=True), NS(submitted=False)
    sent = []
    episode.io = NS(publication=PhysicalPublication(sent.append, sent.append))
    assert episode.tick() == ([], []) and not sent
