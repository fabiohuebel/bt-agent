import math
import threading
from types import SimpleNamespace as NS

import numpy as np
import pytest

from bt_agent_robotics.hardware_inputs import (
    HardwareInputProfile,
    HardwareInputNormalizer,
    StaticTransformTree,
    rigid_matrix,
)
from bt_agent_robotics.hardware_ros import RosHardwareIO
from bt_agent_robotics.platform import CAMERAS, PlatformState


def transform(parent, child, xyz=(0, 0, 0), yaw=0):
    return NS(
        header=NS(frame_id=parent),
        child_frame_id=child,
        transform=NS(
            translation=NS(x=xyz[0], y=xyz[1], z=xyz[2]),
            rotation=NS(x=0, y=0, z=math.sin(yaw / 2), w=math.cos(yaw / 2)),
        ),
    )


def odometry(frame="vision", child="body"):
    return NS(
        header=NS(frame_id=frame, stamp=NS(sec=9, nanosec=900_000_000)),
        child_frame_id=child,
        pose=NS(
            pose=NS(position=NS(x=1.0, y=2.0, z=3.0), orientation=NS(x=0.0, y=0.0, z=0.0, w=1.0))
        ),
        twist=NS(twist=NS(linear=NS(x=0.2, y=0.3), angular=NS(z=0.4))),
    )


def test_spot_profile_matches_recorded_driver_topics():
    profile = HardwareInputProfile.spot_driver()
    assert profile.source_pose_frame == "vision" and profile.body_frame == "body"
    assert profile.camera_topics["left"] == "/camera/left/compressed"
    assert profile.camera_frames["frontleft"] == "frontleft_fisheye"
    with pytest.raises(TypeError):
        profile.camera_frames["left"] = "wrong"


@pytest.mark.parametrize("bad", ["", "/map", "a b", None, 3])
def test_explicit_frame_validation(bad):
    profile = HardwareInputProfile.spot_driver()
    with pytest.raises(ValueError):
        HardwareInputProfile(bad, "body", profile.camera_frames, profile.camera_topics)


def test_profile_copies_mutable_arguments():
    initial = HardwareInputProfile.spot_driver()
    topics = dict(initial.camera_topics)
    profile = HardwareInputProfile("vision", "body", initial.camera_frames, topics)
    topics["left"] = "/wrong"
    assert profile.camera_topics["left"] == "/camera/left/compressed"


@pytest.mark.parametrize(
    "translation,quaternion",
    [
        ([0, 0], [0, 0, 0, 1]),
        ([0, 0, math.inf], [0, 0, 0, 1]),
        ([0, 0, 0], [0, 0, 0, 0]),
        ([0, 0, 0], [0, 0, math.nan, 1]),
    ],
)
def test_rigid_transform_rejects_invalid_values(translation, quaternion):
    with pytest.raises(ValueError):
        rigid_matrix(translation, quaternion)


def test_composed_transform_and_inverse():
    normalizer = HardwareInputNormalizer(HardwareInputProfile.spot_driver())
    normalizer.transforms.update(
        [
            transform("body", "head", (1, 0, 0)),
            transform("head", "frontleft", (2, 0, 0)),
            transform("frontleft", "frontleft_fisheye", (3, 0, 0)),
        ]
    )
    assert normalizer.camera_transform("frontleft")[:3, 3] == pytest.approx([-6, 0, 0])
    forward = normalizer.transforms.lookup("body", "frontleft_fisheye")
    assert normalizer.camera_transform("frontleft") @ forward == pytest.approx(np.eye(4))


def test_map_conversion_is_not_frame_relabeling_and_does_not_edit_input():
    normalizer = HardwareInputNormalizer(HardwareInputProfile.spot_driver())
    normalizer.transforms.update([transform("map", "vision", (10, 20, 0), math.pi / 2)])
    message = odometry()
    xyz, q = normalizer.odometry(message)
    assert xyz == pytest.approx([8, 21, 3])
    assert q == pytest.approx([0, 0, math.sqrt(0.5), math.sqrt(0.5)])
    assert message.header.frame_id == "vision" and message.pose.pose.position.x == 1
    assert message.twist.twist.linear.x == 0.2


@pytest.mark.parametrize("frame,child", [("map", "body"), ("vision", "other")])
def test_odometry_frames_must_match_profile(frame, child):
    with pytest.raises(ValueError, match="frames"):
        HardwareInputNormalizer(HardwareInputProfile.spot_driver()).odometry(odometry(frame, child))


def test_missing_transform_never_defaults_to_identity():
    normalizer = HardwareInputNormalizer(HardwareInputProfile.spot_driver())
    with pytest.raises(ValueError, match="No static transform"):
        normalizer.odometry(odometry())


def test_cycle_is_rejected():
    tree = StaticTransformTree()
    tree.update([transform("a", "b"), transform("b", "a")])
    with pytest.raises(ValueError, match="cyclic"):
        tree.lookup("a", "b")


@pytest.mark.parametrize("change", ["parent", "translation", "rotation"])
def test_changed_static_tf_is_latched_invalid(change):
    tree = StaticTransformTree()
    initial = transform("map", "vision")
    tree.update([initial, initial])  # identical latched retransmission is allowed
    changed = transform("map", "vision")
    if change == "parent":
        changed.header.frame_id = "other"
    elif change == "translation":
        changed.transform.translation.x = 1
    else:
        changed.transform.rotation.w = 0
    tree.update([changed, initial])
    with pytest.raises(ValueError, match="Invalid"):
        tree.lookup("map", "vision")


def test_bad_unrelated_transform_does_not_supply_or_poison_camera_path():
    tree = StaticTransformTree()
    tree.update([transform("map", "vision"), transform("bad", "bad")])
    assert tree.lookup("map", "vision") == pytest.approx(np.eye(4))
    with pytest.raises(ValueError):
        tree.lookup("bad", "bad")


@pytest.fixture
def io(monkeypatch):
    value = RosHardwareIO.__new__(RosHardwareIO)
    value.epoch, value.pose_frame, value.body_frame = "test", "map", "body"
    value._input = HardwareInputNormalizer(HardwareInputProfile.spot_driver())
    value.camera_frames = dict(value._input.profile.camera_frames)
    value.node = NS(get_clock=lambda: NS(now=lambda: NS(nanoseconds=10_000_000_000)))
    value.source_age_s, value.frame_age_s = 0.75, 2.0
    value._state_lock, value._errors, value._source_stamps = threading.RLock(), {}, {}
    value._closed = False
    value.platform = PlatformState("test", clock=lambda: 10.0)
    monkeypatch.setattr("bt_agent_robotics.hardware_ros.time.monotonic", lambda: 10.0)
    return value


def full_tf():
    return [transform("map", "vision", (10, 20, 0), math.pi / 2), transform("body", "head")] + [
        t
        for camera in CAMERAS
        for t in (
            transform("head", camera),
            transform(camera, camera + "_fisheye"),
        )
    ]


def test_real_io_callback_converts_pose_with_chained_frames(io):
    io._transforms(NS(transforms=full_tf()))
    io._odom(odometry())
    assert io.agent_state()["robot_position"] == pytest.approx([8, 21, math.pi / 2])
    assert io.platform.snapshot()["camera_to_body_transforms"].keys() == set(CAMERAS)
    assert io.platform.velocity_ref == pytest.approx([math.hypot(0.2, 0.3), 0.4])


def test_source_freshness_still_enforced_after_normalization(io):
    io._transforms(NS(transforms=full_tf()))
    message = odometry()
    message.header.stamp.sec = 1
    io._odom(message)
    assert io.agent_state()["robot_position"] is None and "odom" in io._errors


def test_alignment_change_masks_existing_pose_immediately(io):
    io._transforms(NS(transforms=full_tf()))
    io._odom(odometry())
    assert io.agent_state()["robot_position"] is not None
    io._transforms(NS(transforms=[transform("map", "vision", (999, 0, 0))]))
    assert io.agent_state()["robot_position"] is None


def test_camera_chain_change_masks_existing_images(io):
    io._transforms(NS(transforms=full_tf()))
    io.platform._images["frontleft"] = b"test-fixture"
    io.platform._frames["frontleft"] = {"received_at": 10.0}
    assert "frontleft" in io.agent_state()["camera_images"]
    io._transforms(NS(transforms=[transform("body", "head", (9, 0, 0))]))
    assert not io.agent_state()["camera_images"]


def test_inconsistent_profile_is_rejected_before_ros(tmp_path):
    with pytest.raises(ValueError, match="Input profile"):
        RosHardwareIO(
            None,
            epoch="test",
            evidence=tmp_path / "unused",
            pose_frame="map",
            body_frame="body",
            camera_frames={},
            input_profile=HardwareInputProfile.spot_driver(),
        )
    assert not (tmp_path / "unused").exists()
