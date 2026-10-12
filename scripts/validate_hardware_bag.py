"""Sample archived driver inputs through release intake, with no ROS output.

Requires optional rosbags plus the installed robotics package. Uses recorded
static TF and first/middle/last sensor samples. The receiver clock is simulated
from bag receipt times; this is not live freshness or physical validation.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3
import threading
from types import SimpleNamespace


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db3", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    path = args.db3.resolve(strict=True)
    if args.out.exists() or path.suffix != ".db3" or Path(str(path) + "-wal").exists():
        raise ValueError("Use a closed db3 without a WAL and a new output path")
    before = path.stat()
    from rosbags.typesys import Stores, get_typestore
    from bt_agent_robotics.hardware_inputs import HardwareInputNormalizer, HardwareInputProfile
    import bt_agent_robotics.hardware_ros as wire
    from bt_agent_robotics.platform import PlatformState

    profile = HardwareInputProfile.spot_driver()
    store = get_typestore(Stores.ROS2_HUMBLE)
    samples, transforms = [], []
    routes = {profile.odometry_topic: "odom", **{v: k for k, v in profile.camera_topics.items()}}
    with sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        for topic in ["/tf_static", *routes]:
            row = db.execute("SELECT id,type FROM topics WHERE name=?", (topic,)).fetchone()
            if row is None:
                raise ValueError("Missing archived topic: " + topic)
            topic_id, typename = row
            count = db.execute(
                "SELECT COUNT(*) FROM messages WHERE topic_id=?", (topic_id,)
            ).fetchone()[0]
            if not count:
                raise ValueError("Empty archived topic: " + topic)
            offsets = range(count) if topic == "/tf_static" else sorted({0, count // 2, count - 1})
            for offset in offsets:
                stamp, raw = db.execute(
                    "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT 1 OFFSET ?",
                    (topic_id, offset),
                ).fetchone()
                message = store.deserialize_cdr(raw, typename)
                if topic == "/tf_static":
                    transforms.extend(message.transforms)
                else:
                    samples.append((stamp, routes[topic], message))
    samples.sort(key=lambda row: row[0])
    clock = dict(wall=samples[0][0], monotonic=1.0)
    io = wire.RosHardwareIO.__new__(wire.RosHardwareIO)  # deliberately no node/publishers
    io.epoch, io.pose_frame, io.body_frame = "archived-intake", "map", "body"
    io.camera_frames = dict(profile.camera_frames)
    io._input = HardwareInputNormalizer(profile)
    io._closed, io._state_lock, io._errors, io._source_stamps = False, threading.RLock(), {}, {}
    io.source_age_s, io.frame_age_s = 0.75, 2.0
    io.node = SimpleNamespace(
        get_clock=lambda: SimpleNamespace(now=lambda: SimpleNamespace(nanoseconds=clock["wall"]))
    )
    io.platform = PlatformState(io.epoch, clock=lambda: clock["monotonic"])
    rows = []
    original_monotonic = wire.time.monotonic
    try:
        wire.time.monotonic = lambda: clock["monotonic"]
        io._transforms(SimpleNamespace(transforms=transforms))
        for stamp, key, message in samples:
            clock.update(wall=stamp, monotonic=1 + (stamp - samples[0][0]) / 1e9)
            if key == "odom":
                io._odom(message)
            else:
                io._camera(key, message)
            state = io.agent_state()
            accepted = (
                state["robot_position"] is not None
                if key == "odom"
                else key in state["camera_images"]
            )
            row = dict(
                sensor=key,
                accepted=accepted,
                error=io._errors.get(key),
                source_frame=message.header.frame_id,
            )
            if accepted and key == "odom":
                row["map_pose"] = state["robot_position"]
            elif accepted:
                row["frame"] = io.platform.frame_records()[key]
            rows.append(row)
        ready = io.readiness()
    finally:
        wire.time.monotonic = original_monotonic
        io.platform.close()
    passed = all(
        any(row["sensor"] == key and row["accepted"] for row in rows) for key in routes.values()
    )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Archive changed during validation")
    result = dict(
        passed=passed,
        archive_sha256=digest.hexdigest(),
        profile=profile.describe(),
        samples=rows,
        last_readiness=ready,
        commands_replayed=False,
        ros_started=False,
        scope="Static-TF-assisted bounded archived intake, simulated receiver clock; not live startup, calibration or physical validation",
    )
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(
        json.dumps(
            dict(
                passed=passed,
                samples=len(rows),
                accepted=sum(row["accepted"] for row in rows),
                ros_started=False,
            )
        )
    )
    raise SystemExit(0 if passed else 1)


if __name__ == "__main__":
    main()
