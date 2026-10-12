"""Read frame contracts from a closed sqlite3 ROS bag without starting ROS.

Optional dependency: rosbags. This samples archived messages; it does not replay
commands, infer calibration, validate live freshness or contact a robot.
"""

import argparse
import hashlib
import json
from pathlib import Path
import sqlite3

TOPICS = ["/spot/platform/odom", "/tf_static"] + [
    f"/spot/camera/{name}/image/compressed"
    for name in ("frontleft", "frontright", "left", "right", "back")
]


def inspect(db_path):
    from rosbags.typesys import Stores, get_typestore

    path = Path(db_path).resolve(strict=True)
    if path.suffix != ".db3" or Path(str(path) + "-wal").exists():
        raise ValueError("Use a closed .db3 archive without a live WAL")
    before = path.stat()
    store = get_typestore(Stores.ROS2_HUMBLE)
    result = dict(
        schema="hardware-bag-frame-sample/v1",
        archive=str(path),
        topics={},
        scope="First/middle/last samples per sensor topic and all static TF; no live freshness or calibration claim",
        commands_replayed=False,
    )
    with sqlite3.connect(path.as_uri() + "?mode=ro&immutable=1", uri=True) as db:
        # Include the driver's actual compressed-camera topics as evidence;
        # discovering one does not silently reconfigure the release adapter.
        camera_topics = [
            row[0]
            for row in db.execute(
                "SELECT name FROM topics WHERE type=? AND (name LIKE '/camera/%' OR name LIKE '/spot/camera/%')",
                ("sensor_msgs/msg/CompressedImage",),
            )
        ]
        for topic in dict.fromkeys(TOPICS + camera_topics):
            row = db.execute("SELECT id, type FROM topics WHERE name=?", (topic,)).fetchone()
            if row is None:
                result["topics"][topic] = dict(present=False)
                continue
            topic_id, typename = row
            count = db.execute(
                "SELECT COUNT(*) FROM messages WHERE topic_id=?", (topic_id,)
            ).fetchone()[0]
            offsets = (
                range(count) if topic == "/tf_static" else sorted({0, count // 2, count - 1} - {-1})
            )
            samples = []
            for offset in offsets:
                raw = db.execute(
                    "SELECT timestamp,data FROM messages WHERE topic_id=? ORDER BY timestamp LIMIT 1 OFFSET ?",
                    (topic_id, offset),
                ).fetchone()
                if raw is None:
                    continue
                bag_stamp, data = raw
                message = store.deserialize_cdr(data, typename)
                if topic == "/tf_static":
                    samples.append(
                        dict(
                            edges=[
                                dict(
                                    parent=t.header.frame_id,
                                    child=t.child_frame_id,
                                    translation=[
                                        t.transform.translation.x,
                                        t.transform.translation.y,
                                        t.transform.translation.z,
                                    ],
                                    rotation_xyzw=[
                                        t.transform.rotation.x,
                                        t.transform.rotation.y,
                                        t.transform.rotation.z,
                                        t.transform.rotation.w,
                                    ],
                                )
                                for t in message.transforms
                            ]
                        )
                    )
                else:
                    stamp = message.header.stamp.sec * 10**9 + message.header.stamp.nanosec
                    sample = dict(
                        frame_id=message.header.frame_id,
                        source_stamp_ns=stamp,
                        bag_minus_source_s=(bag_stamp - stamp) / 1e9,
                    )
                    if topic == "/spot/platform/odom":
                        sample["child_frame_id"] = message.child_frame_id
                    samples.append(sample)
            result["topics"][topic] = dict(
                present=True, type=typename, message_count=count, samples=samples
            )
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(4 * 1024 * 1024), b""):
            digest.update(chunk)
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Archive changed during inspection")
    result["sha256"] = digest.hexdigest()
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--db3", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Use a new result path")
    result = inspect(args.db3)
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
        stream.write("\n")
    print(json.dumps({"archive": str(args.db3), "out": str(args.out), "commands_replayed": False}))


if __name__ == "__main__":
    main()
