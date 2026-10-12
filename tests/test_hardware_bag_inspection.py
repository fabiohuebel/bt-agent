import hashlib
import importlib.util
import json
from pathlib import Path
import sqlite3
import sys
from types import ModuleType, SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/inspect_hardware_bag.py"
spec = importlib.util.spec_from_file_location("bag_inspector", SCRIPT)
inspector = importlib.util.module_from_spec(spec)
spec.loader.exec_module(inspector)


@pytest.fixture
def closed_bag(tmp_path, monkeypatch):
    module = ModuleType("rosbags.typesys")
    module.Stores = SimpleNamespace(ROS2_HUMBLE="test-store")
    module.get_typestore = lambda _: SimpleNamespace(
        deserialize_cdr=lambda raw, _: SimpleNamespace(
            header=SimpleNamespace(
                frame_id=json.loads(raw)["frame"],
                stamp=SimpleNamespace(sec=1, nanosec=0),
            ),
            child_frame_id="body",
        )
    )
    monkeypatch.setitem(sys.modules, "rosbags", ModuleType("rosbags"))
    monkeypatch.setitem(sys.modules, "rosbags.typesys", module)
    path = tmp_path / "closed.db3"
    with sqlite3.connect(path) as db:
        db.execute("CREATE TABLE topics (id INTEGER, name TEXT, type TEXT)")
        db.execute(
            "CREATE TABLE messages (id INTEGER, topic_id INTEGER, timestamp INTEGER, data BLOB)"
        )
        db.executemany(
            "INSERT INTO topics VALUES(?,?,?)",
            [
                (1, "/spot/platform/odom", "nav_msgs/msg/Odometry"),
                (2, "/camera/frontleft/compressed", "sensor_msgs/msg/CompressedImage"),
            ],
        )
        db.executemany(
            "INSERT INTO messages VALUES(?,?,?,?)",
            [
                (1, 1, 1_100_000_000, b'{"frame":"vision"}'),
                (2, 2, 1_200_000_000, b'{"frame":"frontleft_fisheye"}'),
            ],
        )
    return path


def test_inspection_keeps_archive_bytes_and_reports_actual_topics(closed_bag):
    before = closed_bag.read_bytes()
    result = inspector.inspect(closed_bag)
    assert closed_bag.read_bytes() == before
    assert result["sha256"] == hashlib.sha256(before).hexdigest()
    assert result["commands_replayed"] is False
    odom = result["topics"]["/spot/platform/odom"]["samples"][0]
    assert odom["frame_id"] == "vision" and odom["bag_minus_source_s"] == pytest.approx(0.1)
    assert result["topics"]["/camera/frontleft/compressed"]["message_count"] == 1
    assert result["topics"]["/spot/camera/frontleft/image/compressed"] == {"present": False}


def test_live_wal_is_refused(closed_bag):
    Path(str(closed_bag) + "-wal").touch()
    with pytest.raises(ValueError, match="closed"):
        inspector.inspect(closed_bag)


def test_no_sample_fabricated_for_empty_topic(closed_bag):
    with sqlite3.connect(closed_bag) as db:
        db.execute("DELETE FROM messages")
    result = inspector.inspect(closed_bag)
    assert result["topics"]["/spot/platform/odom"]["samples"] == []
