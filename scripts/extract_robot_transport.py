"""Extract shared camera wire codec and ROS waypoint adapter from the evaluated agent."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

COMMIT = "62056af6fcb9ac3ec611760415c0c4f2f91a1fb6"
PREFIX = "ros/ros2_ws/src/lam_v2/lam_v2_tiamat/src/lam_v2_tiamat/ros/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    package = root / "packages/bt-agent-robotics/src/bt_agent_robotics"
    manifest_path = root / "reproduction/transport-source.json"
    old = (
        {r["destination"]: r["sha256"] for r in json.loads(manifest_path.read_text())["files"]}
        if manifest_path.exists()
        else {}
    )
    files = []
    sha = lambda data: hashlib.sha256(data).hexdigest()
    for source, destination in (
        ("image_rotation.py", "_image_rotation.py"),
        ("ros_local_planner.py", "ros_navigation.py"),
    ):
        original = PREFIX + source
        raw = subprocess.check_output(["git", "show", f"{COMMIT}:{original}"], cwd=args.source_repo)
        output = raw.replace(
            b"lam_v2_tiamat.bt_skills.navigate", b"bt_agent_robotics.skills.navigate"
        )
        path = package / destination
        relative = str(path.relative_to(root))
        if args.check:
            assert path.read_bytes() == output, relative
        else:
            if (
                path.exists()
                and path.read_bytes() != output
                and sha(path.read_bytes()) != old.get(relative)
            ):
                raise RuntimeError(f"Refuse edited transport source: {path}")
            path.write_bytes(output)
        files.append(
            dict(source=original, destination=relative, source_sha256=sha(raw), sha256=sha(output))
        )
    receipt = dict(
        reference_commit=COMMIT,
        files=files,
        transformation="ROS navigation interface import only; camera codec byte-identical",
    )
    if args.check:
        assert json.loads(manifest_path.read_text()) == receipt
    else:
        manifest_path.write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt))


if __name__ == "__main__":
    main()
