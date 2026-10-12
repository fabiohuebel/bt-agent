"""Extract the evaluated graph planner without introducing ROS or benchmark imports."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess

COMMIT = "62056af6fcb9ac3ec611760415c0c4f2f91a1fb6"
PREFIX = "ros/ros2_ws/src/lam_v2/lam_v2_tiamat/"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", required=True, type=Path)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    source = PREFIX + "src/lam_v2_tiamat/bt_skills/utils/nav_graph_planner.py"
    raw = subprocess.check_output(["git", "show", f"{COMMIT}:{source}"], cwd=args.source_repo)
    transformed = raw.replace(
        b"lam_v2_tiamat.bt_skills.utils.path_planner",
        b"bt_agent_robotics.skills.utils.path_planner",
    )
    target = root / "packages/bt-agent-robotics/src/bt_agent_robotics/_nav_graph_planner.py"
    manifest_path = root / "reproduction/navigation-source.json"
    sha = lambda value: hashlib.sha256(value).hexdigest()
    manifest = dict(
        commit=COMMIT,
        source=source,
        source_sha256=sha(raw),
        destination=str(target.relative_to(root)),
        extracted_sha256=sha(transformed),
        transformation="Import namespace only; all planner behavior retained",
    )
    if args.check:
        assert (
            target.read_bytes() == transformed and json.loads(manifest_path.read_text()) == manifest
        )
    else:
        if target.exists() and target.read_bytes() != transformed:
            previous = json.loads(manifest_path.read_text())
            if sha(target.read_bytes()) != previous["extracted_sha256"]:
                raise RuntimeError("Refusing to overwrite modified navigation source")
        target.write_bytes(transformed)
        manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    print(json.dumps(manifest))


if __name__ == "__main__":
    main()
