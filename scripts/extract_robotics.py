"""Extract Capture27 skills; only debug-artifact routing is deliberately changed."""

import argparse
import hashlib
import json
from pathlib import Path
import subprocess
from robotics_patches import artifact_writers, navigation_debug

COMMIT = "62056af6fcb9ac3ec611760415c0c4f2f91a1fb6"
PREFIX = "ros/ros2_ws/src/lam_v2/lam_v2_tiamat/"
CORE = "ros/ros2_ws/src/lam_v2/lam_v2_pycore/"
SKILLS = {
    "capture.py",
    "navigate.py",
    "press.py",
    "query.py",
    "surround_survey.py",
    "image_inject.py",
    "utils/compute_pose.py",
    "utils/path_planner.py",
}
TESTS = {
    "test_capture.py",
    "test_capture_skill_call_id.py",
    "test_navigate.py",
    "test_navigate_confirm.py",
    "test_navigate_se2.py",
    "test_press.py",
    "test_query_exclude.py",
}
CORE_TESTS = {
    "test_publish_results.py",
    "test_registry_partition.py",
    "test_safeguard.py",
    "test_unified_skill_schema_parity.py",
}


def relocate(raw, public=True):
    raw = raw.replace(b"lam_v2_tiamat.bt_skills", b"bt_agent_robotics.skills").replace(
        b"lam_v2.", b"bt_agent._engine."
    )
    if not public:
        return raw
    return (
        raw.replace(b"from bt_agent._engine.skills.protocol import", b"from bt_agent.skills import")
        .replace(
            b"from bt_agent._engine.skills.object_store import", b"from bt_agent.skills import"
        )
        .replace(b"from bt_agent._engine.model.backend import", b"from bt_agent.models import")
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    package = root / "packages/bt-agent-robotics"
    manifest_path = root / "reproduction/robotics-source.json"
    previous = (
        {
            row["destination"]: row["extracted_sha256"]
            for row in json.loads(manifest_path.read_text())["files"]
        }
        if manifest_path.exists()
        else {}
    )
    names = (
        subprocess.check_output(
            ["git", "ls-tree", "-r", "--name-only", COMMIT, "--", PREFIX, CORE + "tests/skills"],
            cwd=args.source_repo,
        )
        .decode()
        .splitlines()
    )
    records = []
    for source in names:
        if source.startswith(PREFIX + "src/lam_v2_tiamat/bt_skills/"):
            relative = source.removeprefix(PREFIX + "src/lam_v2_tiamat/bt_skills/")
            if relative not in SKILLS and not (
                relative.startswith("docs/") and relative.endswith((".md", ".svg"))
            ):
                continue
            destination = package / "src/bt_agent_robotics/skills" / relative
        elif source.startswith(PREFIX + "tests/bt_skills/"):
            relative = source.removeprefix(PREFIX + "tests/bt_skills/")
            if relative not in TESTS and not relative.startswith("fixtures/"):
                continue
            destination = package / "tests/reference/skills" / relative
        elif source.startswith(CORE + "tests/skills/") and Path(source).name in CORE_TESTS:
            destination = package / "tests/reference/core" / Path(source).name
        else:
            continue
        raw = subprocess.check_output(["git", "show", f"{COMMIT}:{source}"], cwd=args.source_repo)
        transformed = (
            raw
            if source.endswith((".md", ".svg"))
            else relocate(raw, public="/tests/" not in str(destination))
        )
        if source.startswith(PREFIX + "src/lam_v2_tiamat/bt_skills/") and Path(source).name in {
            "capture.py",
            "surround_survey.py",
        }:
            transformed = artifact_writers(transformed, Path(source).name)
        if source == PREFIX + "src/lam_v2_tiamat/bt_skills/navigate.py":
            transformed = navigation_debug(transformed)
        if source.startswith(PREFIX + "src/lam_v2_tiamat/"):
            reference = (
                root
                / ".build/reference_robotics/lam_v2_tiamat"
                / source.removeprefix(PREFIX + "src/lam_v2_tiamat/")
            )
            reference.parent.mkdir(parents=True, exist_ok=True)
            reference.write_bytes(raw)
        if "/tests/" in str(destination):
            transformed = transformed.replace(
                b'Path(__file__).resolve().parents[2] / "src/lam_v2/',
                b'Path(__import__("bt_agent").__file__).parent / "_engine/',
            )
            transformed = transformed.replace(
                b"from test_navigate import", b"from .test_navigate import"
            )
        if destination.exists() and destination.read_bytes() != transformed:
            if hashlib.sha256(destination.read_bytes()).hexdigest() != previous.get(
                str(destination.relative_to(root))
            ):
                raise RuntimeError(f"Refuse to replace edited extraction: {destination}")
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_bytes(transformed)
        records.append(
            {
                "source": source,
                "destination": str(destination.relative_to(root)),
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "extracted_sha256": hashlib.sha256(transformed).hexdigest(),
            }
        )
    for name in [
        "__init__.py",
        "feature_flags.py",
        "bt_skills/__init__.py",
        "bt_skills/utils/__init__.py",
    ]:
        source = PREFIX + "src/lam_v2_tiamat/" + name
        result = subprocess.run(
            ["git", "show", f"{COMMIT}:{source}"], cwd=args.source_repo, capture_output=True
        )
        if result.returncode == 0:
            target = root / ".build/reference_robotics/lam_v2_tiamat" / name
            target.parent.mkdir(parents=True, exist_ok=True)
            target.write_bytes(result.stdout)
    receipt = {
        "reference_commit": COMMIT,
        "source_cohort": "Habitat Capture27; not all later physical/hardware variants",
        "reference_evidence": "habitat-four-arm-contract-20260912-capture27-s012-r1/root-review.json",
        "reference_evidence_sha256": "267a6fc081d11866fef4b471d5700ea100b23f66f9d60d5e2394464900782903",
        "transformation": "Module relocation; public core imports; test source-path and sibling-import relocation; explicit debug-writer callbacks replace shared-directory dumps",
        "release_patches": {
            "script": "scripts/robotics_patches.py",
            "sha256": hashlib.sha256(
                (root / "scripts/robotics_patches.py").read_bytes()
            ).hexdigest(),
            "scope": "Optional debug artifact routing only; model-visible contracts unchanged",
        },
        "files": records,
        "not_yet_ported": [
            "Full ROS agent node/subscription launcher (navigation adapter extracted separately)",
            "FARM implementation",
            "robot driver/mux",
            "GT perception",
            "push skill and later interaction variants",
            "later hardware guards/capture/press fixes",
        ],
    }
    (root / "reproduction/robotics-source.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps({"files": len(records), "reference_commit": COMMIT}))


if __name__ == "__main__":
    main()
