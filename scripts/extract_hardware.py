"""Preserve the deployed hardware skill/control profile without changing defaults."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

from extract_robotics import relocate
from robotics_patches import artifact_writers, navigation_debug

COMMIT = "517c18ad9f861fce657643f4ee263459279fc9bd"
PREFIX = "ros/ros2_ws/src/lam_v2/lam_v2_tiamat/"
MODULES = (
    "battery_guard.py",
    "battery_hold_state.py",
    "battery_posture.py",
    "bt_controls.py",
    "hardware_execution_guard.py",
    "hardware_survey_description.py",
    "capture_contract.py",
    "ros/bt_controls_bridge.py",
    "ros/battery_guard_bridge.py",
    "ros/ros_local_planner.py",
    "ros/cmd_vel_mux.py",
    "ros/cmd_vel_mux_logic.py",
    "bt_skills/navigate.py",
    "bt_skills/press.py",
    "bt_skills/capture.py",
    "bt_skills/query.py",
)
TESTS = (
    "test_battery_guard.py",
    "test_battery_hold_state.py",
    "test_battery_posture.py",
    "test_bt_controls.py",
    "test_battery_guard_bridge.py",
    "test_hardware_execution_guard.py",
    "test_hardware_survey_description.py",
    "bt_skills/test_capture_local_inspection.py",
    "bt_skills/test_navigate.py",
    "bt_skills/test_navigate_confirm.py",
    "bt_skills/test_press.py",
    "bt_skills/test_capture.py",
    "bt_skills/test_capture_skill_call_id.py",
    "bt_skills/test_query_exclude.py",
    "ros/test_cmd_vel_mux_logic.py",
    "ros/test_battery_mux_output.py",
)


def transform(raw):
    text = raw.decode()
    text = text.replace(
        'Path(__file__).resolve().parents[2] / "src/lam_v2_tiamat/ros/cmd_vel_mux.py"',
        'Path(__import__("bt_agent_robotics").__file__).parent / "_hardware/ros/cmd_vel_mux.py"',
    )
    text = text.replace("from test_bt_controls import", "from .test_bt_controls import")
    for name in ("navigate", "press", "capture", "query"):
        text = text.replace(
            "lam_v2_tiamat.bt_skills." + name, "bt_agent_robotics._hardware.skills." + name
        )
    for name in (
        "battery_guard",
        "battery_hold_state",
        "battery_posture",
        "bt_controls",
        "hardware_execution_guard",
        "hardware_survey_description",
        "capture_contract",
    ):
        text = text.replace("lam_v2_tiamat." + name, "bt_agent_robotics._hardware." + name)
    text = text.replace("lam_v2_tiamat.ros.", "bt_agent_robotics._hardware.ros.")
    text = text.replace(
        "from lam_v2_tiamat.bt_skills import navigate",
        "from bt_agent_robotics._hardware.skills import navigate",
    )
    text = text.replace(
        "from lam_v2_tiamat.bt_skills import press",
        "from bt_agent_robotics._hardware.skills import press",
    )
    text = text.replace(
        "from lam_v2_tiamat.sim_experiment import frozen_agent_config",
        "from bt_agent.profiles import core_config as frozen_agent_config",
    )
    return relocate(text.encode())


def survey_tests(raw):
    text = raw.decode()
    lines = text.splitlines(keepends=True)
    keep = {"strip_descriptions", "test_original_default_and_experiment_changes_only_help"}
    bodies = [
        "".join(lines[node.lineno - 1 : node.end_lineno])
        for node in ast.parse(text).body
        if getattr(node, "name", None) in keep
    ]
    return (
        "from copy import deepcopy\nimport pytest\n"
        "from bt_agent_robotics.skills.surround_survey import SurroundSurveySkill\n"
        "from bt_agent_robotics._hardware.hardware_survey_description import VisualQuestionSurveySkill, make_survey\n\n"
        + "\n\n".join(body.rstrip() for body in bodies)
        + "\n"
    ).encode()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    package = root / "packages/bt-agent-robotics"
    receipt = root / "reproduction/hardware-source.json"
    previous = (
        {r["destination"]: r["sha256"] for r in json.loads(receipt.read_text())["files"]}
        if receipt.exists()
        else {}
    )
    rows = []

    def save(source, destination, patch=None):
        raw = subprocess.check_output(["git", "show", f"{COMMIT}:{source}"], cwd=args.source_repo)
        data = transform(raw) if source.endswith(".py") else raw
        if patch:
            data = patch(data)
        path = package / destination
        name = str(path.relative_to(root))
        if args.check:
            if path.read_bytes() != data:
                raise ValueError("Changed hardware extraction: " + name)
        else:
            if (
                path.exists()
                and path.read_bytes() != data
                and hashlib.sha256(path.read_bytes()).hexdigest() != previous.get(name)
            ):
                raise ValueError("Refusing edited hardware extraction: " + name)
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        rows.append(
            dict(
                source=source,
                source_sha256=hashlib.sha256(raw).hexdigest(),
                destination=name,
                sha256=hashlib.sha256(data).hexdigest(),
            )
        )

    for name in MODULES:
        patch = (
            (lambda raw: artifact_writers(raw, "capture.py"))
            if name == "bt_skills/capture.py"
            else None
        )
        if name == "bt_skills/navigate.py":
            patch = navigation_debug
        save(
            PREFIX + "src/lam_v2_tiamat/" + name,
            "src/bt_agent_robotics/_hardware/" + name.replace("bt_skills/", "skills/"),
            patch,
        )
    for name in TESTS:
        save(
            PREFIX + "tests/" + name,
            "tests/hardware_reference/" + name.replace("bt_skills/", ""),
            survey_tests if name == "test_hardware_survey_description.py" else None,
        )
    for name in ("capture", "navigate", "press", "query", "surround_survey"):
        save(
            PREFIX + "src/lam_v2_tiamat/bt_skills/docs/" + name + ".md",
            "src/bt_agent_robotics/_hardware/docs/" + name + ".md",
        )
    for name in ("navigate_tree.svg", "navigate_tree_bt.svg"):
        save(
            PREFIX + "src/lam_v2_tiamat/bt_skills/docs/" + name,
            "src/bt_agent_robotics/_hardware/docs/" + name,
        )
    result = dict(
        schema="bt-agent-hardware-source/v1",
        commit=COMMIT,
        files=rows,
        transformations="Namespace relocation; explicit capture/navigation debug writers; frozen config alias; pure survey-help test retained",
        excluded_tests={
            "test_hardware_survey_description.py": [
                "test_variant_is_explicit_and_has_distinct_identity",
                "test_manifest_labels_loaded_experiment_and_rejects_false_label",
            ]
        },
        preserved_defaults="Capture27 and later Habitat profiles unchanged",
        scope="Hardware-profile extraction, not live deployment qualification",
    )
    if args.check:
        if json.loads(receipt.read_text()) != result:
            raise ValueError("Hardware source receipt changed")
    else:
        receipt.write_text(json.dumps(result, indent=2) + "\n")
    runtime = {**result, "files": [row for row in rows if "/src/" in row["destination"]]}
    runtime_path = package / "src/bt_agent_robotics/hardware-source.json"
    if args.check:
        if json.loads(runtime_path.read_text()) != runtime:
            raise ValueError("Hardware runtime source receipt changed")
    else:
        runtime_path.write_text(json.dumps(runtime, indent=2) + "\n")
    print(json.dumps(dict(files=len(rows), checked=args.check, commit=COMMIT)))


if __name__ == "__main__":
    main()
