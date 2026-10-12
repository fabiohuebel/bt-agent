"""Extract later physical skill profiles without replacing Capture27 defaults."""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

from extract_robotics import relocate

AGENT = "523641f52c4619da5f3c901f736a407e32e52c01"
WORLD = "5321e348c2a49a3dd7017726ea9240738b929e01"
PREFIX = "ros/ros2_ws/src/lam_v2/lam_v2_tiamat/src/lam_v2_tiamat/"


def select_nodes(text, names):
    lines = text.splitlines(keepends=True)
    return "\n\n".join(
        "".join(
            lines[
                min([n.lineno] + [d.lineno for d in getattr(n, "decorator_list", [])])
                - 1 : n.end_lineno
            ]
        )
        for n in ast.parse(text).body
        if getattr(n, "name", None) in names
    )


def without_tests(text, names):
    lines = text.splitlines(keepends=True)
    remove = set()
    for n in ast.walk(ast.parse(text)):
        if isinstance(n, ast.FunctionDef) and n.name in names:
            start = min([n.lineno] + [d.lineno for d in n.decorator_list])
            remove.update(range(start - 1, n.end_lineno))
    return "".join(line for i, line in enumerate(lines) if i not in remove)


def transform_source(text):
    tree = ast.parse(text)
    lines = text.splitlines(keepends=True)
    wanted = {
        "HAB_TO_ROS_BASIS",
        "_rotation_matrix_y",
        "Transform",
        "build_transform",
        "quat_from_yaw",
    }
    selected = []
    for node in tree.body:
        name = getattr(node, "name", None)
        if isinstance(node, ast.Assign):
            name = getattr(node.targets[0], "id", None)
        if name in wanted:
            start = min([node.lineno] + [d.lineno for d in getattr(node, "decorator_list", [])])
            selected.append("".join(lines[start - 1 : node.end_lineno]))
    if len(selected) != len(wanted):
        raise ValueError("Transform source closure changed")
    return (
        "from __future__ import annotations\nimport math\nfrom dataclasses import dataclass\n"
        "from typing import List\nimport numpy as np\n\n" + "\n\n".join(selected) + "\n"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--agent-repo", type=Path, required=True)
    parser.add_argument("--world-repo", type=Path, required=True)
    parser.add_argument("--check", action="store_true")
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    package = root / "packages/bt-agent-robotics/src/bt_agent_robotics"
    manifest = root / "reproduction/interaction-robotics-source.json"
    previous = (
        {r["destination"]: r["sha256"] for r in json.loads(manifest.read_text())["files"]}
        if manifest.exists()
        else {}
    )
    records = []

    def save(repo, commit, source, destination, transform):
        raw = subprocess.check_output(["git", "show", f"{commit}:{source}"], cwd=repo)
        data = transform(raw.decode()).encode()
        path = package / destination
        if args.check:
            assert path.read_bytes() == data, destination
        else:
            if (
                path.exists()
                and path.read_bytes() != data
                and hashlib.sha256(path.read_bytes()).hexdigest() != previous.get(destination)
            ):
                raise RuntimeError(f"Refusing edited extraction: {path}")
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(data)
        records.append(
            dict(
                commit=commit,
                source=source,
                destination=destination,
                source_sha256=hashlib.sha256(raw).hexdigest(),
                sha256=hashlib.sha256(data).hexdigest(),
            )
        )

    def agent(text):
        return (
            relocate(text.encode())
            .decode()
            .replace("lam_v2_tiamat.ros.push_feedback", "bt_agent_robotics._push.feedback")
            .replace("lam_v2_tiamat.ros.push_telemetry", "bt_agent_robotics._push.telemetry")
            .replace("research.habitat_push.prototype", "bt_agent_robotics._push.controller")
            .replace("examples.habitat_gt_agent_artifacts", "bt_agent_robotics._push.transform")
        )

    save(args.agent_repo, AGENT, PREFIX + "bt_skills/query.py", "_interaction_query.py", agent)
    save(
        args.agent_repo,
        AGENT,
        PREFIX + "bt_skills/push.py",
        "_push/skill.py",
        lambda t: agent(t).replace(
            "from bt_agent_robotics.skills.navigate import (",
            "from bt_agent_robotics._push.navigation import (",
        ),
    )
    save(
        args.agent_repo,
        AGENT,
        PREFIX + "bt_skills/navigate.py",
        "_push/navigation.py",
        lambda t: "from __future__ import annotations\nimport math, time, logging\nimport py_trees\n"
        "from bt_agent_robotics.skills.navigate import (_ALIGN_TOL_RAD, _finite_float, _yaw_difference, "
        "_pose_is_stale, _nav_debug_log, _ALIGN_TIMEOUT_S, _ALIGN_MAX_W, _ALIGN_KP, _FollowPath, _TerminalDiscipline)\n"
        "logger = logging.getLogger(__name__)\n\n" + select_nodes(t, {"_AlignHeading"}) + "\n",
    )
    for source, target in (
        ("push_feedback", "feedback"),
        ("push_telemetry", "telemetry"),
        ("push_speed_limit", "speed_limit"),
        ("ros_local_planner", "ros_navigation"),
    ):
        save(
            args.agent_repo,
            AGENT,
            PREFIX + "ros/" + source + ".py",
            "_push/" + target + ".py",
            agent,
        )
    save(
        args.world_repo,
        WORLD,
        "research/habitat_push/prototype.py",
        "_push/controller.py",
        lambda t: t.replace(
            "from habitat_environment.sim.push_motion import measured_rail_step\n", ""
        ),
    )
    save(
        args.world_repo,
        WORLD,
        "examples/habitat_gt_agent_artifacts.py",
        "_push/transform.py",
        transform_source,
    )
    for name in ("test_query_refresh.py",):
        source = PREFIX.replace("src/lam_v2_tiamat/", "tests/bt_skills/") + name
        save(
            args.agent_repo,
            AGENT,
            source,
            "../../tests/interaction_reference/" + name,
            lambda t: agent(t).replace(
                "bt_agent_robotics.skills.query", "bt_agent_robotics._interaction_query"
            ),
        )
    for name in (
        "test_push_skill",
        "test_push_feedback",
        "test_push_speed_limit",
        "test_push_contact_control",
        "test_push_heading_precision",
        "test_push_episode_resume",
    ):
        source = PREFIX.replace("src/lam_v2_tiamat/", "tests/ros/") + name + ".py"

        def tests(t):
            result = (
                agent(t)
                .replace("bt_agent_robotics.skills.push", "bt_agent_robotics._push.skill")
                .replace(
                    "from bt_agent_robotics.skills import push",
                    "from bt_agent_robotics._push import skill as push",
                )
                .replace(
                    "lam_v2_tiamat.ros.push_speed_limit", "bt_agent_robotics._push.speed_limit"
                )
                .replace(
                    "from bt_agent_robotics.skills.navigate import _AlignHeading",
                    "from bt_agent_robotics._push.navigation import _AlignHeading",
                )
            )
            for old, new in (("push_feedback", "feedback"), ("push_speed_limit", "speed_limit")):
                result = result.replace(
                    'Path(__file__).resolve().parents[2] / "src/lam_v2_tiamat/ros/' + old + '.py"',
                    'Path(__import__("bt_agent_robotics").__file__).parent / "_push/'
                    + new
                    + '.py"',
                )
            result = result.replace("from lam_v2_tiamat.ros.flat_actions import parse_action\n", "")
            result = without_tests(
                result,
                {
                    "test_bridge_requires_both_profile_and_registration",
                    "test_push_remains_rejected_without_registration",
                    "test_enabled_grammar_resolves_same_target_parameter",
                },
            )
            if "from research.habitat_push.test_episode_resume import fixture" in result:
                original = subprocess.check_output(
                    ["git", "show", f"{WORLD}:research/habitat_push/test_episode_resume.py"],
                    cwd=args.world_repo,
                ).decode()
                result = result.replace(
                    "from research.habitat_push.test_episode_resume import fixture",
                    "from bt_agent_robotics._push.controller import Affordance, Observation\n\n"
                    + select_nodes(original, {"fixture"}),
                )
            return result

        save(
            args.agent_repo,
            AGENT,
            source,
            "../../tests/interaction_reference/" + name + ".py",
            tests,
        )
    record = dict(
        schema="interaction-robotics-source/v1",
        agent_commit=AGENT,
        world_commit=WORLD,
        transformations=[
            "Namespace relocation",
            "Unused simulator import removed from controller",
            "Standalone geometric transform and alignment closures",
            "Three flat-baseline parser/dispatch tests omitted; resume fixture inlined from frozen world test",
        ],
        files=records,
    )
    if args.check:
        assert json.loads(manifest.read_text()) == record
    else:
        (package / "_push/__init__.py").write_text(
            '"""Versioned measured-push implementation; never enabled by import."""\n'
        )
        manifest.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(dict(files=len(records), checked=args.check)))


if __name__ == "__main__":
    main()
