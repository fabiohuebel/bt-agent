"""Reproducible mechanical extraction; no prompt or controller redesign.

Read immutable Git objects, never the source worktree. The only transformation
is the module prefix lam_v2 -> bt_agent._engine (not lam_v2_tiamat etc.).
The private engine is a transitional extraction boundary, not the public API.
"""

import argparse
import ast
import hashlib
import json
from pathlib import Path
import subprocess

COMMIT = "62056af6fcb9ac3ec611760415c0c4f2f91a1fb6"
PREFIX = "ros/ros2_ws/src/lam_v2/lam_v2_pycore/"
DEFERRED = {
    "tests/test_framework_agnosticity.py": "Original filesystem-layout audit; replaced by release import-boundary test",
    "tests/skills/test_unified_skill_schema_parity.py": "Physical skill schemas; port with robotics package",
    "tests/model/test_scene_graph_llm_hook.py": "Physical scene-graph integration; port with robotics package",
    "tests/model/test_claude_headless.py": "Optional CLI provider integration is outside first core milestone",
}


def relocate(raw):
    return raw.replace(b"lam_v2.", b"bt_agent._engine.").replace(
        b"from lam_v2 import", b"from bt_agent._engine import"
    )


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-repo", type=Path, required=True)
    parser.add_argument(
        "--reference-only",
        action="store_true",
        help="Regenerate comparison fixtures without touching the maintained engine",
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    if not args.reference_only and (root / "reproduction/core-patches.json").exists():
        # A full extraction would rewrite patched files and restore recorded removals.
        parser.error("release patches exist; use --reference-only")
    old_manifest = root / "reproduction/core-source.json"
    previous = (
        {
            r["destination"]: r["extracted_sha256"]
            for r in json.loads(old_manifest.read_text())["files"]
        }
        if old_manifest.exists()
        else {}
    )
    files = subprocess.check_output(
        ["git", "ls-tree", "-r", "-z", COMMIT, "--", PREFIX + "src/lam_v2", PREFIX + "tests"],
        cwd=args.source_repo,
    ).split(b"\0")
    manifest = {
        "source_commit": COMMIT,
        "source_prefix": PREFIX,
        "transform": "module-prefix-only: lam_v2. -> bt_agent._engine.",
        "deferred_tests": DEFERRED,
        "files": [],
    }
    for entry in files:
        if not entry:
            continue
        meta, path = entry.split(b"\t", 1)
        blob = meta.split()[2].decode()
        source_path = path.decode()
        relative = source_path.removeprefix(PREFIX)
        raw = subprocess.check_output(["git", "cat-file", "blob", blob], cwd=args.source_repo)
        reference = root / ".build/reference" / relative
        reference.parent.mkdir(parents=True, exist_ok=True)
        reference.write_bytes(raw)
        if args.reference_only:
            continue
        if relative in DEFERRED:
            continue
        if relative.startswith("src/lam_v2/"):
            dest = Path("packages/bt-agent/src/bt_agent/_engine") / relative.removeprefix(
                "src/lam_v2/"
            )
        else:
            dest = Path("tests/reference") / relative.removeprefix("tests/")
        transformed = relocate(raw)
        if relative.startswith("tests/"):
            transformed = transformed.replace(
                b'Path(__file__).resolve().parents[2] / "src/lam_v2/',
                b'Path(__file__).resolve().parents[3] / "packages/bt-agent/src/bt_agent/_engine/',
            )
            if relative == "tests/skills/test_publish_results.py":
                # Generic hook tests need only handle/label; physical skill tests
                # in this file stay explicitly deferred by tests/conftest.py.
                transformed = transformed.replace(
                    b"from lam_v2_tiamat.bt_skills.query import SceneObject\n",
                    b"from bt_agent._engine.skills.object_store import SkillObject as SceneObject\n",
                )
        target = root / dest
        if target.exists() and target.read_bytes() not in (transformed, relocate(raw)):
            if hashlib.sha256(target.read_bytes()).hexdigest() != previous.get(str(dest)):
                raise RuntimeError(f"Refuse to replace an edited extraction: {target}")
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(transformed)
        manifest["files"].append(
            {
                "source": source_path,
                "git_blob": blob,
                "destination": str(dest),
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "extracted_sha256": hashlib.sha256(transformed).hexdigest(),
            }
        )
    extras = {
        "ros/ros2_ws/src/lam_v2/lam_v2_textrung/src/lam_v2_textrung/bt_skills/report.py": "report_skill.py",
        "ros/ros2_ws/src/lam_v2/lam_v2_textrung/src/lam_v2_textrung/bt_skills/_helpers.py": "_helpers.py",
        "ros/ros2_ws/src/lam_v2/lam_v2_tiamat/src/lam_v2_tiamat/sim_experiment.py": "episode_bridge.py",
    }
    for source, filename in extras.items():
        raw = subprocess.check_output(["git", "show", f"{COMMIT}:{source}"], cwd=args.source_repo)
        transformed = raw
        transform = "module-prefix-only"
        if filename == "episode_bridge.py":
            tree = ast.parse(raw.decode())
            node = next(
                n for n in tree.body if isinstance(n, ast.ClassDef) and n.name == "RosClaimBridge"
            )
            transformed = (
                "\n".join(raw.decode().splitlines()[node.lineno - 1 : node.end_lineno]) + "\n"
            ).encode()
            transform = "extract RosClaimBridge class verbatim (no ROS imports)"
            fn = next(
                n
                for n in tree.body
                if isinstance(n, ast.FunctionDef) and n.name == "frozen_agent_config"
            )
            ref_profile = (
                "PROMPT = 'tiamat_agent_submit_v5'\n\n"
                + "\n".join(raw.decode().splitlines()[fn.lineno - 1 : fn.end_lineno])
                + "\n"
            )
            support = root / ".build/reference/ref_support"
            support.mkdir(parents=True, exist_ok=True)
            (support / "profile.py").write_text(ref_profile)
        support = root / ".build/reference/ref_support"
        support.mkdir(parents=True, exist_ok=True)
        (support / "__init__.py").write_text("")
        (support / filename).write_bytes(transformed)
        if args.reference_only:
            continue
        transformed = relocate(transformed)
        dest = Path("packages/bt-agent/src/bt_agent/_engine") / filename
        (root / dest).write_bytes(transformed)
        manifest["files"].append(
            {
                "source": source,
                "destination": str(dest),
                "transform": transform,
                "source_sha256": hashlib.sha256(raw).hexdigest(),
                "extracted_sha256": hashlib.sha256(transformed).hexdigest(),
            }
        )
    if args.reference_only:
        print(json.dumps({"source_commit": COMMIT, "reference_only": True}))
        return
    out = root / "reproduction/core-source.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(manifest, indent=2) + "\n")
    print(
        json.dumps(
            {
                "source_commit": COMMIT,
                "extracted_files": len(manifest["files"]),
                "deferred_test_files": len(DEFERRED),
            }
        )
    )


if __name__ == "__main__":
    main()
