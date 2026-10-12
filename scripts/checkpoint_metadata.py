"""Record the installed qualification environment and artifact identities."""

import hashlib
import argparse
from datetime import datetime, timezone
import importlib.metadata
import json
import platform
import subprocess
import sys
from pathlib import Path
from xml.etree import ElementTree

import bt_agent

parser = argparse.ArgumentParser()
parser.add_argument("--references-rerun", action="store_true")
args = parser.parse_args()

root = Path(__file__).resolve().parents[1]
artifacts = root / "artifacts"
packages = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
(artifacts / "tested-environment.txt").write_text(packages)
cases = ElementTree.parse(artifacts / "core-tests.xml").findall(".//testcase")
failures = [c for c in cases if c.find("failure") is not None or c.find("error") is not None]
skipped = [c for c in cases if c.find("skipped") is not None]
physical_skips = [
    c for c in skipped if "Physical skill integration" in c.find("skipped").get("message", "")
]
record = {
    "recorded_utc": datetime.now(timezone.utc).isoformat(),
    "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=root).decode().strip(),
    "worktree_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=root)),
    "references_rerun": args.references_rerun,
    "python": platform.python_version(),
    "package_version": bt_agent.__version__,
    "installed_package_path": bt_agent.__file__,
    "frozen_reference_commit": "62056af6fcb9ac3ec611760415c0c4f2f91a1fb6",
    "scope": "Offline core extraction, not live-model/simulator/robot qualification",
    "tests": {
        "passed": len(cases) - len(failures) - len(skipped),
        "deferred_physical": len(physical_skips),
        "other_skipped_or_xfailed": len(skipped) - len(physical_skips),
        "failures_or_errors": len(failures),
    },
    "reference_parity": (
        json.loads((artifacts / "reference-parity.json").read_text())
        if (artifacts / "reference-parity.json").exists()
        else None
    ),
    "artifact_sha256": {
        str(path.relative_to(root)): hashlib.sha256(path.read_bytes()).hexdigest()
        for path in [
            artifacts / "core-tests.xml",
            artifacts / "tested-environment.txt",
            *sorted((artifacts / "wheels").glob("*.whl")),
        ]
    },
}
robotics_xml = artifacts / "robotics-tests.xml"
if robotics_xml.exists():
    robotics_cases = ElementTree.parse(robotics_xml).findall(".//testcase")
    robotics_failed = [
        c for c in robotics_cases if c.find("failure") is not None or c.find("error") is not None
    ]
    robotics_skipped = [c for c in robotics_cases if c.find("skipped") is not None]
    record["robotics_tests"] = {
        "package_version": importlib.metadata.version("bt-agent-robotics"),
        "passed": len(robotics_cases) - len(robotics_failed) - len(robotics_skipped),
        "failed": len(robotics_failed),
        "skipped": len(robotics_skipped),
        "scope": "Fake-backend unit tests, not physical validation",
    }
    record["robotics_parity"] = (
        json.loads((artifacts / "robotics-parity.json").read_text())
        if (artifacts / "robotics-parity.json").exists()
        else None
    )
    record["artifact_sha256"]["artifacts/robotics-tests.xml"] = hashlib.sha256(
        robotics_xml.read_bytes()
    ).hexdigest()
    failures.extend(robotics_failed)
(artifacts / "qualification.json").write_text(json.dumps(record, indent=2) + "\n")
print(json.dumps(record, indent=2))
if failures:
    raise SystemExit("Qualification contains test failures")
