"""Run installed-package core and robotics tests in separate pytest processes."""

import argparse
import os
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser()
parser.add_argument("--with-reference", action="store_true")
args = parser.parse_args()
root = Path(__file__).resolve().parents[1]
env = {**os.environ, "PYTEST_DISABLE_PLUGIN_AUTOLOAD": "1", "PYTHONDONTWRITEBYTECODE": "1"}
env.pop("PYTHONPATH", None)
for suite, report in [
    ("tests", "core-tests.xml"),
    ("packages/bt-agent-robotics/tests", "robotics-tests.xml"),
]:
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pytest",
            suite,
            "-q",
            "-p",
            "no:cacheprovider",
            f"--junitxml=artifacts/{report}",
        ],
        cwd=root,
        env=env,
        check=True,
    )
subprocess.run(
    [sys.executable, "-I", str(root / "examples/cpu_demo.py")], cwd=root, env=env, check=True
)
subprocess.run([sys.executable, "-m", "pip", "check"], cwd=root, env=env, check=True)
if args.with_reference:
    for script in ["check_reference_parity.py", "check_robotics_parity.py"]:
        subprocess.run(
            [sys.executable, str(root / "scripts" / script)], cwd=root, env=env, check=True
        )
metadata = [sys.executable, str(root / "scripts/checkpoint_metadata.py")]
if args.with_reference:
    metadata.append("--references-rerun")
subprocess.run(metadata, cwd=root, env=env, check=True)
