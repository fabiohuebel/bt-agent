"""Run the hardware ROS checks in isolated containers, never on a robot network.

Uses an already installed Python environment, local image and compiled test-only
Spot messages. Does not build, download, contact models or change robot state.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import time
import uuid

CASES = (
    "interventions",
    "critical",
    "lease-denied",
    "stale-battery",
    "invalid-battery",
    "duplicate-output",
    "sit-denied",
    "sit-unavailable",
    "episodes",
    "driver-episodes",
)
SCRIPTS = (
    "qualify_hardware_ros.py",
    "qualify_hardware_wiring.py",
    "qualify_hardware_episode.py",
)


def checked_directory(value):
    path = Path(value).absolute()
    if not path.is_dir() or path.is_symlink() or "," in str(path):
        raise ValueError("Expected an existing nonsymlink directory without commas: " + str(path))
    if len(path.parts) < 4:
        raise ValueError("Use narrow environment/workspace mounts, not a system or home root")
    return path


def validate_inputs(image, python, interfaces, roots, scripts):
    if not re.fullmatch(r"sha256:[0-9a-f]{64}", image):
        raise ValueError("Supply the immutable ID of an existing local image (sha256:...)")
    python = Path(python).absolute()  # retain venv prefix, not only resolved interpreter
    roots = tuple(checked_directory(root) for root in roots)
    if not python.is_file() or not any(python.is_relative_to(root) for root in roots):
        raise ValueError("The installed Python must be inside a declared read-only root")
    resolved = python.resolve()
    if not str(resolved).startswith("/usr/") and not any(
        resolved.is_relative_to(root) for root in roots
    ):
        raise ValueError("Also mount the Python base interpreter's directory read-only")
    interfaces = checked_directory(interfaces)
    receipt = json.loads((interfaces / "source-receipt.json").read_text())
    if (
        receipt.get("schema") != "spot-telemetry-test-interfaces/v1"
        or receipt.get("test_only") is not True
    ):
        raise ValueError(
            "Requires the exported test-only telemetry workspace, not robot interfaces"
        )
    if not (interfaces / "install/setup.bash").is_file():
        raise ValueError("Compile the test-only workspace before running this check")
    scripts = checked_directory(scripts)
    for name in SCRIPTS:
        if not (scripts / name).is_file():
            raise ValueError("Missing qualification script: " + name)
    for root in (*roots, interfaces, scripts):
        if any(
            root.is_relative_to(Path(prefix))
            for prefix in ("/dev", "/proc", "/sys", "/run", "/var/run")
        ):
            raise ValueError("Devices, kernel and socket directories are not permitted mounts")
    return python, interfaces, roots, scripts, receipt


def container_command(image, python, interfaces, roots, scripts, out, name, case):
    if case not in CASES:
        raise ValueError("Unknown offline case")
    script = (
        SCRIPTS[0]
        if case == "interventions"
        else SCRIPTS[2] if case in ("episodes", "driver-episodes") else SCRIPTS[1]
    )
    invocation = [str(python), "/checks/" + script, "--out", "/evidence/" + case]
    if case == "driver-episodes":
        invocation += ["--driver-inputs"]
    elif case not in ("interventions", "episodes"):
        invocation += ["--case", case]
    command = [
        "docker",
        "run",
        "--detach",
        "--pull",
        "never",
        "--name",
        name,
        "--user",
        f"{os.getuid()}:{os.getgid()}",
        "--network",
        "none",
        "--cap-drop",
        "ALL",
        "--security-opt",
        "no-new-privileges",
        "--read-only",
        "--tmpfs",
        "/tmp:rw,nosuid,size=256m",
        "--entrypoint",
        "/bin/bash",
    ]
    for source, target in [(root, root) for root in roots] + [
        (interfaces, Path("/work")),
        (scripts, Path("/checks")),
    ]:
        command += ["--mount", f"type=bind,src={source},dst={target},readonly"]
    command += ["--mount", f"type=bind,src={out},dst=/evidence"]
    for entry in (
        "ROS_DOMAIN_ID=97",
        "ROS_LOCALHOST_ONLY=1",
        "ROS_LOG_DIR=/evidence/ros-logs",
        "RMW_IMPLEMENTATION=rmw_fastrtps_cpp",
        "PYTHONDONTWRITEBYTECODE=1",
    ):
        command += ["--env", entry]
    command += [
        image,
        "-c",
        "source /opt/ros/humble/setup.bash && source /work/install/setup.bash && "
        + shlex.join([str(python), "-m", "pip", "list", "--format=json"])
        + " > /evidence/"
        + case
        + "-packages.json && exec "
        + shlex.join(invocation),
    ]
    return command


def run_case(command, name, out, case, timeout):
    started = time.monotonic()
    row = dict(case=case, passed=False, command=command)
    created = False
    try:
        launched = subprocess.run(command, text=True, capture_output=True, timeout=30)
        row["launch_exit"] = launched.returncode
        if launched.returncode:
            row["error"] = launched.stderr
            return row
        created = True
        row["container_id"] = launched.stdout.strip()
        waited = subprocess.run(
            ["docker", "wait", name], text=True, capture_output=True, timeout=timeout
        )
        if waited.returncode:
            raise RuntimeError("docker wait failed: " + waited.stderr)
        row["container_exit"] = int(waited.stdout.strip())
        evidence = out / case / "result.json"
        if evidence.is_file():
            result = json.loads(evidence.read_text())
            row["result"] = result
            row["passed"] = row["container_exit"] == 0 and result.get("passed") is True
        else:
            row["error"] = "No result.json; inspect the container log"
    except (OSError, ValueError, subprocess.SubprocessError, RuntimeError) as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        if created:
            # Only our uniquely named container is inspected/stopped. No global prune/stop.
            try:
                log = subprocess.run(["docker", "logs", name], capture_output=True, timeout=15)
                (out / (case + ".log")).write_bytes(log.stdout + log.stderr)
            except (OSError, subprocess.SubprocessError) as exc:
                row["log_error"] = str(exc)
                row["passed"] = False
        # A timed-out launch can still have created a container. Always attempt
        # cleanup of this invocation's unique name, even without an ID receipt.
        try:
            removed = subprocess.run(
                ["docker", "rm", "--force", name], capture_output=True, timeout=20
            )
            row["owned_container_removed"] = removed.returncode == 0
            if created and removed.returncode:
                row["passed"] = False
        except (OSError, subprocess.SubprocessError) as exc:
            row["cleanup_error"] = str(exc)
            row["passed"] = False
        row["elapsed_s"] = time.monotonic() - started
    return row


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--image", required=True)
    parser.add_argument("--python", required=True, type=Path)
    parser.add_argument("--test-workspace", required=True, type=Path)
    parser.add_argument("--read-only-root", action="append", required=True, type=Path)
    parser.add_argument("--out", required=True, type=Path)
    parser.add_argument("--case", choices=CASES, action="append")
    parser.add_argument("--timeout", type=int, default=90, help="Seconds per case")
    args = parser.parse_args()
    if args.timeout < 1:
        parser.error("--timeout must be positive")
    python, interfaces, roots, scripts, receipt = validate_inputs(
        args.image,
        args.python,
        args.test_workspace,
        args.read_only_root,
        Path(__file__).parent,
    )
    out = args.out.absolute()
    if (
        out.exists()
        or "," in str(out)
        or any(out.is_relative_to(root) for root in (*roots, interfaces, scripts))
    ):
        raise ValueError("Use a new output directory outside the read-only input mounts")
    subprocess.run(
        ["docker", "image", "inspect", args.image], check=True, capture_output=True, timeout=15
    )
    out.mkdir(parents=True, exist_ok=False)
    summary = dict(
        schema="hardware-offline-suite/v1",
        passed=False,
        image=args.image,
        python=str(python),
        test_interfaces=receipt,
        script_sha256={
            name: hashlib.sha256((scripts / name).read_bytes()).hexdigest() for name in SCRIPTS
        },
        physical_robot_validated=False,
        scope="Synthetic telemetry and model/vision, real isolated ROS/mux, fake Sit; no robot or model endpoint",
        cases=[],
    )
    try:
        for case in dict.fromkeys(args.case or CASES):
            name = "bt-offline-" + uuid.uuid4().hex
            command = container_command(
                args.image, python, interfaces, roots, scripts, out, name, case
            )
            row = run_case(command, name, out, case, args.timeout)
            summary["cases"].append(row)
            print(
                json.dumps({key: row[key] for key in ("case", "passed", "elapsed_s")}), flush=True
            )
            (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
        summary["passed"] = bool(summary["cases"]) and all(
            row["passed"] for row in summary["cases"]
        )
    finally:
        (out / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    raise SystemExit(0 if summary["passed"] else 1)


if __name__ == "__main__":
    main()
