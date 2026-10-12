"""The offline runner must never acquire network, devices or a robot interface."""

import importlib.util
import json
from pathlib import Path
import subprocess
from types import SimpleNamespace

import pytest

SCRIPT = Path(__file__).resolve().parents[1] / "scripts/check_hardware_offline.py"
spec = importlib.util.spec_from_file_location("offline_runner", SCRIPT)
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


@pytest.fixture
def inputs(tmp_path):
    env = tmp_path / "env"
    env.mkdir()
    python = env / "python"
    python.touch()
    interfaces = tmp_path / "interfaces"
    (interfaces / "install").mkdir(parents=True)
    (interfaces / "install/setup.bash").touch()
    (interfaces / "source-receipt.json").write_text(
        json.dumps(
            {
                "schema": "spot-telemetry-test-interfaces/v1",
                "test_only": True,
            }
        )
    )
    scripts = tmp_path / "scripts"
    scripts.mkdir()
    for name in runner.SCRIPTS:
        (scripts / name).touch()
    return "sha256:" + "a" * 64, python, interfaces, [env], scripts


def test_validate_is_read_only(inputs, tmp_path):
    before = sorted(tmp_path.rglob("*"))
    runner.validate_inputs(*inputs)
    assert before == sorted(tmp_path.rglob("*"))


@pytest.mark.parametrize("image", ["latest", "repo:tag", "sha256:short"])
def test_reject_mutable_or_invalid_images(inputs, image):
    with pytest.raises(ValueError, match="immutable"):
        runner.validate_inputs(image, *inputs[1:])


def test_reject_real_workspace(inputs):
    (inputs[2] / "source-receipt.json").write_text('{"test_only": false}')
    with pytest.raises(ValueError, match="test-only"):
        runner.validate_inputs(*inputs)


def test_require_base_interpreter_mount(inputs, tmp_path):
    base = tmp_path / "base"
    base.mkdir()
    (base / "python").touch()
    inputs[1].unlink()
    inputs[1].symlink_to(base / "python")
    with pytest.raises(ValueError, match="base interpreter"):
        runner.validate_inputs(*inputs)
    runner.validate_inputs(*inputs[:3], inputs[3] + [base], inputs[4])


@pytest.mark.parametrize("case", runner.CASES)
def test_every_case_uses_isolated_container_and_bounded_test_paths(inputs, tmp_path, case):
    command = runner.container_command(*inputs, tmp_path / "evidence", "bt-offline-unit", case)
    assert command[command.index("--network") + 1] == "none"
    assert command[command.index("--pull") + 1] == "never"
    assert command[command.index("--user") + 1] == f"{runner.os.getuid()}:{runner.os.getgid()}"
    assert "--privileged" not in command and "--gpus" not in command and "--device" not in command
    assert "ROS_DOMAIN_ID=97" in command and "ROS_LOCALHOST_ONLY=1" in command
    mounts = [command[i + 1] for i, value in enumerate(command) if value == "--mount"]
    assert all(mount.endswith(",readonly") for mount in mounts[:-1])
    assert mounts[-1].endswith("dst=/evidence")
    assert "/checks/qualify_hardware_" in command[-1]
    assert f"/evidence/{case}" in command[-1]


def test_deadline_stops_only_owned_container_and_never_passes(monkeypatch, tmp_path):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["docker", "wait"]:
            raise subprocess.TimeoutExpired(command, 1)
        raw = kwargs.get("text") is not True
        return SimpleNamespace(returncode=0, stdout=b"" if raw else "id", stderr=b"" if raw else "")

    monkeypatch.setattr(runner.subprocess, "run", run)
    row = runner.run_case(["docker", "run"], "bt-offline-owned", tmp_path, "critical", 1)
    assert not row["passed"] and "TimeoutExpired" in row["error"]
    assert row["owned_container_removed"]
    assert calls[-1] == ["docker", "rm", "--force", "bt-offline-owned"]


def test_ambiguous_launch_also_cleans_up(monkeypatch, tmp_path):
    calls = []

    def run(command, **kwargs):
        calls.append(command)
        if command[:2] == ["docker", "run"]:
            raise subprocess.TimeoutExpired(command, 30)
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr(runner.subprocess, "run", run)
    row = runner.run_case(["docker", "run"], "bt-offline-owned", tmp_path, "critical", 1)
    assert not row["passed"] and row["owned_container_removed"]
    assert calls[-1] == ["docker", "rm", "--force", "bt-offline-owned"]


@pytest.mark.parametrize(
    "exit_code,result,passed", [(0, True, True), (1, True, False), (0, False, False)]
)
def test_both_exit_status_and_result_are_required(monkeypatch, tmp_path, exit_code, result, passed):
    (tmp_path / "critical").mkdir()
    (tmp_path / "critical/result.json").write_text(json.dumps({"passed": result}))

    def run(command, **kwargs):
        if command[:2] == ["docker", "wait"]:
            return SimpleNamespace(returncode=0, stdout=str(exit_code), stderr="")
        raw = kwargs.get("text") is not True
        return SimpleNamespace(returncode=0, stdout=b"" if raw else "id", stderr=b"" if raw else "")

    monkeypatch.setattr(runner.subprocess, "run", run)
    row = runner.run_case(["docker", "run"], "bt-offline-owned", tmp_path, "critical", 1)
    assert row["passed"] is passed
