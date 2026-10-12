import importlib.util
import json
from pathlib import Path
import subprocess
import zipfile

import pytest


@pytest.fixture
def audit():
    path = Path(__file__).parents[1] / "scripts/audit_release.py"
    spec = importlib.util.spec_from_file_location("release_audit", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_secret_values_are_not_reported(audit):
    fake = "ghp_" + "Q" * 36
    report = audit.scan_bytes(("header\n" + fake + "\n").encode())
    assert report == [dict(rule="github_token", level="credential", lines=[2], count=1)]
    assert fake not in json.dumps(report)
    assert fake not in audit.safe_label("folder/" + fake + ".txt")


def test_review_markers_are_not_treated_as_verified_credentials(audit):
    report = audit.scan_bytes(b"/home/example/code http://user:dummy@offline.invalid")
    assert {r["rule"] for r in report} == {"personal_path", "url_credentials"}
    assert all(r["level"] == "review" for r in report)


def test_history_finds_deleted_credential_without_echoing_it(audit, tmp_path):
    def run(*args):
        subprocess.run(["git", "-C", str(tmp_path), *args], check=True, capture_output=True)

    run("init", "-q")
    run("config", "user.name", "Fixture")
    run("config", "user.email", "fixture@example.invalid")
    token = "hf_" + "A" * 30
    path = tmp_path / "fixture.txt"
    path.write_text(token)
    run("add", ".")
    run("commit", "-qm", "fixture")
    path.write_text("removed")
    run("add", ".")
    run("commit", "-qm", "remove fixture value")
    report = audit.history_inventory(tmp_path, 1024)
    assert report["blobs_scanned"] == 2 and not report["shallow"]
    assert report["findings"][0]["findings"][0]["level"] == "credential"
    assert token not in json.dumps(report)


def test_archive_scan_does_not_extract_traversal(audit, tmp_path):
    path = tmp_path / "bad.whl"
    with zipfile.ZipFile(path, "w") as out:
        out.writestr("../escape.txt", "test")
    report = audit.archive_inventory(path, 1024)
    assert any(e["reason"] == "unsafe-or-duplicate-archive-path" for e in report["errors"])
    assert not (tmp_path.parent / "escape.txt").exists()


def test_oversized_archive_member_is_explicitly_omitted(audit, tmp_path):
    path = tmp_path / "large.whl"
    with zipfile.ZipFile(path, "w") as out:
        out.writestr("module.py", "x" * 100)
    report = audit.archive_inventory(path, 10)
    assert report["omitted"][0]["bytes"] == 100


def test_archive_metadata_cannot_echo_a_candidate_secret(audit, tmp_path):
    fake = "ghp_" + "M" * 36
    path = tmp_path / "fixture.whl"
    with zipfile.ZipFile(path, "w") as out:
        out.writestr("fixture.dist-info/METADATA", "Name: " + fake + "\nVersion: 1\n")
    report = audit.archive_inventory(path, 4096)
    assert fake not in json.dumps(report)
    assert "redacted" in report["metadata"]["name"]
