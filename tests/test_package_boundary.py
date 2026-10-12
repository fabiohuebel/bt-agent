import hashlib
import json
from pathlib import Path
import subprocess
import sys


def _expected_hashes(root):
    manifest = json.loads((root / "reproduction/core-source.json").read_text())
    expected = {row["destination"]: row["extracted_sha256"] for row in manifest["files"]}
    patches = root / "reproduction/core-patches.json"
    if patches.exists():
        for patch in json.loads(patches.read_text())["patches"]:
            assert expected[patch["destination"]] == patch["base_sha256"]
            expected[patch["destination"]] = patch["release_sha256"]
    return expected


def test_mechanically_extracted_files_match_manifest():
    root = Path(__file__).resolve().parents[1]
    for destination, expected in _expected_hashes(root).items():
        if expected is None:  # recorded removal
            assert not (root / destination).exists(), destination
            continue
        assert hashlib.sha256((root / destination).read_bytes()).hexdigest() == expected


def test_loaded_engine_not_only_checkout_matches_manifest():
    import bt_agent

    root = Path(__file__).resolve().parents[1]
    prefix = "packages/bt-agent/src/bt_agent/"
    for destination, expected in _expected_hashes(root).items():
        if destination.startswith(prefix):
            installed = Path(bt_agent.__file__).parent / destination.removeprefix(prefix)
            if expected is None:  # recorded removal
                assert not installed.exists(), destination
                continue
            assert hashlib.sha256(installed.read_bytes()).hexdigest() == expected


def test_core_and_offline_demo_do_not_import_ros_or_old_repositories():
    source = """
import importlib.abc, sys, runpy
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = {'rclpy','lam_v2','lam_v2_tiamat','lam_v2_textrung',
                     'tiamat_text','habitat_sim','isaacsim','torch'}
        if fullname.split('.')[0] in forbidden:
            raise ImportError('Forbidden release dependency: ' + fullname)
sys.meta_path.insert(0, Block())
runpy.run_path('examples/cpu_demo.py', run_name='__main__')
"""
    subprocess.run(
        [sys.executable, "-c", source],
        check=True,
        capture_output=True,
        cwd=Path(__file__).resolve().parents[1],
    )
