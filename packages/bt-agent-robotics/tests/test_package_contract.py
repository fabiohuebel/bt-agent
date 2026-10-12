import hashlib
import json
from pathlib import Path
import subprocess
import sys

import bt_agent_robotics
from bt_agent_robotics import inspection_skills


def test_factory_keeps_reference_order_without_creating_services(monkeypatch):
    monkeypatch.setenv("LAM_NAV_CONFIRM", "0")
    skills = inspection_skills(
        navigation=None, path_planning=None, scene_memory=None, perception=None, register_docs=False
    )
    assert [s.name for s in skills] == [
        "query",
        "navigate",
        "press",
        "report",
        "surround_survey",
        "capture",
    ]


def test_loaded_robotics_files_match_source_manifest():
    root = Path(__file__).resolve().parents[3]
    manifest = json.loads((root / "reproduction/robotics-source.json").read_text())
    prefix = "packages/bt-agent-robotics/src/bt_agent_robotics/"
    for row in manifest["files"]:
        if row["destination"].startswith(prefix):
            path = Path(bt_agent_robotics.__file__).parent / row["destination"].removeprefix(prefix)
            assert hashlib.sha256(path.read_bytes()).hexdigest() == row["extracted_sha256"]


def test_import_and_registration_do_not_require_ros_or_old_repositories():
    source = """
import importlib.abc, os, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, path=None, target=None):
        forbidden = {'rclpy', 'lam_v2', 'lam_v2_tiamat', 'tiamat_text',
                     'habitat_sim', 'isaacsim', 'torch'}
        if fullname.split('.')[0] in forbidden:
            raise ImportError('Forbidden dependency: ' + fullname)
sys.meta_path.insert(0, Block())
os.environ['LAM_NAV_CONFIRM'] = '0'
from bt_agent_robotics import inspection_skills
assert len(inspection_skills(navigation=None, path_planning=None,
                            scene_memory=None, perception=None)) == 6
"""
    subprocess.run([sys.executable, "-I", "-c", source], check=True, capture_output=True)
