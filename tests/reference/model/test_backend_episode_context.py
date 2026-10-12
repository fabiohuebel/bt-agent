"""ModelBackend episode-context provider wiring + LAM_VLM_LOG_DIR env
resolution (physics-trace-recording, wave 1).

Every vlm_calls.jsonl entry must carry the episode context active at call
time when the bridge node has registered a provider; unregistered (the
default — baseline/non-ROS callers) must stay byte-identical to the
pre-wave-1 journal shape (no "episode_context" key at all).
"""

from __future__ import annotations

import json
import os
import subprocess
import sys

from bt_agent._engine.model import backend as backend_module
from bt_agent._engine.model.backend import ModelBackend


def _backend_config(model_name: str = "qwen35-vlm") -> dict:
    return {
        "endpoint": "http://model.test:8000/v1",
        "model_name": model_name,
        "agent_tier": "lam",
    }


def test_tags_omit_episode_context_key_when_no_provider_registered(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_module, "_VLM_LOG_DIR", str(tmp_path))
    backend = ModelBackend(_backend_config())

    assert "episode_context" not in backend._tags()


def test_tags_include_episode_context_from_registered_provider(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_module, "_VLM_LOG_DIR", str(tmp_path))
    backend = ModelBackend(_backend_config())
    stamp = {
        "run_id": "run-1",
        "task_id": "task-1",
        "task_index": 0,
        "attempt": 1,
        "out_tag": None,
        "context_valid": True,
    }
    backend.set_episode_context_provider(lambda: stamp)

    assert backend._tags()["episode_context"] == stamp


def test_tags_stay_null_when_provider_reports_no_context_received(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_module, "_VLM_LOG_DIR", str(tmp_path))
    backend = ModelBackend(_backend_config())
    backend.set_episode_context_provider(lambda: None)

    assert backend._tags()["episode_context"] is None


def test_tags_survive_a_raising_provider(tmp_path, monkeypatch):
    """A recorder failure must never disrupt the main VLM call path."""
    monkeypatch.setattr(backend_module, "_VLM_LOG_DIR", str(tmp_path))
    backend = ModelBackend(_backend_config())

    def _boom():
        raise RuntimeError("provider exploded")

    backend.set_episode_context_provider(_boom)

    assert "episode_context" not in backend._tags()


def test_dump_vlm_exchange_writes_episode_context_into_the_journal_line(tmp_path, monkeypatch):
    monkeypatch.setattr(backend_module, "_VLM_LOG_DIR", str(tmp_path))
    backend = ModelBackend(_backend_config())
    stamp = {
        "run_id": "run-1",
        "task_id": "task-1",
        "task_index": 0,
        "attempt": 1,
        "out_tag": None,
        "context_valid": True,
    }
    backend.set_episode_context_provider(lambda: stamp)

    backend_module._dump_vlm_exchange({"model": "m"}, {"choices": []}, tags=backend._tags())

    lines = (tmp_path / "vlm_calls.jsonl").read_text().splitlines()
    record = json.loads(lines[-1])
    assert record["episode_context"] == stamp


def test_default_vlm_log_dir_matches_pre_wave1_default_when_env_unset():
    """LAM_VLM_LOG_DIR unset -> byte-identical /tmp/lam_vlm_log default."""
    env = {k: v for k, v in os.environ.items() if k != "LAM_VLM_LOG_DIR"}
    result = subprocess.run(
        [sys.executable, "-c", "from bt_agent._engine.model import backend; print(backend._VLM_LOG_DIR)"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "/tmp/lam_vlm_log"


def test_lam_vlm_log_dir_env_override_is_honored():
    env = {**os.environ, "LAM_VLM_LOG_DIR": "/tmp/custom_vlm_log_dir_override"}
    result = subprocess.run(
        [sys.executable, "-c", "from bt_agent._engine.model import backend; print(backend._VLM_LOG_DIR)"],
        env=env,
        capture_output=True,
        text=True,
        check=True,
    )
    assert result.stdout.strip() == "/tmp/custom_vlm_log_dir_override"
