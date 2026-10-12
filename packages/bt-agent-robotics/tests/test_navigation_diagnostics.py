from importlib import import_module

import pytest


@pytest.mark.parametrize("profile", ["skills", "_hardware.skills"])
def test_navigation_has_no_implicit_shared_diagnostic_file(tmp_path, monkeypatch, profile):
    module = import_module(f"bt_agent_robotics.{profile}.navigate")
    destination = tmp_path / "implicit" / "nav_debug.jsonl"
    monkeypatch.setattr(module, "_NAV_DEBUG_PATH", str(destination), raising=False)
    monkeypatch.setattr(module, "_NAV_DEBUG_WRITER", None, raising=False)
    module._nav_debug_log({"event": "fixture", "goal": [1, 2, 3]})
    assert not destination.exists()
    assert not destination.parent.exists()


@pytest.mark.parametrize("profile", ["skills", "_hardware.skills"])
def test_optional_navigation_diagnostics_preserve_entry_and_cannot_fail_skill(
    tmp_path, monkeypatch, profile
):
    module = import_module(f"bt_agent_robotics.{profile}.navigate")
    monkeypatch.setattr(module, "_NAV_DEBUG_PATH", str(tmp_path / "nav_debug.jsonl"), raising=False)
    entries = []
    monkeypatch.setattr(module, "_NAV_DEBUG_WRITER", entries.append, raising=False)
    entry = {"event": "fixture", "goal": [1, 2, 3]}
    module._nav_debug_log(entry)
    assert entries == [entry]
    assert entries[0] is entry

    def failing(_):
        raise OSError("Synthetic optional diagnostic failure")

    monkeypatch.setattr(module, "_NAV_DEBUG_WRITER", failing)
    module._nav_debug_log(entry)
