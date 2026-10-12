import json
import time

import py_trees
import pytest

from bt_agent import create_agent
from bt_agent.episode import EpisodeHost
from bt_agent.skills import ActionSkill, ReportSkill
from bt_agent.testing import InspectShelf, ScriptedModel, shelf_model


def host():
    return EpisodeHost(lambda text: None, lambda: None)


def test_complete_episode_records_report_and_acknowledged_submission():
    reports, flushes = [], []
    model = shelf_model()
    with create_agent(
        model=model,
        skills=[InspectShelf(), ReportSkill()],
        host=EpisodeHost(reports.append, lambda: flushes.append(True)),
    ) as agent:
        outcome = agent.run("Inspect the shelf and report what is there.")
        assert outcome.termination == "submitted" and outcome.submitted
        assert not hasattr(outcome, "success")
        assert reports == ["The shelf contains a red mug."] and flushes == [True]
        assert [r["max_tokens"] for r in model.requests] == [1024] * 3
        assert all(r["phase"] == "actor" for r in model.requests)
        assert all(r["temperature"] == 0 for r in model.requests)
        assert all(
            "submit" not in {t["function"]["name"] for t in r["tools"]} for r in model.requests
        )
        assert "Report recorded." in json.dumps(model.requests[-1])
        assert "red mug on the shelf" in json.dumps(model.requests[1])
        assert sum(e.kind == "all_done_submit" for e in agent.events) == 1


def test_submission_rejection_does_not_manufacture_completion():
    class RejectingHost(EpisodeHost):
        def step(self, action):
            if action.startswith("submit "):
                return "Rejected.", 0, False, {"action_ok": False}
            return super().step(action)

    model = ScriptedModel([("claim_complete", {}, "There is nothing else to do.")])
    with create_agent(
        model=model, skills=[], host=RejectingHost(lambda _: None, lambda: None)
    ) as agent:
        result = agent.run("Report completion.", max_ticks=50, tick_interval=0.001)
        assert result.termination == "tick_limit" and not result.submitted
        assert any(e.kind == "all_done_submit_rejected" for e in agent.events)
        assert not any(e.kind == "all_done_submit" for e in agent.events)


def test_reset_starts_a_new_episode_without_old_reports_in_history():
    model = shelf_model()
    model.turns *= 2
    with create_agent(model=model, skills=[InspectShelf(), ReportSkill()], host=host()) as agent:
        assert agent.run("Inspect and report the shelf.").submitted
        assert agent.run("A new task: inspect and report the shelf.").submitted
        assert "Report recorded." not in json.dumps(model.requests[3])
        assert len(model.requests) == 6


def test_duplicate_and_reserved_skill_names_fail_before_runtime_creation():
    with pytest.raises(ValueError, match="unique"):
        create_agent(model=shelf_model(), skills=[ReportSkill(), ReportSkill()], host=host())

    class Submit(InspectShelf):
        name = "submit"

    with pytest.raises(ValueError, match="collides"):
        create_agent(model=shelf_model(), skills=[Submit()], host=host())


def test_process_global_runtime_is_explicitly_exclusive_and_releasable():
    with create_agent(model=shelf_model(), skills=[], host=host()):
        with pytest.raises(RuntimeError, match="One active"):
            create_agent(model=shelf_model(), skills=[], host=host())
    with create_agent(model=shelf_model(), skills=[], host=host()):
        pass


def test_profile_and_hidden_environment_conflicts_fail(monkeypatch):
    with pytest.raises(ValueError, match="Unsupported profile"):
        create_agent(model=shelf_model(), skills=[], host=host(), profile="typo")
    monkeypatch.setenv("LAM_REASONING_MODE", "act")
    with pytest.raises(ValueError, match="conflicts"):
        create_agent(model=shelf_model(), skills=[], host=host())


def test_loaded_manifest_identifies_effective_profile_without_credentials():
    model = shelf_model()
    model.api_key = "never-log-this"
    model.endpoint = "http://user:secret@offline.invalid/v1"
    with create_agent(model=model, skills=[ReportSkill()], host=host()) as agent:
        manifest = agent.describe()
        assert manifest["agent_config"]["ablation"] == ["evaluator", "planner"]
        assert manifest["controller_request_caps"] == {
            "actor_decision": 1024,
            "history_compaction": 512,
        }
        assert len(manifest["prompt_bundle_sha256"]) == 64
        assert {"runtime_support", "blackboard_lifecycle"} <= manifest[
            "loaded_source_sha256"
        ].keys()
        assert all(len(value) == 64 for value in manifest["loaded_source_sha256"].values())
        assert "never-log-this" not in json.dumps(manifest)
        assert "secret" not in json.dumps(manifest)


def test_no_hidden_bridge_override():
    with create_agent(model=shelf_model(), skills=[], host=host()) as agent:
        with pytest.raises(ValueError, match="episode host"):
            agent.start("task", platform_state={"_bridge": object()})


class LongAction(ActionSkill):
    name = "long_action"
    description = "Test action that stays running until cancelled."
    params = {}

    def precondition(self, state):
        return True, ""

    def safeguard_invariant(self, state, handle):
        return True

    def build_subtree(self, params, context, handle):
        self.handle = handle
        return py_trees.behaviours.Running("work")


def test_cancel_stops_active_skill_without_submission():
    skill = LongAction()
    model = ScriptedModel([("long_action", {}, "Start the action.")])
    with create_agent(model=model, skills=[skill], host=host()) as agent:
        agent.start("Run the long action.")
        deadline = time.monotonic() + 2
        while not hasattr(skill, "handle") and time.monotonic() < deadline:
            agent.tick()
            time.sleep(0.005)
        assert hasattr(skill, "handle")
        agent.cancel("test interruption")
        assert skill.handle.is_terminal
        assert agent.done
        assert not any(e.kind == "all_done_submit" for e in agent.events)


@pytest.mark.parametrize("profile", ["task-core-v1", "paper-core-v1"])
def test_keyboard_interrupt_in_run_cancels_the_active_skill(profile, monkeypatch):
    skill = LongAction()
    agent = create_agent(
        model=ScriptedModel([("long_action", {}, "Start.")]),
        skills=[skill],
        host=host(),
        profile=profile,
    )
    ordinary_tick = agent.tick

    def interrupted_tick():
        result = ordinary_tick()
        if hasattr(skill, "handle"):
            raise KeyboardInterrupt()
        return result

    monkeypatch.setattr(agent, "tick", interrupted_tick)
    try:
        with pytest.raises(KeyboardInterrupt):
            agent.run("Work", tick_interval=0.001)
        assert skill.handle.is_terminal and agent.done
        assert not any(event.kind == "all_done_submit" for event in agent.events)
    finally:
        agent.close()


def test_keyboard_interrupt_during_construction_releases_runtime_owner(monkeypatch):
    import bt_agent.runtime as runtime_module

    def interrupted(*args, **kwargs):
        raise KeyboardInterrupt()

    original = runtime_module.LAMAgent
    monkeypatch.setattr(runtime_module, "LAMAgent", interrupted)
    try:
        with pytest.raises(KeyboardInterrupt):
            create_agent(model=shelf_model(), skills=[], host=host(), profile="paper-core-v1")
        assert runtime_module._owner is None
    finally:
        runtime_module._release_runtime(runtime_module._owner)
    monkeypatch.setattr(runtime_module, "LAMAgent", original)
    with create_agent(model=shelf_model(), skills=[], host=host()):
        pass


@pytest.mark.parametrize("profile", ["task-core-v1", "paper-core-v1"])
def test_configuration_drift_cannot_block_cancellation(profile, monkeypatch):
    skill = LongAction()
    agent = create_agent(
        model=ScriptedModel([("long_action", {}, "Start.")]),
        skills=[skill],
        host=host(),
        profile=profile,
    )
    try:
        agent.start("Work")
        deadline = time.monotonic() + 2
        while not hasattr(skill, "handle") and time.monotonic() < deadline:
            agent.tick()
            time.sleep(0.002)
        assert hasattr(skill, "handle")
        monkeypatch.setenv("LAM_REASONING_MODE", "invalid-live-change")
        agent.cancel("Stop despite invalid new configuration")
        assert agent.done and skill.handle.is_terminal
    finally:
        agent.close()
