"""Default promotion must not remove the explicitly selected paper runtime."""

from bt_agent import AgentRuntime, TaskRuntime, create_agent
from bt_agent.episode import EpisodeHost
from bt_agent.profiles import DEFAULT_RUNTIME_PROFILE
from bt_agent.skills import ReportSkill
from bt_agent.testing import InspectShelf, shelf_model


def test_omitting_profile_runs_direct_task_state():
    assert DEFAULT_RUNTIME_PROFILE == "task-core-v1"
    with create_agent(
        model=shelf_model(),
        skills=[InspectShelf(), ReportSkill()],
        host=EpisodeHost(lambda _: None, lambda: None),
    ) as agent:
        assert isinstance(agent, TaskRuntime)
        assert agent.run("Inspect and report the shelf.", tick_interval=0.001).submitted
        assert agent.task.instruction == "Inspect and report the shelf."
        assert agent.describe()["profile"] == "task-core-v1"
        assert agent.describe()["state_model"] == "task/v1"
        assert not any(e.kind in {"plan_generated", "milestone_advanced"} for e in agent.events)


def test_explicit_paper_profile_keeps_legacy_lifecycle():
    with create_agent(
        model=shelf_model(),
        skills=[InspectShelf(), ReportSkill()],
        host=EpisodeHost(lambda _: None, lambda: None),
        profile="paper-core-v1",
    ) as agent:
        assert isinstance(agent, AgentRuntime)
        assert agent.run("Inspect and report the shelf.", tick_interval=0.001).submitted
        assert agent.describe()["profile"] == "paper-core-v1"
        assert "state_model" not in agent.describe()
        assert {"plan_generated", "milestone_advanced"} <= {e.kind for e in agent.events}
