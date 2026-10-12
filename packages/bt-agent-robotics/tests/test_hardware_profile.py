import json
from types import SimpleNamespace
import time

import pytest

from bt_agent.episode import EpisodeHost
from bt_agent.testing import ScriptedModel
from bt_agent_robotics.hardware import HardwareAgent, HardwareProfile, InterventionController


class Navigation:
    def __init__(self):
        self.calls = []

    def get_pose_age(self):
        return 0.01

    def get_robot_pose(self):
        return (0.0, 0.0, 0.0)

    def get_robot_velocity(self):
        return (0.1, 0.0)

    def set_brake(self, enabled):
        self.calls.append(("brake", enabled))

    def send_waypoint(self, *point):
        self.calls.append(("waypoint", point))

    def send_velocity(self, *velocity):
        self.calls.append(("velocity", velocity))

    def cancel(self):
        self.calls.append(("cancel",))


class Planner:
    waypoint_threshold = 0.2

    def plan(self, *_):
        return [(3.0, 0.0)]

    replan = plan

    def update_map(self, _):
        pass


class Gate:
    enabled = True

    def __init__(self):
        self.plans = []
        self.decision = None

    def publish_plan(self, *args):
        self.plans.append(args)
        self.decision = None

    def poll(self):
        value, self.decision = self.decision, None
        return value

    def clear(self):
        pass


class Memory:
    def query(self, *_args, **_kwargs):
        from bt_agent_robotics._hardware.skills.query import SceneGraphResult

        return SceneGraphResult(
            found=[
                dict(
                    object_id="1",
                    object_label="shelf",
                    viewpoint_position=dict(x=3.0, y=0.0, z=0.0),
                    all_viewpoints=[[3.0, 0.0, 0.0, 0.0]],
                    viewpoint_orientation=[0.0, 0.0, 0.0, 1.0],
                )
            ]
        )


@pytest.fixture(autouse=True)
def environment(monkeypatch):
    from bt_agent._engine import skills

    monkeypatch.setattr(skills, "_extra_doc_roots", [])
    for key, value in dict(
        LAM_NAV_CONFIRM="1",
        LAM_CAPTURE_LOCAL_INSPECTION="1",
        LAM_FARM_EXPLORATION_FALLBACK="0",
        LAM_QUERY_TOPK="0",
    ).items():
        monkeypatch.setenv(key, value)
    monkeypatch.delenv("LAM_SIM_EXPERIMENT_ARM", raising=False)


def setup(turns, **options):
    model = ScriptedModel(turns)
    model.temperature, model._enable_thinking = 0, False
    model.endpoint = "http://private:secret@invalid"
    nav, gate, reports, flushes = Navigation(), Gate(), [], []
    host = options.pop("host", EpisodeHost(reports.append, lambda: flushes.append(True)))
    profile = options.pop("profile", HardwareProfile("visual_question_v1", True, False))
    controls = InterventionController(nav)
    agent = HardwareAgent(
        model=model,
        host=host,
        navigation=nav,
        path_planning=Planner(),
        scene_memory=Memory(),
        perception=None,
        confirmation_gate=gate,
        profile=profile,
        interventions=controls,
        **options,
    )
    return agent, model, nav, gate, reports, flushes, controls


def until(agent, predicate, seconds=3):
    end = time.monotonic() + seconds
    while not predicate() and time.monotonic() < end:
        if not agent.done:
            agent.tick()
        time.sleep(0.002)
    assert predicate()


def test_report_submit_and_reset_preserve_frozen_policy():
    turns = [
        ("report", {"text": "Observed the shelf."}, "Report evidence."),
        ("claim_complete", {}, "Done."),
    ]
    agent, model, nav, gate, reports, flushes, controls = setup(turns * 2)
    with agent:
        for task in ("First report", "Second report"):
            agent.start(task)
            until(agent, lambda: agent.done)
        assert reports == ["Observed the shelf."] * 2 and len(flushes) == 2
        assert "Report recorded." not in json.dumps(model.requests[2])
        assert all(r["max_tokens"] == 1024 and r["temperature"] == 0 for r in model.requests)
        assert all(
            "submit" not in [t["function"]["name"] for t in r["tools"]] for r in model.requests
        )
        assert not nav.calls
        manifest = agent.describe()
        from bt_agent import AgentRuntime

        assert isinstance(agent.runtime, AgentRuntime)
        assert manifest["policy"]["profile"] == "paper-core-v1"
        assert "state_model" not in manifest["policy"]
        assert manifest["execution_guard"]["enabled"] and manifest["profile"]["local_inspection"]
        assert set(manifest["adapter_source_sha256"]) == {
            "hardware.py",
            "hardware_ros.py",
            "hardware_inputs.py",
        }
        assert all(len(value) == 64 for value in manifest["adapter_source_sha256"].values())
        assert "secret" not in json.dumps(manifest)
        agent.profile = HardwareProfile("original", True, False)
        with pytest.raises(ValueError, match="Declared hardware profile"):
            agent.describe()


@pytest.mark.parametrize(
    "kind,reason", [("interrupt", "operator_interrupt"), ("timeout", "wall_time_exceeded")]
)
def test_model_continues_to_report_after_motion_failure(kind, reason):
    turns = [
        ("query", {"target_description": "shelf"}, "Find shelf."),
        ("navigate", {"target_object": "shelf_1"}, "Move to shelf."),
        ("report", {"text": "Navigation did not complete."}, "Report failure."),
        ("claim_complete", {}, "Done."),
    ]
    agent, model, nav, gate, reports, _, controls = setup(turns)
    with agent:
        agent.start("Navigate to shelf and report.", platform_state={"robot_position": (0, 0, 0)})
        until(agent, lambda: bool(gate.plans))
        assert not any(row == ("brake", False) for row in nav.calls)
        gate.decision = "go"
        until(agent, lambda: ("brake", False) in nav.calls)
        key = controls.snapshot()["active"]["key"]
        assert controls.request(key, kind, "test-1")
        until(agent, lambda: agent.done)
        assert reports == ["Navigation did not complete."]
        history = json.dumps(model.requests[2]["messages"])
        assert ("operator_interrupt" if kind == "interrupt" else "timed out") in history
        assert [row for row in nav.calls if row[0] == "brake"][-1] == ("brake", True)
        assert ("velocity", (0.0, 0.0, 0.0)) in nav.calls
        assert controls.snapshot()["active"] is None
        if kind == "timeout":
            assert "operator injected" not in history.lower()


def test_offered_tool_guard_rejects_press_without_survey():
    agent, _, nav, _, reports, _, _ = setup(
        [
            ("press", {"target_object": "missing"}, "Invalid action."),
            ("report", {"text": "Need to inspect first."}, "Recover."),
            ("claim_complete", {}, "Done."),
        ]
    )
    with agent:
        agent.start("Inspect first")
        until(agent, lambda: agent.done)
        assert not nav.calls and reports == ["Need to inspect first."]
        assert any(event.kind == "skill_rejected" for event in agent.runtime.events)


def test_submission_refusal_never_becomes_success():
    class Reject(EpisodeHost):
        def step(self, action):
            return "Rejected.", 0, False, {"action_ok": False}

    agent, *_ = setup([("claim_complete", {}, "Done.")], host=Reject(lambda _: None, lambda: None))
    with agent:
        agent.start("test")
        until(
            agent, lambda: any(e.kind == "all_done_submit_rejected" for e in agent.runtime.events)
        )
        assert not agent.done
        agent.cancel("test end")


def test_environment_conflicts_and_changes_are_not_silent(monkeypatch):
    agent, *_ = setup([])
    with agent:
        monkeypatch.setenv("LAM_NAV_CONFIRM", "0")
        with pytest.raises(ValueError, match="LAM_NAV_CONFIRM"):
            agent.start("test")
    with pytest.raises(ValueError, match="LAM_NAV_CONFIRM"):
        setup([])


def test_capture_contract_does_not_turn_empty_instruction_into_a_task():
    agent, *_ = setup([], profile=HardwareProfile("original", True, False, "capture"))
    with agent:
        with pytest.raises(ValueError, match="non-empty"):
            agent.start(" ")


def test_early_safeguard_failure_does_not_leave_intervention_scope_active():
    agent, _, nav, _, _, _, controls = setup(
        [
            ("query", {"target_description": "shelf"}, "Find shelf."),
            ("navigate", {"target_object": "shelf_1"}, "Move."),
            ("report", {"text": "Pose is stale."}, "Report."),
            ("claim_complete", {}, "Done."),
        ]
    )
    nav.get_pose_age = lambda: 2.0
    with agent:
        agent.start("Navigate and report", platform_state={"robot_position": (0, 0, 0)})
        until(agent, lambda: agent.done)
        assert controls.snapshot()["active"] is None
        assert ("brake", False) not in nav.calls
