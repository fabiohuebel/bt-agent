"""Migration checks: authoritative task state, lifecycle and policy parity."""

from copy import deepcopy
from dataclasses import FrozenInstanceError
import json
import threading
import time

import py_trees
import pytest

from bt_agent import TaskContext, TaskRuntime, TaskState, TaskStatus, create_agent
from bt_agent.episode import EpisodeHost
from bt_agent.skills import ActionSkill, ContextSkill, ReportSkill, SkillStatus, register_doc_root
from bt_agent.testing import InspectShelf, ScriptedModel, shelf_model


def host():
    return EpisodeHost(lambda _: None, lambda: None)


def task_agent(model=None, skills=(), episode_host=None):
    return create_agent(
        model=model or shelf_model(),
        skills=skills,
        host=episode_host or host(),
        profile="task-core-v1",
    )


def tick_until(runtime, predicate):
    deadline = time.monotonic() + 3
    while not predicate() and time.monotonic() < deadline:
        runtime.tick()
        time.sleep(0.001)
    assert predicate(), "Offline fixture did not reach its expected state"


def normalized_events(events):
    return [
        {"kind": event.kind, "data": {k: v for k, v in event.data.items() if k != "skill_call_id"}}
        for event in events
        if event.kind
        in {
            "skill_selected",
            "skill_started",
            "skill_completed",
            "completion_claimed",
            "all_done_submit",
            "all_done_submit_rejected",
        }
    ]


def test_task_is_authoritative_without_blackboard_or_plan_bookkeeping():
    before = dict(py_trees.blackboard.Blackboard.storage)
    with task_agent(skills=[InspectShelf(), ReportSkill()]) as agent:
        assert isinstance(agent, TaskRuntime)
        assert agent.task is None and not agent.done
        agent.start("Inspect and report the shelf.", scenario="fixture")
        first = agent.task
        assert first == TaskState("Inspect and report the shelf.", "fixture")
        with pytest.raises(FrozenInstanceError):
            first.instruction = "mutated"
        names = [node.name for node in agent._tree.root.iterate()]
        assert names == ["SelectTaskSkill", "ExecuteTaskSkill", "SubmitTask", "Task"]
        tick_until(agent, lambda: agent.done)
        assert agent.task.status is TaskStatus.SUBMITTED
        assert first.status is TaskStatus.ACTIVE  # snapshots do not change retrospectively
        assert "milestone_advanced" not in [e.kind for e in agent.events]
        assert "plan_generated" not in [e.kind for e in agent.events]
        assert len(agent._history) == 4  # retained task history, not cleared by advancement
        assert agent.objects_snapshot()[0].handle == "red_mug"
    assert py_trees.blackboard.Blackboard.storage == before


def test_legacy_context_is_a_detached_projection_not_mutable_task_storage():
    task = TaskState("Look around.", "room")
    context = TaskContext(task)
    assert context.main_task == task.instruction
    assert context.scenario == task.scenario and context.milestone_index == 0
    context.milestone["description"] = "changed"
    assert context.milestone["description"] == task.instruction
    assert "milestone" not in context.__dict__


class FeedbackSkill(InspectShelf):
    """Return typed feedback and images so both actor ingress paths are exercised."""

    name = "feedback"

    def build_subtree(self, params, context, handle):
        class Feedback(py_trees.behaviour.Behaviour):
            def update(self):
                handle.observation = "Stopped before the target."
                handle.result_data = {
                    "arrival_status": "STOPPED_SHORT",
                    "failure_reason": "wall_time_budget",
                    "memoized": True,
                    "image": b"fixture-image",
                    "nested": {"other": b"fixture-image-2"},
                    "views": [b"fixture-image-3"],
                }
                return py_trees.common.Status.FAILURE

        return Feedback("feedback")


class GatedAction(ActionSkill):
    name = "gated_action"
    description = "Action available only after inspection and with host permission."
    params = {}

    def is_available(self, history, context):
        return any(call.name == "inspect_shelf" for call in history.get("tool_calls", [])), ""

    def precondition(self, state):
        return bool(state.platform_state.get("permit")), "permission required"

    def safeguard_invariant(self, state, handle):
        return True

    def build_subtree(self, params, context, handle):
        return py_trees.behaviours.Success("act")


@pytest.mark.parametrize(
    "case", ["shelf", "empty-report", "feedback", "unknown", "gating", "docs", "model-error"]
)
def test_paired_actor_requests_host_actions_and_skill_events(case, tmp_path):
    if case == "docs":
        (tmp_path / "fixture.md").write_text("Offline skill documentation.")
        register_doc_root(tmp_path)
    all_results = []
    for profile in ["paper-core-v1", "task-core-v1"]:
        reports, attempts = [], []

        class Host(EpisodeHost):
            def step(self, action):
                attempts.append(action)
                return super().step(action)

        model = shelf_model()
        skills = [InspectShelf(), ReportSkill(), FeedbackSkill(), GatedAction()]
        if case == "empty-report":
            model.turns.insert(0, ("report", {"text": ""}, "Try an empty report."))
        elif case == "feedback":
            model.turns.insert(0, ("feedback", {}, "Attempt inspection."))
        elif case == "unknown":
            model.turns.insert(0, ("unknown", {}, "Try an unknown skill."))
        elif case == "gating":
            model.turns.insert(1, ("gated_action", {}, "Inspection is complete."))
        elif case == "docs":
            model.turns.insert(
                0, ("read_skill_docs", {"skill_name": "fixture"}, "Read documentation.")
            )
        elif case == "model-error":
            ordinary = model.query_tool_call

            def flaky(*args, **kwargs):
                if not hasattr(model, "raised"):
                    model.raised = True
                    raise RuntimeError("offline transport error")
                return ordinary(*args, **kwargs)

            model.query_tool_call = flaky
        with create_agent(
            model=model, skills=skills, host=Host(reports.append, lambda: None), profile=profile
        ) as agent:
            outcome = agent.run(
                "Inspect and report.",
                scenario="fixture",
                platform_state={"permit": True},
                tick_interval=0.001,
            )
            assert outcome.submitted and not model.turns
            all_results.append((model.requests, reports, attempts, normalized_events(agent.events)))
        if case == "gating":
            first_tools = {tool["function"]["name"] for tool in model.requests[0]["tools"]}
            later_tools = {tool["function"]["name"] for tool in model.requests[1]["tools"]}
            assert "gated_action" not in first_tools and "gated_action" in later_tools
    assert all_results[0] == all_results[1]


@pytest.mark.parametrize(
    "done,info",
    [
        (False, {"action_ok": True}),
        (True, {"action_ok": False}),
        (True, {"action_ok": 1}),
        (True, {}),
        (True, None),
    ],
)
def test_host_must_acknowledge_before_submission_and_retries_do_not_rerun_model(done, info):
    class RejectOnce(EpisodeHost):
        attempts = 0

        def step(self, action):
            self.attempts += 1
            if self.attempts == 1:
                return "Not acknowledged.", 0, done, info
            return super().step(action)

    model = ScriptedModel([("claim_complete", {}, "Done.")])
    episode_host = RejectOnce(lambda _: None, lambda: None)
    with task_agent(model=model, episode_host=episode_host) as agent:
        agent.start("Finish.")
        tick_until(agent, lambda: episode_host.attempts == 1)
        assert not agent.done and agent.task.status is TaskStatus.SUBMITTING
        agent.tick()
        assert agent.done and agent.task.status is TaskStatus.SUBMITTED
        assert episode_host.attempts == 2 and len(model.requests) == 1
        assert not any(e.kind == "skill_completed" for e in agent.events)


def test_reset_clears_reports_objects_history_and_submission():
    model = shelf_model()
    model.turns *= 2
    with task_agent(model=model, skills=[InspectShelf(), ReportSkill()]) as agent:
        assert agent.run("First task.", tick_interval=0.001).submitted
        agent.start("Second task.")
        assert not agent.done and not agent.objects_snapshot() and not agent.events
        tick_until(agent, lambda: agent.done)
        assert len(model.requests) == 6
        assert "Report recorded." not in json.dumps(model.requests[3])
        assert "red mug on the shelf" not in json.dumps(model.requests[3])


class LongAction(ActionSkill):
    name = "long_action"
    description = "Cancellable test BT."
    params = {}

    def precondition(self, state):
        return True, ""

    def safeguard_invariant(self, state, handle):
        return state.platform_state.get("permit", True)

    def build_subtree(self, params, context, handle):
        self.handle = handle
        self.context = context
        self.running = threading.Event()
        running = self.running

        class Work(py_trees.behaviour.Behaviour):
            def update(self):
                running.set()
                return py_trees.common.Status.RUNNING

        return Work("work")


def test_cancel_quiesces_the_skill_before_reset():
    skill = LongAction()
    model = ScriptedModel([("long_action", {}, "Start."), ("claim_complete", {}, "Done.")])
    with task_agent(model=model, skills=[skill]) as agent:
        agent.start("Old task.")
        tick_until(agent, lambda: hasattr(skill, "handle"))
        assert isinstance(skill.context, TaskContext)
        agent.cancel("operator")
        assert skill.handle.is_terminal and agent.done
        assert agent.task.status is TaskStatus.CANCELLED
        assert not agent._host.submitted
        assert agent.run("New task.", tick_interval=0.001).submitted


def test_skill_safeguard_reads_fresh_host_state_and_failure_returns_to_actor():
    skill = LongAction()
    model = ScriptedModel([("long_action", {}, "Start."), ("claim_complete", {}, "Blocked.")])
    with task_agent(model=model, skills=[skill]) as agent:
        agent.start("Try work.", platform_state={"permit": True})
        tick_until(agent, lambda: hasattr(skill, "running") and skill.running.is_set())
        agent.push_state({"permit": False})
        tick_until(agent, lambda: agent.done)
        assert skill.handle.status is SkillStatus.FAILED
        assert "InvariantGuard" in skill.handle.observation
        assert any(
            e.kind == "skill_completed" and e.data["outcome"] == "failed" for e in agent.events
        )
        assert skill.handle.observation in [
            message.get("content") for message in model.requests[-1]["messages"]
        ]


def test_old_paper_cancellation_cannot_poison_a_new_task_runtime():
    from bt_agent._engine.agent.blackboard_keys import BB

    with create_agent(
        model=shelf_model(), skills=[], host=host(), profile="paper-core-v1"
    ) as paper:
        paper.start("Cancel the old paper task.")
        paper.cancel("done")
    assert py_trees.blackboard.Blackboard.get(BB.CANCEL_REQUESTED) is True
    with task_agent(skills=[InspectShelf(), ReportSkill()]) as agent:
        assert agent.run("Inspect and report.", tick_interval=0.001).submitted
        assert all(
            e.data["outcome"] == "success" for e in agent.events if e.kind == "skill_completed"
        )
    assert py_trees.blackboard.Blackboard.get(BB.CANCEL_REQUESTED) is True


def test_cancelled_inference_cannot_execute_or_overwrite_a_new_task():
    entered, release = threading.Event(), threading.Event()

    class SlowModel(ScriptedModel):
        def query_tool_call(self, *args, **kwargs):
            if not entered.is_set():
                entered.set()
                assert release.wait(3)
            return super().query_tool_call(*args, **kwargs)

    model = SlowModel(
        [("report", {"text": "STALE"}, "Old response."), ("claim_complete", {}, "New task.")]
    )
    reports = []
    try:
        with task_agent(
            model=model,
            skills=[ReportSkill()],
            episode_host=EpisodeHost(reports.append, lambda: None),
        ) as agent:
            agent.start("Old task.")
            agent.tick()
            assert entered.wait(1)
            agent.cancel("reset")
            agent.start("New task.")
            agent.tick()
            release.set()
            tick_until(agent, lambda: agent.done)
            assert reports == []
            assert "STALE" not in json.dumps(model.requests[-1])
            assert agent.task.instruction == "New task."
            assert agent.task.completion_reason == "New task."
    finally:
        release.set()


@pytest.mark.parametrize(
    "owner,other",
    [
        ("paper-core-v1", "task-core-v1"),
        ("task-core-v1", "paper-core-v1"),
        ("task-core-v1", "task-core-v1"),
    ],
)
def test_process_ownership_shared_between_profiles(owner, other):
    with create_agent(model=shelf_model(), skills=[], host=host(), profile=owner):
        with pytest.raises(RuntimeError, match="One active"):
            create_agent(model=shelf_model(), skills=[], host=host(), profile=other)
    with create_agent(model=shelf_model(), skills=[], host=host(), profile=other):
        pass


def test_manifest_explicitly_marks_compatibility_and_no_planner():
    model = shelf_model()
    model.endpoint = "http://secret.invalid"
    with task_agent(model=model) as agent:
        manifest = agent.describe()
        assert manifest["profile"] == "task-core-v1" and manifest["state_model"] == "task/v1"
        assert manifest["planner_supported"] is False
        assert manifest["controller_request_caps"] == {
            "actor_decision": 1024,
            "history_compaction": 512,
        }
        assert "secret.invalid" not in json.dumps(manifest)
        with pytest.raises(TypeError, match="plan"):
            agent.start("task", plan=["not silently accepted"])


@pytest.mark.parametrize("profile", ["paper-core-v1", "task-core-v1"])
def test_internal_wait_is_resolved_without_exposing_it_as_a_tool(profile):
    class Wait(InspectShelf):
        name = "wait"

    model = ScriptedModel([("wait", {}, "Wait."), ("claim_complete", {}, "Done.")])
    with create_agent(
        model=model, skills=[], internal_skills=[Wait()], host=host(), profile=profile
    ) as agent:
        assert agent.run("Wait then finish.", tick_interval=0.001).submitted
        assert all(
            "wait" not in {t["function"]["name"] for t in req["tools"]} for req in model.requests
        )
        assert agent.describe()["internal_handlers"] == ["wait"]


def test_compaction_payloads_and_budgets_are_identical():
    class Verbose(InspectShelf):
        def build_subtree(self, params, context, handle):
            handle.observation = "Detailed observation. " * 6000
            return py_trees.behaviours.Success("verbose")

    class SummarizingModel(ScriptedModel):
        def query(self, messages, **kwargs):
            self.requests.append(deepcopy({"messages": messages, **kwargs}))
            return "Previous observations recorded."

    requests = []
    for profile in ["paper-core-v1", "task-core-v1"]:
        model = SummarizingModel(
            [("inspect_shelf", {}, "Inspect again.")] * 7 + [("claim_complete", {}, "Done.")]
        )
        with create_agent(model=model, skills=[Verbose()], host=host(), profile=profile) as agent:
            assert agent.run("Repeated inspection.", tick_interval=0.001).submitted
        requests.append(model.requests)
        assert any(request["max_tokens"] == 512 for request in model.requests)
        assert all(
            request["max_tokens"] == 1024 for request in model.requests if "tools" in request
        )
    assert requests[0] == requests[1]


def test_unacknowledged_skill_cancellation_retains_ownership():
    class Stubborn(LongAction):
        def start(self, params, context, state_provider=None):
            from bt_agent.skills import SkillHandle

            self.handle = SkillHandle()
            self.handle.status = SkillStatus.RUNNING
            return self.handle

    skill = Stubborn()
    agent = task_agent(model=ScriptedModel([("long_action", {}, "Start.")]), skills=[skill])
    try:
        agent.start("Start stubborn fixture.")
        tick_until(agent, lambda: hasattr(skill, "handle"))
        with pytest.raises(RuntimeError, match="ownership is retained"):
            agent._stop_tree(settle_timeout=0)
        with pytest.raises(RuntimeError, match="One active"):
            task_agent()
        with pytest.raises(RuntimeError, match="Cancellation is pending"):
            agent.tick()
        assert not agent.done
    finally:
        skill.handle.status = SkillStatus.CANCELLED
        agent.close()
    with task_agent():
        pass


def test_task_runtime_validation_and_lifecycle_boundaries(monkeypatch):
    with pytest.raises(ValueError, match="unique"):
        task_agent(skills=[ReportSkill(), ReportSkill()])
    with pytest.raises(TypeError, match="context skills"):
        create_agent(
            model=shelf_model(),
            skills=[],
            host=host(),
            profile="task-core-v1",
            internal_skills=[LongAction()],
        )
    with task_agent() as agent:
        with pytest.raises(RuntimeError, match="No active"):
            agent.tick()
        with pytest.raises(ValueError, match="non-empty"):
            agent.start("")
        with pytest.raises(ValueError, match="episode host"):
            agent.start("task", platform_state={"_bridge": host()})
        with pytest.raises(ValueError, match="episode host"):
            agent.push_state({"_bridge": host()})
        for max_ticks in [0, True, 1.5]:
            with pytest.raises(ValueError, match="max_ticks"):
                agent.run("task", max_ticks=max_ticks)
        agent.start("task")
        with pytest.raises(RuntimeError, match="Finish or cancel"):
            agent.start("another task")
        monkeypatch.setenv("LAM_REASONING_MODE", "act")
        with pytest.raises(ValueError, match="conflicts"):
            agent.tick()
        monkeypatch.delenv("LAM_REASONING_MODE")
    with pytest.raises(RuntimeError, match="closed"):
        agent.start("cannot restart")


class ProseThenScripted(ScriptedModel):
    """Offline fixture: ``None`` turns are text-only replies without a tool call."""

    def query_tool_call(self, messages, tools, **kwargs):
        if self.turns and self.turns[0][0] is None:
            self.requests.append(deepcopy({"messages": messages, "tools": tools, **kwargs}))
            _, _, text = self.turns.pop(0)
            return None, {}, {"choices": [{"message": {"role": "assistant", "content": text}}]}
        return super().query_tool_call(messages, tools, **kwargs)


@pytest.mark.parametrize("profile", ["paper-core-v1", "task-core-v1"])
def test_two_text_only_replies_do_not_poison_later_selections(profile):
    # A prose reply plus a prose re-prompt answer leaves no tool name. Both
    # profiles must ask again and finish; the task profile used to record a
    # nameless tool call that crashed every later request until the tick limit.
    model = ProseThenScripted(
        [(None, {}, "Let me think about the shelf."), (None, {}, "Still thinking.")]
        + shelf_model().turns
    )
    with create_agent(
        model=model, skills=[InspectShelf(), ReportSkill()], host=host(), profile=profile
    ) as agent:
        outcome = agent.run("Inspect and report the shelf.", tick_interval=0.001, max_ticks=3000)
        assert outcome.submitted, outcome
        assert not model.turns
    for request in model.requests:
        for message in request["messages"]:
            for call in message.get("tool_calls") or []:
                assert call["function"]["name"], "nameless tool call entered history"


def test_unoffered_defer_call_is_rejected_like_any_unknown_tool():
    # The milestone-era ``defer`` meta-tool is never offered by this profile.
    # A model that calls it anyway gets an unknown-skill observation and the
    # episode continues; it must not abort as an infrastructure error.
    model = ScriptedModel([("defer", {"reason": "later"}, "I will defer.")] + shelf_model().turns)
    with task_agent(model=model, skills=[InspectShelf(), ReportSkill()]) as agent:
        outcome = agent.run("Inspect and report the shelf.", tick_interval=0.001, max_ticks=3000)
        assert outcome.submitted, outcome
    assert all("defer" not in {t["function"]["name"] for t in r["tools"]} for r in model.requests)
    assert "Unknown skill: defer" in json.dumps(model.requests[1]["messages"])
