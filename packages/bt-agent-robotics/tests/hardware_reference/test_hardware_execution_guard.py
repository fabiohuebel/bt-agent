import json

import py_trees
import pytest
from bt_agent._engine.agent.blackboard_keys import BB
from bt_agent._engine.bt.nodes.execute_skill import ExecuteSkill
from bt_agent._engine.bt.nodes.select_skill import SelectSkill
from bt_agent.skills import ObjectStore
from bt_agent.skills import SkillContext, SkillHandle, SkillStatus
from bt_agent._engine.skills.registry import SkillRegistry
from bt_agent_robotics._hardware.skills.press import PressSkill
from bt_agent_robotics._hardware.skills.query import SceneObject
from bt_agent_robotics._hardware.hardware_execution_guard import (
    DecisionMask,
    GuardedExecuteSkill,
    GuardedSelectSkill,
    install_execution_guard,
)


@pytest.fixture(autouse=True)
def isolated(monkeypatch):
    py_trees.blackboard.Blackboard.clear()
    monkeypatch.setenv("LAM_NAV_CONFIRM", "0")
    yield
    py_trees.blackboard.Blackboard.clear()


def make_case(names, offered=True):
    bb = py_trees.blackboard.Client(name="test_writer")
    keys = [
        BB.SELECTED_SKILL,
        BB.SKILL_PARAMS,
        BB.SKILL_REGISTRY,
        BB.SKILL_HANDLE,
        BB.PENDING_EVENTS,
        BB.SKILL_CALL_ID,
        BB.CONVERSATION,
        BB.MILESTONES,
        BB.CURRENT_MILESTONE_IDX,
        BB.COMPLETION_CLAIMED,
        BB.TASK_COMPLETE,
    ]
    for key in keys:
        bb.register_key(key=key, access=py_trees.common.Access.WRITE)
    history = []
    for i, name in enumerate(names):
        history.extend([
            {
                "role": "assistant",
                "tool_calls": [{"id": str(i), "function": {"name": name, "arguments": json.dumps({})}}],
            },
            {"role": "tool", "tool_call_id": str(i), "content": "success"},
        ])
    skill = PressSkill(None, None)
    registry = SkillRegistry()
    registry.register(skill)
    store = ObjectStore()
    store.add(SceneObject(handle="panel1", label="panel", object_id="1"))
    context = SkillContext(object_store=store)
    for key, value in [
        (BB.SELECTED_SKILL, "press"),
        (BB.SKILL_PARAMS, {"target_object": "panel1"}),
        (BB.SKILL_REGISTRY, registry),
        (BB.SKILL_HANDLE, None),
        (BB.PENDING_EVENTS, []),
        (BB.SKILL_CALL_ID, "decision1"),
        (BB.CONVERSATION, history),
        (BB.MILESTONES, [{}]),
        (BB.CURRENT_MILESTONE_IDX, 0),
        (BB.COMPLETION_CLAIMED, False),
        (BB.TASK_COMPLETE, False),
    ]:
        bb.set(key, value)
    mask = DecisionMask()
    mask.call_id = "decision1"
    mask.offered = frozenset(["press"] if offered else [])
    node = GuardedExecuteSkill(mask)
    node._build_context = lambda: context
    calls = []

    def prepare(params, context):
        calls.append("prepare")
        return params, None

    def start(*args, **kwargs):
        calls.append("start")
        h = SkillHandle()
        h.status = SkillStatus.SUCCEEDED
        return h

    skill.prepare, skill.start = prepare, start
    return node, bb, skill, calls


def test_actual_skip_survey_sequence_is_rejected_before_prepare():
    node, bb, _, calls = make_case(["navigate", "surround_survey", "query", "navigate"], offered=False)
    node.initialise()
    assert calls == []
    assert node._handle.status == SkillStatus.FAILED
    assert "SURROUND_SURVEY after NAVIGATE" in node._handle.observation
    assert node.update() == py_trees.common.Status.SUCCESS  # actor continues
    events = bb.get(BB.PENDING_EVENTS)
    assert any(e["kind"] == "skill_rejected" for e in events)
    assert not any(e["kind"] == "skill_started" for e in events)


def test_offered_press_is_rechecked_against_current_history():
    node, _, _, calls = make_case(["navigate"], offered=True)
    node.initialise()
    assert calls == [] and node._handle.status == SkillStatus.FAILED


def test_unoffered_press_rejected_even_if_conditions_now_pass():
    node, _, _, calls = make_case(["navigate", "surround_survey"], offered=False)
    node.initialise()
    assert calls == [] and "not offered" in node._handle.observation


def test_valid_offered_press_is_unchanged():
    node, _, _, calls = make_case(["navigate", "surround_survey"])
    node.initialise()
    assert calls == ["prepare", "start"]
    assert node._handle.status == SkillStatus.SUCCEEDED


def test_stale_decision_mask_does_not_authorize_new_call():
    node, _, _, calls = make_case(["navigate", "surround_survey"])
    node.mask.call_id = "old_decision"
    node.initialise()
    assert calls == [] and node._handle.status == SkillStatus.FAILED


def test_failed_precondition_and_exception_fail_closed():
    node, _, skill, calls = make_case(["navigate", "surround_survey"])
    skill.precondition = lambda context: (False, "camera lost")
    node.initialise()
    assert calls == [] and "camera lost" in node._handle.observation

    def broken(*args):
        raise RuntimeError("broken condition")

    skill.precondition = broken
    node.initialise()
    assert calls == [] and "check failed" in node._handle.observation


def test_completion_bypass_remains_intact():
    node, bb, _, calls = make_case([], offered=False)
    bb.set(BB.COMPLETION_CLAIMED, True)
    node.initialise()
    assert node.update() == py_trees.common.Status.SUCCESS
    assert calls == [] and node._handle is None


def test_selection_snapshots_actual_offered_tools(monkeypatch):
    _, bb, _, _ = make_case([], offered=False)
    mask = DecisionMask()
    node = GuardedSelectSkill(mask)
    node._epoch = 1
    node._model = None

    def select(self, *args, **kwargs):
        self._result = ("CONTINUE", "press", {}, "invalid model choice", "", {})

    monkeypatch.setattr(SelectSkill, "_select_in_background", select)
    node._select_in_background(1, None, [{"function": {"name": "navigate"}}])
    assert node.update() == py_trees.common.Status.SUCCESS
    assert mask.offered == frozenset({"navigate"})
    assert mask.call_id == bb.get(BB.SKILL_CALL_ID)


def test_installer_replaces_only_execution_boundary_nodes():
    root = py_trees.composites.Sequence("test", memory=True, children=[SelectSkill(), ExecuteSkill()])
    report = install_execution_guard(root)
    assert report["enabled"]
    assert isinstance(root.children[0], GuardedSelectSkill)
    assert isinstance(root.children[1], GuardedExecuteSkill)
    with pytest.raises(RuntimeError):
        install_execution_guard(root)


def test_guard_survives_actual_agent_task_reset():
    from bt_agent._engine.agent.lam_agent import LAMAgent
    from bt_agent._engine.agent.runner import AgentRunner
    from bt_agent.profiles import core_config as frozen_agent_config

    runner = AgentRunner(LAMAgent(None, config=frozen_agent_config()))
    install_execution_guard(runner.tree.root)
    runner.reset("test task")
    assert sum(isinstance(n, GuardedSelectSkill) for n in runner.tree.root.iterate()) == 1
    assert sum(isinstance(n, GuardedExecuteSkill) for n in runner.tree.root.iterate()) == 1
