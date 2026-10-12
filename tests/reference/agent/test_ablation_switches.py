"""CP2/CP3: all four ablation switches go live -- "evaluator", "replan"
(tree/routing, CP2) and "context", "planner" (actor history / planner call,
CP3).

docs/spec/tiamat-ablation-flags.md CP2: with "evaluator" ablated, the
``evaluate`` branch is replaced by an unconditional auto-accept (no evaluator
model call, ever). With "replan" ablated, a REPLAN verdict (raw from the
evaluator, or cap-promoted inside ``EvaluateCompletion.update()``) ends the
episode as a give-up submit instead of wiping ``BB.MILESTONES``. Both must
remain terminating -- no hang -- and the default (empty ablation) tree must
stay byte-identical to today's agent.

docs/spec/tiamat-ablation-flags.md CP3: with "context" ablated, SelectSkill
threads an unbounded token budget into ``actor.select_skill()`` -- full
verbatim history, the compaction functions never run -- and a backend
context-overflow error (HTTP 400, context-length message) ends the episode
with a real give-up submit instead of crashing the child. With "planner"
ablated, PlanTask never imports/calls ``generate_plan``; every plan (initial
or re-entry) is the single static task-text milestone.

Unit-level, no HTTP, no ROS: the evaluator-ablated tests never build
EvaluateCompletion at all (so there is no model to call); the replan-ablated
test forces the verdict via a monkeypatched ``evaluate_completion``, the same
pattern ``test_three_way_verdict.py`` uses. The CP3 tests drive
``PlanTask._plan_in_background`` / ``SelectSkill._select_in_background``
directly (both are plain methods -- no real threading needed to exercise
them), plus three tree-level compositions ticked through a real
``AgentRunner`` with a mock model backend and no network.
"""

from __future__ import annotations

import json
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock

import bt_agent._engine.agent.lam_agent  # noqa: F401 (init-order side effect, see test_structural_cp2.py)
import py_trees
import pytest
import requests
from bt_agent._engine.agent.blackboard_keys import BB
from bt_agent._engine.agent.config import AgentConfig
from bt_agent._engine.agent.lam_agent import LAMAgent
from bt_agent._engine.agent.lam_agent.evaluator import EvaluatorResult, Verdict
from bt_agent._engine.agent.runner import AgentRunner
from bt_agent._engine.bt.nodes.plan_task import PlanTask
from bt_agent._engine.bt.nodes.route_verdict import GiveUpOnReplan
from bt_agent._engine.bt.nodes.select_skill import (
    _UNBOUNDED_TOKEN_BUDGET,
    SelectSkill,
    _is_context_overflow_error,
)
from bt_agent._engine.bt.tree_builder import build_agent_tree

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


class _FakeSubmitBridge:
    """Minimal bridge stub: confirms every submit as an immediate terminal step."""

    def __init__(self, *, terminal: bool = True, action_ok: bool = True) -> None:
        self.terminal = terminal
        self.action_ok = action_ok
        self.actions: List[str] = []

    def step(self, action: str):
        self.actions.append(action)
        info = {"action_ok": self.action_ok, "outcome": "give_up" if self.action_ok else None}
        return "submit response", 0.0, self.terminal, info


def _find(node: py_trees.behaviour.Behaviour, name: str):
    if node.name == name:
        return node
    for child in getattr(node, "children", []):
        hit = _find(child, name)
        if hit is not None:
            return hit
    return None


def _tick_once(node: py_trees.behaviour.Behaviour) -> py_trees.common.Status:
    for _ in node.tick():
        pass
    return node.status


def _writer(*keys: str) -> py_trees.blackboard.Client:
    bb = py_trees.blackboard.Client(name=f"ablation_{id(object())}")
    for key in keys:
        bb.register_key(key=key, access=py_trees.common.Access.WRITE)
    return bb


# ---------------------------------------------------------------------------
# "evaluator" ablation: tree shape
# ---------------------------------------------------------------------------


def test_evaluator_ablated_tree_replaces_evaluate_with_auto_accept() -> None:
    root = build_agent_tree(ablation=frozenset({"evaluator"}))

    skill_or_eval = _find(root, "skill_or_eval")
    assert [c.name for c in skill_or_eval.children] == ["auto_accept", "execute"]

    auto_accept = _find(root, "auto_accept")
    assert [type(c).__name__ for c in auto_accept.children] == ["CheckCompletionClaimed", "AdvanceMilestone"]

    # No verdict machinery at all -- nothing left to route a REPLAN through,
    # and the amend-cap promotion inside EvaluateCompletion.update() (the only
    # place AMEND -> REPLAN cap promotion happens) is unreachable by
    # construction since that node is never built.
    assert _find(root, "verdict_route") is None
    assert _find(root, "EvaluateCompletion") is None


def test_default_ablation_keeps_evaluate_branch() -> None:
    root = build_agent_tree()  # ablation=frozenset() default

    skill_or_eval = _find(root, "skill_or_eval")
    assert [c.name for c in skill_or_eval.children] == ["evaluate", "execute"]
    assert _find(root, "auto_accept") is None
    assert _find(root, "EvaluateCompletion") is not None


# ---------------------------------------------------------------------------
# "replan" ablation: tree shape
# ---------------------------------------------------------------------------


def test_replan_ablated_tree_swaps_apply_replan_for_give_up() -> None:
    root = build_agent_tree(ablation=frozenset({"replan"}))

    # Evaluator still runs -- only the replan action changes.
    assert _find(root, "EvaluateCompletion") is not None
    replan_route = _find(root, "replan_route")
    assert [type(c).__name__ for c in replan_route.children] == ["CheckVerdictReplan", "GiveUpOnReplan"]

    pass_route = _find(root, "pass_route")
    amend_route = _find(root, "amend_route")
    assert [type(c).__name__ for c in pass_route.children] == ["CheckVerdictPass", "AdvanceMilestone"]
    assert [type(c).__name__ for c in amend_route.children] == ["CheckVerdictAmend", "ApplyAmend"]


def test_planner_ablated_alone_also_swaps_apply_replan_for_give_up() -> None:
    """CP3 review round: "planner" ablation, with "replan" NOT also ablated,
    still routes REPLAN to GiveUpOnReplan -- replanning a static plan is
    meaningless and would otherwise cycle to the tick cap."""
    root = build_agent_tree(ablation=frozenset({"planner"}))

    assert _find(root, "EvaluateCompletion") is not None
    replan_route = _find(root, "replan_route")
    assert [type(c).__name__ for c in replan_route.children] == ["CheckVerdictReplan", "GiveUpOnReplan"]


# ---------------------------------------------------------------------------
# "evaluator" ablation: claim advances, episode terminates
# ---------------------------------------------------------------------------


def _seed_auto_accept_bb(bridge: _FakeSubmitBridge, milestones: list) -> py_trees.blackboard.Client:
    bb = _writer(
        BB.CANCEL_REQUESTED,
        BB.MILESTONES,
        BB.CURRENT_MILESTONE_IDX,
        BB.COMPLETED_MILESTONES,
        BB.COMPLETION_CLAIMED,
        BB.REACT_FALLBACK_ACTIVE,
        BB.TASK,
        BB.CORRECTIVE_REPLAN_DONE,
        BB.LAST_VERDICT,
        BB.PLATFORM_STATE,
        BB.TASK_COMPLETE,
        BB.TASK_SUBMITTED,
        BB.PENDING_EVENTS,
    )
    bb.set(BB.CANCEL_REQUESTED, False)
    bb.set(BB.MILESTONES, milestones)
    bb.set(BB.CURRENT_MILESTONE_IDX, 0)
    bb.set(BB.COMPLETED_MILESTONES, set())
    bb.set(BB.COMPLETION_CLAIMED, True)
    bb.set(BB.REACT_FALLBACK_ACTIVE, False)
    bb.set(BB.TASK, "capture the mug")
    # Real starting state of a fresh episode -- no corrective replan spent yet.
    # HandleAllDone(evaluator_ablated=True) must submit on the first plan
    # exhaustion without ever touching this latch (board_runner._build_agent
    # always runs with structural_fixes=True, so this is the exact combination
    # the ablated switch ships in).
    bb.set(BB.CORRECTIVE_REPLAN_DONE, False)
    bb.set(BB.LAST_VERDICT, None)
    bb.set(BB.PLATFORM_STATE, {"_bridge": bridge})
    bb.set(BB.TASK_COMPLETE, False)
    bb.set(BB.TASK_SUBMITTED, False)
    bb.set(BB.PENDING_EVENTS, [])
    return bb


def test_evaluator_ablated_claim_advances_and_reaches_terminal_submit_on_first_pass(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Real (unseeded) tree: with the evaluator ablated, HandleAllDone must not
    burn its one lam-improvement.md CP3 corrective replan on the first plan
    exhaustion -- that would wipe BB.MILESTONES and force an extra planner call
    before ever submitting. Regression coverage for that exact bug (previously
    dodged by seeding CORRECTIVE_REPLAN_DONE=True)."""
    evaluator_calls: List[Any] = []
    planner_calls: List[Any] = []
    monkeypatch.setattr(
        "bt_agent._engine.agent.lam_agent.evaluator.evaluate_completion",
        lambda *a, **kw: evaluator_calls.append((a, kw)),
    )
    monkeypatch.setattr(
        "bt_agent._engine.agent.lam_agent.planner.generate_plan",
        lambda *a, **kw: planner_calls.append((a, kw)),
    )

    bridge = _FakeSubmitBridge()
    milestones = [{"id": "m1", "description": "capture the mug"}]
    bb = _seed_auto_accept_bb(bridge, milestones)

    root = build_agent_tree(structural_fixes=True, ablation=frozenset({"evaluator"}))

    ticks = 0
    for ticks in range(1, 6):
        root.tick_once()
        if bb.get(BB.TASK_COMPLETE):
            break

    assert bb.get(BB.TASK_COMPLETE) is True, f"did not terminate within {ticks} ticks"
    assert bb.get(BB.TASK_SUBMITTED) is True
    assert bb.get(BB.CORRECTIVE_REPLAN_DONE) is False, "must submit on the FIRST pass, no corrective replan spent"
    assert bb.get(BB.MILESTONES) == milestones, "the plan must never be wiped"
    assert bb.get(BB.CURRENT_MILESTONE_IDX) == 1
    assert bb.get(BB.COMPLETED_MILESTONES) == {0}
    assert ticks <= 5, "must terminate within a small bounded number of ticks"
    assert evaluator_calls == [], "the evaluator model must never be invoked when ablated"
    assert planner_calls == [], "no re-plan call after the initial plan"
    assert bridge.actions and bridge.actions[0].startswith("submit "), "termination must go through a real submit"


def test_both_ablated_claim_path_still_terminates(monkeypatch: pytest.MonkeyPatch) -> None:
    """evaluator+replan together: the claim path is unaffected by "replan" --
    "replan" only reshapes replan_route, which does not exist once "evaluator"
    removes the whole evaluate branch (REPLAN can only originate from the
    evaluator model, which never runs)."""
    calls: List[Any] = []
    monkeypatch.setattr(
        "bt_agent._engine.agent.lam_agent.evaluator.evaluate_completion",
        lambda *a, **kw: calls.append((a, kw)),
    )

    bridge = _FakeSubmitBridge()
    milestones = [{"id": "m1", "description": "capture the mug"}]
    bb = _seed_auto_accept_bb(bridge, milestones)

    root = build_agent_tree(structural_fixes=True, ablation=frozenset({"evaluator", "replan"}))
    assert _find(root, "GiveUpOnReplan") is None, "replan has nothing to route without an evaluate branch"

    ticks = 0
    for ticks in range(1, 6):
        root.tick_once()
        if bb.get(BB.TASK_COMPLETE):
            break

    assert bb.get(BB.TASK_COMPLETE) is True, f"did not terminate within {ticks} ticks"
    assert bb.get(BB.TASK_SUBMITTED) is True
    assert bb.get(BB.CORRECTIVE_REPLAN_DONE) is False, "must submit on the FIRST pass"
    assert bb.get(BB.MILESTONES) == milestones, "the plan must never be wiped"
    assert ticks <= 5
    assert calls == []


# ---------------------------------------------------------------------------
# "replan" ablation: forced REPLAN verdict gives up without wiping the plan
# ---------------------------------------------------------------------------


def test_replan_ablated_forced_replan_gives_up_without_wiping_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    result = EvaluatorResult(verdict=Verdict.REPLAN, reasoning="infeasible")
    monkeypatch.setattr(
        "bt_agent._engine.agent.lam_agent.evaluator.evaluate_completion",
        lambda *a, **kw: result,
    )

    milestones = [
        {"id": "m1", "description": "open door"},
        {"id": "m2", "description": "find mug"},
    ]
    bridge = _FakeSubmitBridge()
    bb = _writer(
        BB.MODEL_BACKEND,
        BB.MILESTONES,
        BB.CURRENT_MILESTONE_IDX,
        BB.SKILL_HANDLE,
        BB.SKILL_REASONING,
        BB.SCENARIO,
        BB.SKILL_REGISTRY,
        BB.OBJECT_STORE,
        BB.EVENT_SINK,
        BB.PROMPT_VARIANT,
        BB.AMEND_COUNT,
        BB.AMEND_CAP,
        BB.AMEND_NOTE,
        BB.LAST_VERDICT,
        BB.PENDING_EVENTS,
        BB.CONVERSATION,
        BB.COMPLETION_CLAIMED,
        BB.AGENT_CONFIG,
        BB.PLATFORM_STATE,
        BB.TASK_COMPLETE,
        BB.TASK_SUBMITTED,
    )
    bb.set(BB.MODEL_BACKEND, MagicMock())
    bb.set(BB.MILESTONES, milestones)
    bb.set(BB.CURRENT_MILESTONE_IDX, 0)
    bb.set(BB.SKILL_HANDLE, None)
    bb.set(BB.SKILL_REASONING, "")
    bb.set(BB.SCENARIO, "test scenario")
    bb.set(BB.SKILL_REGISTRY, None)
    bb.set(BB.OBJECT_STORE, None)
    bb.set(BB.EVENT_SINK, None)
    bb.set(BB.PROMPT_VARIANT, "tiamat_agent_submit_v5")
    bb.set(BB.AMEND_COUNT, 0)
    bb.set(BB.AMEND_CAP, 2)
    bb.set(BB.AMEND_NOTE, None)
    bb.set(BB.LAST_VERDICT, None)
    bb.set(BB.PENDING_EVENTS, [])
    bb.set(BB.CONVERSATION, [])
    bb.set(BB.COMPLETION_CLAIMED, True)
    bb.set(BB.AGENT_CONFIG, None)
    bb.set(BB.PLATFORM_STATE, {"_bridge": bridge})
    bb.set(BB.TASK_COMPLETE, False)
    bb.set(BB.TASK_SUBMITTED, False)

    root = build_agent_tree(ablation=frozenset({"replan"}))
    evaluate = _find(root, "evaluate")
    eval_node = _find(root, "EvaluateCompletion")

    # Tick 1: CheckCompletionClaimed succeeds, EvaluateCompletion starts its
    # background thread and returns RUNNING (mirrors test_three_way_verdict.py).
    evaluate.tick_once()
    assert eval_node._thread is not None
    eval_node._thread.join(timeout=2.0)

    # Tick 2: EvaluateCompletion resumes, writes LAST_VERDICT=REPLAN, and the
    # same tick cascades into verdict_route -> replan_route -> GiveUpOnReplan.
    evaluate.tick_once()

    assert bb.get(BB.LAST_VERDICT) == Verdict.REPLAN
    assert bb.get(BB.TASK_COMPLETE) is True, "REPLAN with replan ablated must still terminate"
    assert bb.get(BB.TASK_SUBMITTED) is True
    assert bb.get(BB.MILESTONES) == milestones, "replan ablation must not wipe the plan"
    assert bb.get(BB.CURRENT_MILESTONE_IDX) == 0, "no re-plan re-entry"
    assert bridge.actions, "give-up must go through a real submit step"


def test_give_up_on_replan_without_bridge_does_not_crash() -> None:
    """Missing bridge -> RUNNING, same defensive shape as HandleAllDone's own
    guard (bt/nodes/handle_all_done.py) -- never an exception, never a
    spurious SUCCESS/FAILURE that would let the tree move on falsely."""
    bb = _writer(BB.PLATFORM_STATE, BB.TASK_COMPLETE, BB.TASK_SUBMITTED, BB.PENDING_EVENTS)
    bb.set(BB.PLATFORM_STATE, {})
    bb.set(BB.TASK_COMPLETE, False)
    bb.set(BB.TASK_SUBMITTED, False)
    bb.set(BB.PENDING_EVENTS, [])

    node = GiveUpOnReplan()
    assert _tick_once(node) == py_trees.common.Status.RUNNING
    assert bb.get(BB.TASK_COMPLETE) is False


def test_give_up_on_replan_rejected_submit_stays_running() -> None:
    """Bridge present but the environment does not confirm terminal -> RUNNING,
    TASK_COMPLETE stays False. Mirrors HandleAllDone's own rejection coverage
    (test_structural_cp3.py::test_submit_rejection_does_not_set_completion)."""
    bridge = _FakeSubmitBridge(terminal=False, action_ok=False)
    bb = _writer(BB.PLATFORM_STATE, BB.TASK_COMPLETE, BB.TASK_SUBMITTED, BB.PENDING_EVENTS)
    bb.set(BB.PLATFORM_STATE, {"_bridge": bridge})
    bb.set(BB.TASK_COMPLETE, False)
    bb.set(BB.TASK_SUBMITTED, False)
    bb.set(BB.PENDING_EVENTS, [])

    node = GiveUpOnReplan()
    assert _tick_once(node) == py_trees.common.Status.RUNNING
    assert bb.get(BB.TASK_COMPLETE) is False
    assert bb.get(BB.TASK_SUBMITTED) is False
    assert bridge.actions and bridge.actions[0].startswith("submit ")
    events = bb.get(BB.PENDING_EVENTS)
    assert events[-1]["kind"] == "replan_ablated_submit_rejected"


# ---------------------------------------------------------------------------
# "context" ablation: unbounded token budget threaded into the actor call
# ---------------------------------------------------------------------------

_MILESTONE = {
    "id": "m1",
    "description": "capture the mug",
    "detailed_instructions": "find and capture the mug",
    "verify_condition": "Is the mug captured?",
}


def test_context_ablation_threads_unbounded_token_budget_into_actor_call(monkeypatch: pytest.MonkeyPatch) -> None:
    """SelectSkill._select_in_background overrides ac.token_budget with the
    unbounded sentinel only when "context" is in AgentConfig.ablation; the
    default (empty ablation) config must still pass ac.token_budget verbatim
    -- byte-identical wiring for every existing variant."""
    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        "bt_agent._engine.agent.lam_agent.actor.select_skill",
        lambda *a, **kw: captured.append(kw) or ("CONTINUE", "wait", {}, "", "", {}),
    )

    ablated = SelectSkill()
    ablated._select_in_background(
        my_epoch=0,
        model=MagicMock(),
        tools=[],
        milestone=_MILESTONE,
        milestone_index=0,
        total_milestones=1,
        object_summaries="",
        history=[],
        injected_images=None,
        event_sink=None,
        prompt_variant="tiamat_agent_submit_v5",
        config=AgentConfig(ablation=frozenset({"context"})),
        scenario="",
    )
    assert captured[-1]["max_history_tokens"] == _UNBOUNDED_TOKEN_BUDGET

    default_node = SelectSkill()
    default_config = AgentConfig()
    default_node._select_in_background(
        my_epoch=0,
        model=MagicMock(),
        tools=[],
        milestone=_MILESTONE,
        milestone_index=0,
        total_milestones=1,
        object_summaries="",
        history=[],
        injected_images=None,
        event_sink=None,
        prompt_variant="tiamat_agent_submit_v5",
        config=default_config,
        scenario="",
    )
    assert captured[-1]["max_history_tokens"] == default_config.actor.token_budget


def test_context_ablated_history_is_verbatim_with_no_summarizer_call() -> None:
    """At the unbounded budget CP3 wires in, actor.py's _split_history early-
    return (actor.py:827) always fires: every history entry -- including the
    earliest -- survives verbatim in the composed prompt, and neither
    compaction function (extractive or LLM) ever runs."""
    from bt_agent._engine.agent.lam_agent.actor import select_skill

    marker = "EARLY_MARKER_frisbee_query_result_9f3a"
    history: List[Dict[str, Any]] = [
        {
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": "call_0", "type": "function", "function": {"name": "query", "arguments": "{}"}}],
        },
        {"role": "tool", "tool_call_id": "call_0", "content": marker},
    ]
    # Pad well past both the default token budget and the keep_recent*2
    # message-count shortcut, so a bounded budget would summarize the marker away.
    for i in range(1, 20):
        history.append({
            "role": "assistant",
            "content": "",
            "tool_calls": [{"id": f"call_{i}", "type": "function", "function": {"name": "query", "arguments": "{}"}}],
        })
        history.append({"role": "tool", "tool_call_id": f"call_{i}", "content": "x" * 2000})

    model = MagicMock()
    model.query_tool_call.return_value = ("wait", {}, {"choices": [{"message": {"content": ""}}]})
    model.query.side_effect = AssertionError("LLM compaction must never be invoked when context is ablated")

    select_skill(
        model,
        tools=[],
        milestone=_MILESTONE,
        milestone_index=0,
        total_milestones=1,
        history=history,
        max_history_tokens=_UNBOUNDED_TOKEN_BUDGET,
        compaction="llm",
        reasoning_mode="act",
    )

    assert model.query.call_count == 0, "the summarizer must never be invoked"
    composed_messages = model.query_tool_call.call_args.args[0]
    assert marker in json.dumps(composed_messages), "the earliest history entry must survive verbatim"
    assert "Previous Actions (summarized)" not in composed_messages[0]["content"], "no summary block was built"


# ---------------------------------------------------------------------------
# "context" ablation: backend context-overflow classification + give-up submit
# ---------------------------------------------------------------------------


def _context_overflow_error(message: str) -> requests.exceptions.HTTPError:
    err = requests.exceptions.HTTPError(message)
    err.response = SimpleNamespace(status_code=400)
    return err


def test_is_context_overflow_error_classifies_backend_400_context_length() -> None:
    overflow = _context_overflow_error(
        "400 Client Error: Bad Request. This model's maximum context length is 8192 tokens."
    )
    assert _is_context_overflow_error(overflow) is True

    other_400 = _context_overflow_error("400 Client Error: Bad Request. invalid tool schema.")
    assert _is_context_overflow_error(other_400) is False

    # CP3 review round: "too many tokens" was dropped from the marker list --
    # too generic, a schema-validation 400 can legitimately mention "tokens"
    # without being a context-length overflow at all.
    schema_tokens_400 = _context_overflow_error(
        "400 Client Error: Bad Request. the 'tags' field has too many tokens for this parameter."
    )
    assert _is_context_overflow_error(schema_tokens_400) is False

    connection_error = requests.exceptions.ConnectionError("connection refused")
    assert _is_context_overflow_error(connection_error) is False


def test_context_ablated_overflow_produces_context_overflow_decision(monkeypatch: pytest.MonkeyPatch) -> None:
    overflow = _context_overflow_error("400 ... maximum context length is 8192 tokens ...")
    monkeypatch.setattr(
        "bt_agent._engine.agent.lam_agent.actor.select_skill",
        lambda *a, **kw: (_ for _ in ()).throw(overflow),
    )

    node = SelectSkill()
    node._select_in_background(
        my_epoch=0,
        model=MagicMock(),
        tools=[],
        milestone=_MILESTONE,
        milestone_index=0,
        total_milestones=1,
        object_summaries="",
        history=[],
        injected_images=None,
        event_sink=None,
        prompt_variant="tiamat_agent_submit_v5",
        config=AgentConfig(ablation=frozenset({"context"})),
        scenario="",
    )

    assert node._result[0] == "CONTEXT_OVERFLOW"


def test_context_not_ablated_overflow_falls_back_to_old_wait_behavior(monkeypatch: pytest.MonkeyPatch) -> None:
    """Byte-identical guardrail: the exact same backend error, without
    "context" ablated, must still take the pre-existing generic "wait"
    recovery -- classification is gated on the ablation flag, not on the
    error shape alone."""
    overflow = _context_overflow_error("400 ... maximum context length is 8192 tokens ...")
    monkeypatch.setattr(
        "bt_agent._engine.agent.lam_agent.actor.select_skill",
        lambda *a, **kw: (_ for _ in ()).throw(overflow),
    )

    node = SelectSkill()
    node._select_in_background(
        my_epoch=0,
        model=MagicMock(),
        tools=[],
        milestone=_MILESTONE,
        milestone_index=0,
        total_milestones=1,
        object_summaries="",
        history=[],
        injected_images=None,
        event_sink=None,
        prompt_variant="tiamat_agent_submit_v5",
        config=AgentConfig(),
        scenario="",
    )

    assert node._result[0] == "CONTINUE"
    assert node._result[1] == "wait"


def test_context_overflow_give_up_submits_and_completes() -> None:
    bridge = _FakeSubmitBridge()
    bb = _writer(BB.PLATFORM_STATE, BB.TASK_COMPLETE, BB.TASK_SUBMITTED, BB.PENDING_EVENTS)
    bb.set(BB.PLATFORM_STATE, {"_bridge": bridge})
    bb.set(BB.TASK_COMPLETE, False)
    bb.set(BB.TASK_SUBMITTED, False)
    bb.set(BB.PENDING_EVENTS, [])

    node = SelectSkill()
    node._result = ("CONTEXT_OVERFLOW", "", {}, "backend rejected: maximum context length", "", {})
    status = node.update()

    assert status == py_trees.common.Status.RUNNING, "must not cascade into ExecuteSkill this same tick"
    assert bb.get(BB.TASK_COMPLETE) is True
    assert bb.get(BB.TASK_SUBMITTED) is True
    assert bridge.actions and bridge.actions[0].startswith("submit ")
    events = bb.get(BB.PENDING_EVENTS)
    assert events[-1]["kind"] == "context_overflow_give_up"
    assert "maximum context length" in events[-1]["data"]["detail"]


def test_context_overflow_give_up_without_bridge_stays_running() -> None:
    bb = _writer(BB.PLATFORM_STATE, BB.TASK_COMPLETE, BB.TASK_SUBMITTED, BB.PENDING_EVENTS)
    bb.set(BB.PLATFORM_STATE, {})
    bb.set(BB.TASK_COMPLETE, False)
    bb.set(BB.TASK_SUBMITTED, False)
    bb.set(BB.PENDING_EVENTS, [])

    node = SelectSkill()
    node._result = ("CONTEXT_OVERFLOW", "", {}, "detail", "", {})
    assert node.update() == py_trees.common.Status.RUNNING
    assert bb.get(BB.TASK_COMPLETE) is False

    events = bb.get(BB.PENDING_EVENTS)
    assert events[-1]["kind"] == "context_overflow_submit_rejected"


# ---------------------------------------------------------------------------
# "planner" ablation: zero model calls, static task-text milestone
# ---------------------------------------------------------------------------


def test_planner_ablated_zero_model_calls_static_milestone(monkeypatch: pytest.MonkeyPatch) -> None:
    generate_plan_mock = MagicMock(
        side_effect=AssertionError("generate_plan must never be called when planner is ablated")
    )
    monkeypatch.setattr("bt_agent._engine.agent.lam_agent.planner.generate_plan", generate_plan_mock)

    node = PlanTask()
    node._plan_in_background(
        my_epoch=0,
        model=MagicMock(),
        task="capture the mug",
        scenario="test scenario",
        skill_registry=None,
        object_store=None,
        platform_state={},
        milestone_schema_extension=None,
        milestone_fields_guidance="",
        event_sink=None,
        replan_context=None,
        prompt_variant="tiamat_agent_submit_v5",
        config=AgentConfig(ablation=frozenset({"planner"})),
    )

    assert generate_plan_mock.call_count == 0
    milestones, reasoning, _plan_data = node._result
    assert len(milestones) == 1
    assert milestones[0]["id"] == "fallback_task"
    assert milestones[0]["description"] == "capture the mug"
    assert milestones[0]["type"] == "direct"
    assert "ablated" in reasoning.lower()


def test_planner_ablated_replan_reentry_reinjects_same_static_milestone(monkeypatch: pytest.MonkeyPatch) -> None:
    """ApplyReplan re-fires PlanTask with BB.REPLAN_CONTEXT set; the re-fire
    must land on the exact same static template, with no model call, so a
    planner-ablated replan is a true no-op re-injection."""
    generate_plan_mock = MagicMock(
        side_effect=AssertionError("generate_plan must never be called when planner is ablated")
    )
    monkeypatch.setattr("bt_agent._engine.agent.lam_agent.planner.generate_plan", generate_plan_mock)

    node = PlanTask()
    node._plan_in_background(
        my_epoch=0,
        model=MagicMock(),
        task="capture the mug",
        scenario="test scenario",
        skill_registry=None,
        object_store=None,
        platform_state={},
        milestone_schema_extension=None,
        milestone_fields_guidance="",
        event_sink=None,
        replan_context={
            "failed_milestone_id": "fallback_task",
            "failed_description": "capture the mug",
            "reason": "infeasible",
            "completed_descriptions": [],
        },
        prompt_variant="tiamat_agent_submit_v5",
        config=AgentConfig(ablation=frozenset({"planner"})),
    )

    assert generate_plan_mock.call_count == 0
    milestones, _reasoning, _plan_data = node._result
    assert milestones == [{
        "id": "fallback_task",
        "description": "capture the mug",
        "detailed_instructions": "capture the mug",
        "verify_condition": "Is the task complete?",
        "type": "direct",
        "dependencies": [],
    }]


def test_planner_not_ablated_still_calls_generate_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """Byte-identical guardrail: default (empty) ablation must still reach
    generate_plan exactly as before."""
    calls: List[Any] = []
    monkeypatch.setattr(
        "bt_agent._engine.agent.lam_agent.planner.generate_plan",
        lambda *a, **kw: calls.append((a, kw)) or ([], "empty", {}),
    )

    node = PlanTask()
    node._plan_in_background(
        my_epoch=0,
        model=MagicMock(),
        task="capture the mug",
        scenario="test scenario",
        skill_registry=None,
        object_store=None,
        platform_state={},
        milestone_schema_extension=None,
        milestone_fields_guidance="",
        event_sink=None,
        replan_context=None,
        prompt_variant="tiamat_agent_submit_v5",
        config=AgentConfig(),
    )

    assert len(calls) == 1


def _claim_complete_model() -> MagicMock:
    model = MagicMock()
    model.query_tool_call.return_value = ("claim_complete", {}, {"choices": [{"message": {"content": "done"}}]})
    return model


def test_planner_ablated_alone_replan_verdict_gives_up_without_wiping_plan(monkeypatch: pytest.MonkeyPatch) -> None:
    """CP3 review round: "planner" ablated ALONE (evaluator and replan both
    live) -- a forced REPLAN verdict must still give up rather than cycling
    (ApplyReplan wipe -> PlanTask re-fire -> same static milestone -> claim ->
    REPLAN -> ...) to the tick cap. build_agent_tree routes REPLAN to
    GiveUpOnReplan whenever "planner" is ablated, even with "replan" itself
    not ablated."""
    generate_plan_mock = MagicMock(side_effect=AssertionError("planner is ablated; generate_plan must not be called"))
    monkeypatch.setattr("bt_agent._engine.agent.lam_agent.planner.generate_plan", generate_plan_mock)
    monkeypatch.setattr(
        "bt_agent._engine.agent.lam_agent.evaluator.evaluate_completion",
        lambda *a, **kw: EvaluatorResult(verdict=Verdict.REPLAN, reasoning="infeasible"),
    )

    bridge = _FakeSubmitBridge()
    config = AgentConfig(structural_fixes=True, ablation=frozenset({"planner"}))
    agent = LAMAgent(_claim_complete_model(), config=config)
    runner = AgentRunner(agent)

    completed = runner.run(
        "capture the mug",
        "test scenario",
        max_ticks=40,
        tick_rate=0.02,
        state_provider=lambda: {"_bridge": bridge},
    )

    assert completed, "planner ablated alone must still terminate on REPLAN -- no live-lock"
    assert generate_plan_mock.call_count == 0
    assert agent.blackboard.get(BB.MILESTONES), "GiveUpOnReplan must not wipe the static plan"
    assert agent.blackboard.get(BB.CURRENT_MILESTONE_IDX) == 0, "no re-plan re-entry"
    assert bridge.actions and bridge.actions[0].startswith("submit ")


# ---------------------------------------------------------------------------
# Pairwise compositions: tick a full tree to termination, hard tick bound
# ---------------------------------------------------------------------------


def test_composition_context_and_planner_terminates_without_livelock(monkeypatch: pytest.MonkeyPatch) -> None:
    generate_plan_mock = MagicMock(side_effect=AssertionError("planner is ablated; generate_plan must not be called"))
    monkeypatch.setattr("bt_agent._engine.agent.lam_agent.planner.generate_plan", generate_plan_mock)
    monkeypatch.setattr(
        "bt_agent._engine.agent.lam_agent.evaluator.evaluate_completion",
        lambda *a, **kw: EvaluatorResult(verdict=Verdict.PASS, reasoning="looks done"),
    )

    bridge = _FakeSubmitBridge()
    config = AgentConfig(structural_fixes=True, ablation=frozenset({"context", "planner"}))
    agent = LAMAgent(_claim_complete_model(), config=config)
    runner = AgentRunner(agent)

    completed = runner.run(
        "capture the mug",
        "test scenario",
        max_ticks=40,
        tick_rate=0.02,
        state_provider=lambda: {"_bridge": bridge},
    )

    assert completed, "composition must terminate -- no live-lock"
    assert generate_plan_mock.call_count == 0
    assert agent.blackboard.get(BB.TASK_SUBMITTED) is True
    assert bridge.actions and bridge.actions[0].startswith("submit ")


def test_composition_planner_and_replan_terminates_without_livelock(monkeypatch: pytest.MonkeyPatch) -> None:
    generate_plan_mock = MagicMock(side_effect=AssertionError("planner is ablated; generate_plan must not be called"))
    monkeypatch.setattr("bt_agent._engine.agent.lam_agent.planner.generate_plan", generate_plan_mock)
    monkeypatch.setattr(
        "bt_agent._engine.agent.lam_agent.evaluator.evaluate_completion",
        lambda *a, **kw: EvaluatorResult(verdict=Verdict.REPLAN, reasoning="infeasible"),
    )

    bridge = _FakeSubmitBridge()
    config = AgentConfig(structural_fixes=True, ablation=frozenset({"planner", "replan"}))
    agent = LAMAgent(_claim_complete_model(), config=config)
    runner = AgentRunner(agent)

    completed = runner.run(
        "capture the mug",
        "test scenario",
        max_ticks=40,
        tick_rate=0.02,
        state_provider=lambda: {"_bridge": bridge},
    )

    assert completed, "composition must terminate -- no live-lock"
    assert generate_plan_mock.call_count == 0
    assert agent.blackboard.get(BB.MILESTONES), "GiveUpOnReplan (replan ablated) must not wipe the static plan"
    assert bridge.actions and bridge.actions[0].startswith("submit ")


def test_composition_evaluator_and_context_terminates_without_livelock(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        "bt_agent._engine.agent.lam_agent.planner.generate_plan",
        lambda *a, **kw: ([dict(_MILESTONE)], "reasoning", {}),
    )
    evaluator_calls: List[Any] = []
    monkeypatch.setattr(
        "bt_agent._engine.agent.lam_agent.evaluator.evaluate_completion",
        lambda *a, **kw: evaluator_calls.append((a, kw)),
    )

    bridge = _FakeSubmitBridge()
    config = AgentConfig(structural_fixes=True, ablation=frozenset({"evaluator", "context"}))
    agent = LAMAgent(_claim_complete_model(), config=config)
    runner = AgentRunner(agent)

    completed = runner.run(
        "capture the mug",
        "test scenario",
        max_ticks=40,
        tick_rate=0.02,
        state_provider=lambda: {"_bridge": bridge},
    )

    assert completed, "composition must terminate -- no live-lock"
    assert evaluator_calls == [], "evaluator ablated: the model must never be invoked"
    assert bridge.actions and bridge.actions[0].startswith("submit ")
