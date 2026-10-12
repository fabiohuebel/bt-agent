"""SelectSkill threads BB.PLATFORM_STATE["action_budget"] into
select_skill()'s ``action_budget`` kwarg (tiamat_world docs/spec/
budget-disclosure.md, CP4).

board_runner.drive_episode is the producer (pushes "action_budget" onto
BB.PLATFORM_STATE only when the episode's scaffold_config configured
disclosure); this module tests only the consumer side --
SelectSkill._select_in_background threading the value through to the actor
call, unit-level like test_ablation_switches.py's CP3 "context" coverage.
"""

from __future__ import annotations

from typing import Any, Dict, List
from unittest.mock import MagicMock

import pytest
from bt_agent._engine.bt.nodes.select_skill import SelectSkill

_MILESTONE = {
    "id": "m1",
    "description": "capture the mug",
    "detailed_instructions": "find and capture the mug",
    "verify_condition": "Is the mug captured?",
}


def test_action_budget_threads_into_actor_call(monkeypatch: pytest.MonkeyPatch) -> None:
    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        "bt_agent._engine.agent.lam_agent.actor.select_skill",
        lambda *a, **kw: captured.append(kw) or ("CONTINUE", "wait", {}, "", "", {}),
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
        config=None,
        scenario="",
        action_budget={"skill_call_cap": 15, "wall_time_cap_s": 1800.0, "actions_used": 0},
    )
    assert captured[-1]["action_budget"] == {
        "skill_call_cap": 15,
        "wall_time_cap_s": 1800.0,
        "actions_used": 0,
    }


def test_default_action_budget_is_none(monkeypatch: pytest.MonkeyPatch) -> None:
    """Byte-identical wiring guardrail: a call site that omits action_budget
    (e.g. every existing caller/test predating this feature) must thread
    None through, unchanged."""
    captured: List[Dict[str, Any]] = []
    monkeypatch.setattr(
        "bt_agent._engine.agent.lam_agent.actor.select_skill",
        lambda *a, **kw: captured.append(kw) or ("CONTINUE", "wait", {}, "", "", {}),
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
        config=None,
        scenario="",
    )
    assert captured[-1]["action_budget"] is None
