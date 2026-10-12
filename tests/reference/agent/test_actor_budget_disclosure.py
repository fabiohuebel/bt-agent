"""Symmetric action-budget disclosure in the actor's context.

tiamat_world's docs/spec/budget-disclosure.md, CP4: the model driving this
agent must see the same "Action budget: N skill calls; wall limit: W s."
(episode start) / "Actions used: K of N." (every turn) lines every other
board arm sees. Absent (``action_budget=None``, disclosure not configured)
renders byte-identical to today; configured, BOTH lines render on every
actor turn (K==0 and K>=1 alike) -- this agent rebuilds its context every
turn, so the episode-start line must persist rather than fire once.
"""

from __future__ import annotations

from unittest.mock import MagicMock

from bt_agent._engine.agent.lam_agent.actor import render_action_budget_disclosure, select_skill

_MILESTONE = {
    "id": "m1",
    "description": "capture the mug",
    "detailed_instructions": "find and capture the mug",
    "verify_condition": "Is the mug captured?",
}


def _mock_model() -> MagicMock:
    model = MagicMock()
    model.query_tool_call.return_value = ("wait", {}, {"choices": [{"message": {"content": "looking"}}]})
    return model


def _system_content(model: MagicMock) -> str:
    messages = model.query_tool_call.call_args.args[0]
    return messages[0]["content"]


# ---------------------------------------------------------------------------
# render_action_budget_disclosure — pure rendering
# ---------------------------------------------------------------------------


class TestRenderActionBudgetDisclosure:
    def test_none_renders_empty(self) -> None:
        assert render_action_budget_disclosure(None) == ""

    def test_empty_dict_renders_empty(self) -> None:
        assert render_action_budget_disclosure({}) == ""

    def test_first_turn_shows_both_lines(self) -> None:
        text = render_action_budget_disclosure({
            "skill_call_cap": 15,
            "wall_time_cap_s": 1800.0,
            "actions_used": 0,
        })
        assert "Action budget: 15 skill calls; wall limit: 1800 s." in text
        assert "Actions used: 0 of 15." in text

    def test_later_turn_shows_both_lines_with_live_values(self) -> None:
        text = render_action_budget_disclosure({
            "skill_call_cap": 15,
            "wall_time_cap_s": 1800.0,
            "actions_used": 3,
        })
        assert "Action budget: 15 skill calls; wall limit: 1800 s." in text
        assert "Actions used: 3 of 15." in text


# ---------------------------------------------------------------------------
# select_skill() — actor system-prompt composition
# ---------------------------------------------------------------------------


class TestSelectSkillActionBudgetThreading:
    def test_absent_action_budget_is_byte_identical(self) -> None:
        """Default (action_budget=None) must not append any disclosure text
        -- the tiamat_agent_submit_v5 system prompt is exactly today's bytes."""
        model = _mock_model()
        select_skill(
            model,
            tools=[],
            milestone=_MILESTONE,
            milestone_index=0,
            total_milestones=1,
            prompt_variant="tiamat_agent_submit_v5",
            reasoning_mode="act",
        )
        content = _system_content(model)
        assert "# Action budget" not in content
        assert "Actions used" not in content

    def test_configured_first_turn_shows_banner_and_counter(self) -> None:
        model = _mock_model()
        select_skill(
            model,
            tools=[],
            milestone=_MILESTONE,
            milestone_index=0,
            total_milestones=1,
            prompt_variant="tiamat_agent_submit_v5",
            reasoning_mode="act",
            action_budget={"skill_call_cap": 15, "wall_time_cap_s": 1800.0, "actions_used": 0},
        )
        content = _system_content(model)
        assert "Action budget: 15 skill calls; wall limit: 1800 s." in content
        assert "Actions used: 0 of 15." in content

    def test_configured_later_turn_shows_both_lines_with_live_values(self) -> None:
        model = _mock_model()
        select_skill(
            model,
            tools=[],
            milestone=_MILESTONE,
            milestone_index=0,
            total_milestones=1,
            prompt_variant="tiamat_agent_submit_v5",
            reasoning_mode="act",
            action_budget={"skill_call_cap": 15, "wall_time_cap_s": 1800.0, "actions_used": 4},
        )
        content = _system_content(model)
        assert "Action budget: 15 skill calls; wall limit: 1800 s." in content
        assert "Actions used: 4 of 15." in content
