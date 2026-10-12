"""Tests for format_for_prompt observation cap (bt/nodes/_conversation.py).

FIX 2: observation[:200] truncated query recall-lists mid-list, causing the
evaluator to false-flag 'hallucination' and emit spurious amends. The cap is
raised to 800 (uniformly — this module is substrate-agnostic and must not
hardcode skill names). reasoning stays capped at 150.
"""

from __future__ import annotations

from bt_agent._engine.bt.nodes._conversation import ToolCall, format_for_prompt


def test_observation_cap_raised_to_800():
    obs = "x" * 1000
    line = format_for_prompt([ToolCall(name="query", args={}, observation=obs)])
    # The observation is repr()'d, so count the x's actually rendered.
    assert line.count("x") == 800


def test_recall_list_survives_past_old_200_cap():
    # A query recall-list that the old 200-char cap would have cut mid-list.
    items = ", ".join(f"object_{i}" for i in range(60))  # ~600 chars
    assert 200 < len(items) <= 800
    line = format_for_prompt([ToolCall(name="query", args={}, observation=items)])
    assert "object_59" in line  # last item preserved (would be lost at 200)


def test_reasoning_still_capped_at_150():
    reasoning = "r" * 300
    line = format_for_prompt([ToolCall(name="navigate", args={"reasoning": reasoning}, observation="ok")])
    assert line.count("r") == 150


def test_empty_calls_unchanged():
    assert format_for_prompt([]) == "No actions taken."
