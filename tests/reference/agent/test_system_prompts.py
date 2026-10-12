"""Tests for the system_prompts variant infrastructure (Step 5).

The release ships one variant, ``tiamat_agent_submit_v5``. Covers:
- ``get_prompts(variant)`` registry + discovery via ``list_variants()``
- The released variant exposes every ``PromptBundle`` field and the
  placeholders each call site interpolates.
- Templates render with ``str.format`` (no stray braces).

Byte-level pinning of the released strings lives in
``tests/test_released_prompts.py``.
"""

from __future__ import annotations

import dataclasses

import pytest
from bt_agent._engine.agent.system_prompts import PromptBundle, get_prompts, list_variants

RELEASED = "tiamat_agent_submit_v5"

# ---------------------------------------------------------------------------
# 1. Registry
# ---------------------------------------------------------------------------


def test_list_variants_is_the_released_variant() -> None:
    assert list_variants() == [RELEASED]


def test_get_prompts_unknown_variant_raises_with_helpful_message() -> None:
    with pytest.raises(ValueError, match="Unknown prompt variant"):
        get_prompts("does_not_exist")


def test_get_prompts_returns_promptbundle() -> None:
    bundle = get_prompts(RELEASED)
    assert isinstance(bundle, PromptBundle)
    assert bundle.name == RELEASED


def test_get_prompts_default_is_the_released_variant() -> None:
    assert get_prompts().name == RELEASED


# ---------------------------------------------------------------------------
# 2. Required fields and placeholders
# ---------------------------------------------------------------------------


def test_released_variant_provides_every_field() -> None:
    bundle = get_prompts(RELEASED)
    for field in dataclasses.fields(PromptBundle):
        assert getattr(bundle, field.name), f"{field.name} must be non-empty"


def test_planner_gather_has_required_placeholders() -> None:
    bundle = get_prompts(RELEASED)
    assert "{task}" in bundle.planner_gather
    assert "{scenario}" in bundle.planner_gather


def test_planner_synthesis_has_required_placeholders() -> None:
    bundle = get_prompts(RELEASED)
    for ph in ("{task}", "{scenario}", "{gathered_info}", "{milestone_fields}"):
        assert ph in bundle.planner_synthesis


def test_planner_replan_has_required_placeholders() -> None:
    bundle = get_prompts(RELEASED)
    for ph in (
        "{completed_block}",
        "{failed_description}",
        "{failed_milestone_id}",
        "{reason}",
        "{base_user_prompt}",
    ):
        assert ph in bundle.planner_replan, f"missing replan placeholder {ph}"


def test_actor_has_required_placeholders() -> None:
    bundle = get_prompts(RELEASED)
    for ph in (
        "{milestone_idx}",
        "{total_milestones}",
        "{milestone_description}",
        "{milestone_instructions}",
        "{milestone_verify}",
        "{object_summaries}",
        "{history_summary}",
        "{budget_disclosure}",
    ):
        assert ph in bundle.actor, f"missing actor placeholder {ph}"


def test_planner_gather_mentions_done_gathering() -> None:
    """The planner must be told about the framework's done_gathering tool."""
    assert "done_gathering" in get_prompts(RELEASED).planner_gather


# ---------------------------------------------------------------------------
# 3. Rendering
# ---------------------------------------------------------------------------


def test_planner_replan_renders_correctly() -> None:
    bundle = get_prompts(RELEASED)
    completed_block = "  - milestone_a\n  - milestone_b"
    base = "Begin planning."
    rendered = bundle.planner_replan.format(
        completed_block=completed_block,
        failed_description="reach the target",
        failed_milestone_id="m1",
        reason="path planner found no route",
        base_user_prompt=base,
    )
    assert completed_block in rendered
    assert '"reach the target"' in rendered
    assert "id=m1" in rendered
    assert "path planner found no route" in rendered
    assert rendered.endswith(base)


_ACTOR_KWARGS = dict(
    milestone_idx=1,
    total_milestones=1,
    milestone_description="capture the mug",
    milestone_instructions="find and capture the mug",
    milestone_verify="Is the mug captured?",
    object_summaries="None yet",
    history_summary="",
    scenario="Some scenario",
)


def test_actor_renders_disclosure_when_configured() -> None:
    rendered = get_prompts(RELEASED).actor.format(
        budget_disclosure="\n\n# Action budget\nAction budget: 5 skill calls; wall limit: 60 s.\nActions used: 0 of 5.",
        **_ACTOR_KWARGS,
    )
    assert "Action budget: 5 skill calls; wall limit: 60 s." in rendered


def test_actor_disclosure_placeholder_is_a_trailing_no_op_when_absent() -> None:
    """budget_disclosure="" (disclosure off) renders the template without it."""
    actor = get_prompts(RELEASED).actor
    assert actor.endswith("{budget_disclosure}")
    rendered = actor.format(budget_disclosure="", **_ACTOR_KWARGS)
    assert rendered == actor.removesuffix("{budget_disclosure}").format(**_ACTOR_KWARGS)


def test_synthesis_and_evaluator_render_without_stray_braces() -> None:
    bundle = get_prompts(RELEASED)
    bundle.planner_synthesis.format(task="t", scenario="s", gathered_info="g", milestone_fields="m")
    bundle.evaluator.format()
