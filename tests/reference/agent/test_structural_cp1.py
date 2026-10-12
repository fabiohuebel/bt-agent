"""CP1 — planner + evaluator token budgets & plan-schema slimming.

docs/spec/lam-improvement.md CP1. These tests pin the structural-fix behavior
AND the byte-identical guarantee for the frozen path:

- The frozen path omits ``max_tokens`` on the synthesis call (backend default),
  which truncates the plan into an empty milestone list on weak models.
- The structural path threads an explicit ``synthesis_max_tokens`` so an
  over-long scene synthesizes a NON-EMPTY plan instead of an empty one — no
  silent empty plan (Success criterion 1).
- Schema slimming drops the free-text ``description`` hints (output-token
  pressure) while preserving the required list + enum constraints.
- The evaluator inspection runs when the evidence-selection budget is large
  enough to return a real ``skip_inspection=False`` (Success criterion 5).

No model endpoint is touched — a fake backend stands in for the VLM.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from bt_agent._engine.agent.lam_agent.evaluator import evaluate_completion
from bt_agent._engine.agent.lam_agent.planner import (
    _MILESTONE_CORE_REQUIRED,
    _synthesize_plan,
    build_milestone_schema,
)

_VALID_MILESTONE = {
    "id": "m1",
    "description": "d",
    "detailed_instructions": "i",
    "verify_condition": "v",
    "type": "direct",
    "dependencies": [],
}


class _TruncatingModel:
    """query_structured returns an empty plan below a token threshold.

    Mirrors the real failure: a weak model that runs out of output tokens emits
    a partial/empty milestone list. Above the threshold it returns a full plan.
    """

    def __init__(self, threshold: int) -> None:
        self.threshold = threshold
        self.seen_max_tokens: Any = "UNSET"

    def query_structured(
        self,
        messages: List[Dict[str, Any]],
        schema: Dict[str, Any],
        images: Optional[List[bytes]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        phase: str = "other",
    ) -> Dict[str, Any]:
        self.seen_max_tokens = max_tokens
        budget = max_tokens if max_tokens is not None else 2048  # backend default
        if budget < self.threshold:
            return {"reasoning": "truncated", "milestones": []}
        return {"reasoning": "ok", "milestones": [dict(_VALID_MILESTONE)]}


def test_structural_budget_eliminates_truncation() -> None:
    """Over-long scene + threaded budget → non-empty plan (never a silent empty)."""
    model = _TruncatingModel(threshold=4096)
    milestones, _reasoning = _synthesize_plan(
        model,
        task="capture the umbrella",
        scenario="s" * 5000,  # over-long scene
        gather_history=[],
        prompt_variant="tiamat_agent_submit_v5",
        synthesis_max_tokens=8192,
        slim_schema=True,
    )
    assert model.seen_max_tokens == 8192
    assert len(milestones) == 1, "structural budget must synthesize a non-empty plan"


def test_frozen_path_omits_max_tokens() -> None:
    """Byte-identical guard: with no budget threaded, max_tokens stays None.

    The frozen variants must keep hitting the backend-default path exactly as
    before — the call omits max_tokens (None), never an injected value.
    """
    model = _TruncatingModel(threshold=4096)
    _synthesize_plan(
        model,
        task="capture the umbrella",
        scenario="short scene",
        gather_history=[],
        prompt_variant="tiamat_agent_submit_v5",
    )
    assert model.seen_max_tokens is None


def test_schema_slim_drops_descriptions_keeps_shape() -> None:
    """slim=True removes free-text descriptions but preserves required + enums."""
    ext = {"interaction_type": {"type": "string", "description": "the skill"}}
    full = build_milestone_schema(ext, slim=False)
    slim = build_milestone_schema(ext, slim=True)

    item_full = full["properties"]["milestones"]["items"]
    item_slim = slim["properties"]["milestones"]["items"]

    # Required list + enum constraint are untouched by slimming.
    assert item_slim["required"] == item_full["required"] == _MILESTONE_CORE_REQUIRED
    assert item_slim["properties"]["type"]["enum"] == ["direct", "explore", "dependent"]

    # No milestone property carries a description under slim.
    assert all("description" not in spec for spec in item_slim["properties"].values())
    # The frozen schema still carries them (proof slimming is a real delta).
    assert any("description" in spec for spec in item_full["properties"].values())


def test_slim_default_is_byte_identical() -> None:
    """build_milestone_schema(ext) == build_milestone_schema(ext, slim=False)."""
    ext = {"interaction_type": {"type": "string", "description": "the skill"}}
    assert build_milestone_schema(ext) == build_milestone_schema(ext, slim=False)


class _InspectingModel:
    """Evidence step returns skip_inspection=False; verdict step returns pass."""

    def __init__(self) -> None:
        self.evidence_max_tokens: Any = "UNSET"

    def query_structured(
        self,
        messages: List[Dict[str, Any]],
        schema: Dict[str, Any],
        images: Optional[List[bytes]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        phase: str = "other",
    ) -> Dict[str, Any]:
        props = schema.get("properties", {})
        if "skip_inspection" in props:
            self.evidence_max_tokens = max_tokens
            return {
                "skip_inspection": False,
                "inspect_handles": ["umbrella 1"],
                "reasoning": "need to see it",
            }
        return {"verdict": "pass", "reasoning": "looks right", "correction": None}


class _EmptyStore:
    def get(self, _handle: str) -> None:
        return None


def test_evaluator_inspection_runs_with_budget() -> None:
    """A large evidence budget lets inspection actually run (skip=False resolved)."""
    model = _InspectingModel()
    result = evaluate_completion(
        model,
        milestone={"id": "m1", "description": "photograph umbrella", "verify_condition": "v"},
        skill_observation="captured umbrella 1",
        store=_EmptyStore(),
        store_summaries="umbrella 1",
        prompt_variant="tiamat_agent_submit_v5",
        evidence_max_tokens=2048,
        verdict_max_tokens=2048,
    )
    assert result.skipped_inspection is False, "inspection must run, not rubber-stamp"
    assert result.inspected_handles == ["umbrella 1"]
    assert model.evidence_max_tokens == 2048, "config budget must reach the evidence call"
