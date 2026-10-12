"""Evaluator — two-step milestone verification with a three-way verdict.

Paper §5 Invariant I7: the Evaluator emits ``(d, δ)`` where
``d ∈ {pass, amend, replan}`` and ``δ`` is an optional corrective note
(required iff ``d == amend``).

Two-step flow:
  1. **Evidence selection** — the VLM sees the actor trace, milestone,
     and K store summaries.  It decides which store entries (if any)
     to inspect.  If none are needed, it skips to Step 2 directly
     (single VLM call, same cost as before).
  2. **Verdict** — the VLM sees the resolved evidence (images, metadata
     from the selected store entries) and emits ``(d, δ)``.

The BT node (EvaluateCompletion) handles lifecycle: threading, blackboard,
tick status, evidence resolution from K, and amend-cap promotion. This
module handles reasoning: what to ask, how to parse, and a defensive
escalation to ``replan`` when the model violates the schema.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum
from typing import Any, Dict, List, Optional

from bt_agent._engine.agent.system_prompts import get_prompts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Verdict / result types
# ---------------------------------------------------------------------------


class Verdict(Enum):
    """Three-way Evaluator verdict (paper §5 I7)."""

    PASS = "pass"
    AMEND = "amend"
    REPLAN = "replan"


@dataclass
class EvaluatorResult:
    """Structured Evaluator output."""

    verdict: Verdict
    reasoning: str
    correction: Optional[str] = None  # δ — required iff verdict == AMEND
    # Evidence selection metadata (for observability)
    inspected_handles: List[str] = None  # type: ignore[assignment]
    skipped_inspection: bool = True


# ---------------------------------------------------------------------------
# Schemas for structured output
# ---------------------------------------------------------------------------

_EVIDENCE_SELECTION_SCHEMA = {
    "type": "object",
    "properties": {
        "skip_inspection": {
            "type": "boolean",
            "description": (
                "True if the actor trace and latest observation are "
                "sufficient to judge the milestone — no need to inspect "
                "store objects. False if you need to see specific evidence."
            ),
        },
        "inspect_handles": {
            "type": "array",
            "items": {"type": "string"},
            "description": "Handles from the Known Objects list to inspect. Leave empty if skip_inspection is true.",
        },
        "reasoning": {
            "type": "string",
            "maxLength": 240,
            "description": "One sentence: why you need (or don't need) this evidence.",
        },
    },
    "required": ["skip_inspection"],
}

_VERDICT_SCHEMA = {
    "type": "object",
    "properties": {
        "verdict": {
            "type": "string",
            "enum": ["pass", "amend", "replan"],
            "description": (
                "pass = milestone clearly satisfied; "
                "amend = recoverable mistake, retry with the correction below; "
                "replan = milestone is structurally infeasible, planner must revise."
            ),
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of the verdict.",
        },
        "correction": {
            "type": ["string", "null"],
            "description": (
                "REQUIRED iff verdict == 'amend'. One or two concise sentences "
                "telling the actor what to do differently. Leave null for pass/replan."
            ),
        },
    },
    "required": ["verdict", "reasoning", "correction"],
}

# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------


# User-message templates (_USER_TEMPLATE and _CLAIM_TEMPLATE) are now
# centralized in the system_prompts variant system as
# bundle.evaluator_user and bundle.evaluator_claim. See EVALUATOR_USER and
# EVALUATOR_CLAIM in agent/system_prompts/tiamat_agent_submit_v5/__init__.py.


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def evaluate_completion(
    model: Any,
    milestone: Dict[str, Any],
    skill_observation: str,
    claim_reasoning: str = "",
    actor_trace: str = "",
    store: Any = None,
    store_summaries: str = "",
    scenario: str = "",
    skill_descriptions: str = "",
    event_sink: Any = None,  # OBS-DROP
    prompt_variant: str = "tiamat_agent_submit_v5",
    *,
    evidence_temperature: float = 0.0,
    verdict_temperature: float = 0.0,
    evidence_max_tokens: int = 512,
    verdict_max_tokens: int = 512,
) -> EvaluatorResult:
    """Two-step evaluator: evidence selection from K, then verdict.

    Paper §5: π^eval_θ((d,δ) | m_i, o_t, h̃^act_{m_i}; p^eval).

    Args:
        actor_trace: Compressed action trace h̃^act_{m_i}.
        store: ObjectStore reference for evidence resolution (Step 2).
        store_summaries: LLM-facing summaries of K entries (Step 1).
        scenario: Substrate context (BB.SCENARIO).
        skill_descriptions: Skill name + description list.

    Returns:
        EvaluatorResult with verdict ∈ {PASS, AMEND, REPLAN}.

    Defensive behavior:
        - amend without correction → escalate to replan.
        - any exception → escalate to replan.
    """
    # Lazy import to avoid circular dependency
    from bt_agent._engine.bt.nodes.evaluate_completion import _resolve_evidence

    description = milestone.get("description", "")
    verify_condition = milestone.get("verify_condition", "Is the task complete?")
    interaction_type = milestone.get("interaction_type") or "unspecified"

    # ---- OBS-DROP-BEGIN ----
    if event_sink is not None:
        event_sink.emit(
            "evaluator.start",
            {
                "milestone_id": milestone.get("id", ""),
                "description": description[:160],
                "claim_reasoning": (claim_reasoning or "")[:240],
            },
        )
    # ---- OBS-DROP-END ----

    bundle = get_prompts(prompt_variant)
    claim_section = bundle.evaluator_claim.format(claim_reasoning=claim_reasoning) if claim_reasoning else ""

    user_content = bundle.evaluator_user.format(
        description=description,
        interaction_type=interaction_type,
        verify_condition=verify_condition,
        claim_section=claim_section,
        actor_trace=actor_trace or "No actions recorded yet.",
        observation=skill_observation or "No observation available.",
        scenario=scenario or "No scenario context provided.",
        skill_descriptions=skill_descriptions or "No skill information available.",
        store_summaries=store_summaries or "Store is empty.",
    )

    system_content = bundle.evaluator

    # -- Step 1: Evidence selection -------------------------------------------
    #
    # The VLM sees the full context + K summaries and decides which store
    # entries (if any) to inspect. If skip_inspection=True, we go straight
    # to the verdict (single VLM call).

    evaluation_evidence: Optional[Dict[str, Any]] = None

    try:
        selection = model.query_structured(
            [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            _EVIDENCE_SELECTION_SCHEMA,
            max_tokens=evidence_max_tokens,
            temperature=evidence_temperature,
            phase="evaluator",
        )
    except Exception as e:
        logger.warning("Evidence selection failed (%s), skipping to verdict", e)
        selection = {"skip_inspection": True}

    skip = selection.get("skip_inspection", True)
    handles = selection.get("inspect_handles", []) if not skip else []

    if handles and store is not None:
        evaluation_evidence = _resolve_evidence(store, handles)

    # -- Step 2: Verdict ------------------------------------------------------

    # Build verdict prompt — same context + resolved evidence
    image_list: Optional[List[bytes]] = None
    if evaluation_evidence:
        image_list = evaluation_evidence.get("images")
        extra_text = evaluation_evidence.get("extra_text")
        if extra_text:
            user_content += f"\n\n# Inspected Evidence\n{extra_text}"

    try:
        raw = model.query_structured(
            [
                {"role": "system", "content": system_content},
                {"role": "user", "content": user_content},
            ],
            _VERDICT_SCHEMA,
            images=image_list,
            max_tokens=verdict_max_tokens,
            temperature=verdict_temperature,
            phase="evaluator",
        )
        result = _parse_result(raw)
    except Exception as e:
        logger.warning("Verdict query failed (%s), escalating to REPLAN", e)
        result = EvaluatorResult(
            verdict=Verdict.REPLAN,
            reasoning=f"Evaluator verdict query failed: {e}. Escalating to replan.",
            correction=None,
        )

    # ---- OBS-DROP-BEGIN ----
    if event_sink is not None:
        event_sink.emit(
            "evaluator.done",
            {
                "milestone_id": milestone.get("id", ""),
                "verdict": result.verdict.value,
                "reasoning": result.reasoning[:240],
                "correction": (result.correction or "")[:240],
                "inspected_handles": handles,
                "skipped_inspection": skip,
            },
        )
    # ---- OBS-DROP-END ----

    result.inspected_handles = handles
    result.skipped_inspection = skip
    return result


def _parse_result(raw: Dict[str, Any]) -> EvaluatorResult:
    """Parse a structured-output dict into an EvaluatorResult.

    Defensive: amend without a non-empty correction is promoted to REPLAN.
    """
    verdict_str = str(raw.get("verdict", "")).strip().lower()
    reasoning = str(raw.get("reasoning", "")).strip()
    correction_raw = raw.get("correction")
    correction = str(correction_raw).strip() if correction_raw else ""

    try:
        verdict = Verdict(verdict_str)
    except ValueError:
        logger.warning(
            "Unknown verdict %r from evaluator; treating as REPLAN",
            verdict_str,
        )
        return EvaluatorResult(
            verdict=Verdict.REPLAN,
            reasoning=f"Evaluator returned unknown verdict {verdict_str!r}. Original reasoning: {reasoning}",
            correction=None,
        )

    if verdict == Verdict.AMEND and not correction:
        logger.warning("Evaluator returned AMEND without correction; promoting to REPLAN")
        return EvaluatorResult(
            verdict=Verdict.REPLAN,
            reasoning=f"AMEND verdict without corrective note. Original reasoning: {reasoning}",
            correction=None,
        )

    return EvaluatorResult(
        verdict=verdict,
        reasoning=reasoning,
        correction=correction if verdict == Verdict.AMEND else None,
    )
