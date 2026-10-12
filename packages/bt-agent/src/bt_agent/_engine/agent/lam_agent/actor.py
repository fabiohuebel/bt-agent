"""Actor — select the next skill via LLM tool-calling (one ReAct step).

This is LAM-specific agent intelligence. It builds the conversation context,
calls the model with available tools, and parses the tool call into a
skill selection.

The BT node (SelectSkill) handles lifecycle: threading, blackboard, tick status.
This module handles reasoning: what to ask the model and how to interpret it.

Conversation history
--------------------
The actor maintains multi-turn context across ReAct steps within a single
milestone. Each step produces a (tool_call, observation) pair that the
caller appends to the history via ``build_history_entry()``.

The caller owns the history list (stored on the blackboard as
BB.CONVERSATION). On each call, ``select_skill()`` receives the full
history and handles compaction internally: if the token estimate exceeds
``max_history_tokens``, older entries are summarized while the most recent
steps are kept verbatim as tool-call/result messages.

When a milestone changes, the caller clears the history — the
conversation is scoped to a single milestone.

Skill documentation — progressive disclosure
---------------------------------------------
Brief tool descriptions (name + description + usage + params) are always
present via ``build_tool_definition()``. Full ``.md`` documentation is
available on-demand via the ``read_skill_docs`` pseudo-tool. When the LLM
calls it, the actor loads the doc, appends it to the conversation, and
re-calls the LLM — transparent to the BT node.

Compaction strategies
---------------------
Two strategies, switchable via ``compaction`` parameter for ablation:

- ``"extractive"``: Parse each (tool_call, result) pair into a compact line.
  Free, deterministic, instant. Loses nuance from long observations.
- ``"llm"``: Call ``model.query()`` to produce a natural-language summary.
  Richer context, but costs an extra model call.

Both produce the same interface: older entries become a summary block in the
system prompt, recent entries stay as verbatim messages.
"""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Dict, List, Optional, Tuple

from bt_agent._engine.agent._internals_config import CORRECTION_PREFIX
from bt_agent._engine.agent.system_prompts import get_prompts

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Typed arrival-status → guidance routing (deterministic, no LLM call)
# ---------------------------------------------------------------------------

# One-sentence guidance appended to a skill's tool-result observation, keyed
# by ``arrival_status`` (currently produced by NavigateSkill; framework-level
# so any skill emitting these codes gets the same routing). Pure code lookup
# — the model is never asked to classify the failure.
_ARRIVAL_STATUS_GUIDANCE: Dict[str, str] = {
    "failed_no_path": "No path exists to this goal from the current position — try a different viewpoint or target.",
    "failed_blocked": (
        "The robot got stuck and repeated recovery attempts made no progress — try a different viewpoint or approach."
    ),
    "failed_no_movement": (
        "The robot never started moving toward the goal — the local planner or robot may be unresponsive."
    ),
    "arrived_short": "The robot settled short of the arrival distance — verify position before relying on it.",
    # Non-finite goal coordinates (``_ValidateGoal``). The goal itself is
    # bad, not the route — recovery is re-QUERY, not retry.
    "failed_invalid": (
        "The goal position for this target was invalid — query the target again or pick a different viewpoint."
    ),
}


def route_arrival_guidance(arrival_status: Optional[str]) -> str:
    """Deterministic code → one-sentence guidance lookup (no LLM involved).

    Returns ``""`` for unmapped/empty codes (e.g. ``"arrived"``,
    ``"already_at_goal"``, or non-navigate skills without this status).
    """
    return _ARRIVAL_STATUS_GUIDANCE.get(arrival_status or "", "")


# ---------------------------------------------------------------------------
# Typed press_status → guidance routing (skills_contract_v0 press section, D8)
# ---------------------------------------------------------------------------

# PressSkill's own outcome key. Its failure codes name the same physical
# failures navigate reports via ``arrival_status``, so the sentences are
# *shared by reference* rather than restated — the agent must not learn two
# different recovery stories for one failure just because a different skill
# reported it. ``pressed`` is deliberately unmapped (success needs no hint).
_PRESS_STATUS_GUIDANCE: Dict[str, str] = {
    code: _ARRIVAL_STATUS_GUIDANCE[code]
    for code in ("failed_no_path", "failed_blocked", "failed_no_movement", "failed_invalid")
}


def route_press_status_guidance(press_status: Optional[str]) -> str:
    """Deterministic code → one-sentence guidance lookup (no LLM involved).

    Returns ``""`` for unmapped/empty codes (e.g. ``"pressed"``, or a skill
    that doesn't emit this status at all).
    """
    return _PRESS_STATUS_GUIDANCE.get(press_status or "", "")


# ---------------------------------------------------------------------------
# Typed wrap-level failure_reason → guidance routing (bt-nav-fixes CP5)
# ---------------------------------------------------------------------------

# One-sentence guidance appended to a skill's tool-result observation, keyed
# by ``failure_reason`` — the generic, framework-level termination code any
# ActionSkill's safeguard wrap (CancellationGuard / WallTimeBudgetGuard /
# InvariantGuard) may set on ``handle.result_data``, distinct from a skill's
# own outcome status (e.g. ``arrival_status``). Pure code lookup, no LLM call.
_FAILURE_REASON_GUIDANCE: Dict[str, str] = {
    "cancelled": "The skill was cancelled before it finished — no error, just preempted.",
    "wall_time_exceeded": (
        "The skill ran out of its time budget before finishing — the route or action may be "
        "longer than expected; consider a closer intermediate target or retrying."
    ),
    "stale_pose": "Localization was too old to trust — wait for fresh odometry before retrying.",
    "pose_lost": "Robot localization was unavailable — check perception/odometry before retrying.",
    "path_invalidated": "The planned path became invalid mid-execution — retry to replan, or try a different target.",
}


def route_failure_reason_guidance(failure_reason: Optional[str]) -> str:
    """Deterministic code → one-sentence guidance lookup (no LLM involved).

    Returns ``""`` for unmapped/empty codes (e.g. a skill that terminated
    without a typed wrap-level failure_reason).
    """
    return _FAILURE_REASON_GUIDANCE.get(failure_reason or "", "")


# ---------------------------------------------------------------------------
# Symmetric action-budget disclosure (tiamat_world docs/spec/budget-disclosure.md, CP4)
# ---------------------------------------------------------------------------


def render_action_budget_disclosure(action_budget: Optional[Dict[str, Any]]) -> str:
    """Render the two symmetric-disclosure lines for the actor system prompt.

    ``action_budget`` (when configured) carries the SAME counter
    ``board_runner.drive_episode`` checks against
    ``EpisodeBudget.skill_call_cap`` -- ``bridge.step_count`` -- never a
    parallel count that could drift from the one the cap enforces. Keys:
    ``skill_call_cap`` (N), ``wall_time_cap_s`` (W), ``actions_used`` (K).

    ``None`` (no action budget configured for this episode -- disclosure off
    or no budget rule at all) renders ``""``, so the actor system prompt is
    byte-identical to before this feature existed.

    Both lines render on every actor turn, regardless of ``actions_used``.
    Flat arms keep the episode-start line in their message history for the
    whole episode; this agent rebuilds its context from scratch every turn
    (see module docstring), so persistent per-turn rendering is the
    symmetric equivalent -- an ``actions_used == 0`` gate would silently
    drop the wall-limit line whenever the PlanTask gather loop (QuerySkill
    calls during planning) has already spent budget before the actor's
    first turn. Wording is mode-neutral (never mentions "submit"): correct
    verbatim under MODE 1 and MODE 2 alike, per the design ruling.
    """
    if not action_budget:
        return ""
    cap = action_budget["skill_call_cap"]
    wall_cap_s = action_budget["wall_time_cap_s"]
    used = action_budget["actions_used"]
    lines = [
        f"Action budget: {cap} skill calls; wall limit: {wall_cap_s:.0f} s.",
        f"Actions used: {used} of {cap}.",
    ]
    return "\n\n# Action budget\n" + "\n".join(lines)


def _history_has_amend_correction(history: List[Dict[str, Any]]) -> bool:
    """True if the most recent user message in ``history`` is an evaluator
    correction injected by ApplyAmend. Used to raise sampling temperature
    so the actor doesn't re-emit the action that triggered amend.

    Walks from the tail and stops at the first user message — if that
    message starts with CORRECTION_PREFIX, returns True. Older
    corrections (from prior amends, now superseded) deliberately don't
    count — only the most recent one is the active signal.
    """
    for msg in reversed(history):
        if msg.get("role") != "user":
            continue
        content = msg.get("content", "")
        if isinstance(content, str) and content.startswith(CORRECTION_PREFIX):
            return True
        return False
    return False


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_TOKEN_BUDGET = 24000
_KEEP_RECENT_STEPS = 5  # (assistant tool_call + tool result) pairs to keep
_MAX_DOC_READS = 3  # cap read_skill_docs inner loop iterations

# Actor system prompt lives under ``bt_agent._engine.agent.system_prompts.<variant>/``
# (Step 5). Resolved per-call via ``get_prompts(prompt_variant)``.

# ---------------------------------------------------------------------------
# read_skill_docs tool definition lifted to bt_agent._engine.agent.role in Step 6
# (the SelectSkill BT node calls Role.ACTOR.available_tools(...) and
# passes the resulting list — including read_skill_docs — into select_skill).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def select_skill(
    model: Any,
    tools: List[Dict[str, Any]],
    *,
    milestone: Dict[str, Any],
    milestone_index: int,
    total_milestones: int,
    object_summaries: str = "",
    history: Optional[List[Dict[str, Any]]] = None,
    injected_images: Optional[List[bytes]] = None,
    compaction: str = "extractive",
    max_history_tokens: int = _DEFAULT_TOKEN_BUDGET,
    event_sink: Any = None,  # OBS-DROP: visualizer pubsub broker, may be None
    prompt_variant: str = "tiamat_agent_submit_v5",
    scenario: str = "",
    reasoning_mode: str = "react_single",
    temperature: float = 0.0,
    amend_temperature: float = 0.7,
    action_budget: Optional[Dict[str, Any]] = None,
) -> Tuple[str, str, Dict[str, Any], str, str, Dict[str, Any]]:
    """Run one ReAct step: call LLM with tools, get skill selection.

    Dispatches per-turn mechanics through :func:`select_action`. Four
    reasoning modes are supported:

    - ``"act"`` — pure tool calling, no reasoning captured.
    - ``"act_thinking"`` — single call, reasoning from ``<think>`` tokens.
    - ``"react_single"`` (default) — single call, reasoning from
      ``message.content``. Atomic thought + action. Re-prompts once
      if content or tool_call missing.
    - ``"react_two_step"`` — legacy two-call Reason→Act via
      ``reason_then_act``.

    May call the LLM multiple times if ``read_skill_docs`` is invoked
    (inner loop). Transparent to the caller — always returns a single
    skill selection.

    Args:
        model: ModelBackend instance.
        tools: Available skill tool definitions (OpenAI format).
        milestone: Current milestone dict with description,
            detailed_instructions, verify_condition.
        milestone_index: Zero-based index of the current milestone.
        total_milestones: Total number of milestones in the plan.
        object_summaries: Text summaries of known objects from ObjectStore.
        history: Conversation history — list of OpenAI-format messages.
        injected_images: Image bytes from the previous skill to include
            as vision input in this model call.
        compaction: ``"extractive"`` or ``"llm"`` compaction strategy.
        max_history_tokens: Approximate token budget for history.
        reasoning_mode: One of ``"act" | "act_thinking" | "react_single" |
            "react_two_step"``. See :func:`select_action`.
        action_budget: Symmetric action-budget disclosure payload (``None``
            = not configured, disclosure off). See
            :func:`render_action_budget_disclosure`.

    Returns:
        (decision, skill_name, params, reasoning, memory_note, raw_response)
        where decision is ``"CONTINUE"``, ``"TASK_COMPLETE"``, or ``"DEFER"``.
    """
    history = list(history or [])
    history_len_before = len(history)  # OBS-DROP

    # Build system prompt — variant resolved per-call (Step 5)
    bundle = get_prompts(prompt_variant)

    # Amend-aware temperature: when an evaluator correction was injected
    # into history (by ApplyAmend), the actor's next decision should
    # explore a different response instead of re-emitting the action
    # that triggered the amend. Detection by CORRECTION_PREFIX is
    # stable framework output from route_verdict.ApplyAmend.
    effective_temperature = amend_temperature if _history_has_amend_correction(history) else temperature

    # Split history into (summary_text, recent_messages) if over budget
    history_summary, recent = _split_history(
        history,
        max_history_tokens,
        compaction,
        model,
        compaction_prompt=bundle.actor_compaction,
    )

    # ---- OBS-DROP-BEGIN ----
    if event_sink is not None:
        event_sink.emit(
            "actor.iter.start",
            {
                "milestone_idx": milestone_index,
                "total_milestones": total_milestones,
                "mask_size": len(tools),
                "history_messages": history_len_before,
            },
        )
        if history_summary:
            event_sink.emit(
                "actor.history_compacted",
                {
                    "strategy": compaction,
                    "summarized_messages": history_len_before - len(recent),
                    "kept_recent_messages": len(recent),
                },
            )
    # ---- OBS-DROP-END ----
    system_content = bundle.actor.format(
        milestone_idx=milestone_index + 1,
        total_milestones=total_milestones,
        milestone_description=milestone.get("description", ""),
        milestone_instructions=milestone.get("detailed_instructions", ""),
        milestone_verify=milestone.get("verify_condition", ""),
        object_summaries=object_summaries or "None yet — use QUERY to find objects.",
        history_summary=(f"\n# Previous Actions (summarized)\n{history_summary}\n" if history_summary else ""),
        scenario=scenario or "No scenario context provided.",
        budget_disclosure=render_action_budget_disclosure(action_budget),
    )

    # Build message list
    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_content},
    ]

    if recent:
        messages.append({
            "role": "user",
            "content": "Continue executing the milestone. Select the next action.",
        })
        messages.extend(recent)
    else:
        messages.append({
            "role": "user",
            "content": "Begin this milestone. Select the first action.",
        })

    # Step 6: ``tools`` already contains read_skill_docs (appended by
    # Role.ACTOR.available_tools() upstream in SelectSkill). The actor is
    # a pure consumer of the role's tool list — no per-call augmentation.

    # Inner loop: handle read_skill_docs calls, break on real skill.
    # Each iteration delegates per-turn ReAct mechanics to select_action,
    # which dispatches on reasoning_mode.
    images = list(injected_images) if injected_images else None
    reasoning_text = ""

    for _ in range(_MAX_DOC_READS + 1):
        reasoning_text, tool_name, tool_args, raw_response = select_action(
            model,
            messages,
            tools,
            mode=reasoning_mode,
            images=images,
            temperature=effective_temperature,
        )

        # Only inject images on the first call
        images = None

        if tool_name == "read_skill_docs":
            skill_name = tool_args.get("skill_name", "")
            logger.info("Actor reading docs for: %s", skill_name)

            from bt_agent._engine.skills import load_skill_doc

            doc_content = load_skill_doc(skill_name)

            # ---- OBS-DROP-BEGIN ----
            if event_sink is not None:
                event_sink.emit(
                    "actor.read_docs",
                    {
                        "milestone_idx": milestone_index,
                        "skill_name": skill_name,
                        "doc_chars": len(doc_content or ""),
                    },
                )
            # ---- OBS-DROP-END ----

            # Append tool_call + result to messages so LLM sees the docs
            doc_entry = build_history_entry(
                "read_skill_docs",
                tool_args,
                doc_content,
                reasoning=reasoning_text,
            )
            messages.extend(doc_entry)
            continue

        if tool_name == "claim_complete":
            # ---- OBS-DROP-BEGIN ----
            if event_sink is not None:
                event_sink.emit(
                    "actor.decision",
                    {
                        "kind": "task_complete",
                        "milestone_idx": milestone_index,
                        "summary": reasoning_text[:240],
                    },
                )
            # ---- OBS-DROP-END ----
            return ("TASK_COMPLETE", "", {}, reasoning_text, "", raw_response)

        # Real skill or special decision — break out
        break

    # ---- OBS-DROP-BEGIN ----
    if event_sink is not None:
        decision_kind = "defer" if tool_name == "defer" else "skill"
        event_sink.emit(
            "actor.decision",
            {
                "kind": decision_kind,
                "milestone_idx": milestone_index,
                "skill": tool_name if decision_kind == "skill" else "",
                "params": {k: v for k, v in (tool_args or {}).items() if k != "memory_note"},
                "reasoning": reasoning_text[:240],
            },
        )
    # ---- OBS-DROP-END ----

    # Parse the final tool call
    return _parse_tool_call(tool_name, tool_args, raw_response, reasoning=reasoning_text)


# ---------------------------------------------------------------------------
# History management (public helpers for the BT node)
# ---------------------------------------------------------------------------


def history_call_id(tool_name: str, observation: str) -> str:
    """Return the legacy deterministic-within-process history call ID."""
    return f"call_{abs(hash((tool_name, observation))) % 100000:05d}"


def build_history_entry(
    tool_name: str,
    tool_args: Dict[str, Any],
    observation: str,
    call_id: str = "",
    reasoning: str = "",
) -> List[Dict[str, Any]]:
    """Build the OpenAI-format messages for one completed ReAct step.

    Returns two messages — the assistant's tool_call and the tool's
    result — to append to the history list between ``select_skill()``
    calls.

    The assistant message carries the reasoning text in ``content``
    (from the Reason step) alongside the ``tool_calls`` list (from the
    Act step).  This is valid OpenAI chat format and lets the model see
    its prior deliberation when it reads conversation history.

    Args:
        tool_name: Skill name that was called (e.g., ``"query"``).
        tool_args: Skill-specific arguments (no reasoning — that is
            passed separately).
        observation: The skill's observation text after execution.
        call_id: Tool call ID. Auto-generated if not provided.
        reasoning: Free-text reasoning from the Reason step (two-step
            ReAct) or from model thinking tokens (reasoning models).
    """
    if not call_id:
        call_id = history_call_id(tool_name, observation)

    return [
        {
            "role": "assistant",
            "content": reasoning or None,
            "tool_calls": [
                {
                    "id": call_id,
                    "type": "function",
                    "function": {
                        "name": tool_name,
                        "arguments": json.dumps(tool_args),
                    },
                },
            ],
        },
        {
            "role": "tool",
            "content": observation,
            "tool_call_id": call_id,
        },
    ]


def compact_history(
    history: List[Dict[str, Any]],
    model: Any = None,
    compaction: str = "extractive",
    max_tokens: int = _DEFAULT_TOKEN_BUDGET,
    keep_recent: int = _KEEP_RECENT_STEPS,
) -> List[Dict[str, Any]]:
    """Compact old history entries, return a shorter list.

    Call this periodically on the stored history to prevent unbounded
    growth. Old (tool_call, result) pairs are replaced by a single
    user message summarizing them. Recent pairs are kept verbatim.

    ``select_skill()`` also handles over-long history internally, so
    calling this is optional. Use it to keep memory usage bounded for
    very long tasks (50+ skill calls within one milestone).

    Args:
        history: The full conversation history (list of OpenAI messages).
        model: ModelBackend instance. Required when ``compaction="llm"``.
        compaction: ``"extractive"`` or ``"llm"``.
        max_tokens: Token budget threshold.
        keep_recent: Number of recent (tool_call, result) pairs to keep.

    Returns:
        A new (shorter) list of messages.
    """
    token_est = _estimate_tokens(history)
    keep_msgs = keep_recent * 2

    if token_est <= max_tokens or len(history) <= keep_msgs:
        return list(history)

    old = history[:-keep_msgs]
    recent = history[-keep_msgs:]

    if compaction == "llm" and model is not None:
        summary = _compact_llm(old, model)
    else:
        summary = _compact_extractive(old)

    if not summary:
        return list(recent)

    return [
        {"role": "user", "content": f"Previous actions summary:\n{summary}"},
        *recent,
    ]


# ---------------------------------------------------------------------------
# Two-step Reason→Act helpers
# ---------------------------------------------------------------------------

_THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL)
_TOOL_CALL_RE = re.compile(r"<tool_call>.*?</tool_call>", re.DOTALL)
_TOOL_CALL_OPEN_RE = re.compile(r"<tool_call>.*$", re.DOTALL)


def reason_then_act(
    model: Any,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    *,
    images: Optional[List[bytes]] = None,
) -> Tuple[str, str, Dict[str, Any], Dict[str, Any]]:
    """Two-step Reason→Act for non-reasoning models.

    1. **Reason** — ``model.query()`` (no tools) produces free-text
       deliberation.
    2. **Act** — ``model.query_tool_call()`` selects the tool. The
       reasoning text is in the KV prefix so the tool choice is informed
       by it.

    The three ephemeral messages (reason prompt, reasoning, act prompt)
    are appended to *messages* for the duration of the two calls and
    then removed — the caller stores the result via
    ``build_history_entry()`` in proper history format.

    Returns:
        (reasoning_text, tool_name, tool_args, raw_response)
    """
    # ── Step 1: REASON ──
    reason_prompt = {
        "role": "user",
        "content": "Think step-by-step about what to do next. Do not select a tool yet.",
    }
    messages.append(reason_prompt)
    reasoning_text = model.query(messages, images=images, max_tokens=512, phase="actor")

    # Strip any <tool_call> XML the model leaked into the thought step.
    # Qwen3 (and similar) emits tool_call tags reflexively once it knows
    # what to do, even when instructed not to. Leaving the XML in the
    # assistant.content for step 2 confuses the model: it sees a
    # pre-committed tool call in its own prior turn and tends to emit a
    # DIFFERENT tool in step 2 rather than re-emit the leaked one — which
    # produces a thought/action mismatch.
    reasoning_text = _TOOL_CALL_RE.sub("", reasoning_text)
    reasoning_text = _TOOL_CALL_OPEN_RE.sub("", reasoning_text)
    reasoning_text = reasoning_text.strip()

    # Append reasoning as assistant content (KV prefix for step 2)
    messages.append({"role": "assistant", "content": reasoning_text})

    # ── Step 2: ACT ──
    act_prompt = {
        "role": "user",
        "content": "Now select exactly one tool.",
    }
    messages.append(act_prompt)
    tool_name, tool_args, raw_response = model.query_tool_call(
        messages,
        tools,
        tool_choice="required",
        phase="actor",
    )

    # Pop the 3 injected messages — they'll be re-added in structured
    # form via build_history_entry (reasoning in content, tool in
    # tool_calls on the same assistant message).
    del messages[-3:]

    return reasoning_text, tool_name, tool_args, raw_response


def extract_thinking(raw_response: Dict[str, Any]) -> str:
    """Extract reasoning from a reasoning model's ``<think>`` tokens.

    vLLM returns thinking content either in a dedicated
    ``reasoning_content`` field or inline as ``<think>...</think>``
    tags in ``message.content``.  This helper normalises both.

    Returns an empty string when no thinking content is found.
    """
    message = raw_response.get("choices", [{}])[0].get("message", {})

    # Dedicated field (vLLM ≥ 0.8 with enable_thinking)
    reasoning = message.get("reasoning_content") or ""
    if reasoning:
        return reasoning.strip()

    # Inline <think> tags in content
    content = message.get("content") or ""
    m = _THINK_RE.search(content)
    if m:
        return m.group(1).strip()

    return ""


# ---------------------------------------------------------------------------
# Unified reasoning-mode dispatcher
# ---------------------------------------------------------------------------

# The four supported per-turn modes:
#   "act"            — pure tool calling, no reasoning captured.
#   "act_thinking"   — single call, reasoning from <think> tokens
#                      (reasoning models).
#   "react_single"   — single call, reasoning from message.content
#                      (any model). Default. ReAct atomic.
#   "react_two_step" — two calls (model.query() for thought, then
#                      query_tool_call for action). Legacy.
_VALID_REASONING_MODES = frozenset({"act", "act_thinking", "react_single", "react_two_step"})

# Re-prompt text reused across modes. Phrased lowercase since prompts now
# carry the THOUGHT/tool convention; these are recovery nudges.
_REPROMPT_FOR_TOOL = "You must select a tool. Call one now."
_REPROMPT_FOR_THOUGHT = (
    "You skipped the THOUGHT. Before calling a tool, emit a brief free-text "
    "thought describing what you observed and why this action follows. "
    "Re-emit your turn now: thought first (as message content), then the "
    "tool call."
)


def _extract_content_stripped(raw_response: Dict[str, Any]) -> str:
    """Pull message.content from a raw response and strip <tool_call> XML.

    Used by ``react_single`` mode: the model emits a free-text THOUGHT plus
    a structured tool_call. The thought is the content; any inline
    <tool_call> XML (vLLM parser variation can leave it in content even
    after parsing it into tool_calls) is stripped defensively.
    """
    message = raw_response.get("choices", [{}])[0].get("message", {})
    content = (message.get("content") or "").strip()
    content = _TOOL_CALL_RE.sub("", content)
    return _TOOL_CALL_OPEN_RE.sub("", content).strip()


def select_action(
    model: Any,
    messages: List[Dict[str, Any]],
    tools: List[Dict[str, Any]],
    *,
    mode: str,
    images: Optional[List[bytes]] = None,
    temperature: float = 0.0,
    max_tokens: int = 1024,
) -> Tuple[str, Optional[str], Dict[str, Any], Dict[str, Any]]:
    """Make one ReAct decision in the specified mode.

    Returns ``(reasoning_text, tool_name, tool_args, raw_response)``.
    ``tool_name`` may be ``None`` if the model could not be coerced to
    emit one even after the per-mode re-prompt fallback.

    Mutates ``messages`` only for the duration of re-prompt fallbacks —
    any scaffolding appended is popped before return. Net change to
    ``messages`` is zero.

    Raises:
        ValueError: if ``mode`` is not one of the four supported values.
    """
    if mode not in _VALID_REASONING_MODES:
        raise ValueError(f"Unknown reasoning_mode: {mode!r}. Expected one of: {sorted(_VALID_REASONING_MODES)}.")

    if mode == "react_two_step":
        # Legacy two-call path with <tool_call> strip on the first call.
        return reason_then_act(model, messages, tools, images=images)

    if mode == "act":
        # Pure tool calling. tool_choice="required" *should* guarantee a
        # tool, but vLLM + Qwen3 has been observed unreliable — keep a
        # defensive re-prompt fallback.
        name, args, raw = model.query_tool_call(
            messages,
            tools,
            images=images,
            tool_choice="required",
            temperature=temperature,
            max_tokens=max_tokens,
            phase="actor",
        )
        if name is None:
            messages.append({"role": "user", "content": _REPROMPT_FOR_TOOL})
            name, args, raw = model.query_tool_call(
                messages,
                tools,
                images=images,
                tool_choice="required",
                temperature=temperature,
                max_tokens=max_tokens,
                phase="actor",
            )
            del messages[-1]
        return "", name, args, raw

    # Single-call paths (act_thinking and react_single). They differ only
    # in where reasoning_text comes from and whether content absence is
    # treated as a faithfulness violation worth re-prompting.
    name, args, raw = model.query_tool_call(
        messages,
        tools,
        images=images,
        tool_choice="auto",
        temperature=temperature,
        max_tokens=max_tokens,
        phase="actor",
    )

    if mode == "act_thinking":
        reasoning_text = extract_thinking(raw)
    else:  # react_single
        reasoning_text = _extract_content_stripped(raw)

    # Fallback A: model emitted text only, no tool_call. Re-prompt once.
    if name is None:
        text = raw.get("choices", [{}])[0].get("message", {}).get("content", "") or ""
        if text:
            messages.append({"role": "assistant", "content": text})
        messages.append({"role": "user", "content": _REPROMPT_FOR_TOOL})
        name, args, raw = model.query_tool_call(
            messages,
            tools,
            images=images,
            tool_choice="auto",
            temperature=temperature,
            max_tokens=max_tokens,
            phase="actor",
        )
        # If reasoning wasn't captured the first time, try again from the
        # retry's response.
        if not reasoning_text:
            if mode == "act_thinking":
                reasoning_text = extract_thinking(raw)
            else:
                reasoning_text = _extract_content_stripped(raw)
        # Pop the scaffolding so it doesn't persist in conversation history.
        if text:
            del messages[-2:]
        else:
            del messages[-1:]

    # Fallback B (react_single only): tool emitted but thought skipped —
    # ReAct faithfulness violation. Re-prompt once for thought-first.
    elif mode == "react_single" and name is not None and not reasoning_text:
        messages.append({"role": "user", "content": _REPROMPT_FOR_THOUGHT})
        name, args, raw = model.query_tool_call(
            messages,
            tools,
            images=images,
            tool_choice="auto",
            temperature=temperature,
            max_tokens=max_tokens,
            phase="actor",
        )
        retry_content = _extract_content_stripped(raw)
        if retry_content:
            reasoning_text = retry_content
        del messages[-1]

    return reasoning_text, name, args, raw


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------


def _parse_tool_call(
    tool_name: str,
    tool_args: Dict[str, Any],
    raw_response: Dict[str, Any],
    reasoning: str = "",
) -> Tuple[str, str, Dict[str, Any], str, str, Dict[str, Any]]:
    """Parse a tool call into the standard 6-tuple return.

    Args:
        reasoning: Deliberation text from the Reason step (two-step
            ReAct) or from model thinking tokens (reasoning models).
            Passed through as-is in the return tuple.
    """
    memory_note = tool_args.get("memory_note", "")

    if tool_name == "defer":
        return ("DEFER", "", {}, reasoning, memory_note, raw_response)

    # Normal skill — strip memory_note, keep only skill params
    skill_params = {k: v for k, v in tool_args.items() if k != "memory_note"}
    return ("CONTINUE", tool_name, skill_params, reasoning, memory_note, raw_response)


def _split_history(
    history: List[Dict[str, Any]],
    max_tokens: int,
    compaction: str,
    model: Any,
    keep_recent: int = _KEEP_RECENT_STEPS,
    compaction_prompt: str = "",
) -> Tuple[str, List[Dict[str, Any]]]:
    """Split history into (summary_text, recent_messages).

    If history fits in the token budget, returns ``("", history)``.
    Otherwise, old entries are summarized into text (for the system
    prompt) and only the most recent steps are kept as messages.
    """
    token_est = _estimate_tokens(history)
    keep_msgs = keep_recent * 2

    if token_est <= max_tokens or len(history) <= keep_msgs:
        return "", history

    old = history[:-keep_msgs]
    recent = history[-keep_msgs:]

    if compaction == "llm" and model is not None:
        summary = _compact_llm(old, model, compaction_prompt)
    else:
        summary = _compact_extractive(old)

    return summary, recent


def _compact_extractive(entries: List[Dict[str, Any]]) -> str:
    """Extract key info from old history messages into a compact summary.

    Each (assistant tool_call, tool result) pair becomes one line::

        1. query(target="red mug") -> Found 'red mug'. Handle: red_mug_1
        2. surround_survey(target="is it visible?") -> Target found in frontleft.
    """
    lines: List[str] = []
    step = 1
    i = 0

    while i < len(entries):
        msg = entries[i]

        # Pass through previously compacted summaries
        if msg.get("role") == "user":
            content = msg.get("content", "")
            if content:
                lines.append(content)
            i += 1
            continue

        # Expect (assistant, tool) pair
        if msg.get("role") != "assistant" or i + 1 >= len(entries):
            i += 1
            continue

        assistant_msg = entries[i]
        tool_msg = entries[i + 1]
        i += 2

        tool_calls = assistant_msg.get("tool_calls", [])
        if not tool_calls:
            continue

        name = tool_calls[0]["function"]["name"]

        # Skip read_skill_docs entries — they're meta, not progress
        if name == "read_skill_docs":
            continue

        try:
            args = json.loads(tool_calls[0]["function"]["arguments"])
            args.pop("memory_note", None)
        except (json.JSONDecodeError, KeyError):
            args = {}

        obs = tool_msg.get("content", "")
        if len(obs) > 200:
            obs = obs[:197] + "..."

        args_str = ", ".join(f'{k}="{v}"' for k, v in args.items())
        lines.append(f"{step}. {name}({args_str}) -> {obs}")
        step += 1

    return "\n".join(lines)


def _compact_llm(entries: List[Dict[str, Any]], model: Any, prompt_template: str = "") -> str:
    """Summarize old history entries using an LLM call.

    Produces a richer summary than extractive compaction by asking the
    model to identify key findings, progress, and state. Costs one
    extra model call (~500 tokens output).
    """
    # First build extractive text as input to the LLM
    actions_text = _compact_extractive(entries)
    if not actions_text:
        return ""

    messages = [
        {
            "role": "user",
            "content": prompt_template.format(actions_text=actions_text),
        },
    ]

    try:
        return model.query(messages, max_tokens=512, temperature=0.0, phase="actor")
    except Exception as e:
        logger.warning("LLM compaction failed, falling back to extractive: %s", e)
        return actions_text


def _estimate_tokens(messages: List[Dict[str, Any]]) -> int:
    """Rough token estimate: ~4 characters per token."""
    total_chars = 0
    for msg in messages:
        content = msg.get("content") or ""
        total_chars += len(content)
        for tc in msg.get("tool_calls", []):
            fn = tc.get("function", {})
            total_chars += len(fn.get("arguments", ""))
            total_chars += len(fn.get("name", ""))
    return total_chars // 4
