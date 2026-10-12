"""Conversation utilities — extract tool calls from h^act_{m_i}.

Shared between SelectSkill (for ``is_available()`` history) and
EvaluateCompletion (for the compressed actor trace in the evaluator
prompt).  Single extraction, two consumers.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Dict, List


@dataclass
class ToolCall:
    """A single tool call extracted from the actor conversation."""

    name: str
    args: Dict[str, Any] = field(default_factory=dict)
    observation: str = ""


def extract_tool_calls(conversation: List[Dict[str, Any]]) -> List[ToolCall]:
    """Walk BB.CONVERSATION, pair assistant tool_calls with tool responses.

    Returns a flat list of ToolCall records.  Substrate-agnostic: the
    framework reads tool names, args, and observations without
    interpreting them.
    """
    calls: List[ToolCall] = []
    # Map tool_call_id → (name, args) from assistant messages
    pending: Dict[str, tuple] = {}

    for msg in conversation:
        role = msg.get("role", "")
        if role == "assistant":
            # Reasoning lives in the assistant message's ``content``
            # field (from the Reason step or model thinking tokens),
            # NOT inside tool_args.
            content = msg.get("content") or ""
            for tc in msg.get("tool_calls", []):
                fn = tc.get("function", {})
                tc_id = tc.get("id", "")
                name = fn.get("name", "")
                if not tc_id or not name:
                    continue
                try:
                    args = json.loads(fn.get("arguments", "{}"))
                except (json.JSONDecodeError, TypeError):
                    args = {}
                args = {k: v for k, v in args.items() if k != "memory_note"}
                # Attach reasoning from assistant content for downstream
                # consumers (evaluator prompt, trace display).
                if content:
                    args["reasoning"] = content
                pending[tc_id] = (name, args)
        elif role == "tool":
            tc_id = msg.get("tool_call_id", "")
            entry = pending.pop(tc_id, None)
            if entry is not None:
                name, args = entry
                calls.append(
                    ToolCall(
                        name=name,
                        args=args,
                        observation=msg.get("content", ""),
                    )
                )

    return calls


def format_for_prompt(calls: List[ToolCall]) -> str:
    """Format tool calls as a numbered summary for the evaluator prompt.

    Reasoning (the actor's chain of thought) is shown as a bracketed
    note — it captures visual descriptions after image injection and
    decision rationale for the evaluator.

    Example output::

        1. image_inject(camera='frontleft') → 'Injected frontleft camera.'
        2. navigate(target_object='red_mug_1') [I see the red mug on the counter] → 'Arrived.'
    """
    if not calls:
        return "No actions taken."
    lines: List[str] = []
    for i, tc in enumerate(calls, 1):
        # Separate reasoning from skill params for display
        display_args = {k: v for k, v in tc.args.items() if k != "reasoning"}
        reasoning = tc.args.get("reasoning", "")
        args_str = ", ".join(f"{k}={v!r}" for k, v in display_args.items())
        reasoning_note = f" [{reasoning[:150]}]" if reasoning else ""
        # Raised 200 -> 800: information-bearing skills (query / surround_survey)
        # return recall-lists that a 200-char cut truncated mid-list, causing
        # the evaluator to false-flag "hallucination" and amend. A per-skill cap
        # would need to hardcode substrate skill names here, but this module is
        # substrate-agnostic by contract (see module docstring), so the cap is
        # raised uniformly; navigate/motion observations are short and unaffected.
        obs = tc.observation[:800] if tc.observation else ""
        lines.append(f"{i}. {tc.name}({args_str}){reasoning_note} → {obs!r}")
    return "\n".join(lines)
