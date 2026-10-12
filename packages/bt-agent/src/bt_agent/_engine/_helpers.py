"""Shared helpers for text-rung skills.

The text-rung substrate (``tiamat_text``) returns a *typed* result from every
``step``: the ``info`` dict carries ``action_ok`` (did the skill succeed?) and a
``reason`` code (the typed failure channel) — the substrate's parallel to ALFWorld's
flat ``"Nothing happens."`` string. ``step_bridge`` collapses that into
``(observation, ok, reason)`` so each skill node can map it onto a BT tick.
"""

from __future__ import annotations

import re
from typing import Any, List, Optional, Tuple

from bt_agent._engine.skills.protocol import SkillContext


def get_bridge(context: SkillContext) -> Any:
    """Retrieve the TextRungBridge the harness injected into platform_state."""
    bridge = context.platform_state.get("_bridge")
    if bridge is None:
        raise RuntimeError("platform_state['_bridge'] not set — harness must inject it")
    return bridge


def get_admissible(context: SkillContext) -> List[str]:
    """Valid text actions at the current state (ActionSkill precondition source)."""
    return context.platform_state.get("admissible_commands", [])


def step_bridge(context: SkillContext, action: str) -> Tuple[str, bool, Optional[str]]:
    """Drive the bridge one step; return ``(observation, ok, reason_code)``.

    ``ok`` comes from ``info["action_ok"]`` (defaults True when absent, e.g. a
    manually-built env); ``reason`` is the typed failure code or None on success.
    """
    bridge = get_bridge(context)
    obs, _reward, _done, info = bridge.step(action)
    ok = bool(info.get("action_ok", True))
    reason = info.get("reason")
    return obs, ok, reason


def observe_with_reason(obs: str, reason: Optional[str]) -> str:
    """Tag a failure observation with its typed reason code for the trace/LLM."""
    if reason and not obs.startswith("[reason="):
        return f"[reason={reason}] {obs}"
    return obs


_TRAILING_NUM = re.compile(r"^(.*?)_(\d+)$")


def normalize_handle(ref: str) -> str:
    """Best-effort handle normalization.

    Text-rung handles use spaces + a trailing index ("dining chair 1"). The oracle
    and observations already emit that form, but an LLM may hand back an
    underscore-joined variant ("dining_chair_1"); convert it. Space-form handles
    pass through untouched (the env lowercases + resolves case-insensitively).
    """
    ref = ref.strip()
    m = _TRAILING_NUM.match(ref)
    if m:
        return f"{m.group(1).replace('_', ' ')} {m.group(2)}"
    return ref
