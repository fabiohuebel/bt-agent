"""Shared SkillContext builder.

Used by ``ExecuteSkill`` (skill dispatch), ``SelectSkill`` (action-mask
computation), and ``DecisionMakerAgent`` (alt-tier dispatch). Centralizes
the "BB world-model keys → SkillContext" wiring so the three call sites
stay in sync.
"""

from __future__ import annotations

from typing import Any

from bt_agent._engine.agent.blackboard_keys import BB
from bt_agent._engine.skills.protocol import SkillContext


def build_skill_context(blackboard: Any) -> SkillContext:
    """Construct a SkillContext from blackboard world-model reads.

    Missing keys are tolerated (default to None / 0 / "") — the caller
    typically only needs the subset relevant to its concern (e.g. mask
    computation only reads object_store + platform_state).

    ``platform_state`` (paper §5: ``s_robot``) is an opaque dict written
    by the bridge — the framework passes it through without inspecting
    its contents. Skills read what they need via ``.get()``.
    """
    milestones = _bb_get(blackboard, BB.MILESTONES, [])
    idx = _bb_get(blackboard, BB.CURRENT_MILESTONE_IDX, 0)
    milestone = milestones[idx] if idx < len(milestones) else None

    return SkillContext(
        object_store=_bb_get(blackboard, BB.OBJECT_STORE),
        milestone=milestone,
        milestone_index=idx,
        main_task=_bb_get(blackboard, BB.TASK, ""),
        scenario=_bb_get(blackboard, BB.SCENARIO, ""),
        platform_state=_bb_get(blackboard, BB.PLATFORM_STATE, {}),
        skill_call_id=_bb_get(blackboard, BB.SKILL_CALL_ID, ""),
    )


def _bb_get(blackboard: Any, key: str, default: Any = None) -> Any:
    try:
        return blackboard.get(key)
    except KeyError:
        return default
