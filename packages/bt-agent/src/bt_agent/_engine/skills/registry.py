"""SkillRegistry — name-to-skill mapping with gating analysis.

Skills are partitioned by ``EffectType`` (paper §5, Invariant I2):

* ``CONTEXT`` (Σ_ctx): read-only context-gathering tools — always available.
* ``ACTION`` (Σ_act): effectful skill-BTs — gated by ``is_available()``
  (interaction-history) AND ``precondition(state)`` (current world state)
  for the full declarative action mask
  ``Σ^avail_t = Σ_ctx ∪ {σ ∈ Σ_act : is_available(history) ∧ precondition(state)}``.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from bt_agent._engine.skills.protocol import EffectType, Skill, SkillContext


class SkillRegistry:
    """Central registry of all available skills.

    Skills are registered at startup:

    * Instant skills by the agent itself (no ROS2 deps).
    * Physical skills by the bridge via ``agent.register_skill()``.

    Internally partitioned by effect type. Two role-aware accessors:

    * ``get_context_tools()`` — Σ_ctx tool defs, no filtering.
    * ``get_action_skills_masked(state, history, context)`` — Σ_act tool defs
      whose ``precondition(state)`` AND ``is_available(history, context)``
      both hold (the paper-aligned action mask for Σ_act, Invariant I3).

    """

    def __init__(self) -> None:
        self._ctx_skills: Dict[str, Skill] = {}
        self._act_skills: Dict[str, Skill] = {}

    # -- Registration / lookup ----------------------------------------------

    def register(self, skill: Skill) -> None:
        """Register a skill, routing it to the Σ_ctx or Σ_act bucket by ``effect_type``."""
        if skill.effect_type == EffectType.CONTEXT:
            self._ctx_skills[skill.name] = skill
        elif skill.effect_type == EffectType.ACTION:
            self._act_skills[skill.name] = skill
        else:
            raise TypeError(
                f"Skill {skill.name!r} has invalid effect_type {skill.effect_type!r}; "
                "expected EffectType.CONTEXT or EffectType.ACTION."
            )

    def get(self, name: str) -> Optional[Skill]:
        """Look up a skill by name across both buckets."""
        return self._ctx_skills.get(name) or self._act_skills.get(name)

    def all_skills(self) -> Dict[str, Skill]:
        """Return all registered skills (union of both buckets)."""
        return {**self._ctx_skills, **self._act_skills}

    # -- Role-aware accessors (paper §5, Invariant I3) ----------------------

    def context_skills(self) -> List[Skill]:
        """Return all registered ContextSkill instances (Σ_ctx).

        The planner's gather loop iterates these to build its tool list and
        to dispatch tool calls back to the originating skill via
        ``context_skill_by_tool_name()``. Substrate-agnostic: the registry
        does not know what each skill is named — that's the skill's own
        ``name`` attribute.
        """
        return list(self._ctx_skills.values())

    def context_skill_by_tool_name(self, tool_name: str) -> Optional[Skill]:
        """Look up a ContextSkill by the ``name`` it exposes in its ``tool_def()``.

        Used by the planner's generic gather-loop dispatcher: when the LLM
        calls a non-meta tool, this resolves the call back to the concrete
        skill so ``skill.start(...)`` + ``skill.publish_results(...)``
        can run. Returns ``None`` for unknown names.
        """
        # By convention, skill.name == skill.tool_def()["function"]["name"].
        # The straight dict lookup is the cheap path; iterate as a safety
        # net in case a skill author overrode tool_def with a different name.
        skill = self._ctx_skills.get(tool_name)
        if skill is not None:
            return skill
        for skill in self._ctx_skills.values():
            try:
                if skill.tool_def()["function"]["name"] == tool_name:
                    return skill
            except (KeyError, TypeError):
                continue
        return None

    def get_context_tools(self) -> List[Dict[str, Any]]:
        """Return tool definitions for every Σ_ctx skill, unfiltered.

        Context tools are always in ``Σ^avail_t`` per the paper's mask
        definition (only Σ_act is precondition-gated).
        """
        return [skill.tool_def() for skill in self._ctx_skills.values()]

    def get_action_skills_masked(
        self,
        state: SkillContext,
        history_state: Dict[str, Any],
        context: Dict[str, Any],
    ) -> List[Dict[str, Any]]:
        """Return tool definitions for Σ_act skills whose mask predicates both pass.

        Implements the Σ_act portion of the paper's action mask
        (Invariant I3):
        ``Σ^avail_t = Σ_ctx ∪ {σ ∈ Σ_act : I_σ(s_t) ∧ is_available(history)}``.

        Σ_ctx skills are always available per I3 — use ``get_context_tools()``
        for those; this accessor never inspects them.
        """
        tools: List[Dict[str, Any]] = []
        for skill in self._act_skills.values():
            avail_ok, _ = skill.is_available(history_state, context)
            if not avail_ok:
                continue
            # ActionSkill subclasses always declare precondition (abstract).
            pre_ok, _ = skill.precondition(state)
            if pre_ok:
                tools.append(skill.tool_def())
        return tools
