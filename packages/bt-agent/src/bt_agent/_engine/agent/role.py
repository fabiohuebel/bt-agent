"""Role-aware tool visibility dispatch (paper §5, Invariant I2).

The paper claims **three parameter-tied prompt-conditioned policies**
``π^plan_θ, π^act_θ, π^eval_θ`` over a single VLM backbone, with **layered
tool visibility**:

* ``Role.PLANNER`` — sees framework-defined planner meta-tools
  (``done_gathering``, ``read_skill_docs``) plus the tool def of every
  ContextSkill in the registry. The planner's gather phase is a generic
  ReAct loop over Σ_ctx.
* ``Role.ACTOR`` — sees the full action mask
  ``Σ^avail_t = Σ_ctx ∪ {σ ∈ Σ_act : I_σ(s_t) ∧ is_available(history)}``
  plus the actor-flavored ``read_skill_docs`` meta-tool plus an optional
  ``defer`` meta-tool when gating allows.
* ``Role.EVALUATOR`` — sees no tools (uses ``model.query_structured``).

Step 6.5 framework-agnosticity contract — this module names ZERO
substrate-specific concepts. No "scene graph", no "navigate", no
"capture", no TIAMAT skill names anywhere. The grep guard test in
``tests/test_framework_agnosticity.py`` enforces this. Substrate-
specific tool defs come from skills in the registry via
``skill.tool_def()``; substrate-specific skill names live in skill
modules under ``skills/<family>_skills/``.

Both ``read_skill_docs`` variants share a name but have different schemas:
the planner's variant requires a ``reasoning`` field; the actor's
variant doesn't. Both are framework-defined (skill-doc reading is a
framework affordance, not a substrate concept).
"""

from __future__ import annotations

from enum import Enum
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    from bt_agent._engine.skills.protocol import SkillContext
    from bt_agent._engine.skills.registry import SkillRegistry


# ---------------------------------------------------------------------------
# Framework-defined meta-tools
# ---------------------------------------------------------------------------
#
# Three meta-tools are framework affordances — they don't correspond to any
# registered Skill, but every agent built on this framework can use them:
#
#   * read_skill_docs(skill_name)  — load a skill's .md doc on demand
#   * done_gathering(reasoning)    — planner signals end of ReAct gather loop
#   * defer(reason)                — actor escapes a milestone (gated)
#
# All three are substrate-agnostic. They reference no specific skill name.


def _build_planner_read_skill_docs_tool(available_skills: List[str]) -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "read_skill_docs",
            "description": (
                "Read documentation for a skill to understand its capabilities, "
                "failure modes, and usage patterns. "
                f"Available: {', '.join(available_skills)}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": "Why you need to read this skill's documentation.",
                    },
                    "skill_name": {
                        "type": "string",
                        "description": f"One of: {', '.join(available_skills)}.",
                    },
                },
                "required": ["reasoning", "skill_name"],
            },
        },
    }


def _build_done_gathering_tool() -> Dict[str, Any]:
    return {
        "type": "function",
        "function": {
            "name": "done_gathering",
            "description": (
                "Signal that you have gathered enough information and are ready to "
                "produce the plan. Call this when you have queried all relevant "
                "items and understand the task structure."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "reasoning": {
                        "type": "string",
                        "description": (
                            "Summarize what you learned: which items are found, "
                            "which need exploration, what dependencies exist."
                        ),
                    },
                },
                "required": ["reasoning"],
            },
        },
    }


def _build_actor_read_skill_docs_tool(available_skills: List[str]) -> Dict[str, Any]:
    """Actor-side ``read_skill_docs`` — same name as the planner's variant but
    omits the ``reasoning`` field (the actor's per-turn `reasoning` field on
    every other tool already covers it)."""
    return {
        "type": "function",
        "function": {
            "name": "read_skill_docs",
            "description": (
                "Read the full documentation for a skill. Use this when you need to "
                "understand a skill's detailed behavior, failure modes, recovery "
                "actions, or typical usage sequences. "
                f"Available: {', '.join(available_skills)}."
            ),
            "parameters": {
                "type": "object",
                "properties": {
                    "skill_name": {
                        "type": "string",
                        "description": f"Name of the skill to read docs for. One of: {', '.join(available_skills)}.",
                    },
                },
                "required": ["skill_name"],
            },
        },
    }


_CLAIM_COMPLETE_TOOL_DEF: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "claim_complete",
        "description": (
            "Claim that the current milestone is complete. The evaluator "
            "will verify your claim. Your preceding reasoning should "
            "contain evidence of what was accomplished."
        ),
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}

_DEFER_TOOL_DEF: Dict[str, Any] = {
    "type": "function",
    "function": {
        "name": "defer",
        "description": "Skip this milestone and retry it later.",
        "parameters": {
            "type": "object",
            "properties": {},
            "required": [],
        },
    },
}


# Framework-defined meta-tool names — used by planner/actor dispatch loops to
# distinguish meta-tools (framework-handled) from skill tools (registry-routed).
PLANNER_META_TOOL_NAMES = frozenset({"done_gathering", "read_skill_docs"})
ACTOR_META_TOOL_NAMES = frozenset({"read_skill_docs", "defer", "claim_complete"})


# ---------------------------------------------------------------------------
# Role enum + dispatcher
# ---------------------------------------------------------------------------


class Role(Enum):
    """The three parameter-tied prompt-conditioned policies (paper §5).

    String values are stable — they appear in observability event payloads
    (Step 7/8 will tag ``mask_computed`` and invariant audit events with
    ``role=<value>``).
    """

    PLANNER = "planner"
    ACTOR = "actor"
    EVALUATOR = "evaluator"

    def available_tools(
        self,
        *,
        registry: Optional["SkillRegistry"] = None,
        state: Optional["SkillContext"] = None,
        history_state: Optional[Dict[str, Any]] = None,
        gating_context: Optional[Dict[str, Any]] = None,
    ) -> List[Dict[str, Any]]:
        """Return the OpenAI-format tool list this role sees this turn.

        Per Step 6.5: skill tool defs come from the registry via
        ``skill.tool_def()``; meta-tools are framework-defined here.
        ``role.py`` names ZERO substrate-specific concepts.

        Args:
            registry: Required for ``PLANNER`` and ``ACTOR``. Source of
                Σ_ctx tool defs (planner sees all; actor sees all + masked
                Σ_act) and the skill-name list for ``read_skill_docs``.
            state: Required for ``ACTOR``. Used by
                ``registry.get_action_skills_masked`` to compute the
                paper's I_σ predicate per Σ_act skill.
            history_state: Optional analysis of the current milestone's
                episode history; passed to ``is_available()`` checks.
                Defaults to ``{}``.
            gating_context: Optional turn-level context (e.g.,
                ``can_defer``); passed to ``is_available()`` and consulted
                for ``defer`` inclusion. Defaults to ``{}``.

        Returns:
            Ordered list of OpenAI-format tool definitions.

        Raises:
            ValueError: if a required arg is missing for the chosen role.
        """
        from bt_agent._engine.skills import list_skill_docs

        if self is Role.EVALUATOR:
            return []

        if registry is None:
            raise ValueError(f"Role.{self.name}.available_tools(...) requires registry; got None.")

        if self is Role.PLANNER:
            # Planner gather phase: framework meta-tools first, then every
            # ContextSkill's tool_def. The planner's ReAct loop dispatches
            # by tool name — meta-tool names are handled in-loop, skill
            # names are routed via registry.context_skill_by_tool_name().
            #
            # ``read_skill_docs`` is intentionally NOT exposed to the planner.
            # The planner is a decomposer, not an executor; the per-skill
            # tool_def description+usage strings are enough to plan with,
            # and dumping full markdown docs into the gather conversation
            # bloated context (~11k chars per cascade) and caused observed
            # "lost in the middle" drops of earlier gather observations.
            # The actor still has read_skill_docs (see Role.ACTOR below).
            meta = [_build_done_gathering_tool()]
            skill_tools = [s.tool_def() for s in registry.context_skills()]
            return meta + skill_tools

        # Role.ACTOR
        if state is None:
            raise ValueError("Role.ACTOR.available_tools(...) requires state (SkillContext); got None.")

        ctx_tools = registry.get_context_tools()
        act_tools = registry.get_action_skills_masked(state, history_state or {}, gating_context or {})
        # Order matches the pre-Step-6 actor flow: SelectSkill built
        # ``ctx + act + [?defer]`` and the actor's select_skill() then
        # appended read_skill_docs. Preserved for byte-identical actor
        # regression — see test_actor_byte_identical_pre_step6_select_skill_flow.
        tools: List[Dict[str, Any]] = list(ctx_tools) + list(act_tools)
        tools.append(_CLAIM_COMPLETE_TOOL_DEF)
        if (gating_context or {}).get("can_defer", False):
            tools.append(_DEFER_TOOL_DEF)
        tools.append(_build_actor_read_skill_docs_tool(list_skill_docs()))
        return tools
