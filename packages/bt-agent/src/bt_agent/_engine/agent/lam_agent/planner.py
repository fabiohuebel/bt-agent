"""Planner — agentic task decomposition via a generic ReAct gather loop.

The planner runs a two-phase process:

1. **Gather** (ReAct loop): The planner sees a substrate-agnostic tool
   list (``Role.PLANNER.available_tools``) — framework meta-tools
   (``done_gathering``, ``read_skill_docs``) plus the ``tool_def()`` of
   every ContextSkill in the registry. Each turn the LLM picks a tool;
   the loop dispatches by name (meta-tools are handled in-loop, skill
   tools are routed via ``registry.context_skill_by_tool_name`` and
   executed via ``skill.start()`` + ``skill.publish_results()``).
   Substrate-specific data flows through Step 10's K-promotion hook —
   no field names appear in framework code.
2. **Synthesize**: A structured LLM call produces milestones with a
   dependency DAG and types, conditioned on the gather history.

The BT node (PlanTask) handles lifecycle: threading, blackboard, tick
status. This module handles reasoning: what to ask the model and how to
interpret it.

The two-phase split is the LAM Agent's design choice — see ARCHITECTURE.md
"LAM Agent design note." The framework supports both two-phase planners
(this module) and single-shot planners (DecisionMakerAgent) equally well.
"""

from __future__ import annotations

import json
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

from bt_agent._engine.agent.role import Role
from bt_agent._engine.agent.system_prompts import get_prompts

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Defaults (used when no config is provided; overridden by AgentConfig)
# ---------------------------------------------------------------------------

_MAX_ITERATIONS = 10
_WALL_TIME_BUDGET = 30.0  # seconds
_MAX_SKILL_DOC_READS = 3

# Prompts live under ``bt_agent._engine.agent.system_prompts.<variant>/`` (Step 5).
# Resolved per-call via ``get_prompts(prompt_variant)``.


# ---------------------------------------------------------------------------
# Plan schema for structured output
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# Milestone schema — core (framework) + extensible (substrate)
# ---------------------------------------------------------------------------
#
# The framework defines 6 substrate-agnostic milestone fields. Substrates
# add their own via ``milestone_schema_extension`` (e.g., TIAMAT adds
# ``capture`` + ``capture_target``). The extension is merged at plan time
# by ``build_milestone_schema()``.

_MILESTONE_CORE_PROPERTIES: Dict[str, Any] = {
    "id": {
        "type": "string",
        "description": "Unique milestone ID, e.g. 'find_umbrella', 'open_safe'.",
    },
    "description": {"type": "string"},
    "detailed_instructions": {
        "type": "string",
        "description": "Step-by-step instructions for the agent.",
    },
    "verify_condition": {
        "type": "string",
        "description": "How to verify this milestone is complete.",
    },
    "type": {
        "type": "string",
        "enum": ["direct", "explore", "dependent"],
    },
    "dependencies": {
        "type": "array",
        "items": {"type": "string"},
        "description": "Milestone IDs that must complete before this one starts.",
    },
}

_MILESTONE_CORE_REQUIRED: List[str] = [
    "id",
    "description",
    "detailed_instructions",
    "verify_condition",
    "type",
    "dependencies",
]


def _slim_properties(properties: Dict[str, Any]) -> Dict[str, Any]:
    """Drop the verbose ``description`` guidance strings from a property map.

    Structural-fix schema slimming (docs/spec/lam-improvement.md CP1): the plan
    schema is echoed into the synthesis prompt AND constrains structured decoding.
    Dropping the human-facing ``description`` strings shrinks the schema the weak
    model must parrot without changing the required output shape (same keys, same
    ``required`` list, same enums), reducing the output-token pressure that drives
    truncation. Enum constraints (``type``) are preserved — only free-text
    ``description`` hints are removed.
    """
    slimmed: Dict[str, Any] = {}
    for key, spec in properties.items():
        if isinstance(spec, dict) and "description" in spec:
            spec = {k: v for k, v in spec.items() if k != "description"}
        slimmed[key] = spec
    return slimmed


def build_milestone_schema(
    extension: Optional[Dict[str, Any]] = None,
    *,
    slim: bool = False,
) -> Dict[str, Any]:
    """Build the full plan-output schema from core fields + substrate extension.

    Args:
        extension: Substrate-specific milestone properties to merge (e.g.,
            TIAMAT's ``{"capture": {"type": "boolean", ...}, ...}``).
            Extension fields are added to properties but NOT to the
            required list — the LLM may omit them without schema violation.
        slim: Structural-fix opt-in. When True, drop the verbose per-field
            ``description`` strings (input-side guidance only) to reduce
            output-token pressure. Default False keeps the schema
            BYTE-IDENTICAL to the frozen variants.

    Returns:
        Complete JSON schema for ``model.query_structured()``.
    """
    properties = dict(_MILESTONE_CORE_PROPERTIES)
    required = list(_MILESTONE_CORE_REQUIRED)
    if extension:
        properties.update(extension)
    if slim:
        properties = _slim_properties(properties)

    return {
        "type": "object",
        "properties": {
            "reasoning": {
                "type": "string",
                "description": "Step-by-step analysis of the task, findings, dependencies, and plan strategy.",
            },
            "milestones": {
                "type": "array",
                "items": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        },
        "required": ["reasoning", "milestones"],
    }


# ---------------------------------------------------------------------------
# Tool definitions for the planner loop now live in bt_agent._engine.agent.role
# (Step 6: layered visibility dispatcher).
# ---------------------------------------------------------------------------


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------


def generate_plan(
    model: Any,
    task: str,
    scenario: str,
    *,
    skill_registry: Any = None,
    object_store: Any = None,
    platform_state: Optional[Dict[str, Any]] = None,
    milestone_schema_extension: Optional[Dict[str, Any]] = None,
    milestone_fields_guidance: str = "",
    max_iterations: int = _MAX_ITERATIONS,
    wall_time_budget: float = _WALL_TIME_BUDGET,
    event_sink: Any = None,  # OBS-DROP: visualizer pubsub broker, may be None
    replan_context: Optional[Dict[str, Any]] = None,
    prompt_variant: str = "tiamat_agent_submit_v5",
    synthesis_temperature: float = 0.0,
    replan_temperature: float = 0.7,
    synthesis_max_tokens: Optional[int] = None,
    slim_schema: bool = False,
) -> Tuple[List[Dict[str, Any]], str, Dict[str, Any]]:
    """Generate milestones from a task description via agentic planning.

    Phase 1: Use context-gathering tools to ground the task.
    Phase 2: Synthesize a structured milestone plan with dependency DAG.

    Args:
        model: ModelBackend instance.
        task: High-level task description.
        scenario: Environment/scenario context.
        skill_registry: SkillRegistry to look up context skills (optional).
        object_store: ObjectStore to populate via publish_results (optional).
        platform_state: Opaque platform-specific robot state (paper §5: s_robot).
            Passed through to SkillContext for context-skill execution during
            the gather phase. The framework never inspects the contents.
        max_iterations: Maximum tool-call iterations in the gathering phase.
        wall_time_budget: Maximum wall-clock seconds for the gathering phase.
        replan_context: Optional dict from a prior failed attempt
            ``{failed_milestone_id, failed_description, reason, completed_descriptions}``.
            When present, the gather-phase prompt is augmented with a
            "previous attempt notes" block so the planner knows what's already
            done and what failed (paper §5: suffix-only replan).

    Returns:
        (milestones, reasoning, plan_data) where plan_data is an empty dict
        (reserved for future substrate-agnostic metadata).
    """
    # On replan, raise temperature for both gather and synthesis so the
    # planner explores a different scene-graph query path and emits a
    # different plan shape instead of re-emitting the same failed plan.
    is_replan = replan_context is not None
    effective_temp = replan_temperature if is_replan else synthesis_temperature

    # ---- OBS-DROP-BEGIN ----
    if event_sink is not None:
        event_sink.emit(
            "planner.gather.start",
            {
                "task": task,
                "scenario": scenario,
                "replan": is_replan,
                "temperature": effective_temp,
            },
        )
    # ---- OBS-DROP-END ----

    # -- Phase 1: Information Gathering ----------------------------------------

    gather_history = _gather_information(
        model,
        task,
        scenario,
        skill_registry=skill_registry,
        object_store=object_store,
        platform_state=platform_state or {},
        max_iterations=max_iterations,
        wall_time_budget=wall_time_budget,
        event_sink=event_sink,  # OBS-DROP
        replan_context=replan_context,
        prompt_variant=prompt_variant,
        temperature=effective_temp,
    )

    # ---- OBS-DROP-BEGIN ----
    if event_sink is not None:
        event_sink.emit(
            "planner.gather.done",
            {
                "iterations": len(gather_history),
                "tools_called": [tool for tool, _, _ in gather_history],
            },
        )
        event_sink.emit("planner.synthesis.start", {})
    # ---- OBS-DROP-END ----

    # -- Phase 2: Plan Synthesis -----------------------------------------------

    milestones, reasoning = _synthesize_plan(
        model,
        task,
        scenario,
        gather_history,
        prompt_variant=prompt_variant,
        milestone_schema_extension=milestone_schema_extension,
        milestone_fields_guidance=milestone_fields_guidance,
        replan_context=replan_context,
        temperature=effective_temp,
        synthesis_max_tokens=synthesis_max_tokens,
        slim_schema=slim_schema,
    )

    # ---- OBS-DROP-BEGIN ----
    if event_sink is not None:
        event_sink.emit(
            "planner.synthesis.done",
            {"milestone_count": len(milestones)},
        )
    # ---- OBS-DROP-END ----

    plan_data: Dict[str, Any] = {}

    return milestones, reasoning, plan_data


# ---------------------------------------------------------------------------
# Phase 1: Information Gathering
# ---------------------------------------------------------------------------


def _gather_information(
    model: Any,
    task: str,
    scenario: str,
    *,
    skill_registry: Any,
    object_store: Any,
    platform_state: Dict[str, Any],
    max_iterations: int,
    wall_time_budget: float,
    event_sink: Any = None,  # OBS-DROP: visualizer pubsub broker
    replan_context: Optional[Dict[str, Any]] = None,
    prompt_variant: str = "tiamat_agent_submit_v5",
    temperature: float = 0.0,
) -> List[Tuple[str, Dict[str, Any], str]]:
    """Generic ReAct gather loop over ``Role.PLANNER.available_tools``.

    Substrate-agnostic dispatch:
      * Framework meta-tools (``done_gathering``, ``read_skill_docs``) are
        handled in-loop.
      * Skill tool calls are routed via
        ``registry.context_skill_by_tool_name(tool_name)`` and executed
        through ``skill.start(...)`` + ``skill.publish_results(...)``
        (the Step-10 K-promotion hook). No skill-specific field names
        appear in this function.

    Returns the gather history as a list of ``(tool_name, args, observation)``
    tuples — the synthesis pass formats this list into the
    ``{gathered_info}`` block of its prompt template.
    """
    from bt_agent._engine.agent.lam_agent.actor import build_history_entry, select_action
    from bt_agent._engine.skills import load_skill_doc
    from bt_agent._engine.skills.protocol import SkillContext

    bundle = get_prompts(prompt_variant)
    system_content = bundle.planner_gather.format(task=task, scenario=scenario)

    user_prompt = (
        "Begin planning. Use the available context-gathering tools to "
        "ground the task in the current state of the environment, then "
        "call `done_gathering`."
    )
    if replan_context is not None:
        # Step 3: suffix replan — tell the planner what's already done and what failed
        # so it can decide whether to re-include the failed milestone with a different
        # approach or skip it. The replan template is variant-owned (Step 5) so
        # ablations (no-Reflexion, with-Reflexion) can change its structure.
        completed = replan_context.get("completed_descriptions") or []
        completed_block = "\n".join(f"  - {d}" for d in completed) if completed else "  (none)"
        user_prompt = bundle.planner_replan.format(
            completed_block=completed_block,
            failed_description=replan_context.get("failed_description", ""),
            failed_milestone_id=replan_context.get("failed_milestone_id", ""),
            reason=replan_context.get("reason", "unspecified"),
            base_user_prompt=user_prompt,
        )

    messages: List[Dict[str, Any]] = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": user_prompt},
    ]

    # Layered visibility (paper §5 Invariant I2): the Planner role sees
    # framework meta-tools + every ContextSkill's tool_def. Single source
    # of truth for per-role tool lists is bt_agent._engine.agent.role.
    tools = Role.PLANNER.available_tools(registry=skill_registry)

    # Persistent SkillContext for any context-skill tool calls during gather.
    # platform_state (paper §5: s_robot) is passed through opaquely — the
    # framework never inspects its contents.
    skill_context = SkillContext(
        object_store=object_store,
        main_task=task,
        scenario=scenario,
        platform_state=platform_state,
    )

    # Substrate-agnostic gather history: (tool_name, args, observation).
    history: List[Tuple[str, Dict[str, Any], str]] = []
    skill_doc_reads = 0  # kept for the defensive read_skill_docs dispatch below

    start = time.monotonic()

    for iteration in range(max_iterations):
        elapsed = time.monotonic() - start
        if elapsed > wall_time_budget:
            logger.info("Planner wall-time budget exceeded (%.1fs), moving to synthesis", elapsed)
            break

        # NOTE: the iter-4 doc-read nudge previously lived here. It was
        # removed because read_skill_docs is no longer exposed to the planner
        # (see Role.PLANNER.available_tools). The dispatch branch for
        # read_skill_docs below is kept defensively in case the tool gets
        # re-enabled later, but the model has no way to invoke it.

        call_start = time.monotonic()  # OBS-DROP
        try:
            reasoning_text, tool_name, tool_args, raw = select_action(
                model, messages, tools, mode="react_single", temperature=temperature
            )
        except Exception as e:
            logger.warning("Planner select_action failed: %s", e)
            break
        latency_ms = (time.monotonic() - call_start) * 1000.0  # OBS-DROP

        # select_action's react_single path already has a None-tool re-prompt
        # fallback. If it still comes back None we treat that as a hard stop
        # rather than the old "silent done" auto-mode behavior.
        if tool_name is None:
            logger.warning(
                "Planner produced no tool call at iteration %d after select_action(react_single) re-prompt; aborting"
                " gather",
                iteration + 1,
            )
            break

        logger.info("Planner iteration %d: %s", iteration + 1, tool_name)

        # ---- Framework meta-tools ----------------------------------------

        if tool_name == "done_gathering":
            logger.info("Planner done gathering after %d iterations (%.1fs)", iteration + 1, time.monotonic() - start)
            # ---- OBS-DROP-BEGIN ----
            if event_sink is not None:
                event_sink.emit(
                    "planner.gather.tool_call",
                    {
                        "iter": iteration + 1,
                        "tool": "done_gathering",
                        "args": {},
                        "result_summary": (reasoning_text or tool_args.get("reasoning", ""))[:240],
                        "latency_ms": round(latency_ms, 1),
                    },
                )
            # ---- OBS-DROP-END ----
            break

        if tool_name == "read_skill_docs":
            if skill_doc_reads >= _MAX_SKILL_DOC_READS:
                observation = "Skill doc read limit reached."
            else:
                skill_name = tool_args.get("skill_name", "")
                observation = load_skill_doc(skill_name)
                skill_doc_reads += 1
            messages.extend(build_history_entry("read_skill_docs", tool_args, observation, reasoning=reasoning_text))
            history.append(("read_skill_docs", dict(tool_args), observation))
            # ---- OBS-DROP-BEGIN ----
            if event_sink is not None:
                event_sink.emit(
                    "planner.gather.tool_call",
                    {
                        "iter": iteration + 1,
                        "tool": "read_skill_docs",
                        "args": {"skill_name": tool_args.get("skill_name", "")},
                        "result_summary": observation[:240],
                        "latency_ms": round(latency_ms, 1),
                    },
                )
            # ---- OBS-DROP-END ----
            continue

        # ---- ContextSkill dispatch (substrate-agnostic) ------------------

        skill = skill_registry.context_skill_by_tool_name(tool_name) if skill_registry else None
        if skill is None:
            logger.warning("Planner returned unknown tool: %s", tool_name)
            observation = f"Unknown tool '{tool_name}'. Pick a tool from the list or call done_gathering."
            messages.extend(build_history_entry(tool_name, tool_args, observation, reasoning=reasoning_text))
            history.append((tool_name, dict(tool_args), observation))
            continue

        observation = _run_context_skill(skill, tool_args, skill_context)
        messages.extend(build_history_entry(tool_name, tool_args, observation, reasoning=reasoning_text))
        history.append((tool_name, dict(tool_args), observation))
        # ---- OBS-DROP-BEGIN ----
        if event_sink is not None:
            event_sink.emit(
                "planner.gather.tool_call",
                {
                    "iter": iteration + 1,
                    "tool": tool_name,
                    "args": _summarize_args_for_event(tool_args),
                    "result_summary": observation[:240],
                    "latency_ms": round(latency_ms, 1),
                },
            )
        # ---- OBS-DROP-END ----

    return history


def _run_context_skill(
    skill: Any,
    raw_args: Dict[str, Any],
    context: Any,
) -> str:
    """Generic ContextSkill executor for the planner gather loop.

    Substrate-agnostic: the framework only knows ``skill.prepare(...)``,
    ``skill.start(...)``, and ``skill.publish_results(handle, store)``.
    Skill-specific data (objects, mappings, captures, ...) flows through
    the Step-10 K-promotion hook — no skill-specific field names are
    referenced here.
    """
    from bt_agent._engine.skills.protocol import SkillStatus

    # Strip the auto-added envelope fields the framework's tool_def adds
    # before passing args to the skill.
    skill_args = {k: v for k, v in (raw_args or {}).items() if k not in ("reasoning", "memory_note")}
    try:
        prepared, error = skill.prepare(skill_args, context)
        if error:
            return f"{skill.name} preparation failed: {error}"
        handle = skill.start(prepared, context)
        # Wait for skill completion — Event.wait wakes instantly when
        # the skill thread sets _completion_event at terminal state.
        handle._completion_event.wait(timeout=30.0)
    except Exception as e:
        logger.warning("Context skill '%s' failed: %s", skill.name, e)
        return f"{skill.name} failed: {e}"

    if handle.status == SkillStatus.SUCCEEDED and context.object_store is not None:
        # K-promotion (paper §5 I6) — Step-10 hook. The framework calls
        # this; the skill decides what (if anything) to write to K.
        try:
            skill.publish_results(handle, context.object_store)
        except Exception as e:
            logger.warning("publish_results failed for '%s': %s", skill.name, e)

    return handle.observation or f"No result from {skill.name}."


def _summarize_args_for_event(args: Dict[str, Any]) -> Dict[str, Any]:
    """Compact, observability-only arg summary (drops verbose envelope fields)."""
    return {k: v for k, v in (args or {}).items() if k not in ("reasoning", "memory_note")}


# ---------------------------------------------------------------------------
# Phase 2: Plan Synthesis
# ---------------------------------------------------------------------------


def _synthesize_plan(
    model: Any,
    task: str,
    scenario: str,
    gather_history: List[Tuple[str, Dict[str, Any], str]],
    *,
    prompt_variant: str = "tiamat_agent_submit_v5",
    milestone_schema_extension: Optional[Dict[str, Any]] = None,
    milestone_fields_guidance: str = "",
    replan_context: Optional[Dict[str, Any]] = None,
    temperature: float = 0.0,
    synthesis_max_tokens: Optional[int] = None,
    slim_schema: bool = False,
) -> Tuple[List[Dict[str, Any]], str]:
    """Call model.query_structured() to produce the milestone plan.

    ``synthesis_max_tokens`` (structural fix, CP1): when provided, it is passed
    to ``query_structured`` as an explicit output-token budget. Default ``None``
    is BYTE-IDENTICAL to the frozen path — the call omits ``max_tokens`` and the
    backend falls back to its own default. The frozen variants truncate because
    that backend default (2048 on the text-rung lane) is too small for the plan
    schema; the structural variant threads a generous budget here to kill it.

    Returns:
        (milestones, reasoning)
    """
    gathered_info = _format_gathered(gather_history)
    bundle = get_prompts(prompt_variant)

    # On replan, prepend completed/failed context to the user message so the
    # synthesis LLM knows what's already done and only produces new milestones.
    user_content = "Produce the milestone plan."
    if replan_context is not None:
        completed = replan_context.get("completed_descriptions") or []
        completed_lines = "\n".join(f"  - {d}" for d in completed) if completed else "  (none)"
        failed_desc = replan_context.get("failed_description", "")
        reason = replan_context.get("reason", "unspecified")
        user_content = (
            "# Previous Attempt\n"
            "The following milestones are ALREADY DONE — do NOT re-plan them:\n"
            f"{completed_lines}\n\n"
            f'Failed milestone: "{failed_desc}"\n'
            f"Failure reason: {reason}\n\n"
            "Only produce milestones for REMAINING work. You may re-include "
            "the failed milestone with a different approach if recoverable, "
            "or skip it.\n\n"
            "Produce the milestone plan."
        )

    messages: List[Dict[str, Any]] = [
        {
            "role": "system",
            "content": bundle.planner_synthesis.format(
                task=task,
                scenario=scenario,
                gathered_info=gathered_info,
                milestone_fields=milestone_fields_guidance,
            ),
        },
        {"role": "user", "content": user_content},
    ]

    schema = build_milestone_schema(milestone_schema_extension, slim=slim_schema)
    # BYTE-IDENTICAL guard: omit max_tokens entirely when None so the frozen
    # variants hit the same backend-default path as before.
    structured_kwargs: Dict[str, Any] = {"temperature": temperature}
    if synthesis_max_tokens is not None:
        structured_kwargs["max_tokens"] = synthesis_max_tokens
    try:
        plan = model.query_structured(messages, schema, phase="planner", **structured_kwargs)
    except Exception as e:
        logger.warning("Structured plan synthesis failed (%s), trying text fallback", e)
        plan = _fallback_text_synthesis(model, messages, temperature=temperature, max_tokens=synthesis_max_tokens)

    reasoning = plan.get("reasoning", "")
    milestones = _validate_plan(plan)

    logger.info("Plan: %d milestones", len(milestones))

    return milestones, reasoning


def _fallback_text_synthesis(
    model: Any,
    messages: List[Dict[str, Any]],
    *,
    temperature: float = 0.0,
    max_tokens: Optional[int] = None,
) -> Dict[str, Any]:
    """Fallback: ask for JSON in plain text and parse it.

    ``max_tokens`` defaults to 4096 (the frozen value) when None; the structural
    variant threads its larger synthesis budget here so the text-salvage path
    isn't itself truncated.
    """
    messages = list(messages)
    messages[-1] = dict(messages[-1])
    messages[-1][
        "content"
    ] += '\n\nRespond with ONLY valid JSON matching this structure: {"reasoning": "...", "milestones": [...]}'

    try:
        text = model.query(messages, max_tokens=max_tokens or 4096, temperature=temperature, phase="planner")
        # Try to extract JSON from the response
        start = text.find("{")
        end = text.rfind("}") + 1
        if start >= 0 and end > start:
            return json.loads(text[start:end])
    except Exception as e:
        logger.error("Fallback text synthesis also failed: %s", e)

    return {"reasoning": "Fallback plan", "milestones": []}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _format_gathered(gather_history: List[Tuple[str, Dict[str, Any], str]]) -> str:
    """Format the planner's gather history for the synthesis prompt.

    Substrate-agnostic: each line is ``- tool_name(arg=val, ...): observation``.
    The synthesis pass reads this verbatim — the framework doesn't interpret
    tool names or observation contents.
    """
    if not gather_history:
        return "No context-gathering tools were called."

    lines = []
    for tool_name, args, observation in gather_history:
        # Drop envelope fields; show only substantive args.
        substantive = {k: v for k, v in (args or {}).items() if k not in ("reasoning", "memory_note")}
        args_str = ", ".join(f"{k}={v!r}" for k, v in substantive.items())
        prefix = f"{tool_name}({args_str})" if args_str else f"{tool_name}()"
        lines.append(f"- {prefix}: {observation}")

    return "\n".join(lines)


def _validate_plan(plan: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Validate the plan DAG and apply defaults.

    Raises ValueError on fatal issues (cycles, missing deps).
    """
    milestones = plan.get("milestones", [])

    if not milestones:
        return milestones

    # Check unique IDs
    ids = [m["id"] for m in milestones]
    id_set = set(ids)
    if len(ids) != len(id_set):
        dupes = [i for i in ids if ids.count(i) > 1]
        logger.warning("Duplicate milestone IDs: %s — deduplicating", dupes)
        seen = set()
        unique = []
        for m in milestones:
            if m["id"] not in seen:
                seen.add(m["id"])
                unique.append(m)
        milestones = unique
        id_set = {m["id"] for m in milestones}

    # Check dependency references exist
    for m in milestones:
        deps = m.get("dependencies", [])
        for dep in deps:
            if dep not in id_set:
                logger.warning("Milestone '%s' depends on unknown '%s' — removing", m["id"], dep)
                m["dependencies"] = [d for d in deps if d in id_set]

    # Check for cycles (DFS)
    if _has_cycle(milestones):
        logger.warning("Dependency cycle detected — clearing all dependencies")
        for m in milestones:
            m["dependencies"] = []

    # Apply defaults for core fields only. Substrate-specific fields
    # (e.g., TIAMAT's capture/capture_target) are handled by the substrate
    # — the framework doesn't validate or default them.
    for m in milestones:
        m.setdefault("type", "explore")
        m.setdefault("dependencies", [])
        m.setdefault("detailed_instructions", m.get("description", ""))
        m.setdefault("verify_condition", f"Is '{m.get('description', 'the task')}' complete?")

    return milestones


def _has_cycle(milestones: List[Dict[str, Any]]) -> bool:
    """Check for cycles in the dependency DAG using DFS."""
    adj: Dict[str, List[str]] = {m["id"]: m.get("dependencies", []) for m in milestones}
    visited: set = set()
    in_stack: set = set()

    def dfs(node: str) -> bool:
        if node in in_stack:
            return True
        if node in visited:
            return False
        visited.add(node)
        in_stack.add(node)
        for dep in adj.get(node, []):
            if dfs(dep):
                return True
        in_stack.discard(node)
        return False

    return any(dfs(m_id) for m_id in adj)
