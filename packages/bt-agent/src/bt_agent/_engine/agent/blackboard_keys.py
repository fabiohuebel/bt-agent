"""Typed blackboard key constants.

All blackboard access goes through these constants. This prevents typo bugs
and serves as documentation of the shared state contract between BT nodes,
the bridge, and the agent.
"""


class BB:
    """Blackboard key namespace."""

    # -- World model (written by bridge, read by BT nodes) ------------------

    TASK = "task"  # str — current high-level task
    SCENARIO = "scenario"  # str — scenario/environment description

    # Platform-specific robot state (paper §5: s_robot). The framework is
    # agnostic to the contents — the bridge writes whatever the platform
    # provides as a dict. Skills read what they need via .get().
    PLATFORM_STATE = "platform_state"  # Dict[str, Any] — opaque s_robot

    # -- Planning (written by PlanTask, read by milestone nodes) ------------

    MILESTONES = "milestones"  # List[Dict] — milestone dicts from planner
    CURRENT_MILESTONE_IDX = "milestone_idx"  # int — index of active milestone
    COMPLETED_MILESTONES = "completed_milestones"  # Set[int] — indices of completed milestones
    DEFERRED_MILESTONES = "deferred_milestones"  # List[int] — FIFO queue of deferred indices
    PLAN_REASONING = "plan_reasoning"  # str — planner's chain-of-thought reasoning
    NAV_GRAPH = "nav_graph"  # Dict — navigation graph data for PathPlanner

    # -- Skill execution (written by SelectSkill/ExecuteSkill) ---------------

    SELECTED_SKILL = "selected_skill"  # str — skill name chosen by LLM
    SKILL_PARAMS = "skill_params"  # Dict — raw LLM params for the skill
    SKILL_HANDLE = "skill_handle"  # SkillHandle — active execution handle
    SKILL_REASONING = "skill_reasoning"  # str — LLM reasoning for this choice
    COMPLETION_CLAIMED = "completion_claimed"  # bool — LLM returned TASK_COMPLETE
    # physics-trace-recording wave 2: uuid minted by SelectSkill the instant the
    # actor picks an action. Carried through skill_selected -> skill_started ->
    # skill_completed -> (via SkillContext) capture/submission, so the offline
    # extractor can pair events by id instead of FIFO-by-skill-name.
    SKILL_CALL_ID = "skill_call_id"

    # -- Three-way verdict + amend bookkeeping (paper §5 Invariant I7) -------

    LAST_VERDICT = "last_verdict"  # Optional[Verdict] — pass/amend/replan from last EvaluateCompletion
    AMEND_COUNT = "amend_count"  # int — per-milestone amend counter (reset on advance/replan)
    AMEND_CAP = "amend_cap"  # int — N_max; constant, set once in LAMAgent.__init__
    AMEND_NOTE = (  # Optional[str] — δ corrective note; written by EvaluateCompletion, consumed by ApplyAmend
        "amend_note"
    )
    REPLAN_CONTEXT = "replan_context"  # Optional[Dict] — failed milestone + completed descriptions for next PlanTask

    # -- Structural-fix bundle (docs/spec/lam-improvement.md; gated) ----------
    # Only written/read when AgentConfig.structural_fixes is True. Frozen
    # variants never touch these (the gated nodes that use them are absent from
    # the tree), so the frozen path is byte-identical.

    EMPTY_PLAN_COUNT = "empty_plan_count"  # int — consecutive empty syntheses (CP2)
    REACT_FALLBACK_ACTIVE = "react_fallback_active"  # bool — flat react-fallback plan installed (CP2)
    CORRECTIVE_REPLAN_DONE = "corrective_replan_done"  # bool — the one all_done corrective replan was spent (CP3)

    # -- Ablation switches (docs/spec/tiamat-ablation-flags.md CP3) ----------

    # bool -- set by SelectSkill._give_up_context_overflow on a confirmed
    # give-up submit; drive_episode (lam_v2_textrung/board_runner.py) reads
    # this to stamp termination_reason="context_overflow" instead of the
    # generic "agent_submit".
    CONTEXT_OVERFLOW_GIVE_UP = "context_overflow_give_up"

    # -- Shared services (initialized by agent, read by BT nodes) -----------

    AGENT_CONFIG = "agent_config"  # AgentConfig — all user-tunable constants
    SKILL_REGISTRY = "skill_registry"  # SkillRegistry
    MODEL_BACKEND = "model_backend"  # ModelBackend — unified VLM/LLM client
    MEMORY = "memory"  # MemoryManager
    CONVERSATION = "conversation"  # List[Dict] — OpenAI messages within current milestone
    OBJECT_STORE = "object_store"  # ObjectStore
    PROMPT_VARIANT = "prompt_variant"  # str — system_prompts variant name (Step 5)
    MILESTONE_SCHEMA_EXTENSION = "milestone_schema_extension"  # Optional[Dict] — substrate milestone fields
    MILESTONE_FIELDS_GUIDANCE = "milestone_fields_guidance"  # str — substrate milestone prompt guidance

    # -- Cancellation --------------------------------------------------------

    CANCEL_REQUESTED = "cancel_requested"  # bool
    CANCEL_SOURCE = "cancel_source"  # str — "human" | "agent" | "task_replacement"
    CANCEL_REASON = "cancel_reason"  # str

    # -- Task lifecycle ------------------------------------------------------

    TASK_COMPLETE = "task_complete"  # bool
    TASK_SUBMITTED = "task_submitted"  # bool

    # -- Outputs (written by BT nodes, drained by bridge) --------------------

    PENDING_OUTPUTS = "pending_outputs"  # List[Dict] — captures, submissions to publish
    PENDING_EVENTS = "pending_events"  # List[AgentEvent] — for GUI/logging

    # -- Live observability (OBS-DROP: visualizer-only, safe to remove) ------

    EVENT_SINK = "event_sink"  # Optional[EventSink] — pubsub broker, None if no viz
