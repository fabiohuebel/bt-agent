"""Internal framework constants — not user-configurable.

These are implementation details centralized here for discoverability
and maintainability.  To change framework behavior for experiments,
edit this file.

For user-tunable parameters, see ``config.py`` (AgentConfig).
"""

# -- Tick / polling -----------------------------------------------------------
# Will be replaced by threading.Event (Section 25).  Until then,
# these control busy-wait intervals.

TICK_LOOP_SLEEP: float = 0.01  # seconds between BT subtree ticks
CONTEXT_SKILL_POLL_ITERS: int = 3000  # 3000 * 0.01s = 30s max wait

# -- Schema defaults ----------------------------------------------------------
# Applied to milestones that omit optional fields.

DEFAULT_MILESTONE_TYPE: str = "explore"
MILESTONE_REQUIRED_FIELDS = [
    "id",
    "description",
    "detailed_instructions",
    "verify_condition",
    "type",
    "dependencies",
]

# -- LLM compaction (actor) ---------------------------------------------------
# Only used when compaction="llm" (not default).

LLM_COMPACTION_TEMPERATURE: float = 0.0
LLM_COMPACTION_MAX_TOKENS: int = 512

# -- Evaluator fallbacks ------------------------------------------------------

CORRECTION_PREFIX: str = "\u26a0\ufe0f EVALUATOR CORRECTION: "

# -- Observability (OBS-DROP safe to remove) ----------------------------------

EVENT_SINK_RING_SIZE: int = 200
EVENT_SINK_QUEUE_SIZE: int = 1000
