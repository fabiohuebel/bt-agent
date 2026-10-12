"""Agent affordance configuration.

All user-tunable constants live here as frozen dataclasses with YAML
loading.  Framework code reads from this config instead of module-level
constants.

Two config levels:
  - ``AgentConfig``: framework-level (substrate-agnostic).
  - Substrate configs (e.g., ``realizations/tiamat/config.yaml``):
    loaded separately by the substrate, NOT by this module.

Usage::

    # From YAML file:
    config = AgentConfig.from_yaml("agent_config.yaml")
    agent = LAMAgent(model, config=config)

    # All defaults (no YAML needed):
    agent = LAMAgent(model)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, FrozenSet, Optional

# The text board's four design-decision axes that can each be switched off via
# scaffold_config["ablation"] (docs/spec/tiamat-ablation-flags.md). CP1 only
# validates/stamps this set; CP2/CP3 make the tree actually branch on it.
ABLATION_CHOICES: FrozenSet[str] = frozenset({"evaluator", "replan", "context", "planner"})


def validate_ablation(raw: Any) -> FrozenSet[str]:
    """Validate and normalize a ``scaffold_config["ablation"]`` value.

    ``None`` (absent) or an empty collection means today's agent, unchanged.
    Anything outside :data:`ABLATION_CHOICES` fails fast with a clear error
    instead of silently no-oping somewhere deep in the tree.
    """
    if raw is None:
        return frozenset()
    if isinstance(raw, str) or not isinstance(raw, (list, tuple, set, frozenset)):
        raise ValueError(f"ablation must be a list of strings, got {raw!r}")
    values = frozenset(raw)
    non_strings = sorted(repr(v) for v in values if not isinstance(v, str))
    if non_strings:
        raise ValueError(f"ablation entries must be strings, got {non_strings}")
    unknown = values - ABLATION_CHOICES
    if unknown:
        raise ValueError(f"unknown ablation flag(s) {sorted(unknown)}; expected a subset of {sorted(ABLATION_CHOICES)}")
    return values


@dataclass(frozen=True)
class PlannerConfig:
    """Planner gather + synthesis parameters."""

    # How many tool calls the planner can make during the information-
    # gathering phase before forcing plan synthesis.  Higher = more
    # thorough grounding, but more VLM calls and latency.
    max_gather_iterations: int = 10

    # Wall-clock time limit (seconds) for the gather phase.  On edge
    # VLMs (~2s/call), 30s allows ~15 calls.  Reduce for faster
    # planning at the cost of less grounding.
    gather_wall_time_budget: float = 30.0

    # Max times the planner can call read_skill_docs per gather phase.
    # Prevents doc-read spam loops.  1-5 range.
    max_skill_doc_reads: int = 3

    # Temperature for the plan synthesis VLM call.  0.0 (greedy) makes
    # plans reproducible across runs — combined with a fixed seed, the
    # same task produces the same plan.
    synthesis_temperature: float = 0.0

    # Temperature for the plan synthesis call on a replan re-entry (after
    # a REPLAN verdict or amend-cap escalation).  Higher than the normal
    # synthesis_temperature so the planner explores a different plan
    # shape instead of re-emitting the same failed plan.  Applied to
    # both gather and synthesis when replan_context is set.
    replan_temperature: float = 0.7

    # Max output tokens for plan synthesis.  Plans rarely exceed 2K
    # tokens; 4096 is a generous safety margin.
    synthesis_max_tokens: int = 4096

    # Structural fix (CP2): after this many consecutive empty syntheses, the
    # planner stops re-planning and installs a flat react-fallback plan (a single
    # catch-all milestone the actor works in an observe->reason->act loop).
    # Only consulted when AgentConfig.structural_fixes is True. Spec default N=2.
    empty_plan_fallback_threshold: int = 2


@dataclass(frozen=True)
class ActorConfig:
    """Actor ReAct loop parameters."""

    # When the conversation history exceeds this many tokens, older
    # entries are compacted (summarized).  Too low = frequent
    # compaction and information loss.  Too high = context window
    # overflow.  Depends on your model's context length.
    token_budget: int = 24000

    # After compaction, this many recent (tool_call, result) pairs
    # are kept verbatim — not summarized.  The actor needs recent
    # context to avoid repeating actions.
    keep_recent_steps: int = 5

    # Max times the actor can call read_skill_docs per skill
    # selection turn.  Prevents inner-loop doc-read spam.
    max_doc_reads: int = 3

    # History compaction strategy:
    #   "extractive" — rule-based parsing (fast, free, default)
    #   "llm" — LLM-based summarization (better quality, costs 1 call)
    compaction: str = "llm"

    # Per-turn reasoning mode. One of:
    #   "act"            — pure tool calling, no reasoning captured.
    #   "act_thinking"   — single call + <think> tokens (reasoning models).
    #   "react_single"   — single call, content=thought + tool_calls=action
    #                      atomically. Default. Works on any model.
    #                      Re-prompts once if either is missing.
    #   "react_two_step" — legacy: model.query() for thought + forced
    #                      tool_call (with <tool_call> XML strip).
    # Invalid values fail-fast at first select_action() call.
    reasoning_mode: str = "react_single"

    # Default temperature for actor skill-selection calls.  0.0 (greedy)
    # gives reproducible actions for a fixed conversation + seed.
    temperature: float = 0.0

    # Temperature for the actor's NEXT decision after the evaluator's AMEND
    # verdict injected an "EVALUATOR CORRECTION" message into history.
    # Higher than the base temperature so the actor samples a different
    # response instead of re-emitting the action that triggered amend.
    amend_temperature: float = 0.7

    # Actor-visible terminal failure text. Robot experiment profiles set this
    # once at startup; the framework default preserves the historical typed
    # transcript for every caller that does not opt into profiles.
    failure_presentation: str = "typed"


@dataclass(frozen=True)
class EvaluatorConfig:
    """Evaluator two-step parameters."""

    # Temperature for the verdict VLM call.  0.0 (greedy) keeps verdicts
    # reproducible — verdicts must not depend on sampling noise.
    verdict_temperature: float = 0.0

    # Max output tokens for the verdict response.  Needs room for
    # reasoning + correction text (if amend).  512 is sufficient
    # for most cases; increase to 1024 for verbose CoT.
    verdict_max_tokens: int = 512

    # Temperature for the evidence selection step (Step 1).  0.0 (greedy)
    # keeps verdicts reproducible — evidence selection is deterministic.
    evidence_temperature: float = 0.0

    # Max tokens for evidence selection.  Just outputs handle names
    # and brief reasoning — 512 is generous.
    evidence_max_tokens: int = 512


@dataclass(frozen=True)
class ModelConfig:
    """Default model backend parameters.

    These are defaults — individual calls (planner synthesis, evaluator
    verdict, etc.) override temperature and max_tokens per-call.
    """

    # Default max tokens per request.  The actor's tool-selection calls
    # use this default.  Override per-call where needed.
    max_tokens: int = 4096

    # Default sampling temperature.  The actor uses this for skill
    # selection; planner/evaluator override with their own values.
    temperature: float = 0.0

    # HTTP request timeout in seconds.  Increase for very large models
    # or slow network connections.
    timeout: int = 120

    # Optional vLLM sampling seed (None = let server randomize).
    # When set, every chat-completion payload includes "seed", pinning
    # the PRNG per request on a fixed server instance.
    seed: Optional[int] = None


@dataclass(frozen=True)
class AgentConfig:
    """Top-level agent configuration.

    Load from YAML::

        config = AgentConfig.from_yaml("agent_config.yaml")
        agent = LAMAgent(model, config=config)

    Or use defaults::

        agent = LAMAgent(model)  # all defaults
    """

    # Which system prompt variant to use.  Each variant is a complete
    # set of prompts for planner, actor, and evaluator.  The release
    # ships one variant, "tiamat_agent_submit_v5".
    prompt_variant: str = "tiamat_agent_submit_v5"

    # Max consecutive AMEND verdicts per milestone before the evaluator
    # auto-escalates to REPLAN.  Paper's N_max.  Higher = more retries
    # before giving up; lower = faster escalation to replanning.
    amend_cap: int = 2

    # Master gate for the LAM structural-fix bundle (docs/spec/lam-improvement.md).
    # Default False keeps every existing variant BYTE-IDENTICAL: the tree shape,
    # planner token budgets, and evaluator budgets are untouched. Set True ONLY
    # by the substrate harness for the new structural variant. When True:
    #   - PlanTask threads planner.synthesis_max_tokens into the synthesis call
    #     (kills the truncation → empty-plan → re-plan loop) and slims the schema;
    #   - PlanTask installs a react-fallback plan after N empty syntheses;
    #   - build_agent_tree() adds the react-fallback + all_done submit branches.
    # It never changes behavior on its own — every consumer reads this flag before
    # diverging from the frozen path.
    structural_fixes: bool = False

    # Ablation switches for the text board's four design-decision axes
    # (docs/spec/tiamat-ablation-flags.md). Values are validated against
    # ABLATION_CHOICES; absent/empty means today's agent, byte-identical. All
    # four are live: "evaluator"/"replan" are threaded by LAMAgent.create_tree
    # into build_agent_tree(), which branches the evaluate/verdict_route nodes
    # (CP2); "context"/"planner" are read directly off this config by
    # SelectSkill (bt/nodes/select_skill.py) and PlanTask (bt/nodes/plan_task.py)
    # respectively (CP3).
    ablation: FrozenSet[str] = field(default_factory=frozenset)

    # Default wall-time budget (seconds) per ActionSkill execution.
    # Individual skills can override via wall_time_budget_seconds
    # class attribute.  Navigation may need 300s; capture needs 60s.
    default_wall_time_budget: float = 120.0

    # Max times planner or actor can call read_skill_docs per phase.
    # Prevents doc-read spam loops.
    max_doc_reads: int = 3

    planner: PlannerConfig = field(default_factory=PlannerConfig)
    actor: ActorConfig = field(default_factory=ActorConfig)
    evaluator: EvaluatorConfig = field(default_factory=EvaluatorConfig)
    model: ModelConfig = field(default_factory=ModelConfig)

    def __post_init__(self) -> None:
        # Frozen dataclass: normalize/validate via object.__setattr__. Runs on
        # every construction (including dataclasses.replace), so an unknown
        # ablation flag fails fast wherever AgentConfig gets built.
        object.__setattr__(self, "ablation", validate_ablation(self.ablation))

    @classmethod
    def from_yaml(cls, path: str) -> AgentConfig:
        """Load config from a YAML file.  Missing keys use defaults."""
        import yaml

        with open(path) as f:
            raw = yaml.safe_load(f) or {}
        return cls.from_dict(raw)

    @classmethod
    def from_dict(cls, d: Dict[str, Any]) -> AgentConfig:
        """Build config from a nested dict.  Missing keys use defaults."""
        return cls(
            prompt_variant=d.get("prompt_variant", "tiamat_agent_submit_v5"),
            amend_cap=d.get("amend_cap", 2),
            structural_fixes=d.get("structural_fixes", False),
            ablation=d.get("ablation", ()),
            default_wall_time_budget=d.get("default_wall_time_budget", 120.0),
            max_doc_reads=d.get("max_doc_reads", 3),
            planner=PlannerConfig(
                **{k: v for k, v in d.get("planner", {}).items() if k in PlannerConfig.__dataclass_fields__}
            ),
            actor=ActorConfig(**{k: v for k, v in d.get("actor", {}).items() if k in ActorConfig.__dataclass_fields__}),
            evaluator=EvaluatorConfig(
                **{k: v for k, v in d.get("evaluator", {}).items() if k in EvaluatorConfig.__dataclass_fields__}
            ),
            model=ModelConfig(**{k: v for k, v in d.get("model", {}).items() if k in ModelConfig.__dataclass_fields__}),
        )
