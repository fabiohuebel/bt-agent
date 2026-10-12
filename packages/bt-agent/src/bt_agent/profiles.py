"""Explicit supported profiles; no implicit simulator/environment selection.

``task-core-v1`` (the default) is the maintained runtime. ``paper-core-v1`` is a
research snapshot: the controller as it ran for the paper and the hardware
binding, kept for reproduction. It is unpolished and keeps its historical
structure and limitations.
"""

from ._engine.agent.config import ActorConfig, AgentConfig, EvaluatorConfig, PlannerConfig

DEFAULT_RUNTIME_PROFILE = "task-core-v1"


def core_config(profile: str = DEFAULT_RUNTIME_PROFILE) -> AgentConfig:
    """Frozen model policy, shared by the default task runtime and the paper snapshot."""
    if profile not in {"paper-core-v1", "task-core-v1"}:
        raise ValueError(
            f"Unsupported profile: {profile!r}; expected 'paper-core-v1' or 'task-core-v1'"
        )
    return AgentConfig(
        prompt_variant="tiamat_agent_submit_v5",
        structural_fixes=True,
        ablation=frozenset({"planner", "evaluator"}),
        actor=ActorConfig(reasoning_mode="react_single", failure_presentation="typed"),
        planner=PlannerConfig(synthesis_max_tokens=16384, replan_temperature=0.2),
        evaluator=EvaluatorConfig(evidence_max_tokens=2048, verdict_max_tokens=2048),
    )
