"""DecisionContext + Decision — the simple agent contract.

This is the "plug your function here" interface. An agent is just:

    def my_agent(ctx: DecisionContext) -> Decision:
        # call your LLM, rule engine, or hardcoded logic
        return Decision(skill="navigate", params={"target": "kitchen"})

The framework handles everything else: execution, cancellation,
world state, skill lifecycle.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


@dataclass
class DecisionContext:
    """Everything the agent needs to make a decision.

    Built automatically from the blackboard each time a decision is needed.
    The agent function receives this as a read-only snapshot.
    """

    # Task
    task: str = ""
    scenario: str = ""

    # Available skills (OpenAI tool-call format, ready to pass to an LLM)
    available_skills: List[Dict[str, Any]] = field(default_factory=list)

    # Platform state (paper §5: s_robot) — opaque, substrate-specific
    platform_state: Dict[str, Any] = field(default_factory=dict)

    # History — what happened so far this task
    skill_history: List[Dict[str, Any]] = field(default_factory=list)
    last_skill_name: str = ""
    last_skill_observation: str = ""
    last_skill_outcome: str = ""  # "success" | "failed" | ""
    last_skill_reasoning: str = ""

    # Images injected by the previous skill (e.g. image_inject's bytes)
    pending_images: List[bytes] = field(default_factory=list)

    # Known objects (LLM-visible summaries from ObjectStore)
    known_objects: List[str] = field(default_factory=list)

    # Milestone info (if using milestone-based planning)
    milestone: Optional[Dict[str, Any]] = None
    milestone_index: int = 0
    total_milestones: int = 0


@dataclass
class Decision:
    """What the agent wants the robot to do.

    Return one of:
        Decision(skill="navigate", params={"target": "kitchen"})
        Decision(task_complete=True, reasoning="Object found and delivered")
        Decision(defer=True, reasoning="Can't reach target, will retry later")
    """

    # Skill to execute (mutually exclusive with task_complete / defer)
    skill: str = ""
    params: Dict[str, Any] = field(default_factory=dict)

    # Or signal completion / deferral
    task_complete: bool = False
    defer: bool = False

    # Explanation (optional, for logging / GUI)
    reasoning: str = ""
