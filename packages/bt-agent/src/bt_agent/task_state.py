"""Task-level state for the planner-free runtime; no implicit plan."""

from dataclasses import dataclass, field
from enum import Enum
from typing import Any

from .skills import ObjectStore


class TaskStatus(str, Enum):
    ACTIVE = "active"
    SUBMITTING = "submitting"
    SUBMITTED = "submitted"
    CANCELLED = "cancelled"


@dataclass(frozen=True)
class TaskState:
    """Immutable task snapshot. Submission records termination, not success."""

    instruction: str
    scenario: str = ""
    status: TaskStatus = TaskStatus.ACTIVE
    completion_reason: str = ""
    cancellation_reason: str = ""


def _paper_task_view(task: TaskState) -> dict:
    """Transient input for frozen prompts/skills, never stored as a plan.

    Keep the old wire wording until a separately versioned prompt migration.
    A fresh dict prevents legacy consumers from mutating authoritative state.
    """
    return {
        "id": "fallback_task",
        "description": task.instruction,
        "detailed_instructions": task.instruction,
        "verify_condition": "Is the task complete?",
        "type": "direct",
        "dependencies": [],
    }


@dataclass(frozen=True)
class TaskContext:
    """Skill input with an explicit task and live host state.

    New skills use ``task``. Legacy skill properties are read-only projections;
    there is no milestone list, cursor, or advancement lifecycle behind them.
    """

    task: TaskState
    object_store: ObjectStore | None = None
    skill_call_id: str = ""
    platform_state: dict[str, Any] = field(default_factory=dict)

    @property
    def main_task(self):
        return self.task.instruction

    @property
    def scenario(self):
        return self.task.scenario

    @property
    def milestone(self):
        return _paper_task_view(self.task)

    @property
    def milestone_index(self):
        return 0
