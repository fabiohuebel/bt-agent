"""AdvanceMilestone — mark the current milestone as completed and advance.

This is a simple, instant node that runs after EvaluateCompletion succeeds.
It updates the milestone tracking state on the blackboard.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Set

import py_trees
from bt_agent._engine.agent.blackboard_keys import BB


def _next_executable(
    milestones: List[Dict[str, Any]],
    start: int,
    completed: Set[int],
) -> Optional[int]:
    """Find the first milestone at or after ``start`` whose deps are all met.

    Returns the index, or ``None`` if no executable milestone remains.
    """
    for i in range(start, len(milestones)):
        if i in completed:
            continue
        deps = milestones[i].get("dependencies", [])
        if not deps:
            return i  # no deps — always executable
        dep_indices = {j for j, m in enumerate(milestones) if m["id"] in deps}
        if dep_indices.issubset(completed):
            return i
    return None


class AdvanceMilestone(py_trees.behaviour.Behaviour):
    """Mark current milestone done, advance index to next uncompleted one.

    Always returns SUCCESS. After this node runs, either:
    - CheckMilestonesRemain will find the next milestone, or
    - CheckAllMilestonesDone will detect we're finished.
    """

    def __init__(self) -> None:
        super().__init__(name="AdvanceMilestone")
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=BB.MILESTONES, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.CURRENT_MILESTONE_IDX, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.COMPLETED_MILESTONES, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.COMPLETION_CLAIMED, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.SKILL_HANDLE, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.PENDING_EVENTS, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.CONVERSATION, access=py_trees.common.Access.WRITE)
        # Step 3: amend bookkeeping is per-milestone — reset on advance.
        self.blackboard.register_key(key=BB.AMEND_COUNT, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.AMEND_NOTE, access=py_trees.common.Access.WRITE)

    def update(self) -> py_trees.common.Status:
        milestones = self.blackboard.get(BB.MILESTONES)
        idx = self.blackboard.get(BB.CURRENT_MILESTONE_IDX)
        completed = self.blackboard.get(BB.COMPLETED_MILESTONES)

        # Mark current milestone as done
        completed.add(idx)
        self.blackboard.set(BB.COMPLETED_MILESTONES, completed)

        # Reset per-milestone state
        self.blackboard.set(BB.COMPLETION_CLAIMED, False)
        self.blackboard.set(BB.SKILL_HANDLE, None)
        self.blackboard.set(BB.CONVERSATION, [])
        self.blackboard.set(BB.AMEND_COUNT, 0)
        self.blackboard.set(BB.AMEND_NOTE, None)

        # Advance to the next executable milestone (deps satisfied).
        # Milestones whose dependencies are not yet in the completed set
        # are skipped — the DAG controls execution order.
        resolved_next = _next_executable(milestones, idx + 1, completed)
        next_idx = resolved_next if resolved_next is not None else len(milestones)
        self.blackboard.set(BB.CURRENT_MILESTONE_IDX, next_idx)

        # Emit event
        events = self._bb_get(BB.PENDING_EVENTS, [])
        events.append({
            "kind": "milestone_advanced",
            "data": {
                "completed_idx": idx,
                "next_idx": next_idx,
                "total": len(milestones),
                "all_done": next_idx >= len(milestones),
            },
        })
        self.blackboard.set(BB.PENDING_EVENTS, events)

        self.feedback_message = f"Milestone {idx} done, next: {next_idx}/{len(milestones)}"
        return py_trees.common.Status.SUCCESS

    def _bb_get(self, key: str, default: Any = None) -> Any:
        try:
            return self.blackboard.get(key)
        except KeyError:
            return default
