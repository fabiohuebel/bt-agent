"""Derive a single high-level phase label from the BT tip.

Foxglove's Indicator panel and our custom BT panel both want a single
``phase`` enum (PLANNING / ACTING / EXECUTING / EVALUATING / IDLE / ...).
The BT tip name is the cheapest place to read this — no need to track
state machine history.
"""

from __future__ import annotations

from typing import Any, Dict

_NODE_TO_PHASE = {
    "PlanTask": "PLANNING",
    "SelectSkill": "ACTING",
    "ExecuteSkill": "EXECUTING",
    "EvaluateCompletion": "EVALUATING",
    "AdvanceMilestone": "ADVANCING",
    "HandleDeferredOrSubmit": "ADVANCING",
}


def derive_phase(bt_snapshot: Dict[str, Any], cancel_requested: bool, task_complete: bool) -> str:
    """Walk the snapshot, return the phase name based on the tip node.

    Falls back to IDLE if no tip is set (tree not yet ticked or all
    children INVALID).
    """
    if cancel_requested:
        return "CANCELLED"
    if task_complete:
        return "DONE"

    def _find_tip(node: Dict[str, Any]) -> Dict[str, Any] | None:
        if node.get("tip"):
            return node
        for child in node.get("children", []):
            r = _find_tip(child)
            if r is not None:
                return r
        return None

    tip = _find_tip(bt_snapshot)
    if tip is None:
        return "IDLE"
    return _NODE_TO_PHASE.get(tip["name"], "IDLE")
