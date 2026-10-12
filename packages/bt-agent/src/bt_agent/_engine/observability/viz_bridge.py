"""VizBridge — shared per-tick snapshot emitter for Foxglove visualization.

Emits 8 JSON channels to an :class:`EventSink` each tick, providing a
complete view of the agent's state (BT tree, milestones, skills,
conversation, object store). Used by both the TestHarness and the ROS2
bridge node.

Substrate-agnostic: reads only framework BB keys, never imports from
any realization package.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Dict, List

import py_trees
from bt_agent._engine.agent.blackboard_keys import BB
from bt_agent._engine.agent.interface import AgentEvent
from bt_agent._engine.agent.runner import AgentRunner
from bt_agent._engine.observability.event_sink import EventSink
from bt_agent._engine.observability.phase import derive_phase
from bt_agent._engine.observability.tick_snapshot import (
    attach_skill_subtree,
    snapshot_blackboard,
    snapshot_tree,
)

logger = logging.getLogger(__name__)

# BB keys the viz client needs read access to.
_VIZ_BB_KEYS = [
    BB.TASK,
    BB.SCENARIO,
    BB.PLATFORM_STATE,
    BB.MILESTONES,
    BB.CURRENT_MILESTONE_IDX,
    BB.COMPLETED_MILESTONES,
    BB.DEFERRED_MILESTONES,
    BB.PLAN_REASONING,
    BB.SELECTED_SKILL,
    BB.SKILL_PARAMS,
    BB.SKILL_HANDLE,
    BB.SKILL_REASONING,
    BB.COMPLETION_CLAIMED,
    BB.CONVERSATION,
    BB.CANCEL_REQUESTED,
    BB.CANCEL_SOURCE,
    BB.CANCEL_REASON,
    BB.TASK_COMPLETE,
    BB.TASK_SUBMITTED,
    BB.OBJECT_STORE,
]


class VizBridge:
    """Emit per-tick snapshots to an EventSink for Foxglove visualization.

    Create one instance after the agent and runner are set up, then call
    :meth:`on_tick` after each ``runner.tick()`` and :meth:`mirror_event`
    for each agent event.
    """

    def __init__(self, runner: AgentRunner, sink: EventSink) -> None:
        self._sink = sink
        self._runner = runner

        # Read-only BB client for snapshotting.
        self._bb = py_trees.blackboard.Client(name="VizBridge")
        for k in _VIZ_BB_KEYS:
            self._bb.register_key(key=k, access=py_trees.common.Access.READ)

    def mirror_event(self, event: AgentEvent) -> None:
        """Forward an AgentEvent to ``/agent/event_log``."""
        self._sink.emit("/agent/event_log", {"kind": event.kind, "data": event.data})

    def on_tick(self, tick_idx: int) -> None:
        """Emit all 8 snapshot channels. Call after ``runner.tick()``."""
        try:
            tree_snap = snapshot_tree(self._runner.tree.root)
        except Exception:  # noqa: BLE001
            logger.exception("snapshot_tree failed")
            return

        bb_snap = snapshot_blackboard(self._bb)

        # Splice running skill's internal subtree for panel rendering.
        skill_handle = self._bb_get(BB.SKILL_HANDLE)
        if skill_handle is not None:
            attach_skill_subtree(tree_snap, skill_handle)

        # Phase
        cancel = bool(bb_snap.get(BB.CANCEL_REQUESTED, False))
        done = bool(bb_snap.get(BB.TASK_COMPLETE, False))
        phase = derive_phase(tree_snap, cancel_requested=cancel, task_complete=done)

        skill_handle_view = bb_snap.get(BB.SKILL_HANDLE) or {}
        skill_name = bb_snap.get(BB.SELECTED_SKILL, "") or ""
        skill_params = bb_snap.get(BB.SKILL_PARAMS, {}) or {}

        # Milestones
        milestones_raw = self._bb_get(BB.MILESTONES, []) or []
        completed = self._bb_get(BB.COMPLETED_MILESTONES, set()) or set()
        deferred = self._bb_get(BB.DEFERRED_MILESTONES, []) or []
        cur_idx = self._bb_get(BB.CURRENT_MILESTONE_IDX, 0) or 0

        ms_items: List[Dict[str, Any]] = []
        for i, m in enumerate(milestones_raw):
            if i in completed:
                status = "done"
            elif i in deferred:
                status = "deferred"
            elif i == cur_idx:
                status = "running"
            else:
                status = "pending"
            ms_items.append({
                "id": m.get("id", ""),
                "description": m.get("description", ""),
                "status": status,
                "type": m.get("type", ""),
                "capture": bool(m.get("capture", False)),
                "capture_target": m.get("capture_target", ""),
                "dependencies": list(m.get("dependencies", [])),
            })

        # Object store
        store = self._bb_get(BB.OBJECT_STORE)
        os_handles: List[Dict[str, Any]] = []
        if store is not None:
            with contextlib.suppress(Exception):
                for handle, obj in getattr(store, "_objects", {}).items():
                    pos = list(getattr(obj, "viewpoint_position", None) or getattr(obj, "position", []) or [])
                    os_handles.append({
                        "handle": handle,
                        "label": getattr(obj, "label", "") or "",
                        "position": pos,
                        "is_exploration": bool(getattr(obj, "is_exploration", False)),
                    })

        # Conversation
        convo = self._bb_get(BB.CONVERSATION, []) or []

        # Emit all channels
        self._sink.emit(
            "/agent/phase",
            {
                "tick": tick_idx,
                "phase": phase,
                "task": bb_snap.get(BB.TASK, "") or "",
                "scenario": bb_snap.get(BB.SCENARIO, "") or "",
                "milestone_idx": cur_idx,
                "milestone_total": len(milestones_raw),
                "current_skill": skill_name,
            },
        )
        self._sink.emit("/agent/bt_snapshot", {"tick": tick_idx, "root": tree_snap})
        self._sink.emit(
            "/agent/skill",
            {
                "name": skill_name,
                "status": skill_handle_view.get("status", "IDLE"),
                "progress": float(skill_handle_view.get("progress", 0.0) or 0.0),
                "observation": skill_handle_view.get("observation", "") or "",
                "params": skill_params,
                "is_terminal": bool(skill_handle_view.get("is_terminal", False)),
            },
        )
        self._sink.emit("/agent/milestones", {"current_idx": cur_idx, "items": ms_items})
        self._sink.emit("/agent/conversation", {"milestone_idx": cur_idx, "messages": convo})
        self._sink.emit("/agent/blackboard", bb_snap)
        self._sink.emit("/agent/object_store", {"handles": os_handles})

    def _bb_get(self, key: str, default: Any = None) -> Any:
        try:
            return self._bb.get(key)
        except KeyError:
            return default
