"""Snapshot helpers — render BT and blackboard state into JSON-safe dicts.

Called by the harness once per tick. The output is fed straight into
``EventSink.emit`` for downstream consumption (Foxglove channels, MCAP).

Both functions are tolerant of partially-set blackboards and unserializable
values; they never raise on missing keys or weird types.
"""

from __future__ import annotations

import contextlib
import logging
from typing import Any, Dict, Iterable, List, Optional

import py_trees
from bt_agent._engine.agent.blackboard_keys import BB

logger = logging.getLogger(__name__)


# Curated set of blackboard keys to include in per-tick snapshots.
# Excludes: PLATFORM_STATE (may contain large scene graphs, camera bytes,
# numpy arrays), shared services (SKILL_REGISTRY, MODEL_BACKEND, OBJECT_STORE
# — those go into other channels), CONVERSATION (its own channel).
DEFAULT_BB_KEYS: List[str] = [
    BB.TASK,
    BB.SCENARIO,
    BB.CURRENT_MILESTONE_IDX,
    BB.COMPLETED_MILESTONES,
    BB.DEFERRED_MILESTONES,
    BB.PLAN_REASONING,
    BB.SELECTED_SKILL,
    BB.SKILL_PARAMS,
    BB.SKILL_REASONING,
    BB.COMPLETION_CLAIMED,
    BB.CANCEL_REQUESTED,
    BB.CANCEL_SOURCE,
    BB.CANCEL_REASON,
    BB.TASK_COMPLETE,
    BB.TASK_SUBMITTED,
]


_STATUS_NAMES = {
    py_trees.common.Status.SUCCESS: "SUCCESS",
    py_trees.common.Status.FAILURE: "FAILURE",
    py_trees.common.Status.RUNNING: "RUNNING",
    py_trees.common.Status.INVALID: "INVALID",
}


def _status_str(status: Any) -> str:
    return _STATUS_NAMES.get(status, str(status))


def _node_type(node: py_trees.behaviour.Behaviour) -> str:
    cls = type(node)
    # Common composite types first (cheap, exact name).
    name = cls.__name__
    # Map py_trees built-ins to friendlier labels.
    if name in {"Sequence", "Selector", "Parallel"}:
        return name
    return "Behaviour"


def snapshot_tree(root: py_trees.behaviour.Behaviour) -> Dict[str, Any]:
    """Walk a py_trees tree and return a JSON-safe snapshot.

    The ``tip`` flag marks the leaf currently RUNNING (per ``root.tip()``).
    """
    try:
        tip_node: Optional[py_trees.behaviour.Behaviour] = root.tip()
    except Exception:  # noqa: BLE001
        tip_node = None

    tip_id = id(tip_node) if tip_node is not None else None

    def _walk(node: py_trees.behaviour.Behaviour) -> Dict[str, Any]:
        return {
            "name": node.name,
            "type": _node_type(node),
            "status": _status_str(node.status),
            "feedback": getattr(node, "feedback_message", "") or "",
            "tip": id(node) == tip_id,
            "children": [_walk(c) for c in getattr(node, "children", [])],
        }

    return _walk(root)


# ---- OBS-DROP-BEGIN ----
def attach_skill_subtree(tree_snap: Dict[str, Any], skill_handle: Any) -> None:
    """Splice a running skill's BT subtree under the ExecuteSkill node.

    The skill subtree runs in its own background thread and is not part of
    the main BT. This makes it visible by reading the subtree root off the
    skill handle (set in ``Skill.start()``) and snapshotting it as a child
    of the ``ExecuteSkill`` node.

    Mutates ``tree_snap`` in place. Idempotent — only attaches when the
    main-BT ``ExecuteSkill`` node is RUNNING and the handle exposes a
    subtree. Skipped when no skill is active.

    Reading skill-thread-owned node statuses from the main thread is racy
    by py_trees' design, but py_trees uses simple atomic attribute writes
    and we accept slightly stale reads in exchange for not adding locks.
    """
    if skill_handle is None:
        return
    subtree_root = getattr(skill_handle, "subtree", None)
    if subtree_root is None:
        return

    def _splice(node: Dict[str, Any]) -> bool:
        if node.get("name") == "ExecuteSkill" and node.get("status") == "RUNNING":
            try:
                sub_snap = snapshot_tree(subtree_root)
            except Exception:  # noqa: BLE001
                logger.exception("attach_skill_subtree: snapshot_tree failed")
                return True
            existing = node.get("children") or []
            node["children"] = [*existing, sub_snap]
            # ExecuteSkill is no longer a leaf — the real tip is inside the
            # spliced subtree. Clear its tip flag to avoid two glowing nodes.
            node["tip"] = False
            return True
        return any(_splice(child) for child in node.get("children", []))

    _splice(tree_snap)


# ---- OBS-DROP-END ----


def _safe_value(key: str, value: Any) -> Any:
    """Make a single value JSON-safe."""
    # SkillHandle -> small dict
    if hasattr(value, "is_terminal") and hasattr(value, "status") and hasattr(value, "progress"):
        try:
            status = value.status
            status_name = status.name if hasattr(status, "name") else str(status)
            return {
                "status": status_name,
                "progress": float(value.progress),
                "observation": str(value.observation or ""),
                "is_terminal": bool(value.is_terminal),
                "is_cancel_requested": bool(value.is_cancel_requested),
            }
        except Exception:  # noqa: BLE001
            return {"status": "UNKNOWN"}

    if isinstance(value, set):
        return sorted(value, key=lambda x: (str(type(x)), str(x)))

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    if isinstance(value, (list, tuple)):
        return [_safe_value(key, v) for v in value]

    if isinstance(value, dict):
        return {str(k): _safe_value(key, v) for k, v in value.items()}

    # Fallback: stringify
    return str(value)


def snapshot_blackboard(
    bb: py_trees.blackboard.Client,
    keys: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """Read selected BB keys and return a JSON-safe dict.

    Missing keys are skipped (no error). Unserializable values are coerced
    or stringified. Designed to be cheap enough to call every tick.
    """
    keys = list(keys) if keys is not None else list(DEFAULT_BB_KEYS)
    out: Dict[str, Any] = {}
    for k in keys:
        try:
            value = bb.get(k)
        except KeyError:
            continue
        except (AttributeError, Exception) as e:  # noqa: BLE001
            # AttributeError fires when the client lacks read access to the
            # key. Treat it the same as missing — the snapshotter is not the
            # place to enforce client permissions.
            if isinstance(e, AttributeError):
                continue
            out[k] = f"<read error: {e}>"
            continue
        out[k] = _safe_value(k, value)

    # Always include a small skill_handle digest if present and accessible.
    handle = None
    with contextlib.suppress(KeyError, AttributeError):
        handle = bb.get(BB.SKILL_HANDLE)
    if handle is not None:
        out[BB.SKILL_HANDLE] = _safe_value(BB.SKILL_HANDLE, handle)

    return out
