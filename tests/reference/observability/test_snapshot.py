"""Tests for snapshot_tree and snapshot_blackboard."""

from __future__ import annotations

import py_trees
from bt_agent._engine.agent.blackboard_keys import BB
from bt_agent._engine.observability.tick_snapshot import snapshot_blackboard, snapshot_tree


def _build_tree() -> py_trees.behaviour.Behaviour:
    """Sequence(Success, Running, Failure)."""
    root = py_trees.composites.Sequence(name="root", memory=False)
    root.add_child(py_trees.behaviours.Success(name="ok"))
    root.add_child(py_trees.behaviours.Running(name="busy"))
    root.add_child(py_trees.behaviours.Failure(name="bad"))
    return root


def test_snapshot_tree_basic() -> None:
    root = _build_tree()
    snap = snapshot_tree(root)
    assert snap["name"] == "root"
    assert snap["type"] == "Sequence"
    assert len(snap["children"]) == 3
    assert [c["name"] for c in snap["children"]] == ["ok", "busy", "bad"]


def test_snapshot_tree_status_and_tip() -> None:
    root = _build_tree()
    # Tick to populate statuses.
    for _ in root.tick():
        pass
    snap = snapshot_tree(root)
    # The middle child is Running, so it should be the tip.
    children = snap["children"]
    tips = [c for c in children if c["tip"]]
    assert len(tips) == 1
    assert tips[0]["name"] == "busy"
    assert tips[0]["status"] == "RUNNING"
    # First child succeeded.
    assert children[0]["status"] == "SUCCESS"


def test_snapshot_blackboard_skips_missing() -> None:
    bb = py_trees.blackboard.Client(name="test_bb_missing")
    bb.register_key(key=BB.TASK, access=py_trees.common.Access.WRITE)
    bb.set(BB.TASK, "find the cup")
    snap = snapshot_blackboard(bb, keys=[BB.TASK, BB.SCENARIO])
    assert snap[BB.TASK] == "find the cup"
    # SCENARIO was not set — must be absent, not raise.
    assert BB.SCENARIO not in snap


def test_snapshot_blackboard_serializes_set() -> None:
    bb = py_trees.blackboard.Client(name="test_bb_set")
    bb.register_key(key=BB.COMPLETED_MILESTONES, access=py_trees.common.Access.WRITE)
    bb.set(BB.COMPLETED_MILESTONES, {2, 0, 1})
    snap = snapshot_blackboard(bb, keys=[BB.COMPLETED_MILESTONES])
    assert snap[BB.COMPLETED_MILESTONES] == [0, 1, 2]


def test_snapshot_blackboard_handle_view() -> None:
    """SkillHandle should be coerced to a small JSON-safe digest."""
    from bt_agent._engine.skills.protocol import SkillHandle, SkillStatus

    handle = SkillHandle()
    handle.status = SkillStatus.RUNNING
    handle.progress = 0.42
    handle.observation = "halfway there"

    bb = py_trees.blackboard.Client(name="test_bb_handle")
    bb.register_key(key=BB.SKILL_HANDLE, access=py_trees.common.Access.WRITE)
    bb.set(BB.SKILL_HANDLE, handle)

    snap = snapshot_blackboard(bb, keys=[])  # SKILL_HANDLE auto-included
    assert BB.SKILL_HANDLE in snap
    h = snap[BB.SKILL_HANDLE]
    assert h["status"] == "RUNNING"
    assert h["progress"] == 0.42
    assert h["observation"] == "halfway there"
    assert h["is_terminal"] is False


def test_snapshot_blackboard_handles_unset_keys() -> None:
    """When the requested keys aren't on the blackboard, they're omitted."""
    bb = py_trees.blackboard.Client(name="test_bb_unset")
    # Ask for keys we never registered or set on this client; they should not
    # appear in the snapshot. (Other tests may have set BB.TASK on the global
    # blackboard, but BB.NAV_GRAPH isn't touched anywhere in this module.)
    snap = snapshot_blackboard(bb, keys=[BB.NAV_GRAPH])
    assert BB.NAV_GRAPH not in snap
