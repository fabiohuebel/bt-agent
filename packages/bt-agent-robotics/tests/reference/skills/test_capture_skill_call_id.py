"""physics-trace-recording wave 2, item D: skill_call_id reaches the capture
submission artifact (CaptureResult), carried via SkillContext.skill_call_id.
"""

from __future__ import annotations

import py_trees
from bt_agent._engine.skills.protocol import SkillContext, SkillHandle
from bt_agent_robotics.skills.capture import _FinalizeCapture


def _tick(node: py_trees.behaviour.Behaviour) -> py_trees.common.Status:
    for _ in node.tick():
        pass
    return node.status


def _params() -> dict:
    return {
        "target_label": "red mug",
        "camera_view": "frontleft",
        "image_bytes": b"jpeg bytes",
    }


def test_finalize_capture_carries_skill_call_id_from_context_into_the_result() -> None:
    handle = SkillHandle()
    context = SkillContext(skill_call_id="skillcall-123")
    node = _FinalizeCapture(_params(), handle, {"bbox_1000": None, "segment_result": None}, context)

    status = _tick(node)

    assert status == py_trees.common.Status.SUCCESS
    new_objects = handle.result_data["new_objects"]
    assert len(new_objects) == 1
    assert new_objects[0].skill_call_id == "skillcall-123"


def test_finalize_capture_defaults_to_empty_skill_call_id_when_context_has_none() -> None:
    handle = SkillHandle()
    context = SkillContext()  # skill_call_id defaults to ""
    node = _FinalizeCapture(_params(), handle, {"bbox_1000": None, "segment_result": None}, context)

    _tick(node)

    assert handle.result_data["new_objects"][0].skill_call_id == ""
