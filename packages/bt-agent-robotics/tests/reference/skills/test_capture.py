"""Red-phase coverage for malformed VLM output in capture leaves."""

from __future__ import annotations

from typing import Any

import py_trees
from bt_agent._engine.skills.protocol import SkillHandle
from bt_agent_robotics.skills.capture import _AnalyzeView, _RefinePointing


class _StubModel:
    def __init__(self, result: Any) -> None:
        self._result = result

    def query_structured(self, *args: Any, **kwargs: Any) -> Any:
        return self._result


def _tick(node: py_trees.behaviour.Behaviour) -> py_trees.common.Status:
    for _ in node.tick():
        pass
    return node.status


def _params() -> dict[str, Any]:
    return {
        "target_label": "red mug",
        "camera_view": "frontleft",
        "image_bytes": b"jpeg bytes",
    }


def test_analyze_view_rejects_non_dict_result_via_analysis_failure() -> None:
    handle = SkillHandle()
    node = _AnalyzeView(_params(), handle, {}, _StubModel(["x"]))

    status = _tick(node)

    assert status == py_trees.common.Status.FAILURE
    assert "VLM analysis failed" in handle.observation


def test_analyze_view_rejects_malformed_bbox_via_analysis_failure() -> None:
    handle = SkillHandle()
    result = {"visible": True, "bbox_1000": "n/a", "reasoning": "r"}
    node = _AnalyzeView(_params(), handle, {}, _StubModel(result))

    status = _tick(node)

    assert status == py_trees.common.Status.FAILURE
    assert "VLM analysis failed" in handle.observation


def test_refine_pointing_keeps_last_valid_bbox_on_non_dict_result() -> None:
    handle = SkillHandle()
    original_bbox = [100, 200, 300, 400]
    state = {"bbox_1000": original_bbox.copy()}
    node = _RefinePointing(_params(), handle, state, _StubModel(["x"]))

    status = _tick(node)

    assert status == py_trees.common.Status.SUCCESS
    assert state["bbox_1000"] == original_bbox
