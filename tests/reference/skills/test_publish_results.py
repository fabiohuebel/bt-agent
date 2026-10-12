"""Tests for the K-promotion hook (paper §5 Invariant I6).

Covers Step 10 of the paper-alignment refactor:

- ``Skill.publish_results`` is an overridable hook with a default no-op
  body. The framework calls it; skills decide what (if anything) to do.
- Framework code is **agnostic** to skill-specific field names — no
  ``if field == "new_objects"`` anywhere outside skill modules.
- Class-level audits verify that ``QuerySkill`` and ``CaptureSkill`` override
  the hook, and that the other shipped builtins do NOT (they don't need to).
- The default is no-op (paper-faithful ``R_K = ∅``): a stub skill that
  doesn't override the hook and returns K-bound-looking data writes nothing
  to the ObjectStore.
"""

from __future__ import annotations

from typing import Any, Dict
from unittest.mock import MagicMock

import py_trees
from bt_agent._engine.skills.object_store import ObjectStore
from bt_agent._engine.skills.protocol import (
    ContextSkill,
    Skill,
    SkillContext,
    SkillHandle,
    SkillStatus,
)
from bt_agent._engine.skills.object_store import SkillObject as SceneObject

# ---------------------------------------------------------------------------
# Stub skills
# ---------------------------------------------------------------------------


class _NoopNode(py_trees.behaviour.Behaviour):
    def update(self) -> py_trees.common.Status:
        return py_trees.common.Status.SUCCESS


class _StubNonPromotingSkill(ContextSkill):
    """Concrete Skill subclass that does NOT override ``publish_results``."""

    name = "stub_no_promote"
    description = "stub that returns K-bound-looking data but does not opt in"
    params: Dict[str, str] = {}

    def build_subtree(
        self, params: Dict[str, Any], context: SkillContext, handle: SkillHandle
    ) -> py_trees.behaviour.Behaviour:
        return _NoopNode(name=self.name)

    def tool_def(self) -> Dict[str, Any]:
        # Step 6.5: ContextSkill mandates an explicit tool_def override.
        return super().tool_def()


# ---------------------------------------------------------------------------
# 1–2. Default behavior (no override) is no-op
# ---------------------------------------------------------------------------


def test_skill_default_publish_results_is_noop() -> None:
    """Calling ``Skill.publish_results`` directly does NOT touch the store."""
    skill = _StubNonPromotingSkill()
    handle = SkillHandle()
    handle.result_data = {"new_objects": [SceneObject(handle="x", label="x")]}
    store = MagicMock(spec=ObjectStore)

    skill.publish_results(handle, store)

    store.add.assert_not_called()
    store.register_query.assert_not_called()


def test_publish_results_does_not_call_store_for_unoverriding_skill() -> None:
    """End-to-end: stub skill returns new_objects but ObjectStore stays empty."""
    skill = _StubNonPromotingSkill()
    handle = SkillHandle()
    handle.result_data = {
        "new_objects": [SceneObject(handle="red_mug_1", label="red mug")],
        "query_mappings": {"red mug": "red_mug_1"},
    }
    store = ObjectStore()

    skill.publish_results(handle, store)

    assert store.all_objects() == []
    assert store.get("red mug") is None


# ---------------------------------------------------------------------------
# 3–5. Class-level audits — which builtins override?
# ---------------------------------------------------------------------------


def test_query_skill_overrides_publish_results() -> None:
    from lam_v2_tiamat.bt_skills.query import QuerySkill

    assert QuerySkill.publish_results is not Skill.publish_results


def test_capture_skill_overrides_publish_results() -> None:
    from lam_v2_tiamat.bt_skills.capture import CaptureSkill

    assert CaptureSkill.publish_results is not Skill.publish_results


def test_non_promoting_builtins_inherit_default_noop() -> None:
    """NavigateSkill, SurroundSurveySkill, ImageInjectSkill don't write to K."""
    from lam_v2_tiamat.bt_skills.image_inject import ImageInjectSkill
    from lam_v2_tiamat.bt_skills.navigate import NavigateSkill
    from lam_v2_tiamat.bt_skills.surround_survey import SurroundSurveySkill

    assert NavigateSkill.publish_results is Skill.publish_results
    assert SurroundSurveySkill.publish_results is Skill.publish_results
    assert ImageInjectSkill.publish_results is Skill.publish_results


# ---------------------------------------------------------------------------
# 6–7. Builtin overrides actually write to the store
# ---------------------------------------------------------------------------


def test_query_promote_writes_new_objects_and_query_mappings() -> None:
    from lam_v2_tiamat.bt_skills.query import QuerySkill

    skill = QuerySkill(scene_graph_agent=MagicMock())
    handle = SkillHandle()
    obj = SceneObject(handle="red_mug_1", label="red mug")
    handle.result_data = {
        "new_objects": [obj],
        "query_mappings": {"red mug": "red_mug_1"},
    }
    store = ObjectStore()

    skill.publish_results(handle, store)

    assert any(o.handle == "red_mug_1" for o in store.all_objects())
    # Resolution by query text proves register_query was called
    assert store.get("red mug") is not None
    assert store.get("red mug").handle == "red_mug_1"


def test_capture_promote_writes_new_objects() -> None:
    from lam_v2_tiamat.bt_skills.capture import CaptureSkill

    skill = CaptureSkill(model=MagicMock(), pointing_pipeline=None)
    handle = SkillHandle()
    capture_obj = SceneObject(handle="cap_red_mug_3", label="captured red mug")
    handle.result_data = {"new_objects": [capture_obj]}
    store = ObjectStore()

    skill.publish_results(handle, store)

    assert any(o.handle == "cap_red_mug_3" for o in store.all_objects())


# ---------------------------------------------------------------------------
# 8–9. Defensive against empty / partial result_data
# ---------------------------------------------------------------------------


def test_publish_results_handles_empty_result_data() -> None:
    from lam_v2_tiamat.bt_skills.query import QuerySkill

    skill = QuerySkill(scene_graph_agent=MagicMock())
    handle = SkillHandle()
    # result_data defaults to {}
    store = ObjectStore()

    skill.publish_results(handle, store)

    assert store.all_objects() == []


def test_publish_results_handles_missing_fields() -> None:
    """``observation`` only, no new_objects or query_mappings — no error, no writes."""
    from lam_v2_tiamat.bt_skills.query import QuerySkill

    skill = QuerySkill(scene_graph_agent=MagicMock())
    handle = SkillHandle()
    handle.result_data = {"observation": "Found nothing."}
    store = ObjectStore()

    skill.publish_results(handle, store)

    assert store.all_objects() == []


# ---------------------------------------------------------------------------
# 10. Integration: ExecuteSkill._apply_results calls the hook
# ---------------------------------------------------------------------------


def test_execute_skill_apply_results_invokes_promote_hook() -> None:
    """ExecuteSkill._apply_results delegates K-promotion to the skill's hook."""
    from bt_agent._engine.bt.nodes.execute_skill import ExecuteSkill

    # Skill that records when its promote hook is called.
    call_log: list[tuple[Any, Any]] = []

    class _RecordingSkill(ContextSkill):
        name = "recording"
        description = "records promote calls"
        params: Dict[str, str] = {}

        def build_subtree(self, params, context, handle):
            return _NoopNode(name=self.name)

        def tool_def(self):
            return super().tool_def()

        def publish_results(self, handle, store):
            call_log.append((handle, store))

    skill = _RecordingSkill()
    node = ExecuteSkill()
    node._skill = skill  # bypass initialise() — direct injection
    node._handle = SkillHandle()
    node._handle.status = SkillStatus.SUCCEEDED
    node._handle.result_data = {"observation": "ok"}

    # Inject a store via the BB.OBJECT_STORE key so _bb_get returns it
    store = ObjectStore()
    node.blackboard.set("object_store", store)

    node._apply_results()

    assert len(call_log) == 1
    received_handle, received_store = call_log[0]
    assert received_handle is node._handle
    assert received_store is store


# ---------------------------------------------------------------------------
# 11. Framework agnosticism: no field-name dispatch in framework code
# ---------------------------------------------------------------------------


def test_framework_code_does_not_reference_skill_specific_field_names() -> None:
    """Self-check: TIAMAT-specific field names (``new_objects`` / ``query_mappings``)
    must not appear in framework code outside the skill modules themselves.

    This is a code-style invariant — if a future change re-introduces
    ``if field == "new_objects"`` in execute_skill.py or planner.py, this
    test fails.
    """
    from pathlib import Path

    framework_files = [
        Path(__file__).resolve().parents[3] / "packages/bt-agent/src/bt_agent/_engine/bt/nodes/execute_skill.py",
        Path(__file__).resolve().parents[3] / "packages/bt-agent/src/bt_agent/_engine/skills/protocol.py",
    ]

    forbidden = ('"new_objects"', "'new_objects'", '"query_mappings"', "'query_mappings'")

    for fp in framework_files:
        text = fp.read_text()
        for needle in forbidden:
            assert needle not in text, (
                f"Framework file {fp.name} references skill-specific field "
                f"name {needle}. K-promotion vocabulary belongs to skills "
                "(via publish_results override), not the framework."
            )
