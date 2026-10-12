"""Tests for SkillRegistry effect-type partition (paper §5, Invariant I2).

Covers Step 1 of the paper-alignment refactor: every Skill must declare an
``effect_type`` (Σ_ctx or Σ_act); the registry partitions by effect type and
exposes role-aware accessors.
"""

from __future__ import annotations

from typing import Any, Dict

import py_trees
import pytest
from bt_agent._engine.skills.protocol import (
    ActionSkill,
    ContextSkill,
    EffectType,
    Skill,
    SkillContext,
    SkillHandle,
)
from bt_agent._engine.skills.registry import SkillRegistry

# ---------------------------------------------------------------------------
# Stub skills used to exercise registry behavior in isolation
# ---------------------------------------------------------------------------


class _NoopNode(py_trees.behaviour.Behaviour):
    def update(self) -> py_trees.common.Status:
        return py_trees.common.Status.SUCCESS


class _StubContextSkill(ContextSkill):
    name = "stub_ctx"
    description = "stub context skill"
    params: Dict[str, str] = {}

    def build_subtree(
        self, params: Dict[str, Any], context: SkillContext, handle: SkillHandle
    ) -> py_trees.behaviour.Behaviour:
        return _NoopNode(name=self.name)

    def tool_def(self) -> Dict[str, Any]:
        # Step 6.5: ContextSkill mandates an explicit tool_def override.
        return super().tool_def()


class _StubActionSkill(ActionSkill):
    name = "stub_act"
    description = "stub action skill (no prerequisites)"
    params: Dict[str, str] = {}

    def build_subtree(
        self, params: Dict[str, Any], context: SkillContext, handle: SkillHandle
    ) -> py_trees.behaviour.Behaviour:
        return _NoopNode(name=self.name)

    def precondition(self, state: SkillContext):
        return True, ""

    def safeguard_invariant(self, state: SkillContext, handle: SkillHandle) -> bool:
        return True


class _StubGatedActionSkill(ActionSkill):
    name = "stub_gated_act"
    description = "stub action skill with a custom is_available gate"
    params: Dict[str, str] = {}

    def build_subtree(
        self, params: Dict[str, Any], context: SkillContext, handle: SkillHandle
    ) -> py_trees.behaviour.Behaviour:
        return _NoopNode(name=self.name)

    def is_available(self, state, context):
        if not state.get("gate_open"):
            return False, "BLOCKED: gate_open not set"
        return True, f"{self.name} available"

    def precondition(self, state: SkillContext):
        return True, ""

    def safeguard_invariant(self, state: SkillContext, handle: SkillHandle) -> bool:
        return True


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


def test_register_routes_skills_to_correct_bucket() -> None:
    registry = SkillRegistry()
    ctx = _StubContextSkill()
    act = _StubActionSkill()

    registry.register(ctx)
    registry.register(act)

    assert ctx.name in registry._ctx_skills
    assert act.name in registry._act_skills
    assert ctx.name not in registry._act_skills
    assert act.name not in registry._ctx_skills


def test_get_returns_skills_from_either_bucket() -> None:
    registry = SkillRegistry()
    ctx = _StubContextSkill()
    act = _StubActionSkill()
    registry.register(ctx)
    registry.register(act)

    assert registry.get(ctx.name) is ctx
    assert registry.get(act.name) is act
    assert registry.get("does_not_exist") is None


def test_subclass_without_effect_type_raises_typeerror() -> None:
    """Concrete Skill subclasses without ``effect_type`` are rejected at class definition."""
    with pytest.raises(TypeError, match="effect_type"):

        class _BrokenSkill(Skill):  # noqa: F841 — definition itself must raise
            name = "broken"
            description = "missing effect_type"
            params: Dict[str, str] = {}

            # NOTE: no `effect_type = ...` declared.

            def build_subtree(
                self,
                params: Dict[str, Any],
                context: SkillContext,
                handle: SkillHandle,
            ) -> py_trees.behaviour.Behaviour:
                return _NoopNode(name=self.name)


def test_get_context_tools_returns_all_ctx_unfiltered() -> None:
    """Σ_ctx tools are always available; no filtering applied."""
    registry = SkillRegistry()
    registry.register(_StubContextSkill())
    registry.register(_StubActionSkill())

    ctx_tools = registry.get_context_tools()

    assert len(ctx_tools) == 1
    assert ctx_tools[0]["function"]["name"] == "stub_ctx"


def test_get_action_skills_masked_respects_is_available() -> None:
    """Σ_act tools are gated by ``is_available()`` (history) — precondition trivial here."""
    registry = SkillRegistry()
    registry.register(_StubActionSkill())  # ungated (default is_available → True)
    registry.register(_StubGatedActionSkill())  # custom is_available gate

    state = SkillContext()  # both stubs have trivial preconditions returning True

    # Gate closed — gated skill blocked, ungated skill passes.
    tools = registry.get_action_skills_masked(state, {}, {})
    names = sorted(t["function"]["name"] for t in tools)
    assert names == ["stub_act"]

    # Gate open — both pass.
    tools = registry.get_action_skills_masked(state, {"gate_open": True}, {})
    names = sorted(t["function"]["name"] for t in tools)
    assert names == ["stub_act", "stub_gated_act"]


def test_register_rejects_skill_with_invalid_effect_type() -> None:
    """A skill instance with a non-EffectType ``effect_type`` is rejected at register time."""
    registry = SkillRegistry()
    bogus = _StubContextSkill()
    bogus.effect_type = "not an enum"  # type: ignore[assignment]

    with pytest.raises(TypeError, match="invalid effect_type"):
        registry.register(bogus)


# ---------------------------------------------------------------------------
# Audit: every shipped builtin declares the right effect_type
# ---------------------------------------------------------------------------


def test_builtin_skills_have_expected_effect_types() -> None:
    """Class-level audit (no instantiation, so no dependency mocks needed)."""
    from bt_agent_robotics.skills.capture import CaptureSkill
    from bt_agent_robotics.skills.image_inject import ImageInjectSkill
    from bt_agent_robotics.skills.navigate import NavigateSkill
    from bt_agent_robotics.skills.query import QuerySkill
    from bt_agent_robotics.skills.surround_survey import SurroundSurveySkill

    assert QuerySkill.effect_type == EffectType.CONTEXT
    assert SurroundSurveySkill.effect_type == EffectType.CONTEXT
    assert ImageInjectSkill.effect_type == EffectType.CONTEXT
    assert NavigateSkill.effect_type == EffectType.ACTION
    assert CaptureSkill.effect_type == EffectType.ACTION


# ---------------------------------------------------------------------------
# Step 2: ContextSkill / ActionSkill base-class semantics
# ---------------------------------------------------------------------------


def test_action_skill_without_precondition_is_uninstantiable() -> None:
    """ActionSkill subclasses MUST override precondition (paper §5 I3)."""

    class _ActSkillNoPrecondition(ActionSkill):
        name = "no_precondition"
        description = "broken — forgot precondition"
        params: Dict[str, str] = {}

        def build_subtree(
            self,
            params: Dict[str, Any],
            context: SkillContext,
            handle: SkillHandle,
        ) -> py_trees.behaviour.Behaviour:
            return _NoopNode(name=self.name)

    with pytest.raises(TypeError, match="precondition"):
        _ActSkillNoPrecondition()


def test_context_skill_does_not_require_precondition() -> None:
    """Σ_ctx skills are exempt from precondition (always available per I3)."""

    class _CtxSkillNoPrecondition(ContextSkill):
        name = "ctx_no_precondition"
        description = "context skill with no precondition method"
        params: Dict[str, str] = {}

        def build_subtree(
            self,
            params: Dict[str, Any],
            context: SkillContext,
            handle: SkillHandle,
        ) -> py_trees.behaviour.Behaviour:
            return _NoopNode(name=self.name)

        def tool_def(self) -> Dict[str, Any]:
            return super().tool_def()

    skill = _CtxSkillNoPrecondition()  # must instantiate cleanly
    assert skill.effect_type == EffectType.CONTEXT


# ---------------------------------------------------------------------------
# Step 2: builtin precondition semantics
# ---------------------------------------------------------------------------


def _navigate_skill():
    from bt_agent_robotics.skills.navigate import NavigateSkill

    return NavigateSkill(local_planner=object(), path_planner=object())


def _capture_skill():
    from bt_agent_robotics.skills.capture import CaptureSkill

    return CaptureSkill(model=object(), pointing_pipeline=None)


def _populated_object_store():
    from bt_agent._engine.skills.object_store import ObjectStore
    from bt_agent_robotics.skills.query import SceneObject

    store = ObjectStore()
    store.add(SceneObject(handle="red_mug_1", label="red mug"))
    return store


def test_navigate_precondition_blocks_when_object_store_empty() -> None:
    from bt_agent._engine.skills.object_store import ObjectStore

    nav = _navigate_skill()

    # No store at all
    ok, reason = nav.precondition(SkillContext())
    assert ok is False
    assert "ObjectStore" in reason

    # Empty store
    ok, reason = nav.precondition(SkillContext(object_store=ObjectStore()))
    assert ok is False
    assert "QUERY" in reason


def test_navigate_precondition_passes_with_objects() -> None:
    nav = _navigate_skill()
    state = SkillContext(object_store=_populated_object_store())
    ok, reason = nav.precondition(state)
    assert ok is True
    assert reason == ""


def test_capture_precondition_blocks_when_no_camera_images() -> None:
    cap = _capture_skill()
    state = SkillContext(object_store=_populated_object_store(), platform_state={})
    ok, reason = cap.precondition(state)
    assert ok is False
    assert "camera images" in reason


def test_capture_precondition_passes_with_full_state() -> None:
    cap = _capture_skill()
    state = SkillContext(
        object_store=_populated_object_store(),
        platform_state={"camera_images": {"frontleft": b"\x00"}},
    )
    ok, reason = cap.precondition(state)
    assert ok is True
    assert reason == ""


# ---------------------------------------------------------------------------
# Step 2: get_action_skills_masked AND-s precondition with is_available
# ---------------------------------------------------------------------------


def test_get_action_skills_masked_excludes_when_precondition_fails() -> None:
    """is_available may pass but empty ObjectStore → precondition blocks."""
    registry = SkillRegistry()
    registry.register(_navigate_skill())
    registry.register(_capture_skill())

    state = SkillContext()  # no ObjectStore → precondition fails
    history = {}  # is_available passes

    tools = registry.get_action_skills_masked(state, history, {})
    assert tools == []


def test_get_action_skills_masked_includes_when_both_pass() -> None:
    registry = SkillRegistry()
    registry.register(_navigate_skill())
    registry.register(_capture_skill())

    state = SkillContext(
        object_store=_populated_object_store(),
        platform_state={"camera_images": {"frontleft": b"\x00"}},
    )
    history = {}

    tools = registry.get_action_skills_masked(state, history, {})
    names = sorted(t["function"]["name"] for t in tools)
    assert names == ["capture", "navigate"]


def test_get_action_skills_masked_excludes_when_is_available_fails() -> None:
    """precondition passes but is_available blocks → skill excluded."""
    registry = SkillRegistry()
    registry.register(_StubGatedActionSkill())  # is_available requires gate_open

    state = SkillContext()  # trivial precondition → passes

    # Gate closed → is_available blocks
    tools = registry.get_action_skills_masked(state, {}, {})
    assert tools == []

    # Gate open → both pass
    tools = registry.get_action_skills_masked(state, {"gate_open": True}, {})
    assert len(tools) == 1
    assert tools[0]["function"]["name"] == "stub_gated_act"
