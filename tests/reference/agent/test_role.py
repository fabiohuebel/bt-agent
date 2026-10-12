"""Tests for the Role dispatcher (paper §5, Invariant I2 — layered visibility).

Covers Step 6: lifting per-role tool-list construction into a single seam at
``bt_agent._engine.agent.role``.

Step 6.5 update: ``role.py`` is substrate-agnostic — it owns ONLY framework
meta-tools (``done_gathering``, ``read_skill_docs``, ``defer``). Skill tool
defs come from the registry. These tests pin both that contract and that
the dispatcher respects skill author authority over tool surfaces.
"""

from __future__ import annotations

from typing import Any, Dict

import py_trees
import pytest
from bt_agent._engine.agent.role import Role
from bt_agent._engine.skills.protocol import ActionSkill, ContextSkill, SkillContext, SkillHandle
from bt_agent._engine.skills.registry import SkillRegistry

# ---------------------------------------------------------------------------
# Stub skills (mirrors the pattern in tests/skills/test_registry_partition.py)
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
        # Step 6.5: ContextSkill mandates an explicit tool_def; delegate to
        # the framework default which builds from declared class attrs.
        return super().tool_def()


class _StubActionSkill(ActionSkill):
    name = "stub_act"
    description = "stub action skill"
    params: Dict[str, str] = {}

    def build_subtree(
        self, params: Dict[str, Any], context: SkillContext, handle: SkillHandle
    ) -> py_trees.behaviour.Behaviour:
        return _NoopNode(name=self.name)

    def precondition(self, state: SkillContext):
        return True, ""

    def safeguard_invariant(self, state: SkillContext, handle: SkillHandle) -> bool:
        return True


class _BlockedActionSkill(ActionSkill):
    """Action skill whose precondition always fails — masked out."""

    name = "blocked_act"
    description = "stub action skill that fails precondition"
    params: Dict[str, str] = {}

    def build_subtree(
        self, params: Dict[str, Any], context: SkillContext, handle: SkillHandle
    ) -> py_trees.behaviour.Behaviour:
        return _NoopNode(name=self.name)

    def precondition(self, state: SkillContext):
        return False, "always blocked"

    def safeguard_invariant(self, state: SkillContext, handle: SkillHandle) -> bool:
        return True


def _populated_registry() -> SkillRegistry:
    registry = SkillRegistry()
    registry.register(_StubContextSkill())
    registry.register(_StubActionSkill())
    registry.register(_BlockedActionSkill())
    return registry


# ---------------------------------------------------------------------------
# Enum stability
# ---------------------------------------------------------------------------


def test_role_enum_string_values_are_stable() -> None:
    """String values are forward-compat for Step 7/8 event payloads."""
    assert Role.PLANNER.value == "planner"
    assert Role.ACTOR.value == "actor"
    assert Role.EVALUATOR.value == "evaluator"


# ---------------------------------------------------------------------------
# Role.PLANNER (Step 6.5: framework meta-tools + registry context skills)
# ---------------------------------------------------------------------------


def test_planner_returns_meta_tools_plus_registry_context_skills() -> None:
    """Planner sees framework meta-tools FIRST, then every ContextSkill's tool_def."""
    registry = _populated_registry()
    tools = Role.PLANNER.available_tools(registry=registry)

    names = [t["function"]["name"] for t in tools]
    # Planner docs were deliberately removed in cfda0b6e3c; the actor retains
    # read_skill_docs, while the planner sees one meta-tool plus context skills.
    assert names == ["done_gathering", "stub_ctx"]
    # Action skills are NOT exposed to the planner (paper §5: planner cannot
    # invoke action skills directly — only the actor sees Σ_act).
    assert "stub_act" not in names
    assert "blocked_act" not in names


def test_planner_does_not_expose_read_skill_docs() -> None:
    """The decomposer stays on compact tool defs; only the actor reads docs."""
    registry = _populated_registry()
    tools = Role.PLANNER.available_tools(registry=registry)
    assert "read_skill_docs" not in [tool["function"]["name"] for tool in tools]


def test_planner_raises_without_registry() -> None:
    with pytest.raises(ValueError, match="registry"):
        Role.PLANNER.available_tools()


def test_planner_does_not_name_substrate_specific_tools() -> None:
    """role.py contract: planner meta-tools name no substrate-specific concepts."""
    # Build with an empty registry to isolate the meta-tool surface.
    tools = Role.PLANNER.available_tools(registry=SkillRegistry())
    names = {t["function"]["name"] for t in tools}
    assert names == {"done_gathering"}


# ---------------------------------------------------------------------------
# Role.ACTOR
# ---------------------------------------------------------------------------


def test_actor_returns_ctx_plus_masked_act_plus_read_docs() -> None:
    """Actor sees Σ_ctx + masked Σ_act + actor read_skill_docs."""
    registry = _populated_registry()
    state = SkillContext()  # _StubActionSkill precondition trivially True

    tools = Role.ACTOR.available_tools(registry=registry, state=state)
    names = [t["function"]["name"] for t in tools]

    # Order: ctx, masked act, claim_complete, read_skill_docs (no defer).
    assert names == ["stub_ctx", "stub_act", "claim_complete", "read_skill_docs"]
    assert "blocked_act" not in names  # precondition False → masked out


def test_actor_includes_defer_when_gating_can_defer_true() -> None:
    registry = _populated_registry()
    state = SkillContext()

    tools_with = Role.ACTOR.available_tools(registry=registry, state=state, gating_context={"can_defer": True})
    names_with = [t["function"]["name"] for t in tools_with]
    assert "defer" in names_with
    # defer comes BEFORE read_skill_docs (matches pre-Step-6 ordering:
    # SelectSkill appended defer, then actor's select_skill appended read_docs)
    assert names_with.index("defer") < names_with.index("read_skill_docs")

    # can_defer omitted → no defer
    tools_without = Role.ACTOR.available_tools(registry=registry, state=state)
    assert "defer" not in [t["function"]["name"] for t in tools_without]

    # can_defer explicitly False → no defer
    tools_false = Role.ACTOR.available_tools(registry=registry, state=state, gating_context={"can_defer": False})
    assert "defer" not in [t["function"]["name"] for t in tools_false]


def test_actor_read_skill_docs_has_no_reasoning_field() -> None:
    """Actor variant of read_skill_docs has only skill_name (vs planner's reasoning+skill_name)."""
    registry = _populated_registry()
    state = SkillContext()
    tools = Role.ACTOR.available_tools(registry=registry, state=state)

    read_docs = next(t for t in tools if t["function"]["name"] == "read_skill_docs")
    required = read_docs["function"]["parameters"]["required"]
    assert required == ["skill_name"]
    assert "reasoning" not in required


def test_actor_raises_without_state() -> None:
    registry = _populated_registry()
    with pytest.raises(ValueError, match="state"):
        Role.ACTOR.available_tools(registry=registry)


def test_actor_raises_without_registry() -> None:
    with pytest.raises(ValueError, match="registry"):
        Role.ACTOR.available_tools(state=SkillContext())


# ---------------------------------------------------------------------------
# Role.EVALUATOR
# ---------------------------------------------------------------------------


def test_evaluator_returns_empty_tool_list() -> None:
    """Evaluator uses model.query_structured — never sees tools."""
    assert Role.EVALUATOR.available_tools() == []
    # All kwargs ignored; still returns [] without raising.
    assert (
        Role.EVALUATOR.available_tools(
            registry=_populated_registry(),
            state=SkillContext(),
            history_state={"x": 1},
            gating_context={"can_defer": True},
        )
        == []
    )


# ---------------------------------------------------------------------------
# Step 6.5: skill-author authority
# ---------------------------------------------------------------------------


def test_context_skill_subclass_without_tool_def_is_uninstantiable() -> None:
    """Step 6.5: ContextSkill mandates an explicit tool_def() override."""

    class _CtxNoToolDef(ContextSkill):
        name = "ctx_no_tool_def"
        description = "broken — forgot tool_def override"
        params: Dict[str, str] = {}

        def build_subtree(
            self, params: Dict[str, Any], context: SkillContext, handle: SkillHandle
        ) -> py_trees.behaviour.Behaviour:
            return _NoopNode(name=self.name)

        # NOTE: no tool_def override.

    with pytest.raises(TypeError, match="tool_def"):
        _CtxNoToolDef()


def test_planner_skill_tool_def_is_skill_authored_not_framework_authored() -> None:
    """Step 6.5: the framework calls skill.tool_def() — it does not synthesize names.

    A custom ContextSkill that overrides tool_def with a hand-written dict
    must surface verbatim through Role.PLANNER.
    """

    class _CustomCtxSkill(ContextSkill):
        name = "custom_ctx"
        description = "custom"
        params: Dict[str, str] = {}

        def build_subtree(
            self, params: Dict[str, Any], context: SkillContext, handle: SkillHandle
        ) -> py_trees.behaviour.Behaviour:
            return _NoopNode(name=self.name)

        def tool_def(self) -> Dict[str, Any]:
            # Skill author owns the surface — framework must not rewrite this.
            return {
                "type": "function",
                "function": {
                    "name": "totally_custom_name",
                    "description": "skill author wrote this verbatim",
                    "parameters": {
                        "type": "object",
                        "properties": {"foo": {"type": "string"}},
                        "required": ["foo"],
                    },
                },
            }

    registry = SkillRegistry()
    registry.register(_CustomCtxSkill())
    tools = Role.PLANNER.available_tools(registry=registry)
    custom = next(t for t in tools if t["function"]["name"] == "totally_custom_name")
    assert custom["function"]["description"] == "skill author wrote this verbatim"


def test_registry_context_skill_by_tool_name_routes_correctly() -> None:
    """The planner's gather loop routes tool calls back to skills via this accessor."""
    registry = _populated_registry()

    skill = registry.context_skill_by_tool_name("stub_ctx")
    assert skill is not None
    assert skill.name == "stub_ctx"

    # Action skills are NOT looked up here (planner can't invoke them).
    assert registry.context_skill_by_tool_name("stub_act") is None

    # Unknown tool name returns None (loop handles gracefully).
    assert registry.context_skill_by_tool_name("does_not_exist") is None
