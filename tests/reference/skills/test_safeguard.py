"""Tests for the framework safeguarding wrap (paper §5 Invariant I5).

Covers Step 4 of the paper-alignment refactor:

- ``ActionSkill.safeguard_invariant`` is a second ``@abstractmethod``;
  subclasses MUST override (uninstantiable otherwise).
- ``Skill.start()`` auto-wraps the body with the framework safeguards
  (``CancellationGuard``, ``WallTimeBudgetGuard``, ``InvariantGuard``).
  Wrap shape: ``Sequence(memory=False, [...])`` per Lemma 5.5 hypothesis.
- ContextSkills get only ``CancellationGuard``; ActionSkills get all three.
- Defensive: ``safeguard_invariant`` exception ⇒ FAILURE (not propagated).
- ``_tick_subtree_loop`` walks the wrapped tree post-FAILURE to identify
  the failing leaf and prefixes ``handle.observation`` with ``[leaf=...]``;
  ``handle.result_data["failed_leaf"]`` carries the bare name.
- Builtin audits: ``NavigateSkill`` and ``CaptureSkill`` override the
  predicate; ContextSkills do not.
"""

from __future__ import annotations

import time
from typing import Dict
from unittest.mock import MagicMock

import py_trees
import pytest
from bt_agent._engine.agent.blackboard_keys import BB
from bt_agent._engine.skills.protocol import (
    ActionSkill,
    ContextSkill,
    SkillContext,
    SkillHandle,
    SkillStatus,
    _find_failed_leaf,
    _tick_subtree_loop,
)
from bt_agent._engine.skills.safeguard import CancellationGuard, InvariantGuard, WallTimeBudgetGuard, _wrap_safeguarding

# ---------------------------------------------------------------------------
# Stub nodes / skills
# ---------------------------------------------------------------------------


class _AlwaysSucceedNode(py_trees.behaviour.Behaviour):
    def __init__(self, name: str = "always_succeed") -> None:
        super().__init__(name=name)

    def update(self) -> py_trees.common.Status:
        return py_trees.common.Status.SUCCESS


class _AlwaysFailNode(py_trees.behaviour.Behaviour):
    def __init__(self, name: str = "always_fail") -> None:
        super().__init__(name=name)

    def update(self) -> py_trees.common.Status:
        return py_trees.common.Status.FAILURE


class _StubContextSkill(ContextSkill):
    name = "stub_ctx"
    description = "stub ctx for safeguard tests"
    params: Dict[str, str] = {}

    def build_subtree(self, params, context, handle):
        return _AlwaysSucceedNode(name=self.name)

    def tool_def(self):
        # Step 6.5: ContextSkill mandates an explicit tool_def override.
        return super().tool_def()


class _StubActionSkill(ActionSkill):
    name = "stub_act"
    description = "stub act for safeguard tests"
    params: Dict[str, str] = {}

    def __init__(self, predicate_value: bool = True, predicate_raises: bool = False) -> None:
        self._predicate_value = predicate_value
        self._predicate_raises = predicate_raises
        self.predicate_call_count = 0

    def build_subtree(self, params, context, handle):
        return _AlwaysSucceedNode(name=self.name)

    def precondition(self, state):
        return True, ""

    def safeguard_invariant(self, state, handle):
        self.predicate_call_count += 1
        if self._predicate_raises:
            raise RuntimeError("intentional predicate failure")
        return self._predicate_value


# ---------------------------------------------------------------------------
# Helpers — build a blackboard registered with BB.CANCEL_REQUESTED
# ---------------------------------------------------------------------------


def _make_handle() -> SkillHandle:
    h = SkillHandle()
    h.status = SkillStatus.RUNNING
    return h


def _bb_setup_cancel(value: bool = False) -> None:
    """Ensure BB.CANCEL_REQUESTED is set on the global blackboard."""
    bb = py_trees.blackboard.Client(name=f"test_setup_{id(object())}")
    bb.register_key(key=BB.CANCEL_REQUESTED, access=py_trees.common.Access.WRITE)
    bb.set(BB.CANCEL_REQUESTED, value)


def _tick_once(node: py_trees.behaviour.Behaviour) -> py_trees.common.Status:
    """Tick a node once and return its status."""
    for _ in node.tick():
        pass
    return node.status


# ---------------------------------------------------------------------------
# 1. Wrap shape
# ---------------------------------------------------------------------------


def test_actionskill_wrap_has_three_guards_plus_body() -> None:
    skill = _StubActionSkill()
    handle = _make_handle()
    body = _AlwaysSucceedNode(name="body")
    wrapped = _wrap_safeguarding(skill, body, handle, lambda: SkillContext())

    assert isinstance(wrapped, py_trees.composites.Sequence)
    assert len(wrapped.children) == 4
    assert isinstance(wrapped.children[0], CancellationGuard)
    assert isinstance(wrapped.children[1], WallTimeBudgetGuard)
    assert isinstance(wrapped.children[2], InvariantGuard)
    assert wrapped.children[3] is body


def test_contextskill_wrap_has_only_cancellation_guard() -> None:
    skill = _StubContextSkill()
    handle = _make_handle()
    body = _AlwaysSucceedNode(name="body")
    wrapped = _wrap_safeguarding(skill, body, handle, lambda: SkillContext())

    assert isinstance(wrapped, py_trees.composites.Sequence)
    assert len(wrapped.children) == 2
    assert isinstance(wrapped.children[0], CancellationGuard)
    assert wrapped.children[1] is body


def test_wrap_uses_memory_false() -> None:
    """Lemma 5.5 hypothesis — guards re-evaluate every tick."""
    skill = _StubActionSkill()
    handle = _make_handle()
    wrapped = _wrap_safeguarding(skill, _AlwaysSucceedNode(), handle, lambda: SkillContext())
    # py_trees Sequence stores `memory` as `_memory` attribute
    assert wrapped.memory is False


# ---------------------------------------------------------------------------
# 2. ABC enforcement
# ---------------------------------------------------------------------------


def test_actionskill_without_safeguard_invariant_uninstantiable() -> None:
    """Concrete ActionSkill subclasses MUST override safeguard_invariant."""

    class _ActSkillNoSafeguard(ActionSkill):
        name = "no_safeguard"
        description = "broken — forgot safeguard_invariant"
        params: Dict[str, str] = {}

        def build_subtree(self, params, context, handle):
            return _AlwaysSucceedNode(name=self.name)

        def precondition(self, state):
            return True, ""

        # NOTE: no safeguard_invariant override

    with pytest.raises(TypeError, match="safeguard_invariant"):
        _ActSkillNoSafeguard()


def test_contextskill_does_not_require_safeguard_invariant() -> None:
    """ContextSkills are exempt from the safeguard_invariant abstractmethod."""

    class _CtxNoSafeguard(ContextSkill):
        name = "ctx_no_safeguard"
        description = "ctx skill without safeguard predicate"
        params: Dict[str, str] = {}

        def build_subtree(self, params, context, handle):
            return _AlwaysSucceedNode(name=self.name)

        def tool_def(self):
            return super().tool_def()

    skill = _CtxNoSafeguard()  # must instantiate cleanly
    assert skill.name == "ctx_no_safeguard"


# ---------------------------------------------------------------------------
# 3. CancellationGuard
# ---------------------------------------------------------------------------


def test_cancellation_guard_returns_failure_on_handle_cancel() -> None:
    handle = _make_handle()
    handle.cancel()
    _bb_setup_cancel(False)

    guard = CancellationGuard(handle)
    assert _tick_once(guard) == py_trees.common.Status.FAILURE


def test_cancellation_guard_returns_failure_on_blackboard_cancel() -> None:
    handle = _make_handle()
    _bb_setup_cancel(True)

    guard = CancellationGuard(handle)
    assert _tick_once(guard) == py_trees.common.Status.FAILURE
    _bb_setup_cancel(False)  # cleanup


def test_cancellation_guard_returns_success_when_no_cancel() -> None:
    handle = _make_handle()
    _bb_setup_cancel(False)

    guard = CancellationGuard(handle)
    assert _tick_once(guard) == py_trees.common.Status.SUCCESS


# ---------------------------------------------------------------------------
# 4. WallTimeBudgetGuard
# ---------------------------------------------------------------------------


def test_walltime_guard_returns_success_within_budget() -> None:
    handle = _make_handle()
    guard = WallTimeBudgetGuard(budget_seconds=10.0, handle=handle)
    assert _tick_once(guard) == py_trees.common.Status.SUCCESS


def test_walltime_guard_returns_failure_after_budget() -> None:
    handle = _make_handle()
    guard = WallTimeBudgetGuard(budget_seconds=0.1, handle=handle)

    # Tick once to snapshot start time
    _tick_once(guard)
    # Sleep past the budget and tick again
    time.sleep(0.2)
    assert _tick_once(guard) == py_trees.common.Status.FAILURE


def test_walltime_guard_writes_observation_on_breach() -> None:
    handle = _make_handle()
    guard = WallTimeBudgetGuard(budget_seconds=0.05, handle=handle)
    _tick_once(guard)
    time.sleep(0.1)
    _tick_once(guard)
    assert "wall-time budget exceeded" in handle.observation


def test_actionskill_default_wall_time_budget_is_120s() -> None:
    skill = _StubActionSkill()
    assert skill.wall_time_budget_seconds == 120.0


# ---------------------------------------------------------------------------
# 5. InvariantGuard
# ---------------------------------------------------------------------------


def test_invariant_guard_returns_success_when_predicate_true() -> None:
    skill = _StubActionSkill(predicate_value=True)
    handle = _make_handle()
    guard = InvariantGuard(skill, handle, state_provider=lambda: SkillContext())
    assert _tick_once(guard) == py_trees.common.Status.SUCCESS


def test_invariant_guard_returns_failure_when_predicate_false() -> None:
    skill = _StubActionSkill(predicate_value=False)
    handle = _make_handle()
    guard = InvariantGuard(skill, handle, state_provider=lambda: SkillContext())
    assert _tick_once(guard) == py_trees.common.Status.FAILURE
    assert "safeguard_invariant returned False" in handle.observation


def test_invariant_guard_catches_exception_and_returns_failure() -> None:
    """Defensive default: predicate exception ⇒ FAILURE, not propagated."""
    skill = _StubActionSkill(predicate_raises=True)
    handle = _make_handle()
    guard = InvariantGuard(skill, handle, state_provider=lambda: SkillContext())
    # Should NOT raise
    assert _tick_once(guard) == py_trees.common.Status.FAILURE
    assert "RuntimeError" in handle.observation
    assert "intentional predicate failure" in handle.observation


def test_invariant_guard_invokes_state_provider_per_tick() -> None:
    """Lemma 5.5 hypothesis: state re-evaluated each tick."""
    skill = _StubActionSkill(predicate_value=True)
    handle = _make_handle()
    call_count = [0]

    def provider() -> SkillContext:
        call_count[0] += 1
        return SkillContext()

    guard = InvariantGuard(skill, handle, state_provider=provider)
    _tick_once(guard)
    _tick_once(guard)
    _tick_once(guard)
    assert call_count[0] == 3


# ---------------------------------------------------------------------------
# 6. Failing-leaf identification
# ---------------------------------------------------------------------------


def test_find_failed_leaf_returns_leaf_name() -> None:
    """Single-node tree that fails: returns its own name."""
    node = _AlwaysFailNode(name="my_leaf")
    _tick_once(node)
    assert _find_failed_leaf(node) == "my_leaf"


def test_find_failed_leaf_walks_nested_composites() -> None:
    """Nested Sequence([a, Sequence([b, fail_c])]) — finds 'fail_c'."""
    inner = py_trees.composites.Sequence(
        name="inner",
        memory=True,
        children=[_AlwaysSucceedNode(name="b"), _AlwaysFailNode(name="fail_c")],
    )
    outer = py_trees.composites.Sequence(
        name="outer",
        memory=True,
        children=[_AlwaysSucceedNode(name="a"), inner],
    )
    _tick_once(outer)
    assert _find_failed_leaf(outer) == "fail_c"


def test_find_failed_leaf_returns_none_for_opaque_body() -> None:
    """Library-agnostic safety: non-py_trees opaque body returns None."""

    class _Opaque:
        # No status, no children — looks nothing like py_trees
        pass

    assert _find_failed_leaf(_Opaque()) is None
    assert _find_failed_leaf(None) is None


def test_tick_subtree_loop_writes_leaf_prefix_on_failure() -> None:
    """Body fails at child 'broken_leaf' → handle gets [leaf=broken_leaf] prefix."""
    handle = _make_handle()
    handle.observation = "something broke"
    body = py_trees.composites.Sequence(
        name="body",
        memory=True,
        children=[_AlwaysFailNode(name="broken_leaf")],
    )
    _tick_subtree_loop(body, handle)

    assert handle.status == SkillStatus.FAILED
    assert handle.observation.startswith("[leaf=broken_leaf]")
    assert handle.result_data.get("failed_leaf") == "broken_leaf"


def test_tick_subtree_loop_reports_invariantguard_as_failing_leaf() -> None:
    """When InvariantGuard fails, the leaf tag is 'InvariantGuard'."""
    skill = _StubActionSkill(predicate_value=False)
    handle = _make_handle()
    body = _AlwaysSucceedNode(name="body")
    wrapped = _wrap_safeguarding(skill, body, handle, state_provider=lambda: SkillContext())
    _tick_subtree_loop(wrapped, handle)

    assert handle.status == SkillStatus.FAILED
    assert handle.observation.startswith("[leaf=InvariantGuard]")
    assert handle.result_data.get("failed_leaf") == "InvariantGuard"


# ---------------------------------------------------------------------------
# 7. Per-skill audits — builtins override safeguard_invariant
# ---------------------------------------------------------------------------


def test_navigate_skill_overrides_safeguard_invariant() -> None:
    from lam_v2_tiamat.bt_skills.navigate import NavigateSkill

    assert NavigateSkill.safeguard_invariant is not ActionSkill.safeguard_invariant


def test_capture_skill_overrides_safeguard_invariant() -> None:
    from lam_v2_tiamat.bt_skills.capture import CaptureSkill

    assert CaptureSkill.safeguard_invariant is not ActionSkill.safeguard_invariant


def test_navigate_safeguard_returns_false_on_missing_localization() -> None:
    from lam_v2_tiamat.bt_skills.navigate import NavigateSkill

    skill = NavigateSkill(local_planner=MagicMock(), path_planner=MagicMock())
    handle = _make_handle()
    state = SkillContext(platform_state={})
    # CP5: typed failure-reason string, not plain False.
    assert skill.safeguard_invariant(state, handle) == "pose_lost"


def test_navigate_safeguard_returns_false_on_empty_planned_path() -> None:
    from lam_v2_tiamat.bt_skills.navigate import NavigateSkill

    skill = NavigateSkill(local_planner=MagicMock(), path_planner=MagicMock())
    handle = _make_handle()
    handle.result_data = {"planned_path": []}
    state = SkillContext(platform_state={"robot_position": (0.0, 0.0, 0.0)})
    # CP5: typed failure-reason string, not plain False.
    assert skill.safeguard_invariant(state, handle) == "path_invalidated"


def test_navigate_safeguard_returns_true_pre_planning() -> None:
    """No planned_path yet (still in _ValidateGoal) — vacuously safe."""
    from lam_v2_tiamat.bt_skills.navigate import NavigateSkill

    skill = NavigateSkill(local_planner=MagicMock(), path_planner=MagicMock())
    handle = _make_handle()
    state = SkillContext(platform_state={"robot_position": (0.0, 0.0, 0.0)})
    assert skill.safeguard_invariant(state, handle) is True


def test_capture_safeguard_pre_analysis_is_vacuously_true() -> None:
    from lam_v2_tiamat.bt_skills.capture import CaptureSkill

    skill = CaptureSkill(model=MagicMock(), pointing_pipeline=None)
    handle = _make_handle()
    # result_data is empty — pre-analysis
    state = SkillContext(platform_state={})
    assert skill.safeguard_invariant(state, handle) is True


def test_capture_safeguard_returns_false_post_analysis_with_no_camera() -> None:
    from lam_v2_tiamat.bt_skills.capture import CaptureSkill

    skill = CaptureSkill(model=MagicMock(), pointing_pipeline=None)
    handle = _make_handle()
    handle.result_data = {
        "analyze_view_succeeded": True,
        "analyze_view_camera": "frontleft",
    }
    state = SkillContext(platform_state={})  # camera feed dropped out
    assert skill.safeguard_invariant(state, handle) is False


def test_capture_safeguard_returns_true_post_analysis_with_camera_present() -> None:
    from lam_v2_tiamat.bt_skills.capture import CaptureSkill

    skill = CaptureSkill(model=MagicMock(), pointing_pipeline=None)
    handle = _make_handle()
    handle.result_data = {
        "analyze_view_succeeded": True,
        "analyze_view_camera": "frontleft",
    }
    state = SkillContext(platform_state={"camera_images": {"frontleft": b"\x00"}})
    assert skill.safeguard_invariant(state, handle) is True


# ---------------------------------------------------------------------------
# 8. Skill.start() integration — wrap visible on handle.subtree
# ---------------------------------------------------------------------------


def test_skill_start_exposes_wrapped_tree_on_handle_subtree() -> None:
    """``handle.subtree`` is the safeguard-wrapped tree, not the bare body."""
    skill = _StubActionSkill()
    _bb_setup_cancel(False)

    handle = skill.start({}, SkillContext())
    handle._completion_event.wait(timeout=2.0)

    assert isinstance(handle.subtree, py_trees.composites.Sequence)
    assert handle.subtree.name == "safeguarded_stub_act"
    assert handle.status == SkillStatus.SUCCEEDED


# ---------------------------------------------------------------------------
# 9. Typed wrap-level failure_reason (bt-nav-fixes CP4 + CP5)
# ---------------------------------------------------------------------------


def test_cancellation_guard_handle_cancel_sets_failure_reason() -> None:
    handle = _make_handle()
    handle.cancel()
    _bb_setup_cancel(False)

    guard = CancellationGuard(handle)
    assert _tick_once(guard) == py_trees.common.Status.FAILURE
    assert handle.result_data["failure_reason"] == "cancelled"


def test_cancellation_guard_blackboard_cancel_sets_failure_reason() -> None:
    handle = _make_handle()
    _bb_setup_cancel(True)

    guard = CancellationGuard(handle)
    assert _tick_once(guard) == py_trees.common.Status.FAILURE
    assert handle.result_data["failure_reason"] == "cancelled"
    _bb_setup_cancel(False)  # cleanup


def test_walltime_guard_failure_sets_failure_reason() -> None:
    handle = _make_handle()
    guard = WallTimeBudgetGuard(budget_seconds=0.05, handle=handle)
    _tick_once(guard)
    time.sleep(0.1)
    _tick_once(guard)
    assert handle.result_data["failure_reason"] == "wall_time_exceeded"


def test_walltime_guard_reads_handle_override_each_tick() -> None:
    """CP4: the guard must read a mutable per-execution budget, not a value
    snapshotted once at construction — so a skill can shrink/grow its own
    budget mid-run (e.g. NavigateSkill after a replan)."""
    handle = _make_handle()
    # Static (constructor) budget is generous; the handle override is tiny.
    guard = WallTimeBudgetGuard(budget_seconds=120.0, handle=handle)
    handle.wall_time_budget_seconds = 0.05

    _tick_once(guard)
    time.sleep(0.1)
    status = _tick_once(guard)

    assert status == py_trees.common.Status.FAILURE  # tiny override wins over the 120s static default
    assert handle.result_data["failure_reason"] == "wall_time_exceeded"


def test_walltime_guard_ignores_none_override_uses_static_default() -> None:
    """Backward compat: a skill that never touches handle.wall_time_budget_
    seconds (the default None) behaves exactly as before CP4."""
    handle = _make_handle()
    assert handle.wall_time_budget_seconds is None
    guard = WallTimeBudgetGuard(budget_seconds=0.05, handle=handle)
    _tick_once(guard)
    time.sleep(0.1)
    assert _tick_once(guard) == py_trees.common.Status.FAILURE


def test_invariant_guard_string_reason_sets_typed_failure_reason() -> None:
    class _TypedInvariantSkill(_StubActionSkill):
        def safeguard_invariant(self, state, handle):
            return "stale_pose"

    skill = _TypedInvariantSkill()
    handle = _make_handle()
    guard = InvariantGuard(skill, handle, state_provider=lambda: SkillContext())

    assert _tick_once(guard) == py_trees.common.Status.FAILURE
    assert handle.result_data["failure_reason"] == "stale_pose"
    assert "stale_pose" in handle.observation


def test_invariant_guard_plain_false_does_not_set_failure_reason() -> None:
    """Backward compat: skills that keep returning plain bool (e.g.
    CaptureSkill) get no failure_reason key — same as before CP5."""
    skill = _StubActionSkill(predicate_value=False)
    handle = _make_handle()
    guard = InvariantGuard(skill, handle, state_provider=lambda: SkillContext())

    assert _tick_once(guard) == py_trees.common.Status.FAILURE
    assert "failure_reason" not in handle.result_data


def test_skill_start_state_provider_is_called_per_tick() -> None:
    """ActionSkill body running multiple ticks → state_provider invoked >1x."""
    call_count = [0]

    def provider() -> SkillContext:
        call_count[0] += 1
        return SkillContext()

    class _MultiTickAction(ActionSkill):
        name = "multi_tick"
        description = "runs RUNNING for two ticks then SUCCESS"
        params: Dict[str, str] = {}
        wall_time_budget_seconds = 10.0

        def build_subtree(self, params, context, handle):
            return _MultiTickNode(name="multi_tick_body")

        def precondition(self, state):
            return True, ""

        def safeguard_invariant(self, state, handle):
            return True

    class _MultiTickNode(py_trees.behaviour.Behaviour):
        def __init__(self, name: str) -> None:
            super().__init__(name=name)
            self._ticks = 0

        def update(self) -> py_trees.common.Status:
            self._ticks += 1
            if self._ticks >= 3:
                return py_trees.common.Status.SUCCESS
            return py_trees.common.Status.RUNNING

    skill = _MultiTickAction()
    _bb_setup_cancel(False)

    handle = skill.start({}, SkillContext(), state_provider=provider)
    handle._completion_event.wait(timeout=5.0)

    assert handle.status == SkillStatus.SUCCEEDED
    assert call_count[0] >= 2  # state_provider called at least twice (>=2 ticks)
