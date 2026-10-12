"""Skill protocol — the contract between skills and the BT executor.

Every skill is a behavior tree at its core:

    1. ``build_subtree()`` returns a py_trees subtree that implements the skill.
    2. ``start()`` creates a SkillHandle and runs the subtree in a bg thread.
    3. ``ExecuteSkill`` polls ``handle.status`` each BT tick.
    4. On cancellation, ``handle.cancel()`` propagates to the subtree.

Each skill subclasses ``Skill`` directly and implements ``build_subtree()``
to return its own BT — from a single custom node (Query, ImageInject) to a
multi-node Sequence (SurroundSurvey) to a full composite tree (Navigate).

There are no intermediary base classes. Every skill owns its BT structure.

The BT structure gives every skill:
    - Decomposition: complex skills are trees of sub-operations
    - Preemptability: cancellation propagates through the subtree
    - Debuggability: the subtree is inspectable (py_trees visualization)
    - Uniformity: every skill — from a one-line query to a five-camera
      survey — is the same kind of thing
"""

from __future__ import annotations

import logging
import threading
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum, auto
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union

import py_trees
from bt_agent._engine.agent._internals_config import TICK_LOOP_SLEEP

if TYPE_CHECKING:
    from bt_agent._engine.skills.object_store import ObjectStore

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Skill lifecycle
# ---------------------------------------------------------------------------


class SkillStatus(Enum):
    """Lifecycle states of a skill execution."""

    IDLE = auto()
    RUNNING = auto()
    SUCCEEDED = auto()
    FAILED = auto()
    CANCELLED = auto()


class EffectType(Enum):
    """Tool partition by effect type (paper §5, Invariant I2).

    Every skill belongs to exactly one of:

    * ``CONTEXT`` (Σ_ctx): read-only context-gathering tools. May write to the
      shared context store K (ObjectStore) but must NOT mutate ``s_robot`` or
      the physical world. Always available; no precondition gating on world
      state.
    * ``ACTION`` (Σ_act): effectful skill-BTs that may change ``s_robot`` or
      the world. Realized as ``Sequence(safeguard_subtree, body)`` and gated
      by a declarative precondition ``I_σ``.
    """

    CONTEXT = "context"
    ACTION = "action"


@dataclass
class SkillHandle:
    """Opaque handle returned by ``Skill.start()``.

    Thread-safe: the skill thread writes status/progress/observation,
    the BT tick thread reads them. Protected by a ``threading.Lock``.

    Cancellation: the BT calls ``cancel()`` which sets a
    ``threading.Event``. The skill checks ``is_cancel_requested``
    periodically and aborts gracefully.
    """

    _status: SkillStatus = SkillStatus.IDLE
    _progress: float = 0.0
    _observation: str = ""
    _result_data: Dict[str, Any] = field(default_factory=dict)
    # Per-execution override for WallTimeBudgetGuard's budget (seconds).
    # ``None`` (the default) means "use the static ActionSkill.wall_time_
    # budget_seconds the guard was constructed with" — every skill that
    # never touches this field behaves exactly as before (CP4). Skills
    # with a distance dimension (e.g. NavigateSkill) may set this once a
    # planned path is known and update it on every successful replan.
    _wall_time_budget_seconds: Optional[float] = None
    _cancel_event: threading.Event = field(default_factory=threading.Event)
    _completion_event: threading.Event = field(default_factory=threading.Event)
    _lock: threading.Lock = field(default_factory=threading.Lock)
    # OBS-DROP: visualizer-only — root of the skill's BT subtree, used by
    # tick_snapshot to splice the running skill's internal structure into
    # the main BT view. Not consumed by any non-viz code.
    _subtree: Any = None

    # -- Thread-safe property access ----------------------------------------

    @property
    def status(self) -> SkillStatus:
        with self._lock:
            return self._status

    @status.setter
    def status(self, value: SkillStatus) -> None:
        with self._lock:
            self._status = value

    @property
    def progress(self) -> float:
        with self._lock:
            return self._progress

    @progress.setter
    def progress(self, value: float) -> None:
        with self._lock:
            self._progress = value

    @property
    def observation(self) -> str:
        with self._lock:
            return self._observation

    @observation.setter
    def observation(self, value: str) -> None:
        with self._lock:
            self._observation = value

    @property
    def result_data(self) -> Dict[str, Any]:
        with self._lock:
            return dict(self._result_data)

    @result_data.setter
    def result_data(self, value: Dict[str, Any]) -> None:
        with self._lock:
            self._result_data = value

    @property
    def wall_time_budget_seconds(self) -> Optional[float]:
        """Per-execution WallTimeBudgetGuard override, or ``None`` for the
        skill's static default (see field docstring above)."""
        with self._lock:
            return self._wall_time_budget_seconds

    @wall_time_budget_seconds.setter
    def wall_time_budget_seconds(self, value: Optional[float]) -> None:
        with self._lock:
            self._wall_time_budget_seconds = value

    # -- Cancellation -------------------------------------------------------

    def cancel(self) -> None:
        """Request graceful cancellation. The skill thread checks ``is_cancel_requested``."""
        self._cancel_event.set()

    @property
    def is_cancel_requested(self) -> bool:
        return self._cancel_event.is_set()

    # ---- OBS-DROP-BEGIN ----
    @property
    def subtree(self) -> Any:
        """Root of the running skill's BT subtree (visualization only)."""
        return self._subtree

    @subtree.setter
    def subtree(self, value: Any) -> None:
        self._subtree = value

    # ---- OBS-DROP-END ----

    @property
    def is_terminal(self) -> bool:
        return self.status in (SkillStatus.SUCCEEDED, SkillStatus.FAILED, SkillStatus.CANCELLED)


# ---------------------------------------------------------------------------
# Skill context (read-only world snapshot)
# ---------------------------------------------------------------------------


@dataclass
class SkillContext:
    """Read-only world snapshot passed to skills at start time.

    Skills never touch the agent, blackboard, or BT directly. They receive
    everything they need through this context and return results through
    the ``SkillHandle``.
    """

    # -- Framework fields (substrate-agnostic) ---------------------------------
    object_store: Optional["ObjectStore"] = None
    milestone: Optional[Dict[str, Any]] = None
    milestone_index: int = 0
    main_task: str = ""
    scenario: str = ""
    # physics-trace-recording wave 2: id of the actor decision that selected
    # the skill this context is being built for ("" if none has been minted
    # yet, e.g. the DEFER/alt-tier callers that never write BB.SKILL_CALL_ID).
    # Skills that produce a submittable artifact (e.g. CaptureSkill) carry it
    # into that artifact so the offline extractor can link decision -> world
    # outcome -> submission by id instead of FIFO-by-skill-name.
    skill_call_id: str = ""

    # -- Platform state (paper §5: s_robot) ------------------------------------
    platform_state: Dict[str, Any] = field(default_factory=dict)
    # Opaque platform-specific robot state (paper §5: s_robot).
    #
    # The framework is agnostic to what's in here — it passes the dict
    # through without inspecting it. The bridge populates it with whatever
    # the platform provides (perception data, sensor feeds, joint states,
    # text observations, etc.). Skills read what they need via .get();
    # the framework never inspects the contents. Consuming skills should
    # document which platform_state keys they depend on in their .md docs.


# ---------------------------------------------------------------------------
# Subtree runner (shared by Skill.start default and compound skills)
# ---------------------------------------------------------------------------


def _find_failed_leaf(node: Any) -> Optional[str]:
    """Walk a py_trees subtree to find the leaf that actually caused failure.

    Returns the leaf's ``name`` attribute, or ``None`` if no failed leaf
    can be located. Library-agnostic safety: returns ``None`` for opaque
    bodies that don't expose a py_trees-shaped ``children`` attribute, so
    skills built on BehaviorTree.cpp shims or skill-server stubs don't crash.

    Recurses into the LAST child with status==FAILURE, not the first.
    Selectors try children in order and only report overall FAILURE once
    every child has failed, so the first FAILURE child is typically an
    earlier fall-through leaf (e.g. a guard/precondition check) rather than
    the child whose failure actually ended the attempt. Sequences only ever
    have a single FAILURE child (later siblings stay INVALID), so "last" and
    "first" coincide there — this is a strict correctness improvement for
    Selectors with no behavior change for Sequences.
    """
    if node is None:
        return None
    try:
        status = node.status
    except AttributeError:
        return None
    if status != py_trees.common.Status.FAILURE:
        return None
    children = getattr(node, "children", None)
    if not children:
        # Leaf node — return its name
        return getattr(node, "name", None)
    last_failed_child = None
    for child in children:
        try:
            if child.status == py_trees.common.Status.FAILURE:
                last_failed_child = child
        except AttributeError:
            continue
    if last_failed_child is not None:
        return _find_failed_leaf(last_failed_child)
    # Composite reports FAILURE but no child does — return composite's own name
    return getattr(node, "name", None)


def _tick_subtree_loop(
    subtree: py_trees.behaviour.Behaviour,
    handle: SkillHandle,
) -> None:
    """Tick a skill's BT subtree until terminal or cancelled.

    Runs in a background thread. Subtree nodes may block (sync LLM calls)
    because this is NOT the main BT tick thread.
    """
    while True:
        if handle.is_cancel_requested:
            if subtree.status == py_trees.common.Status.RUNNING:
                subtree.stop(py_trees.common.Status.INVALID)
            handle.status = SkillStatus.CANCELLED
            handle._completion_event.set()
            return

        try:
            # Exhaust the tick generator — this runs the subtree
            for _ in subtree.tick():
                pass
        except Exception as e:
            # Convert a crashed tick into a terminal state instead of leaving the handle frozen RUNNING.
            logger.exception("Skill subtree tick raised")
            # Stop the subtree first so terminal decorators still fire —
            # navigate/press wrap their bodies in _TerminalDiscipline, whose
            # terminate() parks at self + engages the /stop brake. Marking
            # FAILED without stopping would leave the brake released and the
            # last goal latched: the path follower keeps emitting fresh
            # /cmd_vel (the mux zero-Twist failsafe never triggers).
            try:
                subtree.stop(py_trees.common.Status.INVALID)
            except Exception:
                logger.exception("Subtree stop after crash also raised")
            handle.observation = f"Subtree exception {type(e).__name__}: {e}"
            data = handle.result_data
            data["failure_reason"] = "subtree_exception"
            handle.result_data = data
            handle.status = SkillStatus.FAILED
            handle._completion_event.set()
            return

        status = subtree.status
        if status == py_trees.common.Status.SUCCESS:
            if handle.status == SkillStatus.RUNNING:
                handle.status = SkillStatus.SUCCEEDED
            handle._completion_event.set()
            return
        if status == py_trees.common.Status.FAILURE:
            if handle.is_cancel_requested:
                handle.status = SkillStatus.CANCELLED
            elif handle.status == SkillStatus.RUNNING:
                leaf_name = _find_failed_leaf(subtree)
                if leaf_name and not handle.observation.startswith("[leaf="):
                    handle.observation = f"[leaf={leaf_name}] {handle.observation}"
                    data = handle.result_data
                    data["failed_leaf"] = leaf_name
                    handle.result_data = data
                handle.status = SkillStatus.FAILED
            handle._completion_event.set()
            return

        # RUNNING — tick-rate throttle (not a sync wait)
        time.sleep(TICK_LOOP_SLEEP)


# ---------------------------------------------------------------------------
# Skill ABC
# ---------------------------------------------------------------------------


class Skill(ABC):
    """Base class for all skills. Every skill is a BT at its core.

    Subclass this directly and implement ``build_subtree()`` to return a
    py_trees subtree — from a single custom Behaviour node to a full
    composite tree. The default ``start()`` runs the subtree in a
    background thread via ``_tick_subtree_loop``.

    Simple skills (Query, ImageInject) return a single node whose
    ``update()`` completes in one tick. Complex skills (SurroundSurvey,
    Navigate) return composites with multiple children.
    """

    name: str = ""
    description: str = ""
    usage: str = ""
    params: Dict[str, str] = {}

    # -- Effect type (paper §5, Invariant I2) -------------------------------
    # Required class attribute on every concrete subclass. Enforced by
    # ``__init_subclass__`` below. CONTEXT = Σ_ctx (read-only); ACTION = Σ_act
    # (effectful). No default — every skill must declare which it is.
    effect_type: "EffectType" = None  # type: ignore[assignment]

    # -- Subclass validation -------------------------------------------------

    def __init_subclass__(cls, **kwargs: Any) -> None:
        super().__init_subclass__(**kwargs)
        # Concrete subclasses (those that implement build_subtree and have no
        # remaining abstract methods) must declare effect_type. Intermediate
        # abstract bases are skipped.
        is_concrete = not getattr(cls, "__abstractmethods__", frozenset())
        if is_concrete and not isinstance(cls.effect_type, EffectType):
            raise TypeError(
                f"Skill subclass {cls.__name__} must declare a class "
                "attribute `effect_type: EffectType` (one of "
                "EffectType.CONTEXT, EffectType.ACTION). Per paper §5, "
                "every tool belongs to exactly one of Σ_ctx (read-only) "
                "or Σ_act (effectful)."
            )

    # -- BT construction (the core of every skill) --------------------------

    @abstractmethod
    def build_subtree(
        self,
        params: Dict[str, Any],
        context: SkillContext,
        handle: SkillHandle,
    ) -> py_trees.behaviour.Behaviour:
        """Build the BT subtree that implements this skill.

        The subtree's nodes update the ``handle`` with observations,
        result_data, and progress. The subtree should:
          - Return ``SUCCESS`` when the skill completes successfully
          - Return ``FAILURE`` when the skill fails
          - Return ``RUNNING`` while work is in progress
        """
        ...

    # -- Lifecycle -----------------------------------------------------------

    def prepare(
        self,
        params: Dict[str, Any],
        context: SkillContext,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        """Transform raw LLM parameters into execution-ready data.

        Returns:
            ``(prepared_params, error_or_None)``. When error is not None,
            execution is aborted and the error is fed back to the LLM.
        """
        return params, None

    def start(
        self,
        params: Dict[str, Any],
        context: SkillContext,
        state_provider: Optional[Callable[[], "SkillContext"]] = None,
    ) -> SkillHandle:
        """Start skill execution and return a handle for polling.

        Builds the subtree, wraps it with the framework safeguards
        (paper §5 Invariant I5), and ticks the wrapped tree in a
        background thread. For skills whose body completes in one tick,
        the thread exits immediately.

        ``state_provider`` is a callable returning a fresh ``SkillContext``
        on demand. Used by ``InvariantGuard`` to re-evaluate the per-skill
        ``safeguard_invariant`` predicate against current world state each
        tick (Lemma 5.5 hypothesis). If not provided, falls back to the
        start-time snapshot — acceptable for ContextSkills (no
        ``InvariantGuard``) but the start-time snapshot will be stale for
        ActionSkills under fast-changing state.
        """
        # Local import to avoid circular dependency: safeguard imports from protocol
        from bt_agent._engine.skills.safeguard import _wrap_safeguarding

        handle = SkillHandle()
        handle.status = SkillStatus.RUNNING
        body = self.build_subtree(params, context, handle)
        provider = state_provider if state_provider is not None else (lambda: context)
        wrapped = _wrap_safeguarding(self, body, handle, provider)
        handle.subtree = wrapped  # OBS-DROP: viz sees the wrapped tree
        threading.Thread(
            target=_tick_subtree_loop,
            args=(wrapped, handle),
            daemon=True,
        ).start()
        return handle

    # -- Gating --------------------------------------------------------------

    def is_available(
        self,
        state: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[bool, str]:
        """Check whether this skill may be selected this turn.

        Part of the paper's action mask (§5 Invariant I3):
        ``Σ^avail_t = Σ_ctx ∪ {σ ∈ Σ_act : is_available(history) ∧ precondition(state)}``.

        ``state`` contains the tool-call history for the current milestone
        (extracted from h^act_{m_i}). ``context`` carries turn-level info
        (milestone index, can_defer, etc.).

        **Default: always available.** Override in skills that need
        interaction-history gating (e.g., ordering constraints between
        skills within a milestone).
        """
        return True, f"{self.name} available"

    # -- K-publication hook (paper §5, Invariant I6) -------------------------

    def publish_results(self, handle: SkillHandle, store: "ObjectStore") -> None:
        """Hook: write select ``handle.result_data`` fields into the shared
        context store K.

        Paper §5 Invariant I6: ``R_K = ∅`` by default — skills opt in by
        overriding this method.  The framework calls it after the skill
        succeeds (from ``ExecuteSkill._apply_results`` and from
        ``planner._run_context_skill``).

        This hook is **orthogonal to the ctx/act partition**.  Both
        ContextSkills and ActionSkills may override it; most don't.
        ``handle.observation`` always flows into the reasoning trace *h*
        regardless — ``publish_results`` is only for structured data that
        OTHER skills need to consume via the ObjectStore (the K channel).

        The framework is intentionally agnostic to *what* a skill publishes
        and *how* it writes to ``ObjectStore``. The skill author owns both.
        This method is the only K-write seam between framework and skill code.

        **Default: no-op.** Override in skills that publish data to K.

        Args:
            handle: The skill's terminal ``SkillHandle`` (status SUCCEEDED).
                The override reads ``handle.result_data`` to find what to
                publish.
            store: The shared context store. Override calls ``store.add(...)``
                and/or other ``ObjectStore`` methods.

        Implementations should be defensive: ``handle.result_data`` may be
        empty or missing fields; check before iterating.

        New skill authors: if your skill produces objects or mappings that
        downstream skills need to reference (across-skill data), add a
        ``publish_results`` override here AND a class-level audit test
        confirming the override is in place. There is no runtime safety net.
        """

    # -- Tool definition -----------------------------------------------------

    def tool_def(self) -> Dict[str, Any]:
        """OpenAI/Anthropic-style tool definition this skill exposes to the agent.

        Default implementation builds the dict from the skill author's
        ``name`` / ``description`` / ``usage`` / ``params`` class attrs —
        no skill-specific framework code, no hardcoded names. Concrete
        skill subclasses MAY override to hand-write their tool def
        (e.g., for a non-string parameter type, or to omit the auto-added
        ``memory_note`` field).

        ``ContextSkill`` re-declares this as ``@abstractmethod`` so context
        skill authors are forced to think about what their tool exposes
        to the LLM (paper §5: skill author owns the tool surface, not the
        framework). ``ActionSkill`` does NOT re-declare it — action skills
        get a uniform actor-side tool surface and the default is fine.
        """
        desc = self.description
        if self.usage:
            desc += " " + self.usage

        properties: Dict[str, Any] = {}
        required: List[str] = []

        for param_name, param_desc in self.params.items():
            properties[param_name] = {"type": "string", "description": param_desc}
            required.append(param_name)

        properties["memory_note"] = {
            "type": "string",
            "description": "Brief note for your future self about key findings or state.",
        }

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": properties,
                    "required": required,
                },
            },
        }


# ---------------------------------------------------------------------------
# Effect-typed Skill bases (paper §5)
# ---------------------------------------------------------------------------
#
# Skill authors subclass one of these — never `Skill` directly. The base
# class sets the effect_type and declares the per-tier abstract API. The
# framework checks (precondition, safeguard_invariant); skills define them.
# Skill ABC's __init_subclass__ effect_type check above is now redundant
# (always satisfied by the base) but kept as a safety net for any future
# code that accidentally subclasses Skill directly.


class ContextSkill(Skill):
    """Σ_ctx skill base — read-only context-gathering tool (paper §5).

    May write to the shared context store K (ObjectStore) but must NOT
    mutate s_robot or the physical world. Always available per Invariant
    I3 — the framework does not ask context skills for a precondition or
    safeguard invariant; they are exempt by definition.

    **Per Step 6.5 (framework agnosticity):** ``tool_def()`` is
    ``@abstractmethod`` here — every ContextSkill MUST explicitly own its
    tool surface. The default implementation in ``Skill.tool_def()``
    remains usable: a ContextSkill can override with one line that calls
    ``super().tool_def()`` (it'll build from the class's ``name`` /
    ``description`` / ``usage`` / ``params`` attrs). The point of the
    abstractmethod is to force the skill author to make a deliberate
    choice instead of silently inheriting an empty default.
    """

    effect_type = EffectType.CONTEXT

    @abstractmethod
    def tool_def(self) -> Dict[str, Any]:
        """OpenAI/Anthropic-style tool definition this context skill exposes.

        See ``Skill.tool_def`` for the default implementation; overrides
        may delegate to ``super().tool_def()`` (this body forwards to
        ``Skill.tool_def(self)``) or hand-write a custom dict.

        Note: the ``@abstractmethod`` enforcement is checked at
        instantiation time via Python's ABC machinery — the body below
        is the fall-through used when subclasses delegate via ``super()``.
        """
        return Skill.tool_def(self)


class ActionSkill(Skill):
    """Σ_act skill base — effectful skill that may change s_robot or the world.

    Concrete subclasses MUST declare:

      - ``precondition(state) -> (bool, str)``         — I_σ, eligibility (this step)
      - ``safeguard_invariant(state, handle) -> bool`` — φ_σ, trajectory safety
      - ``build_subtree(...)``                          — μ_σ, the skill body

    The framework only checks these; it never defines them. Skill authors
    own the policy (what counts as eligible / safe); the framework owns
    the mechanism (when to ask, what to do with the answer).

    The framework auto-wraps every ``ActionSkill`` body inside ``Skill.start()``:

        Sequence(memory=False, [
            CancellationGuard,
            WallTimeBudgetGuard,
            InvariantGuard,
            build_subtree(),
        ])

    ``memory=False`` is required so guards re-evaluate every tick (Safe-Set
    Lifting Theorem hypothesis, Colledanchise & Ögren Lemma 5.5).
    """

    effect_type = EffectType.ACTION

    # Wall-time budget enforced by ``WallTimeBudgetGuard``. Override per skill.
    wall_time_budget_seconds: float = 120.0

    @abstractmethod
    def precondition(self, state: SkillContext) -> Tuple[bool, str]:
        """Eligibility predicate I_σ over current state.

        Cheap, offline, side-effect-free predicate evaluation (NOT a BT
        tick). Evaluated once by the scheduler before invocation to
        determine whether σ appears in the action mask
        ``Σ^avail_t = Σ_ctx ∪ {σ ∈ Σ_act : is_available(history) ∧ precondition(state)}``.

        May be incomplete — does NOT need to imply success or trajectory
        safety. Mid-execution failures are caught by ``safeguard_invariant``
        via the framework's ``InvariantGuard`` wrap, which re-evaluates
        on every tick.

        Returns:
            ``(True, "")`` if eligible; ``(False, reason)`` if not.
            The reason string is human-readable and may surface to the
            LLM (or to logs) explaining why the skill is masked out.
        """

    @abstractmethod
    def safeguard_invariant(
        self,
        state: SkillContext,
        handle: SkillHandle,
    ) -> Union[bool, str]:
        """Trajectory-safety predicate φ_σ — re-evaluated every tick by ``InvariantGuard``.

        Returning ``True`` lets the body continue. Returning ``False``
        aborts the in-flight skill within one tick with no typed reason
        (backward-compatible plain-bool contract — most skills use this).

        Returning a non-empty ``str`` also aborts the skill, but the string
        is treated as a typed ``failure_reason`` code (e.g. ``"stale_pose"``,
        ``"pose_lost"``, ``"path_invalidated"`` — see ``bt-nav-fixes`` CP5)
        that ``InvariantGuard`` writes into ``handle.result_data`` and that
        the actor guidance router keys off. Skills that don't need to
        distinguish *which* clause failed may keep returning plain ``bool``;
        the framework never requires the richer return.

        I5 discipline (NOT type-enforced — relies on author care):
            MUST consult only ``s_robot`` + observation projection
            (read from ``state.platform_state`` — e.g., position,
            perception feeds, sensor data) and skill execution state
            (``handle.result_data``).
            MUST NOT read ``state.object_store``, ``state.milestone``,
            ``state.main_task``, ``state.scenario`` — those are K, h, P
            projections that belong to other roles per Invariant I5.

        The ``handle`` argument exposes skill-execution artifacts
        (e.g., a path computed earlier in the body) so the predicate
        can detect "we still have what we needed to start" conditions.

        Defensive default (framework-side): any exception raised by this
        method is caught by ``InvariantGuard``, treated as ``False``, and
        emitted to the audit stream as ``safeguard_invariant_exception``
        (audit wiring lands in Step 8). Skill authors should NOT rely on
        exceptions for control flow — return ``False`` explicitly.

        Honest disclaimer: the framework cannot prevent an author from
        writing ``return True`` unconditionally. The Safe-Set Lifting
        Theorem is conditional on this predicate being a faithful
        encoding of the user-declared safe set.
        """
