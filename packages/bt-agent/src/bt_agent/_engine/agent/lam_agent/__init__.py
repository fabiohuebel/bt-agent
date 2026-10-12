"""LAMAgent — the default BT-based agent implementation.

Uses py_trees behavior tree with LLM-driven skill selection.
This is the concrete agent that the bridge creates for autonomous
robot task execution.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

import py_trees
from bt_agent._engine.agent.blackboard_keys import BB
from bt_agent._engine.agent.config import AgentConfig
from bt_agent._engine.agent.interface import AgentEvent, AgentInterface
from bt_agent._engine.bt.tree_builder import build_agent_tree
from bt_agent._engine.skills.object_store import ObjectStore
from bt_agent._engine.skills.protocol import Skill
from bt_agent._engine.skills.registry import SkillRegistry


class LAMAgent(AgentInterface):
    """BT agent with VLM-based planning and skill selection.

    The bridge:
      1. Calls ``create_tree()`` to get the root BT node.
      2. Pushes world data to ``self.blackboard`` each tick.
      3. Calls ``tree.tick()`` at 10Hz.
      4. Calls ``drain_events()`` / ``drain_outputs()`` to dispatch results.

    Physical skills are injected via ``register_skill()`` before the first tick.
    """

    def __init__(
        self,
        model_backend: Any,
        event_sink: Any = None,
        config: Optional[AgentConfig] = None,
        amend_cap: int = 2,
        prompt_variant: str = "tiamat_agent_submit_v5",
        milestone_schema_extension: Optional[Dict[str, Any]] = None,
        milestone_fields_guidance: str = "",
    ) -> None:
        self._config = config or AgentConfig()
        self._bb = py_trees.blackboard.Client(name="LAMAgent")
        self._registry = SkillRegistry()
        self._object_store = ObjectStore()
        self._model = model_backend
        self._event_sink = event_sink  # OBS-DROP: visualizer pubsub broker, may be None
        # Config takes precedence over individual kwargs if config is provided
        self._amend_cap = self._config.amend_cap if config else amend_cap
        self._prompt_variant = self._config.prompt_variant if config else prompt_variant
        self._milestone_schema_extension = milestone_schema_extension
        self._milestone_fields_guidance = milestone_fields_guidance

        # Register BB keys that the agent manages
        self._bb.register_key(key=BB.TASK, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.SCENARIO, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.MILESTONES, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.CURRENT_MILESTONE_IDX, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.COMPLETED_MILESTONES, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.DEFERRED_MILESTONES, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.COMPLETION_CLAIMED, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.CANCEL_REQUESTED, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.CANCEL_SOURCE, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.CANCEL_REASON, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.TASK_COMPLETE, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.TASK_SUBMITTED, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.SKILL_REGISTRY, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.MODEL_BACKEND, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.PENDING_EVENTS, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.PENDING_OUTPUTS, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.SKILL_HANDLE, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.OBJECT_STORE, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.CONVERSATION, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.PLAN_REASONING, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.EVENT_SINK, access=py_trees.common.Access.WRITE)  # OBS-DROP
        # Step 3: three-way verdict + amend bookkeeping (paper §5 I7).
        self._bb.register_key(key=BB.LAST_VERDICT, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.AMEND_COUNT, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.AMEND_CAP, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.AMEND_NOTE, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.REPLAN_CONTEXT, access=py_trees.common.Access.WRITE)
        # Structural-fix bundle (docs/spec/lam-improvement.md; gated). Registered
        # unconditionally (harmless — only the gated nodes read them), reset per task.
        self._bb.register_key(key=BB.EMPTY_PLAN_COUNT, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.REACT_FALLBACK_ACTIVE, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.CORRECTIVE_REPLAN_DONE, access=py_trees.common.Access.WRITE)
        # "context" ablation give-up marker (CP3 review round); reset per task.
        self._bb.register_key(key=BB.CONTEXT_OVERFLOW_GIVE_UP, access=py_trees.common.Access.WRITE)
        # Step 5: which system_prompts variant to use for planner/actor/evaluator
        self._bb.register_key(key=BB.PROMPT_VARIANT, access=py_trees.common.Access.WRITE)
        # Extensible milestone schema: substrate adds custom fields + prompt guidance
        self._bb.register_key(key=BB.MILESTONE_SCHEMA_EXTENSION, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.MILESTONE_FIELDS_GUIDANCE, access=py_trees.common.Access.WRITE)

        # Agent config (all user-tunable constants)
        self._bb.register_key(key=BB.AGENT_CONFIG, access=py_trees.common.Access.WRITE)

        # Initialize shared services on the blackboard
        self._bb.set(BB.AGENT_CONFIG, self._config)
        self._bb.set(BB.SKILL_REGISTRY, self._registry)
        self._bb.set(BB.MODEL_BACKEND, self._model)
        self._bb.set(BB.OBJECT_STORE, self._object_store)
        self._bb.set(BB.EVENT_SINK, self._event_sink)  # OBS-DROP: None if viz not enabled
        self._bb.set(BB.PENDING_EVENTS, [])
        self._bb.set(BB.PENDING_OUTPUTS, [])
        self._bb.set(BB.PLAN_REASONING, "")

        # Initialize state
        self._bb.set(BB.CANCEL_REQUESTED, False)
        self._bb.set(BB.TASK_COMPLETE, False)
        self._bb.set(BB.TASK_SUBMITTED, False)
        self._bb.set(BB.COMPLETION_CLAIMED, False)
        self._bb.set(BB.SKILL_HANDLE, None)
        self._bb.set(BB.CONVERSATION, [])

        # Step 3: three-way verdict state (cap is constant; the rest reset per task/milestone)
        self._bb.set(BB.AMEND_CAP, self._amend_cap)
        self._bb.set(BB.AMEND_COUNT, 0)
        self._bb.set(BB.AMEND_NOTE, None)
        self._bb.set(BB.LAST_VERDICT, None)
        self._bb.set(BB.REPLAN_CONTEXT, None)
        # Structural-fix bundle state (reset per task; only read when gated on).
        self._bb.set(BB.EMPTY_PLAN_COUNT, 0)
        self._bb.set(BB.REACT_FALLBACK_ACTIVE, False)
        self._bb.set(BB.CORRECTIVE_REPLAN_DONE, False)
        self._bb.set(BB.CONTEXT_OVERFLOW_GIVE_UP, False)
        # Step 5: write the prompt variant once (constant for the agent's lifetime).
        self._bb.set(BB.PROMPT_VARIANT, self._prompt_variant)
        # Extensible milestone schema (constant for the agent's lifetime).
        self._bb.set(BB.MILESTONE_SCHEMA_EXTENSION, self._milestone_schema_extension)
        self._bb.set(BB.MILESTONE_FIELDS_GUIDANCE, self._milestone_fields_guidance)

    # -- AgentInterface: Tree lifecycle ----------------------------------------

    def create_tree(self) -> py_trees.behaviour.Behaviour:
        # Gate the structural-fix branches (CP2 react-fallback, CP3 all_done
        # submit) on the config flag. False (every frozen variant) returns the
        # byte-identical frozen tree. ``ablation`` (empty by default, also
        # byte-identical) branches the evaluator/replan routes independently.
        return build_agent_tree(structural_fixes=self._config.structural_fixes, ablation=self._config.ablation)

    @property
    def blackboard(self) -> py_trees.blackboard.Client:
        return self._bb

    # -- AgentInterface: Task lifecycle ----------------------------------------

    def reset_for_task(self, task: str, scenario: str) -> None:
        self._bb.set(BB.TASK, task)
        self._bb.set(BB.SCENARIO, scenario)

        # Clear planning state
        self._bb.set(BB.MILESTONES, [])
        self._bb.set(BB.CURRENT_MILESTONE_IDX, 0)
        self._bb.set(BB.COMPLETED_MILESTONES, set())
        self._bb.set(BB.DEFERRED_MILESTONES, [])

        # Clear execution state
        self._bb.set(BB.COMPLETION_CLAIMED, False)
        self._bb.set(BB.SKILL_HANDLE, None)

        # Clear cancellation / completion
        self._bb.set(BB.CANCEL_REQUESTED, False)
        self._bb.set(BB.CANCEL_SOURCE, "")
        self._bb.set(BB.CANCEL_REASON, "")
        self._bb.set(BB.TASK_COMPLETE, False)
        self._bb.set(BB.TASK_SUBMITTED, False)

        # Clear conversation history
        self._bb.set(BB.CONVERSATION, [])

        # Clear object store
        self._object_store.reset()

        # Clear planner state
        self._bb.set(BB.PLAN_REASONING, "")

        # Clear output queues
        self._bb.set(BB.PENDING_EVENTS, [])
        self._bb.set(BB.PENDING_OUTPUTS, [])

        # Step 3: reset three-way verdict state (AMEND_CAP stays at constructor value)
        self._bb.set(BB.LAST_VERDICT, None)
        self._bb.set(BB.AMEND_COUNT, 0)
        self._bb.set(BB.AMEND_NOTE, None)
        self._bb.set(BB.REPLAN_CONTEXT, None)
        # Structural-fix bundle state.
        self._bb.set(BB.EMPTY_PLAN_COUNT, 0)
        self._bb.set(BB.REACT_FALLBACK_ACTIVE, False)
        self._bb.set(BB.CORRECTIVE_REPLAN_DONE, False)
        self._bb.set(BB.CONTEXT_OVERFLOW_GIVE_UP, False)

    @property
    def is_task_complete(self) -> bool:
        try:
            return self._bb.get(BB.TASK_COMPLETE)
        except KeyError:
            return False

    # -- AgentInterface: Skill injection ---------------------------------------

    def register_skill(self, name: str, skill: Skill) -> None:
        self._registry.register(skill)

    # -- AgentInterface: Cancellation ------------------------------------------

    def request_cancel(self, source: str, reason: str = "") -> None:
        self._bb.set(BB.CANCEL_REQUESTED, True)
        self._bb.set(BB.CANCEL_SOURCE, source)
        self._bb.set(BB.CANCEL_REASON, reason)

    # -- AgentInterface: Observability -----------------------------------------

    def drain_events(self) -> List[AgentEvent]:
        raw_events = self._bb.get(BB.PENDING_EVENTS) or []
        self._bb.set(BB.PENDING_EVENTS, [])
        return [AgentEvent(kind=e["kind"], data=e.get("data", {})) for e in raw_events]

    def drain_outputs(self) -> List[Dict[str, Any]]:
        outputs = self._bb.get(BB.PENDING_OUTPUTS) or []
        self._bb.set(BB.PENDING_OUTPUTS, [])
        return outputs  # noqa: R504
