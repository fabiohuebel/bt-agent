"""AgentRunner — generic BT lifecycle orchestrator.

The runner manages the py_trees BehaviourTree, pushes world state to the
blackboard, ticks the tree, and drains events/outputs.  It does NOT
create agents, register skills, or know about any substrate.

Two usage patterns:

**Blocking loop** (test harnesses, CLI scripts)::

    agent = LAMAgent(model, ...)
    agent.register_skill("query", QuerySkill(sg_agent))
    agent.register_skill("navigate", NavigateSkill(planner, path))

    runner = AgentRunner(agent)
    runner.push_state(initial_platform_state)
    success = runner.run("Find the red mug", scenario="Office")

**Consumer-controlled loop** (ROS2 bridges, event-driven adapters)::

    runner = AgentRunner(agent)

    # In a 10 Hz timer callback:
    runner.push_state(sensor_data)
    events, outputs = runner.tick()
    for o in outputs:
        dispatch_to_ros(o)
"""

from __future__ import annotations

import logging
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

import py_trees
from bt_agent._engine.agent.blackboard_keys import BB
from bt_agent._engine.agent.interface import AgentEvent, AgentInterface

logger = logging.getLogger(__name__)


class AgentRunner:
    """Generic BT loop for any :class:`AgentInterface` implementation.

    The consumer (test harness, ROS2 bridge, OpenClaw adapter, …):
      1. Creates the agent and registers skills.
      2. Passes the configured agent to ``AgentRunner``.
      3. Calls :meth:`run` for a blocking loop **or**
         :meth:`push_state` + :meth:`tick` for a consumer-controlled loop.
    """

    def __init__(self, agent: AgentInterface) -> None:
        self._agent = agent
        root = agent.create_tree()
        self._tree = py_trees.trees.BehaviourTree(root=root)

        # Bridge-side BB client — owns the world-model key and reads
        # cancellation state for the is_cancelled property.
        self._bb = py_trees.blackboard.Client(name="AgentRunner")
        self._bb.register_key(
            key=BB.PLATFORM_STATE,
            access=py_trees.common.Access.WRITE,
        )
        self._bb.register_key(
            key=BB.CANCEL_REQUESTED,
            access=py_trees.common.Access.READ,
        )

        self._tick_count = 0

    # -- Properties ----------------------------------------------------------

    @property
    def agent(self) -> AgentInterface:
        """The wrapped agent (for advanced access)."""
        return self._agent

    @property
    def tree(self) -> py_trees.trees.BehaviourTree:
        """The py_trees BehaviourTree (for snapshot / debug)."""
        return self._tree

    @property
    def tick_count(self) -> int:
        return self._tick_count

    @property
    def is_complete(self) -> bool:
        return self._agent.is_task_complete

    @property
    def is_cancelled(self) -> bool:
        try:
            return self._bb.get(BB.CANCEL_REQUESTED)
        except KeyError:
            return False

    # -- Primitives (consumer-controlled loop) -------------------------------

    def push_state(self, platform_state: Dict[str, Any]) -> None:
        """Push substrate-specific world state to the blackboard.

        The framework is agnostic to the shape of *platform_state*.
        Skills read what they need via ``context.platform_state.get(key)``.
        """
        self._bb.set(BB.PLATFORM_STATE, platform_state)

    def tick(self) -> Tuple[List[AgentEvent], List[Dict[str, Any]]]:
        """Execute a single BT tick and drain events + outputs.

        Returns:
            Tuple of (events, outputs) produced during this tick.
        """
        self._tree.tick()
        self._tick_count += 1
        events = self._agent.drain_events()
        outputs = self._agent.drain_outputs()
        return events, outputs

    def reset(self, task: str, scenario: str = "") -> None:
        """Reset the agent for a new task.

        Order matters: we stop the BT *before* clearing the blackboard.

        Why: threaded BT leaves (PlanTask, SelectSkill, EvaluateCompletion)
        each spawn a daemon thread for their LLM call. py_trees considers
        such a node "running" until its update() returns SUCCESS/FAILURE.
        If we cleared the blackboard while one of these threads was still
        in-flight, the thread would eventually return and (with the
        epoch/cancelled guards added to those leaves on 2026-05-22) drop
        its result on the floor — but py_trees still thinks the node is
        running and would re-tick it next time around. Calling
        ``root.stop(INVALID)`` propagates ``terminate(INVALID)`` down to
        every RUNNING leaf, which sets each leaf's ``_cancelled`` flag and
        resets py_trees-side state so the next tick re-enters
        ``initialise()`` cleanly for the new task.

        This is the runtime half of the T5→T6 boundary fix; the BB-clear
        below is the data half. Neither alone is sufficient.
        """
        try:
            self._tree.root.stop(py_trees.common.Status.INVALID)
        except Exception:  # noqa: BLE001 — defensive: never let a stop failure block reset
            logger.exception("AgentRunner.reset: tree.root.stop raised; continuing")
        self._agent.reset_for_task(task, scenario)
        self._tick_count = 0

    # -- Blocking loop -------------------------------------------------------

    def run(
        self,
        task: str,
        scenario: str = "",
        *,
        max_ticks: int = 50,
        tick_rate: float = 0.5,
        state_provider: Optional[Callable[[], Dict[str, Any]]] = None,
        on_event: Optional[Callable[[AgentEvent], None]] = None,
        on_output: Optional[Callable[[Dict[str, Any]], None]] = None,
    ) -> bool:
        """Run the agent loop until completion, cancellation, or timeout.

        Args:
            task: High-level task description.
            scenario: Environment / scenario description.
            max_ticks: Maximum number of BT ticks before giving up.
            tick_rate: Seconds between ticks.
            state_provider: Called each tick to get fresh platform_state.
                If ``None``, the caller must have called :meth:`push_state`
                before :meth:`run`.
            on_event: Callback for each :class:`AgentEvent`.
            on_output: Callback for each output action dict.

        Returns:
            ``True`` if the task completed, ``False`` on timeout or cancel.
        """
        self.reset(task, scenario)

        for tick_idx in range(1, max_ticks + 1):
            if state_provider is not None:
                self.push_state(state_provider())

            events, outputs = self.tick()

            for event in events:
                if on_event is not None:
                    on_event(event)

            for output in outputs:
                if on_output is not None:
                    on_output(output)

            if self.is_complete:
                logger.info("Task completed after %d ticks", tick_idx)
                return True

            if self.is_cancelled:
                logger.info("Task cancelled after %d ticks", tick_idx)
                return False

            time.sleep(tick_rate)

        logger.warning("Max ticks reached (%d)", max_ticks)
        return False
