"""AgentInterface — the swappable contract between bridge and agent.

Any agent (LAMAgent, OpenClaw adapter, Claude Code adapter, test harness)
implements this interface. The bridge node only knows about AgentInterface;
it pushes world data to the blackboard, ticks the tree, and reads outputs.

Swapping agents:
    - LAMAgent: BT with LLM-based skill selection (default)
    - OpenClaw: Map WebSocket events to prompt()/get_events()
    - Claude Code: Map tool calls to skills
    - TestAgent: Scripted plans with mock skills
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import TYPE_CHECKING, Any, Dict, List, Optional

if TYPE_CHECKING:
    import py_trees
    from bt_agent._engine.skills.protocol import Skill


class AgentEvent:
    """Event emitted by the agent for external observers (GUI, logging, OpenClaw bridge).

    Attributes:
        kind: Event type — "skill_started", "skill_completed", "plan_generated",
              "milestone_advanced", "cancelled", "task_complete", etc.
        data: Arbitrary payload specific to the event kind.
    """

    __slots__ = ("kind", "data")

    def __init__(self, kind: str, data: Optional[Dict[str, Any]] = None):
        self.kind = kind
        self.data = data or {}

    def __repr__(self) -> str:
        return f"AgentEvent({self.kind!r}, {self.data!r})"


class AgentInterface(ABC):
    """Contract that ANY agent must fulfill to run inside a bridge.

    The bridge node:
      1. Calls ``create_tree()`` once at startup to get the py_trees root.
      2. Pushes world data to ``blackboard`` each tick.
      3. Calls ``py_trees.trees.BehaviourTree.tick()`` at 10Hz.
      4. Reads ``pending_outputs`` to dispatch ROS2 publications.
      5. Reads ``events`` for logging / GUI / external bridges.

    The agent:
      - Owns the BT structure and all node logic.
      - Receives physical skill implementations via ``register_skill()``.
      - Handles cancellation via ``request_cancel()``.
    """

    # -- Tree lifecycle ------------------------------------------------------

    @abstractmethod
    def create_tree(self) -> "py_trees.behaviour.Behaviour":
        """Build and return the root behaviour of the agent's BT.

        Called once by the bridge at startup. The bridge wraps this in a
        ``py_trees.trees.BehaviourTree`` and calls ``tick()`` each cycle.
        """
        ...

    @property
    @abstractmethod
    def blackboard(self) -> "py_trees.blackboard.Client":
        """Blackboard client used by the bridge to push world model updates.

        The bridge writes: platform_state (s_robot), task, scenario, etc.
        The agent's BT nodes read these keys during tick.
        """
        ...

    # -- Task lifecycle ------------------------------------------------------

    @abstractmethod
    def reset_for_task(self, task: str, scenario: str) -> None:
        """Reset all agent state for a brand-new task.

        Clears milestones, memory, conversation, and reinitializes the
        blackboard for a fresh run.
        """
        ...

    @property
    @abstractmethod
    def is_task_complete(self) -> bool:
        """Whether the current task has been completed or cancelled."""
        ...

    # -- Skill injection -----------------------------------------------------

    @abstractmethod
    def register_skill(self, name: str, skill: "Skill") -> None:
        """Register a skill implementation.

        Called by the bridge to inject physical skills (navigate, press, push,
        explore) that are backed by ROS2 action clients. Instant skills
        (query, image_analysis, capture, submit) are registered by the agent
        itself since they have no ROS2 dependencies.
        """
        ...

    # -- Cancellation --------------------------------------------------------

    @abstractmethod
    def request_cancel(self, source: str, reason: str = "") -> None:
        """Request cancellation of the current activity.

        Args:
            source: Who is cancelling — ``"human"``, ``"agent"``, or
                ``"task_replacement"``.
            reason: Human-readable explanation.
        """
        ...

    # -- Observability -------------------------------------------------------

    def drain_events(self) -> List[AgentEvent]:
        """Return and clear pending events since the last drain.

        Override to emit events for external observers (GUI, OpenClaw bridge,
        logging). Default implementation returns an empty list.
        """
        return []

    def drain_outputs(self) -> List[Dict[str, Any]]:
        """Return and clear pending output actions (captures, submissions).

        The bridge reads these each tick and publishes to ROS2 topics.
        Each dict has at minimum ``{"type": str, "data": dict}``.
        Default implementation returns an empty list.
        """
        return []

    # -- Human interaction (for chat-based agents) ---------------------------

    def inject_message(self, text: str) -> None:
        """Inject a human message into the agent's conversation.

        Used by OpenClaw / chat bridges. Default is a no-op for BT agents
        that don't support freeform human input mid-task.
        """
        pass
