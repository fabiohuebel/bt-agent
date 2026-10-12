"""Tests for AgentRunner — generic BT lifecycle orchestrator."""

from __future__ import annotations

from typing import Any, Dict, List

import py_trees
from bt_agent._engine.agent.blackboard_keys import BB
from bt_agent._engine.agent.interface import AgentEvent, AgentInterface
from bt_agent._engine.agent.runner import AgentRunner

# ---------------------------------------------------------------------------
# Minimal stub agent
# ---------------------------------------------------------------------------


class _StubAgent(AgentInterface):
    """Stub that builds a trivial SUCCESS tree and tracks calls."""

    def __init__(self) -> None:
        self._bb = py_trees.blackboard.Client(name="StubAgent")
        self._bb.register_key(key=BB.CANCEL_REQUESTED, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.TASK, access=py_trees.common.Access.WRITE)
        self._bb.register_key(key=BB.SCENARIO, access=py_trees.common.Access.WRITE)
        self._bb.set(BB.CANCEL_REQUESTED, False)
        self._complete = False
        self._events: List[AgentEvent] = []
        self._outputs: List[Dict[str, Any]] = []
        self.reset_calls: List[tuple] = []

    def create_tree(self) -> py_trees.behaviour.Behaviour:
        return py_trees.behaviours.Success(name="stub_root")

    @property
    def blackboard(self) -> py_trees.blackboard.Client:
        return self._bb

    def reset_for_task(self, task: str, scenario: str) -> None:
        self.reset_calls.append((task, scenario))

    @property
    def is_task_complete(self) -> bool:
        return self._complete

    def register_skill(self, name: str, skill: Any) -> None:
        pass

    def request_cancel(self, source: str, reason: str = "") -> None:
        self._bb.set(BB.CANCEL_REQUESTED, True)

    def drain_events(self) -> List[AgentEvent]:
        result = list(self._events)
        self._events.clear()
        return result  # noqa: R504

    def drain_outputs(self) -> List[Dict[str, Any]]:
        result = list(self._outputs)
        self._outputs.clear()
        return result  # noqa: R504


# ---------------------------------------------------------------------------
# Tests
# ---------------------------------------------------------------------------


class TestAgentRunnerPrimitives:
    def test_push_state_sets_platform_state(self):
        agent = _StubAgent()
        runner = AgentRunner(agent)
        runner.push_state({"robot_position": (1.0, 2.0, 0.5)})

        bb = py_trees.blackboard.Client(name="reader")
        bb.register_key(key=BB.PLATFORM_STATE, access=py_trees.common.Access.READ)
        state = bb.get(BB.PLATFORM_STATE)
        assert state["robot_position"] == (1.0, 2.0, 0.5)

    def test_tick_increments_count(self):
        agent = _StubAgent()
        runner = AgentRunner(agent)
        runner.push_state({})
        assert runner.tick_count == 0
        runner.tick()
        assert runner.tick_count == 1
        runner.tick()
        assert runner.tick_count == 2

    def test_tick_returns_events_and_outputs(self):
        agent = _StubAgent()
        runner = AgentRunner(agent)
        runner.push_state({})

        agent._events.append(AgentEvent("test_event", {"key": "val"}))
        agent._outputs.append({"type": "test_output"})

        events, outputs = runner.tick()
        assert len(events) == 1
        assert events[0].kind == "test_event"
        assert len(outputs) == 1
        assert outputs[0]["type"] == "test_output"

    def test_reset_delegates_to_agent(self):
        agent = _StubAgent()
        runner = AgentRunner(agent)
        runner.push_state({})
        runner.tick()
        runner.tick()
        runner.reset("my task", "my scenario")
        assert agent.reset_calls[-1] == ("my task", "my scenario")
        assert runner.tick_count == 0

    def test_is_complete_mirrors_agent(self):
        agent = _StubAgent()
        runner = AgentRunner(agent)
        assert not runner.is_complete
        agent._complete = True
        assert runner.is_complete

    def test_is_cancelled_reads_blackboard(self):
        agent = _StubAgent()
        runner = AgentRunner(agent)
        assert not runner.is_cancelled
        agent.request_cancel("test")
        assert runner.is_cancelled


class TestAgentRunnerRun:
    def test_run_completes_on_task_complete(self):
        agent = _StubAgent()
        runner = AgentRunner(agent)
        runner.push_state({})

        # Mark complete after 2 ticks
        tick_count = [0]
        _orig_tick = runner.tick

        def _tick_and_mark():
            out = _orig_tick()
            tick_count[0] += 1
            if tick_count[0] >= 2:
                agent._complete = True
            return out  # noqa: R504

        runner.tick = _tick_and_mark

        result = runner.run("test", max_ticks=10, tick_rate=0.0)
        assert result is True

    def test_run_returns_false_on_max_ticks(self):
        agent = _StubAgent()
        runner = AgentRunner(agent)
        runner.push_state({})

        result = runner.run("test", max_ticks=3, tick_rate=0.0)
        assert result is False

    def test_run_calls_state_provider(self):
        agent = _StubAgent()
        runner = AgentRunner(agent)

        calls = []

        def provider():
            calls.append(1)
            return {"tick": len(calls)}

        # Will hit max_ticks
        runner.run("test", max_ticks=3, tick_rate=0.0, state_provider=provider)
        assert len(calls) == 3

    def test_run_calls_on_event_and_on_output(self):
        agent = _StubAgent()
        runner = AgentRunner(agent)
        runner.push_state({})

        received_events = []
        received_outputs = []

        # Inject an event on tick 1
        agent._events.append(AgentEvent("hello", {}))
        agent._outputs.append({"type": "submission"})

        agent._complete = True  # complete immediately

        runner.run(
            "test",
            max_ticks=5,
            tick_rate=0.0,
            on_event=received_events.append,
            on_output=received_outputs.append,
        )

        assert len(received_events) == 1
        assert received_events[0].kind == "hello"
        assert len(received_outputs) == 1
        assert received_outputs[0]["type"] == "submission"

    def test_run_stops_on_cancel(self):
        agent = _StubAgent()
        runner = AgentRunner(agent)
        runner.push_state({})

        tick_count = [0]
        _orig_tick = runner.tick

        def _tick_and_cancel():
            out = _orig_tick()
            tick_count[0] += 1
            if tick_count[0] >= 2:
                agent.request_cancel("test")
            return out  # noqa: R504

        runner.tick = _tick_and_cancel

        result = runner.run("test", max_ticks=10, tick_rate=0.0)
        assert result is False
