"""Offline fixtures for examples/tests. These are not benchmark results."""

from copy import deepcopy
import json

import py_trees

from .skills import ContextSkill, SkillObject


class ScriptedModel:
    """Return explicit synthetic responses and record model-level requests."""

    def __init__(self, turns):
        self.turns = list(turns)
        self.requests = []

    def query_tool_call(self, messages, tools, **kwargs):
        self.requests.append(deepcopy({"messages": messages, "tools": tools, **kwargs}))
        if not self.turns:
            raise RuntimeError("Offline model fixture exhausted")
        name, arguments, thought = self.turns.pop(0)
        return (
            name,
            arguments,
            {
                "choices": [
                    {
                        "finish_reason": "stop",
                        "message": {
                            "role": "assistant",
                            "content": thought,
                            "tool_calls": [
                                {
                                    "id": f"fixture-{len(self.requests)}",
                                    "type": "function",
                                    "function": {"name": name, "arguments": json.dumps(arguments)},
                                }
                            ],
                        },
                    }
                ]
            },
        )

    def query(self, *args, **kwargs):
        raise AssertionError("Unexpected extra model call in the offline fixture")


class InspectShelf(ContextSkill):
    """Example authored BT: a single observation leaf, no physical sensing."""

    name = "inspect_shelf"
    description = "Read the fixture shelf contents. This is an offline example."
    params = {}

    def tool_def(self):
        return super().tool_def()

    def build_subtree(self, params, context, handle):
        class Inspect(py_trees.behaviour.Behaviour):
            def update(self):
                handle.observation = "The shelf contains a red mug."
                return py_trees.common.Status.SUCCESS

        return Inspect("inspect_shelf")

    def publish_results(self, handle, store):
        store.add(SkillObject(handle="red_mug", label="red mug on the shelf"))


def shelf_model():
    return ScriptedModel(
        [
            ("inspect_shelf", {}, "I should inspect the shelf before reporting."),
            (
                "report",
                {"text": "The shelf contains a red mug."},
                "I can report the observed contents.",
            ),
            ("claim_complete", {}, "The requested inspection and report are complete."),
        ]
    )
