"""ReportSkill — log a finding / give-up answer back to the operator (ContextSkill).

Read-only: many inspection tasks end in a report rather than a capture ("the
requested item is not visible from any reachable viewpoint"). The binary goal predicate can read
the report log (``goals.reported(substring)``). Mirrors ALFWorld's look/inventory
context skills — no world effect. Only failure is an empty report.

# Substrate-specific deviation from the Σ_ctx "no world mutation" contract
#
# The pycore protocol (bt_agent._engine.skills.protocol.ContextSkill) defines Σ_ctx skills as
# read-only: they may write to the shared context store K (ObjectStore) but must NOT
# mutate ``s_robot`` or the physical world.  ReportSkill calls bridge.step("report ..."),
# which appends a string to the substrate's ``env.state.reports`` list.  This mutates a
# bookkeeping list inside TextEnv — not a physical robot state and not the physical world.
# The deviation is accepted because:
#   1. The reports list is symbolic bookkeeping (it feeds the goal predicate, not physics).
#   2. Appending to reports is monotone: prior context is never erased by this call.
#   3. The text rung has no physical world; Σ_ctx's "no world mutation" guarantee is
#      interpreted here as "no change to the scored physical state of the scene".
# Future substrates with a real physical layer must ensure report does not trigger physics.
"""

from __future__ import annotations

from typing import Any, Dict

import py_trees
from bt_agent._engine.skills.protocol import ContextSkill, SkillContext, SkillHandle

from ._helpers import observe_with_reason, step_bridge


class ReportSkill(ContextSkill):
    name = "report"
    description = (
        "State a finding back to the operator — use when the task asks you to check something or "
        "when an objective cannot be photographed and needs a written answer instead."
    )
    usage = (
        "Use to answer a state-check or to report that a target is unreachable / has no viewpoint. "
        "When reporting reachability, state explicitly whether the target CAN or CANNOT be reached, "
        "and name the target the way the task names it."
    )
    params = {
        "text": "The finding to report (a full sentence).",
    }

    def build_subtree(
        self,
        params: Dict[str, Any],
        context: SkillContext,
        handle: SkillHandle,
    ) -> py_trees.behaviour.Behaviour:
        return _ReportNode(params, context, handle)

    def tool_def(self) -> Dict[str, Any]:
        """The inherited Skill.tool_def() surface is correct for ReportSkill.

        The base builds the definition from name/description/usage/params class
        attributes, which fully describes the report interface to the LLM.
        No hand-written override is needed; super() is called to satisfy the
        ContextSkill @abstractmethod requirement.
        """
        return super().tool_def()


class _ReportNode(py_trees.behaviour.Behaviour):
    def __init__(self, params: Dict[str, Any], context: SkillContext, handle: SkillHandle) -> None:
        super().__init__(name="report")
        self._params = params
        self._context = context
        self._handle = handle

    def update(self) -> py_trees.common.Status:
        text = str(self._params.get("text", "")).strip()
        if not text:
            self._handle.observation = "No text provided to report."
            return py_trees.common.Status.FAILURE

        obs, ok, reason = step_bridge(self._context, f"report {text}")
        self._handle.observation = observe_with_reason(obs, reason if not ok else None)
        return py_trees.common.Status.SUCCESS if ok else py_trees.common.Status.FAILURE
