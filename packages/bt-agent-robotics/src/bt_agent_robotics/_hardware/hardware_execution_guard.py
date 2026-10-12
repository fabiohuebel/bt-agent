"""Hardware execution mask enforcement without changing frozen core sources."""

import hashlib
import threading
from pathlib import Path

import py_trees
from bt_agent._engine.agent.blackboard_keys import BB
from bt_agent._engine.bt.nodes._conversation import extract_tool_calls
from bt_agent._engine.bt.nodes.execute_skill import ExecuteSkill
from bt_agent._engine.bt.nodes.select_skill import SelectSkill
from bt_agent.skills import EffectType, SkillHandle, SkillStatus


class DecisionMask:
    def __init__(self):
        self.call_id = None
        self.offered = frozenset()


class GuardedSelectSkill(SelectSkill):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask
        self._offered_by_epoch = {}
        self._mask_lock = threading.Lock()

    def _select_in_background(self, epoch, model, tools, *args, **kwargs):
        names = frozenset(tool.get("function", {}).get("name") for tool in tools)
        with self._mask_lock:
            self._offered_by_epoch[epoch] = names
        return super()._select_in_background(epoch, model, tools, *args, **kwargs)

    def update(self):
        decision = self._result[0] if self._result is not None else None
        status = super().update()
        if status == py_trees.common.Status.SUCCESS and decision == "CONTINUE":
            with self._mask_lock:
                names = self._offered_by_epoch.pop(self._epoch, frozenset())
                self._offered_by_epoch = {k: v for k, v in self._offered_by_epoch.items() if k > self._epoch}
            self.mask.call_id = self._bb_get(BB.SKILL_CALL_ID, "")
            self.mask.offered = names
        return status


class GuardedExecuteSkill(ExecuteSkill):
    def __init__(self, mask):
        super().__init__()
        self.mask = mask
        self.blackboard.register_key(key=BB.CONVERSATION, access=py_trees.common.Access.READ)

    def initialise(self):
        if self._bb_get(BB.COMPLETION_CLAIMED) or self._bb_get(BB.TASK_COMPLETE):
            return super().initialise()
        name = self._bb_get(BB.SELECTED_SKILL, "")
        registry = self._bb_get(BB.SKILL_REGISTRY)
        skill = registry.get(name) if registry is not None else None
        if skill is None:
            return super().initialise()  # Existing unknown-skill rejection.
        call_id = self._bb_get(BB.SKILL_CALL_ID, "")
        offered = self.mask.call_id == call_id and name in self.mask.offered
        reason = "Tool was not offered for this decision; choose a currently available tool."
        current_ok = True
        try:
            if skill.effect_type == EffectType.ACTION:
                history = {"tool_calls": extract_tool_calls(self._bb_get(BB.CONVERSATION, []) or [])}
                milestones = self._bb_get(BB.MILESTONES, []) or []
                gating = {"current_idx": self._bb_get(BB.CURRENT_MILESTONE_IDX, 0), "total_milestones": len(milestones)}
                current_ok, detail = skill.is_available(history, gating)
                if current_ok:
                    current_ok, detail = skill.precondition(self._build_context())
                if not current_ok:
                    reason = detail
        except Exception as error:
            current_ok = False
            reason = f"Execution-condition check failed: {type(error).__name__}: {error}"
        if offered and current_ok:
            return super().initialise()

        # Reject BEFORE prepare(), start(), model calls or motion side effects.
        self._completion_bypass = False
        self._skill, self._skill_name = skill, name
        self._handle = SkillHandle()
        self._handle.status = SkillStatus.FAILED
        self._handle.observation = f"[execution_guard] Rejected '{name}' before execution: {reason}"
        self._handle.result_data = {
            "failure_reason": "skill_not_available",
            "execution_rejected": True,
            "was_offered": offered,
            "current_conditions_satisfied": current_ok,
        }
        self.blackboard.set(BB.SKILL_HANDLE, self._handle)
        events = self._bb_get(BB.PENDING_EVENTS, []) or []
        events.append({
            "kind": "skill_rejected",
            "data": {"skill": name, "reason": reason, "was_offered": offered, "skill_call_id": call_id},
        })
        self.blackboard.set(BB.PENDING_EVENTS, events)
        # Inherited update() reports FAILURE as feedback and continues the agent.
        return None


def install_execution_guard(root):
    selected = [node for node in root.iterate() if type(node) is SelectSkill]
    executed = [node for node in root.iterate() if type(node) is ExecuteSkill]
    if len(selected) != 1 or len(executed) != 1:
        raise RuntimeError("Execution guard requires exactly one SelectSkill and ExecuteSkill")
    mask = DecisionMask()
    for old, new in ((selected[0], GuardedSelectSkill(mask)), (executed[0], GuardedExecuteSkill(mask))):
        if old.parent is None:
            raise RuntimeError("Cannot replace a root execution node")
        old.parent.replace_child(old, new)
    return {
        "enabled": True,
        "selector": "GuardedSelectSkill",
        "executor": "GuardedExecuteSkill",
        "source_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "scope": "offered tool snapshot and fresh action conditions; before prepare/start",
    }
