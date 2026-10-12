"""BT condition nodes — stateless checks that return SUCCESS or FAILURE.

These are the "guards" in the tree. They read the blackboard and make
binary decisions. They never modify state or run long operations.
"""

import py_trees
from bt_agent._engine.agent.blackboard_keys import BB


class CheckCancellation(py_trees.behaviour.Behaviour):
    """FAILURE if cancellation has been requested, SUCCESS otherwise.

    Sits at the top of the root Sequence. When it fails, py_trees calls
    terminate() on all running siblings, propagating cancellation through
    the entire tree.
    """

    def __init__(self) -> None:
        super().__init__(name="CheckCancellation")
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=BB.CANCEL_REQUESTED, access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        try:  # noqa: SIM105
            if self.blackboard.get(BB.CANCEL_REQUESTED):
                return py_trees.common.Status.FAILURE
        except KeyError:
            pass
        return py_trees.common.Status.SUCCESS


class CheckHasPlan(py_trees.behaviour.Behaviour):
    """SUCCESS if milestones exist on the blackboard."""

    def __init__(self) -> None:
        super().__init__(name="CheckHasPlan")
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=BB.MILESTONES, access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        try:
            milestones = self.blackboard.get(BB.MILESTONES)
            if milestones and len(milestones) > 0:
                return py_trees.common.Status.SUCCESS
        except KeyError:
            pass
        return py_trees.common.Status.FAILURE


class CheckMilestonesRemain(py_trees.behaviour.Behaviour):
    """SUCCESS if the current milestone is uncompleted and its deps are met."""

    def __init__(self) -> None:
        super().__init__(name="CheckMilestonesRemain")
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=BB.MILESTONES, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.CURRENT_MILESTONE_IDX, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.COMPLETED_MILESTONES, access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        try:
            milestones = self.blackboard.get(BB.MILESTONES)
            idx = self.blackboard.get(BB.CURRENT_MILESTONE_IDX)
            completed = self.blackboard.get(BB.COMPLETED_MILESTONES)
            if idx < len(milestones) and idx not in completed:
                # Verify dependencies are satisfied before entering
                deps = milestones[idx].get("dependencies", [])
                if deps:
                    dep_indices = {i for i, m in enumerate(milestones) if m["id"] in deps}
                    if not dep_indices.issubset(completed):
                        return py_trees.common.Status.FAILURE
                return py_trees.common.Status.SUCCESS
        except KeyError:
            pass
        return py_trees.common.Status.FAILURE


class CheckAllMilestonesDone(py_trees.behaviour.Behaviour):
    """SUCCESS if every milestone index is in the completed set."""

    def __init__(self) -> None:
        super().__init__(name="CheckAllMilestonesDone")
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=BB.MILESTONES, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.COMPLETED_MILESTONES, access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        try:
            milestones = self.blackboard.get(BB.MILESTONES)
            completed = self.blackboard.get(BB.COMPLETED_MILESTONES)
            if len(completed) >= len(milestones):
                return py_trees.common.Status.SUCCESS
        except KeyError:
            pass
        return py_trees.common.Status.FAILURE


class CheckCompletionClaimed(py_trees.behaviour.Behaviour):
    """SUCCESS if the LLM has returned TASK_COMPLETE for this milestone."""

    def __init__(self) -> None:
        super().__init__(name="CheckCompletionClaimed")
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=BB.COMPLETION_CLAIMED, access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        try:  # noqa: SIM105
            if self.blackboard.get(BB.COMPLETION_CLAIMED):
                return py_trees.common.Status.SUCCESS
        except KeyError:
            pass
        return py_trees.common.Status.FAILURE


class CheckReactFallbackActive(py_trees.behaviour.Behaviour):
    """SUCCESS iff the CP2 react-fallback plan is active.

    Structural-fix only — this node is added to the tree solely when
    ``structural_fixes`` is on, so the frozen variants never evaluate it.
    """

    def __init__(self) -> None:
        super().__init__(name="CheckReactFallbackActive")
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=BB.REACT_FALLBACK_ACTIVE, access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        try:  # noqa: SIM105
            if self.blackboard.get(BB.REACT_FALLBACK_ACTIVE):
                return py_trees.common.Status.SUCCESS
        except KeyError:
            pass
        return py_trees.common.Status.FAILURE


class CheckSkillRunning(py_trees.behaviour.Behaviour):
    """SUCCESS if a skill handle exists and is still RUNNING.

    This lets the BT skip SelectSkill and go straight to ExecuteSkill
    when a multi-tick skill is in progress.
    """

    def __init__(self) -> None:
        super().__init__(name="CheckSkillRunning")
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=BB.SKILL_HANDLE, access=py_trees.common.Access.READ)

    def update(self) -> py_trees.common.Status:
        try:
            handle = self.blackboard.get(BB.SKILL_HANDLE)
            if handle is not None and not handle.is_terminal:
                return py_trees.common.Status.SUCCESS
        except KeyError:
            pass
        return py_trees.common.Status.FAILURE
