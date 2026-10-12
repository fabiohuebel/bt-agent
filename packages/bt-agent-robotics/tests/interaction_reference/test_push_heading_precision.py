from types import SimpleNamespace

import py_trees
from bt_agent_robotics._push.navigation import _AlignHeading


class Planner:
    def __init__(self):
        self.commands = []

    def get_robot_pose(self):
        return (0, 0, 0.1)

    def get_pose_age(self):
        return 0

    def send_velocity(self, *values):
        self.commands.append(values)

    def cancel(self):
        self.commands.append("cancel")


def align(params):
    planner = Planner()
    handle = SimpleNamespace(is_cancel_requested=False, observation="arrived", result_data={})
    node = _AlignHeading(params, handle, planner)
    node.initialise()
    return node, handle, planner


def test_default_navigation_heading_acceptance_is_preserved():
    node, handle, planner = align({"goal_theta": 0})
    assert node.update() == py_trees.common.Status.SUCCESS
    assert handle.result_data["heading_alignment"]["tolerance_rad"] == 0.15


def test_push_can_require_precise_heading_using_the_existing_rotation_primitive():
    node, handle, planner = align({"goal_theta": 0, "heading_tolerance_rad": 0.02})
    assert node.update() == py_trees.common.Status.RUNNING
    assert planner.commands[-1][0:2] == (0, 0) and planner.commands[-1][2] < 0


def test_precision_override_cannot_relax_the_existing_tolerance():
    node, handle, planner = align({"goal_theta": 0, "heading_tolerance_rad": 0.3})
    assert node.update() == py_trees.common.Status.FAILURE
    assert handle.result_data["heading_alignment"]["within_tolerance"] is False
