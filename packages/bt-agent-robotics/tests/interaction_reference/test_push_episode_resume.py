import threading
from dataclasses import asdict
from types import SimpleNamespace

import py_trees
import pytest
from bt_agent_robotics._push import skill as push
from bt_agent_robotics._push.telemetry import RosPushTelemetry
from bt_agent_robotics._push.controller import Affordance, Observation

def fixture(progress=.315):
    asset=Affordance('dining_chair',(0,0),(.9215,0),(-.7,0,0),
                     tuple((-.7+1.4715*i/15,0,0) for i in range(1,16)),radius=.08)
    obs=Observation('same-episode',381,'dining_chair',700,10.,(progress,0),(progress-.7,0),'none',
                    episode_initial_xy=(0,0),episode_robot_travel=5.,episode_motion_observed=True)
    return asset,obs



def test_existing_path_nodes_receive_rebased_retry_approach(monkeypatch):
    monkeypatch.setenv("LAM_HABITAT_PUSH_EPISODE_RESUME", "1")
    monkeypatch.setattr(push, "time", SimpleNamespace(monotonic=lambda: 10.0))
    asset, observation = fixture()
    seen = []

    class CheckPlan(py_trees.behaviour.Behaviour):
        def __init__(self, params, *args):
            super().__init__("check-existing-plan")
            self.params = params

        def update(self):
            seen.append((self.params["goal_x"], self.params["goal_y"]))
            return py_trees.common.Status.RUNNING

    monkeypatch.setattr(push, "_PlanPath", CheckPlan)
    monkeypatch.setattr(push, "_FollowPath", lambda *args: py_trees.behaviours.Running("follow"))
    local = SimpleNamespace(
        get_robot_pose=lambda: (-0.7, 0, 0),
        get_pose_age=lambda: 0,
        push_telemetry=SimpleNamespace(
            begin=lambda _: dict(asset=asdict(asset), episode_token="same-episode", native_instance_id=381),
            drain=lambda now: [observation],
        ),
    )
    handle = SimpleNamespace(is_cancel_requested=False, result_data={})
    node = push._PushExecution(
        dict(target_handle="dining chair 1", target_label="dining chair"), None, handle, local, None
    )
    node.initialise()
    assert node.update() == py_trees.common.Status.RUNNING
    assert len(seen) == 1 and seen[0] == pytest.approx((-0.385, 0))
    assert node.controller.resumed


def test_telemetry_transforms_initialization_certificate_only_for_opt_in(monkeypatch):
    receiver = RosPushTelemetry.__new__(RosPushTelemetry)
    receiver._lock = threading.RLock()
    receiver._error = None
    receiver._selected = {"asset": {}}
    receiver._observed = True
    receiver._context = ("run", "task", 1)
    receiver._transform = SimpleNamespace(point=lambda p: (p[0], p[2], p[1]))
    data = dict(
        frame="habitat_y_up",
        robot_pose_correction=False,
        actual_pose=[0.315, 0.4, 0],
        robot_pose=[-0.385, 0, 0],
        native_instance_id=381,
        canonical_key="dining_chair",
        sample=700,
        activation="none",
        episode_initial_pose=[0, 0.4, 0],
        episode_robot_travel_m=5.0,
        episode_motion_observed=True,
    )
    receiver.buffer = SimpleNamespace(drain=lambda now: [({"evidence": data}, 10.0)])
    monkeypatch.setenv("LAM_HABITAT_PUSH_EPISODE_RESUME", "1")
    observation = receiver.drain(10.0)[0]
    assert observation.episode_initial_xy == (0, 0) and observation.episode_robot_travel == 5.0
    monkeypatch.setenv("LAM_HABITAT_PUSH_EPISODE_RESUME", "0")
    assert receiver.drain(10.0)[0].episode_initial_xy is None
