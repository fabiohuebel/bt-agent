"""Recorded-view and shared-heading regressions; CPU only, no policy episodes."""

import json
import math
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import py_trees
import pytest
from bt_agent._engine.skills.protocol import SkillContext
from bt_agent_robotics.skills import navigate as nav
from bt_agent_robotics.skills.query import SceneObject
from scipy.spatial.transform import Rotation
from .test_navigate import (
    _FixedPathPlanner,
    _make_handle,
    _make_params,
    _navigate_tree,
    _StuckLocalPlanner,
    _tick,
)


def object_with(poses, cameras=None):
    return SimpleNamespace(label="fixture", all_viewpoints=poses, all_viewpoint_camera_ids=cameras or [""] * len(poses))


def pose(x=1.0, y=1.0, yaw=0.0):
    return [x, y, 0.44, *Rotation.from_euler("z", yaw).as_quat().tolist()]


def test_se2_diversity_preserves_raw_indices_and_collapses_duplicates():
    target = object_with([pose(), pose(), pose(yaw=math.pi / 2), pose(yaw=math.pi / 2)])
    result = nav._viewpoint_catalog(target)
    assert result["distinct_indices"] == [0, 2]
    assert result["valid_indices"] == [0, 1, 2, 3]
    assert result["effective_goals"][2]["theta"] == pytest.approx(math.pi / 2)


def test_position_heading_thresholds_and_wraparound():
    origin = dict(x=0.0, y=0.0, theta=0.0)
    assert not nav._same_se2_view(origin, dict(x=0.4, y=0.0, theta=0.0))
    assert nav._same_se2_view(origin, dict(x=0.399, y=0.0, theta=0.149))
    assert not nav._same_se2_view(origin, dict(x=0.0, y=0.0, theta=0.151))
    assert nav._same_se2_view(dict(x=0, y=0, theta=math.pi - 0.02), dict(x=0, y=0, theta=-math.pi + 0.02))


def test_different_camera_same_effective_body_pose_collapses():
    a, b = np.eye(4), np.eye(4)
    b[:3, :3] = Rotation.from_euler("z", -math.pi / 2).as_matrix()
    target = object_with([pose(), pose(yaw=math.pi / 2)], ["a", "b"])
    assert nav._viewpoint_catalog(target, {"a": a, "b": b})["distinct_indices"] == [0]


def test_malformed_viewpoints_not_offered_and_diagnostic_is_explicit():
    target = object_with([pose(), [1, 2], [1, 2, 3, 0, 0, 0, 0], pose(x=float("nan")), pose(yaw=1)])
    result = nav._viewpoint_catalog(target)
    assert result["distinct_indices"] == [0, 4]
    assert [x["raw_index"] for x in result["invalid_entries"]] == [1, 2, 3]


def recorded():
    x = json.loads((Path(__file__).parent / "fixtures/recorded_wallet_glasses_se2.json").read_text())
    tf = {}
    for name, item in x["tf"]["transforms"].items():
        t = np.eye(4)
        t[:3, :3] = Rotation.from_quat(item["rotation_xyzw"]).as_matrix()
        t[:3, 3] = item["translation"]
        tf[name] = np.linalg.inv(t)
    return x["objects"], tf


@pytest.mark.parametrize("index,expected", [(0, [0, 1]), (1, [0, 2])])
def test_saved_wallet_glasses_preserve_pose_and_expose_turn(index, expected):
    objects, tf = recorded()
    source = objects[index]
    obj = SceneObject.from_raw(
        "fixture",
        dict(
            object_label=source["label"],
            all_viewpoints=source["stored_viewpoints"],
            all_viewpoint_camera_ids=source["camera_ids"],
        ),
    )
    result = nav._viewpoint_catalog(obj, tf)
    assert result["distinct_indices"] == expected
    for actual, old in zip(result["effective_goals"], source["effective_se2_goals"]):
        for key in ("x", "y", "theta"):
            assert actual[key] == pytest.approx(old[key], abs=1e-10)
    # The selected default remains the exact original raw index0 pose.
    assert nav.compute_pose(obj, camera_to_body_transforms=tf) == pytest.approx(source["effective_se2_goals"][0])


class TurningPlanner(_StuckLocalPlanner):
    def __init__(self, turn=True):
        super().__init__(pose=(5, 5, 0))
        self.turn = turn
        self._pose_age = 0.0
        self.velocities = []

    def send_velocity(self, x, y, w):
        self.velocities.append((x, y, w))
        if self.turn:
            self._pose = (*self._pose[:2], self._pose[2] + 0.5 * w)


def test_already_positioned_turns_before_success_then_stops_and_brakes():
    lp = TurningPlanner()
    handle = _make_handle()
    params = _make_params(goal_theta=1.4)
    params["viewpoint_catalog"] = dict(distinct_indices=[0, 2])
    tree = _navigate_tree(lp, _FixedPathPlanner([(5, 5)]), handle, params=params)
    assert _tick(tree) == py_trees.common.Status.RUNNING
    assert any(w != 0 for _, _, w in lp.velocities)
    for _ in range(30):
        if _tick(tree) != py_trees.common.Status.RUNNING:
            break
    assert tree.status == py_trees.common.Status.SUCCESS
    assert abs(nav._yaw_difference(1.4, lp._pose[2])) <= nav._ALIGN_TOL_RAD
    assert handle.result_data["arrival_status"] == "already_at_goal"
    assert handle.result_data["heading_alignment"]["status"] == "aligned"
    assert "raw indices): [0, 2]" in handle.observation
    assert lp.brake_calls == [False, True]
    assert lp.velocities[-1] == (0, 0, 0)
    assert len([n for n in tree.iterate() if isinstance(n, nav._AlignHeading)]) == 1


def test_already_aligned_has_no_nonzero_turn():
    lp = TurningPlanner()
    handle = _make_handle()
    tree = _navigate_tree(lp, _FixedPathPlanner([(5, 5)]), handle)
    assert _tick(tree) == py_trees.common.Status.SUCCESS
    assert not any(w for _, _, w in lp.velocities)


def test_full_arrival_also_reaches_the_same_alignment(monkeypatch):
    # Physical-path leaves are test doubles; this checks the actual composite
    # wiring rather than creating a simulator or claiming a real navigation run.
    monkeypatch.setattr(nav._CheckAlreadyAtGoal, "update", lambda self: py_trees.common.Status.FAILURE)
    for cls in (nav._ValidateGoal, nav._PlanPath, nav._ConfirmPath, nav._FollowPath, nav._ConfirmArrival):
        monkeypatch.setattr(cls, "initialise", lambda self: None)
        monkeypatch.setattr(cls, "update", lambda self: py_trees.common.Status.SUCCESS)
        monkeypatch.setattr(cls, "terminate", lambda self, status: None)
    lp = TurningPlanner()
    handle = _make_handle()
    handle.result_data = {"arrival_status": "arrived"}
    tree = _navigate_tree(lp, _FixedPathPlanner([(5, 5)]), handle, params=_make_params(goal_theta=1.4))
    for _ in range(30):
        if _tick(tree) != py_trees.common.Status.RUNNING:
            break
    assert tree.status == py_trees.common.Status.SUCCESS
    assert handle.result_data["arrival_status"] == "arrived"
    assert handle.result_data["heading_alignment"]["within_tolerance"]
    assert len([n for n in tree.iterate() if isinstance(n, nav._AlignHeading)]) == 1


def test_alignment_timeout_is_explicit_not_false_aligned(monkeypatch):
    clock = [0.0]
    monkeypatch.setattr(nav.time, "monotonic", lambda: clock[0])
    lp = TurningPlanner(turn=False)
    handle = _make_handle()
    tree = _navigate_tree(lp, _FixedPathPlanner([(5, 5)]), handle, params=_make_params(goal_theta=1.4))
    assert _tick(tree) == py_trees.common.Status.RUNNING
    clock[0] = nav._ALIGN_TIMEOUT_S + 0.01
    assert _tick(tree) == py_trees.common.Status.SUCCESS  # existing best-effort semantics
    info = handle.result_data["heading_alignment"]
    assert info["status"] == "timeout" and info["timed_out"] and not info["within_tolerance"]
    assert info["residual_yaw_rad"] == pytest.approx(1.4)
    assert "Heading alignment: timeout" in handle.observation
    assert lp.brake_calls[-1] and lp.velocities[-1] == (0, 0, 0)


@pytest.mark.parametrize("interruption", ["cancel", "invalidate", "stale"])
def test_alignment_interruption_stops_and_brakes(interruption):
    lp = TurningPlanner(turn=False)
    handle = _make_handle()
    tree = _navigate_tree(lp, _FixedPathPlanner([(5, 5)]), handle, params=_make_params(goal_theta=1.4))
    assert _tick(tree) == py_trees.common.Status.RUNNING
    if interruption == "invalidate":
        tree.stop(py_trees.common.Status.INVALID)
        expected = "interrupted"
    else:
        if interruption == "cancel":
            handle.cancel()
            expected = "cancelled"
        else:
            lp._pose_age = 5.0
            expected = "stale_pose"
        assert _tick(tree) == py_trees.common.Status.FAILURE
    assert handle.result_data["heading_alignment"]["status"] == expected
    assert lp.brake_calls[-1] and lp.velocities[-1] == (0, 0, 0)


@pytest.mark.parametrize("bad", [None, "not-a-number", float("nan"), float("inf")])
@pytest.mark.parametrize("field", ["goal", "pose"])
def test_bad_heading_or_pose_terminates_without_nonzero_motion(bad, field):
    lp = TurningPlanner(turn=False)
    handle = _make_handle()
    handle.result_data = {"arrival_status": "arrived"}
    params = _make_params(goal_theta=bad if field == "goal" else 1.4)
    if field == "pose":
        lp._pose = (5, 5, bad)
    node = nav._AlignHeading(params, handle, lp)
    node.initialise()
    assert node.update() == py_trees.common.Status.FAILURE
    info = handle.result_data["heading_alignment"]
    assert info["status"] == "invalid_pose" and info["residual_yaw_rad"] is None
    assert not any(w for _, _, w in lp.velocities)
    json.dumps(info, allow_nan=False)


@pytest.mark.parametrize("valid_index", [2, 4])
def test_keepout_fallback_uses_explicit_noncontiguous_raw_indices(valid_index):
    poses = [pose(x=1), *([[0, 0]] * (valid_index - 1)), pose(x=4)]
    obj = object_with(poses)
    catalog = nav._viewpoint_catalog(obj)
    assert catalog["distinct_indices"] == [0, valid_index]
    context = SkillContext(platform_state={}, object_store=SimpleNamespace(get=lambda _: obj))
    params = _make_params(goal_x=1, goal_y=1)
    params["viewpoint_catalog"] = catalog
    params["total_viewpoints"] = 2  # Must not become range(2).

    class Planner(_FixedPathPlanner):
        has_keep_out = True

        def __init__(self):
            super().__init__([])
            self.goals = []

        def plan_strict(self, start, goal):
            self.goals.append(goal)
            return [goal] if goal[0] == 4 else None

    planner = Planner()
    node = nav._PlanPath(params, _make_handle(), planner, TurningPlanner(), context)
    path = node._plan_preferring_keep_out_clean_viewpoint((0, 0))
    assert path == [(4, 1)]
    assert params["viewpoint_index"] == valid_index
    assert planner.goals == [(1, 1), (4, 1)]
