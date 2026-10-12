import hashlib
import json
from pathlib import Path
import subprocess
import sys

import pytest

from bt_agent_robotics.navigation import NavGraphPlanner


def grid():
    nodes = [{"id": x * 10 + y, "world": [x * 0.5, y * 0.5, 0]} for x in range(7) for y in range(7)]
    links = []
    for x in range(7):
        for y in range(7):
            for xx, yy in ((x + 1, y), (x, y + 1)):
                if xx < 7 and yy < 7:
                    links.append(dict(source=x * 10 + y, target=xx * 10 + yy, weight=0.5))
    return dict(directed=False, multigraph=False, graph={}, nodes=nodes, links=links)


def planner():
    value = NavGraphPlanner()
    value.update_map(grid())
    return value


def test_source_matches_frozen_reference():
    import bt_agent_robotics._nav_graph_planner as source

    root = Path(__file__).resolve().parents[3]
    record = json.loads((root / "reproduction/navigation-source.json").read_text())
    assert (
        hashlib.sha256(Path(source.__file__).read_bytes()).hexdigest() == record["extracted_sha256"]
    )


def test_navigation_import_needs_no_ros_or_benchmark():
    code = """
import importlib.abc, sys
class Block(importlib.abc.MetaPathFinder):
    def find_spec(self, fullname, *args):
        if fullname.split('.')[0] in {'rclpy', 'robot_benchmark', 'robot_benchmark_habitat',
                                      'habitat_sim', 'lam_v2_tiamat', 'tiamat_text'}:
            raise RuntimeError(fullname)
sys.meta_path.insert(0, Block())
from bt_agent_robotics.navigation import NavGraphPlanner
assert NavGraphPlanner().waypoint_threshold == .45
"""
    subprocess.run([sys.executable, "-I", "-c", code], check=True)


def test_no_map_is_no_path():
    p = NavGraphPlanner()
    assert p.plan((0, 0), (3, 3)) is None


@pytest.mark.parametrize("shape", ["links", "edges", "json"])
def test_graph_input_formats_and_real_coordinates(shape):
    data = grid()
    if shape == "edges":
        data["edges"] = data.pop("links")
    if shape == "json":
        data = json.dumps(data)
    p = NavGraphPlanner()
    p.update_map(data)
    route = p.plan((0, 0), (3, 3))
    assert route[0] == (0, 0) and route[-1] == (3, 3)
    assert p.waypoint_threshold == 0.45


def test_disconnected_component_is_not_selected():
    data = grid()
    data["nodes"].append(dict(id=999, world=[50, 50, 0]))
    p = NavGraphPlanner()
    p.update_map(data)
    assert 999 not in p._graph


def test_stuck_attempt_does_not_destroy_later_navigation():
    p = planner()
    original = p._graph.number_of_edges()
    for _ in range(6):
        assert p.plan((0, 0), (3, 3))
        p.replan((0, 0), (3, 3))
        assert p._graph.number_of_edges() < original
        p.replan((0, 0), (3, 3))
        p.replan((0, 0), (3, 3))
    assert p.plan((0, 0), (3, 3))
    assert p._graph.number_of_edges() == original


def test_keepout_avoids_other_panels_and_tightens_tracking():
    p = planner()
    p.set_keep_out([(1.5, 1.5, "green")], radius=0.6, goal_exempt_radius=0.1)
    route = p.plan((1.5, 0), (1.5, 3))
    assert all((x - 1.5) ** 2 + (y - 1.5) ** 2 > 0.6**2 for x, y in route)
    assert p.waypoint_threshold_at((1.5, 1.5)) == 0.3
    assert p.waypoint_threshold_at((0, 0)) == 0.45


def test_named_target_and_last_pressed_are_exempt():
    p = planner()
    p.set_keep_out([(1.5, 1, "green"), (1.5, 2, "red")], radius=0.6)
    p.set_active_target(" green ")
    p.set_last_pressed("RED")
    route = p.plan((1.5, 0), (1.5, 3))
    assert (1.5, 1) in route and (1.5, 2) in route


def test_strict_vs_penalised_fallback():
    p = planner()
    p.set_keep_out([(x * 0.5, 1.5, str(x)) for x in range(7)], radius=0.3, goal_exempt_radius=0.1)
    assert p.plan_strict((1.5, 0), (1.5, 3)) is None
    route = p.plan((1.5, 0), (1.5, 3))
    assert route and len([xy for xy in route if xy[1] == 1.5]) == 1
