"""NavGraphPlanner — Dijkstra shortest path on a NetworkX navigation graph.

Ported from lam_core Navigator class. Pure Python + NetworkX, no ROS2.

Graph format: NetworkX node_link_data JSON.
    Node attributes: {"id": int, "world": [x, y, z]}
    Edge attributes: {"source": int, "target": int, "weight": float}
"""

from __future__ import annotations

import logging
import math
from typing import Any, List, Optional, Tuple

import networkx as nx
from bt_agent_robotics.skills.utils.path_planner import PathPlanner

logger = logging.getLogger(__name__)


# A trigger this close to the start is under the robot (it is standing on
# it). The trigger it just pressed is exempt by IDENTITY (set_last_pressed):
# a distance rule at ~0.9 m confused the neighbor 1 m away once the robot had
# stepped back a little short of its viewpoint.
# otherwise the robot cannot leave.
_KEEP_OUT_START_EXEMPT_M = 0.5
# Waypoint tracking near a keep-out point: _FollowPath normally advances to
# the next waypoint within 0.45 m, which on a 0.5 m lattice lets the robot
# track a straight row up to 0.45 m off the planned line -- toward a panel,
# that is the difference between 0.8 m and 0.35 m of clearance (green fired
# on the blue->red leg of final_t3_safe_meds_glasses that way). Within
# radius + _KEEP_OUT_TIGHT_BAND_M of any keep-out point the threshold drops
# to _KEEP_OUT_TIGHT_THRESHOLD_M so the robot actually visits those waypoints.
# 0.3 m = 2x the path follower's stopDisThre (0.15 m, local_planner_habitat_
# params.yaml); 0.2 m was too close to it and the robot never registered
# arrival at its first waypoint (final_t3_safe_meds_glasses r4, every leg
# failed). _FollowPath also keeps the default for the FIRST waypoint (the
# node nearest the robot, which the local planner may treat as reached).
_KEEP_OUT_TIGHT_THRESHOLD_M = 0.3
_KEEP_OUT_TIGHT_BAND_M = 0.6
# When the keep-out disconnects the goal, plan with keep-out nodes PENALISED
# instead of dropping the keep-out: the route then crosses the fewest
# keep-out nodes (typically only the target's own, or the last few metres to
# an enclosed viewpoint) rather than walking straight over a neighbour panel.
_KEEP_OUT_PENALTY_M = 10.0


def _penalised_weight(hidden: set):
    """Dijkstra weight function: the edge's own weight plus
    _KEEP_OUT_PENALTY_M for every hidden endpoint."""

    def weight(u, v, data):
        base = float(data.get("weight", 1.0))
        return base + _KEEP_OUT_PENALTY_M * ((u in hidden) + (v in hidden))

    return weight


class NavGraphPlanner(PathPlanner):
    """Dijkstra path planning on a navigation graph.

    The graph is loaded from a NetworkX ``node_link_data`` dict.
    On load, only the largest connected component is retained.
    Replanning removes the edge near the robot and re-runs Dijkstra.
    """

    def __init__(
        self,
        threshold: float = 0.45,
        max_replan_attempts: int = 3,
    ) -> None:
        # threshold: _FollowPath advances to the next waypoint once the robot is
        # within it. 0.45 m (nodes are 0.5 m apart) keeps the robot on the
        # planned polyline; the old 0.8 m let it cut corners by most of a metre,
        # which is how a keep-out-clean route still stepped on a floor trigger.
        self._threshold = threshold
        self._max_replan_attempts = max_replan_attempts
        self._graph: Optional[nx.Graph] = None
        # Pristine copy of the loaded graph, taken at update_map() time.
        # replan() permanently removes edges from self._graph to steer
        # Dijkstra away from a blocked edge; without a restore point, a
        # stationary robot that keeps getting stuck (e.g. immobilized) can
        # accumulate edge removals across many navigate attempts until the
        # graph is disconnected and plan() fails forever. self._graph is
        # restored from this copy at the start of every fresh top-level
        # plan() call (never mid-replan — see _in_replan below).
        self._pristine_graph: Optional[nx.Graph] = None
        self._edges_removed = False
        self._in_replan = False
        self._replan_count = 0
        self._last_blocked_edge: Optional[Tuple[int, int]] = None
        # Cache the map_data hash to skip redundant reloads
        self._loaded_hash: Optional[int] = None
        # Interactable keep-out: (x, y, label) points (floor triggers such as
        # press panels) that plan() routes around unless they are the goal.
        self._keep_out: List[Tuple[float, float, str]] = []
        self._keep_out_radius = 1.0
        self._keep_out_goal_exempt = 0.45
        self._active_target: Optional[str] = None
        self._last_pressed: Optional[str] = None

    def set_last_pressed(self, handle: Optional[str]) -> None:
        """Name the trigger the robot just pressed: exempt from the keep-out
        on the following legs regardless of how far the robot stepped back."""
        self._last_pressed = (handle or "").strip().lower() or None

    def set_active_target(self, handle: Optional[str]) -> None:
        """Name the handle the current leg heads to (navigate/press target) so
        its own keep-out point is exempt. Skills set it before plan()."""
        self._active_target = (handle or "").strip().lower() or None

    def set_keep_out(
        self,
        points: List[Tuple[float, float, str]],
        radius: float = 1.0,
        goal_exempt_radius: float = 0.45,
    ) -> None:
        """Route around *points* (e.g. floor press panels): nodes within
        ``radius`` of a keep-out point are hidden from Dijkstra. Default 1.0 m:
        the final scene's panels sit 1 m apart on a 0.5 m lattice, and the
        lattice row 0.8 m from the panel line is where the robot, tracking
        within 0.3 m, still came within 0.5 m of a neighbour panel. Exempt: the
        start and goal nodes, nodes within ``goal_exempt_radius`` of the goal,
        the keep-out point within 1 m of the start (the panel just pressed, from
        its own viewpoint after stepping back)
        and the point whose label equals the active target
        (``set_active_target``, the panel being approached). Empty list
        disables the behavior."""
        self._keep_out = [(float(x), float(y), str(label)) for x, y, label in points]
        self._keep_out_radius = float(radius)
        self._keep_out_goal_exempt = float(goal_exempt_radius)

    def _keep_out_hidden_nodes(
        self,
        start_node: int,
        goal_node: int,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
    ) -> set:
        hidden = set()
        if not self._keep_out or self._graph is None:
            return hidden
        # A keep-out point that belongs to this leg is exempt: the trigger the
        # robot just pressed (it stands on it or at its viewpoint, <= 1 m) and the
        # trigger named as this leg's target (set_active_target). Distance to
        # the goal is deliberately NOT used: panels sit 1 m apart and a panel's
        # viewpoint can be closer to a neighbor than to the panel itself.
        active = [
            (kx, ky, label)
            for kx, ky, label in self._keep_out
            if math.hypot(kx - start_xy[0], ky - start_xy[1]) > _KEEP_OUT_START_EXEMPT_M
            and (self._active_target is None or label.strip().lower() != self._active_target)
            and (self._last_pressed is None or label.strip().lower() != self._last_pressed)
        ]
        if not active:
            return hidden
        for node_id, attrs in self._graph.nodes(data=True):
            if node_id in (start_node, goal_node):
                continue
            world = attrs.get("world", [0, 0, 0])
            if math.hypot(world[0] - goal_xy[0], world[1] - goal_xy[1]) <= self._keep_out_goal_exempt:
                continue
            for kx, ky, _label in active:
                if math.hypot(world[0] - kx, world[1] - ky) <= self._keep_out_radius:
                    hidden.add(node_id)
                    break
        return hidden

    @property
    def waypoint_threshold(self) -> float:
        return self._threshold

    def keep_out_clearance(self, waypoints: List[Tuple[float, float]]) -> float:
        """Smallest distance from any waypoint to any keep-out point that is
        NOT exempt for the current leg (active target / last pressed). inf
        when there is no keep-out or no non-exempt point. navigate uses it to
        pick, among viewpoints that are all only reachable through the
        keep-out, the route that stays farthest from a neighbour trigger."""
        active = [
            (kx, ky)
            for kx, ky, label in self._keep_out
            if (self._active_target is None or label.strip().lower() != self._active_target)
            and (self._last_pressed is None or label.strip().lower() != self._last_pressed)
        ]
        if not active or not waypoints:
            return math.inf
        return min(math.hypot(x - kx, y - ky) for x, y in waypoints for kx, ky in active)

    def waypoint_threshold_at(self, xy: Tuple[float, float]) -> float:
        """Arrival threshold for the waypoint at *xy*: the default, or the
        tight _KEEP_OUT_TIGHT_THRESHOLD_M when *xy* lies within keep-out
        radius + _KEEP_OUT_TIGHT_BAND_M of any keep-out point (exempt ones
        included: tracking tightly near the panel being approached costs
        nothing and keeps its neighbours clear)."""
        if not self._keep_out:
            return self._threshold
        band = self._keep_out_radius + _KEEP_OUT_TIGHT_BAND_M
        for kx, ky, _label in self._keep_out:
            if math.hypot(kx - xy[0], ky - xy[1]) <= band:
                return min(self._threshold, _KEEP_OUT_TIGHT_THRESHOLD_M)
        return self._threshold

    def update_map(self, map_data: Any) -> None:
        """Load from a node_link_data dict. Skips reload if data unchanged.

        Tolerant of empty / wrong-shaped input: callers in the BT path
        sometimes pass the scene graph or an empty dict here, which we
        silently ignore (the previously-loaded graph stays valid).
        """
        if not map_data:
            return  # None or empty dict — nothing to load
        if not isinstance(map_data, (dict, str)):
            return

        # Identity check — skip if same object already loaded
        data_hash = id(map_data)
        if data_hash == self._loaded_hash and self._graph is not None:
            return

        if isinstance(map_data, str):
            import json

            try:
                map_data = json.loads(map_data)
            except (json.JSONDecodeError, ValueError):
                return

        # Sanity gate — must look like NetworkX node_link_data
        if not isinstance(map_data, dict) or "nodes" not in map_data:
            return  # not a nav graph (e.g. a scene graph passed by mistake)

        try:
            # NetworkX renamed the edges key over its 3.x line: 3.3 wants
            # "links", 3.4+ defaults to "edges". Try as-is first, then fall
            # back to renaming.
            try:
                raw = nx.node_link_graph(map_data)
            except KeyError:
                if "links" in map_data and "edges" not in map_data:
                    renamed = dict(map_data)
                    renamed["edges"] = renamed.pop("links")
                    raw = nx.node_link_graph(renamed)
                elif "edges" in map_data and "links" not in map_data:
                    renamed = dict(map_data)
                    renamed["links"] = renamed.pop("edges")
                    raw = nx.node_link_graph(renamed)
                else:
                    raise
            if raw.number_of_nodes() == 0:
                logger.warning("Loaded empty navigation graph")
                return

            # Keep only the largest connected component
            undirected = raw.to_undirected() if raw.is_directed() else raw
            largest_cc = max(nx.connected_components(undirected), key=len)
            self._graph = raw.subgraph(largest_cc).copy()
            self._pristine_graph = self._graph.copy()
            self._edges_removed = False
            self._loaded_hash = data_hash
            self._replan_count = 0

            removed = raw.number_of_nodes() - self._graph.number_of_nodes()
            if removed > 0:
                logger.info(
                    "Nav graph: %d nodes retained, %d disconnected removed",
                    self._graph.number_of_nodes(),
                    removed,
                )
            else:
                logger.info("Nav graph loaded: %d nodes", self._graph.number_of_nodes())
        except Exception as e:
            logger.error("Failed to load navigation graph: %s", e)

    @property
    def has_keep_out(self) -> bool:
        return bool(self._keep_out)

    def plan_strict(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
    ) -> Optional[List[Tuple[float, float]]]:
        """Like plan(), but returns None instead of falling back to the plain
        graph when the keep-out leaves no route (used by navigate to try a
        target's other viewpoints before accepting a route over a trigger)."""
        return self.plan(start_xy, goal_xy, strict=True)

    def plan(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
        strict: bool = False,
    ) -> Optional[List[Tuple[float, float]]]:
        """Plan path via Dijkstra. Returns list of (x, y) waypoints.

        Fresh top-level calls (i.e. NOT the internal call replan() makes
        after removing a blocked edge) restore the graph from the pristine
        copy first, undoing any edge removals accumulated by earlier
        replan() escalations. This bounds the damage a stuck robot can do
        to the graph to "within one navigate attempt" instead of "forever".
        """
        if self._graph is None:
            return None

        if not self._in_replan and self._edges_removed:
            self._graph = self._pristine_graph.copy()
            self._edges_removed = False

        start_node = self._find_closest_node(start_xy)
        goal_node = self._find_closest_node(goal_xy)

        if start_node is None or goal_node is None:
            return None

        graph = self._graph
        hidden = self._keep_out_hidden_nodes(start_node, goal_node, start_xy, goal_xy)
        if hidden:
            graph = nx.restricted_view(self._graph, hidden, [])
        try:
            path_nodes = nx.dijkstra_path(graph, start_node, goal_node, weight="weight")
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            if hidden and strict:
                return None
            if hidden:
                # Keep-out made the goal unreachable: keep the keep-out as a
                # penalty (every edge into a hidden node costs
                # _KEEP_OUT_PENALTY_M extra) so the route crosses as few
                # keep-out nodes as possible, rather than dropping it.
                logger.warning(
                    "Keep-out (%d nodes hidden) left no path to goal %s; planning with keep-out as a penalty",
                    len(hidden),
                    goal_xy,
                )
                try:
                    path_nodes = nx.dijkstra_path(
                        self._graph,
                        start_node,
                        goal_node,
                        weight=_penalised_weight(hidden),
                    )
                except (nx.NetworkXNoPath, nx.NodeNotFound):
                    return None
            else:
                return None

        if not path_nodes:
            return None

        self._replan_count = 0
        return self._nodes_to_waypoints(path_nodes)

    def replan(
        self,
        robot_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
    ) -> Optional[List[Tuple[float, float]]]:
        """Remove edge near robot and re-plan."""
        if self._graph is None:
            return None

        self._replan_count += 1
        if self._replan_count > self._max_replan_attempts:
            logger.warning("Replan limit reached (%d attempts)", self._max_replan_attempts)
            return None

        # Find the two closest nodes to the robot and remove the edge between them
        closest = self._find_closest_node(robot_xy)
        if closest is None:
            return None

        # Remove edges from the closest node to its neighbors (pick the nearest neighbor)
        neighbors = list(self._graph.neighbors(closest))
        if neighbors:
            # Find neighbor closest to the robot (likely the edge we're stuck on)
            best_neighbor = min(
                neighbors,
                key=lambda n: self._node_distance(n, robot_xy),
            )
            if self._graph.has_edge(closest, best_neighbor):
                self._graph.remove_edge(closest, best_neighbor)
                self._edges_removed = True
                logger.warning(
                    "Replan %d/%d: removed edge %s-%s",
                    self._replan_count,
                    self._max_replan_attempts,
                    closest,
                    best_neighbor,
                )

        self._in_replan = True
        try:
            return self.plan(robot_xy, goal_xy)
        finally:
            self._in_replan = False

    # -- Internal helpers ------------------------------------------------------

    def _find_closest_node(self, xy: Tuple[float, float]) -> Optional[int]:
        """Find the graph node closest to (x, y)."""
        if self._graph is None or self._graph.number_of_nodes() == 0:
            return None

        best_node = None
        best_dist = float("inf")

        for node_id, attrs in self._graph.nodes(data=True):
            world = attrs.get("world", [0, 0, 0])
            dx = world[0] - xy[0]
            dy = world[1] - xy[1]
            dist = math.hypot(dx, dy)
            if dist < best_dist:
                best_dist = dist
                best_node = node_id

        return best_node

    def _node_distance(self, node_id: int, xy: Tuple[float, float]) -> float:
        """Euclidean distance from a graph node to (x, y)."""
        world = self._graph.nodes[node_id].get("world", [0, 0, 0])
        return math.hypot(world[0] - xy[0], world[1] - xy[1])

    def _nodes_to_waypoints(self, path_nodes: List[int]) -> List[Tuple[float, float]]:
        """Convert node IDs to (x, y) world coordinates."""
        waypoints = []
        for node_id in path_nodes:
            world = self._graph.nodes[node_id].get("world", [0, 0, 0])
            waypoints.append((float(world[0]), float(world[1])))
        return waypoints
