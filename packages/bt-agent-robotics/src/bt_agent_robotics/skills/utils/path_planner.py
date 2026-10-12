"""PathPlanner ABC — abstract path planning backend.

Implementations produce a list of 2D waypoints from start to goal.
The NavigateSkill doesn't know or care which backend is used.

Two backends:
    NavGraphPlanner — Dijkstra on a NetworkX navigation graph (Habitat)
    GridPlanner     — A* on an occupancy grid (Isaac) [future]
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, List, Optional, Tuple


class PathPlanner(ABC):
    """Compute a list of 2D waypoints from start to goal."""

    @abstractmethod
    def plan(
        self,
        start_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
    ) -> Optional[List[Tuple[float, float]]]:
        """Plan a path from start to goal.

        Returns:
            List of (x, y) waypoints in world coordinates, or None
            if no path exists.
        """
        ...

    @abstractmethod
    def replan(
        self,
        robot_xy: Tuple[float, float],
        goal_xy: Tuple[float, float],
    ) -> Optional[List[Tuple[float, float]]]:
        """Find an alternative path around a blockage.

        Called when the robot is stuck on a waypoint. The implementation
        should mark the current area as blocked and find a new route.

        Returns:
            New waypoint list, or None if no alternative exists.
        """
        ...

    @abstractmethod
    def update_map(self, map_data: Any) -> None:
        """Update the underlying map data.

        Called before planning to ensure the planner has current data.
        The implementation should handle diffing internally (skip reload
        if data hasn't changed).

        Args:
            map_data: Backend-specific map data. NavGraphPlanner expects
                a node_link_data dict. GridPlanner would expect an
                occupancy grid array.
        """
        ...

    @property
    @abstractmethod
    def waypoint_threshold(self) -> float:
        """Distance (m) at which a waypoint is considered reached."""
        ...
