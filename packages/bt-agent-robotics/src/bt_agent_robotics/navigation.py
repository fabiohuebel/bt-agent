"""Graph path planning for supplied maps, independent of ROS and benchmark tasks.

Install the ``navigation`` extra. Coordinates are already in the application's
map frame; this planner performs no camera/robot/scene-frame conversion.
"""

from ._nav_graph_planner import NavGraphPlanner

__all__ = ["NavGraphPlanner"]
