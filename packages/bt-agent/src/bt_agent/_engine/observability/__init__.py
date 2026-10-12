"""Transport-agnostic observability primitives.

This package contains the broker (``EventSink``) and snapshot helpers used
by the live visualization tooling. Pure Python, no Foxglove or MCAP deps —
those live in ``lam_v2_testing.viz``.

Everything in this package, plus every line tagged ``# OBS-DROP:`` or
wrapped in ``# ---- OBS-DROP-BEGIN/END ----`` blocks elsewhere in the
codebase, is purely instrumentation. See :mod:`bt_agent._engine.observability.markers`
for the strip recipe.
"""

from bt_agent._engine.observability.event_sink import EventSink, SinkEvent
from bt_agent._engine.observability.fg_server import FoxgloveWsServer
from bt_agent._engine.observability.phase import derive_phase
from bt_agent._engine.observability.tick_snapshot import (
    DEFAULT_BB_KEYS,
    attach_skill_subtree,
    snapshot_blackboard,
    snapshot_tree,
)
from bt_agent._engine.observability.timeline_recorder import TimelineRecorder
from bt_agent._engine.observability.viz_bridge import VizBridge

__all__ = [
    "EventSink",
    "FoxgloveWsServer",
    "SinkEvent",
    "TimelineRecorder",
    "VizBridge",
    "derive_phase",
    "snapshot_tree",
    "snapshot_blackboard",
    "attach_skill_subtree",
    "DEFAULT_BB_KEYS",
]
