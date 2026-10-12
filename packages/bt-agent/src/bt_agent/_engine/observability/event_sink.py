"""Compatibility imports for the maintained event broker.

The original implementation is preserved in the paper source snapshot. This
release fixes accepted-event loss during shutdown; see core-patches.json.
"""
from bt_agent.tracing import EventSink, SinkEvent

__all__ = ["EventSink", "SinkEvent"]
