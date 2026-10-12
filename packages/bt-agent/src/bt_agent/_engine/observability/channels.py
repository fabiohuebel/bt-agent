"""Channel registry — maps Foxglove channel names to JSON schemas + sink filters.

Each channel says which ``EventSink`` channels feed it. Some channels mirror
a single sink topic (``/agent/phase`` ← ``phase`` events); others aggregate
many (``/agent/event_log`` swallows everything).

The schemas are loose — extra keys are permitted. They exist so Foxglove can
infer types for the Raw Messages panel and for MCAP indexing.
"""

from __future__ import annotations

from typing import Any, Dict, List

# JSON Schemas (Draft 2020-12 compatible; Foxglove uses Draft 7). Kept loose
# (``additionalProperties: true``) so we can add fields without breaking
# old recordings.

_PHASE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "title": "AgentPhase",
    "properties": {
        "tick": {"type": "integer"},
        "phase": {
            "type": "string",
            "enum": [
                "IDLE",
                "PLANNING",
                "ACTING",
                "EXECUTING",
                "EVALUATING",
                "ADVANCING",
                "CANCELLED",
                "DONE",
            ],
        },
        "task": {"type": "string"},
        "scenario": {"type": "string"},
        "milestone_idx": {"type": "integer"},
        "milestone_total": {"type": "integer"},
        "current_skill": {"type": "string"},
    },
}

# Foxglove Studio's JSON Schema parser does NOT support recursive $ref, so the
# BT tree node schema is left permissive (additionalProperties=true). The shape
# is documented in code via tick_snapshot.snapshot_tree() and the custom
# Foxglove panel reads it directly.
_BT_SNAPSHOT_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "title": "BtSnapshot",
    "additionalProperties": True,
    "properties": {
        "tick": {"type": "integer"},
        "root": {"type": "object", "additionalProperties": True},
    },
    "required": ["tick", "root"],
}

_SKILL_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "title": "SkillState",
    "properties": {
        "name": {"type": "string"},
        "status": {"type": "string"},
        "progress": {"type": "number"},
        "observation": {"type": "string"},
        "params": {"type": "object", "additionalProperties": True},
        "is_terminal": {"type": "boolean"},
    },
}

_MILESTONES_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "title": "Milestones",
    "properties": {
        "current_idx": {"type": "integer"},
        "items": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "id": {"type": "string"},
                    "description": {"type": "string"},
                    "status": {"type": "string", "enum": ["pending", "running", "done", "deferred"]},
                    "type": {"type": "string"},
                    "capture": {"type": "boolean"},
                    "capture_target": {"type": "string"},
                    "dependencies": {"type": "array", "items": {"type": "string"}},
                },
            },
        },
    },
}

# Foxglove's parser doesn't accept JSON-Schema features like ["string","null"]
# unions or `items` arrays without a typed item schema. Keep this loose; the
# Log panel renders messages as text regardless of schema specificity.
_CONVERSATION_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "title": "Conversation",
    "additionalProperties": True,
    "properties": {
        "milestone_idx": {"type": "integer"},
        "messages": {
            "type": "array",
            "items": {"type": "object", "additionalProperties": True},
        },
    },
}

_EVENT_LOG_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "title": "EventLog",
    "properties": {
        "kind": {"type": "string"},
        "data": {"type": "object", "additionalProperties": True},
    },
    "required": ["kind"],
}

_BLACKBOARD_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "title": "BlackboardSnapshot",
    "additionalProperties": True,
}

_OBJECT_STORE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "title": "ObjectStore",
    "properties": {
        "handles": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "handle": {"type": "string"},
                    "label": {"type": "string"},
                    "position": {"type": "array", "items": {"type": "number"}},
                    "is_exploration": {"type": "boolean"},
                },
            },
        },
    },
}


# Public channel manifest. The harness (lam_v2_testing.harness) emits
# directly to these channel names via EventSink.
CHANNELS: List[Dict[str, Any]] = [
    {"topic": "/agent/phase", "encoding": "json", "schema": _PHASE_SCHEMA, "schema_name": "AgentPhase"},
    {"topic": "/agent/bt_snapshot", "encoding": "json", "schema": _BT_SNAPSHOT_SCHEMA, "schema_name": "BtSnapshot"},
    {"topic": "/agent/skill", "encoding": "json", "schema": _SKILL_SCHEMA, "schema_name": "SkillState"},
    {"topic": "/agent/milestones", "encoding": "json", "schema": _MILESTONES_SCHEMA, "schema_name": "Milestones"},
    {"topic": "/agent/conversation", "encoding": "json", "schema": _CONVERSATION_SCHEMA, "schema_name": "Conversation"},
    {"topic": "/agent/event_log", "encoding": "json", "schema": _EVENT_LOG_SCHEMA, "schema_name": "EventLog"},
    {
        "topic": "/agent/blackboard",
        "encoding": "json",
        "schema": _BLACKBOARD_SCHEMA,
        "schema_name": "BlackboardSnapshot",
    },
    {"topic": "/agent/object_store", "encoding": "json", "schema": _OBJECT_STORE_SCHEMA, "schema_name": "ObjectStore"},
]

# Fast lookup
CHANNEL_TOPICS: List[str] = [c["topic"] for c in CHANNELS]
CHANNELS_BY_TOPIC: Dict[str, Dict[str, Any]] = {c["topic"]: c for c in CHANNELS}
