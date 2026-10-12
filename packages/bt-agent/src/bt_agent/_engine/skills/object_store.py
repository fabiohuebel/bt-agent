"""ObjectStore — symbolic inter-skill communication via semantic handles.

The ObjectStore is the paper's K store (§5): a typed, append-only context
store that lets skills exchange structured data (coordinates, poses,
segmentation masks) **without passing through the LLM**.  The LLM operates
on semantic handles (e.g. ``"obj_mug_1"``); structured data flows outside
the language channel through this generic container.

This eliminates a class of failure where numerical precision is lost through
natural language round-tripping — especially important for smaller models
deployed at the edge.

Three layers::

    LLM sees:       "navigate to obj_mug_1"             (semantic handle)
    ObjectStore:     SkillObject(handle="obj_mug_1", ...)  (structured data)
    Skill receives:  position=(2.3, 1.5)                 (via prepare())

Producing objects (inside a skill's ``publish_results`` override)::

    for obj in handle.result_data.get("new_objects", []):
        store.add(obj)

Consuming objects (inside a skill's ``prepare()``)::

    obj = context.object_store.get(params["target_object"])
    if obj is None:
        return {}, "Unknown target."
    return {"position": ...}, None

The framework never inspects what is inside a ``SkillObject`` beyond
``handle`` and ``label``.  All substrate-specific data types (e.g.
TIAMAT's ``SceneObject``, ``CaptureResult``) live in their respective
``realizations/<name>/objects.py`` and subclass ``SkillObject``.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Dict, List, Optional

# ---------------------------------------------------------------------------
# Base
# ---------------------------------------------------------------------------


@dataclass
class SkillObject:
    """Base class for all skill-produced objects in K.

    Every object has a human-readable ``handle`` (e.g. ``"red_mug_1"``)
    that the LLM uses to reference it.  An optional ``label`` provides a
    natural-language description for fuzzy resolution and LLM summaries.

    The ``to_llm_summary()`` method returns what the LLM sees — semantic
    information only, never coordinates or structured payloads.

    Substrate-specific types (SceneObject, CaptureResult, InventoryItem,
    etc.) subclass this and add their own fields.
    """

    handle: str
    label: str = ""

    def to_llm_summary(self) -> str:
        """What the LLM sees. Override in subclasses for richer summaries."""
        if self.label:
            return f"'{self.handle}' ({self.label})"
        return f"'{self.handle}'"


# ---------------------------------------------------------------------------
# ObjectStore
# ---------------------------------------------------------------------------

_HANDLE_RE = re.compile(r"[^a-z0-9_]")


class ObjectStore:
    """Generic container for inter-skill object references (paper's K).

    Resolution order in ``get()``: exact handle → label match → query text match.

    Write path:  Skills write objects in their ``publish_results()`` override
                 (called by the framework after skill success).
    Read path:   Skills receive the store via ``SkillContext.object_store``
                 and call ``get()`` in ``prepare()``.

    The framework only calls ``add()``, ``get()``, ``all_llm_summaries()``,
    ``generate_handle()``, ``register_query()``, and ``all_objects()``.
    No ``isinstance`` checks — the store is fully substrate-agnostic.
    """

    def __init__(self) -> None:
        self._objects: Dict[str, SkillObject] = {}
        self._by_label: Dict[str, str] = {}  # label.lower() → handle
        self._by_query: Dict[str, str] = {}  # query_text.lower() → handle
        self._counter: int = 0

    # -- Write operations ------------------------------------------------------

    def add(self, obj: SkillObject) -> str:
        """Add an object and return its handle.

        Label-based resolution index is updated automatically when
        ``obj.label`` is non-empty.  Deduplication by substrate-specific
        keys (e.g. scene-graph ``object_id``) is the producing skill's
        responsibility — the framework does not inspect subclass fields.
        """
        self._objects[obj.handle] = obj
        if obj.label:
            self._by_label[obj.label.lower().strip()] = obj.handle
        return obj.handle

    def register_query(self, query_text: str, handle: str) -> None:
        """Map a query text to a handle for fuzzy resolution."""
        self._by_query[query_text.lower().strip()] = handle

    def reset(self) -> None:
        """Clear all stored objects."""
        self._objects.clear()
        self._by_label.clear()
        self._by_query.clear()
        self._counter = 0

    # -- Read operations -------------------------------------------------------

    def get(self, handle_or_text: str) -> Optional[SkillObject]:
        """Resolve a reference: exact handle → label match → query text match."""
        key = handle_or_text.strip()
        # Strip Object<...> wrapper that LLMs sometimes copy from prompt context
        if key.startswith("Object<") and key.endswith(">"):
            key = key[7:-1]
        if key in self._objects:
            return self._objects[key]
        lower = key.lower()
        label_handle = self._by_label.get(lower)
        if label_handle and label_handle in self._objects:
            return self._objects[label_handle]
        query_handle = self._by_query.get(lower)
        if query_handle and query_handle in self._objects:
            return self._objects[query_handle]
        return None

    def all_objects(self) -> List[SkillObject]:
        """All objects currently in the store."""
        return list(self._objects.values())

    def all_llm_summaries(self) -> List[str]:
        """LLM-visible summaries of all objects. For building prompts."""
        return [obj.to_llm_summary() for obj in self._objects.values()]

    # -- Handle generation -----------------------------------------------------

    def generate_handle(self, label: str, prefix: str = "") -> str:
        """Generate a human-readable handle like ``red_mug_1``."""
        self._counter += 1
        if prefix:
            return f"{prefix}_{self._counter}"
        slug = _HANDLE_RE.sub("_", label.lower().strip())
        slug = slug.strip("_")[:30]
        if not slug:
            slug = "obj"
        return f"{slug}_{self._counter}"
