"""QuerySkill — search the scene graph for an object's location.

Takes a text description (e.g. "red mug on the kitchen counter"), queries
the scene graph, and returns a semantic handle that downstream skills
(Navigate, Press, Push, Explore) use to access the object's coordinates.

Three outcomes:
    1. **Found**: Object exists in the scene graph → SceneObject with
       position + viewpoints. LLM gets a handle like "obj_red_mug_1".
    2. **Exploration suggested**: Object not found, but an anchor location
       was identified → SceneObject with is_exploration=True. LLM gets
       a handle to navigate near the anchor and look again.
    3. **Failed**: Nothing found → observation text only, no objects added.

Body subtree: single-leaf ``_QueryNode``. ContextSkill — framework
auto-wraps with CancellationGuard only. K-promotion via
``publish_results`` (paper §5 Invariant I6).

Agent-facing docs: ``tiamat_skills/docs/query.md`` — surfaced at runtime
via ``read_skill_docs("query")``.

The scene graph agent is injected via constructor and called synchronously.
It is NOT ported here — implementations live in lam_v2_roscore (real scene
graph) or lam_v2_testing (mock).

# Optional ``exclude`` param (unified-skill-api CP2, ported from the text
# rung's tiamat_agent commit 649d6f9d6)
#
# ``exclude`` is a list of previously-revealed ObjectStore handles (e.g.
# 'red_umbrella_1', as rendered in an earlier FOUND observation). The text
# rung passes exclude straight through to its engine, which speaks handles
# natively. Physics has no such layer — ``SceneGraphAgent`` speaks
# ``object_id``, not handle; the ObjectStore is the only place that knows
# the handle↔object_id mapping. So the port resolves excluded handles to
# object_ids via the store and drops matching rows from the found/
# retrieval-candidate lists (``_apply_exclude``) rather than threading
# ``exclude`` into the scene-graph agent call. Exploration suggestions are
# a single anchor, not a search list, so exclude does not touch them.
# Absent/empty exclude is byte-identical to the pre-CP2 behavior.
"""

from __future__ import annotations

import os
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional

import py_trees
from bt_agent.skills import ObjectStore, SkillObject
from bt_agent.skills import ContextSkill, SkillContext, SkillHandle

_QUERY_CANDIDATE_LIMIT = 5

# ---------------------------------------------------------------------------
# SceneObject — producer-local type (never imported by consumer skills)
# ---------------------------------------------------------------------------


@dataclass
class SceneObject(SkillObject):
    """A scene-graph object or exploration suggestion.

    This type is local to QuerySkill (the sole producer). Consumer skills
    (Navigate, Capture) access fields via duck-typed ``getattr`` — they
    never import this class. The contract is in the skill docs, not in
    shared types.
    """

    # -- VLM-visible (semantic) --
    reasoning: str = ""
    is_exploration: bool = False

    # -- VLM-hidden (structured) --
    object_id: str = ""
    position: Dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})
    viewpoint_position: Dict[str, float] = field(default_factory=lambda: {"x": 0.0, "y": 0.0, "z": 0.0})
    viewpoint_orientation: List[float] = field(default_factory=lambda: [0.0, 0.0, 0.0, 1.0])
    viewpoint_camera_id: str = ""
    all_viewpoints: List[List[float]] = field(default_factory=list)
    all_viewpoint_camera_ids: List[str] = field(default_factory=list)
    cropped_image: str = ""
    cropped_images: List[str] = field(default_factory=list)
    crop_metadata: Dict = field(default_factory=dict)
    crop_metadata_list: List[Dict] = field(default_factory=list)
    retrieval_score: Optional[float] = None
    rerank_score: Optional[float] = None
    navigation_unavailable: bool = False
    retriever_cluster_score: Optional[float] = None
    retriever_cluster_rank: Optional[int] = None

    # -- Exploration-specific --
    exploration_target: str = ""
    anchor_name: str = ""

    def to_llm_summary(self) -> str:
        if self.is_exploration:
            parts = [f"Exploration<{self.handle}>: search near '{self.label}'"]
            if self.exploration_target:
                parts.append(f"looking for '{self.exploration_target}'")
            return " — ".join(parts)
        return f"Object<{self.handle}>: '{self.label}'"

    @classmethod
    def from_raw(cls, handle: str, raw: Dict[str, Any]) -> SceneObject:
        """Create from a raw scene-graph result dict."""
        return cls(
            handle=handle,
            label=raw.get("object_label", ""),
            reasoning=raw.get("reasoning", ""),
            is_exploration=bool(raw.get("is_exploration", False)),
            object_id=str(raw.get("object_id", "")),
            position=dict(raw.get("position", {"x": 0.0, "y": 0.0, "z": 0.0})),
            viewpoint_position=(
                None
                if "viewpoint_position" in raw and raw["viewpoint_position"] is None
                else dict(raw.get("viewpoint_position", {"x": 0.0, "y": 0.0, "z": 0.0}))
            ),
            viewpoint_orientation=list(raw.get("viewpoint_orientation", [0.0, 0.0, 0.0, 1.0])),
            viewpoint_camera_id=raw.get("viewpoint_camera_id", ""),
            all_viewpoints=[list(vp) for vp in raw.get("all_viewpoints", [])],
            all_viewpoint_camera_ids=list(raw.get("all_viewpoint_camera_ids", [])),
            cropped_image=raw.get("cropped_image", ""),
            cropped_images=list(raw.get("cropped_images", [])),
            crop_metadata=dict(raw.get("crop_metadata", {})),
            crop_metadata_list=[dict(d) for d in raw.get("crop_metadata_list", [])],
            retrieval_score=raw.get("retrieval_score"),
            rerank_score=raw.get("rerank_score"),
            navigation_unavailable=bool(raw.get("navigation_unavailable", False)),
            retriever_cluster_score=raw.get("retriever_cluster_score"),
            retriever_cluster_rank=raw.get("retriever_cluster_rank"),
            exploration_target=raw.get("target_object", ""),
            anchor_name=raw.get("anchor_name", ""),
        )


def _find_by_object_id(store: ObjectStore, object_id: str) -> Optional[SceneObject]:
    """Dedup helper: find existing SceneObject by scene-graph object_id."""
    if not object_id:
        return None
    for obj in store.all_objects():
        if isinstance(obj, SceneObject) and obj.object_id == object_id:
            return obj
    return None


def _has_handle(store: ObjectStore, handle: str) -> bool:
    """Whether *handle* is already reserved in the object store."""
    return any(getattr(obj, "handle", None) == handle for obj in store.all_objects())


def _clean_exclude(raw: Any) -> List[str]:
    """Normalize the ``exclude`` param: strip whitespace, drop empty and
    non-string entries so callers can treat "no exclusions" uniformly.

    Ported from the text rung's ``_clean_exclude`` (commit 649d6f9d6); the
    text-rung version also drops entries containing ``;`` because ``;`` is
    the reserved joiner for its emitted bridge action. Physics passes
    ``exclude`` through as a real list (never serialized into an action
    string), so that guard doesn't apply here.
    """
    if not raw:
        return []
    cleaned = []
    for handle in raw:
        if not isinstance(handle, str):
            continue
        stripped = handle.strip()
        if stripped:
            cleaned.append(stripped)
    return cleaned


def _apply_exclude(
    result: SceneGraphResult,
    exclude: List[str],
    store: ObjectStore,
) -> SceneGraphResult:
    """Drop already-known objects the LLM asked to exclude.

    ``exclude`` holds ObjectStore handles; resolve them back to
    ``object_id`` via the store (the only place that knows the mapping —
    see the module docstring) and filter both ``found`` and
    ``retrieval_candidates``. Exploration suggestions are untouched.

    Production (``RosSceneGraphAgent.query``) always returns a single
    ``found`` entry plus up to four ``retrieval_candidates`` (the primary
    included). Excluding that one entry would otherwise empty ``found``
    and degrade "exclude the match I already saw" into a false "not
    found" — so when exclusion empties ``found`` but non-excluded
    candidates remain, promote the best-ranked survivor (the first
    remaining candidate, order preserved) into ``found``. It is left in
    ``retrieval_candidates`` too: ``_build_found_result`` already dedupes
    alternatives against ``found[0]``'s object_id, so this mirrors the
    normal found+candidates shape rather than introducing a new one.
    """
    excluded_ids = {
        obj.object_id
        for obj in store.all_objects()
        if isinstance(obj, SceneObject) and obj.handle in exclude and obj.object_id
    }
    if not excluded_ids:
        return result

    found = [raw for raw in result.found if str(raw.get("object_id", "")) not in excluded_ids]
    candidates = [raw for raw in result.retrieval_candidates if str(raw.get("object_id", "")) not in excluded_ids]
    if not found and candidates:
        found = [candidates[0]]
    return SceneGraphResult(found=found, exploration=result.exploration, retrieval_candidates=candidates)


# ---------------------------------------------------------------------------
# SceneGraphAgent — the interface QuerySkill depends on
# ---------------------------------------------------------------------------


class SceneGraphAgent(ABC):
    """Abstract interface for scene graph querying.

    Implementations:
        - lam_v2_roscore: wraps the real scene graph + VLM pipeline
        - lam_v2_testing: returns canned results
    """

    @abstractmethod
    def query(
        self,
        query_text: str,
        *,
        task: str = "",
        scenario: str = "",
        camera_images: Optional[Dict[str, bytes]] = None,
        scene_graph: Optional[Dict[str, Any]] = None,
    ) -> SceneGraphResult:
        """Search the scene graph for objects matching query_text.

        Returns a SceneGraphResult with found/exploration/empty outcome.
        """
        ...


class SceneGraphResult:
    """Result of a scene graph query.

    Exactly one of: found objects, exploration suggestion, or empty.
    """

    def __init__(
        self,
        *,
        found: Optional[List[Dict[str, Any]]] = None,
        exploration: Optional[Dict[str, Any]] = None,
        retrieval_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self.found = found or []
        self.exploration = exploration
        self.retrieval_candidates = retrieval_candidates or list(self.found)

    @property
    def is_found(self) -> bool:
        return len(self.found) > 0

    @property
    def is_exploration(self) -> bool:
        return self.exploration is not None and not self.is_found

    @property
    def is_empty(self) -> bool:
        return not self.is_found and not self.is_exploration


# ---------------------------------------------------------------------------
# BT node
# ---------------------------------------------------------------------------


class _QueryNode(py_trees.behaviour.Behaviour):
    """Single BT node that queries the scene graph.

    Completes in one tick: calls the scene graph agent synchronously,
    writes observation + result_data to the handle, returns SUCCESS.
    """

    def __init__(
        self,
        sg: SceneGraphAgent,
        params: Dict[str, Any],
        context: SkillContext,
        handle: SkillHandle,
    ) -> None:
        super().__init__(name="query")
        self._sg = sg
        self._params = params
        self._context = context
        self._handle = handle

    def update(self) -> py_trees.common.Status:
        query_text = self._params.get("target_description", "")
        if not query_text:
            self._handle.observation = "No target_description provided."
            return py_trees.common.Status.FAILURE

        store = self._context.object_store
        if store is None:
            self._handle.observation = "ObjectStore not available."
            return py_trees.common.Status.FAILURE

        exclude = _clean_exclude(self._params.get("exclude"))
        canonical = getattr(self._sg, "canonical_query", None)
        if canonical is not None:
            result = canonical(query_text, exclude)
            handles = list(result.get("handles") or [])
            built = (
                _build_found_result(
                    [self._sg._raw(handle, result.get("query_layer", "")) for handle in handles], query_text, store
                )
                if handles
                else {}
            )
            self._handle.observation = result["observation"]
            self._handle.result_data = {
                **built,
                **result,
                "observation": result["observation"],
                "failure_reason": result["reason"],
            }
            return py_trees.common.Status.SUCCESS if result["ok"] else py_trees.common.Status.FAILURE

        try:
            result = self._sg.query(
                query_text,
                task=self._context.main_task,
                scenario=self._context.scenario,
                camera_images=self._context.platform_state.get("camera_images"),
                scene_graph=self._context.platform_state.get("scene_graph"),
            )
        except Exception as e:
            self._handle.observation = f"Scene graph query failed: {e}"
            return py_trees.common.Status.FAILURE

        if exclude:
            result = _apply_exclude(result, exclude, store)

        if result.is_found:
            data = _build_found_result(
                result.found,
                query_text,
                store,
                retrieval_candidates=result.retrieval_candidates,
            )
        elif result.is_exploration:
            data = _build_exploration_result(result.exploration, query_text, store)
        else:
            data = {"observation": f"'{query_text}' not found in scene graph."}

        if exclude:
            data["observation"] = data.get("observation", "") + f"\nexcluding: {', '.join(exclude)}"

        self._handle.observation = data.get("observation", "")
        self._handle.result_data = data
        return py_trees.common.Status.SUCCESS


# ---------------------------------------------------------------------------
# Result builders (pure functions)
# ---------------------------------------------------------------------------


def _build_found_result(
    found: List[Dict[str, Any]],
    query_text: str,
    store: Any,
    *,
    retrieval_candidates: Optional[List[Dict[str, Any]]] = None,
) -> Dict[str, Any]:
    new_objects: List[SceneObject] = []
    all_handles: List[str] = []
    primary_handle = ""

    for raw in found:
        object_id = str(raw.get("object_id", ""))

        # Reuse existing handle if this object_id is already in the store
        existing = _find_by_object_id(store, object_id) if object_id else None

        if existing is not None:
            handle = existing.handle
        else:
            label = raw.get("object_label", query_text)
            preferred = raw.get("preferred_handle")
            handle = (
                preferred
                if isinstance(preferred, str) and preferred and not _has_handle(store, preferred)
                else store.generate_handle(label)
            )
            obj = SceneObject.from_raw(handle, raw)
            new_objects.append(obj)

        all_handles.append(handle)
        if not primary_handle:
            primary_handle = handle

    if len(all_handles) == 1:
        label = new_objects[0].label if new_objects else query_text
        obs = f"Found '{label}'. Use '{primary_handle}' for navigation."
    else:
        handles_str = ", ".join(f"'{h}'" for h in all_handles)
        obs = f"Found {len(all_handles)} matches for '{query_text}': {handles_str}. Use any handle for navigation."

    if _query_topk_enabled():
        alternatives: List[str] = []
        candidates = (retrieval_candidates or found)[:_QUERY_CANDIDATE_LIMIT]
        primary_id = str(found[0].get("object_id", "") or "")
        for raw in candidates:
            object_id = str(raw.get("object_id", "") or "")
            if object_id == primary_id:
                continue

            existing = _find_by_object_id(store, object_id) if object_id else None
            if existing is not None:
                handle = existing.handle
            else:
                label = raw.get("object_label", query_text)
                preferred = raw.get("preferred_handle")
                handle = (
                    preferred
                    if isinstance(preferred, str) and preferred and not _has_handle(store, preferred)
                    else store.generate_handle(label)
                )
                new_objects.append(SceneObject.from_raw(handle, raw))

            alternatives.append(f"{handle} ({_format_retrieval_score(raw)})")
            if len(alternatives) >= _QUERY_CANDIDATE_LIMIT - 1:
                break

        if alternatives:
            obs = f"{obs}\nalternatives: {', '.join(alternatives)}"

    return {
        "observation": obs,
        "new_objects": new_objects,
        "query_mappings": {query_text: primary_handle},
    }


def _query_topk_enabled(value: Optional[str] = None) -> bool:
    """Parse the opt-in top-k observation switch using the local env idiom."""
    raw = os.environ.get("LAM_QUERY_TOPK", "") if value is None else value
    return raw.strip().lower() in ("1", "true")


def _format_retrieval_score(raw: Dict[str, Any]) -> str:
    """Format the retriever's final score, with cluster/rerank fallbacks."""
    for key in ("retrieval_score", "retriever_cluster_score", "rerank_score"):
        score = raw.get(key)
        if score is None:
            continue
        try:
            return f"{float(score):.2f}"
        except (TypeError, ValueError):
            continue
    return "n/a"


def _build_exploration_result(
    exploration: Dict[str, Any],
    query_text: str,
    store: Any,
) -> Dict[str, Any]:
    label = exploration.get("object_label", exploration.get("anchor_name", "area"))
    handle = store.generate_handle(label, prefix="explore")

    raw = dict(exploration)
    raw.setdefault("is_exploration", True)
    raw.setdefault("target_object", query_text)
    obj = SceneObject.from_raw(handle, raw)

    return {
        "observation": f"'{query_text}' not found directly. Explore near '{label}' using '{handle}'.",
        "new_objects": [obj],
        "query_mappings": {query_text: handle},
    }


# ---------------------------------------------------------------------------
# QuerySkill
# ---------------------------------------------------------------------------


class QuerySkill(ContextSkill):
    """Search the scene graph for an object's location.

    Returns a semantic handle (e.g. "obj_red_mug_1") that downstream
    skills use to resolve the object's position via the ObjectStore.
    The LLM never sees raw coordinates — only the handle.
    """

    name = "query"
    description = "Search the scene graph for an object or location."
    usage = "Use this FIRST when you need to find something. Returns a handle you can pass to NAVIGATE, PRESS, or PUSH."
    params = {
        "target_description": "What to search for (e.g. 'red umbrella' )",
    }

    def __init__(self, scene_graph_agent: SceneGraphAgent) -> None:
        self._sg = scene_graph_agent

    def build_subtree(
        self,
        params: Dict[str, Any],
        context: SkillContext,
        handle: SkillHandle,
    ) -> py_trees.behaviour.Behaviour:
        return _QueryNode(self._sg, params, context, handle)

    def tool_def(self) -> Dict[str, Any]:
        """Hand-written: the generic builder (``Skill.tool_def``) marks every
        declared param as a required string, so it cannot express an
        optional array param. ``exclude`` needs exactly that shape.

        Ported from the text rung's identical override (tiamat_agent
        commit 649d6f9d6). ``target_description`` mirrors what the generic
        builder would have produced byte-for-byte (same name, same
        description source, still required); ``exclude`` is the only
        schema diff.
        """
        desc = self.description
        if self.usage:
            desc += " " + self.usage

        return {
            "type": "function",
            "function": {
                "name": self.name,
                "description": desc,
                "parameters": {
                    "type": "object",
                    "properties": {
                        "target_description": {
                            "type": "string",
                            "description": self.params["target_description"],
                        },
                        "exclude": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": (
                                "Object handles to exclude from this search (as returned by a "
                                "prior QUERY, e.g. 'red_umbrella_1'). Use this to find "
                                "alternatives when the first match isn't the right one."
                            ),
                        },
                        "memory_note": {
                            "type": "string",
                            "description": "Brief note for your future self about key findings or state.",
                        },
                    },
                    "required": ["target_description"],
                },
            },
        }

    def publish_results(self, handle: SkillHandle, store) -> None:
        """K-publication (paper §5 I6): adds discovered SceneObjects + query mappings."""
        result = handle.result_data or {}
        for obj in result.get("new_objects", []):
            store.add(obj)
        for query_text, h in result.get("query_mappings", {}).items():
            store.register_query(query_text, h)
