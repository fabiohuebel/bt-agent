"""ROS-free coverage for the physics QuerySkill's ``exclude`` param.

Checkpoint 2 of unified-skill-api (docs/spec/unified-skill-api.md): physics
gains ``exclude`` parity with the text rung's QuerySkill (tiamat_agent
commit 649d6f9d6). Physics has no engine-side handle layer, so the port
resolves excluded handles to ``object_id`` via the ObjectStore and filters
the scene-graph result before it's rendered (see ``_apply_exclude`` /
``_clean_exclude`` in ``bt_skills/query.py``).
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from bt_agent.skills import ObjectStore
from bt_agent.skills import SkillContext, SkillHandle
from bt_agent_robotics._hardware.skills.query import (
    QuerySkill,
    SceneGraphAgent,
    SceneGraphResult,
    _apply_exclude,
    _clean_exclude,
)

# ---------------------------------------------------------------------------
# _clean_exclude — normalization
# ---------------------------------------------------------------------------


def test_clean_exclude_absent_normalizes_to_empty_list() -> None:
    assert _clean_exclude(None) == []
    assert _clean_exclude([]) == []


def test_clean_exclude_strips_whitespace_and_drops_empty_and_non_string() -> None:
    assert _clean_exclude([" red_umbrella_1 ", "", "   ", None, 42, "blue_mug_2"]) == [
        "red_umbrella_1",
        "blue_mug_2",
    ]


# ---------------------------------------------------------------------------
# _apply_exclude — filtering against the ObjectStore's handle<->object_id map
# ---------------------------------------------------------------------------


def _candidate(object_id: str, label: str) -> Dict[str, Any]:
    return {"object_id": object_id, "object_label": label}


def test_apply_exclude_drops_matching_found_and_candidates_with_no_survivor() -> None:
    """When exclusion leaves no surviving candidate either, both lists end
    up empty — no promotion possible (see the promotion test below for the
    case where a survivor exists)."""
    store = ObjectStore()
    # Pretend a prior query already published "red_umbrella_1" -> obj_1.
    from bt_agent_robotics._hardware.skills.query import SceneObject

    store.add(SceneObject(handle="red_umbrella_1", label="red umbrella", object_id="obj_1"))

    result = SceneGraphResult(
        found=[_candidate("obj_1", "red umbrella")],
        retrieval_candidates=[_candidate("obj_1", "red umbrella")],
    )

    filtered = _apply_exclude(result, ["red_umbrella_1"], store)

    assert filtered.found == []
    assert filtered.retrieval_candidates == []


def test_apply_exclude_no_match_is_a_noop() -> None:
    store = ObjectStore()
    result = SceneGraphResult(found=[_candidate("obj_1", "red umbrella")])

    filtered = _apply_exclude(result, ["unrelated_handle_9"], store)

    assert filtered.found == result.found


def test_apply_exclude_promotes_surviving_candidate_when_found_is_emptied() -> None:
    """Production shape (RosSceneGraphAgent.query): found=[primary],
    retrieval_candidates=[primary, ...alts]. Excluding the primary must not
    degrade to empty ``found`` when a non-excluded alternative remains.
    """
    store = ObjectStore()
    from bt_agent_robotics._hardware.skills.query import SceneObject

    store.add(SceneObject(handle="red_umbrella_1", label="red umbrella", object_id="obj_1"))

    result = SceneGraphResult(
        found=[_candidate("obj_1", "red umbrella")],
        retrieval_candidates=[
            _candidate("obj_1", "red umbrella"),
            _candidate("obj_2", "red umbrella (2)"),
            _candidate("obj_3", "red umbrella (3)"),
        ],
    )

    filtered = _apply_exclude(result, ["red_umbrella_1"], store)

    assert filtered.found == [_candidate("obj_2", "red umbrella (2)")]
    assert filtered.retrieval_candidates == [
        _candidate("obj_2", "red umbrella (2)"),
        _candidate("obj_3", "red umbrella (3)"),
    ]


# ---------------------------------------------------------------------------
# End-to-end: QuerySkill.build_subtree()._QueryNode.update() honors exclude
# ---------------------------------------------------------------------------


class _StubSceneGraphAgent(SceneGraphAgent):
    """Returns a fixed result regardless of query text.

    Mirrors production shape (RosSceneGraphAgent.query /
    ros_scene_graph_agent.py's ``_top_retrieval_candidates``): ``found``
    holds exactly one entry, and ``retrieval_candidates`` holds that same
    primary followed by up to four alternates.
    """

    def __init__(
        self,
        found: List[Dict[str, Any]],
        retrieval_candidates: Optional[List[Dict[str, Any]]] = None,
    ) -> None:
        self._found = found
        self._retrieval_candidates = retrieval_candidates

    def query(
        self,
        query_text: str,
        *,
        task: str = "",
        scenario: str = "",
        camera_images: Optional[Dict[str, bytes]] = None,
        scene_graph: Optional[Dict[str, Any]] = None,
    ) -> SceneGraphResult:
        return SceneGraphResult(
            found=list(self._found),
            retrieval_candidates=list(self._retrieval_candidates) if self._retrieval_candidates else None,
        )


def _run_query(skill: QuerySkill, params: Dict[str, Any], store: ObjectStore) -> SkillHandle:
    handle = SkillHandle()
    context = SkillContext(object_store=store)
    node = skill.build_subtree(params, context, handle)
    node.update()
    return handle


def test_query_without_exclude_is_byte_identical_to_legacy_observation() -> None:
    store = ObjectStore()
    skill = QuerySkill(_StubSceneGraphAgent([_candidate("obj_1", "red umbrella")]))

    handle = _run_query(skill, {"target_description": "red umbrella"}, store)

    assert handle.observation == "Found 'red umbrella'. Use 'red_umbrella_1' for navigation."
    assert "excluding" not in handle.observation


def test_query_excludes_previously_found_handle_and_offers_next_match() -> None:
    """Production shape: found=[obj_1], retrieval_candidates=[obj_1, obj_2]
    (the scene-graph agent always returns exactly one ``found`` entry — see
    ``RosSceneGraphAgent.query``). Excluding obj_1 must promote obj_2 into
    ``found`` rather than degrading to "not found".
    """
    store = ObjectStore()
    skill = QuerySkill(
        _StubSceneGraphAgent(
            [_candidate("obj_1", "red umbrella")],
            retrieval_candidates=[_candidate("obj_1", "red umbrella"), _candidate("obj_2", "red umbrella")],
        )
    )

    # First call: discovers obj_1, publish_results (as the framework would) writes it to the store.
    first = _run_query(skill, {"target_description": "red umbrella"}, store)
    skill.publish_results(first, store)
    first_handle = first.result_data["query_mappings"]["red umbrella"]

    # Second call: exclude the just-found handle, expect obj_2 instead.
    second = _run_query(skill, {"target_description": "red umbrella", "exclude": [first_handle]}, store)

    assert first_handle not in second.observation.split("\n")[0]
    assert "excluding: " + first_handle in second.observation
    assert second.result_data["query_mappings"]["red umbrella"] != first_handle


def test_query_exclude_that_removes_all_matches_reports_not_found() -> None:
    store = ObjectStore()
    skill = QuerySkill(_StubSceneGraphAgent([_candidate("obj_1", "red umbrella")]))

    first = _run_query(skill, {"target_description": "red umbrella"}, store)
    skill.publish_results(first, store)
    only_handle = first.result_data["query_mappings"]["red umbrella"]

    second = _run_query(skill, {"target_description": "red umbrella", "exclude": [only_handle]}, store)

    assert second.observation == f"'red umbrella' not found in scene graph.\nexcluding: {only_handle}"


def test_query_tool_def_exposes_optional_exclude_array() -> None:
    skill = QuerySkill(_StubSceneGraphAgent([]))

    schema = skill.tool_def()["function"]["parameters"]

    assert schema["properties"]["exclude"]["type"] == "array"
    assert schema["properties"]["exclude"]["items"] == {"type": "string"}
    assert "exclude" not in schema["required"]
    assert schema["required"] == ["target_description"]
