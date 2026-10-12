from bt_agent.skills import ObjectStore
from bt_agent_robotics._interaction_query import SceneObject, _build_found_result


def _raw(viewpoints):
    return {
        "object_id": "42",
        "object_label": "pills",
        "preferred_handle": "pills 1",
        "position": {"x": 1.0, "y": 2.0, "z": 3.0},
        "all_viewpoints": viewpoints,
        "all_viewpoint_camera_ids": [""] * len(viewpoints),
        "navigation_unavailable": not viewpoints,
    }


def test_requery_refreshes_same_handle_viewpoints_after_native_gate():
    store = ObjectStore()
    existing = SceneObject.from_raw("pills 1", _raw([]))
    store.add(existing)
    result = _build_found_result([_raw([[4, 5, 6, 0, 0, 0, 1]])], "medicine", store)
    assert result["new_objects"] == []
    assert store.get("pills 1") is existing
    assert existing.all_viewpoints == [[4, 5, 6, 0, 0, 0, 1]]
    assert existing.navigation_unavailable is False

    store.reset()
    closed = SceneObject.from_raw("pills 1", _raw([]))
    store.add(closed)
    assert store.get("pills 1") is closed
    assert closed.all_viewpoints == []
    assert closed.navigation_unavailable is True
