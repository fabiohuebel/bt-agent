import gc

import py_trees
import pytest

from bt_agent import create_agent
from bt_agent.episode import EpisodeHost
from bt_agent.skills import ReportSkill
from bt_agent.testing import InspectShelf, shelf_model
from bt_agent._blackboard_lifecycle import track_client, track_tree


@pytest.mark.parametrize("profile", ["task-core-v1", "paper-core-v1"])
def test_closed_runtimes_do_not_accumulate_blackboard_clients(profile):
    gc.collect()
    before = len(py_trees.blackboard.Blackboard.clients)
    for _ in range(12):
        with create_agent(
            model=shelf_model(),
            skills=[InspectShelf(), ReportSkill()],
            host=EpisodeHost(lambda _: None, lambda: None),
            profile=profile,
        ) as agent:
            assert agent.run("Inspect shelf", tick_interval=0.001).submitted
        del agent
        gc.collect()
    # The paper blackboard can retain its most recent skill snapshot for inspection.
    assert len(py_trees.blackboard.Blackboard.clients) <= before + 1


@pytest.mark.parametrize("profile", ["task-core-v1", "paper-core-v1"])
def test_caller_client_and_value_survive_runtime_cleanup(profile):
    with create_agent(
        model=shelf_model(),
        skills=[InspectShelf(), ReportSkill()],
        host=EpisodeHost(lambda _: None, lambda: None),
        profile=profile,
    ) as agent:
        caller = py_trees.blackboard.Client(name="caller-owned")
        caller.register_key("/release_test_caller", access=py_trees.common.Access.WRITE)
        caller.set("/release_test_caller", 42)
        assert agent.run("Inspect shelf", tick_interval=0.001).submitted
    del agent
    gc.collect()
    try:
        assert caller.id() in py_trees.blackboard.Blackboard.clients
        assert caller.get("/release_test_caller") == 42
    finally:
        caller.unregister()


def test_tracked_live_client_keeps_permissions_and_dead_client_keeps_value():
    board = py_trees.blackboard.Blackboard
    client = py_trees.blackboard.Client(name="tracked-owned")
    client.register_key("/release_owned_value", access=py_trees.common.Access.WRITE)
    client.set("/release_owned_value", 17)
    identifier = client.id()
    track_client(client)
    track_client(client)
    gc.collect()
    assert identifier in board.clients and client.get("/release_owned_value") == 17
    del client
    gc.collect()
    assert identifier not in board.clients
    assert identifier not in board.metadata["/release_owned_value"].write
    assert board.storage["/release_owned_value"] == 17
    board.storage.pop("/release_owned_value")
    board.metadata.pop("/release_owned_value")


def test_explicit_unregister_before_collection_is_idempotent():
    client = py_trees.blackboard.Client(name="explicit-close")
    track_client(client)
    identifier = client.id()
    client.unregister()
    del client
    gc.collect()
    assert identifier not in py_trees.blackboard.Blackboard.clients


def test_guard_handle_self_reference_does_not_recurse():
    from types import SimpleNamespace

    root = py_trees.behaviours.Success("guard-fixture")
    root._handle = SimpleNamespace(subtree=root)
    track_tree(root)


def test_failed_paper_construction_does_not_leak_clients_or_ownership():
    gc.collect()
    before = len(py_trees.blackboard.Blackboard.clients)
    for _ in range(8):
        skill = InspectShelf()
        skill.effect_type = "invalid"
        with pytest.raises(TypeError):
            create_agent(
                model=shelf_model(),
                skills=[skill],
                host=EpisodeHost(lambda _: None, lambda: None),
                profile="paper-core-v1",
            )
        gc.collect()
    assert len(py_trees.blackboard.Blackboard.clients) <= before
    with create_agent(
        model=shelf_model(), skills=[], host=EpisodeHost(lambda _: None, lambda: None)
    ):
        pass


@pytest.mark.parametrize("name", [None, "", "  ", 3, []])
def test_invalid_skill_names_fail_before_runtime_ownership(name):
    skill = InspectShelf()
    skill.name = name
    with pytest.raises(ValueError, match="non-empty"):
        create_agent(
            model=shelf_model(), skills=[skill], host=EpisodeHost(lambda _: None, lambda: None)
        )
