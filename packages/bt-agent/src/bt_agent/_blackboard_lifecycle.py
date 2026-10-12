"""Collect dead client registrations without changing live tree permissions.

Only clients reached through this runtime's owned trees are tracked. Never
clear global blackboard storage or deregister a client while it is still alive.
The frozen skill worker can therefore finish using its clients after a stop.
"""

import weakref

import py_trees

_tracked = weakref.WeakSet()


def _forget(identifier):
    board = py_trees.blackboard.Blackboard
    board.clients.pop(identifier, None)
    for metadata in list(board.metadata.values()):
        metadata.read.discard(identifier)
        metadata.write.discard(identifier)
        metadata.exclusive.discard(identifier)
    # Keep key/value data and empty metadata: an external consumer may still
    # inspect it, and concurrent client registration must not lose its entry.


def track_client(client):
    if isinstance(client, py_trees.blackboard.Client) and client not in _tracked:
        _tracked.add(client)
        finalizer = weakref.finalize(client, _forget, client.id())
        finalizer.atexit = False


def track_tree(root):
    pending, seen = [root], set()
    while pending:
        node = pending.pop()
        if not isinstance(node, py_trees.behaviour.Behaviour) or id(node) in seen:
            continue
        seen.add(id(node))
        for client in node.blackboards:
            track_client(client)
        pending.extend(node.children)
        # ExecuteSkill retains its most recent subtree even when the paper
        # controller clears SKILL_HANDLE in the same supervisor tick.
        handle = getattr(node, "_handle", None)
        if handle is not None:
            pending.append(getattr(handle, "subtree", None))
