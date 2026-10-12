"""Deterministic blocked-callback fixture; no robot, simulator or model service."""

import argparse
from importlib import import_module
import json
import threading

parser = argparse.ArgumentParser()
parser.add_argument("--reference", action="store_true")
args = parser.parse_args()
module = "lam_v2.observability.event_sink" if args.reference else "bt_agent.tracing"
sink = import_module(module).EventSink()
started, release = threading.Event(), threading.Event()
received, errors = [], []


def callback(event):
    if event.payload["i"] == 0:
        started.set()
        assert release.wait(2)
    received.append(event.payload["i"])


sink.subscribe(callback)
subscriber = sink._subscribers[0]
sink.emit("fixture", {"i": 0})
assert started.wait(1)
sink.emit("fixture", {"i": 1})
sink.emit("fixture", {"i": 2})


def close():
    try:
        sink.close()
    except Exception as error:
        errors.append(repr(error))


closer = threading.Thread(target=close)
closer.start()
assert subscriber._stop.wait(1)
release.set()
closer.join(2)
assert not closer.is_alive() and not errors
print(
    json.dumps(
        {
            "reference": args.reference,
            "accepted": 3,
            "delivered": received,
            "lost_on_shutdown": 3 - len(received),
        }
    )
)
if not args.reference:
    assert received == [0, 1, 2]
