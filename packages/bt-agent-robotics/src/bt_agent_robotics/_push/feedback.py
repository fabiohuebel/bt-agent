"""Controller-only bounded push telemetry; never serialize this into LLM input.

Not registered with a ROS node yet. Bind before motion, feed every native
sample, and drain every skill tick. Gaps and overflow are sticky failures until
a fresh invocation is bound. The receiver supplies monotonic arrival times.
"""

import math
import threading
from collections import deque
from copy import deepcopy


class PushFeedbackBuffer:
    def __init__(self, capacity=4096):
        if type(capacity) is not int or capacity < 2:
            raise ValueError("Bounded capacity >=2 required")
        self.capacity = capacity
        self._lock = threading.Lock()
        self._queue = deque()
        self._binding = None
        self._sample = None
        self._received = None
        self._error = None

    def bind(self, episode, canonical_key, native_id):
        if (
            len(episode) != 3
            or not all(isinstance(x, str) and x for x in episode[:2])
            or type(episode[2]) is not int
            or episode[2] < 1
        ):
            raise ValueError("Explicit episode identity required")
        if canonical_key not in {"step_stool", "dining_chair"} or type(native_id) is not int or native_id < 0:
            raise ValueError("Exact supported native instance required")
        with self._lock:
            self._binding = (tuple(episode), canonical_key, native_id)
            self._queue.clear()
            self._sample = None
            self._received = None
            self._error = None

    def reset(self):
        with self._lock:
            self._binding = None
            self._queue.clear()
            self._sample = None
            self._received = None
            self._error = None

    def observe(self, event, received_at):
        with self._lock:
            if self._binding is None or self._error:
                return False
            episode, key, native_id = self._binding
            if event.get("mechanism_id") != "push:" + key:
                return False
            if tuple(event.get(k) for k in ("run_id", "task_id", "task_index")) != episode:
                # Old DDS samples are ignored; matching-episode corruption fails.
                return False
            e = event.get("evidence", {})
            if (
                event.get("schema") != "native-push-observation/v1"
                or event.get("status") not in {"push_observation", "push_completed"}
                or e.get("canonical_key") != key
                or e.get("native_instance_id") != native_id
            ):
                self._error = "feedback_identity_mismatch"
                return False
            if (
                type(received_at) not in (int, float)
                or not math.isfinite(received_at)
                or received_at < 0
                or (self._received is not None and received_at < self._received)
            ):
                self._error = "feedback_clock_invalid"
                return False
            sample = e.get("sample")
            if type(sample) is not int or sample < 1 or (self._sample is not None and sample != self._sample + 1):
                self._error = "feedback_sequence_gap"
                return False
            if len(self._queue) >= self.capacity:
                self._error = "feedback_overflow"
                return False
            self._sample = sample
            self._received = received_at
            self._queue.append((deepcopy(event), received_at))
            return True

    def drain(self, now, stale_after=1.0):
        with self._lock:
            if self._error:
                raise RuntimeError(self._error)
            if self._binding is None:
                raise RuntimeError("feedback_not_bound")
            if not math.isfinite(now) or not math.isfinite(stale_after) or stale_after <= 0:
                raise ValueError("Finite clock/limit required")
            if self._received is None or not 0 <= now - self._received <= stale_after:
                self._error = "feedback_stale"
                raise RuntimeError(self._error)
            result = list(self._queue)
            self._queue.clear()
            return result  # noqa: R504 - preserve the copy before clearing the queue

    @property
    def failure(self):
        with self._lock:
            return self._error
