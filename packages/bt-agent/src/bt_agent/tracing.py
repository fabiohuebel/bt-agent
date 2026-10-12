"""Non-blocking event delivery with explicit, bounded shutdown.

Successful shutdown drains callbacks for every accepted event. Queue overflow
and callback errors remain possible and are exposed by statistics(). Shutdown
does not itself establish durable storage or physical robot stopping.
"""

from collections import deque
import contextlib
from dataclasses import dataclass
import logging
import math
import queue
import threading
import time
from typing import Any, Callable

logger = logging.getLogger(__name__)


def _timeout(value: float) -> float:
    value = float(value)
    if not math.isfinite(value) or value < 0:
        raise ValueError("Shutdown timeout must be finite and non-negative")
    return value


@dataclass
class SinkEvent:
    ts: float  # Wall clock, retained for compatibility; not monotonic time.
    channel: str
    payload: dict[str, Any]


class _Subscriber:
    def __init__(self, callback: Callable, name: str, max_queue: int):
        self._callback = callback
        self._name = name
        self._q = queue.Queue(maxsize=max_queue)
        self._lock = threading.Lock()
        self._stop = threading.Event()
        self._accepted = self._delivered = self._errors = self._dropped = 0
        self._thread = threading.Thread(target=self._run, name=f"sink-{name}", daemon=True)
        self._thread.start()

    def push(self, event: SinkEvent):
        with self._lock:
            if self._stop.is_set():
                return
            try:
                self._q.put_nowait(event)
                self._accepted += 1
            except queue.Full:
                self._dropped += 1

    def request_stop(self):
        with self._lock:
            if not self._stop.is_set():
                self._stop.set()
                # FIFO sentinel follows accepted events. A full queue drains
                # before the worker observes Empty and exits without a sentinel.
                with contextlib.suppress(queue.Full):
                    self._q.put_nowait(None)

    def stop(self, timeout: float = 1.0):
        timeout = _timeout(timeout)
        if threading.current_thread() is self._thread:
            raise RuntimeError("A subscriber cannot synchronously close its own callback")
        self.request_stop()
        self._thread.join(timeout=timeout)
        if self._thread.is_alive():
            raise TimeoutError(f"Subscriber {self._name!r} is still draining; retry close")
        if self.statistics()["pending"]:
            raise RuntimeError(f"Subscriber {self._name!r} terminated with undelivered events")

    def statistics(self):
        with self._lock:
            return {
                "name": self._name,
                "accepted": self._accepted,
                "delivered": self._delivered,
                "callback_errors": self._errors,
                "overflow_dropped": self._dropped,
                "pending": self._accepted - self._delivered - self._errors,
                "closing": self._stop.is_set(),
                "stopped": not self._thread.is_alive(),
            }

    def _run(self):
        while True:
            try:
                event = self._q.get(timeout=0.05)
            except queue.Empty:
                # Empty may have been observed just before an emitter enqueued
                # an event and close set the stop flag. Recheck atomically with
                # push/request_stop rather than discarding that final batch.
                with self._lock:
                    if self._stop.is_set() and self._q.empty():
                        return
                continue
            if event is None:
                return
            try:
                self._callback(event)
            except Exception:
                with self._lock:
                    self._errors += 1
                logger.exception("EventSink subscriber %r raised", self._name)
            else:
                with self._lock:
                    self._delivered += 1


class EventSink:
    """Fan-out broker. Emit never waits for callbacks or queue space.

    close() and unsubscribe wait for accepted callbacks; timeout raises instead
    of falsely claiming a flush. They may be retried, outside the callbacks.
    Emissions after close are ignored; subscriptions after close are rejected.
    """

    def __init__(self, ring_size: int = 200, max_subscriber_queue: int = 1000):
        self._ring = deque(maxlen=ring_size)
        self._subscribers = []
        self._retiring = []
        self._finished = []
        self._lock = threading.Lock()
        self._max_q = max_subscriber_queue
        self._closed = False

    def emit(self, channel: str, payload: dict[str, Any], ts: float | None = None):
        event = SinkEvent(time.time() if ts is None else ts, channel, payload)
        # Acceptance and enqueue share the shutdown cutoff. Close cannot
        # overtake an event accepted into the ring but not yet enqueued.
        with self._lock:
            if self._closed:
                return
            self._ring.append(event)
            for subscriber in self._subscribers:
                subscriber.push(event)

    def subscribe(self, callback: Callable, name: str = "anon") -> Callable:
        with self._lock:
            if self._closed:
                raise RuntimeError("Cannot subscribe to a closed EventSink")
            subscriber = _Subscriber(callback, name, self._max_q)
            self._subscribers.append(subscriber)

        def unsubscribe(timeout: float = 1.0):
            timeout = _timeout(timeout)
            if threading.current_thread() is subscriber._thread:
                raise RuntimeError("A subscriber cannot synchronously close its own callback")
            with self._lock:
                if subscriber in self._subscribers:
                    self._subscribers.remove(subscriber)
                    self._retiring.append(subscriber)
                subscriber.request_stop()
            subscriber.stop(timeout)
            self._record_finished(subscriber)

        return unsubscribe

    def _record_finished(self, subscriber):
        with self._lock:
            if subscriber in self._retiring:
                self._retiring.remove(subscriber)
                self._finished.append(subscriber.statistics())

    def tail(self, n: int = 200):
        with self._lock:
            return list(self._ring)[-n:]

    def statistics(self):
        with self._lock:
            return [dict(row) for row in self._finished] + [
                sub.statistics() for sub in self._subscribers + self._retiring
            ]

    def close(self, timeout: float = 1.0):
        deadline = time.monotonic() + _timeout(timeout)
        with self._lock:
            subscribers = self._subscribers + self._retiring
            if any(threading.current_thread() is sub._thread for sub in subscribers):
                raise RuntimeError("Close the EventSink outside its subscriber callbacks")
            self._closed = True
            self._retiring = subscribers
            self._subscribers = []
            for subscriber in subscribers:
                subscriber.request_stop()
        pending = []
        for subscriber in subscribers:
            try:
                subscriber.stop(max(0.0, deadline - time.monotonic()))
            except TimeoutError:
                pending.append(subscriber._name)
            else:
                self._record_finished(subscriber)
        if pending:
            raise TimeoutError(f"Subscribers still draining: {pending}; retry close")
