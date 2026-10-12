# Accepted-event shutdown fix (0.1.0.dev1)

The frozen subscriber set its stop flag before processing its queue. If a
callback was active when close arrived, the worker could return immediately
after that callback and discard the remaining accepted events. A second race
allowed close to overtake an emission between ring insertion and queue insertion.

The maintained broker is `bt_agent.tracing`. The old import location is a
compatibility shim, listed as an intentional deviation in
`reproduction/core-patches.json`. Controller decisions and prompt text are not
changed. The original implementation remains in the frozen source snapshots.

## Contract

- Emit does not wait for callbacks or queue space. Queue overflow is still
  possible and is counted, not presented as delivered data.
- Close and unsubscribe stop accepting events and drain previously accepted
  callbacks in order. They return only after the worker exits, or raise an
  explicit timeout. Retry close after the slow callback can finish.
- JSONL/timeline recorders keep their file open when unsubscribe times out.
- Subscribing after close and synchronously closing from one's own callback
  are rejected. Emissions after close remain ignored for compatibility.
- `statistics()` exposes accepted/delivered/pending counts, callback errors and
  overflow drops. Zero pending does not imply successful writes if callback
  errors are nonzero. Callback completion is not `fsync` or durable storage.

## Evidence

`scripts/prove_trace_shutdown.py` blocks the first callback, enqueues two more,
then closes the sink. The original delivers `[0]`; the release delivers
`[0, 1, 2]`. This is a synthetic regression, not a measurement from robot trials
and not evidence that a particular historical bag lost records.

Tests cover full queues, FIFO drain, immediate recorder close, timeout/retry,
file lifetime, callback errors, self-close, and an Empty-poll race at the final
enqueue. The original identity-stamp regression is now an ordinary passing test.
Core model-request parity is checked independently against the frozen reference.
