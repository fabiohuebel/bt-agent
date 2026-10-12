# First extraction checkpoint

This page records the first extraction checkpoint and is kept as history. The
README describes the current release.

| Existing component | Destination now | Next boundary |
|---|---|---|
| Frozen `lam_v2_pycore/src/lam_v2` at `62056af6f` | `bt_agent._engine` with module-prefix relocation only | Replace historical internals incrementally behind tested public API |
| Agent configuration and registration | Public `create_agent`, one explicit core profile | Positive feature configuration after contract tests |
| `RosClaimBridge` / text terminal bridge | Explicit injected episode host | Shared report/submit mechanism, separate from independent grading |
| Authored physical skills in `lam_v2_tiamat` | Capture27 reference extracted into optional `bt-agent-robotics` | Reconcile later fixes and separate ROS confirmation/configuration |
| ROS launch/node/mux/platform integrations | Hardware profile and ROS adapters in `bt-agent-robotics` | Launch files and the robot application stay outside this release |
| Text tasks, environment and graders | Companion `robot-benchmark` repository (not yet public) | |
| Habitat / Isaac code and binary environments | Habitat backend in the companion `robot-benchmark` repository (not yet public) | No Isaac physics backend is released |

The reference is the frozen physical controller named in the deployment parity
audit, not a claim that every text, ablation and robot profile used identical
files. Reproduction of those cohorts continues to use their preserved manifests.
Later component-ablation branches add six changed/new core files; they are not
silently merged into this first extraction.

The internal `_engine` package is deliberate transitional scaffolding. Users should
not import it. At this checkpoint it preserved the historical prompt families and
disabled planner/evaluator code while we built behavior checks. This checkpoint
does not claim that all historical variants are supported or that the cleanup is
finished. The `0.1.0rc1` release removed the development prompt variants and keeps
only the flattened `tiamat_agent_submit_v5` set.

The first profile disables planner/evaluator through the existing controller
configuration and preserves `tiamat_agent_submit_v5` prompt bytes. No watchdog
is added. Physical checks remain the responsibility of authored skills.

The event broker has been moved into maintained `bt_agent.tracing`, with a
recorded compatibility shim in the extracted engine. Shutdown drains accepted
callbacks or reports a timeout. See `TRACE_SHUTDOWN.md` and the release-patch
manifest for this intentional instrumentation-only change.

Known qualification boundaries: process-global py_trees blackboard; one runtime
per process; no new async reasoning; physical skill tests use fake backends;
no ROS, simulator or live-model qualification;
no guarantee of physical stop just because cancellation was requested.

Source provenance is in `reproduction/core-source.json`.

## Task state migration

The default `task-core-v1` runtime stores the current task in `TaskState`, not
a fabricated one-element milestone list. Explicit `paper-core-v1` and the internal
engine retain their historical representation. No optional planner API is
implemented; the new profile rejects unsupported plan inputs.

Completion acknowledgement, reports, reset/cancellation, skill availability and
feedback are tested independently of milestone advancement. Frozen prompt and
skill wording remains behind a transient compatibility view. Skill BTs and their
local ticking are unchanged. See [migration evidence and remaining work](TASK_STATE_MIGRATION.md).

No live robot test is part of this release. Hardware adapters can be checked with mocks,
recorded traces and isolated non-motion ROS, with any post-refactor physical
qualification explicitly left unclaimed.
