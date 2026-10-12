# Core extraction qualification — 25 September 2026

This is the historical extraction record. For the current `0.1.0rc1` candidate,
see [release scope](RELEASE_READINESS.md). The maintained default is now
`task-core-v1`; `paper-core-v1` remains explicit for reproduction and hardware.

Scope: the standalone CPU package and its default `paper-core-v1` profile, plus
the optional reference robot skill library tested with fake backends. This does
not qualify ROS, real robot motion, simulator outcomes, benchmark results or all
historical prompt/feature variants retained in the internal engine at that time.
The `0.1.0rc1` release removed the development prompt variants.

## Checks

- Fresh Python 3.13 virtual environment; ordinary wheel installation, not an
  editable link or a checkout-dependent `PYTHONPATH`.
- CPU demo runs in isolated Python mode (`python -I`) using the installed wheel.
  Three scripted decisions produce inspection, a report and one acknowledged
  terminal submission. No network/model inference or GPU is used.
- Core `0.1.0.dev3`: 423 tests passed and 22 physical-integration tests are
  kept outside the core-only suite. No expected-failure marker remains for the
  shutdown issue. The initial dev0 snapshot recorded 399 passes and one XPASS.
  Four additional integration/provider test files are not extracted; their
  reasons are preserved in the extraction manifest.
- Full serialized HTTP request payloads, all 19 prompt bundles, configuration,
  report outputs and terminal calls match the original controller in three
  replay cases: report, empty report then correction, and rejected submission
  then acknowledgement. Both processes fix `PYTHONHASHSEED=0`; no payload fields
  are stripped to obtain equality. All decision requests use 1,024 output tokens.
- A separate wire-level test verifies the 512-token history-summary override.
- File hashes check both the checkout and the loaded installed engine against
  the extraction manifest. An import blocker runs the demo with ROS, original
  agent/benchmark packages, simulators and Torch explicitly unavailable.
- Fresh environment dependency check reports no broken requirements.
- Optional robotics package dev3: 234 tests pass with navigation/vision extras
  (skills, graph planning, physical I/O and fake ROS transport), including the physical skill tests
  kept out of the core-only suite. These suites overlap in generic contract
  coverage; their counts are not a count of unique experiments.
- Tool schemas/order, declared perception templates and five actor-facing skill
  documents match the Capture27 source reference. Installed skill files match
  the extraction manifest. Import/registration works while ROS, old repositories,
  simulators and Torch are explicitly blocked.

Reports: `artifacts/core-tests.xml`, `artifacts/reference-parity.json`,
`artifacts/robotics-tests.xml` and `artifacts/robotics-parity.json`.
Current environment package pins are saved with the checkpoint backup, not
represented as the historical paper's serving environment.

## Issues intentionally not hidden

1. The original asynchronous event subscriber could drop queued records on close.
   This is fixed in dev1 and documented in [TRACE_SHUTDOWN.md](TRACE_SHUTDOWN.md).
   A successful close drains accepted callbacks; callback errors, queue overflow
   and durable-storage guarantees must still be accounted for separately.
2. The original py_trees blackboard is process-global. The new public facade
   rejects simultaneous runtimes in one process and requires explicit close.
   Process-parallel benchmark execution is the first supported route.
3. Physical skill tests formerly hidden under the old core directory now run
   against the separately installed robotics package, not a developer's old
   repository. Robotics dev4
   adds opt-in simulation query-refresh and measured-push profiles, without
   replacing the Capture27 defaults; 278 robotics tests pass locally.
   Robotics dev5 separately adds the [hardware profile](HARDWARE_PROFILE.md):
   508 total robotics tests, six archived tool-schema matches, 43 recorded battery
   decision checkpoints, and a network-isolated zero-motion ROS intervention
   probe. Dev6 extends this to 531 robotics tests and the
   [typed ROS/mux/posture/publication binding](HARDWARE_WIRING_20261006.md).
   All hardware checks remain offline; physical validation is deferred.
4. `_engine` is temporary compatibility scaffolding, not the finished cleaned
   internals or a public API. The default profile is supported; planner,
   evaluator and watchdog activation are not exposed as qualified release APIs.
5. Dev3 replaces capture/survey implicit shared-directory debug writes with
   optional callbacks. This artifact-routing patch is recorded separately from
   frozen skill/prompt parity. The physical-session integration records required
   evidence independently, including actual VLM input bytes.

Incremental internal cleanup continues behind these checks. The independent
benchmark repository now contains primary50, transfer111 and matched fault/control
symbolic environments, plus separate native and final paper graders. Its optional
BT-Agent connector is offline-qualified against the original text controller in
15 complete request/observation replays. A local POSIX parallel/resumable text
runner is now available; historical campaign protocols and live simulator/hardware
qualification remain separate work.

Core dev4 adds an opt-in task-centric runtime, without modifying `_engine` or
changing the paper default. Its [migration checks](TASK_STATE_MIGRATION.md) include
full HTTP parity, cancellation/reset and Habitat session fixtures. Robotics dev7
only updates the core dependency; existing simulator installations remain pinned.
