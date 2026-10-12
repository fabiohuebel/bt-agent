# BT-Agent

Code release for the paper *BT-Agent: Deploying Onboard Robot Agents with
Behavior Trees and Multi-Fidelity Evaluation* (Hübel et al., 2026,
[arXiv:2610.13862](https://arxiv.org/abs/2610.13862)). The version
is `0.1.0rc1`; see [release notes](CHANGELOG.md) and [release scope](docs/RELEASE_READINESS.md).
Project-authored code and docs use [Apache-2.0](LICENSE), with
[third-party and data exclusions](LICENSE-SCOPE.md).

This is research software. Robot use is experimental: it needs your own platform
services, an independent operator stop and supervised acceptance testing.

An independently installable Python package for an agent that selects authored
behavior-tree skills. The maintained runtime (`task-core-v1`) stores the task
directly. A separate research profile (`paper-core-v1`) keeps the paper controller
as it ran, for reproduction. Both use the same released prompt set. The internal
`_engine` package holds the frozen research controller and is not a public API.

## Choose a starting point

| Goal | Guide |
|---|---|
| Try the agent without a robot or model | CPU example below |
| Implement your own skill | [Authoring BT skills](docs/ADDING_SKILLS.md) |
| Run text or Habitat tasks | The companion benchmark `robot-benchmark`, not yet public; its release is pending redistribution permission for the task suite and scene data |
| Integrate an existing robot stack | [Hardware preparation and deployment day](docs/HARDWARE_DEPLOYMENT.md) |
| Reproduce the paper controller (research state) | [Runtime profiles](docs/TASK_STATE_MIGRATION.md) and [source mapping](docs/MIGRATION_MAP.md) |

The core is independent of the benchmark. Robotics is optional; Habitat's native
renderer and Habitat's ROS bridge live in the benchmark repository, not this package.

## Run the CPU-only example

```bash
python -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/python examples/cpu_demo.py
```

No GPU, ROS, simulator, model server or API key is needed for the example.
It uses three explicitly scripted model decisions: inspect a synthetic shelf,
report its contents, then claim completion. The real controller executes the BT
skills and submits through the host's acknowledgement protocol. This is a test
fixture, not an experimental result.

## Public entry points

- `bt_agent.create_agent(model=..., skills=..., host=...)`
- `bt_agent.skills`: `ContextSkill`, `ActionSkill`, `ReportSkill`, context/handle,
  object-store and status types.
- `bt_agent.episode.EpisodeHost`: report/artifact callbacks and terminal
  acknowledgement, without a task-success oracle.
- `bt_agent.models.ChatModel`: explicit compatible HTTP endpoint configuration.
- `bt_agent.recording.RecordedModel`: model-interface call and exact input-image
  recording, without changing forwarded arguments or responses. This is not a
  raw HTTP/usage recorder; provider-hidden data is not reconstructed.

`ChatModel` does not create the reference client's implicit `/tmp` conversation
log or snapshot environment variables. Use `RecordedModel` for an explicit
recording directory. Setting `LAM_VLM_LOG_DIR` deliberately enables the preserved
request/exchange recorder; those artifacts contain task text and possibly images
and need appropriate storage permissions. Transport failures raise
`bt_agent.models.ModelTransportError` (a `requests.RequestException`) with
`error_type` and `status_code`, without copying raw exception messages or
credential-bearing URLs into agent feedback. Importing the internal reference
client directly does not provide these release logging protections.

`create_agent(..., internal_skills=[...])` can provide context-only internal
fallback handlers without advertising them as model tools. The text benchmark
uses this for the frozen `wait` no-op. Names cannot shadow registered or terminal
tools; internal physical actions are rejected. This is a dispatch interface, not
a security boundary for untrusted skill code.

Use the runtime as a context manager so active work is cancelled on close.
`run()` returns an `EpisodeOutcome` containing the termination reason, submission
flag and tick count. **Submitted does not mean task success.** External grading
stays separate. A new task must not start while an earlier task is active.
`objects_snapshot()` gives an output adapter detached skill artifacts without
exposing mutable controller state. Call it on the controller thread.

The default `task-core-v1` profile stores the task directly instead of constructing
a synthetic milestone. Select `profile="paper-core-v1"` explicitly to use the
paper controller. Both retain the frozen `tiamat_agent_submit_v5` prompt and host
protocol, with planner/evaluator disabled and no watchdog added. The hardware
binding explicitly stays on `paper-core-v1`; it does not inherit this default.
See [task-state migration](docs/TASK_STATE_MIGRATION.md) for checks and limits.

`paper-core-v1` is a research snapshot. It is the controller as it ran for the
paper's experiments, kept so those runs can be reproduced. It is unpolished: it
keeps the milestone-era structure, historical names and the limitations listed
below, and it will not receive new features. Use `task-core-v1` for new work
outside the hardware binding.

In both profiles, actor decisions use 1,024 output tokens, and history summaries
use 512. A model backend's 4,096-token fallback is not the decision cap.
`agent.describe()` records the loaded configuration, prompt/tool identities and
source hashes without including API keys or authenticated endpoint URLs.

Use `create_agent` to select a runtime. The exported `AgentRuntime` class is the
legacy paper facade; the factory now returns `TaskRuntime` unless paper mode is
requested. Install a tagged checkout in a fresh environment; do not overwrite a
preserved paper installation to resume an old campaign.

For custom skills, see the [skill-authoring guide](docs/ADDING_SKILLS.md) and
`InspectShelf` in `bt_agent.testing`: author a
`build_subtree(params, context, handle)` implementation. Physical action skills
also implement their precondition and running invariant. This package does not
provide a robot's controller or guarantee physical stopping.

## Verification

```bash
.venv/bin/python -m pip install '.[test]'
.venv/bin/python -m pytest
```

GitHub CI installs the core and optional robot skills on Python 3.10 and 3.13
and runs their CPU suites. It starts no robot, model server or simulator.
Reference replays against the original research controller are a separate
qualification check. They need the original research repository, which is not public.

The core-only suite keeps physical-integration cases separate. Their counterparts
are now exercised by the optional robotics package's tests. Historical test
file boundaries are recorded in `reproduction/core-source.json`. The original
logging shutdown race is fixed in `0.1.0.dev1`; its test is no longer marked
expected-to-fail. The public `bt_agent.tracing.EventSink` drains accepted callbacks
on close or raises a retryable timeout. Queue overflows and callback errors remain
visible through `statistics()`; callback completion is not a guarantee of durable storage.

To regenerate the immutable reference fixture and compare full serialized
requests, the released prompt bundle and report/submission behavior:

```bash
python scripts/extract_reference.py --reference-only --source-repo /path/to/original/tiamat_agent
python scripts/check_reference_parity.py
```

Only the parity check needs the original Git objects. Installation, ordinary
tests and the CPU example do not. `scripts/extract_reference.py` is a mechanical
migration tool; use `--reference-only` once release patches exist. Approved
departures from the extracted files are recorded in `reproduction/core-patches.json`.

The parity check fixes `PYTHONHASHSEED=0` in both subprocesses because the
historical controller derives history call IDs from Python's randomized hash.
This test setting does not change the library's production environment.

## Known limitations

- One active runtime per process: the preserved py_trees blackboard is global.
  The task runtime stores its task locally, but the preserved skill guards still
  share a cancellation flag. Both profiles reject a second runtime. Parallel
  episodes require separate processes.
- Cancellation acknowledgement from the software is not a measurement of a
  robot's velocity or posture.
- An optional [robotics skill package](packages/bt-agent-robotics/README.md) is
  extracted and checked with fake backends. Its camera/publication adapters and
  ROS navigation wire are tested. Habitat integration is qualified separately in
  the companion benchmark, which is not yet public. The [hardware profile](docs/HARDWARE_PROFILE.md) has migrated
  the deployed fixes and guard logic. Its [ROS/mux binding](docs/HARDWARE_WIRING_20261006.md)
  passes offline checks with typed synthetic telemetry and fake services.
  No robot test has been run for this release; physical validation is deferred.
- See the [migration map](docs/MIGRATION_MAP.md) for the extraction boundary.
- `run()` does not raise for model errors. A failed model call becomes a `wait`
  selection: an internal `wait` handler makes it a no-op, and without one the
  model sees `Unknown skill: wait`. A persistently failing endpoint therefore ends
  the episode as `tick_limit`. Record calls with `RecordedModel` when an episode stalls.
- `run()` bounds an episode by ticks, not wall time. The defaults
  (`max_ticks=10000`, `tick_interval=0.01`) allow roughly 100 s; pass larger values
  for slow models.
- After `cancel()` or `close()`, a selection thread that is already running may
  finish its model call and the frozen actor's re-prompts or document reads. It
  cannot execute a skill or change a new task.
- In `paper-core-v1` (also used by the hardware binding), a model call to the
  un-offered `defer` tool makes the frozen controller re-run the previous skill
  with its previous parameters, or fail if there is none. `task-core-v1` rejects
  it as an unknown skill.
- The internal engine still contains the research controller's milestone
  planner and evaluator. Both profiles switch them off, and they are not
  supported in this release.
- The frozen default actor prompt was written for the paper's task setting. It
  refers to milestones, an evaluator and the environment's `submit` command,
  although the released profiles disable the planner and evaluator. Keep this
  wording in mind when you add your own skills.
- See [release scope](docs/RELEASE_READINESS.md) for what was verified and what
  this release does not claim.

## Citation

```bibtex
@misc{huebel2026btagent,
  title         = {{BT-Agent}: Deploying Onboard Robot Agents with Behavior
                   Trees and Multi-Fidelity Evaluation},
  author        = {H{\"u}bel, Fabio and Lilja, Adam and Yin, Yue and He, Siming
                   and Tomlin, Claire and Sastry, Shankar and Frey, Jonas and
                   Pavone, Marco},
  year          = {2026},
  eprint        = {2610.13862},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
}
```

This work was supported in part by the DARPA TIAMAT program.
