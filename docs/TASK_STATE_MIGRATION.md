# Task state migration

The planner-free agent holds its task directly by default as of core dev5.
`task-core-v1` uses an immutable `TaskState` and a select → execute → submit supervisor tree.
There is no stored plan, milestone cursor or milestone-advancement node.
Select `paper-core-v1` explicitly for the preserved reproduction path. That
profile is an unpolished research snapshot of the paper controller, kept for
reproduction and the hardware binding.

This checkpoint separates a state refactor from a policy change. It does not
change prompts, token budgets, skill definitions, feedback or grading. It has CPU
and bounded rendered Habitat qualification, not new live robot validation.

## Use

```python
from bt_agent import create_agent

with create_agent(model=model, skills=skills, host=host, profile="task-core-v1") as agent:
    result = agent.run("Inspect the shelf and report its contents.")
    print(agent.task.instruction, agent.task.status)
```

New skills can read `context.task`. Existing skills retain `main_task`,
`scenario`, `milestone` and `milestone_index` compatibility properties. The
milestone property returns a fresh dictionary for legacy consumers; it is not
the task's storage. Mutating it cannot change the task.

The Actor still sees the frozen wording, including references to a milestone.
The terminal command also retains its historical sentence about an exhausted
plan. These are compatibility boundaries, not descriptions of the new state
model. Cleaning up that wording requires a separately versioned policy change.

## Lifecycle

An Actor completion claim moves the task to `submitting`. Only a terminal host
response with `action_ok is True` moves it to `submitted`. Rejected submissions
retry without another model call. Submission is termination, not a success grade.

Task history and objects persist until the next task starts. Skill failures
return their typed feedback to the Actor. Cancellation stops the active skill
and waits for its acknowledgment before reset or ownership release. Late model
responses cannot execute skills or overwrite a replacement task. A selection
thread that is already running may finish its model call and the frozen actor's
re-prompts or document reads in the background; the runtime does not claim to
cancel remote inference.

The original skill guards still read a global cancellation flag. The new runtime
scopes that flag to its ownership period and clears stale cancellation at task
start. Both profiles therefore still enforce one active runtime per process.
This bridge does not store task, history or completion data on the blackboard.

Observers keep the skill and submission event names. They should not expect
`plan_generated` or `milestone_advanced` events from the new profile. Supervisor
tick counts also differ because the empty planning/advancement stages are gone.

## Verification

Initial migration checks on 7 October 2026, before default promotion:

- 451 core tests pass on Python 3.10 and 3.13, with the existing 22 core-only
  physical tests kept separate. The 28 new tests cover state ownership, request
  parity, gating, compaction, images, typed failures, cancellation and reset.
- All 18 paired HTTP replay cases match across both profiles: 118 complete
  requests plus host actions and available grades. Fifteen cases use the text
  benchmark's primary, transfer, fault and invalid-call fixtures.
- The paper profile still matches the original controller in the three frozen
  reference replays, including all 19 prompt bundles present at that checkpoint.
  The `0.1.0rc1` release ships only the released bundle, and the reference check
  now compares that bundle.
- All 531 robotics regression tests pass. The new task profile additionally
  passes 14 Habitat session integration fixtures using the real skill code with
  synthetic pixels, scripted model responses and fake navigation. No GPU,
  simulator, model endpoint or robot was started.

Reproduce the core and paired replay checks:

```bash
python -m pytest
python scripts/check_task_state_parity.py --benchmark-root /path/to/robot-benchmark
```

The benchmark argument is optional. Without it, the script runs the three core
HTTP cases; the benchmark-fixture cases need the companion benchmark, which is not
yet public. With benchmark dev5 or later, it uses the adapter's explicit
`runtime_profile` argument and verifies the loaded choice; it does not edit
profiles or qualified installations. Its receipts are written under
`artifacts/task-state-20261007/`, which is not part of this repository.

## Maintained default and compatibility

Core dev5, robotics dev8, benchmark dev6 and Habitat host dev7 make the task
runtime the maintained default. The factory, text adapter, campaign definitions
and Habitat session/launcher all resolve an omitted profile to `task-core-v1`.
The loaded manifest still records the resolved choice.

The hardware binding now explicitly selects `paper-core-v1`: its execution guard
targets the frozen tree. Historical replay helpers and the legacy workstation launcher
also retain paper selection. This promotion does not migrate hardware or remove
the paper runtime. Use a fresh installation and new campaign directory; old
code/configuration identities are not rewritten to make a resumed run fit.

Both profiles passed paired CPU tests, ROS/native preflight and eight bounded
rendered Habitat checks before promotion. They cover live capture, safe interaction,
reset, reporting and motion/budget cancellation. That record is in the companion
benchmark, which is not yet public.
Default-specific regressions exercise omitted and explicit profile arguments.

Planner support and prompt-wording cleanup remain outside this change. Live robot
validation is not part of this release.
