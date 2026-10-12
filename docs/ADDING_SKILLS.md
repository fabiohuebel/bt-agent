# Adding an authored BT skill

Start with `InspectShelf` in `bt_agent.testing` and the CPU example. A skill owns
its behavior tree; the runtime selects skills, supplies context and records their
outcomes. Import the public types from `bt_agent.skills`, not `_engine`.

## Choose the interface

Use `ContextSkill` for retrieval, observation or reporting without physical
actions. Context skills are always offered to the Actor. Use `ActionSkill` for
physical actions and provide a precondition and a running invariant.

| Method or field | Responsibility |
|---|---|
| `name`, `description`, `params` | Declare the model-visible skill interface. |
| `tool_def()` | Return the tool schema. It is abstract; `return super().tool_def()` builds it from the fields above. |
| `prepare(params, context)` | Validate/normalize arguments; return `(prepared, None)` or an error string alongside the parameters. |
| `build_subtree(params, context, handle)` | Return a fresh `py_trees` behavior/subtree for this invocation. |
| `is_available(history, gating)` | Optionally restrict the advertised action menu using interaction history. |
| `precondition(context)` | For actions, return `(allowed, reason)` for the advertised menu. |
| `safeguard_invariant(context, handle)` | For actions, cheaply check fresh execution state on each local tick. Return `True`, `False` or a typed failure-reason string. |
| `publish_results(handle, store)` | Optionally publish selected artifacts after success. The default does nothing. |

Tool visibility is not an execution-security boundary. The generic controller
does not promise that a model can never name a hidden tool. Enforce required
execution checks in `prepare`, an authored BT guard or the running invariant.
The separate hardware binding adds its own execution-time guard. Do not change
frozen tool schemas/prompts as an incidental refactor.

## Execute and report

Leaves return `RUNNING`, `SUCCESS` or `FAILURE`. Set `handle.observation` to useful
feedback and place structured artifacts in `handle.result_data`. Keep diagnostic
status separate from a claim that the user's task succeeded. The runtime calls
the publication hook only after skill success and preserves failure feedback for
the next Actor decision.

The default `Skill.start` wraps the authored subtree with cancellation safeguards;
actions also get wall-time and invariant guards. Skill trees tick in a background
thread, independently of the task supervisor. Keep ticks short and bound backend
calls: a guard cannot preempt an indefinitely blocked Python call. Cancellation
acknowledgment is a software event, not proof that a robot has stopped physically.
The application must supply an appropriate stop path and fresh telemetry.

Use `context.platform_state` for execution state. The runtime supplies fresh
contexts to running invariant checks; do not retain an initial sensor snapshot
as your only safety input. Invariants should not depend on model reasoning or
the task-local object store. New task-runtime skills can read `context.task`.
For compatibility with the explicit paper runtime, use the common `main_task`
and `scenario` properties instead of assuming `task` exists there.

## Test and integrate

Register the instance through `create_agent(skills=[...], model=..., host=...)`.
Use injected fake backends and `ScriptedModel` first. Test invalid arguments,
blocked preconditions, mid-execution invariant failure, cancellation and result
publication. Then qualify the real backend separately. Use a fresh skill tree
per call and one active runtime per process.

Do not register `submit`, `claim_complete`, `defer` or `read_skill_docs` as user
skills: those names belong to the controller. `defer` is a retired milestone tool
that the frozen actor still intercepts by name. The Actor claims completion; the
runtime submits through the host and waits for acknowledgment. External grading
remains independent. `paper-core-v1` is the reproduction profile, an unpolished
research snapshot of the paper controller. The default `task-core-v1` stores the
task directly.
