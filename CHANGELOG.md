# Release notes

## 0.1.0rc1

First public release candidate, accompanying the paper. It reports no new
benchmark result.

- Standalone BT-Agent core and optional robotics skill package.
- Task-centric default runtime with an explicit paper-reproduction profile.
  `paper-core-v1` is an unpolished research snapshot of the paper controller.
- Typed skill feedback, object/history state, acknowledged submission and bounded
  cancellation, with request/prompt regression checks.
- Dead blackboard-client registrations are collected without clearing shared
  values or revoking live client permissions, avoiding growth across repeated tasks.
- Keyboard interrupts cancel active work, and invalid live configuration cannot
  disable cancellation. Invalid skill registration fails before runtime allocation.
- Loaded manifests identify both the frozen policy and the maintained runtime /
  hardware adapters by source hash.
- Interrupted model calls retain a terminal recording event and an explicit
  interruption flag; they are not left looking pending after the recorder closes.
- Public ChatModel no longer makes implicit conversation/environment dumps and
  sanitizes transport errors. Its HTTP payloads and response parsing are unchanged;
  recording remains available through explicit opt-in interfaces.
- Navigation documentation includes the original tree diagrams from each pinned
  skill source, without changing the model-visible Markdown.
- Core and optional robotics tests can run together. Core-only skip rules no
  longer suppress the corresponding physical-skill tests in a combined invocation.
- Simulation navigation no longer appends diagnostics to a shared `/tmp` file.
  Its optional internal diagnostic hook matches the hardware profile; navigation
  decisions, skill observations and model-visible documentation are unchanged.
- Hardware execution, intervention and battery adapters with isolated ROS tests.
- Partial ROS startup cleans up owned resources; cleanup errors are retryable.
  Cancellation still attempts brake/zero outputs when a current pose is unavailable.
- Opt-in driver camera/odometry input profile, including static TF chain resolution
  and coordinate conversion into the waypoint frame. Missing or changed transforms
  withhold inputs; the adapter does not calibrate the map.
- The task runtime asks the model again after text-only replies instead of
  recording a nameless tool call, and rejects an unoffered `defer` call as an
  unknown skill instead of aborting the episode.
- Removed an unused, workstation-specific CLI model provider from the internal engine.
- The release ships one prompt set, `tiamat_agent_submit_v5`, used by both profiles.
  It is flattened into one module with byte-identical strings. The 18 other
  development prompt variants are removed, including the six it was assembled from.
  The pure-ReAct ablation baseline, `DecisionMakerAgent`, `ExternalAgent` and an
  empty ALFWorld skills package are also removed.
- Apache-2.0 for project-authored code/docs, with third-party and data exclusions.

One runtime per process remains the supported execution model. A selection
thread already running at cancellation may finish its model call and the frozen
actor's re-prompts, but cannot execute a late skill.
Hardware retains the paper runtime and still requires platform services and live
acceptance testing. Dynamic map TF, robot drivers, FARM/model hosting and lifecycle
bringup are not silently supplied by the library. See the hardware deployment guide.
