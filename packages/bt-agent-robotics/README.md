# BT-Agent robotics

This optional package contains the authored query, navigate, surround-survey,
capture and press BTs from the recorded Habitat Capture27 reference at `62056af6f`.
`inspection_skills(...)` builds that set plus the core's report skill in the
original registration order. The optional image-injection implementation is
retained for compatibility tests but is not in the default skill set.

The application supplies retrieval/scene memory, navigation/path planning and
VLM perception backends. Installation/import does not connect to a robot or
start a simulator. No GT perception backend is provided or silently substituted.

```bash
# From the bt-agent repository root, after installing the core:
.venv/bin/python -m pip install 'packages/bt-agent-robotics[navigation,vision]'
.venv/bin/python -m pytest packages/bt-agent-robotics/tests
```

The CPU suite covers skills, graph planning, camera
intake, publication, hardware-profile composition and fake ROS transport.
Tool schemas/order, declared
perception templates and five actor-facing skill documents match the pinned
reference (`scripts/check_robotics_parity.py`). This does not assert physical
outcome equivalence or include a live VLM test.

The [hardware profile](../../docs/HARDWARE_PROFILE.md) isolates the deployed
fixes and intervention/battery policies. The
[telemetry/mux/publication binding](../../docs/HARDWARE_WIRING_20261006.md) is
offline-qualified, not yet validated on the physical robot after refactoring.
Use the [hardware preparation guide](../../docs/HARDWARE_DEPLOYMENT.md) for the
repeatable offline suite, application integration boundary and live acceptance plan.
Later simulation interaction profiles are opt-in as described below. Existing
environment-variable settings and the legacy optional ROS confirmation helper
are retained in this initial extraction; separating those is follow-up work.
Actual ROS confirmation requires a ROS adapter/environment. The unit tests use
fake backends and fake ROS messages, not a live robot.

## Later physical profiles

`interaction_skills(...)` keeps the original inspection
tool order and adds the evaluated query-refresh behavior: after a native gate
opens, querying an existing object refreshes its viewpoints without changing its
handle. The default `inspection_skills(...)` remains the Capture27 profile.

`interaction_skills(measured_push=True, ...)` also registers `push` between
`press` and `report`, matching the archived factory. Its navigation adapter must
provide private, episode-bound push telemetry and an acknowledged speed lease.
Missing/stale feedback, wrong instances, overflow, cancellation and lack of
progress stop the skill. A completed navigation waypoint is not a completed push.

This optional push profile is specific to the benchmark's authored kinematic
rails; it is not a general force-based manipulation controller or a hardware
deployment profile. Native poses remain controller inputs, not Actor observations.
The Habitat adapter selects the archived chair waypoint mode or stool contact-
velocity mode and the explicit resume flags. It keeps the old stopping thresholds.

`reproduction/interaction-robotics-source.json` records the additional sources
and import transformations. Forty-four added tests cover query refresh, feedback,
speed leases, contact commands and retry geometry. Source-level baseline-parser
tests are excluded because baseline productization is out of scope. Live Habitat
checks and evidence are recorded in the companion benchmark repository, which is
not yet public; its release is pending redistribution permission for the task suite and scene data.

Source identities and scope exclusions are recorded in
`reproduction/robotics-source.json` at the repository root. Project-authored code
and docs use [Apache-2.0](LICENSE), with [licensing exclusions](LICENSE-SCOPE.md).
Live robot validation remains a separately recorded acceptance step.

## Graph navigation backend

`from bt_agent_robotics.navigation import NavGraphPlanner` provides the recorded
graph planner independently of ROS, Habitat and benchmark tasks. It consumes
NetworkX node-link dictionaries whose coordinates are already in the application's
map frame. Each node needs a `world: [x, y, z]` attribute; a node without it is
treated as the map origin. It retains the 0.45 m default waypoint threshold, tighter tracking near
named keep-outs, strict/penalised routing, and restoration of removed edges when
a fresh top-level navigation attempt starts. It is not a local collision controller.

The implementation is source-identical apart from its interface import; see
`reproduction/navigation-source.json`. It retains the reference's identity-based
map cache: supply a new dictionary when changing a map. Keep a loaded input alive
for its planner lifetime. Broader algorithm changes are a separate step, not
silently included in this extraction.

## Physical I/O adapters

`platform.PlatformState` stores one episode's five camera views, transforms and
observed odometry. It never fabricates an observed origin pose. Inputs use
monotonic receive times and a caller-owned episode epoch. The camera codec keeps
the reference rotations; use the `vision` extra for CPU image encoding.

`ros_navigation.RosLocalPlanner` is the preserved ROS waypoint/velocity adapter.
It publishes `/goal_waypoint`, `/spot/cmd_vel_primitive` and the `/stop` brake
bitmask. It does **not** own final `/spot/cmd_vel`; the mux does. Import this module
only inside an installed ROS environment. Fake publisher tests check topic,
frame, quaternion, cancellation and brake behavior without running ROS.

`publication.PhysicalPublication` formats captures/reports and publishes terminal
control only after the caller confirms acknowledged submission. Deduplication is
per episode/capture handle, not distributed exactly-once delivery. A successful
ROS publish is not remote acknowledgement. The application owns transport errors,
independent grading and physical stopping.

The optional `capture_evidence` callback receives the exact capture model image.
`PlatformState.on_frame` can preserve raw and corrected sensor bytes. These are
separate from the legacy capture wire, which has no model-frame identifier.

### Deliberate artifact-only change

Capture/survey do not write implicitly to shared `/tmp/lam_captures`
or `LAM_CAPTURE_DIR`. Pass `capture_debug_writer` / `survey_debug_writer` to
`inspection_skills` to opt into debug artifacts. As in the reference, debug-writer
errors are best-effort and do not determine task outcome; mandatory evidence
recording belongs outside those callbacks. The extraction manifest records this
patch. Prompts, tool schemas, response parsing and observations are unchanged.

The release candidate also disables navigation's implicit shared debug-file write,
matching the hardware profile's optional internal diagnostic hook. Normal skill
events and observations remain available for application-owned recording. This
does not change waypoint selection, guards, replanning or stopping behavior.
