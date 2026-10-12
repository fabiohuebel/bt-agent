# Hardware profile and offline qualification

The hardware-profile adapter and its
[offline-qualified ROS binding and mux](HARDWARE_WIRING_20261006.md) preserve the
deployed skill/control sources separately from the default Capture27 and later
Habitat profiles. It does **not** install software on a robot or provide a
production bringup command. Offline integration checks pass. The
[deployment guide](HARDWARE_DEPLOYMENT.md) defines what an application must wire
and what remains to check during a prospective robot day.

## What is included

`bt_agent_robotics.hardware.HardwareAgent` composes the unchanged `paper-core-v1`
controller with the hardware query, navigation, press and capture variants.
`paper-core-v1` is an unpolished research snapshot of the paper controller.
Robotics dev8 selects this profile explicitly even though core dev5 defaults to
the task runtime. The hardware execution guard is not silently moved to a new tree.
It retains the core report/acknowledged-submission path: `claim_complete` is the
actor-facing completion tool; `submit` is not added as another skill.

The hardware profile includes the deployed local-inspection capture fix, operator
confirmation for navigate/press, and execution-time checks of the offered tool
set and current conditions. Survey help is an explicit choice: `original` or
`visual_question_v1`. The latter changes tool help, not the perception algorithm.
Planner and evaluator remain disabled, with actor/summary output caps 1024/512.

Optional intervention and battery adapters are explicit services. Interruption
and injected timeout/failure end the current skill while leaving the task agent
active. Requests bind to one execution key, and stale scopes cannot issue motion.
The battery policy blocks new motion below 10%, latches at or below 5%, and treats
missing/stale data separately. Stop-to-sit requires fresh stationary, brake,
ownership, hold and mux-suspension evidence. An acknowledgment alone is not proof
of sitting. Task reset and clearing a simulated reading do not release the hold.

Sources are pinned to `517c18ad9f861fce657643f4ee263459279fc9bd`, with original and
relocated hashes in `reproduction/hardware-source.json`. The source matches the
physical skill hashes in the saved deployment manifest. The core policy is
still the independently pinned `62056af6f` profile. This does not retroactively
make all Berkeley/Stanford missions identical configurations.

## Construct with explicit services

The caller supplies navigation, path planning, scene memory, perception, operator
confirmation and the episode host. Constructing the adapter starts no ROS node,
robot connection or model server. The application's sensor and publication
adapters remain responsible for freshness, coordinate frames and durable output.

```python
from bt_agent_robotics.hardware import HardwareAgent, HardwareProfile

profile = HardwareProfile(
    survey_description="visual_question_v1",
    local_inspection=True,
    exploration_fallback=False,
    task_contract="general",
)
agent = HardwareAgent(
    model=actor_model, host=episode_host,
    navigation=navigation, path_planning=path_planning,
    scene_memory=scene_memory, perception=perception_model,
    confirmation_gate=operator_confirmation, profile=profile,
)
```

This is an integration fragment, not a robot launch recipe. For compatibility,
the adapter requires explicit `LAM_NAV_CONFIRM=1`, matching
`LAM_CAPTURE_LOCAL_INSPECTION` and `LAM_FARM_EXPLORATION_FALLBACK` values (`0`/`1`),
and no top-k query or simulator experiment arm. It validates these retained
switches instead of silently changing process-wide environment state. The
confirmation timeout defaults to 300 seconds and must be finite and positive.
Runtime changes to those settings cancel the active episode and raise an error.

Use `start`, `push_state`, `tick`, `cancel` and `close` on this adapter, preferably
as a context manager. One active runtime per process remains the limit. Different
profiles with conflicting skill docs require separate processes; the adapter
refuses mismatched first-registered documentation.

`describe()` records the actual controller/tool identities, physical-source
bundle, selected survey/capture/query settings and installed execution guard.
It omits authenticated endpoint URLs. It explicitly does not claim whole-stack
parity or post-refactor hardware validation.

## Verified in this checkpoint

- 222 migrated hardware-reference tests: guard transitions, persistent holds,
  stop-to-posture state machine, fake ROS callbacks and deployed skill behavior.
- Eight new full-controller/profile tests: report/submit/reset, rejected
  submission, execution-time tool rejection, interrupt/timeout feedback followed
  by reporting, configuration conflicts and early safeguard rejection.
- Six tool schemas, the actor prompt and physical skill-source hashes match a
  preserved deployed manifest. Forty-three logged battery checkpoints produce
  matching policy decisions from their recorded inputs/latch state. This is not
  continuous sensor replay or a remeasurement of robot stopping latency.
- An installed-wheel ROS probe on isolated domain 97 sends intervention requests
  through the preserved bridge and receives acknowledgments plus brake/zero/park
  messages. All velocities and goal positions are zero. No final actuator-output
  publisher, driver, model, simulator or robot is started.
- The default skill contract and the core prompt-bundle reference checks pass. At
  this checkpoint they covered 19 bundles; `0.1.0rc1` ships and checks one.

Release changes beyond namespace relocation are explicit debug-artifact callbacks
and a terminal-scope cleanup in the new adapter. The latter fixes an early
safeguard rejection that could leave a never-ticked motion body marked active.
It retires only terminal handles and never clears a battery hold. The original
source snapshot stays intact. No frozen simulator policy/skill behavior changes.

## Integration boundary and rollback

The offline gate includes typed Spot telemetry, the preserved mux's
persistent ownership/hold protocol, the press-precision wire, fake stop-to-sit
sequencing and report/capture/terminal publication. See the wiring qualification
for exact coverage and limits. FARM/model hosting, maps, calibration, physical
controllers, drivers and lifecycle dashboard remain application inputs, not
automatically started by this package.

The `0.1.0rc1` candidate additionally supplies an opt-in driver input profile:
compressed camera topics, source-stamped vision-to-map odometry conversion and
static camera-transform chain resolution. See the deployment guide for its
fail-closed behavior and unsupported dynamic-TF case. This does not change the
frozen agent/skill policy or supply a robot-specific lifecycle application.

The prospective deployment day is optional and has not happened. Label this
adapter offline-qualified until the exact installed candidate passes the live
acceptance checks; do not transfer an old deployment result to the new release.

Rollback of this checkpoint is local: stop the owned test container, close its
runtime, and use the previously recorded wheel/environment. Nothing was deployed
to the robot computer. A future robot rollback must preserve brakes and any persistent battery
hold before switching versions. Never remove hold/latch files, restart an old
unfenced mux, stand, or release GO as part of an automatic package rollback.

Evidence: `artifacts/hardware-20261006/`. Source verification:
`python scripts/extract_hardware.py --source-repo /path/to/tiamat_agent --check`.
The original repository is needed only for extraction/source comparison, not
for installing or using the released profile.
