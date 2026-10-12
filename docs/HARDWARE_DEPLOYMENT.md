# Hardware preparation and deployment day

The release provides a hardware agent and ROS binding, not a replacement for the
robot's driver, map system or lifecycle dashboard. Offline checks pass using
synthetic sensors and fake services. Physical validation of this release is still
pending. This guide prepares a bounded deployment day on the existing robot stack.

The [8 October offline results](HARDWARE_OFFLINE_20261008.md) record the current
checks and the map-frame configuration fix.

## Freeze the candidate before robot time

Use the coordinated `0.1.0rc1` core and robotics candidate in a separate environment.
Keep the known-working robot installation and its maps untouched. Record wheel
hashes, Git commits, Python/ROS versions, the robot architecture, model checkpoint,
serving settings, scene/navgraph hashes and frame configuration.

Hardware explicitly uses `paper-core-v1` with planner/evaluator disabled. That
profile is an unpolished research snapshot of the paper controller. The
task-centric runtime is the text/Habitat default, not an implicit hardware upgrade.
Keep the selected survey description, local-inspection and exploration options
fixed for the acceptance run. Save `episode.agent.describe()` from the loaded
agent; an environment-variable printout alone is not configuration evidence.

The current runtime checks ran on Linux x86-64, Python 3.10 and ROS Humble. A Jetson
installation needs its own ARM64 dependency/import check against the robot's ROS
environment. Do not copy the x86-64 virtual environment or the simulator's native
Python 3.9 environment onto it. The package wheels are Python code, but NumPy,
SciPy, OpenCV and ROS bindings have platform-specific dependencies.

## Run the offline suite

Install the core and robotics wheels with `navigation,vision` dependencies into
a new Python 3.10 environment. Build the test-only Spot message workspace using
[the wiring guide](HARDWARE_WIRING_20261006.md#checks-and-reproduction). It must
remain separate from any actual robot ROS workspace.

From this repository, use a local immutable ROS image ID and explicit paths:

```bash
python scripts/check_hardware_offline.py \
  --image sha256:REPLACE_WITH_LOCAL_IMAGE_ID \
  --python /srv/bt-release/python/bin/python \
  --test-workspace /srv/bt-release/test-interfaces \
  --read-only-root /srv/bt-release/python \
  --read-only-root /srv/bt-release/base-python \
  --out /srv/bt-evidence/offline-candidate
```

Mount the virtual environment and its base interpreter at their original paths.
The runner uses ten separate containers with no network or device access, domain
97, a read-only filesystem and a new evidence directory. It runs as the invoking
user, bounds each case to 90 seconds, and removes only its own containers. It never
downloads an image, contacts a model or starts a robot driver. No live-robot mode
exists in this command. `--case critical` selects a single case for diagnosis.

The suite covers interruption/timeout/failure acknowledgments, low-battery
admission, critical-battery stop and fake Sit, stale/unknown telemetry, denied
lease, duplicate output ownership, denied/unavailable Sit, and two complete
synthetic inspection episodes, plus two using the driver's camera topics, chained
transforms and a nonidentity vision-to-map transform. It checks report/submit publication while motion
remains held, including a mux restart. `summary.json` records per-case outcomes
and input script hashes. A failed launch or timeout remains a failure, not a
silently retried passing trial. These are software tests, not stopping-time,
perception or physical-safety measurements.

## Connect the existing application

The remaining integration work is at these boundaries. The release does not
discover or start the old deployment checkout automatically.

| Application input | Required check before motion |
|---|---|
| ROS node and executor | Correct robot interfaces installed; no test-only `spot_msgs` overlay; sensors continue updating while the actor waits |
| Pose and camera transforms | Fresh map-frame odometry and five correctly oriented cameras, either normalized by the application or through the explicit driver input profile; physically check map alignment |
| FARM and graph planner | Explicit service adapters, same map frame and graph identity; retained hardware planner, not Habitat's planner |
| Actor and perception models | Endpoint health, exact checkpoint, temperature zero and thinking disabled; record loaded identities without credentials |
| Confirmation gate | `publish_plan`, `poll`, `clear` and `enabled`; a fresh approval belongs to the current goal, not a previous one |
| Motion mux and guards | One actuator-output owner; persistent hold shared with battery guard; press-precision receiver patch installed if press is tested |
| Recording and lifecycle | Durable images, reports, terminal signal, model requests and ROS bag; independent operator stop remains available |

Construct `RosHardwareIO` with output `pose_frame="map"`, matching its preserved
waypoint publisher. By default it expects already normalized camera topics,
map-frame odometry and direct camera transforms. The optional driver profile now
accepts the recorded driver's input layout and resolves static transform chains:

```python
from bt_agent_robotics.hardware_inputs import HardwareInputProfile
from bt_agent_robotics.hardware_ros import RosHardwareIO

inputs = HardwareInputProfile.spot_driver()
io = RosHardwareIO(
    node, epoch=task_id, evidence=evidence_directory, pose_frame="map",
    body_frame=inputs.body_frame, camera_frames=inputs.camera_frames,
    input_profile=inputs,
)
```

This is application wiring, not a robot launcher. The profile subscribes to the
driver's compressed-camera topics and converts its `vision` odometry using
explicit `/tf_static` transforms. It preserves source stamps, body-frame twist
and the skill camera-orientation convention. It does not assume identity alignment.
Missing transforms withhold inputs; changed/conflicting static edges invalidate
their dependent inputs until the application creates a fresh session. Dynamic TF
alignment is not supported by this profile and must be normalized explicitly by
the application. Record `io.readiness()` alongside `episode.agent.describe()`.
Sensor readiness is not evidence that the map is physically aligned.

A read-only sample of the archived Stanford control bag found `vision`/`body`
odometry, `/camera/{name}/compressed` images with `{name}_fisheye` frames, and
`body → head → camera → camera_fisheye` transform chains. Its `vision → map`
transform was identity. The old application used camera relays and TF lookup;
the release binding's default expects normalized inputs. The opt-in driver profile
replaces that adaptation for the recorded compressed/static-TF layout only.
Do not assume that identity alignment survives a reboot or a different map. The
bag need not contain every relay topic, and the sample does not establish current
robot configuration. Inspect another closed bag without ROS using the optional
`rosbags` dependency:

```bash
python scripts/inspect_hardware_bag.py \
  --db3 /path/to/closed_episode_0.db3 --out /new/frame-sample.json
```

Compose `BatteryGuardBridge` around `io.navigation`, then give its fenced
`battery.planner` to `BTControlsBridge`. Pass that bridge's `controller` and the
battery bridge into `HardwareROSEpisode`, alongside the application services and
an explicit `HardwareProfile`. All motion paths must use the same fenced planner.
The hardware module rejects inconsistent adapter identities. The full concrete
synthetic integration is in `scripts/qualify_hardware_wiring.py`; it is not a
live-robot launch script.

Use the profile's checked `LAM_NAV_CONFIRM`, `LAM_CAPTURE_LOCAL_INSPECTION` and
`LAM_FARM_EXPLORATION_FALLBACK` values. Enable battery/press settings deliberately
in the application, including a stable absolute hold-file path. Do not reuse the
offline `/release/test/cmd_vel` or fake Sit remaps on hardware.

`claim_complete` remains the actor-facing terminal tool. The host carries out
acknowledged submission; there is no second actor-visible `submit` skill.
Close and recreate the episode for each task. Persistent motion holds are not
task state and must survive episode reset and application restart.

## One day acceptance plan

This is an ordered plan, not a promise of completion in one day. Do the dependency
installation and application wiring before reserving that day if possible. Stop
at the first failed prerequisite; a partial day can qualify a narrower subset.

In particular, confirm the selected input profile, actual TF layout,
FARM/planner adapters and loaded release manifest before describing the candidate
as ready to drive. The offline suite does not supply a live application launcher.

| Stage | Evidence required to move on |
|---|---|
| Stationary bringup | Candidate imports, loaded manifest, correct image orientation and map pose, endpoint checks, sufficient battery/disk, recording verified; no automatic lease claim or GO |
| No-motion task | Live survey/capture/report and terminal delivery, with motion inhibited; task reset creates a fresh episode and retains the hold |
| Short supervised navigation | Correct displayed goal, explicit operator approval, arrival and cancel-to-stop behavior; no competing velocity publisher |
| Skill interruption and timeout | Inject only into the active execution key; commanded velocity stops, stale requests fail, agent remains able to retry or report; record command and measured motion separately |
| Simulated battery condition | Use the guarded test override, not a physically depleted battery; admission block, mid-skill hold and report; posture only after ownership/stationary checks with operator clearance |
| One inspection mission and repeat | Query → navigate → survey → capture → report → terminal, then a fresh task without stale goals or duplicated output |
| Press, if needed | Separate clearance, verified target/viewpoint, confirmation and receiver-side precision behavior; otherwise leave press unqualified |

For basic live support, retain passing evidence for the intended skills and
interruption path on the exact candidate. Claim battery/posture or press support
only if their additional checks pass. Record genuine perception/task failures
separately from launch/transport failures; do not replace them with the synthetic
suite's successful fixtures.

## Stop and rollback

The operator's independent stop takes priority. Stop owned agent activity while
preserving the brake and persistent motion hold, then verify that no competing
command publisher remains before changing installations. Do not clear latch files,
stand, reclaim a lease or release GO automatically during rollback. Return to the
recorded working installation only under operator control.

Retain manifests, logs, inference images, ROS bags, explicit operator interventions
and failed attempts. A live support statement should name the robot/compute stack,
candidate versions and tested skills, not simply say that the package is safe.
