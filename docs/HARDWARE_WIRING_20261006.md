# Offline hardware wiring qualification

Robotics `0.1.0.dev6` completes the offline hardware-integration gate. It adds
explicit ROS sensor/publication wiring and the source-pinned motion mux. All
checks use synthetic telemetry, scripted model/vision responses and a fake Sit
service inside network-isolated containers. **No robot was contacted and no
physical stopping, posture, perception accuracy or task success is claimed.**

This document records the 6 October checkpoint. For the current offline command
and prospective live acceptance plan, use [hardware preparation](HARDWARE_DEPLOYMENT.md).

## Installed binding

`RosHardwareIO` takes an existing ROS node, an epoch, a new evidence directory,
and explicit pose/body/camera frame names. It subscribes to five camera streams,
odometry and direct body-to-camera static transforms. It uses the preserved
camera orientation and inverse-transform convention. It neither estimates map
alignment nor assumes that odometry coordinates match an arbitrary navgraph.

The current dev11 binding requires `pose_frame="map"`, matching the preserved
waypoint publisher, and a distinct body frame. The application must transform
odometry and align its graph to this frame; changing a frame label is not a transform.

The new input boundary rejects stale/future/repeated source stamps, wrong frame
names, malformed images and invalid odometry/rotations. Unobserved pose is not
treated as the origin. Invalid/stale inputs are withheld from the task state;
the navigation proxy reports invalid pose as stale. These are explicit release
input checks, not changes to the frozen simulator skills or controller prompts.

`HardwareROSEpisode` binds the hardware-profile agent to one episode's
`EpisodeHost` and `PhysicalPublication`. Captures and reports are persisted and
published on `/task_submission`; acknowledged completion publishes `task complete`
on the deployed hardware topic `/sim_control`, **not** Habitat's `/control`.
Capture files contain the exact image bytes used by the skill and are hash checked.
Failed persistence/publication is not accepted as successful terminal delivery.
A pending terminal publication can retry without ticking a completed agent again.

This remains a local acknowledgment boundary: durable local evidence plus a
successful ROS publish. It does not provide distributed exactly-once delivery,
prove a remote consumer persisted the message, or independently grade success.
Close and recreate the binding for each task; external persistent motion holds
survive task/session reset. The caller owns its ROS executor and driver lifecycle.

## Typed telemetry and independent mux

The test workspace compiles the exact eight battery/foot/lease message definitions
from Spot ROS commit `818f368806938fe215b1a03389d44bf67be62587`, the deployed
agent's pinned submodule. The publisher/subscriber path uses generated ROS types,
not String shims or manually invoked sensor callbacks.

The subset is **test-only**: it is not the full Spot interface package and must
never replace one on a robot. No driver, SDK, robot connection or lease claimant
is included. `prepare_spot_test_interfaces.py` records the source hashes and
license in a fresh workspace.

The preserved mux runs in a separate process and publishes only to
`/release/test/cmd_vel`. Tests inject nonzero intentions there to exercise gating,
but no publisher exists on either physical actuator-output topic. They verify
braked startup, motion fencing, output suspension before posture change,
publisher-count/owner checks and persistent suspension after process restart.
Clearing the coordinator file or simulated reading does not clear the mux latch.

| Scenario | Verified behavior |
|---|---|
| Low battery admission | 9% prevents the motion body from starting |
| Critical battery | 5% interrupts the body; stop and mux suspension precede one fake Sit |
| Stale battery feed | Motion held; no Sit inferred from missing data |
| Unknown battery status | Motion held; zero-percent unknown is not treated as valid critical telemetry |
| Lease denied | Motion held; no Sit |
| Duplicate output publisher | Ownership proof refused; no Sit |
| Sit denied | One request, held error, no retry |
| Sit unavailable | Held error, no request sent |

Seven independent ROS scenarios cover those transitions. In every scenario,
the real task controller then reports the constraint and submits its report
while the motion hold remains active. The model is scripted; this is lifecycle
qualification, not learned recovery evidence. A separate fixture completes two
query/navigation-at-current-pose/survey/capture/report/submit episodes with fresh
per-episode output state.

The press-precision sender publishes the expected Bool heartbeat and release
with a 250 ms ROS lifespan. The patched follower's 500 ms lease/control behavior
and physical approach accuracy are **not** revalidated by that wire test.
Receiver-side sensor freshness also has limits: the preserved battery/lease/feet
bridge uses receive time; the new pose/camera boundary checks source timestamps.
Neither mechanism authenticates a publisher or proves calibration.

## Checks and reproduction

- 531 robotics CPU tests pass, including the new mux and I/O failure tests.
- 423 core tests pass; the 22 existing core-only physical skips are unchanged.
- The seven typed ROS scenarios and two inspection episodes pass on the installed
  dev6 wheel with the public-base ROS image from the simulator release.
- The default skill/source contracts, the 19 prompt bundles present at this
  checkpoint and archived hardware comparison were separately checkable; no paper
  results are replaced. The `0.1.0rc1` release ships only the released prompt
  bundle, and the reference check now compares that bundle.

Build the test interfaces, using a new workspace:

```bash
python scripts/prepare_spot_test_interfaces.py \
  --spot-repo /path/to/spot_ros2 --out /new/test-workspace
```

Map that workspace at `/work` in the qualified ROS image and build `spot_msgs`
with `colcon build --packages-select spot_msgs --executor sequential
--cmake-args -DBUILD_TESTING=OFF`. No GPU or network is required for compilation.
Keep the installed Python environment at its original absolute path when mounted.

Run these commands **inside Docker with `--network none`, domain 97 and
`ROS_LOCALHOST_ONLY=1`**, after sourcing `/work/install/setup.bash`:

```bash
python scripts/qualify_hardware_wiring.py --case critical --out /new/critical-check
python scripts/qualify_hardware_episode.py --out /new/episode-check
```

Other cases are `lease-denied`, `stale-battery`, `invalid-battery`,
`duplicate-output`, `sit-denied` and `sit-unavailable`. The scripts refuse
non-isolated execution and existing output directories. The parent should impose
a timeout and stop only its owned test container after failure.

Evidence: `artifacts/hardware-wiring-20261006/`. Final scenario folders end in
`-final`; `episode-final` contains the two inspection fixtures. The first critical
attempt stopped at a startup assertion before an output sample arrived; the
probe now waits for observed output. That attempt remains preserved, not counted.

FARM/model hosting, actual path planning/CMU control, maps/fiducial calibration,
robot power/lease handling and dashboard operation remain application inputs.
This work establishes the software boundary under controlled offline inputs,
not whole-stack deployment equivalence or certified safety. Future physical
validation, if pursued, must be separately recorded without relabeling these tests.
