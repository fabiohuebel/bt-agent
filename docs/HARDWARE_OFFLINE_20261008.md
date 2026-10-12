# Hardware offline checks on 8 October 2026

Core `0.1.0.dev7` and robotics `0.1.0.dev11` pass the current offline suite in a
fresh Python 3.10 installation. The robotics wheel was built from its sdist.
No robot, simulator, model endpoint or GPU was used. These results support the
[deployment preparation plan](HARDWARE_DEPLOYMENT.md), not a claim of live validation.

## Results

| Check | Result |
|---|---|
| Core and offline-tool CPU tests | 483 passed; 22 existing physical/core-only skips |
| Robotics CPU tests | 542 passed, including ten new frame-configuration cases |
| Isolated ROS suite | All nine cases passed |
| Full synthetic inspection path | Two episodes with capture, report and acknowledged terminal delivery |
| Package dependencies | `pip check` passed in the fresh environment |
| Archived deployment policy | Six tool schemas, actor prompt and 43 battery decision checkpoints match |

The ROS suite exercises real generated telemetry messages and the preserved mux,
but scripted model/vision responses and a fake Sit service. It includes persistent
hold across mux restart, constraint reporting while held, rejected/unavailable
Sit, denied lease, duplicate output publisher, invalid/stale battery telemetry
and scoped interrupt/timeout/failure requests. Each container has no network or
device access and is removed after its bounded check.

Robotics dev11 adds a fail-fast check to the hardware ROS adapter: pose input must
use `map`, matching the frozen waypoint publisher, and body frame must be explicit
and distinct. Other names previously passed construction despite this mismatch.
The new tests verify rejection before ROS imports or evidence creation. The frozen
skills, controller, prompts, budgets and simulator profiles were not changed.

## Evidence and limitations

Local evidence is `artifacts/hardware-release-20261008-final/`, with the suite
summary, package lists, per-case results, guard journals, synthetic capture images
and container logs. CPU receipts are `artifacts/hardware-release-20261008-core.xml`
and `artifacts/hardware-release-20261008-robotics.xml`.

The first runner attempt failed before the tests could create their output
directories: dropping container capabilities exposed a bind-mount ownership
mismatch. The runner now uses the invoking UID/GID. That failed attempt is retained
under `artifacts/hardware-release-20261008/`; it is not counted as a passing trial.
The intermediate `-r2` run passed on dev10. The `-final` run uses dev11 and is the
current evidence; no earlier installation was overwritten.

The optional bag inspector sampled a closed Stanford control recording without
ROS replay. It found `vision`-frame odometry, driver camera topics, fisheye frame
names and chained static transforms; that recording's `vision → map` transform
was identity. The application must retain or replace the old camera relays/TF
lookup before connecting these inputs to the normalized release interface.
The result is recorded in `archive-frames-complete.json`. Archived bag timestamps
are not a live clock/freshness test, and omitted relay topics do not establish
that those relays were absent during recording.

ARM64 installation, live sensor/clock/frame conventions, FARM/path-planner
integration, calibration, driver/lease behavior, press receiver behavior and
physical stop/posture still require the target robot. The release remains
offline-qualified until that candidate's live acceptance checks are recorded.
