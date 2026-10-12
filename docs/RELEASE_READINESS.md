# Release scope

This page states what the `0.1.0rc1` release contains, how it was checked and
what it does not claim.

## Contents

- `bt-agent`: the core runtime, public skill and episode interfaces, model
  client, recording and tracing. The internal `_engine` package is the frozen
  paper controller, relocated by module prefix, with the recorded departures in
  `reproduction/core-patches.json`. The default `task-core-v1` profile is the
  maintained runtime. The `paper-core-v1` profile is an unpolished research
  snapshot of the paper controller, kept for reproduction and the hardware binding.
  Both use the single released prompt set.
- `bt-agent-robotics`: optional query, navigate, survey, capture, press and push
  skills, a graph planner, platform adapters and an opt-in hardware profile.
- Documentation, the CPU example, tests and source-provenance records.

The companion text and Habitat benchmark, `robot-benchmark`, is not yet public; its release is pending redistribution permission for the task suite and scene data. Model weights, simulator assets, robot recordings and deployment
configuration are not included.

## Verification

- CI installs both packages on Python 3.10 and 3.13 and runs their CPU suites.
  It starts no robot, model server or simulator.
- Release wheels are built from source archives and tested in fresh environments.
- Against the original research controller, the release reproduces complete
  serialized model requests, the released prompt bundle and report and submission
  behavior. This comparison reads the original research repository, which is
  not public. `scripts/check_task_state_parity.py` compares the default task
  runtime with the paper controller; its core cases run from this repository,
  and its benchmark cases need the companion benchmark.
- The hardware profile passes an offline suite with synthetic telemetry, fake
  services and archived sensor intake. See [hardware preparation](HARDWARE_DEPLOYMENT.md).
- `scripts/audit_release.py` scans tracked files and archives for credential
  patterns and found none. A heuristic scan cannot prove that no secret exists.

Dated records, such as [the extraction qualification](QUALIFICATION.md) and the
hardware check pages, cite receipt files under `artifacts/`. Those receipts are
not part of this repository.

## Not claimed

- Live robot validation of this release. Robot use needs platform services, an
  independent operator stop and supervised acceptance testing.
- Model performance. Scripted fixtures test wiring and do not measure task success.
- Physical stopping. A cancellation acknowledgement is not a measurement of
  robot velocity or posture.

## Data and third-party material

Apache-2.0 covers project-authored code and documentation; see
[LICENSE-SCOPE.md](../LICENSE-SCOPE.md). It grants no license for task text, scene
data or recordings. One robotics test fixture records the scene-graph objects,
viewpoints and camera transforms of one simulated episode in a scene supplied by
an independent evaluation team. The task suite and scene data are not part of
this repository.

## Source history

This repository starts from a single release commit. The extraction records under
`reproduction/` keep the original source commits and file hashes. One recorded
fixture keeps a provenance field that names a path on the original workstation,
and frozen docstrings keep some original development notes.

The `Private :: Do Not Upload` classifier stops an accidental PyPI upload. Install
from this repository.
