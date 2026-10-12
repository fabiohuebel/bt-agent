# Source provenance

BT-Agent accompanies the paper *BT-Agent: Deploying Onboard Robot Agents with
Behavior Trees and Multi-Fidelity Evaluation* (Hübel, Lilja, Yin, He, Tomlin,
Sastry, Frey and Pavone, 2026). This work was supported in part by the DARPA
TIAMAT program.

The extracted controller originates from the project's `tiamat_agent` research
repository, commit `62056af6fcb9ac3ec611760415c0c4f2f91a1fb6`. Its package
metadata names Fabio Hübel and Jonas Frey as authors. Original source and
docstrings are retained apart from recorded departures. The mechanical module-prefix
relocation is recorded file by file in `reproduction/core-source.json`, and approved
departures are recorded in `reproduction/core-patches.json`. They include the
removal of the development prompt variants and unused agent classes; the released
prompt set is flattened into one module with unchanged bytes. The research repository is not public, so
the parity scripts that read it cannot run from this repository alone.

Project-authored software and associated documentation are licensed under
Apache-2.0; see LICENSE and LICENSE-SCOPE.md. Existing third-party notices are
retained. Benchmark content, research data, assets, model weights and recordings
are not relicensed by this code grant.

One robotics test fixture records the scene-graph objects, viewpoints and camera
transforms of one simulated episode in a scene supplied by an independent
evaluation team. The task suite and scene data are not part of this repository.
