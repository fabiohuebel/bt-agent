# BT-Agent

BT-Agent runs a language-model agent on a robot. The model chooses the next
skill. Each skill is a behavior tree. Skills that act on the robot check their
own conditions while they run, and a failed skill reports what went wrong. This
repository is the code
release for the paper *BT-Agent: Deploying Onboard Robot Agents with Behavior
Trees and Multi-Fidelity Evaluation* (Hübel et al., 2026).

[Paper (arXiv:2610.13862)](https://arxiv.org/abs/2610.13862) |
[Project page](https://goldengait.github.io/bt-agent/) |
[Release notes](CHANGELOG.md)

> **Research software.** Robot use is experimental. It needs your own robot
> drivers and services, an independent emergency stop held by an operator and
> supervised acceptance testing. No robot test has been run for this release
> (`0.1.0rc1`).

## What is in this repository

- **`bt-agent`**: the core agent runtime. Pure Python, no ROS or GPU needed.
- **`bt-agent-robotics`** (optional): query, navigate, survey, capture and press
  skills, a simulation-only push skill, a graph planner and an opt-in hardware
  profile, tested with fake backends. See its [README](packages/bt-agent-robotics/README.md).

The companion benchmark with the text and Habitat tasks, `robot-benchmark`, is
not public yet. Its release is waiting for permission to redistribute the task
suite and scene data.

## Try it on a CPU

```bash
python -m venv .venv
.venv/bin/python -m pip install .
.venv/bin/python examples/cpu_demo.py
```

The example needs no GPU, ROS, simulator, model server or API key. A scripted
model has the agent inspect a synthetic shelf, report what it found and submit.

## Use your own model and skills

```python
from bt_agent import create_agent
from bt_agent.episode import EpisodeHost
from bt_agent.models import ChatModel
from bt_agent.skills import ReportSkill

# An OpenAI-compatible endpoint with tool calling, such as vLLM started
# with --enable-auto-tool-choice and a --tool-call-parser.
model = ChatModel(base_url="http://localhost:8000/v1", model="your-model")
host = EpisodeHost(publish_report=print, flush_artifacts=lambda: None)

with create_agent(model=model, skills=[*your_skills, ReportSkill()], host=host) as agent:
    outcome = agent.run("Find the red mug and report where it is.")
```

`outcome.submitted` only records that the agent submitted. Grade task success
separately. To write skills, see [Authoring BT skills](docs/ADDING_SKILLS.md);
`InspectShelf` in `bt_agent.testing` is a small worked example.

## Runtime profiles

A runtime profile picks which version of the agent controller runs.

| Profile | Use it for |
|---|---|
| `task-core-v1` (default) | New work. This is the maintained runtime. |
| `paper-core-v1` | Reproducing the paper. It is an unpolished research snapshot of the controller as it ran in the experiments and gets no new features. The hardware profile in `bt-agent-robotics` uses it. |

Both profiles use the same released prompts. Pass
`profile="paper-core-v1"` to `create_agent` to select the paper controller.

## Documentation

| Topic | Guide |
|---|---|
| Writing skills | [Authoring BT skills](docs/ADDING_SKILLS.md) |
| Interfaces, recording and all known limitations | [Runtime reference](docs/RUNTIME_REFERENCE.md) |
| Running on a robot | [Hardware preparation and deployment day](docs/HARDWARE_DEPLOYMENT.md) |
| Reproducing the paper controller | [Runtime profiles](docs/TASK_STATE_MIGRATION.md) and [source mapping](docs/MIGRATION_MAP.md) |
| What this release covers and how it was checked | [Release scope](docs/RELEASE_READINESS.md) |

## Limitations to know first

- One runtime per process. Run parallel episodes in separate processes.
- `run()` stops an episode after `max_ticks` ticks. The defaults (10000 ticks,
  0.01 s apart) allow about 100 s, so pass a larger `max_ticks` for slow models.
- `run()` does not raise on model errors. A failing model endpoint ends the
  episode at the tick limit. Record calls with `bt_agent.recording.RecordedModel` to debug.
- A cancellation acknowledgement from the software says nothing about the
  robot's actual velocity or posture.

The [runtime reference](docs/RUNTIME_REFERENCE.md#known-limitations) has the full list.

## Tests

```bash
.venv/bin/python -m pip install '.[test]'
.venv/bin/python -m pytest
```

CI runs the core and robotics CPU suites on Python 3.10 and 3.13.

## License

Project code and documentation are licensed under [Apache-2.0](LICENSE).
Third-party code keeps its own terms. This release grants no license for data
such as task text, scene data, recordings or model weights; see
[LICENSE-SCOPE.md](LICENSE-SCOPE.md) and [NOTICE.md](NOTICE.md).

## Citation

```bibtex
@misc{huebel2026btagent,
  title         = {{BT-Agent}: Deploying Onboard Robot Agents with Behavior
                   Trees and Multi-Fidelity Evaluation},
  author        = {H{\"u}bel, Fabio and Lilja, Adam and Yin, Yue and He, Siming
                   and Tomlin, Claire and Sastry, Shankar and Frey, Jonas and
                   Pavone, Marco},
  year          = {2026},
  eprint        = {2610.13862},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
}
```

This work was supported in part by the DARPA TIAMAT program.
