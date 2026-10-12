"""CP3-desk replay harness for the profiles v1 Actor-text seam.

No recorded robot episodes are checked into this tree, so the offline gate uses
a synthetic fixture with the same post-ExecuteSkill fields. The harness accepts
recorded fixtures in this shape when they become available.
"""

from __future__ import annotations

import copy
import json
from pathlib import Path

from bt_agent._engine.agent.lam_agent.actor import build_history_entry, history_call_id
from bt_agent._engine.bt.nodes.select_skill import actor_history_observation
from bt_agent._engine.skills.protocol import SkillHandle, SkillStatus

_FIXTURE = Path(__file__).resolve().parents[1] / "fixtures" / "profiles" / "synthetic_episode.json"


def replay_actor_transcript(episode: dict, failure_presentation: str) -> list[list[dict]]:
    """Replay post-execution episode steps through Actor history ingestion."""
    registry = {name: object() for name in episode["registered_skills"]}
    transcript = []
    for step in episode["steps"]:
        handle = SkillHandle()
        handle.status = SkillStatus[step["status"]]
        handle.observation = step["observation"]
        handle.result_data = copy.deepcopy(step["result_data"])
        observation = actor_history_observation(
            step["skill"],
            handle,
            registry,
            failure_presentation,
        )
        transcript.append(
            build_history_entry(
                step["skill"],
                step["params"],
                observation,
                call_id=history_call_id(step["skill"], step["observation"]),
                reasoning=step["reasoning"],
            )
        )
    return transcript


def test_replay_diff_is_only_registered_failed_skill_observation() -> None:
    episode = json.loads(_FIXTURE.read_text())
    original_episode = copy.deepcopy(episode)

    typed = replay_actor_transcript(episode, "typed")
    flat = replay_actor_transcript(episode, "flat")

    expected_diff_indices = {
        index
        for index, step in enumerate(episode["steps"])
        if step["status"] == "FAILED" and step["skill"] in episode["registered_skills"]
    }
    actual_diff_indices = {index for index, pair in enumerate(zip(typed, flat)) if pair[0] != pair[1]}
    assert actual_diff_indices == expected_diff_indices == {1}

    for index, (typed_entry, flat_entry) in enumerate(zip(typed, flat)):
        if index in expected_diff_indices:
            assert typed_entry[0] == flat_entry[0]
            assert typed_entry[1]["tool_call_id"] == flat_entry[1]["tool_call_id"]
            assert typed_entry[1]["content"] == episode["steps"][index]["observation"]
            assert flat_entry[1]["content"] == "failure of skill navigate"
        else:
            assert typed_entry == flat_entry

    assert episode == original_episode
