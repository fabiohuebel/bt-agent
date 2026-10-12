"""The release ships exactly the prompt bytes that the frozen source sent.

The frozen source assembled ``tiamat_agent_submit_v5`` through a chain of
development variants. The release flattens it into one module, so any edit
to a prompt string now changes what both runtime profiles send to the model.
These digests were computed from the assembled bundle at source commit
62056af6f; ``scripts/check_reference_parity.py`` checks the same bytes
against the frozen source itself.
"""

import dataclasses
import hashlib
from pathlib import Path
import re

from bt_agent._engine.agent.system_prompts import get_prompts, list_variants
from bt_agent.profiles import core_config

RELEASED = "tiamat_agent_submit_v5"

FROZEN_SHA256 = {
    "planner_gather": "9b99a89a460cf6a39a2e54b76a513fbb54ccbfc1e611acd44a2612e66951fa07",
    "planner_synthesis": "5290968d268da96ec4614d865be841a333956ca3743c3e4255afbc93765bdf2c",
    "planner_replan": "8c6ce6c2a2b02e26e7bca1c6e2c246380a45cf871a332998ba763299ca560fb0",
    "actor": "3d4b385f8f3b8039839e7cf1b3489622dc9393d37f3ea749169378d6af9718a1",
    "evaluator": "f14cc602c23c759b0149542614fe97c888169adc9117311f9835b1d9e5b77f79",
    "actor_compaction": "d52cf63a5a6b074aedcb23cc5af41c23e1a247245580038545482726fbb774a1",
    "evaluator_user": "165fa02baa2bea84792efbfdefcfc4d9b1a85ced16a6c59bf70fd9085f24427f",
    "evaluator_claim": "78ce1749bf024d614d42882371e8f59a11dd03a0db5df35018b9adf0fadcf808",
}


def test_both_profiles_use_the_only_shipped_variant():
    assert list_variants() == [RELEASED]
    for profile in ("paper-core-v1", "task-core-v1"):
        assert core_config(profile).prompt_variant == RELEASED


def test_released_prompt_bytes_match_the_frozen_bundle():
    bundle = get_prompts(RELEASED)
    fields = {f.name for f in dataclasses.fields(bundle)} - {"name"}
    assert fields == set(FROZEN_SHA256)
    for name, expected in FROZEN_SHA256.items():
        assert hashlib.sha256(getattr(bundle, name).encode()).hexdigest() == expected, name


def test_release_sources_name_no_challenge_task_ids():
    """The externally supplied challenge suite's task IDs stay out of the release.

    The removed development variants named them in their notes. Project-authored
    task IDs (for example ``final_t3_*``) are not matched.
    """
    root = Path(__file__).resolve().parents[1]
    task_id = re.compile(r"\b(?:final|rr)_(?:isaac_)?\d{2}")
    paths = [*root.glob("*.md")]
    for top in ("packages", "tests", "scripts", "docs", "examples"):
        paths += [p for p in (root / top).rglob("*") if p.is_file() and "__pycache__" not in p.parts]
    assert paths
    for path in paths:
        if "handoffs" in path.parts:
            continue
        try:
            text = path.read_text(encoding="utf-8")
        except UnicodeDecodeError:
            continue
        assert not task_id.search(text), path
