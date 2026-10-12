"""Fix 4 regression: the "do not demand evidence" bullet of the released
evaluator (originally the p7-leniency evaluator) must describe what the text
substrate actually returns as of the 2026-08-27 tiamat_world change (room
banners + doorway adjacency removed from observations). Room names still
surface, but only inside query recall metadata."""

from __future__ import annotations

from bt_agent._engine.agent.system_prompts import get_prompts


def test_p7_leniency_evaluator_no_longer_claims_room_adjacency_banners() -> None:
    evaluator = get_prompts("tiamat_agent_submit_v5").evaluator
    assert "room/adjacency" not in evaluator


def test_p7_leniency_evaluator_names_current_substrate_outputs() -> None:
    evaluator = get_prompts("tiamat_agent_submit_v5").evaluator
    assert "object handles" in evaluator
    assert "query recall metadata" in evaluator
    assert "skill outcome strings" in evaluator
    assert "camera framing" in evaluator
    # The nuance survives: room names still appear, just inside query results.
    assert "last-seen room" in evaluator or "last seen" in evaluator


def test_p7_leniency_evaluator_new_bullet_has_no_em_dash() -> None:
    evaluator = get_prompts("tiamat_agent_submit_v5").evaluator
    marker = "Do not demand evidence the environment cannot produce."
    idx = evaluator.index(marker)
    bullet_end = evaluator.index("\n- ", idx)
    bullet = evaluator[idx:bullet_end]
    assert "—" not in bullet
