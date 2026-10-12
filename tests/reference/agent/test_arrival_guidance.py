"""Tests for the deterministic arrival-status → guidance pre-router.

Covers docs/spec/bt-nav-fixes.md Checkpoint 2: a code (not LLM) lookup that
maps typed navigate failure/short-arrival codes to a one-sentence guidance
string, appended to (never replacing) a skill's tool-result observation.
"""

from __future__ import annotations

from bt_agent._engine.agent.lam_agent.actor import route_arrival_guidance


def test_route_arrival_guidance_covers_all_five_codes_with_distinct_strings() -> None:
    codes = ["failed_no_path", "failed_blocked", "failed_no_movement", "arrived_short", "failed_invalid"]
    guidance = {code: route_arrival_guidance(code) for code in codes}

    for code, text in guidance.items():
        assert text, f"expected non-empty guidance for {code!r}"

    # Distinct strings — no generic catch-all reused across codes.
    assert len(set(guidance.values())) == len(codes)


def test_route_arrival_guidance_failed_invalid_uses_the_ruled_sentence() -> None:
    """``failed_invalid`` is written by ``navigate._ValidateGoal`` on non-finite
    goal coordinates but was never routed (skills_contract_v0 D8 gap). The
    sentence below is the ruled wording — it tells the agent the goal itself
    was bad, not the route, so re-QUERY is the recovery, not a retry."""
    assert (
        route_arrival_guidance("failed_invalid")
        == "The goal position for this target was invalid — query the target again or pick a different viewpoint."
    )


def test_route_arrival_guidance_is_one_sentence() -> None:
    for code in ["failed_no_path", "failed_blocked", "failed_no_movement", "arrived_short", "failed_invalid"]:
        text = route_arrival_guidance(code)
        # One sentence: a single terminal period, not in the middle.
        assert text.count(".") == 1
        assert text.endswith(".")


def test_route_arrival_guidance_unmapped_code_returns_empty() -> None:
    assert route_arrival_guidance("arrived") == ""
    assert route_arrival_guidance("already_at_goal") == ""
    assert route_arrival_guidance("") == ""
    assert route_arrival_guidance(None) == ""
