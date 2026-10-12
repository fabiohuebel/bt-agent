"""Tests for the deterministic press-status → guidance pre-router.

Mirror of ``test_arrival_guidance.py`` for ``press_status`` — PressSkill's own
outcome key (skills_contract_v0 press section, D8). Shared failure codes must
resolve to the *same* sentence navigate's ``arrival_status`` router returns:
the agent should not learn two different recovery stories for one physical
failure just because a different skill reported it.
"""

from __future__ import annotations

from bt_agent._engine.agent.lam_agent.actor import (
    route_arrival_guidance,
    route_press_status_guidance,
)

_PRESS_FAILURE_CODES = ["failed_no_path", "failed_blocked", "failed_no_movement", "failed_invalid"]


def test_route_press_status_guidance_covers_every_press_failure_code() -> None:
    for code in _PRESS_FAILURE_CODES:
        assert route_press_status_guidance(code), f"expected non-empty guidance for {code!r}"


def test_press_guidance_is_byte_identical_to_arrival_guidance_for_shared_codes() -> None:
    """Same physical failure ⇒ same sentence, byte for byte."""
    for code in _PRESS_FAILURE_CODES:
        assert route_press_status_guidance(code) == route_arrival_guidance(code)


def test_route_press_status_guidance_is_one_sentence() -> None:
    for code in _PRESS_FAILURE_CODES:
        text = route_press_status_guidance(code)
        assert text.count(".") == 1
        assert text.endswith(".")


def test_route_press_status_guidance_success_and_unmapped_return_empty() -> None:
    assert route_press_status_guidance("pressed") == ""
    assert route_press_status_guidance("") == ""
    assert route_press_status_guidance(None) is not None
    assert route_press_status_guidance(None) == ""
