"""AgentConfig.ablation: CP1 config + validation (docs/spec/tiamat-ablation-flags.md).

CP1 only parses/validates/stamps the ablation set into AgentConfig; the tree
does not branch on it yet (CP2/CP3). These tests pin that contract: absent or
empty is today's agent (byte-identical default), unknown flags fail fast with
a clear error, and valid flags normalize into a frozenset regardless of the
input container type.
"""

from __future__ import annotations

import pytest
from bt_agent._engine.agent.config import ABLATION_CHOICES, AgentConfig, validate_ablation


def test_ablation_choices_are_the_four_design_decision_axes() -> None:
    assert ABLATION_CHOICES == frozenset({"evaluator", "replan", "context", "planner"})


class TestValidateAblation:
    def test_none_is_empty(self) -> None:
        assert validate_ablation(None) == frozenset()

    def test_empty_list_is_empty(self) -> None:
        assert validate_ablation([]) == frozenset()

    def test_valid_values_normalize_to_frozenset(self) -> None:
        assert validate_ablation(["evaluator", "planner"]) == frozenset({"evaluator", "planner"})
        assert validate_ablation(("replan",)) == frozenset({"replan"})
        assert validate_ablation({"context"}) == frozenset({"context"})

    def test_unknown_flag_rejected_fast_with_clear_error(self) -> None:
        with pytest.raises(ValueError, match="unknown ablation flag"):
            validate_ablation(["evaluator", "bogus"])

    def test_non_string_entry_rejected(self) -> None:
        with pytest.raises(ValueError, match="ablation entries must be strings"):
            validate_ablation([1])

    def test_non_iterable_rejected(self) -> None:
        with pytest.raises(ValueError, match="ablation must be a list"):
            validate_ablation("evaluator")


class TestAgentConfigAblation:
    def test_default_is_empty_byte_identical(self) -> None:
        assert AgentConfig().ablation == frozenset()

    def test_valid_ablation_threads_through(self) -> None:
        config = AgentConfig(ablation=["evaluator", "planner"])
        assert config.ablation == frozenset({"evaluator", "planner"})

    def test_unknown_ablation_fails_fast_at_construction(self) -> None:
        with pytest.raises(ValueError, match="unknown ablation flag"):
            AgentConfig(ablation=["not-a-real-switch"])

    def test_from_dict_absent_is_empty(self) -> None:
        assert AgentConfig.from_dict({}).ablation == frozenset()

    def test_from_dict_parses_ablation(self) -> None:
        config = AgentConfig.from_dict({"ablation": ["replan", "context"]})
        assert config.ablation == frozenset({"replan", "context"})

    def test_from_dict_unknown_ablation_fails_fast(self) -> None:
        with pytest.raises(ValueError, match="unknown ablation flag"):
            AgentConfig.from_dict({"ablation": ["bogus"]})
