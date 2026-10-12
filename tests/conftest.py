"""Documented qualifications of imported historical tests, never hidden skips."""

from pathlib import Path

import pytest

PHYSICAL = {
    "test_non_promoting_builtins_inherit_default_noop",
    "test_builtin_skills_have_expected_effect_types",
    "test_get_action_skills_masked_excludes_when_precondition_fails",
    "test_get_action_skills_masked_includes_when_both_pass",
    "test_navigate_precondition_blocks_when_object_store_empty",
    "test_navigate_precondition_passes_with_objects",
    "test_capture_precondition_blocks_when_no_camera_images",
    "test_capture_precondition_passes_with_full_state",
    "test_query_skill_overrides_publish_results",
    "test_capture_skill_overrides_publish_results",
    "test_query_promote_writes_new_objects_and_query_mappings",
    "test_capture_promote_writes_new_objects",
    "test_publish_results_handles_empty_result_data",
    "test_publish_results_handles_missing_fields",
    "test_navigate_skill_overrides_safeguard_invariant",
    "test_capture_skill_overrides_safeguard_invariant",
    "test_navigate_safeguard_returns_false_on_missing_localization",
    "test_navigate_safeguard_returns_false_on_empty_planned_path",
    "test_navigate_safeguard_returns_true_pre_planning",
    "test_capture_safeguard_pre_analysis_is_vacuously_true",
    "test_capture_safeguard_returns_false_post_analysis_with_no_camera",
    "test_capture_safeguard_returns_true_post_analysis_with_camera_present",
}


def pytest_collection_modifyitems(items):
    reference = Path(__file__).resolve().parent / "reference"
    for item in items:
        if not Path(item.path).resolve().is_relative_to(reference):
            continue
        if item.name in PHYSICAL:
            item.add_marker(
                pytest.mark.skip(
                    reason="Physical skill integration runs in packages/bt-agent-robotics/tests"
                )
            )
