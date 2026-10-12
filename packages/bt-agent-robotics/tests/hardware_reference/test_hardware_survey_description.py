from copy import deepcopy
import pytest
from bt_agent_robotics.skills.surround_survey import SurroundSurveySkill
from bt_agent_robotics._hardware.hardware_survey_description import VisualQuestionSurveySkill, make_survey

def strip_descriptions(value):
    if isinstance(value, dict):
        return {key: strip_descriptions(item) for key, item in value.items() if key != "description"}
    if isinstance(value, list):
        return [strip_descriptions(item) for item in value]
    return value

def test_original_default_and_experiment_changes_only_help():
    original = make_survey(None)
    before = deepcopy(original.tool_def())
    experimental = make_survey(None, "visual_question_v1")
    assert type(original) is SurroundSurveySkill
    assert type(experimental) is VisualQuestionSurveySkill
    assert experimental.build_subtree.__func__ is original.build_subtree.__func__
    assert strip_descriptions(before) == strip_descriptions(experimental.tool_def())
    assert make_survey(None).tool_def() == before
    text = str(experimental.tool_def()).lower()
    for forbidden in ("london", "clock", "safe door", "example"):
        assert forbidden not in text
    with pytest.raises(ValueError):
        make_survey(None, "unknown")
