"""Opt-in hardware experiment: tool help only, not perception implementation.

The original SurroundSurveySkill and its model prompts stay unchanged.
Never select this variant implicitly or from a task's wording.
"""

from bt_agent_robotics.skills.surround_survey import SurroundSurveySkill

VARIANTS = ("original", "visual_question_v1")
ENV_KEY = "LAM_HARDWARE_SURVEY_DESCRIPTION"


class VisualQuestionSurveySkill(SurroundSurveySkill):
    description = (
        "Ask one visual question about the robot's current camera views. "
        "Use this to locate objects, inspect their state, or read visible text and values."
    )
    usage = (
        "Pass the specific question needed to complete the task, rather than only an object description. "
        "The skill checks the current camera images and returns the selected camera with a text observation. "
        "For reading questions, request the actual text or value and ask for unreadable or uncertain evidence "
        "to be stated explicitly. This skill does not move the robot. Use report to record a finding."
    )
    params = {
        "target_description": (
            "One specific question answerable from the current camera images. "
            "Specify the information or condition to inspect. For readings, request the actual text or value "
            "and an explicit statement if it cannot be read reliably."
        ),
    }


def make_survey(model, variant="original"):
    if variant not in VARIANTS:
        raise ValueError(f"Unknown hardware survey description: {variant!r}")
    return (SurroundSurveySkill if variant == "original" else VisualQuestionSurveySkill)(model)
