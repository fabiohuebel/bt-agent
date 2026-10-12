"""Optional physical skill library; no robot connection at import time."""

from pathlib import Path

from bt_agent.skills import ReportSkill, register_doc_root

from .skills.capture import CaptureSkill
from .skills.navigate import NavigateSkill
from .skills.press import PressSkill
from .skills.query import QuerySkill
from .skills.surround_survey import SurroundSurveySkill


def inspection_skills(
    *,
    navigation,
    path_planning,
    scene_memory,
    perception,
    pointing=None,
    register_docs=True,
    capture_debug_writer=None,
    survey_debug_writer=None,
):
    """Build the reference inspection set without GT perception or ROS startup.

    Adapters supply the actual services. Preserve the reference registry order,
    since tool order is part of the model-visible interface.
    """
    if register_docs:
        register_doc_root(Path(__file__).parent / "skills" / "docs")
    return [
        QuerySkill(scene_memory),
        NavigateSkill(navigation, path_planning),
        PressSkill(navigation, path_planning),
        ReportSkill(),
        SurroundSurveySkill(perception, debug_writer=survey_debug_writer),
        CaptureSkill(perception, pointing, debug_writer=capture_debug_writer),
    ]


def interaction_skills(*, measured_push=False, **services):
    """Later physical profile: refreshed query geometry and optional measured push.

    Capture27 defaults remain in inspection_skills. The adapter must supply
    episode-bound native feedback; model prose is never completion evidence.
    """
    from ._interaction_query import QuerySkill as RefreshingQuerySkill

    skills = inspection_skills(**services)
    skills[0] = RefreshingQuerySkill(services["scene_memory"])
    if measured_push:
        from ._push.skill import PushSkill

        navigation = services["navigation"]
        if getattr(navigation, "push_telemetry", None) is None:
            raise ValueError("Measured pushing requires a controller-only telemetry adapter")
        skills.insert(3, PushSkill(navigation, services["path_planning"]))
    return skills


__all__ = [
    "CaptureSkill",
    "NavigateSkill",
    "PressSkill",
    "QuerySkill",
    "SurroundSurveySkill",
    "inspection_skills",
    "interaction_skills",
]
