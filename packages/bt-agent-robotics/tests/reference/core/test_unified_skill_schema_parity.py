"""Physics-side parity test (unified-skill-api CP2, docs/spec/unified-skill-api.md).

Pins ``Skill.tool_def()`` for the 5 shared skills (navigate, capture, query,
surround_survey, press). Physics names are canonical — this test locks in
the current rendering so a future rename/required-flip is caught, and it
proves ``query`` has gained the optional ``exclude`` array (CP2's sole
behavior change; see ``lam_v2_tiamat/bt_skills/query.py`` and
``lam_v2_tiamat/tests/bt_skills/test_query_exclude.py``) and that
``navigate`` renders ``viewpoint_index`` as an optional integer per the
spec's canonical table (see ``lam_v2_tiamat/bt_skills/navigate.py``).
"""

from __future__ import annotations

from typing import Any, Dict, Set
from unittest.mock import MagicMock

from bt_agent_robotics.skills.capture import CaptureSkill
from bt_agent_robotics.skills.navigate import NavigateSkill
from bt_agent_robotics.skills.press import PressSkill
from bt_agent_robotics.skills.query import QuerySkill
from bt_agent_robotics.skills.surround_survey import SurroundSurveySkill


def _properties(tool_def: Dict[str, Any]) -> Dict[str, Any]:
    return tool_def["function"]["parameters"]["properties"]


def _required(tool_def: Dict[str, Any]) -> Set[str]:
    return set(tool_def["function"]["parameters"]["required"])


def test_navigate_schema() -> None:
    skill = NavigateSkill(local_planner=None, path_planner=None)
    tool_def = skill.tool_def()

    assert tool_def["function"]["name"] == "navigate"
    props = _properties(tool_def)
    assert props["target_object"]["type"] == "string"
    assert props["viewpoint_index"]["type"] == "integer"
    assert props["viewpoint_index"]["minimum"] == 0
    assert _required(tool_def) == {"target_object"}


def test_capture_schema() -> None:
    skill = CaptureSkill(model=MagicMock(), pointing_pipeline=None)
    tool_def = skill.tool_def()

    assert tool_def["function"]["name"] == "capture"
    props = _properties(tool_def)
    assert props["target_object"]["type"] == "string"
    assert props["camera_view"]["type"] == "string"
    assert _required(tool_def) == {"target_object", "camera_view"}


def test_query_schema_has_target_description_required_and_exclude_optional() -> None:
    skill = QuerySkill(scene_graph_agent=MagicMock())
    tool_def = skill.tool_def()

    assert tool_def["function"]["name"] == "query"
    props = _properties(tool_def)
    assert props["target_description"]["type"] == "string"
    assert props["exclude"]["type"] == "array"
    assert props["exclude"]["items"] == {"type": "string"}
    assert _required(tool_def) == {"target_description"}


def test_surround_survey_schema() -> None:
    skill = SurroundSurveySkill(model=MagicMock())
    tool_def = skill.tool_def()

    assert tool_def["function"]["name"] == "surround_survey"
    props = _properties(tool_def)
    assert props["target_description"]["type"] == "string"
    assert _required(tool_def) == {"target_description"}


def test_press_schema() -> None:
    skill = PressSkill(local_planner=None, path_planner=None)
    tool_def = skill.tool_def()

    assert tool_def["function"]["name"] == "press"
    props = _properties(tool_def)
    assert props["target_object"]["type"] == "string"
    assert _required(tool_def) == {"target_object"}
