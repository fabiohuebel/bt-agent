import json

import pytest

from bt_agent import create_agent
from bt_agent.episode import EpisodeHost
from bt_agent.testing import InspectShelf, ScriptedModel
from bt_agent.skills import EffectType


def test_internal_handler_is_dispatchable_but_never_advertised():
    class Wait(InspectShelf):
        name = "wait"

    model = ScriptedModel(
        [("wait", {}, "Fixture internal dispatch."), ("claim_complete", {}, "Finished.")]
    )
    with create_agent(
        model=model,
        skills=[],
        internal_skills=[Wait()],
        host=EpisodeHost(lambda _: None, lambda: None),
    ) as runtime:
        result = runtime.run("Check internal fallback.", tick_interval=0.001)
        assert result.submitted
        assert runtime.describe()["internal_handlers"] == ["wait"]
    assert all("wait" not in [t["function"]["name"] for t in r["tools"]] for r in model.requests)
    assert "shelf contains a red mug" in json.dumps(model.requests[-1])


def test_internal_handlers_cannot_shadow_advertised_or_terminal_skills():
    host = EpisodeHost(lambda _: None, lambda: None)
    with pytest.raises(ValueError, match="unique"):
        create_agent(
            model=object(), skills=[InspectShelf()], internal_skills=[InspectShelf()], host=host
        )

    class Submit(InspectShelf):
        name = "submit"

    with pytest.raises(ValueError, match="collides"):
        create_agent(model=object(), skills=[], internal_skills=[Submit()], host=host)
    with pytest.raises(TypeError, match="context skills"):
        create_agent(model=object(), skills=[], internal_skills=[object()], host=host)


def test_internal_context_class_cannot_override_its_effect_to_action():
    class DisguisedAction(InspectShelf):
        effect_type = EffectType.ACTION

    with pytest.raises(TypeError, match="context skills"):
        create_agent(
            model=object(),
            skills=[],
            internal_skills=[DisguisedAction()],
            host=EpisodeHost(lambda _: None, lambda: None),
        )
