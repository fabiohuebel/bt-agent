import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize("core_reference", [True, False])
def test_core_only_skips_do_not_hide_optional_robotics_coverage(core_reference):
    tests = Path(__file__).resolve().parent
    spec = importlib.util.spec_from_file_location("collection_scope_fixture", tests / "conftest.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    path = (
        tests / "reference/skills/test_publish_results.py"
        if core_reference
        else tests.parent
        / "packages/bt-agent-robotics/tests/reference/core/test_publish_results.py"
    )
    markers = []
    item = SimpleNamespace(
        path=path,
        name="test_query_skill_overrides_publish_results",
        add_marker=markers.append,
    )
    module.pytest_collection_modifyitems([item])
    assert bool(markers) is core_reference
