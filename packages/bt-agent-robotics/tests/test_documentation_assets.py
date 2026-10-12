from importlib.resources import files
import re
import xml.etree.ElementTree as ET

import pytest


@pytest.mark.parametrize("directory", ["skills/docs", "_hardware/docs"])
def test_navigation_diagrams_are_packaged_with_the_served_documentation(directory):
    root = files("bt_agent_robotics").joinpath(directory)
    links = re.findall(r"!\[[^]]*\]\(([^)]+)\)", root.joinpath("navigate.md").read_text())
    assert set(links) == {"navigate_tree.svg", "navigate_tree_bt.svg"}
    for target in links:
        diagram = root.joinpath(target)
        assert diagram.is_file(), f"Missing packaged documentation asset: {target}"
        document = ET.fromstring(diagram.read_bytes())
        assert document.tag == "{http://www.w3.org/2000/svg}svg"
