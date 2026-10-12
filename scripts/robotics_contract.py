"""Emit model-visible skill schemas, perception templates and served docs."""

import argparse
from importlib import import_module
import json
from pathlib import Path

parser = argparse.ArgumentParser()
parser.add_argument("--reference", action="store_true")
args = parser.parse_args()
prefix = "lam_v2_tiamat.bt_skills" if args.reference else "bt_agent_robotics.skills"
modules = {
    name: import_module(prefix + "." + name)
    for name in ["query", "navigate", "press", "surround_survey", "capture"]
}
report = import_module(
    "ref_support.report_skill" if args.reference else "bt_agent.skills"
).ReportSkill
skills = [
    modules["query"].QuerySkill(None),
    modules["navigate"].NavigateSkill(None, None),
    modules["press"].PressSkill(None, None),
    report(),
    modules["surround_survey"].SurroundSurveySkill(None),
    modules["capture"].CaptureSkill(None, None),
]
docs = Path(modules["query"].__file__).parent / "docs"
data = {
    "tools": [s.tool_def() for s in skills],
    "perception_templates": {
        name: {
            key: value
            for key, value in vars(module).items()
            if ("PROMPT" in key or "SCHEMA" in key) and isinstance(value, (str, dict))
        }
        for name, module in modules.items()
    },
    "served_docs": {path.name: path.read_text().strip() for path in sorted(docs.glob("*.md"))},
}
print(json.dumps(data, sort_keys=True))
