"""No model server, ROS, simulator or GPU. All observations are synthetic."""

import json
from dataclasses import asdict

from bt_agent import create_agent
from bt_agent.episode import EpisodeHost
from bt_agent.skills import ReportSkill
from bt_agent.testing import InspectShelf, shelf_model


def main():
    reports = []
    model = shelf_model()
    host = EpisodeHost(reports.append, lambda: None)
    with create_agent(model=model, skills=[InspectShelf(), ReportSkill()], host=host) as agent:
        outcome = agent.run("Inspect the shelf and report what is there.")
        result = {
            "synthetic_offline_example": True,
            "termination": asdict(outcome),
            "reports": reports,
            "model_calls": len(model.requests),
            "decision_output_caps": [r["max_tokens"] for r in model.requests],
            "events": [e.kind for e in agent.events],
        }
    print(json.dumps(result, indent=2))
    if not outcome.submitted or len(model.requests) != 3:
        raise SystemExit("Offline qualification failed")


if __name__ == "__main__":
    main()
