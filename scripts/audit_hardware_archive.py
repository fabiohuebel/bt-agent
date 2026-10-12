"""Read-only compatibility checks against one archived deployed manifest/journals."""

import argparse
import hashlib
import json
import os
from pathlib import Path

from bt_agent.profiles import core_config
from bt_agent._engine.agent.system_prompts import get_prompts
from bt_agent_robotics.hardware import verify_sources, BatteryGuardPolicy


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--battery-journals", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    if args.out.exists():
        raise ValueError("Use a new audit output")
    archived = json.loads(args.manifest.read_text())
    source = verify_sources()
    from bt_agent_robotics._hardware.skills.capture import CaptureSkill
    from bt_agent_robotics._hardware.skills.query import QuerySkill
    from bt_agent_robotics._hardware.skills.navigate import NavigateSkill
    from bt_agent_robotics._hardware.skills.press import PressSkill
    from bt_agent_robotics._hardware.hardware_survey_description import make_survey
    from bt_agent.skills import ReportSkill
    from types import SimpleNamespace

    # Only construct skill schemas; no task, model request, ROS or physical call.
    os.environ["LAM_NAV_CONFIRM"] = "1"
    os.environ["LAM_CAPTURE_LOCAL_INSPECTION"] = "1"
    skills = [
        QuerySkill(None),
        NavigateSkill(None, None, SimpleNamespace(enabled=True)),
        PressSkill(None, None, SimpleNamespace(enabled=True)),
        ReportSkill(),
        make_survey(None, archived["survey_description_variant"]),
        CaptureSkill(None),
    ]
    tools = {s.name: s.tool_def() for s in skills}
    if tools != {name: row["tool"] for name, row in archived["skills"].items()}:
        raise AssertionError("Loaded tool schema differs from archived deployment")
    by_name = {
        Path(row["source"]).name: row for row in source["files"] if row["source"].endswith(".py")
    }
    for name in ("query", "navigate", "press", "capture"):
        if archived["skills"][name]["source"]["sha256"] != by_name[name + ".py"]["source_sha256"]:
            raise AssertionError("Wrong historical skill source: " + name)
    prompt = hashlib.sha256(get_prompts(core_config().prompt_variant).actor.encode()).hexdigest()
    if prompt != archived["actor_template_sha256"]:
        raise AssertionError("Actor prompt differs")
    decisions = []
    # These logs do not record every real battery update. Reconstruct each
    # explicit checkpoint from its logged inputs and prior latch state. This is
    # a decision-table replay, not continuous sensor or physical-timing replay.
    for journal in sorted(args.battery_journals.glob("battery_guard_*.jsonl")):
        for number, line in enumerate(journal.read_text().splitlines(), 1):
            record = json.loads(line)
            state = record.get("state")
            if not isinstance(state, dict) or "can_start" not in state:
                continue
            now = record["monotonic_s"]
            policy = BatteryGuardPolicy(clock=lambda: now)
            policy.real_percent = state["real_percent"]
            policy.received_monotonic = state["received_monotonic"]
            policy.override = state["override_percent"]
            policy.motion_active = state["motion_active"]
            policy.held = state["held"]
            policy.critical_latched = state["critical_latched"]
            policy.hold_reason = state["reason"] if state["held"] else None
            policy.hold_id = state["hold_id"]
            actual = policy.snapshot()
            fields = (
                "can_start",
                "can_continue",
                "held",
                "critical_latched",
                "sit_requested",
                "effective_percent",
                "reason",
            )
            if any(actual[key] != state[key] for key in fields):
                raise AssertionError(f"Battery decision mismatch: {journal.name}:{number}")
            decisions.append(
                dict(
                    journal=journal.name,
                    line=number,
                    input={
                        key: state[key]
                        for key in (
                            "real_percent",
                            "received_monotonic",
                            "override_percent",
                            "motion_active",
                            "held",
                            "critical_latched",
                            "reason",
                            "hold_id",
                        )
                    },
                    at=now,
                    decision={key: actual[key] for key in fields},
                )
            )
    if not decisions:
        raise ValueError("No archived battery checkpoints found")
    result = dict(
        passed=True,
        schemas=len(tools),
        actor_template_sha256=prompt,
        manifest_sha256=hashlib.sha256(args.manifest.read_bytes()).hexdigest(),
        battery_checkpoints=len(decisions),
        checkpoints=decisions,
        scope="Tool/source/prompt match and logged battery decision checkpoints; no physical timing or whole-stack equivalence",
    )
    with args.out.open("x") as stream:
        json.dump(result, stream, indent=2)
    print(json.dumps({key: value for key, value in result.items() if key != "checkpoints"}))


if __name__ == "__main__":
    main()
