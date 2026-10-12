"""Replay synthetic responses through the real HTTP serialization and BT loop.

No network: requests.post is replaced before constructing/running the model.
Run in separate processes for the frozen source and the extracted facade.
"""

import argparse
from dataclasses import asdict
from importlib import import_module
import json
import tempfile
from unittest.mock import Mock, patch


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", action="store_true")
    parser.add_argument(
        "--profile", choices=["paper-core-v1", "task-core-v1"], default="paper-core-v1"
    )
    parser.add_argument(
        "--case", choices=["report", "empty-report", "submit-retry"], default="report"
    )
    args = parser.parse_args()
    if args.reference and args.profile != "paper-core-v1":
        parser.error("The historical reference only supports paper-core-v1")
    prefix = "lam_v2" if args.reference else "bt_agent._engine"
    support = "ref_support" if args.reference else "bt_agent._engine"
    backend_module = import_module(prefix + ".model.backend")
    report_class = import_module(support + ".report_skill").ReportSkill
    host_class = import_module(support + ".episode_bridge").RosClaimBridge
    prompt_module = import_module(prefix + ".agent.system_prompts")
    config = (
        import_module("ref_support.profile").frozen_agent_config()
        if args.reference
        else import_module("bt_agent.profiles").core_config(args.profile)
    )
    reports, payloads, attempts = [], [], []

    class Host(host_class):
        def step(self, action):
            attempts.append(action)
            if (
                args.case == "submit-retry"
                and action.startswith("submit ")
                and sum(a.startswith("submit ") for a in attempts) == 1
            ):
                return "Not acknowledged yet.", 0, False, {"action_ok": False}
            return super().step(action)

    host = Host(reports.append, lambda: None)
    turns = [
        ("report", {"text": "The fixture contains a red mug."}, "Report the observation."),
        ("claim_complete", {}, "The report is recorded."),
    ]
    if args.case == "empty-report":
        turns.insert(0, ("report", {"text": ""}, "Attempt a report."))

    def post(url, **kwargs):
        payloads.append({"url": url, "json": kwargs["json"]})
        if not turns:
            raise AssertionError("Unexpected model request")
        name, params, thought = turns.pop(0)
        response = Mock()
        response.raise_for_status.return_value = None
        response.json.return_value = {
            "choices": [
                {
                    "finish_reason": "stop",
                    "message": {
                        "role": "assistant",
                        "content": thought,
                        "tool_calls": [
                            {
                                "type": "function",
                                "id": f"fixture-{len(payloads)}",
                                "function": {"name": name, "arguments": json.dumps(params)},
                            }
                        ],
                    },
                }
            ]
        }
        return response

    with (
        tempfile.TemporaryDirectory() as tmp,
        patch("requests.post", side_effect=post),
        patch.object(backend_module, "_VLM_LOG_DIR", tmp),
    ):
        model = backend_module.ModelBackend(
            {
                "endpoint": "http://offline.invalid/v1",
                "model_name": "fixture",
                "seed": 0,
                "temperature": 0.0,
                "max_tokens": 4096,
                "enable_thinking": False,
            }
        )
        if args.reference:
            agent = import_module(prefix + ".agent.lam_agent").LAMAgent(model, config=config)
            agent.register_skill("report", report_class())
            runner = import_module(prefix + ".agent.runner").AgentRunner(agent)
            runner.push_state({"_bridge": host})
            completed = runner.run(
                "Report the observed fixture contents.", max_ticks=500, tick_rate=0.001
            )
        else:
            with import_module("bt_agent").create_agent(
                model=model, skills=[report_class()], host=host, profile=args.profile
            ) as runtime:
                result = runtime.run(
                    "Report the observed fixture contents.", max_ticks=500, tick_interval=0.001
                )
                completed = result.submitted
        assert completed and host.submitted and not turns
    data = {
        "case": args.case,
        "submitted": host.submitted,
        "reports": reports,
        "host_actions": attempts,
        "wire_requests": payloads,
        "profile": asdict(config),
        # The release ships only the profile's prompt variant; compare it against
        # the frozen source's assembled bundle of the same name.
        "prompts": {config.prompt_variant: asdict(prompt_module.get_prompts(config.prompt_variant))},
    }
    print(json.dumps(data, sort_keys=True, default=lambda x: sorted(x)))


if __name__ == "__main__":
    main()
