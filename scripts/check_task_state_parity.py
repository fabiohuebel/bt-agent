"""Compare both release runtimes using synthetic, network-free HTTP replays.

Optionally include the benchmark's existing primary/transfer/fault fixtures.
The benchmark's explicit runtime argument selects each implementation; no files
or installed defaults change during the check.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--benchmark-root", type=Path)
    parser.add_argument(
        "--output", type=Path, default=Path("artifacts/task-state-20261007/parity.json")
    )
    args = parser.parse_args()
    root = Path(__file__).resolve().parents[1]
    cases = [("core", case) for case in ["report", "empty-report", "submit-retry"]]
    if args.benchmark_root:
        from importlib.util import module_from_spec, spec_from_file_location

        benchmark = args.benchmark_root.resolve()
        sys.path.insert(0, str(benchmark / "src"))
        spec = spec_from_file_location("fixtures", benchmark / "integrations/bt_agent/fixtures.py")
        fixtures = module_from_spec(spec)
        spec.loader.exec_module(fixtures)
        cases += [("benchmark", case) for case in fixtures.CASES + ["malformed", "action-limit"]]
    records = []
    for suite, case in cases:
        results = []
        for profile in ["paper-core-v1", "task-core-v1"]:
            paths = [str(root / "packages/bt-agent/src")]
            if suite == "core":
                command = [
                    sys.executable,
                    str(root / "scripts/replay_controller.py"),
                    "--case",
                    case,
                    "--profile",
                    profile,
                ]
            else:
                paths.append(str(benchmark / "src"))
                command = [
                    sys.executable,
                    str(benchmark / "scripts/replay_bt_connector.py"),
                    "--case",
                    case,
                    "--runtime-profile",
                    profile,
                ]
            env = {
                **os.environ,
                "PYTHONPATH": os.pathsep.join(paths),
                "PYTHONHASHSEED": "0",
                "PYTHONDONTWRITEBYTECODE": "1",
            }
            result = subprocess.run(
                command, cwd=root, env=env, capture_output=True, text=True, timeout=120
            )
            if result.returncode:
                raise RuntimeError(f"{suite}/{case}/{profile}: {result.stderr[-5000:]}")
            results.append(json.loads(result.stdout))
        if results[0] != results[1]:
            args.output.parent.mkdir(parents=True, exist_ok=True)
            for name, value in zip(["paper", "task"], results):
                (args.output.parent / f"{suite}-{case}-{name}.json").write_text(
                    json.dumps(value, indent=2) + "\n"
                )
            raise AssertionError(f"Full replay mismatch: {suite}/{case}")
        records.append(
            dict(
                suite=suite,
                case=case,
                matched=True,
                wire_requests=len(results[0]["wire_requests"]),
                sha256=hashlib.sha256(json.dumps(results[0], sort_keys=True).encode()).hexdigest(),
            )
        )
        print(f"PASS {suite}/{case}: {records[-1]['wire_requests']} identical requests", flush=True)
    data = dict(
        schema="bt-agent-task-migration-parity/v1",
        passed=True,
        profiles=["paper-core-v1", "task-core-v1"],
        scope="Complete HTTP payloads, tool definitions, host actions and available grades; scripted offline replay, not model sampling or physical validation",
        cases=records,
    )
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(data, indent=2) + "\n")


if __name__ == "__main__":
    main()
