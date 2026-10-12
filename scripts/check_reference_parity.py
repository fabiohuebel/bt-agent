"""Compare complete wire requests, prompts and terminal/report behavior."""

import json
import os
from pathlib import Path
import subprocess
import sys
import hashlib

root = Path(__file__).resolve().parents[1]
out = root / "artifacts"
out.mkdir(exist_ok=True)
cases = []
for case in ["report", "empty-report", "submit-retry"]:
    results = []
    for reference in [True, False]:
        # The reference uses Python's hash() to generate history tool-call IDs.
        # Pin hash randomization in both processes, not by rewriting payloads.
        env = {**os.environ, "PYTHONDONTWRITEBYTECODE": "1", "PYTHONHASHSEED": "0"}
        env["PYTHONPATH"] = (
            str(root / ".build/reference/src") + os.pathsep + str(root / ".build/reference")
            if reference
            else str(root / "packages/bt-agent/src")
        )
        args = [sys.executable, str(root / "scripts/replay_controller.py"), "--case", case]
        if reference:
            args.append("--reference")
        result = subprocess.run(args, env=env, cwd=root, capture_output=True, text=True, check=True)
        data = json.loads(result.stdout)
        results.append(data)
    if results[0] != results[1]:
        (out / f"{case}-reference.json").write_text(json.dumps(results[0], indent=2))
        (out / f"{case}-extracted.json").write_text(json.dumps(results[1], indent=2))
        raise AssertionError(f"Reference parity mismatch: {case}")
    digest = hashlib.sha256(json.dumps(results[0], sort_keys=True).encode()).hexdigest()
    cases.append(
        {
            "case": case,
            "passed": True,
            "reference_sha256": digest,
            "wire_requests": len(results[0]["wire_requests"]),
            "prompt_variants": len(results[0]["prompts"]),
            "actor_wire_caps": [r["json"]["max_tokens"] for r in results[0]["wire_requests"]],
        }
    )
(out / "reference-parity.json").write_text(
    json.dumps({"passed": True, "cases": cases}, indent=2) + "\n"
)
print(json.dumps(cases, indent=2))
