"""Compare reference and installed robotics package without robot/VLM access."""

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

root = Path(__file__).resolve().parents[1]
values = []
for reference in [True, False]:
    env = {**os.environ, "LAM_NAV_CONFIRM": "0", "LAM_IMAGE_INJECT": "off", "PYTHONHASHSEED": "0"}
    env.pop("PYTHONPATH", None)
    if reference:
        env["PYTHONPATH"] = os.pathsep.join(
            str(root / path)
            for path in [".build/reference/src", ".build/reference", ".build/reference_robotics"]
        )
    args = [sys.executable, str(root / "scripts/robotics_contract.py")]
    if reference:
        args.append("--reference")
    output = subprocess.run(args, cwd=root, env=env, capture_output=True, text=True, check=True)
    values.append(json.loads(output.stdout))
assert values[0] == values[1], "Robot skill interface/template/doc parity mismatch"
result = {
    "passed": True,
    "reference_commit": "62056af6fcb9ac3ec611760415c0c4f2f91a1fb6",
    "tool_order": [t["function"]["name"] for t in values[0]["tools"]],
    "served_docs": sorted(values[0]["served_docs"]),
    "contract_sha256": hashlib.sha256(json.dumps(values[0], sort_keys=True).encode()).hexdigest(),
    "scope": (
        "Tool schemas, registry order, declared perception templates and documentation; "
        "not physical outcome parity"
    ),
}
(root / "artifacts/robotics-parity.json").write_text(json.dumps(result, indent=2) + "\n")
print(json.dumps(result, indent=2))
