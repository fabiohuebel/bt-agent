"""Opt-in shared success information for the physical capture comparison."""

import hashlib

VERSION = "capture-evidence/v1"
TEXT = (
    "For this capture task, each requested target requires a submitted photograph showing that target. "
    "Locating a target through query or surround_survey alone does not satisfy the task."
)
HEADER = "ENVIRONMENT SUCCESS CONTRACT"
SHA256 = hashlib.sha256(TEXT.encode()).hexdigest()


def resolve_contract(environ):
    value = environ.get("LAM_SIM_CAPTURE_COMPLETION_CONTRACT", "")
    if not value:
        return None
    if value != VERSION or environ.get("LAM_SIM_EXPERIMENT_ARM") not in {"bt-core", "act", "react", "capx"}:
        raise ValueError("Capture contract requires its exact version and a physical comparison arm")
    return {"version": VERSION, "text": TEXT, "sha256": SHA256, "scope": "physical-capture-tasks-only"}


def policy_brief(original_task, contract):
    """Preserve the complete original task, including Task/Max-time wrapper."""
    if contract is None:
        return original_task, None
    if contract.get("version") != VERSION or contract.get("text") != TEXT or contract.get("sha256") != SHA256:
        raise ValueError("Resolved capture contract changed")
    delivered = original_task + "\n\n" + HEADER + "\n" + TEXT
    return delivered, {
        "version": VERSION,
        "contract_sha256": SHA256,
        "original_task": original_task,
        "original_task_sha256": hashlib.sha256(original_task.encode()).hexdigest(),
        "delivered_policy_brief": delivered,
        "delivered_policy_brief_sha256": hashlib.sha256(delivered.encode()).hexdigest(),
        "native_task_and_grader_changed": False,
    }
