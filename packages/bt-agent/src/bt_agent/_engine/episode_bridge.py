class RosClaimBridge:
    """Report/submit bookkeeping only. A terminal claim is NOT task success.

    The runner independently grades physical evidence. No goal, answer, grader,
    scene lookup, or perception substitution is available to this bridge.
    """

    def __init__(self, publish_report, flush_captures):
        self._publish_report = publish_report
        self._flush_captures = flush_captures
        self.reset()

    def reset(self):
        self.submitted = False

    def step(self, action):
        verb, _, text = action.strip().partition(" ")
        if self.submitted:
            return "Task already submitted.", 0.0, True, {"action_ok": False, "reason": "already_submitted"}
        if verb == "report" and text.strip():
            self._publish_report(text.strip())
            return "Report recorded.", 0.0, False, {"action_ok": True}
        if verb == "submit" and text.strip():
            self._flush_captures()
            self.submitted = True
            return (
                "Task submitted for independent grading.",
                0.0,
                True,
                {"action_ok": True, "outcome": "submitted-ungraded"},
            )
        return "Unsupported or empty report/submit.", 0.0, False, {"action_ok": False, "reason": "invalid_action"}
