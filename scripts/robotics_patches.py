"""Narrow, reproducible release changes after reference namespace relocation.

Debug dumps become explicit callbacks instead of implicit shared /tmp writes.
Prompts, schemas, model-call arguments, skill observations and results stay put.
"""

import ast


def navigation_debug(raw):
    """Keep legacy diagnostic calls optional, never write to a shared path."""
    text = raw.decode()
    lines = text.splitlines(keepends=True)
    node = next(n for n in ast.parse(text).body if getattr(n, "name", None) == "_nav_debug_log")
    replacement = (
        "def _nav_debug_log(entry):\n"
        "    if _NAV_DEBUG_WRITER is not None:\n"
        "        try:\n            _NAV_DEBUG_WRITER(entry)\n"
        "        except Exception:\n            pass  # Optional diagnostics only.\n"
    )
    lines[node.lineno - 1 : node.end_lineno] = [replacement]
    text = "".join(lines)
    old = '_NAV_DEBUG_PATH = os.path.join(os.environ.get("LAM_CAPTURE_DIR", "/tmp/lam_captures"), "nav_debug.jsonl")'
    if text.count(old) != 1:
        raise ValueError("Navigation debug hook changed")
    return text.replace(old, "_NAV_DEBUG_WRITER = None").encode()


def artifact_writers(raw, filename):
    text = raw.decode()

    def replace(old, new):
        nonlocal text
        if text.count(old) != 1:
            raise ValueError(f"Unexpected {filename} artifact patch target: {old[:70]!r}")
        text = text.replace(old, new)

    if filename == "capture.py":
        replace(
            "        pointing_pipeline: Optional[PointingPipelineInterface] = None,\n",
            "        pointing_pipeline: Optional[PointingPipelineInterface] = None,\n        *, debug_writer=None,\n",
        )
        replace(
            "        self._pointing = pointing_pipeline\n",
            "        self._pointing = pointing_pipeline\n        self._debug_writer = debug_writer\n",
        )
        replace(
            "_FinalizeCapture(params, handle, state, context)",
            "_FinalizeCapture(params, handle, state, context, debug_writer=self._debug_writer)",
        )
        replace(
            '        context: SkillContext,\n    ) -> None:\n        super().__init__(name="finalize_capture")',
            '        context: SkillContext,\n        *, debug_writer=None,\n    ) -> None:\n        super().__init__(name="finalize_capture")\n        self._debug_writer = debug_writer',
        )
        replace(
            "            _save_capture_debug(image_bytes, cap.bbox_1000, label, cap_handle, self._context.main_task)",
            "            if self._debug_writer is not None:\n                self._debug_writer(image_bytes, cap.bbox_1000, label, cap_handle, self._context.main_task)",
        )
    elif filename == "surround_survey.py":
        replace(
            "        analyses: List[Dict[str, Any]],\n    ) -> None:",
            "        analyses: List[Dict[str, Any]],\n        *, debug_writer=None,\n    ) -> None:",
        )
        replace(
            "        self._analyses = analyses\n\n    def update",
            "        self._analyses = analyses\n        self._debug_writer = debug_writer\n\n    def update",
        )
        replace(
            "                self._milestone_desc,\n            )",
            "                self._milestone_desc,\n                debug_writer=self._debug_writer,\n            )",
        )
        replace(
            "    milestone_desc: str,\n) -> Dict[str, Any]:",
            "    milestone_desc: str,\n    *, debug_writer=None,\n) -> Dict[str, Any]:",
        )
        begin = text.index("    # XXX DEBUG: dump raw bytes sent to VLM")
        end = text.index("    # XXX END DEBUG", begin) + len("    # XXX END DEBUG")
        text = (
            text[:begin]
            + "    if debug_writer is not None:\n        try:\n            debug_writer(camera, image_bytes)\n        except Exception:\n            pass  # Optional diagnostics never change perception behavior."
            + text[end:]
        )
        replace(
            "    def __init__(self, model: ModelBackend) -> None:\n        self._model = model",
            "    def __init__(self, model: ModelBackend, *, debug_writer=None) -> None:\n        self._model = model\n        self._debug_writer = debug_writer",
        )
        replace(
            "                    analyses=analyses,\n",
            "                    analyses=analyses,\n                    debug_writer=self._debug_writer,\n",
        )
    return text.encode()
