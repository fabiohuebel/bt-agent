"""ImageInjectSkill — inject a single camera image into the agent's context.

Pulls raw image tokens from one of the robot's five cameras into the LLM's
visual context. The image becomes available for the NEXT agent decision —
the actor includes it when building the prompt for the following
SelectSkill call.

This is the "expensive but detailed" option: the LLM sees the full image.
Use it when the agent needs to visually inspect something specific.
For a cheaper overview of all cameras, use SurroundSurvey instead.

Data flow:
    1. LLM selects image_inject(camera="frontleft")
    2. Skill grabs JPEG bytes from SkillContext.camera_images["frontleft"]
    3. Returns result_data["injected_images"] = {"frontleft": <bytes>}
    4. Actor picks up injected_images and includes them in the next
       model.query_tool_call() as vision input

Body subtree: single-leaf ``_ImageInjectNode``. ContextSkill — framework
auto-wraps with CancellationGuard only.

Agent-facing docs: ``bt_skills/docs/image_inject/image_inject.md`` — surfaced at
runtime via ``read_skill_docs("image_inject")``.
"""

from __future__ import annotations

from typing import Any, Dict

import py_trees
from bt_agent.skills import ContextSkill, SkillContext, SkillHandle

VALID_CAMERAS = frozenset({"frontleft", "frontright", "left", "right", "back"})


# ---------------------------------------------------------------------------
# BT node
# ---------------------------------------------------------------------------


class _ImageInjectNode(py_trees.behaviour.Behaviour):
    """Single BT node that grabs a camera image for context injection.

    Completes in one tick: validates camera, grabs JPEG bytes, writes
    to the handle, returns SUCCESS.
    """

    def __init__(
        self,
        params: Dict[str, Any],
        context: SkillContext,
        handle: SkillHandle,
    ) -> None:
        super().__init__(name="image_inject")
        self._params = params
        self._context = context
        self._handle = handle

    def update(self) -> py_trees.common.Status:
        camera = self._params.get("camera", "").strip().lower()

        if camera not in VALID_CAMERAS:
            self._handle.observation = f"Invalid camera '{camera}'. Valid cameras: {', '.join(sorted(VALID_CAMERAS))}."
            return py_trees.common.Status.FAILURE

        images = self._context.platform_state.get("camera_images")
        if not images or camera not in images:
            self._handle.observation = f"No image available from {camera} camera."
            return py_trees.common.Status.FAILURE

        self._handle.observation = f"Injected {camera} camera image into context."
        self._handle.result_data = {
            "observation": self._handle.observation,
            "injected_images": {camera: images[camera]},
        }
        return py_trees.common.Status.SUCCESS


# ---------------------------------------------------------------------------
# ImageInjectSkill
# ---------------------------------------------------------------------------


class ImageInjectSkill(ContextSkill):
    """Inject a single camera's image into the agent's visual context."""

    name = "image_inject"
    description = "Pull a camera image into your visual context so you can see it on the next step."
    usage = (
        "Use when you need to visually inspect something specific. "
        "Choose ONE camera. The image will be visible in your next decision."
    )
    params = {
        "camera": "Which camera to view. One of: frontleft, frontright, left, right, back.",
    }

    def build_subtree(
        self,
        params: Dict[str, Any],
        context: SkillContext,
        handle: SkillHandle,
    ) -> py_trees.behaviour.Behaviour:
        return _ImageInjectNode(params, context, handle)

    def tool_def(self) -> Dict[str, Any]:
        # ContextSkill mandates an explicit tool_def (Step 6.5).
        return super().tool_def()
