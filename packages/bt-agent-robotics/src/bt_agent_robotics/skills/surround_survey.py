"""SurroundSurveySkill — sequential multi-camera visual analysis as a BT.

Body subtree (framework auto-wraps with CancellationGuard — see safeguard.py):

    SurroundSurvey (Sequence, memory=True)
    ├── AnalyzeView("frontleft")      [sync LLM call, blocks in bg thread]
    ├── AnalyzeView("frontright")
    ├── AnalyzeView("left")
    ├── AnalyzeView("right")
    ├── AnalyzeView("back")
    └── CompileResults                [sync: picks best view, builds summary]

Effect type: ``ContextSkill`` (Σ_ctx). Read-only.

Agent-facing docs (leaf catalog, recovery hints):
``tiamat_skills/docs/surround_survey.md`` — surfaced at runtime via
``read_skill_docs("surround_survey")``.

Each AnalyzeView node makes one VLM call asking whether the target is
visible in that camera's image. The Sequence runs in a background thread
(via Skill.start default), so the sync LLM calls don't block the 10Hz
main BT tick. Progress and cancellation propagate through the SkillHandle.

This is the "cheap but broad" option: five small LLM calls producing
text summaries (~500 tokens total) instead of five full images injected
into context (~7500 tokens). The agent gets a textual answer about the
surroundings without the token cost.

Mirrors v1's MultiViewAnalyzer.analyze_multi_view_sequential().
"""

from __future__ import annotations

from typing import Any, Dict, List

import py_trees
from bt_agent.models import ModelBackend
from bt_agent.skills import ContextSkill, SkillContext, SkillHandle

CAMERAS = ["frontleft", "frontright", "left", "right", "back"]

# -- Response schemas for structured VLM output --------------------------------

# XXX CONCLUSION ENUM — schema evolution log:
#   v1 (orig):  status enum {"detected","not_detected"}                   — drift FP/FN ~6-15%
#   v2 (1st):   conclusion enum (sentence completion w/ "IS" vs "is NOT")  — MODE COLLAPSE
#               (surface-form prior P("is") >> P("IS") forced 100% is_NOT regardless of evidence)
#   v3 (this): visible: boolean (matches CaptureSkill which works empirically) — CoT preserved
SINGLE_VIEW_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Brief objective description of the visible area (max 40 words).",
        },
        "evaluation": {
            "type": "string",
            "description": "Direct answer to the query based ONLY on visual evidence.",
        },
        "visible": {
            "type": "boolean",
            "description": "Whether the queried target is visible in this view.",
        },
        "bounding_box": {
            "type": "array",
            "items": {"type": "number"},
            "description": "Bounding box [x_min, y_min, x_max, y_max] of the target if visible.",
        },
    },
    "required": ["reasoning", "evaluation", "visible"],
}
# XXX END CONCLUSION ENUM

COMPARE_VIEWS_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of why this view is better.",
        },
        "chosen_view": {
            "type": "string",
            "enum": ["A", "B"],
            "description": "Which view is better: 'A' or 'B'.",
        },
    },
    "required": ["reasoning", "chosen_view"],
}


# ---------------------------------------------------------------------------
# BT nodes (internal — run in background thread, may block)
# ---------------------------------------------------------------------------


class _AnalyzeViewNode(py_trees.behaviour.Behaviour):
    """Analyze a single camera view for the target.

    Makes a synchronous VLM call (blocks in the background thread).
    Appends the result to the shared ``analyses`` list.
    """

    def __init__(
        self,
        camera: str,
        image_bytes: bytes | None,
        target: str,
        milestone_desc: str,
        model: ModelBackend,
        handle: SkillHandle,
        analyses: List[Dict[str, Any]],
        *, debug_writer=None,
    ) -> None:
        super().__init__(name=f"analyze_{camera}")
        self._camera = camera
        self._image_bytes = image_bytes
        self._target = target
        self._milestone_desc = milestone_desc
        self._model = model
        self._handle = handle
        self._analyses = analyses
        self._debug_writer = debug_writer

    def update(self) -> py_trees.common.Status:
        # Check cancellation before making LLM call
        if self._handle.is_cancel_requested:
            return py_trees.common.Status.FAILURE

        if self._image_bytes is None:
            self._analyses.append({
                "camera": self._camera,
                "reasoning": "No image available.",
                "evaluation": "",
                "visible": False,  # XXX CONCLUSION ENUM
                "found": False,
            })
            self._handle.progress = len(self._analyses) / len(CAMERAS)
            return py_trees.common.Status.SUCCESS

        try:
            result = _analyze_single_view(
                self._model,
                self._camera,
                self._image_bytes,
                self._target,
                self._milestone_desc,
                debug_writer=self._debug_writer,
            )
            self._analyses.append(result)
        except Exception as e:
            self._analyses.append({
                "camera": self._camera,
                "reasoning": f"Analysis failed: {e}",
                "evaluation": "",
                "visible": False,  # XXX CONCLUSION ENUM
                "found": False,
            })

        self._handle.progress = len(self._analyses) / len(CAMERAS)
        return py_trees.common.Status.SUCCESS


class _CompileResultsNode(py_trees.behaviour.Behaviour):
    """Compile per-camera analyses into a final observation.

    Runs after all AnalyzeView nodes complete. Determines:
    - 0 candidates: summarizes what WAS seen (may call VLM)
    - 1 candidate: returns that camera's analysis
    - 2+ candidates: compares views to pick best (may call VLM)
    """

    def __init__(
        self,
        target: str,
        model: ModelBackend,
        handle: SkillHandle,
        analyses: List[Dict[str, Any]],
        images: Dict[str, bytes],
    ) -> None:
        super().__init__(name="compile_results")
        self._target = target
        self._model = model
        self._handle = handle
        self._analyses = analyses
        self._images = images

    def update(self) -> py_trees.common.Status:
        if self._handle.is_cancel_requested:
            return py_trees.common.Status.FAILURE

        candidates = [a for a in self._analyses if a["found"]]

        if len(candidates) == 0:
            observation = _summarize_not_found(
                self._model,
                self._analyses,
                self._target,
            )
            self._handle.observation = observation
            self._handle.result_data = {
                "verdict": False,
                "camera": "none",
                "analyses": list(self._analyses),
            }

        elif len(candidates) == 1:
            best = candidates[0]
            self._handle.observation = f"Target found in {best['camera']} camera. {best['evaluation']}"
            self._handle.result_data = {
                "verdict": True,
                "camera": best["camera"],
                "analyses": list(self._analyses),
            }

        else:
            try:
                best = _compare_views(
                    self._model,
                    candidates[:2],
                    self._target,
                    self._images,
                )
            except Exception:
                best = candidates[0]

            self._handle.observation = f"Target found in {best['camera']} camera. {best['evaluation']}"
            self._handle.result_data = {
                "verdict": True,
                "camera": best["camera"],
                "analyses": list(self._analyses),
            }

        return py_trees.common.Status.SUCCESS


# ---------------------------------------------------------------------------
# VLM call helpers (pure functions, called from BT nodes)
# ---------------------------------------------------------------------------


def _analyze_single_view(
    model: ModelBackend,
    camera: str,
    image_bytes: bytes,
    target: str,
    milestone_desc: str,
    *, debug_writer=None,
) -> Dict[str, Any]:
    """Analyze a single camera view. Returns analysis dict."""
    # XXX CONCLUSION ENUM — OBJECT-priority disclaimer stays dropped (it caused FPs on
    # relational queries). The "visible: boolean" field is referenced as step 4.
    messages = [
        {
            "role": "user",
            "content": (
                f"Analyze the {camera} camera feed from the Spot robot.\n\n"
                f"Task Context: {milestone_desc}\n"
                f'User Query: "{target}"\n\n'
                "Instructions:\n"
                "1. Observe: Describe the main elements of the scene "
                "relevant to the context, noting their spatial positions.\n"
                '2. Evaluate: Specifically address the "User Query" based '
                "ONLY on visual evidence.\n"
                "3. Locate: If a specific object is the focus of the query, "
                "provide its bounding box.\n"
                "4. Decide: Set visible=true if the target is clearly or "
                "partially visible in this frame; visible=false otherwise. "
                "Your visible flag MUST match what you wrote in evaluation."
            ),
        },
    ]

    if debug_writer is not None:
        try:
            debug_writer(camera, image_bytes)
        except Exception:
            pass  # Optional diagnostics never change perception behavior.

    result = model.query_structured(
        messages,
        SINGLE_VIEW_SCHEMA,
        images=[image_bytes],
        max_tokens=1024,
        temperature=0.0,
        phase="surround_survey",
    )

    # XXX CONCLUSION ENUM — boolean visible field
    visible = bool(result.get("visible", False))
    return {
        "camera": camera,
        "reasoning": result.get("reasoning", ""),
        "evaluation": result.get("evaluation", ""),
        "visible": visible,
        "found": visible,
    }


def _compare_views(
    model: ModelBackend,
    candidates: List[Dict[str, Any]],
    target: str,
    images: Dict[str, bytes],
) -> Dict[str, Any]:
    """Compare two candidate views to pick the best one."""
    cam_a = candidates[0]["camera"]
    cam_b = candidates[1]["camera"]

    img_a = images.get(cam_a)
    img_b = images.get(cam_b)
    vision_images = [img for img in [img_a, img_b] if img is not None]

    messages = [
        {
            "role": "user",
            "content": (
                "You are comparing two camera views to determine which "
                "one BEST shows the target object.\n\n"
                f'Target Object: "{target}"\n\n'
                f'View A: "{cam_a}" camera (first image)\n'
                f'View B: "{cam_b}" camera (second image)\n\n'
                "Previous analyses suggested BOTH views might contain "
                "the target. Your job is to pick the BETTER view.\n\n"
                "Criteria for selection:\n"
                "1. CONFIDENCE: Which view are you MORE CERTAIN actually "
                "contains the target (not a hallucination)?\n"
                "2. VISIBILITY: Which view shows the object more clearly "
                "(not partially occluded, not at edge)?\n"
                "3. SIZE: Which view shows the object at a better size "
                "for identification?\n\n"
                "IMPORTANT: If one view seems to be a FALSE POSITIVE "
                "(hallucination, misidentification), choose the other.\n"
                "If both seem equally valid, prefer the one with better "
                "visibility.\n\n"
                "Answer with chosen_view = 'A' or 'B'."
            ),
        },
    ]

    result = model.query_structured(
        messages,
        COMPARE_VIEWS_SCHEMA,
        images=vision_images,
        max_tokens=128,
        temperature=0.0,
        phase="surround_survey",
    )

    chosen = result.get("chosen_view", "").strip().upper()
    if chosen == "B":
        return candidates[1]
    return candidates[0]


def _summarize_not_found(
    model: ModelBackend,
    analyses: List[Dict[str, Any]],
    target: str,
) -> str:
    """Compile a summary when no camera detected the target."""
    view_lines = []
    for a in analyses:
        if a["reasoning"] and a["reasoning"] != "No image available.":
            view_lines.append(f"- {a['camera']}: {a['reasoning']}")

    if not view_lines:
        return f"'{target}' not detected in any camera view."

    view_summary = "\n".join(view_lines)

    messages = [
        {
            "role": "user",
            "content": (
                "The robot analyzed all camera views for the following "
                "query but none returned a positive detection.\n\n"
                f'Query: "{target}"\n\n'
                f"Per-camera observations:\n{view_summary}\n\n"
                "Write a concise 1-2 sentence summary explaining what "
                "was seen and why the query could not be confirmed. "
                "Be specific — reference what the cameras actually observed."
            ),
        },
    ]

    try:
        return model.query(messages, max_tokens=128, temperature=0.0, phase="surround_survey")
    except Exception:
        return f"'{target}' not detected in any camera view."


# ---------------------------------------------------------------------------
# SurroundSurveySkill
# ---------------------------------------------------------------------------


class SurroundSurveySkill(ContextSkill):
    """Sequential multi-camera visual analysis — compound BT skill.

    ``build_subtree()`` returns a Sequence of AnalyzeView nodes (one per
    camera) followed by a CompileResults node. The default ``start()``
    runs this subtree in a background thread.
    """

    name = "surround_survey"
    description = (
        "Check whether ONE specific target is visible by running a per-camera VLM "
        "check across all five robot cameras, then picking the best camera for capture."
    )
    usage = (
        "Use after NAVIGATE to confirm arrival and identify which camera shows the target. "
        "Returns a verdict (true/false) and the chosen camera name. "
        "One atomic question per call — for multiple targets, issue one call each. "
        "For spatial reasoning ('where is X relative to Y?'), use IMAGE_INJECT instead."
    )
    params = {
        "target_description": (
            "ONE atomic visibility question. "
            "GOOD: 'is the red umbrella visible?', 'is the safe door open?'. "
            "BAD (compound): 'is X AND Y visible?' → split into two calls. "
            "BAD (spatial): 'where is X relative to Y?' → use IMAGE_INJECT. "
            "BAD (vague): 'safe door' → say what condition to verify."
        ),
    }

    # Static survey: VLM analysis of the five current camera frames, no robot
    # motion. A future ActiveSurroundSurveySkill that rotates the robot would
    # subclass ActionSkill instead.

    def __init__(self, model: ModelBackend, *, debug_writer=None) -> None:
        self._model = model
        self._debug_writer = debug_writer

    def build_subtree(
        self,
        params: Dict[str, Any],
        context: SkillContext,
        handle: SkillHandle,
    ) -> py_trees.behaviour.Behaviour:
        target = params.get("target_description", "")
        # A store handle ("safe box 1") is bookkeeping vocabulary; the VLM is
        # asked about the object's LABEL ("safe box") when the handle resolves,
        # mirroring CaptureSkill.prepare's ``target_label = scene_obj.label``.
        store = getattr(context, "object_store", None)
        scene_obj = store.get(target) if (store is not None and target) else None
        label = getattr(scene_obj, "label", "") if scene_obj is not None else ""
        if label:
            target = label
        images = context.platform_state.get("camera_images") or {}
        milestone = context.milestone or {}
        milestone_desc = milestone.get("description", context.main_task)

        # Shared mutable list — safe because Sequence is serial
        analyses: List[Dict[str, Any]] = []

        children: list[py_trees.behaviour.Behaviour] = []
        for camera in CAMERAS:
            children.append(
                _AnalyzeViewNode(
                    camera=camera,
                    image_bytes=images.get(camera),
                    target=target,
                    milestone_desc=milestone_desc,
                    model=self._model,
                    handle=handle,
                    analyses=analyses,
                    debug_writer=self._debug_writer,
                )
            )

        children.append(
            _CompileResultsNode(
                target=target,
                model=self._model,
                handle=handle,
                analyses=analyses,
                images=images,
            )
        )

        return py_trees.composites.Sequence(
            name="surround_survey",
            memory=True,
            children=children,
        )

    def tool_def(self) -> Dict[str, Any]:
        # ContextSkill mandates an explicit tool_def (Step 6.5).
        return super().tool_def()
