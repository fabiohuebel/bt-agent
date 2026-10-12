"""CaptureSkill — photograph a target object from a specified camera.

Multi-step BT skill that:
1. Validates target + camera + image availability
2. Asks the VLM if the target is visible and to provide a tight bounding box
3. Optionally refines the bounding box (1-2 VLM iterations)
4. Optionally runs a segmentation pipeline (SAM2 or alternative)
5. Builds a CaptureResult and adds it to the ObjectStore

The actual transmission of the captured image to an external task evaluator
is a system concern (handled by the bridge based on configuration). The skill
itself only produces the artifact — submission is automatic on milestone advance.

Body subtree (framework auto-wraps with safeguards — see safeguard.py):

    Sequence("capture", memory=True)
    ├── _ValidateTarget       — check target_object handle, camera, image bytes
    ├── _AnalyzeView          — VLM: visible? bbox?
    ├── _RefinePointing       — VLM: tighten bbox (best-effort, optional)
    ├── _SegmentMask          — pointing pipeline (SAM2), if injected
    └── _FinalizeCapture      — build CaptureResult, write to handle

Agent-facing docs (leaf catalog, safeguard semantics, recovery hints):
``tiamat_skills/docs/capture.md`` — surfaced at runtime via
``read_skill_docs("capture")``.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from collections.abc import Sequence
from dataclasses import dataclass
from io import BytesIO
from numbers import Number
from typing import Any, Dict, List, Optional, Tuple

import py_trees
from bt_agent.skills import SkillObject
from bt_agent.skills import ActionSkill, SkillContext, SkillHandle

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# CaptureResult — producer-local type (never imported by consumer skills)
# ---------------------------------------------------------------------------


@dataclass
class CaptureResult(SkillObject):
    """A capture result from the pointing/segmentation pipeline.

    This type is local to CaptureSkill (the sole producer). Consumer code
    (evaluator, submission) accesses fields via duck-typed ``getattr``.
    """

    # -- VLM-visible --
    description: str = ""
    camera: str = ""

    # -- VLM-hidden --
    target_coords: Tuple[int, int] = (0, 0)
    bbox_1000: Tuple[int, int, int, int] = (0, 0, 0, 0)
    rotated_image_width: int = 0
    rotated_image_height: int = 0
    mask_backend: str = ""
    mask_similarity: float = 0.0
    mask_candidate_count: int = 0
    debug_id: str = ""
    image: Optional[Any] = None  # np.ndarray, for evaluator
    # physics-trace-recording wave 2: the actor decision (SelectSkill's
    # SKILL_CALL_ID) that produced this capture — links the submission back
    # to its skill_selected/skill_started/skill_completed events.
    skill_call_id: str = ""

    def to_llm_summary(self) -> str:
        return f"Capture<{self.handle}>: '{self.description}' from {self.camera} camera"


# ---------------------------------------------------------------------------
# PointingPipelineInterface ABC
# ---------------------------------------------------------------------------


@dataclass
class SegmentResult:
    """Result of running the segmentation pipeline on a bbox."""

    mask_data: bytes = b""
    similarity: float = 0.0
    backend: str = ""
    candidate_count: int = 0
    target_pixel: Optional[Tuple[int, int]] = None  # interior point, falls back to bbox center


class PointingPipelineInterface(ABC):
    """Proxy to the segmentation pipeline (SAM2 or alternative).

    Implementations:
        - lam_v2_roscore: wraps the real SAM2 + pointing logic
        - lam_v2_testing: returns canned SegmentResult
    """

    @abstractmethod
    def segment(
        self,
        image: bytes,
        bbox_1000: List[int],
        target_description: str,
    ) -> SegmentResult:
        """Run segmentation on the image+bbox region for the target.

        Args:
            image: JPEG bytes of the camera frame.
            bbox_1000: Bounding box in [0..1000] normalized scale [x1, y1, x2, y2].
            target_description: What to segment (for prompt-based pipelines).

        Returns:
            SegmentResult with mask data and an optional interior point.
        """
        ...


# ---------------------------------------------------------------------------
# VLM response schemas
# ---------------------------------------------------------------------------

ANALYZE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "reasoning": {
            "type": "string",
            "description": "Brief explanation of what you see (max 30 words).",
        },
        "visible": {
            "type": "boolean",
            "description": "Whether the target is visible in the image.",
        },
        "bbox_1000": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 4,
            "maxItems": 4,
            "description": (
                "Tight bounding box [x1, y1, x2, y2] in 0..1000 normalized coords. Use [0,0,0,0] if not visible."
            ),
        },
    },
    "required": ["reasoning", "visible", "bbox_1000"],
}

REFINE_SCHEMA: Dict[str, Any] = {
    "type": "object",
    "properties": {
        "acceptable": {
            "type": "boolean",
            "description": "Whether the current bounding box is tight enough on the target.",
        },
        "bbox_1000": {
            "type": "array",
            "items": {"type": "integer"},
            "minItems": 4,
            "maxItems": 4,
            "description": "Refined bounding box in 0..1000 coords. Same as input if acceptable.",
        },
        "reasoning": {
            "type": "string",
            "description": "Brief explanation.",
        },
    },
    "required": ["acceptable", "bbox_1000", "reasoning"],
}


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_CAMERAS = frozenset({"frontleft", "frontright", "left", "right", "back"})

_REFINE_MAX_ITERS = 2

_ANALYZE_PROMPT = (
    "You are inspecting a camera frame from a Boston Dynamics Spot robot. "
    "Identify whether the target object is visible. The target name always "
    "denotes one WHOLE object, never one of its parts (a 'claw hammer' is the "
    "head and the handle; an 'umbrella curved handle' is the entire umbrella). "
    "If visible, provide a bounding box in 0..1000 normalized coordinates "
    "[x1, y1, x2, y2] that encloses every visible part of that object. A box "
    "that cuts off part of the object is wrong; prefer a slightly generous "
    "box over a partial one, but do not include neighboring objects. If several "
    "similar objects are visible, box the one that best matches the name. "
    "If not visible, set visible=false and return [0,0,0,0]."
)

_REFINE_PROMPT = (
    "Below is a camera frame and a proposed bounding box for the target. "
    "Decide if the box encloses the WHOLE target object (every visible part). "
    "If yes, set acceptable=true and return the same box. If it covers only a "
    "part of the object (for example just a handle or a lid), enlarge it to "
    "the whole object. If it is on a different object or far too loose, "
    "correct it."
)


# ---------------------------------------------------------------------------
# CaptureSkill
# ---------------------------------------------------------------------------


class CaptureSkill(ActionSkill):
    """Photograph a target object from a specified camera.

    The actor selects which camera to use (typically informed by a prior
    SURROUND_SURVEY) and the target handle. The skill verifies visibility,
    refines the bounding box, optionally runs segmentation, and produces
    a CaptureResult in the ObjectStore.
    """

    name = "capture"
    description = "Capture a photo of a target object using a specified camera."
    usage = (
        "Use after NAVIGATE confirmed proximity and SURROUND_SURVEY identified "
        "the camera with the best view. The captured image is recorded for "
        "submission when the milestone advances."
    )
    params = {
        "target_object": "Handle from QUERY (e.g. 'red_umbrella_1').",
        "camera_view": "Camera name: frontleft, frontright, left, right, or back.",
    }

    def __init__(
        self,
        model: Any,
        pointing_pipeline: Optional[PointingPipelineInterface] = None,
        *, debug_writer=None,
    ) -> None:
        self._model = model
        self._pointing = pointing_pipeline
        self._debug_writer = debug_writer
        self._local_inspection = os.environ.get("LAM_CAPTURE_LOCAL_INSPECTION", "0") == "1"
        if self._local_inspection:
            self.params = {
                **type(self).params,
                "target_object": (
                    "Object handle from QUERY, or a visual description of a newly observed object. "
                    "An object seen by SURROUND_SURVEY need not be in scene memory to be captured."
                ),
            }

    def precondition(self, state: SkillContext) -> Tuple[bool, str]:
        if state.object_store is None:
            return False, "ObjectStore not available."
        if not self._local_inspection and not state.object_store.all_objects():
            return False, "No object handles in ObjectStore (call QUERY first)."
        if not state.platform_state.get("camera_images"):
            return False, "No camera images available."
        # TIAMAT-specific: only available on capture milestones. The milestone
        # dict is substrate-opaque; CaptureSkill knows its own gating rule.
        if state.milestone and not state.milestone.get("capture", True):
            return False, "This milestone is not a capture milestone."
        return True, ""

    def safeguard_invariant(self, state: SkillContext, handle: SkillHandle) -> bool:
        """Pre-analysis vacuously safe; post-analysis target camera still present.

        I5-compliant: reads only ``state.platform_state["camera_images"]``
        (observation projection) and ``handle.result_data["analyze_view_*"]``
        (skill execution state). Does not consult ``object_store``,
        ``milestone``, ``main_task``, or ``scenario``.
        """
        if not handle.result_data.get("analyze_view_succeeded"):
            return True  # Pre-analysis: nothing to invalidate yet
        target_cam = handle.result_data.get("analyze_view_camera")
        if target_cam is None:
            return False
        camera_images = state.platform_state.get("camera_images")
        if camera_images is None or target_cam not in camera_images:
            return False
        return True

    def is_available(
        self,
        state: Dict[str, Any],
        context: Dict[str, Any],
    ) -> Tuple[bool, str]:
        # TODO: gate on "agent has pulled camera images into context this
        # milestone" — only then has the VLM credibly seen what it wants
        # to capture. Check state["tool_calls"] for a prior image_inject
        # or surround_survey call.
        return True, f"{self.name} available"

    def prepare(
        self,
        params: Dict[str, Any],
        context: SkillContext,
    ) -> Tuple[Dict[str, Any], Optional[str]]:
        target_ref = params.get("target_object", "")
        if not isinstance(target_ref, str) or not target_ref.strip():
            return {}, "No target_object provided."
        target_ref = target_ref.strip()

        store = context.object_store
        if store is None:
            return {}, "ObjectStore not available."

        scene_obj = store.get(target_ref)
        if scene_obj is None and not self._local_inspection:
            available = [o.handle for o in store.all_objects()]
            avail_str = ", ".join(available[:10]) if available else "none"
            return {}, f"'{target_ref}' not in ObjectStore. Available: {avail_str}. Use QUERY first."

        camera = params.get("camera_view", "").strip().lower()
        if camera not in VALID_CAMERAS:
            return {}, f"Unknown camera_view '{camera}'. Must be one of: {', '.join(sorted(VALID_CAMERAS))}."

        images = context.platform_state.get("camera_images") or {}
        image_bytes = images.get(camera)
        if image_bytes is None:
            available_cams = ", ".join(sorted(images.keys())) if images else "none"
            return {}, f"No image available from '{camera}'. Available cameras: {available_cams}."

        prepared = {
            "target_handle": scene_obj.handle if scene_obj is not None else "",
            "target_label": (scene_obj.label or target_ref) if scene_obj is not None else target_ref.replace("_", " "),
            "camera_view": camera,
            "image_bytes": image_bytes,
        }
        if scene_obj is None:
            prepared["target_source"] = "local_visual_description"
        return prepared, None

    def build_subtree(
        self,
        params: Dict[str, Any],
        context: SkillContext,
        handle: SkillHandle,
    ) -> py_trees.behaviour.Behaviour:
        # Shared mutable state passed between BT nodes (mirrors SurroundSurvey pattern)
        state: Dict[str, Any] = {
            "bbox_1000": None,
            "segment_result": None,
        }
        return py_trees.composites.Sequence(
            name="capture",
            memory=True,
            children=[
                _ValidateTarget(params, handle, state),
                _AnalyzeView(params, handle, state, self._model),
                _RefinePointing(params, handle, state, self._model),
                _SegmentMask(params, handle, state, self._pointing),
                _FinalizeCapture(params, handle, state, context, debug_writer=self._debug_writer),
            ],
        )

    def publish_results(self, handle: SkillHandle, store) -> None:
        """K-publication (paper §5 I6): adds the CaptureResult artifact for downstream submission."""
        result = handle.result_data or {}
        for obj in result.get("new_objects", []):
            store.add(obj)


# ---------------------------------------------------------------------------
# BT Nodes
# ---------------------------------------------------------------------------


class _ValidateTarget(py_trees.behaviour.Behaviour):
    """Single tick: input validation. prepare() did most of this; this is a guard."""

    def __init__(
        self,
        params: Dict[str, Any],
        handle: SkillHandle,
        state: Dict[str, Any],
    ) -> None:
        super().__init__(name="validate_target")
        self._params = params
        self._handle = handle
        self._state = state

    def update(self) -> py_trees.common.Status:
        if self._handle.is_cancel_requested:
            return py_trees.common.Status.FAILURE

        if not self._params.get("image_bytes"):
            self._handle.observation = "No image bytes available for capture."
            return py_trees.common.Status.FAILURE

        label = self._params.get("target_label", "target")
        camera = self._params.get("camera_view", "?")
        self._handle.observation = f"Inspecting '{label}' in {camera} camera..."
        self._handle.progress = 0.05
        return py_trees.common.Status.SUCCESS


class _AnalyzeView(py_trees.behaviour.Behaviour):
    """VLM call: is the target visible? Tight bounding box?"""

    def __init__(
        self,
        params: Dict[str, Any],
        handle: SkillHandle,
        state: Dict[str, Any],
        model: Any,
    ) -> None:
        super().__init__(name="analyze_view")
        self._params = params
        self._handle = handle
        self._state = state
        self._model = model

    def update(self) -> py_trees.common.Status:
        if self._handle.is_cancel_requested:
            return py_trees.common.Status.FAILURE

        label = self._params["target_label"]
        camera = self._params["camera_view"]
        image_bytes = self._params["image_bytes"]

        messages = [
            {"role": "system", "content": _ANALYZE_PROMPT},
            {"role": "user", "content": f"Target: '{label}'. Camera: {camera}."},
        ]

        try:
            result = self._model.query_structured(
                messages,
                ANALYZE_SCHEMA,
                images=[image_bytes],
                max_tokens=1024,
                temperature=0.0,
                phase="capture",
            )
            if not isinstance(result, dict):
                raise ValueError(f"expected a JSON object, got {type(result).__name__}")

            visible = bool(result.get("visible", False))
            bbox = result.get("bbox_1000", [0, 0, 0, 0])
            if (visible or "bbox_1000" in result) and (
                not isinstance(bbox, Sequence) or len(bbox) != 4 or not all(isinstance(v, Number) for v in bbox)
            ):
                raise ValueError("bbox_1000 must be a length-4 sequence of numbers")
            bbox = [int(v) for v in bbox]
            analysis = result.get("reasoning", "")
        except Exception as e:
            logger.warning("AnalyzeView VLM call failed: %s", e)
            self._handle.observation = f"VLM analysis failed: {e}"
            self._handle.result_data = {"observation": self._handle.observation}
            return py_trees.common.Status.FAILURE

        if not visible:
            self._handle.observation = f"Target '{label}' not visible in {camera} camera. {analysis}"
            self._handle.result_data = {"observation": self._handle.observation}
            return py_trees.common.Status.FAILURE

        self._state["bbox_1000"] = bbox
        # Safeguard-readable markers consulted by CaptureSkill.safeguard_invariant.
        # ``analyze_view_succeeded`` flips the predicate from vacuously-True
        # (pre-analysis) to "target camera must still be present in feed."
        data = self._handle.result_data
        data["analyze_view_succeeded"] = True
        data["analyze_view_camera"] = camera
        self._handle.result_data = data
        self._handle.observation = f"Target '{label}' visible in {camera}. Initial bbox: {bbox}."
        self._handle.progress = 0.3
        return py_trees.common.Status.SUCCESS


class _RefinePointing(py_trees.behaviour.Behaviour):
    """Iteratively refine the bounding box (best-effort, max 2 iterations)."""

    def __init__(
        self,
        params: Dict[str, Any],
        handle: SkillHandle,
        state: Dict[str, Any],
        model: Any,
    ) -> None:
        super().__init__(name="refine_pointing")
        self._params = params
        self._handle = handle
        self._state = state
        self._model = model

    def update(self) -> py_trees.common.Status:
        if self._handle.is_cancel_requested:
            return py_trees.common.Status.FAILURE

        if not self._state.get("bbox_1000"):
            # Nothing to refine
            return py_trees.common.Status.SUCCESS

        label = self._params["target_label"]
        image_bytes = self._params["image_bytes"]
        bbox = self._state["bbox_1000"]

        for i in range(_REFINE_MAX_ITERS):
            if self._handle.is_cancel_requested:
                return py_trees.common.Status.FAILURE

            messages = [
                {"role": "system", "content": _REFINE_PROMPT},
                {
                    "role": "user",
                    "content": f"Target: '{label}'. Current bbox (0..1000): {bbox}.",
                },
            ]

            try:
                result = self._model.query_structured(
                    messages,
                    REFINE_SCHEMA,
                    images=[image_bytes],
                    max_tokens=256,
                    temperature=0.0,
                    phase="capture",
                )
                if not isinstance(result, dict):
                    raise ValueError(f"expected a JSON object, got {type(result).__name__}")

                acceptable = bool(result.get("acceptable", False))
                new_bbox = result.get("bbox_1000", bbox)
                if (
                    not isinstance(new_bbox, Sequence)
                    or len(new_bbox) != 4
                    or not all(isinstance(v, Number) for v in new_bbox)
                ):
                    raise ValueError("bbox_1000 must be a length-4 sequence of numbers")
                new_bbox = [int(v) for v in new_bbox]
            except Exception as e:
                logger.warning("RefinePointing iteration %d failed: %s", i + 1, e)
                break  # Best-effort — keep current bbox

            if acceptable:
                logger.info("RefinePointing accepted bbox after %d iterations", i + 1)
                bbox = new_bbox
                break

            bbox = new_bbox

        self._state["bbox_1000"] = bbox
        self._handle.progress = 0.6
        return py_trees.common.Status.SUCCESS


class _SegmentMask(py_trees.behaviour.Behaviour):
    """Run the segmentation pipeline if injected. Otherwise skip."""

    def __init__(
        self,
        params: Dict[str, Any],
        handle: SkillHandle,
        state: Dict[str, Any],
        pipeline: Optional[PointingPipelineInterface],
    ) -> None:
        super().__init__(name="segment_mask")
        self._params = params
        self._handle = handle
        self._state = state
        self._pipeline = pipeline

    def update(self) -> py_trees.common.Status:
        if self._handle.is_cancel_requested:
            return py_trees.common.Status.FAILURE

        if self._pipeline is None:
            # No pipeline — bbox-only capture
            self._state["segment_result"] = None
            self._handle.progress = 0.8
            return py_trees.common.Status.SUCCESS

        bbox = self._state.get("bbox_1000")
        if not bbox:
            self._state["segment_result"] = None
            return py_trees.common.Status.SUCCESS

        try:
            seg = self._pipeline.segment(
                self._params["image_bytes"],
                list(bbox),
                self._params["target_label"],
            )
            self._state["segment_result"] = seg
        except Exception as e:
            logger.warning("Segmentation failed: %s — proceeding with bbox-only result", e)
            self._state["segment_result"] = None

        self._handle.progress = 0.85
        return py_trees.common.Status.SUCCESS


class _FinalizeCapture(py_trees.behaviour.Behaviour):
    """Build the CaptureResult and write it to the handle."""

    def __init__(
        self,
        params: Dict[str, Any],
        handle: SkillHandle,
        state: Dict[str, Any],
        context: SkillContext,
        *, debug_writer=None,
    ) -> None:
        super().__init__(name="finalize_capture")
        self._debug_writer = debug_writer
        self._params = params
        self._handle = handle
        self._state = state
        self._context = context

    def update(self) -> py_trees.common.Status:
        label = self._params["target_label"]
        camera = self._params["camera_view"]
        image_bytes = self._params["image_bytes"]
        bbox = tuple(self._state.get("bbox_1000") or (0, 0, 0, 0))
        seg = self._state.get("segment_result")

        # Image dimensions for bbox→pixel conversion
        width, height = _read_image_dims(image_bytes)

        # Target pixel: from segmentation if available, else bbox center
        if seg is not None and seg.target_pixel is not None:
            target_coords = seg.target_pixel
        else:
            cx_1000 = (bbox[0] + bbox[2]) // 2
            cy_1000 = (bbox[1] + bbox[3]) // 2
            target_coords = (
                int(cx_1000 / 1000 * width),
                int(cy_1000 / 1000 * height),
            )

        store = self._context.object_store
        cap_handle = (
            store.generate_handle(label, prefix="cap") if store is not None else f"cap_{label.replace(' ', '_')}"
        )

        cap = CaptureResult(
            handle=cap_handle,
            description=label,
            camera=camera,
            target_coords=target_coords,
            bbox_1000=tuple(int(b) for b in bbox),
            rotated_image_width=width,
            rotated_image_height=height,
            mask_backend=seg.backend if seg else "",
            mask_similarity=seg.similarity if seg else 0.0,
            mask_candidate_count=seg.candidate_count if seg else 0,
            image=image_bytes,
            skill_call_id=self._context.skill_call_id,
        )

        # Save raw + annotated debug images for Foxglove / publication
        try:
            if self._debug_writer is not None:
                self._debug_writer(image_bytes, cap.bbox_1000, label, cap_handle, self._context.main_task)
        except Exception as e:
            logger.warning("Failed to save capture debug images: %s", e)

        observation = f"Captured '{label}' from {camera} camera. Handle: '{cap_handle}'."
        self._handle.observation = observation
        self._handle.progress = 1.0
        self._handle.result_data = {
            "observation": observation,
            "new_objects": [cap],
        }
        return py_trees.common.Status.SUCCESS


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _read_image_dims(image_bytes: bytes) -> Tuple[int, int]:
    """Read (width, height) from JPEG bytes via PIL. Falls back to (1024, 768)."""
    try:
        from PIL import Image

        img = Image.open(BytesIO(image_bytes))
        return img.size  # (width, height)
    except Exception as e:
        logger.warning("Could not read image dims (%s) — using default 1024x768", e)
        return (1024, 768)


# ---------------------------------------------------------------------------
# Debug visualization — annotated capture images for Foxglove / publication
# ---------------------------------------------------------------------------

_CAPTURE_DEBUG_DIR = os.environ.get("LAM_CAPTURE_DIR", "/tmp/lam_captures")


def _save_capture_debug(
    image_bytes: bytes,
    bbox_1000: Tuple[int, int, int, int],
    label: str,
    cap_handle: str,
    task: str = "",
) -> None:
    """Save raw + bbox-annotated images to ``_CAPTURE_DEBUG_DIR``.

    Overwrites ``latest_raw.jpg`` and ``latest_annotated.jpg`` each time.
    Timestamped copies include the task number and description for easy
    identification across runs.
    """
    import os

    try:
        from PIL import Image, ImageDraw, ImageFont
    except ImportError:
        logger.warning("PIL not available — skipping capture debug images")
        return

    import re as _re
    import time as _time

    os.makedirs(_CAPTURE_DEBUG_DIR, exist_ok=True)
    _ts_str = _time.strftime("%Y%m%dT%H%M%S", _time.localtime())

    # Extract "Task N" prefix if present, else use first 20 chars of task
    _task_match = _re.match(r"(Task\s*\d+)", task) if task else None
    _task_tag = (
        _re.sub(r"\s+", "", _task_match.group(1))
        if _task_match
        else _re.sub(r"[^a-zA-Z0-9]", "_", task[:20]) if task else ""
    )
    _safe_label = _re.sub(r"[^a-zA-Z0-9_-]", "_", label)[:60]
    _safe_handle = _re.sub(r"[^a-zA-Z0-9_-]", "_", cap_handle)
    _stem = f"{_ts_str}_{_task_tag}_{_safe_handle}_{_safe_label}"

    # -- Raw --
    with open(os.path.join(_CAPTURE_DEBUG_DIR, "latest_raw.jpg"), "wb") as f:
        f.write(image_bytes)
    with open(os.path.join(_CAPTURE_DEBUG_DIR, f"{_stem}_raw.jpg"), "wb") as f:
        f.write(image_bytes)

    # -- Annotated --
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGBA")
    except Exception as e:
        logger.warning("Cannot decode image for annotation: %s", e)
        return

    w, h = img.size
    x1 = int(bbox_1000[0] / 1000 * w)
    y1 = int(bbox_1000[1] / 1000 * h)
    x2 = int(bbox_1000[2] / 1000 * w)
    y2 = int(bbox_1000[3] / 1000 * h)

    overlay = Image.new("RGBA", img.size, (0, 0, 0, 0))
    draw = ImageDraw.Draw(overlay)

    box_color = (0, 210, 200, 220)
    line_w = max(2, min(w, h) // 200)
    radius = max(4, line_w * 3)
    draw.rounded_rectangle(
        [x1, y1, x2, y2],
        radius=radius,
        outline=box_color,
        width=line_w,
    )

    # Label
    font_size = max(14, min(w, h) // 40)
    font = None
    for path in [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf",
    ]:
        try:
            font = ImageFont.truetype(path, font_size)
            break
        except Exception:
            continue
    if font is None:
        font = ImageFont.load_default()

    display_label = f"{label}  [{cap_handle}]"
    tb = draw.textbbox((0, 0), display_label, font=font)
    tw, th = tb[2] - tb[0], tb[3] - tb[1]
    pad = max(4, line_w)

    # Place label above box; fall inside if no room
    lx = x1
    ly = y1 - th - pad * 2
    if ly < 0:
        ly = y1 + pad

    draw.rounded_rectangle(
        [lx, ly, lx + tw + pad * 2, ly + th + pad * 2],
        radius=radius // 2,
        fill=(0, 0, 0, 170),
    )
    draw.text(
        (lx + pad, ly + pad),
        display_label,
        fill=(255, 255, 255, 240),
        font=font,
    )

    composite = Image.alpha_composite(img, overlay).convert("RGB")
    composite.save(
        os.path.join(_CAPTURE_DEBUG_DIR, "latest_annotated.jpg"),
        "JPEG",
        quality=95,
    )
    composite.save(
        os.path.join(_CAPTURE_DEBUG_DIR, f"{_stem}_annotated.jpg"),
        "JPEG",
        quality=95,
    )
    logger.info("Capture debug images saved to %s", _CAPTURE_DEBUG_DIR)
