"""EvaluateCompletion — LLM verification of milestone completion (3-way verdict).

Runs the evaluator in a background thread, parses an :class:`EvaluatorResult`
with verdict ∈ {pass, amend, replan}, applies the amend-cap promotion
(amend→replan when ``AMEND_COUNT >= AMEND_CAP``), and writes the resolved
verdict + correction to the blackboard.

The evaluator uses a two-step flow:
  1. **Evidence selection** — VLM sees the actor trace + K store summaries
     and decides which store entries (if any) to inspect.
  2. **Verdict** — VLM sees the resolved evidence (images, metadata) and
     emits ``(d, δ)``.

If the evaluator skips inspection (Step 1 returns ``skip_inspection=True``),
only one VLM call is made (same cost as before).

Always returns SUCCESS once the verdict is written — downstream verdict-route
nodes (``CheckVerdictPass`` / ``CheckVerdictAmend`` / ``CheckVerdictReplan``)
take it from there.
"""

from __future__ import annotations

import logging
import threading
from typing import Any, Dict, List, Optional

import py_trees
from bt_agent._engine.agent.blackboard_keys import BB
from bt_agent._engine.agent.lam_agent.evaluator import EvaluatorResult, Verdict
from bt_agent._engine.bt.nodes._conversation import extract_tool_calls, format_for_prompt

logger = logging.getLogger(__name__)


def _build_skill_descriptions(registry: Any) -> str:
    """Build a newline-separated list of skill name + description."""
    if registry is None:
        return ""
    return "\n".join(f"- {s.name}: {s.description}" for s in registry.all_skills().values())


def _is_image_array(val: Any) -> bool:
    """Check if a value looks like a numpy image array."""
    try:
        import numpy as np

        return isinstance(val, np.ndarray) and val.ndim in (2, 3)
    except ImportError:
        return False


def _encode_image(img: Any) -> bytes:
    """Encode an image to JPEG bytes. Handles bytes passthrough + np.ndarray."""
    if isinstance(img, bytes):
        return img
    try:
        import cv2

        _, buf = cv2.imencode(".jpg", img)
        return buf.tobytes()
    except Exception:
        return b""


def _resolve_evidence(store: Any, handles: List[str]) -> Optional[Dict[str, Any]]:
    """Resolve requested handles from K into displayable evidence.

    For each object: iterate its attributes (beyond handle/label).
    - bytes / np.ndarray → treat as image, encode for VLM
    - everything else → format as "attr: repr(value)" text

    The framework doesn't know what fields exist — it just displays
    whatever the producer put on the object.
    """
    if store is None or not handles:
        return None
    images: List[bytes] = []
    text_blocks: List[str] = []
    for h in handles:
        obj = store.get(h)
        if obj is None:
            text_blocks.append(f"{h}: not found in store")
            continue
        block_lines = [f"## {obj.to_llm_summary()}"]
        for attr in sorted(vars(obj)):
            if attr.startswith("_") or attr in ("handle", "label"):
                continue
            val = getattr(obj, attr)
            if isinstance(val, bytes):
                images.append(val)
                block_lines.append(f"  {attr}: [image attached]")
            elif _is_image_array(val):
                encoded = _encode_image(val)
                if encoded:
                    images.append(encoded)
                block_lines.append(f"  {attr}: [image attached]")
            else:
                val_repr = repr(val)
                if len(val_repr) > 200:
                    val_repr = val_repr[:200] + "..."
                block_lines.append(f"  {attr}: {val_repr}")
        text_blocks.append("\n".join(block_lines))
    if not images and not text_blocks:
        return None
    return {
        "images": images or None,
        "extra_text": "\n\n".join(text_blocks) or None,
    }


class EvaluateCompletion(py_trees.behaviour.Behaviour):
    """Ask the evaluator for a 3-way verdict and write it to the blackboard.

    Lifecycle:
        initialise() — gather milestone + skill result, build compressed
                        actor trace, launch evaluator thread (two-step:
                        evidence selection from K, then verdict)
        update()     — poll thread; RUNNING until done; write LAST_VERDICT +
                       AMEND_NOTE; apply cap promotion; SUCCESS once written
        terminate()  — abandon result if preempted
    """

    def __init__(self) -> None:
        super().__init__(name="EvaluateCompletion")
        self.blackboard = self.attach_blackboard_client()
        self.blackboard.register_key(key=BB.MILESTONES, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.CURRENT_MILESTONE_IDX, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.SKILL_HANDLE, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.MODEL_BACKEND, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.PENDING_EVENTS, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.SKILL_REASONING, access=py_trees.common.Access.READ)

        # Conversation history for compressed actor trace
        self.blackboard.register_key(key=BB.CONVERSATION, access=py_trees.common.Access.READ)
        # Scenario context + skill registry + object store
        self.blackboard.register_key(key=BB.SCENARIO, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.SKILL_REGISTRY, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.OBJECT_STORE, access=py_trees.common.Access.READ)

        self.blackboard.register_key(key=BB.EVENT_SINK, access=py_trees.common.Access.READ)  # OBS-DROP

        # Three-way verdict bookkeeping (paper §5 I7)
        self.blackboard.register_key(key=BB.LAST_VERDICT, access=py_trees.common.Access.WRITE)
        self.blackboard.register_key(key=BB.AMEND_COUNT, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.AMEND_CAP, access=py_trees.common.Access.READ)
        self.blackboard.register_key(key=BB.AMEND_NOTE, access=py_trees.common.Access.WRITE)
        # Step 5: prompt variant
        self.blackboard.register_key(key=BB.PROMPT_VARIANT, access=py_trees.common.Access.READ)
        # Agent config
        self.blackboard.register_key(key=BB.AGENT_CONFIG, access=py_trees.common.Access.READ)

        self._thread: Optional[threading.Thread] = None
        self._result: Optional[EvaluatorResult] = None
        # Race-free abort handles — see plan_task.py for the rationale.
        self._epoch: int = 0
        self._cancelled: bool = False

    def initialise(self) -> None:
        model = self.blackboard.get(BB.MODEL_BACKEND)
        milestones = self.blackboard.get(BB.MILESTONES)
        idx = self.blackboard.get(BB.CURRENT_MILESTONE_IDX)
        milestone = milestones[idx]

        # Gather the latest skill result for context
        handle = self._bb_get(BB.SKILL_HANDLE)
        skill_observation = handle.observation if handle else ""

        # Actor's claim reasoning (text response that triggered evaluation)
        claim_reasoning = self._bb_get(BB.SKILL_REASONING, "") or ""

        # Compressed actor trace h̃^act_{m_i} (paper §5)
        conversation = self._bb_get(BB.CONVERSATION, []) or []
        calls = extract_tool_calls(conversation)
        actor_trace = format_for_prompt(calls)

        # K store summaries + store reference for evidence resolution
        store = self._bb_get(BB.OBJECT_STORE)
        store_summaries = ""
        if store:
            summaries = store.all_llm_summaries()
            if summaries:
                store_summaries = "\n".join(summaries)

        # Scenario context + skill descriptions for evaluator prompt
        scenario = self._bb_get(BB.SCENARIO, "") or ""
        registry = self._bb_get(BB.SKILL_REGISTRY)
        skill_descriptions = _build_skill_descriptions(registry)

        event_sink = self._bb_get(BB.EVENT_SINK)  # OBS-DROP
        prompt_variant = self._bb_get(BB.PROMPT_VARIANT) or "tiamat_agent_submit_v5"
        config = self._bb_get(BB.AGENT_CONFIG)

        self._cancelled = False
        self._epoch += 1
        my_epoch = self._epoch
        self._result = None
        self._thread = threading.Thread(
            target=self._evaluate_in_background,
            args=(
                my_epoch,
                model,
                milestone,
                skill_observation,
                claim_reasoning,
                actor_trace,
                store,
                store_summaries,
                scenario,
                skill_descriptions,
                event_sink,
                prompt_variant,
                config,
            ),
            daemon=True,
        )
        self._thread.start()

    def _evaluate_in_background(
        self,
        my_epoch: int,
        model: Any,
        milestone: Dict,
        skill_observation: str,
        claim_reasoning: str,
        actor_trace: str,
        store: Any,
        store_summaries: str,
        scenario: str,
        skill_descriptions: str,
        event_sink: Any = None,  # OBS-DROP
        prompt_variant: str = "tiamat_agent_submit_v5",
        config: Any = None,
    ) -> None:
        try:
            from bt_agent._engine.agent.lam_agent.evaluator import evaluate_completion

            eval_kwargs: Dict[str, Any] = {}
            if config is not None:
                ec = config.evaluator
                eval_kwargs["evidence_temperature"] = ec.evidence_temperature
                eval_kwargs["verdict_temperature"] = ec.verdict_temperature
                eval_kwargs["evidence_max_tokens"] = ec.evidence_max_tokens
                eval_kwargs["verdict_max_tokens"] = ec.verdict_max_tokens

            result = evaluate_completion(
                model,
                milestone=milestone,
                skill_observation=skill_observation,
                claim_reasoning=claim_reasoning,
                actor_trace=actor_trace,
                store=store,
                store_summaries=store_summaries,
                scenario=scenario,
                skill_descriptions=skill_descriptions,
                event_sink=event_sink,  # OBS-DROP
                prompt_variant=prompt_variant,
                **eval_kwargs,
            )
        except Exception as e:
            # Defensive: any uncaught failure escalates to REPLAN
            result = EvaluatorResult(
                verdict=Verdict.REPLAN,
                reasoning=f"Evaluation thread failed: {e}",
                correction=None,
            )
        # Only commit if we are still the live generation (see plan_task.py).
        if my_epoch == self._epoch and not self._cancelled:
            self._result = result

    def update(self) -> py_trees.common.Status:
        if self._result is None:
            return py_trees.common.Status.RUNNING

        result = self._result

        # Apply amend-cap promotion: if the evaluator wants AMEND but we've
        # already hit the cap, escalate to REPLAN. Routing logic downstream
        # never has to know about the cap.
        amend_count = self._bb_get(BB.AMEND_COUNT, 0)
        amend_cap = self._bb_get(BB.AMEND_CAP, 2)
        cap_promoted = False

        if result.verdict == Verdict.AMEND and amend_count >= amend_cap:
            verdict = Verdict.REPLAN
            reasoning = (
                f"Amend cap ({amend_cap}) reached after {amend_count} attempts. "
                f"Original AMEND reasoning: {result.reasoning}"
            )
            correction = None
            cap_promoted = True
        else:
            verdict = result.verdict
            reasoning = result.reasoning
            correction = result.correction if verdict == Verdict.AMEND else None

        # Write the resolved verdict + δ (if AMEND)
        self.blackboard.set(BB.LAST_VERDICT, verdict)
        self.blackboard.set(BB.AMEND_NOTE, correction)

        # Emit event
        events = self._bb_get(BB.PENDING_EVENTS, [])
        events.append({
            "kind": "completion_evaluated",
            "data": {
                "verdict": verdict.value,
                "reasoning": reasoning,
                "correction": correction or "",
                "cap_promoted": cap_promoted,
                "amend_count": amend_count,
                "inspected_handles": result.inspected_handles or [],
                "skipped_inspection": result.skipped_inspection,
            },
        })
        self.blackboard.set(BB.PENDING_EVENTS, events)

        self.feedback_message = f"verdict={verdict.value}" + (" (cap-promoted)" if cap_promoted else "")
        return py_trees.common.Status.SUCCESS

    def terminate(self, new_status: py_trees.common.Status) -> None:
        # Drop in-flight thread's result on the floor (see plan_task.py).
        self._cancelled = True
        self._result = None

    def _bb_get(self, key: str, default: Any = None) -> Any:
        try:
            return self.blackboard.get(key)
        except KeyError:
            return default
