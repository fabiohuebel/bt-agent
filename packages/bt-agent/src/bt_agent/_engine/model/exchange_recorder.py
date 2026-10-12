"""Full model-exchange recorder (physics-trace-recording wave 2).

Shared journaling hook for every VLM/LLM transport in the tree: the primary
``ModelBackend._post_chat`` path (lam_v2) and the legacy scene-graph
``LLMInterface`` (lam_core) both call ``begin_exchange`` / ``end_exchange``
around their own HTTP transport rather than routing traffic through a
common client. Writes two JSONL lines per call to
``<LAM_VLM_LOG_DIR>/exchanges.jsonl``:

    {"kind": "request", "exchange_id": ..., "seq": ..., "phase": ..., ...}
    {"kind": "response", "exchange_id": ..., "seq": ..., "phase": ..., ...}

The request line is written BEFORE network I/O; the response line is
written after, whether the call succeeded, errored, or timed out — every
call leaves a paired request/response, and a request with no matching
response (crash mid-call) is a legitimate "pending" entry for the offline
finalizer to flag.

Gating: recording is live only when ``LAM_VLM_LOG_DIR`` is SET in the
process environment at call time. This is checked fresh on every
``begin_exchange`` call (not cached), independent of
``bt_agent._engine.model.backend._VLM_LOG_DIR`` (which always has a value — it
defaults to ``/tmp/lam_vlm_log`` — because it drives the legacy
``vlm_calls.jsonl`` writer that must stay byte-identical to pre-wave-2
behavior). When unset: no ``exchanges.jsonl``, no ``blobs/`` directory, no
extra network-path latency beyond a single ``os.environ.get`` check —
deployed behavior is unaffected.

Images: each ``data:<mime>;base64,<...>`` image_url part in a request's
messages is decoded, SHA-256 hashed, and written once to
``<LAM_VLM_LOG_DIR>/blobs/<sha256><ext>`` (content-addressed — a repeat
image is never written twice). The message part is replaced in the logged
copy by an ``image_ref`` carrying ``{mime, bytes, sha256, blob_path}``; the
same refs are collected in call order under the request line's
``image_refs``. Non-``data:`` URLs and undecodable images are left
unresolved and recorded as a ``limitations`` entry — the request is never
silently claimed complete when its images could not be preserved.

Credentials: this module never receives headers — only request bodies and
provider responses — and additionally scrubs any key that looks like a
credential (``api_key``, ``authorization``, ``secret*``, ``password``,
``token``/``access_token``/``x-api-key``, ...) recursively before every
write, in case a provider ever echoes one back. Matching is on the whole
key, not a substring — a naive ``"token" in key`` scrub also caught
``max_tokens``/``prompt_tokens``/``completion_tokens``/``total_tokens``,
destroying usage and generation-setting data in every journal line. Any key
ending in ``_tokens``/``tokens`` (a token-count field) is never scrubbed.
"""

from __future__ import annotations

import base64
import hashlib
import json
import logging
import mimetypes
import os
import re
import threading
import time
import uuid
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger(__name__)

_seq_lock = threading.Lock()
_seq_counter = 0

_DATA_URL_RE = re.compile(r"^data:(?P<mime>[\w/+.-]+);base64,(?P<data>.*)$", re.DOTALL)

# Exact (whole-key, case/separator-insensitive) credential key names. Deliberately
# NOT a substring match: "token" as a substring also matches max_tokens/
# prompt_tokens/completion_tokens/total_tokens, which are usage/generation-setting
# fields, never credentials.
_CREDENTIAL_KEY_EXACT = frozenset({
    "apikey",
    "api_key",
    "authorization",
    "bearer",
    "password",
    "token",
    "access_token",
    "auth_token",
    "id_token",
    "refresh_token",
    "x_api_key",
})


def _is_credential_key(key: str) -> bool:
    """Whole-key credential predicate — never a substring match.

    Token-count fields (``max_tokens``, ``max_completion_tokens``,
    ``prompt_tokens``, ``completion_tokens``, ``total_tokens``, bare
    ``tokens``) are excluded up front, before the ``secret*`` prefix or the
    exact-name set are even considered, so no future addition to either can
    accidentally reclaim them.
    """
    normalized = key.strip().lower().replace("-", "_")
    if normalized == "tokens" or normalized.endswith("_tokens") or "max_tokens" in normalized:
        return False
    if normalized in _CREDENTIAL_KEY_EXACT:
        return True
    return normalized.startswith("secret")


_GENERATION_SETTING_KEYS = (
    "max_tokens",
    "temperature",
    "top_p",
    "seed",
    "response_format",
    "chat_template_kwargs",
    "stream",
)


def _next_seq() -> int:
    global _seq_counter  # noqa: PLW0603
    with _seq_lock:
        _seq_counter += 1
        return _seq_counter


def _recording_log_dir() -> Optional[str]:
    """Fresh (uncached) read of ``LAM_VLM_LOG_DIR`` — the wave-2 gate.

    Unset -> ``None`` -> every function below becomes a no-op. This is
    intentionally independent of ``backend._VLM_LOG_DIR``'s import-time
    default so the new exchanges.jsonl/blobs artifacts never appear for a
    deployment that hasn't opted into recording.
    """
    return os.environ.get("LAM_VLM_LOG_DIR") or None


@dataclass
class ExchangeHandle:
    """Opaque handle returned by :func:`begin_exchange`.

    Pass it to :func:`end_exchange` to complete the pair. Always carries a
    real ``exchange_id``/``seq`` (cheap to mint) regardless of whether
    recording is enabled, so callers (e.g. the actor's skill-selection
    path) can always link a decision to its exchange.
    """

    exchange_id: str
    seq: int
    phase: str
    t_start: float
    request_hash: Optional[str]
    _recording: bool
    _log_dir: Optional[str]


def _scrub_credentials(obj: Any) -> Any:
    if isinstance(obj, dict):
        out: Dict[str, Any] = {}
        for k, v in obj.items():
            if isinstance(k, str) and _is_credential_key(k):
                out[k] = "<scrubbed>"
            else:
                out[k] = _scrub_credentials(v)
        return out
    if isinstance(obj, list):
        return [_scrub_credentials(v) for v in obj]
    return obj


def _hash_request(payload: Dict[str, Any]) -> str:
    """Stable hash of the logical request (same payload -> same hash)."""
    canon = json.dumps(payload, sort_keys=True, default=str)
    return hashlib.sha256(canon.encode("utf-8")).hexdigest()


def _generation_settings(payload: Dict[str, Any]) -> Dict[str, Any]:
    return {k: payload[k] for k in _GENERATION_SETTING_KEYS if k in payload}


def _write_blob(blob_dir: str, raw: bytes, mime: str) -> Dict[str, Any]:
    sha256 = hashlib.sha256(raw).hexdigest()
    ext = mimetypes.guess_extension(mime) or ".bin"
    if ext == ".jpe":
        ext = ".jpg"
    blob_path = os.path.join(blob_dir, f"{sha256}{ext}")
    if not os.path.exists(blob_path):
        os.makedirs(blob_dir, exist_ok=True)
        tmp_path = f"{blob_path}.tmp{os.getpid()}-{threading.get_ident()}"
        with open(tmp_path, "wb") as f:
            f.write(raw)
        os.replace(tmp_path, blob_path)  # atomic within the blob dir
    return {"mime": mime, "bytes": len(raw), "sha256": sha256, "blob_path": blob_path}


def _extract_and_blob_images(
    messages: List[Dict[str, Any]],
    blob_dir: str,
) -> Tuple[List[Dict[str, Any]], List[Dict[str, Any]], List[str]]:
    """Replace inline base64 images with content-addressed blob refs.

    Returns ``(messages_with_refs, ordered_image_refs, limitations)``.
    """
    out_messages: List[Dict[str, Any]] = []
    refs: List[Dict[str, Any]] = []
    limitations: List[str] = []

    for msg in messages:
        content = msg.get("content") if isinstance(msg, dict) else None
        if not isinstance(content, list):
            out_messages.append(msg)
            continue

        new_content: List[Any] = []
        for part in content:
            if not (isinstance(part, dict) and part.get("type") == "image_url"):
                new_content.append(part)
                continue

            url = ((part.get("image_url") or {}).get("url")) or ""
            match = _DATA_URL_RE.match(url)
            if not match:
                new_content.append({"type": "image_ref", "ref": None})
                limitations.append("image_url was not an inline data: URL — bytes not retained")
                continue

            try:
                raw = base64.b64decode(match.group("data"))
            except Exception as e:  # noqa: BLE001
                new_content.append({"type": "image_ref", "ref": None})
                limitations.append(f"image base64 decode failed ({e}) — bytes not retained")
                continue

            ref = _write_blob(blob_dir, raw, match.group("mime"))
            refs.append(ref)
            new_content.append({"type": "image_ref", "ref": ref})

        out_messages.append({**msg, "content": new_content})

    return out_messages, refs, limitations


def describe_error(e: BaseException) -> Dict[str, Any]:
    """Structured transport/provider error: type, status, message."""
    status = None
    response = getattr(e, "response", None)
    if response is not None:
        status = getattr(response, "status_code", None)
    return {"type": type(e).__name__, "status": status, "message": str(e)}


def _append_jsonl(log_dir: str, entry: Dict[str, Any]) -> None:
    os.makedirs(log_dir, exist_ok=True)
    path = os.path.join(log_dir, "exchanges.jsonl")
    line = json.dumps(_scrub_credentials(entry), default=str)
    with open(path, "a") as f:
        f.write(line + "\n")


def begin_exchange(
    *,
    phase: str,
    payload: Dict[str, Any],
    episode_context: Optional[Dict[str, Any]] = None,
    retry_attempt: int = 0,
) -> ExchangeHandle:
    """Journal the request line BEFORE network I/O. Always returns a handle.

    ``payload`` is the exact OpenAI-format chat-completions body (model,
    messages, tools, tool_choice, generation settings) — the "exact logical
    request after prompt construction and image insertion".
    """
    # log_dir is resolved FIRST: an unset LAM_VLM_LOG_DIR (the deployed
    # default) must return a no-op handle with nothing beyond the env check
    # — no serialization, no hashing, no allocation — per the module's
    # "single os.environ.get" no-op contract.
    log_dir = _recording_log_dir()
    exchange_id = uuid.uuid4().hex
    seq = _next_seq()
    t_start = time.time()
    handle = ExchangeHandle(
        exchange_id=exchange_id,
        seq=seq,
        phase=phase,
        t_start=t_start,
        request_hash=None,
        _recording=bool(log_dir),
        _log_dir=log_dir,
    )
    if not handle._recording:
        return handle

    try:
        limitations: List[str] = []
        try:
            request_hash: Optional[str] = _hash_request(payload)
        except Exception:  # noqa: BLE001 — hashing must never fail the call
            request_hash = None
            limitations.append("failed to hash request payload")
        handle.request_hash = request_hash
        blob_dir = os.path.join(log_dir, "blobs")
        messages, image_refs, image_limitations = _extract_and_blob_images(payload.get("messages", []) or [], blob_dir)
        limitations.extend(image_limitations)
        entry: Dict[str, Any] = {
            "kind": "request",
            "exchange_id": exchange_id,
            "seq": seq,
            "phase": phase,
            "ts": t_start,
            "retry_attempt": retry_attempt,
            "request_hash": request_hash,
            "model": payload.get("model"),
            "messages": messages,
            "tools": payload.get("tools"),
            "tool_choice": payload.get("tool_choice"),
            "generation_settings": _generation_settings(payload),
            "image_refs": image_refs,
            "completeness": "incomplete" if limitations else "complete",
        }
        if episode_context is not None:
            entry["episode_context"] = episode_context
        if limitations:
            entry["limitations"] = limitations
        _append_jsonl(log_dir, entry)
    except Exception:  # noqa: BLE001 — recording must never break the call path
        logger.debug("Failed to journal model request", exc_info=True)
    return handle


def end_exchange(
    handle: ExchangeHandle,
    *,
    response: Optional[Dict[str, Any]] = None,
    error: Optional[Dict[str, Any]] = None,
) -> None:
    """Journal the response line AFTER the call completes, succeeds or not.

    Exactly one of ``response`` (raw provider JSON) / ``error`` (structured
    transport/provider error — see :func:`describe_error`) should be given.
    """
    if not handle._recording:
        return

    try:
        t_end = time.time()
        entry: Dict[str, Any] = {
            "kind": "response",
            "exchange_id": handle.exchange_id,
            "seq": handle.seq,
            "phase": handle.phase,
            "ts": t_end,
            "t_start": handle.t_start,
            "t_end": t_end,
            "duration_s": t_end - handle.t_start,
            "request_hash": handle.request_hash,
        }
        if error is not None:
            entry["error"] = error
            entry["completeness"] = "error"
        else:
            message: Dict[str, Any] = {}
            choices = (response or {}).get("choices") or []
            if choices:
                message = choices[0].get("message") or {}
            entry["response"] = response
            entry["content"] = message.get("content")
            entry["reasoning"] = message.get("reasoning_content")
            entry["tool_calls"] = message.get("tool_calls")
            entry["usage"] = (response or {}).get("usage")
            entry["finish_reason"] = choices[0].get("finish_reason") if choices else None
            entry["completeness"] = "complete"
        _append_jsonl(handle._log_dir, entry)
    except Exception:  # noqa: BLE001 — recording must never break the call path
        logger.debug("Failed to journal model response", exc_info=True)
