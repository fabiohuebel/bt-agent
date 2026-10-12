"""ModelBackend — thin, model-agnostic VLM/LLM client.

This is NOT agent logic. It is the gate between the VLM/LLM server and
everything else. It exposes three generic primitives:

    query()            — text/vision completion → str
    query_structured() — structured output (JSON schema) → dict
    query_tool_call()  — tool calling → tool name + args

The backend handles:
    - HTTP transport (OpenAI-compatible API)
    - Image encoding (JPEG → base64)
    - Retry / error handling
    - API format differences (openai, anthropic, ollama)

The backend does NOT handle:
    - Prompt construction (that's the agent's job)
    - Response interpretation (that's the caller's job)
    - Conversation history (that's the agent's job)

Configuration via dict or YAML:

    model = ModelBackend({
        "endpoint": "http://localhost:8000/v1",
        "model_name": "Qwen/Qwen3-VL-8B-Instruct",
        "api_format": "openai",
        "max_tokens": 4096,
        "temperature": 0.0,
        "seed": 42,  # optional; pins vLLM sampling PRNG
    })
"""

from __future__ import annotations

import base64
import contextlib
import json
import logging
import os
import re
import time
import uuid
from typing import Any, Callable, Dict, List, Optional, Tuple

import requests
from bt_agent._engine.model import exchange_recorder

logger = logging.getLogger(__name__)

_VLM_LOG_DIR = os.environ.get("LAM_VLM_LOG_DIR", "/tmp/lam_vlm_log")
_vlm_call_counter = 0

# Env-var prefixes captured into each run's run_start config record.
# Keep these stable — the trace viewer reads them as-is.
_CONFIG_ENV_PREFIXES = ("LAM_", "VLLM_", "HUMAN_BASELINE_", "OPENCLAW_")


def _strip_b64(obj: Any) -> Any:
    """Strip base64 images from a logged payload."""
    if isinstance(obj, dict):
        if obj.get("type") == "image_url":
            return {"type": "image_url", "image_url": {"url": "<base64-stripped>"}}
        return {k: _strip_b64(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_strip_b64(v) for v in obj]
    return obj


def _dump_vlm_exchange(
    payload: Dict[str, Any],
    response: Dict[str, Any],
    tags: Optional[Dict[str, Any]] = None,
) -> None:
    """Append request + response to a JSONL file in ``_VLM_LOG_DIR``.

    ``tags`` (optional) carries run-identification fields and experiment
    identity stamps. Written into the entry so the trace viewer can
    group/filter calls without parsing prompts.
    """
    global _vlm_call_counter  # noqa: PLW0603
    try:
        os.makedirs(_VLM_LOG_DIR, exist_ok=True)
        _vlm_call_counter += 1

        entry: Dict[str, Any] = {
            "call_id": _vlm_call_counter,
            "ts": time.time(),
        }
        if tags:
            entry.update(tags)
        entry["request"] = _strip_b64(payload)
        entry["response"] = response

        path = os.path.join(_VLM_LOG_DIR, "vlm_calls.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        logger.debug("Failed to dump VLM exchange: %s", e)


def _write_run_start(
    run_id: str,
    agent_tier: str,
    config: Dict[str, Any],
    tags: Optional[Dict[str, str]] = None,
) -> None:
    """Write a ``kind: "run_start"`` record to the VLM log.

    Snapshots the runtime config (env vars + backend settings) and optional
    experiment identity tags once per backend instance so traces are
    self-describing.
    """
    try:
        os.makedirs(_VLM_LOG_DIR, exist_ok=True)
        entry = {
            "kind": "run_start",
            "run_id": run_id,
            "agent_tier": agent_tier,
            "ts": time.time(),
            "config": config,
        }
        if tags:
            entry.update(tags)
        path = os.path.join(_VLM_LOG_DIR, "vlm_calls.jsonl")
        with open(path, "a") as f:
            f.write(json.dumps(entry, default=str) + "\n")
    except Exception as e:
        logger.debug("Failed to dump run_start: %s", e)


def _snapshot_env_config(backend_config: Dict[str, Any]) -> Dict[str, Any]:
    """Capture relevant env vars + backend config for the run_start record."""
    env_snapshot = {k: v for k, v in os.environ.items() if any(k.startswith(p) for p in _CONFIG_ENV_PREFIXES)}
    # Backend config (without secrets) — useful when env wasn't used.
    backend_snapshot = {
        k: v for k, v in backend_config.items() if k not in ("api_key",) and not isinstance(v, (bytes, bytearray))
    }
    return {**env_snapshot, "_backend": backend_snapshot}


def _repair_truncated_json(text: str) -> str:
    """Best-effort repair of truncated/malformed JSON.

    Single pass tracks string/escape state and the bracket stack (ignoring
    brackets inside strings); then closes an unterminated string, drops a
    dangling trailing comma, and appends the unclosed ``{}``/``[]`` closers
    in reverse order. Returns the repaired text (still may be invalid — the
    caller re-parses and decides).
    """
    stack: List[str] = []
    in_string = False
    escape = False
    for ch in text:
        if in_string:
            if escape:
                escape = False
            elif ch == "\\":
                escape = True
            elif ch == '"':
                in_string = False
            continue
        if ch == '"':
            in_string = True
        elif ch in "{[":
            stack.append(ch)
        elif ch in "}]" and stack:
            stack.pop()

    repaired = text
    if in_string:
        repaired += '"'
    repaired = re.sub(r",\s*$", "", repaired.rstrip())  # drop a dangling comma
    for opener in reversed(stack):
        repaired += "}" if opener == "{" else "]"
    return repaired


def _loads_salvage_json(content: Optional[str]) -> Dict[str, Any]:
    """Parse structured-output JSON, salvaging truncated responses.

    Tries a straight ``json.loads`` first; on failure attempts a single
    repair pass (see :func:`_repair_truncated_json`) and re-parses. Raises
    the original decode error if the repaired text still will not parse, or
    if there is no string content at all.
    """
    try:
        return json.loads(content)
    except (json.JSONDecodeError, TypeError) as e:
        if isinstance(content, str):
            repaired = _repair_truncated_json(content)
            try:
                result = json.loads(repaired)
            except json.JSONDecodeError:
                pass
            else:
                # LOUD on purpose: a silent salvage here is how a plan becomes
                # 0 milestones and the agent runs planless. The recovered object
                # may be INCOMPLETE.
                logger.warning(
                    "!!! query_structured SALVAGE: truncated JSON repaired (%s) — "
                    "recovered %d of %d chars; parsed object may be INCOMPLETE "
                    "(likely max_tokens truncation → possible 0-milestone plan). "
                    "Raise the caller's max_tokens.",
                    e,
                    len(repaired),
                    len(content),
                )
                return result
        raise


class ModelBackend:
    """Model-agnostic VLM/LLM client.

    Supports OpenAI-compatible APIs (vLLM, OpenAI, Azure, Together, etc.).
    """

    def __init__(
        self,
        config: Dict[str, Any],
        *,
        identity_tags: Optional[Dict[str, str]] = None,
    ) -> None:
        self.endpoint = config["endpoint"].rstrip("/")
        self.model_name = config["model_name"]
        self.api_format = config.get("api_format", "openai")
        self.max_tokens = config.get("max_tokens", 4096)
        self.temperature = config.get("temperature", 0.0)
        self.api_key = config.get("api_key", "EMPTY")
        # Transport timeouts: requests accepts a (connect, read) tuple at the
        # post site. Serving topology is DECLARED via VLLM_MODE, not inferred
        # from the URL — an ssh-tunnel fallback makes a remote server look
        # like localhost. local: connect is instant-or-dead. remote: a few
        # seconds of tunnel grace. Read stays the model-speed knob
        # (VLLM_TIMEOUT_S via config "timeout").
        read_timeout = config.get("timeout", 120)
        mode = str(config.get("mode") or os.environ.get("VLLM_MODE") or "remote").strip().lower()
        if mode not in ("local", "remote"):
            logger.warning("Unknown VLLM_MODE %r — using 'remote' preset", mode)
            mode = "remote"
        connect_env = os.environ.get("VLLM_CONNECT_TIMEOUT_S")
        connect_timeout = config.get("connect_timeout") or (int(connect_env) if connect_env else None)
        if connect_timeout is None:
            connect_timeout = 2 if mode == "local" else 10
        self.timeout = (connect_timeout, read_timeout)

        # Optional vLLM/OpenAI sampling seed. When set, every chat-completion
        # payload includes "seed", making the PRNG reproducible per-call on
        # a fixed server instance. None = server randomizes (default).
        self.seed: Optional[int] = config.get("seed")

        # vLLM chat_template_kwargs — controls Qwen3.5 thinking mode.
        # Set enable_thinking=False to disable <think> tokens (faster,
        # avoids tool-calling conflicts).  None = use model default.
        self._enable_thinking: Optional[bool] = config.get("enable_thinking")

        # Run identification — tagged into every VLM log entry so the
        # trace viewer can split/filter without parsing prompts.
        self._run_id: str = uuid.uuid4().hex[:8]
        self._task_id: str = ""
        self._agent_tier: str = config.get("agent_tier", os.environ.get("LAM_AGENT_TIER", "unknown"))
        self._identity_tags = dict(identity_tags or {})

        # Optional: the ROS bridge registers a callable returning the current
        # episode-context stamp (or None) so every vlm_calls.jsonl entry
        # carries the context active when that call was made. Unset (the
        # default, e.g. baseline/non-ROS callers) => no "episode_context"
        # key at all, i.e. byte-identical to pre-wave-1 journal entries.
        self._episode_context_provider: Optional[Callable[[], Optional[Dict[str, Any]]]] = None

        # physics-trace-recording wave 2: exchange_id of the most recent
        # _post_chat call. The actor's SELECT_SKILL path reads this right
        # after its final query_tool_call() to attach the deciding exchange
        # to the skill_selected event — cheaper than threading exchange_id
        # through every reasoning-mode return path.
        self.last_exchange_id: Optional[str] = None

        self._record_run_start(config)

    def _record_run_start(self, config):
        # Historical diagnostics; the public client overrides this hook.
        _write_run_start(
            self._run_id,
            self._agent_tier,
            _snapshot_env_config(config),
            tags=self._identity_tags,
        )

    def _record_response(self, payload, result, tags):
        _dump_vlm_exchange(payload, result, tags=tags)

    def _record_error(self, exchange, error):
        if isinstance(error, requests.exceptions.RequestException):
            logger.error("Model request failed: %s", error)
        exchange_recorder.end_exchange(exchange, error=exchange_recorder.describe_error(error))

    def set_task(self, task_text: str) -> None:
        """Update the current task tag. Call when a new task arrives."""
        slug = re.sub(r"[^a-z0-9]+", "-", (task_text or "").lower())[:48].strip("-")
        ts = time.strftime("%Y%m%dT%H%M%S")
        self._task_id = f"{slug}_{ts}" if slug else f"task_{ts}"

    def set_episode_context_provider(
        self,
        provider: Optional[Callable[[], Optional[Dict[str, Any]]]],
    ) -> None:
        """Register a callable returning the current episode-context stamp.

        Called fresh on every ``_tags()`` invocation (i.e. per VLM call) so
        vlm_calls.jsonl carries the context that was active when each call
        was made. ``None`` (the default) omits the "episode_context" key
        entirely — pre-wave-1 callers see byte-identical journal entries.
        """
        self._episode_context_provider = provider

    def _tags(self) -> Dict[str, Any]:
        """Run and experiment tags written into every VLM log entry."""
        tags: Dict[str, Any] = {
            **self._identity_tags,
            "run_id": self._run_id,
            "agent_tier": self._agent_tier,
            "task_id": self._task_id,
        }
        if self._episode_context_provider is not None:
            with contextlib.suppress(Exception):
                tags["episode_context"] = self._episode_context_provider()
        return tags

    # -- Generic primitives ----------------------------------------------------

    def query(
        self,
        messages: List[Dict[str, Any]],
        images: Optional[List[bytes]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        phase: str = "other",
    ) -> str:
        """Text/vision completion. Returns the model's text response.

        Args:
            messages: Chat messages in OpenAI format.
            images: Optional JPEG bytes to include as vision input.
            max_tokens: Override default max_tokens.
            temperature: Override default temperature.
            phase: Calling phase for the exchange journal (planner, actor,
                evaluator, capture, surround_survey, scene_graph, other).

        Returns:
            The model's text response.
        """
        if images:
            messages = self._inject_images(messages, images)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
        }
        if self.seed is not None:
            payload["seed"] = self.seed

        response = self._post_chat(payload, phase=phase)
        return response["choices"][0]["message"]["content"]

    def query_structured(
        self,
        messages: List[Dict[str, Any]],
        schema: Dict[str, Any],
        images: Optional[List[bytes]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        phase: str = "other",
    ) -> Dict[str, Any]:
        """Structured output via JSON schema.

        Instructs the model to return JSON matching the given schema.
        Falls back to prompting + parsing if the API doesn't support
        native structured output.

        Args:
            messages: Chat messages in OpenAI format.
            schema: JSON schema for the expected response.
            images: Optional JPEG bytes.
            phase: Calling phase for the exchange journal (planner, actor,
                evaluator, capture, surround_survey, scene_graph, other).

        Returns:
            Parsed dict matching the schema.
        """
        if images:
            messages = self._inject_images(messages, images)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "response_format": {
                "type": "json_schema",
                "json_schema": {"name": "response", "schema": schema},
            },
        }
        if self.seed is not None:
            payload["seed"] = self.seed

        response = self._post_chat(payload, phase=phase)
        message = response["choices"][0]["message"]
        content = message.get("content")
        if content is None:
            # qwen3 thinking mode emits the structured payload in
            # reasoning_content when the content field comes back null.
            content = message.get("reasoning_content")
        parsed = _loads_salvage_json(content)
        if not isinstance(parsed, dict):
            raise ValueError(f"query_structured expected a JSON object, got {type(parsed).__name__}")
        return parsed

    def query_tool_call(
        self,
        messages: List[Dict[str, Any]],
        tools: List[Dict[str, Any]],
        images: Optional[List[bytes]] = None,
        max_tokens: Optional[int] = None,
        temperature: Optional[float] = None,
        tool_choice: str = "auto",
        phase: str = "other",
    ) -> Tuple[Optional[str], Dict[str, Any], Dict[str, Any]]:
        """Tool calling. Returns the selected tool name, args, and raw response.

        Args:
            messages: Chat messages in OpenAI format.
            tools: Tool definitions in OpenAI format.
            images: Optional JPEG bytes.
            tool_choice: ``"required"`` forces a tool call every turn.
                ``"auto"`` lets the model choose — a text-only response
                returns ``(None, {}, raw_response)``.
            phase: Calling phase for the exchange journal (planner, actor,
                evaluator, capture, surround_survey, scene_graph, other).

        Returns:
            (tool_name, tool_args, raw_response). tool_name is ``None``
            when the model responded with text instead of a tool call.

        Raises:
            ValueError: If tool_choice="required" and no tool call.
        """
        if images:
            messages = self._inject_images(messages, images)

        payload = {
            "model": self.model_name,
            "messages": messages,
            "max_tokens": max_tokens or self.max_tokens,
            "temperature": temperature if temperature is not None else self.temperature,
            "tools": tools,
            "tool_choice": tool_choice,
        }
        if self.seed is not None:
            payload["seed"] = self.seed

        response = self._post_chat(payload, phase=phase)
        message = response["choices"][0]["message"]

        tool_calls = message.get("tool_calls")
        if not tool_calls:
            if tool_choice == "required":
                raise ValueError(f"Model did not return a tool call. Content: {message.get('content', '')}")
            return None, {}, response

        tool_call = tool_calls[0]
        name = tool_call["function"]["name"]
        args = json.loads(tool_call["function"]["arguments"])

        return name, args, response

    # -- Transport -------------------------------------------------------------

    def _post_chat(self, payload: Dict[str, Any], phase: str = "other") -> Dict[str, Any]:
        """POST to the chat completions endpoint."""
        url = f"{self.endpoint}/chat/completions"
        headers = {
            "Content-Type": "application/json",
            "Authorization": f"Bearer {self.api_key}",
        }

        # vLLM extension: control Qwen3.5 thinking mode per-request.
        if self._enable_thinking is not None:
            payload["chat_template_kwargs"] = {
                "enable_thinking": self._enable_thinking,
            }

        # Snapshot the trace tags BEFORE sending. Critical at task boundaries:
        # if the sim auto-advances to the next task (publishing a new /scenario)
        # while this VLM call is mid-flight, ``set_task`` is invoked on the
        # bridge thread and ``self._task_id`` flips to the next task. Reading
        # the tags *after* the response would attribute this call's content
        # (which is about the previous task) to the new task_id — see the
        # T5/T6 boundary bug investigated 2026-05-22.
        tags = self._tags()

        # physics-trace-recording wave 2: journal the request BEFORE network
        # I/O so a crash mid-call still leaves a "pending" exchange on disk.
        # No-op (returns a cheap in-memory handle only) when LAM_VLM_LOG_DIR
        # is unset — see exchange_recorder module docstring.
        exchange = exchange_recorder.begin_exchange(
            phase=phase,
            payload=payload,
            episode_context=tags.get("episode_context"),
        )
        self.last_exchange_id = exchange.exchange_id

        try:
            resp = requests.post(url, headers=headers, json=payload, timeout=self.timeout)
            resp.raise_for_status()
            result = resp.json()
        except requests.exceptions.RequestException as e:
            self._record_error(exchange, e)
            raise
        except Exception as e:  # noqa: BLE001 — e.g. resp.json() on a malformed body
            self._record_error(exchange, e)
            raise

        exchange_recorder.end_exchange(exchange, response=result)
        self._record_response(payload, result, tags)
        return result

    # -- Image encoding --------------------------------------------------------

    def _inject_images(
        self,
        messages: List[Dict[str, Any]],
        images: List[bytes],
    ) -> List[Dict[str, Any]]:
        """Inject base64-encoded images into the last user message.

        Converts the last user message's content from a string to a
        multimodal content list with text + image_url entries.
        """
        messages = [dict(m) for m in messages]  # shallow copy

        # Find the last user message
        for i in range(len(messages) - 1, -1, -1):
            if messages[i]["role"] == "user":
                content = messages[i]["content"]
                if isinstance(content, str):
                    content = [{"type": "text", "text": content}]
                elif isinstance(content, list):
                    content = list(content)
                else:
                    content = [{"type": "text", "text": str(content)}]

                for img in images:
                    b64 = base64.b64encode(img).decode("utf-8")
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"},
                    })

                messages[i]["content"] = content
                break

        return messages
