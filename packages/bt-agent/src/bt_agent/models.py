"""Explicit chat client with preserved requests and opt-in recording."""

import requests

from ._engine.model.backend import ModelBackend
from ._engine.model import exchange_recorder


def _status_code(error):
    status = getattr(getattr(error, "response", None), "status_code", None)
    return status if type(status) is int and 100 <= status <= 599 else None


class ModelTransportError(requests.RequestException):
    """Transport failure without the original credential-bearing URL/message.

    ``error_type`` retains the provider exception class name and ``status_code``
    its HTTP status when available. Request/response objects are not retained.
    """

    def __init__(self, error):
        self.error_type = type(error).__name__
        self.status_code = _status_code(error)
        super().__init__(f"Model request failed ({self.error_type})")


class ChatModel(ModelBackend):
    """Compatible chat endpoint; per-role controller caps override the fallback.

    Construction performs no inference. Actual server/checkpoint qualification
    remains the deployment owner's responsibility. Unlike the private reference
    client, it does not snapshot environment variables or write a legacy log in
    /tmp. Use RecordedModel or explicitly set LAM_VLM_LOG_DIR for recording.
    """

    def __init__(
        self,
        *,
        base_url: str,
        model: str,
        api_key: str = "EMPTY",
        seed: int | None = 0,
        temperature: float = 0.0,
        thinking: bool = False,
        fallback_output_tokens: int = 4096,
        timeout: float = 120,
    ):
        if not base_url or not model:
            raise ValueError("An explicit base_url and model name are required")
        super().__init__(
            {
                "endpoint": base_url,
                "model_name": model,
                "api_key": api_key,
                "seed": seed,
                "temperature": temperature,
                "enable_thinking": thinking,
                "max_tokens": fallback_output_tokens,
                "timeout": timeout,
            }
        )

    def _record_run_start(self, config):
        # agent.describe() records the supported loaded configuration without
        # copying environment credentials or endpoint URLs into an implicit log.
        pass

    def _record_response(self, payload, result, tags):
        # Explicit exchange recording already occurs in the inherited transport.
        # Do not also append full conversations to its implicit legacy log.
        pass

    def _record_error(self, exchange, error):
        exchange_recorder.end_exchange(
            exchange,
            error={
                "type": type(error).__name__,
                "message": "Model request failed",
                "status": _status_code(error),
            },
        )

    def _post_chat(self, payload, phase="other"):
        try:
            return super()._post_chat(payload, phase)
        except requests.RequestException as error:
            # The Actor may include exception text in its next observation.
            # Keep transport diagnostics useful without leaking credentials.
            safe_error = ModelTransportError(error)
        # Leaving the handler also drops the original exception context; merely
        # using "from None" hides it from printing but still retains it.
        raise safe_error from None
