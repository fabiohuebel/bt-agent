"""Durable model-interface recording, including the exact supplied image bytes.

This records calls before provider serialization, not raw HTTP or a transport's
hidden reasoning fields. Model arguments and returned values are not rewritten.
"""

from copy import deepcopy
import hashlib
import json
import math
import os
from pathlib import Path
import threading
import time


class RecordedModel:
    def __init__(self, model, output):
        self._model = model
        self.output = Path(output)
        self.output.mkdir(parents=True, exist_ok=False)
        (self.output / "images").mkdir()
        self._stream = (self.output / "calls.jsonl").open("x")
        self._lock = threading.RLock()
        self._records = []
        self._active = 0
        self._closed = False
        self.failure = None

    def __getattr__(self, name):
        return getattr(self._model, name)

    def _event(self, value):
        try:
            self._stream.write(json.dumps(self._json_value(value), allow_nan=False) + "\n")
            self._stream.flush()
            os.fsync(self._stream.fileno())
        except Exception:
            self.failure = "recording_error"
            raise

    @classmethod
    def _json_value(cls, value):
        # Record invalid model scalars without rewriting the model's reply or
        # turning a skill-level validation failure into an infrastructure error.
        if isinstance(value, float) and not math.isfinite(value):
            return {"nonfinite_float": repr(value)}
        if isinstance(value, dict):
            return {key: cls._json_value(item) for key, item in value.items()}
        if isinstance(value, (tuple, list)):
            return [cls._json_value(item) for item in value]
        return value

    def _image(self, data):
        if not isinstance(data, bytes):
            raise TypeError("Image recording expects the byte inputs used by these skills")
        digest = hashlib.sha256(data).hexdigest()
        path = self.output / "images" / (digest + ".image")
        if path.exists():
            if path.read_bytes() != data:
                raise IOError("Image artifact changed")
        else:
            with path.open("xb") as stream:
                stream.write(data)
                stream.flush()
                os.fsync(stream.fileno())
        return dict(sha256=digest, bytes=len(data), path=str(path.relative_to(self.output)))

    def _call(self, method, args, kwargs):
        with self._lock:
            if self._closed:
                raise RuntimeError("Model recorder is closed")
            try:
                images = [self._image(data) for data in (kwargs.get("images") or [])]
                record = dict(
                    index=len(self._records) + 1,
                    method=method,
                    arguments=deepcopy(args),
                    options={
                        k: deepcopy(v)
                        for k, v in kwargs.items()
                        if k in {"max_tokens", "temperature", "phase", "seed", "max_retries"}
                    },
                    images=images,
                    status="pending",
                    started_monotonic=time.monotonic(),
                )
                self._event(dict(kind="model_call_start", call=record))
            except Exception:
                self.failure = "recording_error"
                raise
            self._records.append(record)
            self._active += 1
        try:
            result = getattr(self._model, method)(*args, **kwargs)
        except BaseException as exc:
            with self._lock:
                interrupted = not isinstance(exc, Exception)
                response_error = isinstance(
                    exc, (ValueError, TypeError, KeyError)
                ) and not isinstance(exc, OSError)
                record.update(
                    status=(
                        "interrupted"
                        if interrupted
                        else "response_error" if response_error else "error"
                    ),
                    error_type=f"{type(exc).__module__}.{type(exc).__qualname__}",
                    elapsed_seconds=time.monotonic() - record["started_monotonic"],
                )
                if interrupted:
                    self.failure = "model_interrupted"
                elif not response_error:
                    self.failure = "model_error"
                self._event(dict(kind="model_call_end", call=record))
            raise
        else:
            with self._lock:
                record.update(
                    status="complete",
                    response=deepcopy(result),
                    elapsed_seconds=time.monotonic() - record["started_monotonic"],
                )
                self._event(dict(kind="model_call_end", call=record))
            return result
        finally:
            with self._lock:
                self._active -= 1

    def query_tool_call(self, *args, **kwargs):
        return self._call("query_tool_call", args, kwargs)

    def query_structured(self, *args, **kwargs):
        return self._call("query_structured", args, kwargs)

    def query(self, *args, **kwargs):
        return self._call("query", args, kwargs)

    def records(self):
        with self._lock:
            return deepcopy(self._records)

    def close(self):
        with self._lock:
            if self._active:
                raise RuntimeError(
                    "Model calls still pending; finalize after they return or end the worker process"
                )
            if not self._closed:
                self._stream.close()
                self._closed = True
