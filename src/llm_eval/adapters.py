"""OpenAI-compatible HTTP adapter for API-only evaluation of domestic LLMs."""

from __future__ import annotations

import json
import os
import random
import re
import time
import uuid
from collections.abc import Callable
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, Protocol
from urllib.parse import urlsplit

import requests

from .io_utils import atomic_write_json


class AdapterError(RuntimeError):
    """Base class for errors that are technical rather than model-answer errors."""

    error_kind = "technical"


class ConfigurationError(AdapterError):
    error_kind = "configuration"


class AuthenticationError(AdapterError):
    error_kind = "authentication"


class RateLimitError(AdapterError):
    error_kind = "rate_limit"


class TransientTransportError(AdapterError):
    error_kind = "transport"


class ServerResponseError(AdapterError):
    error_kind = "server"


class ResponseFormatError(AdapterError):
    error_kind = "response_format"


class BudgetExceededError(AdapterError):
    error_kind = "budget"


class HttpTransport(Protocol):
    def post(
        self,
        url: str,
        *,
        headers: dict[str, str],
        json: dict[str, Any],
        timeout: float,
    ) -> requests.Response: ...


@dataclass
class _CallResult:
    text: str
    usage: dict[str, Any]
    latency_seconds: float
    message: dict[str, Any]
    tool_calls: list[dict[str, Any]]
    finish_reason: str | None


class OpenAICompatibleAdapter:
    """Minimal, auditable adapter that does not depend on a vendor SDK."""

    def __init__(
        self,
        *,
        model_id: str,
        base_url: str,
        api_key_env: str,
        timeout_seconds: float = 120.0,
        max_retries: int = 4,
        temperature: float = 0.0,
        max_tokens: int = 4096,
        extra_body: dict[str, Any] | None = None,
        max_requests: int | None = None,
        raw_log_dir: Path | str | None = None,
        http_transport: HttpTransport | None = None,
        trust_env: bool = False,
        retry_base_seconds: float = 1.0,
        sleeper: Callable[[float], None] = time.sleep,
    ) -> None:
        if not model_id.strip():
            raise ConfigurationError("model_id must not be empty")
        if not base_url.startswith(("https://", "http://")):
            raise ConfigurationError("base_url must start with https:// or http://")
        parsed_url = urlsplit(base_url)
        if parsed_url.query or parsed_url.fragment:
            raise ConfigurationError(
                "base_url must not contain a query string or fragment"
            )
        if parsed_url.username or parsed_url.password:
            raise ConfigurationError("base_url must not embed credentials")
        if not re.fullmatch(r"[A-Za-z_][A-Za-z0-9_]*", api_key_env):
            raise ConfigurationError("api_key_env must be an environment variable name")
        api_key = os.environ.get(api_key_env)
        if not api_key:
            raise ConfigurationError(
                f"Missing API key environment variable: {api_key_env}"
            )
        if max_retries < 0:
            raise ConfigurationError("max_retries must be non-negative")
        if max_requests is not None and max_requests < 1:
            raise ConfigurationError("max_requests must be positive when supplied")

        self.model_id = model_id
        self.base_url = base_url.rstrip("/")
        self.api_provider = "openai_compatible"
        # BeyondBench dispatches on this legacy identifier before calling
        # model_handler.generate(). The concrete provider remains distinct in
        # api_provider and model_info.
        self.backend = "openai"
        self.timeout_seconds = float(timeout_seconds)
        self.max_retries = int(max_retries)
        self.default_temperature = float(temperature)
        self.default_max_tokens = int(max_tokens)
        self.extra_body = dict(extra_body or {})
        self.max_requests = max_requests
        self.raw_log_dir = Path(raw_log_dir) if raw_log_dir else None
        self.retry_base_seconds = max(0.0, float(retry_base_seconds))
        self._sleep = sleeper
        if http_transport is not None:
            self._transport = http_transport
        else:
            session = requests.Session()
            session.trust_env = trust_env
            self._transport = session
        self._api_key_env = api_key_env
        self._api_key = api_key
        self._endpoint = (
            self.base_url
            if self.base_url.endswith("/chat/completions")
            else f"{self.base_url}/chat/completions"
        )
        self._attempts = 0
        self._successes = 0
        self._started = time.monotonic()
        self._stats: dict[str, Any] = {
            "attempted_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "input_tokens": 0,
            "output_tokens": 0,
            "total_latency_seconds": 0.0,
            "errors_by_kind": {},
        }

    @property
    def endpoint(self) -> str:
        """Public endpoint accessor without exposing credentials."""
        return self._endpoint

    def __repr__(self) -> str:
        return (
            f"OpenAICompatibleAdapter(model_id={self.model_id!r}, "
            f"endpoint={self._endpoint!r}, api_key_env={self._api_key_env!r})"
        )

    def _redact(self, value: Any) -> Any:
        """Remove the API key from structured values before persistence."""
        if isinstance(value, str):
            return value.replace(self._api_key, "[REDACTED]")
        if isinstance(value, list):
            return [self._redact(item) for item in value]
        if isinstance(value, dict):
            return {str(key): self._redact(item) for key, item in value.items()}
        return value

    def _record_error(self, error: AdapterError) -> None:
        counts = self._stats["errors_by_kind"]
        counts[error.error_kind] = counts.get(error.error_kind, 0) + 1

    def _write_raw(
        self,
        *,
        sequence: int,
        prompt: str,
        request_body: dict[str, Any],
        status_code: int | None,
        response_body: Any,
        error_kind: str | None,
        latency_seconds: float,
        attempt: int,
    ) -> None:
        if self.raw_log_dir is None:
            return
        timestamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%S%fZ")
        destination = (
            self.raw_log_dir / f"{sequence:06d}-{timestamp}-{uuid.uuid4().hex[:8]}.json"
        )
        atomic_write_json(
            destination,
            self._redact(
                {
                    "model": self.model_id,
                    "endpoint": self._endpoint,
                    "sequence": sequence,
                    "attempt": attempt,
                    "prompt": prompt,
                    "request": request_body,
                    "status_code": status_code,
                    "response": response_body,
                    "error_kind": error_kind,
                    "latency_seconds": latency_seconds,
                    "timestamp_utc": datetime.now(UTC).isoformat(),
                }
            ),
        )

    def _response_json(self, response: requests.Response) -> dict[str, Any]:
        try:
            payload = response.json()
        except (ValueError, TypeError) as exc:
            body = getattr(response, "text", "")[:500]
            raise ResponseFormatError(
                f"Provider returned non-JSON content: {body!r}"
            ) from exc
        if not isinstance(payload, dict):
            raise ResponseFormatError("Provider JSON root must be an object")
        return payload

    @staticmethod
    def _extract_message(
        payload: dict[str, Any],
    ) -> tuple[str, dict[str, Any], list[dict[str, Any]], str | None]:
        choices = payload.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ResponseFormatError("Response is missing a non-empty choices list")
        first = choices[0]
        if not isinstance(first, dict):
            raise ResponseFormatError("Response choice must be an object")
        message = first.get("message")
        if not isinstance(message, dict):
            raise ResponseFormatError("Response choice is missing message")
        content = message.get("content")
        text = ""
        if isinstance(content, str):
            text = content
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    text_parts.append(item["text"])
            text = "".join(text_parts)
        if not text.strip():
            for key in ("text", "output_text"):
                if isinstance(first.get(key), str) and first[key].strip():
                    text = first[key]
                    break

        tool_calls: list[dict[str, Any]] = []
        raw_calls = message.get("tool_calls")
        if isinstance(raw_calls, list):
            for call in raw_calls:
                if not isinstance(call, dict):
                    continue
                function = call.get("function")
                if not isinstance(function, dict):
                    continue
                name = function.get("name")
                arguments = function.get("arguments")
                if not isinstance(name, str) or not name:
                    continue
                if not isinstance(arguments, str):
                    arguments = json.dumps(arguments or {}, ensure_ascii=False)
                tool_calls.append(
                    {
                        "id": str(call.get("id") or f"call_{len(tool_calls)}"),
                        "type": "function",
                        "function": {"name": name, "arguments": arguments},
                    }
                )
        if not text.strip() and not tool_calls:
            raise ResponseFormatError(
                "Response content is empty and contains no tool calls"
            )
        return text, message, tool_calls, first.get("finish_reason")

    @staticmethod
    def _retry_after(response: requests.Response | None) -> float | None:
        if response is None:
            return None
        value = response.headers.get("Retry-After")
        if value is None:
            return None
        try:
            return max(0.0, float(value))
        except ValueError:
            return None

    def _raise_for_status(self, response: requests.Response, payload: Any) -> None:
        status = int(response.status_code)
        if 200 <= status < 300:
            return
        summary = json.dumps(self._redact(payload), ensure_ascii=False)[:800]
        if status in {401, 403}:
            raise AuthenticationError(
                f"HTTP {status}: authentication failed: {summary}"
            )
        if status == 429:
            raise RateLimitError(f"HTTP 429: rate limited: {summary}")
        if status >= 500:
            raise ServerResponseError(
                f"HTTP {status}: provider server error: {summary}"
            )
        raise ResponseFormatError(
            f"HTTP {status}: request rejected or malformed: {summary}"
        )

    def _record_response_accounting(
        self,
        payload: Any,
        latency_seconds: float,
    ) -> None:
        """Account for provider-reported tokens even when content parsing fails."""
        self._stats["total_latency_seconds"] += latency_seconds
        if not isinstance(payload, dict):
            return
        usage = payload.get("usage")
        if not isinstance(usage, dict):
            return
        self._stats["input_tokens"] += int(usage.get("prompt_tokens", 0) or 0)
        self._stats["output_tokens"] += int(usage.get("completion_tokens", 0) or 0)

    def _post_once(
        self,
        *,
        prompt: str,
        request_body: dict[str, Any],
        sequence: int,
        attempt: int,
    ) -> _CallResult:
        headers = {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            "Accept": "application/json",
        }
        started = time.monotonic()
        response: requests.Response | None = None
        payload: Any = None
        try:
            response = self._transport.post(
                self._endpoint,
                headers=headers,
                json=request_body,
                timeout=self.timeout_seconds,
            )
            latency = time.monotonic() - started
            try:
                payload = response.json()
            except (ValueError, TypeError):
                payload = {"raw_text": getattr(response, "text", "")[:4000]}
            self._record_response_accounting(payload, latency)
            self._raise_for_status(response, payload)
            text, message, tool_calls, finish_reason = self._extract_message(payload)
            usage = payload.get("usage")
            if not isinstance(usage, dict):
                usage = {}
            self._write_raw(
                sequence=sequence,
                prompt=prompt,
                request_body=request_body,
                status_code=int(response.status_code),
                response_body=payload,
                error_kind=None,
                latency_seconds=latency,
                attempt=attempt,
            )
            return _CallResult(
                text=text,
                usage=usage,
                latency_seconds=latency,
                message=message,
                tool_calls=tool_calls,
                finish_reason=finish_reason,
            )
        except requests.Timeout as exc:
            latency = time.monotonic() - started
            self._record_response_accounting(payload, latency)
            error = TransientTransportError("Provider request timed out")
            self._write_raw(
                sequence=sequence,
                prompt=prompt,
                request_body=request_body,
                status_code=getattr(response, "status_code", None),
                response_body=payload,
                error_kind=error.error_kind,
                latency_seconds=latency,
                attempt=attempt,
            )
            raise error from exc
        except requests.RequestException as exc:
            latency = time.monotonic() - started
            self._record_response_accounting(payload, latency)
            error = TransientTransportError(
                f"Provider transport error: {type(exc).__name__}"
            )
            self._write_raw(
                sequence=sequence,
                prompt=prompt,
                request_body=request_body,
                status_code=getattr(response, "status_code", None),
                response_body=payload,
                error_kind=error.error_kind,
                latency_seconds=latency,
                attempt=attempt,
            )
            raise error from exc
        except AdapterError as exc:
            latency = time.monotonic() - started
            self._write_raw(
                sequence=sequence,
                prompt=prompt,
                request_body=request_body,
                status_code=getattr(response, "status_code", None),
                response_body=payload,
                error_kind=exc.error_kind,
                latency_seconds=latency,
                attempt=attempt,
            )
            raise

    def _request_with_retries(
        self,
        *,
        request_body: dict[str, Any],
        prompt: str,
        sequence: int,
    ) -> _CallResult:
        """Execute one chat-completion body with bounded retries and accounting."""
        last_error: AdapterError | None = None
        for attempt in range(self.max_retries + 1):
            if self.max_requests is not None and self._attempts >= self.max_requests:
                error = BudgetExceededError(
                    f"Request budget exhausted before prompt {sequence} "
                    f"(cap={self.max_requests})"
                )
                self._record_error(error)
                raise error
            self._attempts += 1
            self._stats["attempted_requests"] += 1
            try:
                result = self._post_once(
                    prompt=prompt,
                    request_body=request_body,
                    sequence=sequence,
                    attempt=attempt,
                )
            except AdapterError as exc:
                last_error = exc
                self._record_error(exc)
                retryable = isinstance(
                    exc,
                    (RateLimitError, ServerResponseError, TransientTransportError),
                )
                if not retryable or attempt >= self.max_retries:
                    self._stats["failed_requests"] += 1
                    raise
                delay = self.retry_base_seconds * (2**attempt) + random.uniform(0, 0.2)
                self._sleep(min(delay, 30.0))
                continue
            self._successes += 1
            self._stats["successful_requests"] += 1
            return result
        if last_error:  # pragma: no cover - loop returns or raises
            raise last_error
        raise RuntimeError("Request loop exited without a result")

    def generate(
        self,
        prompts: list[str],
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float = 1.0,
        **kwargs: Any,
    ) -> list[str]:
        """Generate one response per prompt, preserving input order."""
        if not isinstance(prompts, list):
            raise TypeError("prompts must be a list")
        responses: list[str] = []
        effective_temperature = (
            self.default_temperature if temperature is None else float(temperature)
        )
        effective_max_tokens = (
            self.default_max_tokens if max_tokens is None else int(max_tokens)
        )

        for sequence, prompt in enumerate(prompts, start=1):
            body = {
                "model": self.model_id,
                "messages": [{"role": "user", "content": prompt}],
                "temperature": effective_temperature,
                "top_p": float(top_p),
                "max_tokens": effective_max_tokens,
                "stream": False,
                **self.extra_body,
            }
            result = self._request_with_retries(
                request_body=body,
                prompt=prompt,
                sequence=sequence,
            )
            if result.tool_calls:
                raise ResponseFormatError("Tool calls returned for a text-only request")
            responses.append(result.text)
        return responses

    def chat(
        self,
        messages: list[dict[str, Any]],
        *,
        tools: list[dict[str, Any]] | None = None,
        tool_choice: str | dict[str, Any] | None = None,
        max_tokens: int | None = None,
        temperature: float | None = None,
        top_p: float = 1.0,
        sequence: int = 1,
    ) -> dict[str, Any]:
        """Return one structured assistant message, including normalized tool calls."""
        if not isinstance(messages, list) or not messages:
            raise TypeError("messages must be a non-empty list")
        body: dict[str, Any] = {
            "model": self.model_id,
            "messages": messages,
            "temperature": (
                self.default_temperature if temperature is None else float(temperature)
            ),
            "top_p": float(top_p),
            "max_tokens": (
                self.default_max_tokens if max_tokens is None else int(max_tokens)
            ),
            "stream": False,
            **self.extra_body,
        }
        if tools:
            body["tools"] = tools
        if tool_choice is not None:
            body["tool_choice"] = tool_choice
        prompt_for_log = json.dumps(messages[-1], ensure_ascii=False, default=str)
        result = self._request_with_retries(
            request_body=body,
            prompt=prompt_for_log,
            sequence=sequence,
        )
        message = dict(result.message)
        if result.tool_calls:
            message["tool_calls"] = result.tool_calls
        return {
            "message": message,
            "content": result.text,
            "tool_calls": result.tool_calls,
            "finish_reason": result.finish_reason,
            "usage": result.usage,
        }

    def get_model_info(self) -> dict[str, Any]:
        """Return BeyondBench-compatible model metadata without secrets."""
        return {
            "model_name": self.model_id,
            "backend": self.backend,
            "api_provider": self.api_provider,
            "model_type": "api_based",
            "endpoint": self._endpoint,
            "api_key_env": self._api_key_env,
        }

    def get_statistics(self) -> dict[str, Any]:
        """Return usage and reliability statistics for the run."""
        result = dict(self._stats)
        result["elapsed_seconds"] = time.monotonic() - self._started
        result["max_requests"] = self.max_requests
        result["model_id"] = self.model_id
        return result

    def warm_up(self) -> float:
        """Avoid spending a hidden request on warm-up in API-only mode."""
        return 0.0

    def count_tokens(self, text: str) -> int:
        """Use tiktoken when available, otherwise a conservative estimate."""
        if not text:
            return 0
        try:
            import tiktoken

            return len(tiktoken.get_encoding("cl100k_base").encode(text))
        except (ImportError, LookupError, TypeError, ValueError):
            return max(1, int(len(text.split()) * 1.3))

    def cleanup(self) -> None:
        """No local model resources to release."""
        return


def load_and_validate_config(config_path: Path | str) -> dict[str, Any]:
    """Compatibility helper for API callers that need YAML validation elsewhere."""
    from .runner import load_config, validate_config

    return validate_config(load_config(config_path), require_keys=False)
