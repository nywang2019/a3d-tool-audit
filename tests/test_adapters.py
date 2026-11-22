from __future__ import annotations

import json
from pathlib import Path

import pytest

from llm_eval.adapters import (
    AuthenticationError,
    BudgetExceededError,
    ConfigurationError,
    OpenAICompatibleAdapter,
    ResponseFormatError,
)


class FakeResponse:
    def __init__(
        self, status_code: int, payload: object, headers: dict[str, str] | None = None
    ):
        self.status_code = status_code
        self._payload = payload
        self.headers = headers or {}
        self.text = payload if isinstance(payload, str) else json.dumps(payload)

    def json(self) -> object:
        if isinstance(self._payload, str):
            raise ValueError("not json")  # noqa: TRY004 - mirrors Response.json()
        return self._payload


class FakeTransport:
    def __init__(self, responses: list[FakeResponse]):
        self.responses = iter(responses)
        self.calls = 0
        self.request_bodies: list[dict] = []

    def post(self, url: str, *, headers: dict[str, str], json: dict, timeout: float):
        self.calls += 1
        self.request_bodies.append(json)
        return next(self.responses)


def _success(text: str) -> FakeResponse:
    return FakeResponse(
        200,
        {
            "choices": [{"message": {"content": text}}],
            "usage": {"prompt_tokens": 11, "completion_tokens": 7},
        },
    )


def _adapter(
    monkeypatch: pytest.MonkeyPatch,
    transport: FakeTransport,
    *,
    max_requests: int | None = None,
    raw_log_dir: Path | None = None,
) -> OpenAICompatibleAdapter:
    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    return OpenAICompatibleAdapter(
        model_id="model-v1",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_API_KEY",
        max_retries=1,
        max_requests=max_requests,
        raw_log_dir=raw_log_dir,
        http_transport=transport,
        retry_base_seconds=0,
        sleeper=lambda _: None,
    )


def test_success_preserves_order_and_writes_redacted_log(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    transport = FakeTransport([_success("first"), _success("second")])
    adapter = _adapter(monkeypatch, transport, raw_log_dir=tmp_path)
    assert adapter.generate(["p1", "p2"]) == ["first", "second"]
    stats = adapter.get_statistics()
    assert stats["attempted_requests"] == 2
    assert stats["input_tokens"] == 22
    logs = list(tmp_path.glob("*.json"))
    assert len(logs) == 2
    assert "secret-value" not in logs[0].read_text(encoding="utf-8")


def test_auth_error_is_not_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = FakeTransport([FakeResponse(401, {"error": "bad key"})])
    adapter = _adapter(monkeypatch, transport)
    with pytest.raises(AuthenticationError):
        adapter.generate(["p"])
    assert transport.calls == 1
    assert adapter.get_statistics()["errors_by_kind"]["authentication"] == 1


def test_rate_limit_is_retried(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = FakeTransport(
        [
            FakeResponse(429, {"error": "slow down"}),
            _success("ok"),
        ]
    )
    adapter = _adapter(monkeypatch, transport)
    assert adapter.generate(["p"]) == ["ok"]
    assert transport.calls == 2


def test_budget_is_enforced_before_request(monkeypatch: pytest.MonkeyPatch) -> None:
    transport = FakeTransport([_success("one")])
    adapter = _adapter(monkeypatch, transport, max_requests=1)
    assert adapter.generate(["p1"]) == ["one"]
    with pytest.raises(BudgetExceededError):
        adapter.generate(["p2"])
    assert transport.calls == 1


def test_base_url_cannot_embed_credentials(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    with pytest.raises(ConfigurationError, match="must not"):
        OpenAICompatibleAdapter(
            model_id="model-v1",
            base_url="https://user:pass@example.invalid/v1",
            api_key_env="TEST_API_KEY",
        )


def test_session_can_ignore_environment_proxy(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    adapter = OpenAICompatibleAdapter(
        model_id="model-v1",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_API_KEY",
        trust_env=False,
    )
    assert adapter._transport.trust_env is False


def test_direct_connection_is_the_default(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("TEST_API_KEY", "secret-value")
    adapter = OpenAICompatibleAdapter(
        model_id="model-v1",
        base_url="https://example.invalid/v1",
        api_key_env="TEST_API_KEY",
    )
    assert adapter._transport.trust_env is False


def test_chat_normalizes_tool_calls(monkeypatch: pytest.MonkeyPatch) -> None:
    tools = [
        {
            "type": "function",
            "function": {
                "name": "compute",
                "description": "compute a value",
                "parameters": {
                    "type": "object",
                    "properties": {"x": {"type": "integer"}},
                    "required": ["x"],
                },
            },
        }
    ]
    transport = FakeTransport(
        [
            FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "finish_reason": "tool_calls",
                            "message": {
                                "role": "assistant",
                                "content": None,
                                "tool_calls": [
                                    {
                                        "id": "call_1",
                                        "type": "function",
                                        "function": {
                                            "name": "compute",
                                            "arguments": '{"x": 7}',
                                        },
                                    }
                                ],
                            },
                        }
                    ],
                    "usage": {"prompt_tokens": 10, "completion_tokens": 4},
                },
            )
        ]
    )
    adapter = _adapter(monkeypatch, transport)
    result = adapter.chat(
        [{"role": "user", "content": "compute"}],
        tools=tools,
    )
    assert result["tool_calls"][0]["function"]["name"] == "compute"
    assert result["tool_calls"][0]["function"]["arguments"] == '{"x": 7}'
    assert transport.request_bodies[0]["tools"] == tools


def test_failed_content_still_counts_reported_tokens(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    transport = FakeTransport(
        [
            FakeResponse(
                200,
                {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": "", "reasoning_content": "long"},
                        }
                    ],
                    "usage": {"prompt_tokens": 13, "completion_tokens": 9},
                },
            )
        ]
    )
    adapter = _adapter(monkeypatch, transport)
    with pytest.raises(ResponseFormatError):
        adapter.generate(["p"])
    stats = adapter.get_statistics()
    assert stats["input_tokens"] == 13
    assert stats["output_tokens"] == 9
