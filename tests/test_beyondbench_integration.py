from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import pytest
from beyondbench.tasks.easy.sum_task import SumTask

from llm_eval.adapters import OpenAICompatibleAdapter


class LocalFakeResponse:
    def __init__(self, payload: dict):
        self.status_code = 200
        self._payload = payload
        self.headers: dict[str, str] = {}
        self.text = json.dumps(payload)

    def json(self) -> dict:
        return self._payload


class SolverTransport:
    """Offline transport that solves the real BeyondBench sum prompt."""

    def post(self, url: str, *, headers: dict, json: dict, timeout: float):
        prompt = json["messages"][0]["content"]
        match = re.search(r"Add the following list of numbers:\s*(\[[^\n]+\])", prompt)
        if not match:
            return LocalFakeResponse(
                {"choices": [{"message": {"content": r"\boxed{0}"}}], "usage": {}}
            )
        values = ast.literal_eval(match.group(1))
        answer = sum(values)
        return LocalFakeResponse(
            {
                "choices": [{"message": {"content": rf"\boxed{{{answer}}}"}}],
                "usage": {"prompt_tokens": 20, "completion_tokens": 5},
            }
        )


def test_real_beyondbench_task_with_offline_adapter(
    monkeypatch: pytest.MonkeyPatch,
    tmp_path: Path,
) -> None:
    monkeypatch.setenv("OFFLINE_API_KEY", "not-a-real-secret")
    adapter = OpenAICompatibleAdapter(
        model_id="offline-solver",
        base_url="https://offline.invalid/v1",
        api_key_env="OFFLINE_API_KEY",
        max_retries=0,
        max_requests=5,
        raw_log_dir=tmp_path / "raw",
        http_transport=SolverTransport(),
    )
    task = SumTask(
        model_handler=adapter,
        output_dir=str(tmp_path / "evaluation"),
        min_val=1,
        max_val=100,
        num_folds=1,
        num_samples=2,
        store_details=True,
        temperature=0.0,
        top_p=1.0,
        max_tokens=64,
        seed=20261029,
    )
    metrics = task.run_evaluation([8])
    assert len(metrics) == 1
    assert metrics[0]["accuracy"] == 1.0
    details = list((tmp_path / "evaluation").rglob("detailed_results_fold_*.json"))
    assert len(details) == 1
    records = json.loads(details[0].read_text(encoding="utf-8"))
    assert len(records) == 2
    assert all(item["accuracy"] == 1 for item in records)
    assert adapter.get_statistics()["successful_requests"] == 2
