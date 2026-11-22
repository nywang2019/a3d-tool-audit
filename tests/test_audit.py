from __future__ import annotations

import json
from pathlib import Path

from llm_eval.audit import (
    canonical_answer_text,
    replay_model,
    scan_secret_leaks,
    select_replay_records,
)


def _write_details(model_dir: Path) -> None:
    for task, count in {"sum": 4, "sorting": 4}.items():
        destination = model_dir / "task_results" / task / "test_case_0"
        destination.mkdir(parents=True)
        records = [
            {
                "prompt": f"{task}-{index}",
                "model_response": rf"\boxed{{{index}}}",
                "generation_success": True,
                "parsing_success": True,
                "accuracy": 1,
            }
            for index in range(count)
        ]
        (destination / "detailed_results_fold_0.json").write_text(
            json.dumps(records),
            encoding="utf-8",
        )


class StableAdapter:
    def generate(self, prompts, **_kwargs):
        index = int(prompts[0].split("-")[-1])
        return [rf"\boxed{{{index}}}"]


def test_replay_selection_covers_tasks_and_compares_answers(tmp_path: Path) -> None:
    _write_details(tmp_path)
    selected = select_replay_records(tmp_path, sample_size=4)
    assert {item["task"] for item in selected} == {"sum", "sorting"}
    report = replay_model(adapter=StableAdapter(), model_dir=tmp_path, sample_size=4)
    assert report["sample_size"] == 4
    assert report["final_answer_equal_rate"] == 1.0


def test_secret_scan_detects_secret_without_returning_it(
    tmp_path: Path,
    monkeypatch,
) -> None:
    monkeypatch.setenv("AUDIT_SECRET", "very-secret-value")
    safe = tmp_path / "safe.json"
    unsafe = tmp_path / "unsafe.json"
    safe.write_text('{"ok": true}', encoding="utf-8")
    unsafe.write_text('{"value": "very-secret-value"}', encoding="utf-8")
    report = scan_secret_leaks(tmp_path, api_key_envs=["AUDIT_SECRET"])
    assert report["passed"] is False
    assert report["files_with_secret_material"] == [str(unsafe)]
    assert "very-secret-value" not in json.dumps(report)


def test_canonical_answer_handles_dict_ordering() -> None:
    left = "<answer>{1: True, 2: False}</answer>"
    right = "<answer>{2: False, 1: True}</answer>"
    assert canonical_answer_text(left) == canonical_answer_text(right)
