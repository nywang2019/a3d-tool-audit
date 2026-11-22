from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from llm_eval.analysis import (
    analyse_results,
    answers_equivalent,
    paired_model_comparison,
    summarise_all_models,
    summarize_raw_usage,
)


def _write_model(
    root: Path,
    alias: str,
    outcomes: dict[str, list[str]],
) -> None:
    for task, task_outcomes in outcomes.items():
        detail_dir = root / alias / "task_results" / task / "test_case_0"
        detail_dir.mkdir(parents=True)
        records = []
        for index, outcome in enumerate(task_outcomes):
            records.append(
                {
                    "prompt": f"{task}-{index}",
                    "model_response": r"\boxed{1}",
                    "generation_success": outcome != "technical_failure",
                    "parsing_success": outcome
                    not in {"technical_failure", "format_failure"},
                    "accuracy": int(outcome == "correct"),
                    "ground_truth": "1",
                    "predicted_answer": "1" if outcome == "correct" else None,
                }
            )
        (detail_dir / "detailed_results_fold_0.json").write_text(
            json.dumps(records),
            encoding="utf-8",
        )


def test_summary_classifies_outcomes_and_bootstraps(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_model(
        root,
        "model-a",
        {
            "sum": ["correct", "correct"],
            "sorting": ["model_error", "format_failure"],
            "median": ["technical_failure", "correct"],
        },
    )
    summary, _ = summarise_all_models(root, n_bootstrap=200, seed=7)
    row = summary.iloc[0]
    assert row["n_instances"] == 6
    assert row["correct_n"] == 3
    assert row["technical_failure_n"] == 1
    assert row["format_failure_n"] == 1
    assert row["model_error_n"] == 1
    assert row["accuracy"] == 0.5
    assert 0.0 <= row["accuracy_ci95_task_cluster"][0] <= 1.0


def test_correct_answer_is_recovered_when_upstream_score_is_wrong(
    tmp_path: Path,
) -> None:
    root = tmp_path / "run"
    detail_dir = (
        root / "model-a" / "task_results" / "geometric_sequence" / "test_case_0"
    )
    detail_dir.mkdir(parents=True)
    records = [
        {
            "prompt": "2, 4, 8, ?",
            "model_response": r"\boxed{16}",
            "generation_success": True,
            "parsing_success": True,
            "accuracy": 0,
            "ground_truth": 16,
            "predicted_answer": 16,
        }
    ]
    (detail_dir / "detailed_results_fold_0.json").write_text(
        json.dumps(records),
        encoding="utf-8",
    )
    summary, task_summary = summarise_all_models(root, n_bootstrap=200, seed=7)
    assert answers_equivalent(16, "16.0")
    assert summary.iloc[0]["accuracy"] == 1.0
    assert summary.iloc[0]["upstream_scoring_override_n"] == 1
    assert task_summary.iloc[0]["upstream_scoring_override_n"] == 1


def test_rank_divergence_uses_separate_static_benchmarks(tmp_path: Path) -> None:
    root = tmp_path / "run"
    patterns = {
        "model-a": ["correct", "correct", "correct"],
        "model-b": ["correct", "correct", "model_error"],
        "model-c": ["correct", "model_error", "model_error"],
    }
    for alias, outcomes in patterns.items():
        _write_model(
            root,
            alias,
            {"sum": outcomes[:1], "sorting": outcomes[1:2], "median": outcomes[2:]},
        )
    static_csv = tmp_path / "static.csv"
    pd.DataFrame(
        [
            ["model-a", "C-Eval", 90.0, "https://example.invalid/a", "2026-10-29", ""],
            ["model-b", "C-Eval", 70.0, "https://example.invalid/b", "2026-10-29", ""],
            ["model-c", "C-Eval", 50.0, "https://example.invalid/c", "2026-10-29", ""],
            ["model-a", "CMMLU", 70.0, "https://example.invalid/a", "2026-10-29", ""],
            ["model-b", "CMMLU", 90.0, "https://example.invalid/b", "2026-10-29", ""],
            ["model-c", "CMMLU", 50.0, "https://example.invalid/c", "2026-10-29", ""],
        ],
        columns=[
            "model_alias",
            "benchmark",
            "score",
            "source_url",
            "access_date",
            "notes",
        ],
    ).to_csv(static_csv, index=False)

    output = tmp_path / "analysis"
    payload = analyse_results(
        root,
        static_scores_path=static_csv,
        output_dir=output,
        n_bootstrap=200,
        seed=7,
    )
    assert payload["n_models"] == 3
    assert (output / "analysis.json").exists()
    assert (output / "rank_comparison.csv").exists()
    assert (output / "rank_divergence.svg").exists()
    correlations = pd.read_csv(output / "rank_correlations.csv")
    assert set(correlations["benchmark"]) == {"C-Eval", "CMMLU"}


def test_pairwise_comparison_uses_matched_prompts(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_model(
        root,
        "model-a",
        {
            "sum": ["correct", "correct", "model_error"],
            "sorting": ["correct", "correct", "correct"],
        },
    )
    _write_model(
        root,
        "model-b",
        {
            "sum": ["correct", "model_error", "correct"],
            "sorting": ["correct", "correct", "model_error"],
        },
    )
    paired = paired_model_comparison(root)
    row = paired.iloc[0]
    assert row["n_matched_prompts"] == 6
    assert row["both_correct"] == 3
    assert row["a_only_correct"] == 2
    assert row["b_only_correct"] == 1


def test_raw_usage_includes_failed_attempt_cost(tmp_path: Path) -> None:
    root = tmp_path / "run"
    _write_model(root, "model-a", {"sum": ["correct"]})
    raw_dir = root / "model-a" / "raw"
    raw_dir.mkdir()
    (raw_dir / "000001.json").write_text(
        json.dumps(
            {
                "status_code": 200,
                "error_kind": "response_format",
                "response": {
                    "choices": [
                        {
                            "finish_reason": "length",
                            "message": {"content": ""},
                        }
                    ],
                    "usage": {
                        "prompt_tokens": 11,
                        "completion_tokens": 7,
                        "completion_tokens_details": {"reasoning_tokens": 6},
                    },
                },
            }
        ),
        encoding="utf-8",
    )
    usage = summarize_raw_usage(root)
    row = usage.iloc[0]
    assert row["n_raw_attempts"] == 1
    assert row["input_tokens"] == 11
    assert row["output_tokens"] == 7
    assert row["reasoning_tokens"] == 6
    assert row["empty_content_attempts"] == 1
