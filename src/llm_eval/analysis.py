"""Auditable aggregation and rank-divergence analysis for evaluation outputs."""

from __future__ import annotations

import json
import math
import numbers
from collections.abc import Iterable, Sequence
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from scipy.stats import binomtest, kendalltau, spearmanr

from .io_utils import atomic_write_json, write_csv

DETAIL_GLOB = "detailed_results_fold_*.json"


def answers_equivalent(predicted: Any, ground_truth: Any) -> bool:
    """Compare parsed answers without trusting a potentially buggy task score."""
    if predicted is None or ground_truth is None:
        return False
    if isinstance(predicted, (list, tuple)) and isinstance(ground_truth, (list, tuple)):
        return len(predicted) == len(ground_truth) and all(
            answers_equivalent(left, right)
            for left, right in zip(predicted, ground_truth, strict=True)
        )
    if isinstance(predicted, dict) and isinstance(ground_truth, dict):
        return predicted.keys() == ground_truth.keys() and all(
            answers_equivalent(predicted[key], ground_truth[key]) for key in predicted
        )
    if isinstance(predicted, bool) or isinstance(ground_truth, bool):
        return predicted is ground_truth
    if isinstance(predicted, numbers.Number) and isinstance(
        ground_truth, numbers.Number
    ):
        return math.isclose(
            float(predicted),
            float(ground_truth),
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
    predicted_text = str(predicted).strip()
    ground_truth_text = str(ground_truth).strip()
    try:
        return math.isclose(
            float(predicted_text),
            float(ground_truth_text),
            rel_tol=1e-6,
            abs_tol=1e-6,
        )
    except (TypeError, ValueError):
        return predicted_text == ground_truth_text


def _task_from_path(path: Path) -> str:
    """Extract the BeyondBench task name from a task-result path."""
    parts = path.parts
    if "task_results" not in parts:
        return path.parent.parent.name
    index = parts.index("task_results")
    return parts[index + 1] if index + 1 < len(parts) else "unknown"


def discover_detail_files(results_dir: Path | str) -> list[Path]:
    """Return all BeyondBench detailed-result files in stable order."""
    root = Path(results_dir)
    if not root.exists():
        raise FileNotFoundError(f"Results directory does not exist: {root}")
    return sorted(p for p in root.rglob(DETAIL_GLOB) if p.is_file())


def _normalise_record(
    record: dict[str, Any], task: str, source: Path
) -> dict[str, Any]:
    generation_ok = bool(record.get("generation_success", False))
    parse_ok = bool(record.get("parsing_success", False)) if generation_ok else False
    raw_correct = bool(int(record.get("accuracy", 0) or 0))
    recomputed_correct = parse_ok and answers_equivalent(
        record.get("predicted_answer"),
        record.get("ground_truth"),
    )
    correct = raw_correct or recomputed_correct
    if not generation_ok:
        outcome = "technical_failure"
    elif not parse_ok:
        outcome = "format_failure"
    elif correct:
        outcome = "correct"
    else:
        outcome = "model_error"
    return {
        "task": task,
        "prompt": record.get("prompt", ""),
        "response": record.get("model_response", ""),
        "ground_truth": str(record.get("ground_truth", "")),
        "predicted_answer": str(record.get("predicted_answer", "")),
        "generation_success": generation_ok,
        "parsing_success": parse_ok,
        "correct": int(correct),
        "raw_correct": int(raw_correct),
        "scoring_override": int(recomputed_correct and not raw_correct),
        "outcome": outcome,
        "source_file": str(source),
    }


def find_run_directories(results_root: Path | str) -> list[Path]:
    """Find run/model directories that contain at least one detail file."""
    root = Path(results_root)
    candidates: set[Path] = set()
    for path in discover_detail_files(root):
        for parent in path.parents:
            if parent.name == "task_results":
                candidates.add(parent.parent)
                break
    # A model output is normally root/<run_id>/<model_alias>/task_results/...
    return sorted(candidates)


def load_detail_records(model_dir: Path | str) -> list[dict[str, Any]]:
    """Load and classify every detailed result under one model output directory."""
    root = Path(model_dir)
    records: list[dict[str, Any]] = []
    for path in sorted(root.rglob(DETAIL_GLOB)):
        data = json.loads(path.read_text(encoding="utf-8"))
        if not isinstance(data, list):
            raise TypeError(f"Expected a JSON list in {path}")
        task = _task_from_path(path)
        records.extend(
            _normalise_record(item, task, path)
            for item in data
            if isinstance(item, dict)
        )
    if not records:
        raise ValueError(f"No detailed records found under {root}")
    return records


def _cluster_bootstrap_ci(
    frame: pd.DataFrame,
    *,
    n_bootstrap: int,
    rng: np.random.Generator,
) -> tuple[float, float]:
    """Bootstrap equal-weight task means; returns a percentile 95% interval."""
    task_means = frame.groupby("task", sort=True)["correct"].mean().to_numpy()
    if len(task_means) < 2:
        return (float("nan"), float("nan"))
    draws = rng.choice(
        task_means, size=(n_bootstrap, len(task_means)), replace=True
    ).mean(axis=1)
    low, high = np.quantile(draws, [0.025, 0.975])
    return float(low), float(high)


def summarise_model(
    records: Sequence[dict[str, Any]],
    *,
    n_bootstrap: int = 2_000,
    seed: int = 20261029,
    deduplicate_prompts: bool = True,
) -> dict[str, Any]:
    """Compute accuracy, failure categories, and task-clustered uncertainty."""
    if not records:
        raise ValueError("Cannot summarise an empty result set")
    frame = pd.DataFrame(records)
    n_records = len(frame)
    if deduplicate_prompts:
        frame = frame.drop_duplicates(subset=["prompt"], keep="first")
    n = len(frame)
    outcome_counts = frame["outcome"].value_counts().to_dict()
    overall = float(frame["correct"].mean())
    answerable = frame[frame["generation_success"] & frame["parsing_success"]]
    low, high = _cluster_bootstrap_ci(
        frame,
        n_bootstrap=n_bootstrap,
        rng=np.random.default_rng(seed),
    )
    return {
        "n_records": n_records,
        "n_instances": n,
        "duplicate_prompt_n": n_records - n,
        "n_tasks": int(frame["task"].nunique()),
        "accuracy": overall,
        "answerable_accuracy": (
            float(answerable["correct"].mean())
            if not answerable.empty
            else float("nan")
        ),
        "n_answerable": len(answerable),
        "accuracy_ci95_task_cluster": [low, high],
        "correct_n": int(outcome_counts.get("correct", 0)),
        "technical_failure_n": int(outcome_counts.get("technical_failure", 0)),
        "format_failure_n": int(outcome_counts.get("format_failure", 0)),
        "model_error_n": int(outcome_counts.get("model_error", 0)),
        "upstream_scoring_override_n": int(frame["scoring_override"].sum()),
        "technical_failure_rate": float(
            frame["outcome"].eq("technical_failure").mean()
        ),
        "format_failure_rate": float(frame["outcome"].eq("format_failure").mean()),
        "model_error_rate": float(frame["outcome"].eq("model_error").mean()),
    }


def paired_model_comparison(results_root: Path | str) -> pd.DataFrame:
    """Compare every model pair on the intersection of prompt hashes."""
    root = Path(results_root)
    records_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for model_dir in find_run_directories(root):
        records = load_detail_records(model_dir)
        records_by_model[model_dir.name] = {
            str(record["prompt"]): record for record in records
        }
    aliases = sorted(records_by_model)
    rows: list[dict[str, Any]] = []
    for index, left_alias in enumerate(aliases):
        for right_alias in aliases[index + 1 :]:
            left_records = records_by_model[left_alias]
            right_records = records_by_model[right_alias]
            common_prompts = sorted(set(left_records) & set(right_records))
            if not common_prompts:
                continue
            left_correct = np.asarray(
                [left_records[prompt]["correct"] for prompt in common_prompts],
                dtype=bool,
            )
            right_correct = np.asarray(
                [right_records[prompt]["correct"] for prompt in common_prompts],
                dtype=bool,
            )
            answerable = np.asarray(
                [
                    left_records[prompt]["generation_success"]
                    and left_records[prompt]["parsing_success"]
                    and right_records[prompt]["generation_success"]
                    and right_records[prompt]["parsing_success"]
                    for prompt in common_prompts
                ],
                dtype=bool,
            )
            left_only = int(np.sum(left_correct & ~right_correct))
            right_only = int(np.sum(~left_correct & right_correct))
            discordant = left_only + right_only
            pvalue = (
                float(binomtest(left_only, discordant, p=0.5).pvalue)
                if discordant
                else 1.0
            )
            rows.append(
                {
                    "model_a": left_alias,
                    "model_b": right_alias,
                    "n_matched_prompts": len(common_prompts),
                    "raw_accuracy_a": float(left_correct.mean()),
                    "raw_accuracy_b": float(right_correct.mean()),
                    "raw_accuracy_difference": float(
                        left_correct.mean() - right_correct.mean()
                    ),
                    "n_both_answerable": int(answerable.sum()),
                    "answerable_accuracy_a": (
                        float(left_correct[answerable].mean())
                        if answerable.any()
                        else float("nan")
                    ),
                    "answerable_accuracy_b": (
                        float(right_correct[answerable].mean())
                        if answerable.any()
                        else float("nan")
                    ),
                    "both_correct": int(np.sum(left_correct & right_correct)),
                    "a_only_correct": left_only,
                    "b_only_correct": right_only,
                    "both_incorrect": int(np.sum(~left_correct & ~right_correct)),
                    "mcnemar_exact_pvalue": pvalue,
                }
            )
    return pd.DataFrame(rows)


def summarize_raw_usage(results_root: Path | str) -> pd.DataFrame:
    """Recompute actual token cost and terminal reasons from raw HTTP logs."""
    root = Path(results_root)
    rows: list[dict[str, Any]] = []
    for model_dir in find_run_directories(root):
        status_counts: dict[str, int] = {}
        finish_counts: dict[str, int] = {}
        error_counts: dict[str, int] = {}
        input_tokens = 0
        output_tokens = 0
        reasoning_tokens = 0
        empty_content = 0
        total_latency_seconds = 0.0
        logs = sorted((model_dir / "raw").glob("*.json"))
        for path in logs:
            data = json.loads(path.read_text(encoding="utf-8"))
            total_latency_seconds += float(data.get("latency_seconds", 0.0) or 0.0)
            status = str(data.get("status_code"))
            status_counts[status] = status_counts.get(status, 0) + 1
            error_kind = str(data.get("error_kind") or "success")
            error_counts[error_kind] = error_counts.get(error_kind, 0) + 1
            response = data.get("response")
            if not isinstance(response, dict):
                continue
            usage = response.get("usage")
            if isinstance(usage, dict):
                input_tokens += int(usage.get("prompt_tokens", 0) or 0)
                output_tokens += int(usage.get("completion_tokens", 0) or 0)
                details = usage.get("completion_tokens_details")
                if isinstance(details, dict):
                    reasoning_tokens += int(details.get("reasoning_tokens", 0) or 0)
            choices = response.get("choices")
            if not isinstance(choices, list) or not choices:
                continue
            first = choices[0] if isinstance(choices[0], dict) else {}
            finish_reason = str(first.get("finish_reason"))
            finish_counts[finish_reason] = finish_counts.get(finish_reason, 0) + 1
            message = first.get("message")
            content = message.get("content") if isinstance(message, dict) else None
            empty_content += not bool(content)
        rows.append(
            {
                "model_alias": model_dir.name,
                "n_raw_attempts": len(logs),
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "reasoning_tokens": reasoning_tokens,
                "empty_content_attempts": empty_content,
                "total_latency_seconds": total_latency_seconds,
                "http_status_counts": json.dumps(status_counts, ensure_ascii=False),
                "finish_reason_counts": json.dumps(finish_counts, ensure_ascii=False),
                "error_kind_counts": json.dumps(error_counts, ensure_ascii=False),
            }
        )
    return pd.DataFrame(rows)


def summarise_all_models(
    results_root: Path | str,
    *,
    n_bootstrap: int = 2_000,
    seed: int = 20261029,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Summarise all model runs and return model and task-level tables."""
    root = Path(results_root)
    model_rows: list[dict[str, Any]] = []
    record_rows: list[dict[str, Any]] = []
    for model_dir in find_run_directories(root):
        records = load_detail_records(model_dir)
        model_alias = model_dir.name
        summary = summarise_model(records, n_bootstrap=n_bootstrap, seed=seed)
        summary["model_alias"] = model_alias
        summary["model_dir"] = str(model_dir)
        model_rows.append(summary)
        raw_frame = pd.DataFrame(records)
        frame = raw_frame.drop_duplicates(subset=["prompt"], keep="first")
        raw_counts = raw_frame.groupby("task", sort=True).agg(
            n_records=("correct", "size"),
            n_duplicate_prompts=(
                "prompt",
                lambda values: len(values) - values.nunique(),
            ),
        )
        task_summary = frame.groupby("task", sort=True).agg(
            n_instances=("prompt", "nunique"),
            accuracy=("correct", "mean"),
            upstream_scoring_override_n=("scoring_override", "sum"),
            technical_failure_rate=(
                "outcome",
                lambda x: float((x == "technical_failure").mean()),
            ),
            format_failure_rate=(
                "outcome",
                lambda x: float((x == "format_failure").mean()),
            ),
        )
        task_summary = task_summary.join(raw_counts)
        task_summary = task_summary.reset_index()
        task_summary.insert(0, "model_alias", model_alias)
        record_rows.extend(task_summary.to_dict(orient="records"))
    if not model_rows:
        raise ValueError(f"No detailed model results found under {root}")
    columns = [
        "model_alias",
        "n_records",
        "n_instances",
        "duplicate_prompt_n",
        "n_tasks",
        "accuracy",
        "answerable_accuracy",
        "n_answerable",
        "accuracy_ci95_task_cluster",
        "correct_n",
        "technical_failure_n",
        "format_failure_n",
        "model_error_n",
        "upstream_scoring_override_n",
        "technical_failure_rate",
        "format_failure_rate",
        "model_error_rate",
        "model_dir",
    ]
    return pd.DataFrame(model_rows)[columns], pd.DataFrame(record_rows)


def load_static_scores(path: Path | str) -> pd.DataFrame:
    """Load and validate model-level static benchmark scores."""
    frame = pd.read_csv(path)
    required = {"model_alias", "benchmark", "score", "source_url", "access_date"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Static score CSV is missing columns: {sorted(missing)}")
    frame["score"] = pd.to_numeric(frame["score"], errors="raise")
    if frame["score"].isna().any():
        raise ValueError("Static scores must not contain missing values")
    duplicated = frame.duplicated(["model_alias", "benchmark"])
    if duplicated.any():
        raise ValueError("Duplicate model_alias/benchmark rows are not allowed")
    return frame


def _safe_correlation(
    x: Iterable[float], y: Iterable[float], method: str
) -> tuple[float, float]:
    x_array = np.asarray(list(x), dtype=float)
    y_array = np.asarray(list(y), dtype=float)
    if len(x_array) < 3 or len(np.unique(x_array)) < 2 or len(np.unique(y_array)) < 2:
        return float("nan"), float("nan")
    if method == "kendall":
        value, pvalue = kendalltau(x_array, y_array)
    else:
        value, pvalue = spearmanr(x_array, y_array)
    return float(value), float(pvalue)


def rank_divergence(
    model_summary: pd.DataFrame,
    static_scores: pd.DataFrame,
    *,
    n_bootstrap: int = 2_000,
    seed: int = 20261029,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Compare dynamic accuracy rank with each static benchmark separately."""
    comparison_rows: list[dict[str, Any]] = []
    correlation_rows: list[dict[str, Any]] = []
    rng = np.random.default_rng(seed)

    for benchmark, static_group in static_scores.groupby("benchmark", sort=True):
        merged = model_summary.merge(
            static_group[["model_alias", "score", "source_url", "access_date"]],
            on="model_alias",
            how="inner",
            validate="one_to_one",
        )
        if len(merged) < 3:
            correlation_rows.append(
                {
                    "benchmark": benchmark,
                    "n_models": len(merged),
                    "kendall_tau": float("nan"),
                    "kendall_pvalue": float("nan"),
                    "spearman_rho": float("nan"),
                    "spearman_pvalue": float("nan"),
                    "bootstrap_ci95_tau": [float("nan"), float("nan")],
                    "status": "insufficient_matched_models",
                }
            )
            continue

        merged = merged.sort_values("accuracy", ascending=False).reset_index(drop=True)
        merged["dynamic_rank"] = np.arange(1, len(merged) + 1)
        merged["static_rank"] = merged["score"].rank(ascending=False, method="average")
        merged["rank_change"] = merged["static_rank"] - merged["dynamic_rank"]
        for row in merged.to_dict(orient="records"):
            comparison_rows.append(
                {
                    "benchmark": benchmark,
                    "model_alias": row["model_alias"],
                    "dynamic_accuracy": row["accuracy"],
                    "dynamic_rank": int(row["dynamic_rank"]),
                    "static_score": row["score"],
                    "static_rank": row["static_rank"],
                    "rank_change": row["rank_change"],
                    "source_url": row["source_url"],
                    "access_date": row["access_date"],
                }
            )

        tau, tau_p = _safe_correlation(merged["accuracy"], merged["score"], "kendall")
        rho, rho_p = _safe_correlation(merged["accuracy"], merged["score"], "spearman")
        bootstrap_taus: list[float] = []
        for _ in range(n_bootstrap):
            # Resample model observations with replacement; equal-score ties are retained.
            indices = rng.integers(0, len(merged), size=len(merged))
            boot = merged.iloc[indices]
            boot_tau, _ = _safe_correlation(boot["accuracy"], boot["score"], "kendall")
            if math.isfinite(boot_tau):
                bootstrap_taus.append(boot_tau)
        if bootstrap_taus:
            ci = np.quantile(bootstrap_taus, [0.025, 0.975]).tolist()
        else:
            ci = [float("nan"), float("nan")]
        correlation_rows.append(
            {
                "benchmark": benchmark,
                "n_models": len(merged),
                "kendall_tau": tau,
                "kendall_pvalue": tau_p,
                "spearman_rho": rho,
                "spearman_pvalue": rho_p,
                "bootstrap_ci95_tau": ci,
                "status": "ok",
            }
        )

    return pd.DataFrame(comparison_rows), pd.DataFrame(correlation_rows)


def _render_rank_chart(comparison: pd.DataFrame, destination: Path) -> None:
    """Render a compact static-vs-dynamic rank chart without claiming a causal effect."""
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    benchmarks = list(comparison["benchmark"].drop_duplicates())
    if not benchmarks:
        return
    fig, axes = plt.subplots(
        1, len(benchmarks), figsize=(5.2 * len(benchmarks), 4.4), squeeze=False
    )
    for axis, benchmark in zip(axes[0], benchmarks):
        group = comparison[comparison["benchmark"] == benchmark].sort_values(
            "static_rank"
        )
        for row in group.itertuples():
            axis.plot(
                [0, 1], [row.static_rank, row.dynamic_rank], marker="o", linewidth=1.4
            )
            axis.text(
                -0.04,
                row.static_rank,
                row.model_alias,
                ha="right",
                va="center",
                fontsize=8,
            )
            axis.text(
                1.04,
                row.dynamic_rank,
                row.model_alias,
                ha="left",
                va="center",
                fontsize=8,
            )
        axis.set_xlim(-0.5, 1.5)
        axis.set_xticks([0, 1], ["Static", "Dynamic"])
        axis.invert_yaxis()
        axis.set_ylabel("Rank (1 = best)")
        axis.set_title(benchmark)
        axis.grid(axis="y", alpha=0.2)
    fig.tight_layout()
    destination.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(destination, format="svg", bbox_inches="tight")
    plt.close(fig)


def analyse_results(
    results_root: Path | str,
    *,
    static_scores_path: Path | str | None = None,
    output_dir: Path | str | None = None,
    n_bootstrap: int = 2_000,
    seed: int = 20261029,
) -> dict[str, Any]:
    """Run the complete descriptive and rank-divergence analysis."""
    root = Path(results_root)
    out = Path(output_dir) if output_dir else root / "analysis"
    out.mkdir(parents=True, exist_ok=True)
    summary, task_summary = summarise_all_models(
        root, n_bootstrap=n_bootstrap, seed=seed
    )
    paired = paired_model_comparison(root)
    raw_usage = summarize_raw_usage(root)
    write_csv(summary, out / "accuracy_summary.csv")
    write_csv(task_summary, out / "task_summary.csv")
    write_csv(paired, out / "paired_model_comparison.csv")
    write_csv(raw_usage, out / "raw_usage_summary.csv")

    payload: dict[str, Any] = {
        "results_root": str(root),
        "n_models": len(summary),
        "models": summary.to_dict(orient="records"),
        "paired_model_comparison": paired.to_dict(orient="records"),
        "raw_usage": raw_usage.to_dict(orient="records"),
        "rank_analysis": None,
    }
    if static_scores_path:
        static = load_static_scores(static_scores_path)
        comparison, correlations = rank_divergence(
            summary,
            static,
            n_bootstrap=n_bootstrap,
            seed=seed,
        )
        write_csv(comparison, out / "rank_comparison.csv")
        write_csv(correlations, out / "rank_correlations.csv")
        _render_rank_chart(comparison, out / "rank_divergence.svg")
        payload["rank_analysis"] = {
            "static_scores_path": str(static_scores_path),
            "correlations": correlations.to_dict(orient="records"),
            "note": "Rank correlations are descriptive; the model sample is small.",
        }

    atomic_write_json(out / "analysis.json", payload)
    return payload
