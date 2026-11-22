"""Combine strict-paired baseline and distractor-dose analyses."""

from __future__ import annotations

import math
import random
from collections.abc import Iterable
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd

from .funcbench_errors import _mcnemar_exact_p, _wilson_interval
from .io_utils import atomic_write_json, write_csv


def _bootstrap_delta(
    baseline: pd.Series,
    treatment: pd.Series,
) -> tuple[float, float]:
    differences = (treatment.astype(int) - baseline.astype(int)).tolist()
    rng = random.Random(20261030)
    samples = []
    for _ in range(4000):
        resampled = [
            differences[rng.randrange(len(differences))]
            for _ in range(len(differences))
        ]
        samples.append(sum(resampled) / len(resampled))
    samples.sort()
    return (
        samples[int(0.025 * (len(samples) - 1))],
        samples[int(0.975 * (len(samples) - 1))],
    )


def analyze_funcbench_distractor_dose(
    baseline_analysis: Path | str,
    dose_analysis: Path | str | Iterable[Path | str],
    *,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Merge 0/5 results with a separate 2-distractor run by graph seed."""
    baseline = pd.read_csv(Path(baseline_analysis) / "trial_failures.csv")
    dose_analyses = (
        [dose_analysis]
        if isinstance(dose_analysis, (str, Path))
        else list(dose_analysis)
    )
    dose = pd.concat(
        [
            pd.read_csv(Path(directory) / "trial_failures.csv")
            for directory in dose_analyses
        ],
        ignore_index=True,
    )
    combined = pd.concat([baseline, dose], ignore_index=True)
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)

    condition_rows = []
    effect_rows = []
    validation_errors = []
    metric_columns = [
        "required_recall",
        "selection_precision",
        "dependency_order_score",
        "argument_score",
        "dependency_trace_fidelity",
    ]
    for model_alias, model_rows in combined.groupby("model_alias", sort=True):
        conditions = {
            int(distractor): group
            for distractor, group in model_rows.groupby("distractor_count")
        }
        if 0 not in conditions or 2 not in conditions or 5 not in conditions:
            validation_errors.append(
                f"{model_alias} missing one of distractor conditions 0/2/5"
            )
            continue
        common_all_seeds = (
            set(conditions[0]["graph_seed"])
            & set(conditions[2]["graph_seed"])
            & set(conditions[5]["graph_seed"])
        )
        key_columns = ["graph_seed", "required_graph_fingerprint", "target_value"]
        baseline_rows = conditions[0].set_index("graph_seed")
        for distractor in (2, 5):
            treatment_rows = conditions[distractor].set_index("graph_seed")
            common_seeds = sorted(set(baseline_rows.index) & set(treatment_rows.index))
            if not common_seeds:
                validation_errors.append(
                    f"{model_alias}/x{distractor} has no matching graph seeds"
                )
                continue
            baseline_common = baseline_rows.loc[common_seeds]
            treatment_common = treatment_rows.loc[common_seeds]
            for column in key_columns[1:]:
                if not (
                    baseline_common[column].astype(str)
                    == treatment_common[column].astype(str)
                ).all():
                    validation_errors.append(
                        f"{model_alias}/x{distractor} {column} mismatch"
                    )
            baseline_correct = baseline_common["correct"].astype(bool)
            treatment_correct = treatment_common["correct"].astype(bool)
            baseline_only = int((baseline_correct & ~treatment_correct).sum())
            treatment_only = int((~baseline_correct & treatment_correct).sum())
            ci_low, ci_high = _bootstrap_delta(baseline_correct, treatment_correct)
            effect_rows.append(
                {
                    "model_alias": model_alias,
                    "distractor_count": distractor,
                    "pairs": len(common_seeds),
                    "baseline_accuracy": float(baseline_correct.mean()),
                    "dose_accuracy": float(treatment_correct.mean()),
                    "delta": float(treatment_correct.mean() - baseline_correct.mean()),
                    "baseline_only": baseline_only,
                    "dose_only": treatment_only,
                    "delta_ci_low": ci_low,
                    "delta_ci_high": ci_high,
                    "mcnemar_exact_p": _mcnemar_exact_p(
                        baseline_only,
                        treatment_only,
                    ),
                }
            )

        for distractor in (0, 2, 5):
            group = conditions[distractor].loc[
                conditions[distractor]["graph_seed"].isin(common_all_seeds)
            ]
            correct = int(group["correct"].sum())
            total = len(group)
            ci_low, ci_high = _wilson_interval(correct, total)
            row = {
                "model_alias": model_alias,
                "distractor_count": distractor,
                "trials": total,
                "correct": correct,
                "accuracy": correct / total if total else math.nan,
                "accuracy_ci_low": ci_low,
                "accuracy_ci_high": ci_high,
            }
            for metric in metric_columns:
                if metric in group:
                    row[metric] = float(group[metric].mean())
            condition_rows.append(row)

    condition_summary = pd.DataFrame(condition_rows)
    effect_summary = pd.DataFrame(effect_rows)
    write_csv(condition_summary, output / "dose_condition_summary.csv")
    write_csv(effect_summary, output / "dose_effect_summary.csv")

    if not condition_summary.empty:
        models = list(dict.fromkeys(condition_summary["model_alias"]))
        figure, axis = plt.subplots(figsize=(8, 5), constrained_layout=True)
        for model in models:
            rows = condition_summary.loc[
                condition_summary["model_alias"] == model
            ].sort_values("distractor_count")
            values = rows["accuracy"].to_numpy(dtype=float)
            lower = values - rows["accuracy_ci_low"].to_numpy(dtype=float)
            upper = rows["accuracy_ci_high"].to_numpy(dtype=float) - values
            axis.errorbar(
                rows["distractor_count"],
                values,
                yerr=[lower, upper],
                marker="o",
                capsize=4,
                label=model,
            )
        axis.set_xticks([0, 2, 5])
        axis.set_ylim(0, 1.05)
        axis.set_xlabel("Connected distractors")
        axis.set_ylabel("Exact success")
        axis.set_title("Distractor-dose response on paired graphs")
        axis.grid(alpha=0.2)
        axis.legend(frameon=False, fontsize=8)
        figure.savefig(output / "dose_response.png", dpi=180)
        figure.savefig(output / "dose_response.svg")
        plt.close(figure)

    payload = {
        "baseline_analysis": str(baseline_analysis),
        "dose_analyses": [str(path) for path in dose_analyses],
        "passed": not validation_errors,
        "validation_errors": validation_errors,
        "models": list(condition_summary["model_alias"].unique())
        if not condition_summary.empty
        else [],
        "conditions": sorted(
            condition_summary["distractor_count"].unique().tolist()
        )
        if not condition_summary.empty
        else [],
    }
    atomic_write_json(output / "dose_summary.json", payload)
    return payload
