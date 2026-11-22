"""Static figures for strict paired FuncBenchGen analyses."""

from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def plot_funcbench_summary(
    analysis_dir: Path | str,
    *,
    output_dir: Path | str | None = None,
) -> dict[str, str]:
    """Plot condition accuracy and paired distractor effects."""
    source = Path(analysis_dir)
    destination = Path(output_dir) if output_dir else source
    destination.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(source / "metric_summary.csv")
    effects = pd.read_csv(source / "paired_effect_summary.csv")
    if metrics.empty or effects.empty:
        raise ValueError("Metric and paired effect summaries must not be empty")

    models = list(dict.fromkeys(metrics["model_alias"].astype(str)))
    positions = np.arange(len(models))
    baseline = metrics.loc[metrics["distractor_count"] == 0].set_index(
        "model_alias"
    )
    treatment = metrics.loc[metrics["distractor_count"] != 0].set_index(
        "model_alias"
    )
    effect_index = effects.set_index("model_alias")

    figure, (accuracy_axis, effect_axis) = plt.subplots(
        1,
        2,
        figsize=(12, 4.8),
        constrained_layout=True,
    )
    width = 0.36
    baseline_values = np.array(
        [baseline.loc[model, "accuracy"] for model in models],
        dtype=float,
    )
    treatment_values = np.array(
        [treatment.loc[model, "accuracy"] for model in models],
        dtype=float,
    )
    baseline_error = np.vstack(
        [
            baseline_values
            - np.array([baseline.loc[model, "accuracy_ci_low"] for model in models]),
            np.array([baseline.loc[model, "accuracy_ci_high"] for model in models])
            - baseline_values,
        ]
    )
    treatment_error = np.vstack(
        [
            treatment_values
            - np.array([treatment.loc[model, "accuracy_ci_low"] for model in models]),
            np.array([treatment.loc[model, "accuracy_ci_high"] for model in models])
            - treatment_values,
        ]
    )
    accuracy_axis.bar(
        positions - width / 2,
        baseline_values,
        width,
        yerr=baseline_error,
        capsize=4,
        label="0 distractors",
        color="#4C78A8",
    )
    accuracy_axis.bar(
        positions + width / 2,
        treatment_values,
        width,
        yerr=treatment_error,
        capsize=4,
        label="5 distractors",
        color="#F58518",
    )
    accuracy_axis.set_xticks(positions, models, rotation=18, ha="right")
    accuracy_axis.set_ylim(0, 1.05)
    accuracy_axis.set_ylabel("Exact success")
    accuracy_axis.set_title("Accuracy with Wilson 95% intervals")
    accuracy_axis.legend(frameon=False)
    accuracy_axis.grid(axis="y", alpha=0.2)

    deltas = np.array([effect_index.loc[model, "delta"] for model in models])
    lower = deltas - np.array(
        [effect_index.loc[model, "delta_ci_low"] for model in models]
    )
    upper = np.array(
        [effect_index.loc[model, "delta_ci_high"] for model in models]
    ) - deltas
    effect_axis.bar(
        positions,
        deltas,
        color=["#54A24B" if delta >= 0 else "#E45756" for delta in deltas],
    )
    effect_axis.errorbar(
        positions,
        deltas,
        yerr=np.vstack([lower, upper]),
        fmt="none",
        capsize=4,
        color="black",
    )
    effect_axis.axhline(0, color="black", linewidth=0.8)
    effect_axis.set_xticks(positions, models, rotation=18, ha="right")
    effect_axis.set_ylim(-1, 0.35)
    effect_axis.set_ylabel("Paired success difference")
    effect_axis.set_title("5 minus 0 distractors, graph bootstrap 95% CI")
    effect_axis.grid(axis="y", alpha=0.2)

    png_path = destination / "funcbench_paired_summary.png"
    svg_path = destination / "funcbench_paired_summary.svg"
    figure.savefig(png_path, dpi=180)
    figure.savefig(svg_path)
    plt.close(figure)
    return {"png": str(png_path), "svg": str(svg_path)}


def plot_control_comparison(
    analysis_dir: Path | str,
    *,
    output_dir: Path | str | None = None,
) -> dict[str, str]:
    """Plot baseline, disconnected, incompatible and compatible controls."""
    source = Path(analysis_dir)
    destination = Path(output_dir) if output_dir else source
    destination.mkdir(parents=True, exist_ok=True)
    metrics = pd.read_csv(source / "metric_summary.csv")
    models = list(dict.fromkeys(metrics["model_alias"].astype(str)))
    conditions = [
        ("none", 0, "No distractor", "#4C78A8"),
        ("disconnected", 5, "Disconnected", "#B279A2"),
        ("connected_incompatible", 5, "Incompatible", "#E45756"),
        ("connected_compatible", 5, "Compatible", "#F58518"),
    ]
    positions = np.arange(len(models))
    width = 0.18
    figure, axis = plt.subplots(figsize=(11, 5), constrained_layout=True)
    for offset, (mode, count, label, color) in enumerate(conditions):
        subset = metrics.loc[
            (metrics["distractor_mode"] == mode)
            & (metrics["distractor_count"] == count)
        ].set_index("model_alias")
        values = np.array([subset.loc[model, "accuracy"] for model in models])
        lower = values - np.array(
            [subset.loc[model, "accuracy_ci_low"] for model in models]
        )
        upper = np.array(
            [subset.loc[model, "accuracy_ci_high"] for model in models]
        ) - values
        axis.bar(
            positions + (offset - 1.5) * width,
            values,
            width,
            yerr=np.vstack([lower, upper]),
            capsize=3,
            label=label,
            color=color,
        )
    axis.set_xticks(positions, models, rotation=18, ha="right")
    axis.set_ylim(0, 1.05)
    axis.set_ylabel("Exact success")
    axis.set_title("Distractor affordance controls on paired graphs")
    axis.grid(axis="y", alpha=0.2)
    axis.legend(frameon=False, ncol=2, fontsize=8)
    png_path = destination / "funcbench_control_modes.png"
    svg_path = destination / "funcbench_control_modes.svg"
    figure.savefig(png_path, dpi=180)
    figure.savefig(svg_path)
    plt.close(figure)
    return {"png": str(png_path), "svg": str(svg_path)}


def plot_intervention_effect(
    analysis_dir: Path | str,
    *,
    output_dir: Path | str | None = None,
) -> dict[str, str]:
    """Plot paired success change for an intervention."""
    source = Path(analysis_dir)
    destination = Path(output_dir) if output_dir else source
    destination.mkdir(parents=True, exist_ok=True)
    effects = pd.read_csv(source / "intervention_effect_summary.csv")
    models = effects["model_alias"].astype(str).tolist()
    deltas = effects["delta"].to_numpy(dtype=float)
    lower = deltas - effects["delta_ci_low"].to_numpy(dtype=float)
    upper = effects["delta_ci_high"].to_numpy(dtype=float) - deltas
    positions = np.arange(len(models))
    figure, axis = plt.subplots(figsize=(8, 4.8), constrained_layout=True)
    axis.bar(
        positions,
        deltas,
        color=["#54A24B" if delta >= 0 else "#E45756" for delta in deltas],
    )
    axis.errorbar(
        positions,
        deltas,
        yerr=np.vstack([lower, upper]),
        fmt="none",
        color="black",
        capsize=4,
    )
    axis.axhline(0, color="black", linewidth=0.8)
    axis.set_xticks(positions, models, rotation=18, ha="right")
    axis.set_ylim(-1, 0.4)
    axis.set_ylabel("Paired success change")
    axis.set_title("Prerequisite gating minus ungated tools")
    axis.grid(axis="y", alpha=0.2)
    png_path = destination / "funcbench_gating_effect.png"
    svg_path = destination / "funcbench_gating_effect.svg"
    figure.savefig(png_path, dpi=180)
    figure.savefig(svg_path)
    plt.close(figure)
    return {"png": str(png_path), "svg": str(svg_path)}
