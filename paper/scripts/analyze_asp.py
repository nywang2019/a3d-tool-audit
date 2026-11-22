#!/usr/bin/env python3
"""Compute ASP as a graph-paired loss summary."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
CONTROL = ROOT / "results" / "funcbench-control-five-model-analysis-20261101"
OUTPUT = CONTROL / "asp"
MODELS = [
    "deepseek-flash",
    "kimi-k3.1-pro",
    "glm-5.3-flashx",
    "minimax-m3.1-flash",
    "qwen3.8-max",
]


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def bootstrap_mean(values: list[float], seed: int) -> tuple[float, float]:
    rng = random.Random(seed)
    samples = []
    for _ in range(4_000):
        sample = [values[rng.randrange(len(values))] for _ in values]
        samples.append(sum(sample) / len(sample))
    samples.sort()
    return samples[99], samples[3_899]


def sign_flip_p(values: list[float]) -> float:
    """Exact two-sided sign-flip test for a mean paired score."""
    doubled = [round(2 * value) for value in values]
    observed = abs(sum(doubled))
    nonzero = [value for value in doubled if value]
    exceed = 0
    for mask in range(1 << len(nonzero)):
        permuted = sum(
            value if mask & (1 << index) else -value
            for index, value in enumerate(nonzero)
        )
        exceed += abs(permuted) >= observed
    return exceed / (1 << len(nonzero))


def holm_adjust(p_values: list[float]) -> list[float]:
    order = sorted(range(len(p_values)), key=p_values.__getitem__)
    adjusted = [0.0] * len(p_values)
    running = 0.0
    for rank, index in enumerate(order):
        running = max(running, min(1.0, p_values[index] * (len(p_values) - rank)))
        adjusted[index] = running
    return adjusted


def main() -> None:
    frame = pd.read_csv(CONTROL / "trial_failures.csv")
    frame["correct"] = as_bool(frame["correct"])
    required_modes = {
        "none",
        "disconnected",
        "connected_incompatible",
        "connected_compatible",
    }
    if set(frame["distractor_mode"]) != required_modes:
        raise ValueError("ASP requires all four tool-usability conditions")
    if set(frame["model_alias"]) != set(MODELS):
        raise ValueError("ASP requires the five-model control panel")

    rows = []
    graph_rows = []
    for model_index, model in enumerate(MODELS):
        model_rows = frame[frame["model_alias"] == model]
        pivot = model_rows.pivot(
            index="graph_seed",
            columns="distractor_mode",
            values="correct",
        ).sort_index()
        if len(pivot) != 15 or pivot.isna().any().any():
            raise ValueError(f"Incomplete tool-usability grid for {model}")
        scores = (
            pivot["disconnected"].astype(int)
            + pivot["connected_incompatible"].astype(int)
            - 2 * pivot["connected_compatible"].astype(int)
        ) / 2
        for graph_seed, score in scores.items():
            graph_rows.append(
                {
                    "model_alias": model,
                    "graph_seed": int(graph_seed),
                    "asp": float(score),
                }
            )
        ci_low, ci_high = bootstrap_mean(
            scores.astype(float).tolist(),
            seed=20261106 + model_index,
        )
        solvable_scores = scores[pivot["none"]].astype(float)
        solvable_ci_low, solvable_ci_high = bootstrap_mean(
            solvable_scores.tolist(),
            seed=20261206 + model_index,
        )
        rows.append(
            {
                "model_alias": model,
                "graphs": len(scores),
                "none_correct": int(pivot["none"].sum()),
                "disconnected_correct": int(pivot["disconnected"].sum()),
                "incompatible_correct": int(
                    pivot["connected_incompatible"].sum()
                ),
                "compatible_correct": int(
                    pivot["connected_compatible"].sum()
                ),
                "baseline_delta": float(
                    pivot["connected_compatible"].mean()
                    - pivot["none"].mean()
                ),
                "connected_component": float(
                    pivot["disconnected"].mean()
                    - pivot["connected_compatible"].mean()
                ),
                "callability_component": float(
                    pivot["connected_incompatible"].mean()
                    - pivot["connected_compatible"].mean()
                ),
                "asp": float(scores.mean()),
                "asp_ci_low": ci_low,
                "asp_ci_high": ci_high,
                "sign_flip_p": sign_flip_p(scores.astype(float).tolist()),
                "positive_graphs": int((scores > 0).sum()),
                "zero_graphs": int((scores == 0).sum()),
                "negative_graphs": int((scores < 0).sum()),
                "solvable_graphs": len(solvable_scores),
                "asp_solvable": float(solvable_scores.mean()),
                "asp_solvable_ci_low": solvable_ci_low,
                "asp_solvable_ci_high": solvable_ci_high,
                "asp_solvable_sign_flip_p": sign_flip_p(
                    solvable_scores.tolist()
                ),
                "asp_solvable_positive_graphs": int(
                    (solvable_scores > 0).sum()
                ),
                "asp_solvable_zero_graphs": int((solvable_scores == 0).sum()),
                "asp_solvable_negative_graphs": int(
                    (solvable_scores < 0).sum()
                ),
            }
        )

    summary = pd.DataFrame(rows)
    summary["holm_p"] = holm_adjust(summary["sign_flip_p"].tolist())
    summary["asp_solvable_holm_p"] = holm_adjust(
        summary["asp_solvable_sign_flip_p"].tolist()
    )
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "asp_summary.csv", index=False)
    pd.DataFrame(graph_rows).to_csv(OUTPUT / "asp_graph_scores.csv", index=False)
    payload = {
        "definition": (
            "ASP = mean(disconnected - compatible) / 2 + "
            "mean(incompatible - compatible) / 2"
        ),
        "models": MODELS,
        "graphs_per_model": 15,
        "summary": summary.to_dict(orient="records"),
    }
    (OUTPUT / "asp_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
