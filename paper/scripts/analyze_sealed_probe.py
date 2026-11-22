#!/usr/bin/env python3
"""Compare executable and sealed executable goal-irrelevant tool conditions."""

from __future__ import annotations

import json
import random
from pathlib import Path

import pandas as pd

from llm_eval.funcbench import prepare_condition

ROOT = Path(__file__).resolve().parents[2]
ANALYSIS = ROOT / "results" / "funcbench-sealed-probe-analysis-20261101"
OUTPUT = ANALYSIS / "eap"
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
    frame = pd.read_csv(ANALYSIS / "trial_failures.csv")
    frame["correct"] = as_bool(frame["correct"])
    schema_matches = 0
    for graph_index in range(10):
        kwargs = {
            "depth": 3,
            "distractor_count": 5,
            "graph_seed": 20261029 + 3 * 1000 + graph_index,
            "core_nodes": 4,
            "trial_seed": 20261029 + 3 * 10000 + graph_index,
            "use_types": True,
            "use_subtypes": True,
            "num_supertypes": 5,
            "show_variable_names": False,
            "rename_variables": False,
            "num_noise_inputs": 0,
            "paired_core_graph": True,
            "stable_tool_order": True,
        }
        executable = prepare_condition(
            distractor_mode="connected_compatible",
            **kwargs,
        )
        sealed = prepare_condition(
            distractor_mode="sealed_compatible",
            **kwargs,
        )
        if not (
            executable.tools == sealed.tools
            and executable.prompt == sealed.prompt
            and executable.graph_fingerprint == sealed.graph_fingerprint
            and executable.required_graph_fingerprint
            == sealed.required_graph_fingerprint
            and executable.target_value == sealed.target_value
        ):
            raise ValueError(
                f"Sealed/executable schema or graph mismatch at graph {graph_index}"
            )
        schema_matches += 1
    frame = frame[frame["distractor_count"] == 5]
    rows = []
    graph_rows = []
    for model_index, model in enumerate(MODELS):
        model_rows = frame[frame["model_alias"] == model]
        pivot = model_rows.pivot_table(
            index="graph_seed",
            columns="distractor_mode",
            values="correct",
            aggfunc="first",
        ).sort_index()
        if set(pivot.columns) != {"connected_compatible", "sealed_compatible"}:
            raise ValueError(f"Missing sealed/executable pair for {model}")
        if len(pivot) != 10 or pivot.isna().any().any():
            raise ValueError(f"Incomplete probe grid for {model}")
        scores = (
            pivot["sealed_compatible"].astype(int)
            - pivot["connected_compatible"].astype(int)
        )
        for graph_seed, score in scores.items():
            graph_rows.append(
                {
                    "model_alias": model,
                    "graph_seed": int(graph_seed),
                    "eap": int(score),
                }
            )
        ci_low, ci_high = bootstrap_mean(
            scores.astype(float).tolist(),
            seed=20261120 + model_index,
        )
        rows.append(
            {
                "model_alias": model,
                "graphs": len(scores),
                "executable_correct": int(
                    pivot["connected_compatible"].sum()
                ),
                "sealed_correct": int(pivot["sealed_compatible"].sum()),
                "eap": float(scores.mean()),
                "eap_ci_low": ci_low,
                "eap_ci_high": ci_high,
                "sign_flip_p": sign_flip_p(scores.astype(float).tolist()),
                "positive_graphs": int((scores > 0).sum()),
                "zero_graphs": int((scores == 0).sum()),
                "negative_graphs": int((scores < 0).sum()),
            }
        )

    summary = pd.DataFrame(rows)
    summary["holm_p"] = holm_adjust(summary["sign_flip_p"].tolist())
    OUTPUT.mkdir(parents=True, exist_ok=True)
    summary.to_csv(OUTPUT / "eap_summary.csv", index=False)
    pd.DataFrame(graph_rows).to_csv(OUTPUT / "eap_graph_scores.csv", index=False)
    payload = {
        "definition": "EAP = mean(Y_sealed - Y_executable)",
        "models": MODELS,
        "graphs_per_model": 10,
        "schema_match_graphs": schema_matches,
        "summary": summary.to_dict(orient="records"),
    }
    (OUTPUT / "eap_summary.json").write_text(
        json.dumps(payload, indent=2) + "\n",
        encoding="utf-8",
    )
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
