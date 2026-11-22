"""Graph-cluster tests of model-specific distractor effects."""

from __future__ import annotations

import random
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from .io_utils import atomic_write_json, write_csv


def analyze_distractor_heterogeneity(
    analysis_dir: Path | str,
    *,
    output_dir: Path | str,
    distractor_count: int = 5,
    distractor_mode: str = "connected_compatible",
    n_bootstrap: int = 4000,
) -> dict[str, Any]:
    """Test whether paired distractor effects differ across models."""
    frame = pd.read_csv(Path(analysis_dir) / "trial_failures.csv")
    baseline = frame.loc[frame["distractor_count"] == 0, [
        "model_alias",
        "graph_seed",
        "correct",
    ]].rename(columns={"correct": "baseline_correct"})
    treatment = frame.loc[
        (frame["distractor_count"] == distractor_count)
        & (frame["distractor_mode"] == distractor_mode),
        ["model_alias", "graph_seed", "correct"],
    ].rename(columns={"correct": "treatment_correct"})
    matched = baseline.merge(
        treatment,
        on=["model_alias", "graph_seed"],
        how="inner",
    )
    matched["difference"] = (
        matched["treatment_correct"].astype(int)
        - matched["baseline_correct"].astype(int)
    )
    effect_matrix = matched.pivot(
        index="graph_seed",
        columns="model_alias",
        values="difference",
    ).sort_index()
    models = list(effect_matrix.columns)
    effect_rows = [
        {
            "model_alias": model,
            "pairs": int(effect_matrix[model].notna().sum()),
            "delta": float(effect_matrix[model].mean()),
            "variance": float(effect_matrix[model].var(ddof=1)),
        }
        for model in models
    ]

    rng = random.Random(20261030)
    nprng = np.random.default_rng(20261030)
    pairwise_rows = []
    for index, first in enumerate(models):
        for second in models[index + 1 :]:
            paired = effect_matrix[[first, second]].dropna()
            deltas = (paired[first] - paired[second]).to_list()
            bootstrap = []
            for _ in range(n_bootstrap):
                sample = [
                    deltas[rng.randrange(len(deltas))]
                    for _ in range(len(deltas))
                ]
                bootstrap.append(float(np.mean(sample)))
            bootstrap.sort()
            pairwise_rows.append(
                {
                    "model_a": first,
                    "model_b": second,
                    "pairs": len(paired),
                    "delta_a_minus_b": float(np.mean(deltas)),
                    "ci_low": bootstrap[int(0.025 * (n_bootstrap - 1))],
                    "ci_high": bootstrap[int(0.975 * (n_bootstrap - 1))],
                }
            )

    observed_range = float(
        effect_matrix.mean().max() - effect_matrix.mean().min()
    )
    permutation_count = 0
    complete = effect_matrix.dropna().to_numpy(dtype=int)
    for _ in range(n_bootstrap):
        permuted = complete.copy()
        for row_index in range(permuted.shape[0]):
            nprng.shuffle(permuted[row_index])
        means = permuted.mean(axis=0)
        if float(means.max() - means.min()) >= observed_range:
            permutation_count += 1
    omnibus_p = (permutation_count + 1) / (n_bootstrap + 1)

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(pd.DataFrame(effect_rows), output / "model_effect_summary.csv")
    write_csv(
        pd.DataFrame(pairwise_rows),
        output / "pairwise_effect_differences.csv",
    )
    payload = {
        "analysis_dir": str(analysis_dir),
        "models": models,
        "graph_clusters": len(effect_matrix),
        "effect_range": observed_range,
        "omnibus_permutation_p": omnibus_p,
    }
    atomic_write_json(output / "heterogeneity_summary.json", payload)
    return payload
