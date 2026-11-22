"""Paired comparison of two tool-use intervention settings."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .funcbench_dose import _bootstrap_delta
from .funcbench_errors import _mcnemar_exact_p
from .io_utils import atomic_write_json, write_csv


def compare_funcbench_intervention(
    analysis_dir: Path | str,
    *,
    output_dir: Path | str,
    distractor_count: int = 5,
    distractor_mode: str = "connected_compatible",
    reference_gating: bool = False,
    treatment_gating: bool = True,
) -> dict[str, Any]:
    """Compare two interventions on identical graph seeds."""
    frame = pd.read_csv(Path(analysis_dir) / "trial_failures.csv")
    common_filters = (
        (frame["distractor_count"] == distractor_count)
        & (frame["distractor_mode"] == distractor_mode)
    )
    reference = frame.loc[
        common_filters & (frame["tool_gating"].astype(bool) == reference_gating)
    ]
    treatment = frame.loc[
        common_filters & (frame["tool_gating"].astype(bool) == treatment_gating)
    ]
    key = ["model_alias", "graph_seed"]
    merged = reference[key + [
        "correct",
        "required_graph_fingerprint",
        "target_value",
        "primary_category",
    ]].rename(
        columns={
            "correct": "reference_correct",
            "primary_category": "reference_category",
        }
    ).merge(
        treatment[key + [
            "correct",
            "required_graph_fingerprint",
            "target_value",
            "primary_category",
        ]].rename(
            columns={
                "correct": "treatment_correct",
                "primary_category": "treatment_category",
            }
        ),
        on=key,
        how="inner",
    )
    if merged.empty:
        raise ValueError("No paired intervention trials matched")
    if (
        merged["required_graph_fingerprint_x"].astype(str)
        != merged["required_graph_fingerprint_y"].astype(str)
    ).any():
        raise ValueError("Required graph fingerprint mismatch")
    if merged["target_value_x"].astype(int).ne(
        merged["target_value_y"].astype(int)
    ).any():
        raise ValueError("Target value mismatch")

    rows = []
    category_rows = []
    for model_alias, group in merged.groupby("model_alias", sort=True):
        reference_correct = group["reference_correct"].astype(bool)
        treatment_correct = group["treatment_correct"].astype(bool)
        reference_only = int((reference_correct & ~treatment_correct).sum())
        treatment_only = int((~reference_correct & treatment_correct).sum())
        ci_low, ci_high = _bootstrap_delta(reference_correct, treatment_correct)
        rows.append(
            {
                "model_alias": model_alias,
                "pairs": len(group),
                "reference_gating": reference_gating,
                "treatment_gating": treatment_gating,
                "reference_accuracy": float(reference_correct.mean()),
                "treatment_accuracy": float(treatment_correct.mean()),
                "delta": float(treatment_correct.mean() - reference_correct.mean()),
                "reference_only": reference_only,
                "treatment_only": treatment_only,
                "both_correct": int((reference_correct & treatment_correct).sum()),
                "both_wrong": int((~reference_correct & ~treatment_correct).sum()),
                "delta_ci_low": ci_low,
                "delta_ci_high": ci_high,
                "mcnemar_exact_p": _mcnemar_exact_p(
                    reference_only,
                    treatment_only,
                ),
            }
        )
        for setting, column in (
            ("reference", "reference_category"),
            ("treatment", "treatment_category"),
        ):
            counts = group[column].value_counts()
            for category, count in counts.items():
                category_rows.append(
                    {
                        "model_alias": model_alias,
                        "setting": setting,
                        "primary_category": category,
                        "count": int(count),
                    }
                )

    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    effect_summary = pd.DataFrame(rows)
    categories = pd.DataFrame(category_rows)
    write_csv(effect_summary, output / "intervention_effect_summary.csv")
    write_csv(categories, output / "intervention_category_summary.csv")
    payload = {
        "analysis_dir": str(analysis_dir),
        "distractor_count": distractor_count,
        "distractor_mode": distractor_mode,
        "reference_gating": reference_gating,
        "treatment_gating": treatment_gating,
        "models": sorted(effect_summary["model_alias"].tolist()),
    }
    atomic_write_json(output / "intervention_summary.json", payload)
    return payload
