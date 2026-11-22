"""Merge compatible FuncBenchGen analyses into one paper-level table."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from .funcbench_errors import _write_metric_summaries
from .io_utils import atomic_write_json, write_csv


def merge_funcbench_analyses(
    analysis_dirs: list[Path | str],
    *,
    output_dir: Path | str,
    run_name: str,
    graph_indices: list[int] | None = None,
    model_aliases: list[str] | None = None,
) -> dict[str, Any]:
    """Combine analyses that share graph definitions without duplicate trials."""
    frames = [
        pd.read_csv(Path(directory) / "trial_failures.csv")
        for directory in analysis_dirs
    ]
    if not frames:
        raise ValueError("At least one analysis directory is required")
    combined = pd.concat(frames, ignore_index=True)
    if "distractor_mode" not in combined:
        combined["distractor_mode"] = "connected_compatible"
    if "tool_gating" not in combined:
        combined["tool_gating"] = False
    if "repeat_known_variable_values" not in combined:
        combined["repeat_known_variable_values"] = False
    inferred_graph_index = pd.to_numeric(
        combined["trial_id"].astype(str).str.extract(r"-g(\d+)-")[0],
        errors="coerce",
    )
    if "graph_index" in combined:
        combined["graph_index"] = pd.to_numeric(
            combined["graph_index"],
            errors="coerce",
        ).fillna(
            inferred_graph_index
        )
    else:
        combined["graph_index"] = inferred_graph_index
    combined = combined.dropna(subset=["graph_index"]).copy()
    combined["graph_index"] = combined["graph_index"].astype(int)
    if model_aliases is not None:
        combined = combined.loc[
            combined["model_alias"].isin(model_aliases)
        ].copy()
        if combined.empty:
            raise ValueError("No trials remain after model filtering")
    if graph_indices is not None:
        combined = combined.loc[
            combined["graph_index"].isin(graph_indices)
        ].copy()
        if combined.empty:
            raise ValueError("No trials remain after graph-index filtering")
    combined["distractor_mode"] = combined["distractor_mode"].fillna(
        "connected_compatible"
    )
    combined["tool_gating"] = combined["tool_gating"].fillna(False).astype(bool)
    combined["repeat_known_variable_values"] = (
        combined["repeat_known_variable_values"]
        .fillna(False)
        .astype(bool)
    )
    combined["graph_seed"] = pd.to_numeric(
        combined["graph_seed"],
        errors="raise",
    ).astype(int)

    duplicate_key = [
        "model_alias",
        "graph_seed",
        "distractor_count",
        "distractor_mode",
        "tool_gating",
        "repeat_known_variable_values",
    ]
    duplicates = combined.duplicated(duplicate_key, keep=False)
    if duplicates.any():
        duplicate_rows = combined.loc[duplicates, duplicate_key].drop_duplicates()
        raise ValueError(
            "Duplicate merged experiment cells: "
            f"{duplicate_rows.to_dict(orient='records')}"
        )

    graph_groups = combined.groupby(["model_alias", "graph_seed"], dropna=False)
    for key, group in graph_groups:
        required = group["required_graph_fingerprint"].dropna().astype(str)
        if required.nunique() > 1:
            raise ValueError(f"Required graph fingerprint mismatch for {key}")
        targets = pd.to_numeric(group["target_value"], errors="coerce").dropna()
        if targets.nunique() > 1:
            raise ValueError(f"Target value mismatch for {key}")

    combined["source_run"] = combined["run"]
    combined["run"] = run_name
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    write_csv(combined, output / "trial_failures.csv")
    _write_metric_summaries(combined, output)
    payload = {
        "run_name": run_name,
        "analysis_dirs": [str(path) for path in analysis_dirs],
        "trials": len(combined),
        "models": sorted(combined["model_alias"].unique().tolist()),
        "distractor_modes": sorted(combined["distractor_mode"].unique().tolist()),
        "distractor_counts": sorted(
            int(value) for value in combined["distractor_count"].unique()
        ),
        "graph_seeds": sorted(int(value) for value in combined["graph_seed"].unique()),
        "graph_indices": sorted(int(value) for value in combined["graph_index"].unique()),
        "tool_gating": sorted(bool(value) for value in combined["tool_gating"].unique()),
        "model_filter": model_aliases,
    }
    atomic_write_json(output / "merge_summary.json", payload)
    return payload
