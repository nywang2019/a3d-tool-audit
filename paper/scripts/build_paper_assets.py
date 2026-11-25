#!/usr/bin/env python3
"""Build deterministic paper assets from existing FuncBenchGen analyses.

This script performs no model/API calls. It reads only the canonical derived
CSV/JSON files and paired trial tables in the repository, then writes tables,
figures, and a compact machine-readable fact sheet under ``paper/``.
"""

from __future__ import annotations

import json
from collections.abc import Iterable
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import yaml
from matplotlib import pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

from build_method_overview_v2 import build as build_method_overview

matplotlib.use("Agg")


ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"
SUPPLEMENT = PAPER / "supplement"
MANUSCRIPT = PAPER / "manuscript"

MAIN = ROOT / "results" / "funcbench-main20-five-model-analysis-20261101"
DOSE = ROOT / "results" / "funcbench-dose-five-model-20261101"
CONTROL = ROOT / "results" / "funcbench-control-five-model-analysis-20261101"
ASP = CONTROL / "asp"
GATING = ROOT / "results" / "funcbench-gating-five-model-analysis-20261101"
REMINDER = ROOT / "results" / "funcbench-reminder-five-model-analysis-20261101"
SEALED_PROBE = ROOT / "results" / "funcbench-sealed-probe-analysis-20261101"
EAP = SEALED_PROBE / "eap"

MODEL_LABELS = {
    "deepseek-flash": "DeepSeek-V4.1-Flash",
    "kimi-k3.1-pro": "Kimi K3.1 Pro",
    "glm-5.3-flashx": "GLM-5.3-Flashx",
    "minimax-m3.1-flash": "MiniMax-M3.1-Flash",
    "qwen3.8-max": "Qwen3.8-Max",
}
TABLE_MODEL_LABELS = {
    "deepseek-flash": "DeepSeek-V4.1-Flash",
    "kimi-k3.1-pro": "Kimi K3.1 Pro",
    "glm-5.3-flashx": "GLM-5.3-Flashx",
    "minimax-m3.1-flash": "MiniMax-M3.1-Flash",
    "qwen3.8-max": "Qwen3.8-Max",
}
MODEL_ORDER = list(MODEL_LABELS)
COLORS = {
    "deepseek-flash": "#1f77b4",
    "kimi-k3.1-pro": "#d95f02",
    "glm-5.3-flashx": "#7570b3",
    "minimax-m3.1-flash": "#8c564b",
    "qwen3.8-max": "#1b9e77",
}


def ensure_dirs() -> None:
    for path in (TABLES, FIGURES, SUPPLEMENT, MANUSCRIPT):
        path.mkdir(parents=True, exist_ok=True)


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path, keep_default_na=False)


def pct(value: float, digits: int = 1) -> str:
    return f"{100.0 * float(value):.{digits}f}"


def pct_pp(value: float, digits: int = 1) -> str:
    return f"{100.0 * float(value):+.{digits}f}"


def tex_escape(value: object) -> str:
    text = str(value)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    return text


def md_table(headers: list[str], rows: Iterable[Iterable[object]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "|" + "|".join(["---"] + ["---:"] * (len(headers) - 1)) + "|",
    ]
    for row in rows:
        lines.append("| " + " | ".join(str(cell) for cell in row) + " |")
    return "\n".join(lines) + "\n"


def tex_table(
    caption: str,
    label: str,
    headers: list[str],
    rows: Iterable[Iterable[object]],
    *,
    source: str,
    widths: str | None = None,
    footnote: str = "",
    column_sep: str = "3pt",
) -> str:
    column_spec = widths or ("l" + "r" * (len(headers) - 1))
    column_spec = column_spec.replace(
        "X", ">{\\raggedright\\arraybackslash}X"
    )
    def render_header(header: str) -> str:
        if header.startswith("@RAW@"):
            return header.removeprefix("@RAW@")
        return tex_escape(header)

    lines = [
        "\\begin{table}[htbp]",
        "\\centering",
        f"\\caption{{{caption}}}",
        f"\\label{{{label}}}",
        "\\footnotesize",
        f"\\setlength{{\\tabcolsep}}{{{column_sep}}}",
        "\\renewcommand{\\arraystretch}{1.08}",
        f"\\begin{{tabularx}}{{\\textwidth}}{{{column_spec}}}",
        "\\toprule",
        " & ".join(render_header(h) for h in headers) + r" \\",
        "\\midrule",
    ]
    for row in rows:
        lines.append(" & ".join(tex_escape(cell) for cell in row) + r" \\")
    lines.extend(
        [
            "\\bottomrule",
            "\\end{tabularx}",
            (
                "\\par\\parbox{\\linewidth}{\\raggedright\\footnotesize "
                f"Source: \\texttt{{{tex_escape(source).replace('/', '/\\allowbreak{}')}}}."
                "}"
            ),
        ]
    )
    if footnote:
        lines.append(f"\\par\\parbox{{\\linewidth}}{{\\raggedright\\footnotesize {footnote}}}")
    lines.extend(["\\end{table}", ""])
    return "\n".join(lines)


def write_table(
    stem: str,
    *,
    headers: list[str],
    rows: list[list[object]],
    caption: str,
    label: str,
    source: str,
    widths: str | None = None,
    footnote: str = "",
    column_sep: str = "3pt",
) -> Path:
    csv_rows = [dict(zip(headers, row, strict=True)) for row in rows]
    pd.DataFrame(csv_rows).to_csv(TABLES / f"{stem}.csv", index=False)
    (TABLES / f"{stem}.md").write_text(md_table(headers, rows), encoding="utf-8")
    (TABLES / f"{stem}.tex").write_text(
        tex_table(
            caption,
            label,
            headers,
            rows,
            source=source,
            widths=widths,
            footnote=footnote,
            column_sep=column_sep,
        ),
        encoding="utf-8",
    )
    return TABLES / f"{stem}.tex"


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def holm_adjust(values: list[float]) -> list[float]:
    order = sorted(range(len(values)), key=lambda idx: values[idx])
    adjusted = [0.0] * len(values)
    running = 0.0
    m = len(values)
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * values[idx]))
        adjusted[idx] = running
    return adjusted


def graph_cluster_summary(
    frame: pd.DataFrame,
    *,
    cluster_column: str,
    value_column: str,
    seed: int = 20261122,
    draws: int = 20000,
) -> dict[str, float]:
    """Summarize a fixed panel with graph-cluster resampling.

    The model panel is treated as fixed. Resampling and sign flipping operate on
    graph clusters so that model rows sharing a graph are not treated as
    independent observations.
    """

    cluster_values = (
        frame.groupby(cluster_column, sort=True)[value_column].mean().to_numpy()
    )
    estimate = float(cluster_values.mean())
    rng = np.random.default_rng(seed)
    bootstrap = np.array(
        [
            rng.choice(cluster_values, size=len(cluster_values), replace=True).mean()
            for _ in range(draws)
        ]
    )
    signs = rng.choice([-1.0, 1.0], size=(draws, len(cluster_values)))
    permuted = (signs * cluster_values).mean(axis=1)
    p_value = float(
        (np.count_nonzero(np.abs(permuted) >= abs(estimate) - 1e-15) + 1)
        / (draws + 1)
    )
    ci_low, ci_high = np.percentile(bootstrap, [2.5, 97.5])
    return {
        "clusters": int(len(cluster_values)),
        "estimate": estimate,
        "ci_low": float(ci_low),
        "ci_high": float(ci_high),
        "sign_flip_p": p_value,
        "positive_graphs": int((cluster_values > 0).sum()),
        "zero_graphs": int((cluster_values == 0).sum()),
        "negative_graphs": int((cluster_values < 0).sum()),
    }


def build_study_level_summary(trials: pd.DataFrame) -> dict[str, object]:
    """Summarize the existing five-model panel without new model calls."""

    paired = trials.pivot_table(
        index=["model_alias", "graph_index"],
        columns="distractor_count",
        values="correct",
        aggfunc="first",
    )
    paired = paired[[0, 5]].astype(bool)
    paired_delta = (
        paired[5].astype(int) - paired[0].astype(int)
    ).rename("paired_delta").reset_index()
    main = graph_cluster_summary(
        paired_delta,
        cluster_column="graph_index",
        value_column="paired_delta",
    )
    main.update(
        {
            "models": int(paired_delta["model_alias"].nunique()),
            "model_graph_pairs": int(len(paired_delta)),
            "baseline_correct": int(paired[0].sum()),
            "distractor_correct": int(paired[5].sum()),
            "baseline_only": int((paired[0] & ~paired[5]).sum()),
            "distractor_only": int((~paired[0] & paired[5]).sum()),
        }
    )

    asp_scores = read_csv(ASP / "asp_graph_scores.csv")
    asp = graph_cluster_summary(
        asp_scores,
        cluster_column="graph_seed",
        value_column="asp",
    )
    asp["models"] = int(asp_scores["model_alias"].nunique())
    return {"main": main, "asp": asp}


def build_main_table() -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics = read_csv(MAIN / "metric_summary.csv")
    effects = read_csv(MAIN / "paired_effect_summary.csv")
    base = metrics[metrics["distractor_count"] == 0].set_index("model_alias")
    dist = metrics[metrics["distractor_count"] == 5].set_index("model_alias")
    effects = effects.set_index("model_alias").loc[MODEL_ORDER].reset_index()
    effects["holm_p"] = holm_adjust(effects["mcnemar_exact_p"].astype(float).tolist())
    effects = effects[
        [
            "model_alias",
            "pairs",
            "baseline_only",
            "distractor_only",
            "both_correct",
            "both_wrong",
            "delta",
            "delta_ci_low",
            "delta_ci_high",
            "mcnemar_exact_p",
            "holm_p",
        ]
    ]

    rows: list[list[object]] = []
    for row in effects.itertuples(index=False):
        model = row.model_alias
        delta_interval = (
            f"{pct_pp(row.delta)} pp "
            f"[{pct_pp(row.delta_ci_low)}, {pct_pp(row.delta_ci_high)}]"
        )
        rows.append(
            [
                TABLE_MODEL_LABELS[model],
                f"{int(base.loc[model, 'correct'])}/{int(base.loc[model, 'trials'])} ({pct(base.loc[model, 'accuracy'])}%)",
                f"{int(dist.loc[model, 'correct'])}/{int(dist.loc[model, 'trials'])} ({pct(dist.loc[model, 'accuracy'])}%)",
                delta_interval,
                f"{row.mcnemar_exact_p:.4f}",
                f"{row.holm_p:.4f}",
            ]
        )
    write_table(
        "main20_effect",
        headers=[
            "Model",
            "0 extra tools",
            r"@RAW@\shortstack[r]{5 executable\\goal-irrelevant}",
            "Delta [95% CI]",
            "Exact McNemar p",
            "Holm p",
        ],
        rows=rows,
        caption=(
            "Five executable goal-irrelevant tools showed a negative paired "
            "change in exact success in every model."
        ),
        label="tab:main20",
        source=(
            "results/funcbench-main20-five-model-analysis-20261101/"
            "paired_effect_summary.csv"
        ),
        widths="Xrrrrr",
        column_sep="3pt",
        footnote=(
            "Each model has 20 paired graphs. Intervals are paired graph-level "
            "bootstrap intervals. McNemar tests are uncorrected in the source "
            "file; Holm values are derived here."
        ),
    )
    return metrics, effects


def build_trace_table(metrics: pd.DataFrame) -> None:
    rows: list[list[object]] = []
    trace_metrics = [
        ("required_recall", "Req."),
        ("selection_precision", "Sel."),
        ("dependency_order_score", "Order"),
        ("argument_score", "Arg."),
        ("dependency_trace_fidelity", "Fidelity"),
    ]
    for model in MODEL_ORDER:
        for count, setting in ((0, "0"), (5, "5")):
            row = metrics[
                (metrics["model_alias"] == model)
                & (metrics["distractor_count"] == count)
            ].iloc[0]
            rows.append(
                [
                    TABLE_MODEL_LABELS[model],
                    setting,
                    *[f"{float(row[key]):.3f}" for key, _ in trace_metrics],
                ]
            )
    write_table(
        "trace_metrics",
        headers=[
            "Model",
            "Extra tools",
            "Req.",
            "Sel.",
            "Order",
            "Arg.",
            "Fidelity",
        ],
        rows=rows,
        caption=(
            "Structured tool-trace diagnostics for five model aliases on the "
            "20-graph main experiment."
        ),
        label="tab:trace",
        source=(
            "results/funcbench-main20-five-model-analysis-20261101/"
            "metric_summary.csv"
        ),
        widths="Xrrrrrr",
        footnote=(
            "Trace fidelity is the equal-weight mean of required recall, selection "
            "precision, dependency order, and argument score; it is diagnostic, "
            "not a replacement for exact success."
        ),
    )


def build_dose_table() -> pd.DataFrame:
    conditions = read_csv(DOSE / "dose_condition_summary.csv")
    effects = read_csv(DOSE / "dose_effect_summary.csv")
    rows: list[list[object]] = []
    for model in MODEL_ORDER:
        frame = conditions[conditions["model_alias"] == model].set_index(
            "distractor_count"
        )
        effect2 = effects[
            (effects["model_alias"] == model) & (effects["distractor_count"] == 2)
        ].iloc[0]
        effect5 = effects[
            (effects["model_alias"] == model) & (effects["distractor_count"] == 5)
        ].iloc[0]
        cells = [
            f"{int(frame.loc[count, 'correct'])}/{int(frame.loc[count, 'trials'])}"
            for count in (0, 2, 5)
        ]
        rows.append(
            [
                TABLE_MODEL_LABELS[model],
                *cells,
                f"{pct_pp(effect2.delta)} pp [{pct_pp(effect2.delta_ci_low)}, {pct_pp(effect2.delta_ci_high)}]",
                f"{pct_pp(effect5.delta)} pp [{pct_pp(effect5.delta_ci_low)}, {pct_pp(effect5.delta_ci_high)}]",
            ]
        )
    write_table(
        "dose_response",
        headers=[
            "Model",
            "0",
            "2",
            "5",
            "Delta 2-0 [95% CI]",
            "Delta 5-0 [95% CI]",
        ],
        rows=rows,
        caption=(
            "Exact success rate for 0, 2, and 5 executable goal-irrelevant "
            "tools on 15 shared graphs for five model aliases."
        ),
        label="tab:dose",
        source="results/funcbench-dose-five-model-20261101/",
        widths="Xrrrrr",
        footnote=(
            "Cells are correct/trials; deltas are paired proportions in percentage "
            "points. These data indicate model-specific plateaus or threshold-like "
            "patterns, not a common parametric threshold."
        ),
    )
    return conditions


def build_control_table() -> pd.DataFrame:
    metrics = read_csv(CONTROL / "metric_summary.csv")
    mode_order = [
        ("none", "0"),
        ("disconnected", "Disconnected 5"),
        ("connected_incompatible", "Connected-incompatible 5"),
        ("connected_compatible", "Executable goal-irrelevant 5"),
    ]
    rows: list[list[object]] = []
    for model in MODEL_ORDER:
        row: list[object] = [TABLE_MODEL_LABELS[model]]
        for mode, _ in mode_order:
            item = metrics[
                (metrics["model_alias"] == model)
                & (metrics["distractor_mode"] == mode)
            ].iloc[0]
            row.append(
                f"{int(item['correct'])}/{int(item['trials'])} ({pct(item['accuracy'])}%)"
            )
        rows.append(row)
    write_table(
        "affordance_controls",
        headers=["Model", *[label for _, label in mode_order]],
        rows=rows,
        caption=(
            "Tool-usability controls on the same 15 core graphs. All "
            "five-extra-tool conditions expose the same number of tools, and "
            "all five model aliases are included."
        ),
        label="tab:usability-controls",
        source=(
            "results/funcbench-control-five-model-analysis-20261101/"
            "metric_summary.csv"
        ),
        widths="Xrrrr",
        footnote=(
            "Each model has 15 trials per condition. The disconnected and "
            "connected-incompatible controls do not reproduce the degradation "
            "observed with executable goal-irrelevant tools. DeepSeek-V4.1-Flash has one "
            "technical/budget failure in each connected condition."
        ),
    )
    return metrics


def build_asp_tables() -> pd.DataFrame:
    asp = read_csv(ASP / "asp_summary.csv").set_index("model_alias")
    rows: list[list[object]] = []
    solvable_rows: list[list[object]] = []
    for model in MODEL_ORDER:
        item = asp.loc[model]
        rows.append(
            [
                TABLE_MODEL_LABELS[model],
                f"{pct_pp(item['connected_component'])} pp",
                f"{pct_pp(item['callability_component'])} pp",
                (
                    f"{pct_pp(item['asp'])} pp "
                    f"[{pct_pp(item['asp_ci_low'])}, "
                    f"{pct_pp(item['asp_ci_high'])}]"
                ),
                f"{float(item['sign_flip_p']):.3f}",
                f"{float(item['holm_p']):.3f}",
                (
                    f"{int(item['positive_graphs'])}/"
                    f"{int(item['zero_graphs'])}/"
                    f"{int(item['negative_graphs'])}"
                ),
            ]
        )
        solvable_rows.append(
            [
                TABLE_MODEL_LABELS[model],
                int(item["solvable_graphs"]),
                (
                    f"{pct_pp(item['asp_solvable'])} pp "
                    f"[{pct_pp(item['asp_solvable_ci_low'])}, "
                    f"{pct_pp(item['asp_solvable_ci_high'])}]"
                ),
                f"{float(item['asp_solvable_sign_flip_p']):.3f}",
                f"{float(item['asp_solvable_holm_p']):.3f}",
            ]
        )

    write_table(
        "asp",
        headers=[
            "Model",
            "D-GI",
            "I-GI",
            "ASP [95% CI]",
            "Sign-flip p",
            "Holm p",
            "+/0/- graphs",
        ],
        rows=rows,
        caption=(
            "ASP (historical metric name) reports the average paired loss "
            "relative to the same-count disconnected and connected-incompatible "
            "controls."
        ),
        label="tab:asp",
        source=(
            "results/funcbench-control-five-model-analysis-20261101/"
            "asp/asp_summary.csv"
        ),
        widths="Xrrrrrr",
        footnote=(
            "D-GI is the exact success rate under disconnected tools minus that "
            "under executable goal-irrelevant tools; I-GI is the corresponding "
            "difference for connected-incompatible tools. Intervals resample graphs; p-values "
            "use exact two-sided sign flips; Holm adjustment covers five models."
        ),
    )
    write_table(
        "asp_solvable",
        headers=[
            "Model",
            "Solvable graphs",
            "ASP_solvable [95% CI]",
            "Sign-flip p",
            "Holm p",
        ],
        rows=solvable_rows,
        caption=(
            "Sensitivity analysis restricted to graphs solved without extra tools."
        ),
        label="tab:aspsolvable",
        source=(
            "results/funcbench-control-five-model-analysis-20261101/"
            "asp/asp_summary.csv"
        ),
        widths="Xrrrr",
        footnote=(
            "Small solvable subsets, especially for GLM, make this a sensitivity "
            "analysis rather than a primary outcome."
        ),
    )
    return asp.reset_index()


def build_eap_table() -> pd.DataFrame:
    eap = read_csv(EAP / "eap_summary.csv").set_index("model_alias")
    rows: list[list[object]] = []
    for model in MODEL_ORDER:
        item = eap.loc[model]
        rows.append(
            [
                TABLE_MODEL_LABELS[model],
                f"{int(item['executable_correct'])}/{int(item['graphs'])}",
                f"{int(item['sealed_correct'])}/{int(item['graphs'])}",
                (
                    f"{pct_pp(item['eap'])} pp "
                    f"[{pct_pp(item['eap_ci_low'])}, "
                    f"{pct_pp(item['eap_ci_high'])}]"
                ),
                f"{float(item['sign_flip_p']):.3f}",
                f"{float(item['holm_p']):.3f}",
                (
                    f"{int(item['positive_graphs'])}/"
                    f"{int(item['zero_graphs'])}/"
                    f"{int(item['negative_graphs'])}"
                ),
            ]
        )
    write_table(
        "eap",
        headers=[
            "Model",
            "Executable",
            "Sealed",
            "EAP [95% CI]",
            "Sign-flip p",
            "Holm p",
            "+/0/- graphs",
        ],
        rows=rows,
        caption=(
            "EAP (historical metric name) reports the paired loss between "
            "sealed and executable goal-irrelevant tools on 10 shared graphs."
        ),
        label="tab:eap",
        source=(
            "results/funcbench-sealed-probe-analysis-20261101/eap/eap_summary.csv"
        ),
        widths="Xrrrrrr",
        footnote=(
            "Executable and sealed tools have identical names, descriptions, "
            "schemas, order, prompts, required graphs, and target values. Only "
            "runtime execution differs. Positive EAP means that sealing improved the "
            "exact success rate."
        ),
    )
    return eap.reset_index()


def build_experiment_table() -> None:
    def load_experiment(relative: str) -> dict[str, object]:
        data = yaml.safe_load((ROOT / relative).read_text(encoding="utf-8"))
        return dict(data["experiment"])

    main = load_experiment("configs/funcbench_paired_main_v2.yaml")
    extension = load_experiment("configs/funcbench_paired_extension20.yaml")
    dose = load_experiment("configs/funcbench_paired_dose2.yaml")
    dose_extension = load_experiment(
        "configs/funcbench_paired_dose2_extension.yaml"
    )
    control = load_experiment("configs/funcbench_control_disconnected.yaml")
    control_extension = load_experiment(
        "configs/funcbench_control_disconnected_extension15.yaml"
    )
    gating = load_experiment("configs/funcbench_prerequisite_gating.yaml")
    reminder = load_experiment("configs/funcbench_paired_intervention.yaml")
    repeatability = load_experiment("configs/funcbench_repeatability8.yaml")
    sealed_probe = load_experiment("configs/funcbench_sealed10.yaml")

    main_graphs = len(main["graph_indices"]) + len(extension["graph_indices"])
    dose_graphs = len(dose["graph_indices"]) + len(
        dose_extension["graph_indices"]
    )
    control_graphs = len(control["graph_indices"]) + len(
        control_extension["graph_indices"]
    )
    gating_graphs = len(gating["graph_indices"])
    reminder_graphs = len(reminder["graph_indices"])
    repeatability_graphs = len(repeatability["graph_indices"])
    sealed_probe_graphs = len(sealed_probe["graph_indices"])
    specs = [
        (
            "Main",
            main_graphs,
            "0 / 5 executable goal-irrelevant",
            2,
            5,
        ),
        (
            "Dose",
            dose_graphs,
            "0 / 2 / 5 executable goal-irrelevant",
            3,
            5,
        ),
        (
            "Controls",
            control_graphs,
            "0 / disconnected / connected-incompatible / executable goal-irrelevant",
            4,
            5,
        ),
        (
            "Prerequisite hiding",
            gating_graphs,
            "unhidden / hidden",
            2,
            5,
        ),
        (
            "Reminder",
            reminder_graphs,
            "base / reminder",
            2,
            5,
        ),
        (
            "Repeatability",
            repeatability_graphs,
            "0 / 5 executable goal-irrelevant",
            2,
            5,
        ),
        (
            "Sealed probe",
            sealed_probe_graphs,
            "executable / sealed",
            2,
            5,
        ),
    ]
    rows = [
        [
            name,
            graphs,
            conditions,
            model_count,
            graphs * model_count * condition_count,
        ]
        for name, graphs, conditions, condition_count, model_count in specs
    ]
    write_table(
        "experiment_design",
        headers=[
            "Study",
            "Graph clusters",
            "Conditions",
            "Models",
            "Condition-level attempts",
        ],
        rows=rows,
        caption=(
            "The main comparison and supporting studies use the same five models. "
            "Reused baseline cells are not additional independent tasks."
        ),
        label="tab:design",
        source="configs/funcbench_*.yaml",
        widths="XrXrr",
        footnote=(
            "All five model aliases are evaluated in every study. Graph "
            "clusters recur across studies; reused conditions and cells are "
            "not additional independent tasks."
        ),
    )


def build_case_table() -> None:
    trials = read_csv(MAIN / "trial_failures.csv")
    selected = [
        ("D1", "deepseek-flash", "d3-x5-g5-r0", "Executable goal-irrelevant tool"),
        ("K1", "kimi-k3.1-pro", "d3-x5-g0-r0", "Wrong argument value"),
        ("G1", "glm-5.3-flashx", "d3-x5-g2-r0", "Premature dependency"),
        ("M1", "minimax-m3.1-flash", "d3-x5-g1-r0", "Irrelevant function calls"),
        ("Q1", "qwen3.8-max", "d3-x5-g13-r0", "Premature final answer"),
    ]
    rows: list[list[object]] = []
    for case_id, model, trial_id, label in selected:
        item = trials[
            (trials["model_alias"] == model) & (trials["trial_id"] == trial_id)
        ].iloc[0]
        rows.append(
            [
                case_id,
                TABLE_MODEL_LABELS[model],
                label,
                f"{float(item['dependency_trace_fidelity']):.3f}",
                item["trial_id"],
            ]
        )
    write_table(
        "diagnostic_cases",
        headers=[
            "Case",
            "Model",
            "Rule-derived diagnosis",
            "Trace fidelity",
            "Trial ID",
        ],
        rows=rows,
        caption=(
            "Representative saved trajectories used to demonstrate the diagnostic "
            "report. These are illustrative cases, not a human-validated sample."
        ),
        label="tab:cases",
        source=(
            "results/funcbench-main20-five-model-analysis-20261101/"
            "trial_failures.csv"
        ),
        widths="XXrXX",
    )


def build_repeatability_tables() -> None:
    conditions_path = TABLES / "repeatability_conditions_full.csv"
    agreement_path = TABLES / "repeatability_agreement_full.csv"
    if not conditions_path.exists() or not agreement_path.exists():
        raise FileNotFoundError(
            "Run paper/scripts/analyze_repeatability.py before rebuilding the paper."
        )

    conditions = read_csv(conditions_path)
    condition_rows = [
        [
            "DeepSeek-V4.1-Flash" if row["Model"] == "DeepSeek" else row["Model"],
            f"{int(row['Original 0'])}/8",
            f"{int(row['Repeat 0'])}/8",
            f"{int(row['Original 5'])}/8",
            f"{int(row['Repeat 5'])}/8",
            f"{row['Original delta pp']} pp",
            f"{row['Repeat delta pp']} pp",
        ]
        for _, row in conditions.iterrows()
    ]
    write_table(
        "repeatability_conditions",
        headers=[
            "Model",
            "Original 0",
            "Repeat 0",
            "Original 5",
            "Repeat 5",
            "Original delta",
            "Repeat delta",
        ],
        rows=condition_rows,
        caption=(
            "Eight-graph sampling-repeatability audit for five model aliases. "
            "The repeat uses the same required graphs and generator "
            "seed with a fresh model sample."
        ),
        label="tab:repeatability",
        source=(
            "paper/tables/repeatability_conditions_full.csv; original run and "
            "funcbench-paired-repeatability8-20261101T081739Z"
        ),
        widths="Xrrrrrr",
        footnote=(
            "The repeat is a sampling audit, not a new independent graph sample. "
            "Delta is five-minus-zero success in percentage points."
        ),
    )

    agreement = read_csv(agreement_path)
    agreement_rows = [
        [
            "DeepSeek-V4.1-Flash" if row["Model"] == "DeepSeek" else row["Model"],
            row["Original correct"],
            row["Repeat correct"],
            row["Same correctness"],
            row["Correctness kappa"],
            row["Same primary category"],
            row["Repeat 5-0 bootstrap 95% CI pp"],
        ]
        for _, row in agreement.iterrows()
    ]
    write_table(
        "repeatability_agreement",
        headers=[
            "Model",
            "Original",
            "Repeat",
            "Same correctness",
            "Kappa",
            "Same category",
            "Repeat effect 95% CI",
        ],
        rows=agreement_rows,
        caption="Per-model agreement between the original and repeated sample.",
        label="tab:repeatabilityagreement",
        source="paper/tables/repeatability_agreement_full.csv",
        widths="Xrrrrrr",
        footnote=(
            "Agreement compares the same graph/condition/model cells. The repeat "
            "effect interval is a paired graph bootstrap interval in percentage points."
        ),
    )


def build_error_tables() -> tuple[pd.DataFrame, pd.DataFrame]:
    trials = read_csv(MAIN / "trial_failures.csv")
    categories = [
        "correct",
        "wrong_argument_value",
        "premature_dependency",
        "irrelevant_function",
        "premature_final",
        "technical_or_budget",
    ]
    trial_rows: list[list[object]] = []
    for category in categories:
        trial_rows.append(
            [
                {
                    "correct": "Correct",
                    "wrong_argument_value": "Wrong argument value",
                    "premature_dependency": "Premature dependency",
                    "irrelevant_function": "Irrelevant function call",
                    "premature_final": "Premature final answer",
                    "technical_or_budget": "Technical/budget failure",
                }[category],
                *[
                    int(
                        (
                            trials[
                                trials["distractor_count"] == count
                            ]["primary_category"]
                            == category
                        ).sum()
                    )
                    for count in (0, 5)
                ],
            ]
        )
    write_table(
        "error_taxonomy",
        headers=["Primary category", "0 extra tools", "5 executable goal-irrelevant tools"],
        rows=trial_rows,
        caption=(
            "Rule-derived primary categories in the 200 main-experiment trials "
            "for five model aliases. All table cells sum to 200 trials."
        ),
        label="tab:errors",
        source=(
            "results/funcbench-main20-five-model-analysis-20261101/"
            "trial_failures.csv"
        ),
        widths="Xrr",
        footnote=(
            "Each trial receives one primary category. Counts are diagnostic and "
            "are not human-validated labels."
        ),
    )

    trials["exact_dependency_set"] = as_bool(trials["exact_dependency_set"])
    trials["dependency_order_score"] = pd.to_numeric(
        trials["dependency_order_score"], errors="coerce"
    )
    trials["argument_score"] = pd.to_numeric(trials["argument_score"], errors="coerce")
    trials["correct"] = as_bool(trials["correct"])
    trials["structural_trace_complete"] = (
        trials["exact_dependency_set"]
        & (trials["dependency_order_score"] >= 0.999999)
        & (trials["argument_score"] >= 0.999999)
    )
    trace_rows: list[list[object]] = []
    for model in MODEL_ORDER:
        for count in (0, 5):
            frame = trials[
                (trials["model_alias"] == model)
                & (trials["distractor_count"] == count)
            ]
            technical = int((frame["outcome"] == "technical_failure").sum())
            valid = frame[frame["outcome"] != "technical_failure"]
            complete_trace = frame[frame["structural_trace_complete"]]
            wrong_with_complete_trace = int((~complete_trace["correct"]).sum())
            trace_rows.append(
                [
                    TABLE_MODEL_LABELS[model],
                    count,
                    len(frame),
                    technical,
                    f"{int(valid['correct'].sum())}/{len(valid)}",
                    f"{len(complete_trace)}/{len(frame)}",
                    wrong_with_complete_trace,
                ]
            )
    write_table(
        "valid_trace_summary",
        headers=[
            "Model",
            "Extra tools",
            "Attempts",
            "Technical/budget failures",
            "Success among valid runs",
            "Complete structural trace",
            "Wrong despite complete trace",
        ],
        rows=trace_rows,
        caption="Sensitivity to technical/budget failures and end-to-end trace completeness.",
        label="tab:validtrace",
        source=(
            "results/funcbench-main20-five-model-analysis-20261101/"
            "trial_failures.csv"
        ),
        widths="Xrrrrrr",
        footnote=(
            "A complete structural trace requires the exact required-function "
            "set, dependency order score 1, and argument score 1. The table is "
            "supplementary because final-answer correctness remains the primary outcome."
        ),
    )
    return trials, pd.DataFrame(trace_rows, columns=[
        "model", "extra_tools", "attempts", "technical_failures",
        "valid_success", "complete_trace", "wrong_complete_trace",
    ])


def build_intervention_table(metrics: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    gating_effects = read_csv(GATING / "intervention" / "intervention_effect_summary.csv")
    gating = (
        gating_effects.set_index("model_alias")
        .loc[MODEL_ORDER]
        .reset_index()
    )
    reminder_trials = read_csv(REMINDER / "trial_failures.csv")
    reminder_trials["correct"] = as_bool(reminder_trials["correct"])
    reminder_trials["repeat_known_variable_values"] = as_bool(
        reminder_trials["repeat_known_variable_values"]
    )

    reminder_rows: list[list[object]] = []
    for model in MODEL_ORDER:
        frame = reminder_trials[reminder_trials["model_alias"] == model]
        base = frame[~frame["repeat_known_variable_values"]].set_index("graph_fingerprint")
        treat = frame[frame["repeat_known_variable_values"]].set_index("graph_fingerprint")
        joined = base.join(
            treat, suffixes=("_base", "_treat"), how="inner"
        )
        wrong_to_right = int(((~joined["correct_base"]) & joined["correct_treat"]).sum())
        right_to_wrong = int((joined["correct_base"] & (~joined["correct_treat"])).sum())
        reminder_rows.append(
            [
                model,
                int(joined["correct_base"].sum()),
                int(joined["correct_treat"].sum()),
                wrong_to_right,
                right_to_wrong,
            ]
        )

    write_table(
        "interventions",
        headers=[
            "Model",
            "Prerequisite hiding: unhidden -> hidden",
            "Hiding delta",
            "Reminder: baseline -> treated",
            "W->R / R->W",
        ],
        rows=[
            [
                TABLE_MODEL_LABELS[row.model_alias],
                (
                    f"{round(row.reference_accuracy * row.pairs)}/{int(row.pairs)} -> "
                    f"{round(row.treatment_accuracy * row.pairs)}/{int(row.pairs)}"
                ),
                f"{pct_pp(row.delta)} pp",
                (
                    f"{reminder_rows[MODEL_ORDER.index(row.model_alias)][1]}/4 -> "
                    f"{reminder_rows[MODEL_ORDER.index(row.model_alias)][2]}/4"
                ),
                (
                    f"{reminder_rows[MODEL_ORDER.index(row.model_alias)][3]} / "
                    f"{reminder_rows[MODEL_ORDER.index(row.model_alias)][4]}"
                ),
            ]
            for row in gating.itertuples(index=False)
        ],
        caption=(
            "Two same-graph interventions for five model aliases: "
            "prerequisite hiding on eight graphs and a state reminder on four graphs."
        ),
        label="tab:interventions",
        source=(
            "results/funcbench-gating-five-model-analysis-20261101/intervention/"
            "intervention_effect_summary.csv and "
            "results/funcbench-reminder-five-model-analysis-20261101/"
            "trial_failures.csv"
        ),
        widths="XrXXX",
        footnote=(
            "W->R denotes wrong-to-right and R->W denotes right-to-wrong. Neither "
            "intervention yields a positive net repair."
        ),
    )
    return gating, pd.DataFrame(
        reminder_rows,
        columns=["model", "baseline_correct", "reminder_correct", "wrong_to_right", "right_to_wrong"],
    )


def build_gating_error_transitions() -> pd.DataFrame:
    trials = read_csv(GATING / "trial_failures.csv")
    trials["tool_gating"] = as_bool(trials["tool_gating"])
    reference = trials[~trials["tool_gating"]].set_index(
        ["model_alias", "graph_seed"]
    )
    treatment = trials[trials["tool_gating"]].set_index(
        ["model_alias", "graph_seed"]
    )
    joined = reference.join(
        treatment,
        suffixes=("_reference", "_treatment"),
        how="inner",
    ).reset_index()
    transition_rows: list[list[object]] = []
    for model in MODEL_ORDER:
        frame = joined[joined["model_alias"] == model]
        changed = frame[
            frame["primary_category_reference"]
            != frame["primary_category_treatment"]
        ]
        pairs = [
            (
                row.primary_category_reference,
                row.primary_category_treatment,
            )
            for row in changed.itertuples(index=False)
        ]
        for transition, count in pd.Series(pairs).value_counts().items():
            if not isinstance(transition, tuple):
                continue
            transition_rows.append(
                [MODEL_LABELS[model], transition[0], transition[1], int(count)]
            )
    result = pd.DataFrame(
        transition_rows,
        columns=["model", "reference_category", "hidden_category", "count"],
    )
    result.to_csv(SUPPLEMENT / "gating_error_transitions.csv", index=False)
    lines = [
        "# Prerequisite-hiding error transitions",
        "",
        "Only trials whose primary category changed after graph-matched prerequisite "
        "hiding are shown.",
        "",
        md_table(
            ["Model", "Unhidden category", "Hidden category", "Count"],
            result.itertuples(index=False, name=None),
        ),
    ]
    (SUPPLEMENT / "gating_error_transitions.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )
    return result


def build_figures(
    main_metrics: pd.DataFrame,
    dose_conditions: pd.DataFrame,
    control_metrics: pd.DataFrame,
    trials: pd.DataFrame,
    asp: pd.DataFrame,
    eap: pd.DataFrame,
) -> None:
    plt.rcParams.update(
        {
            "font.size": 9,
            "axes.spines.top": False,
            "axes.spines.right": False,
            "figure.dpi": 180,
            "savefig.bbox": "tight",
        }
    )

    # Story figure: a valid call can still create an invalid trajectory.
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(8.8, 4.2),
        gridspec_kw={"width_ratios": [1.15, 1.0]},
        constrained_layout=True,
    )
    for ax in (ax_left, ax_right):
        ax.set_xlim(0, 10)
        ax.set_ylim(0, 6)
        ax.axis("off")

    ax_left.text(
        0.0,
        5.75,
        "A  Same task, different extra tools",
        ha="left",
        va="top",
        fontsize=12,
        weight="bold",
    )
    condition_rows = [
        ("0 extra tools", "required chain only", "#dbeafe", "#1d4ed8"),
        ("+5 disconnected", "dead ends", "#f1f5f9", "#64748b"),
        ("+5 incompatible", "type-blocked routes", "#f3e8ff", "#7e22ce"),
        ("+5 executable\n   goal-irrelevant", "an irrelevant branch\ncan run", "#ffedd5", "#c2410c"),
    ]
    for row_index, (title, detail, fill, edge) in enumerate(condition_rows):
        y = 4.55 - row_index * 1.25
        ax_left.text(
            0.15,
            y + 0.28,
            title,
            ha="left",
            va="center",
            fontsize=9.5,
            weight="bold",
        )
        ax_left.text(
            0.15,
            y - 0.12,
            detail,
            ha="left",
            va="center",
            fontsize=8.8,
            color="#475569",
        )
        chain_x = [3.15, 4.25, 5.35, 6.45]
        for index, x in enumerate(chain_x):
            rect = FancyBboxPatch(
                (x, y - 0.24),
                0.72,
                0.48,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor="#dbeafe",
                edgecolor="#2563eb",
                linewidth=0.7,
            )
            ax_left.add_patch(rect)
            ax_left.text(
                x + 0.36,
                y,
                f"R{index + 1}",
                ha="center",
                va="center",
                fontsize=8,
                color="#1e3a8a",
            )
            if index < len(chain_x) - 1:
                ax_left.annotate(
                    "",
                    xy=(chain_x[index + 1] - 0.04, y),
                    xytext=(x + 0.76, y),
                    arrowprops={"arrowstyle": "->", "lw": 0.7, "color": "#64748b"},
                )
        if row_index > 0:
            rect = FancyBboxPatch(
                (7.65, y - 0.24),
                0.58,
                0.48,
                boxstyle="round,pad=0.02,rounding_size=0.06",
                facecolor=fill,
                edgecolor=edge,
                linewidth=1.0,
            )
            ax_left.add_patch(rect)
            ax_left.text(
                7.94,
                y,
                "D",
                ha="center",
                va="center",
                fontsize=8,
                color=edge,
                weight="bold",
            )
            if row_index >= 2:
                ax_left.annotate(
                    "",
                    xy=(7.61, y),
                    xytext=(6.24, y),
                    arrowprops={
                        "arrowstyle": "-|>",
                        "lw": 0.8,
                        "color": edge,
                        "linestyle": "--" if row_index == 2 else "-",
                    },
                )
            if row_index == 2:
                ax_left.text(
                    6.93,
                    y + 0.30,
                    "type block",
                    ha="center",
                    va="center",
                    fontsize=7,
                    color=edge,
                )
    ax_left.text(
        5.0,
        0.38,
        "Required graph and target stay fixed",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#334155",
    )

    ax_right.text(
        0.0,
        5.75,
        "B  A valid call, an invalid trajectory",
        ha="left",
        va="top",
        fontsize=12,
        weight="bold",
    )
    trace_boxes = [
        (0.45, 4.45, 2.15, 0.78, "Start", "goal state"),
        (2.95, 4.45, 2.45, 0.78, "D1 call", "valid schema\nand value"),
        (5.90, 4.45, 2.55, 0.78, "Returns z", "state now has z"),
        (2.95, 2.65, 2.45, 0.80, "R2 expects", "R1 output,\nnot z"),
        (5.90, 2.65, 2.55, 0.80, "Trajectory fails", "target is not\nreached"),
    ]
    colors = ["#dbeafe", "#ffedd5", "#ffedd5", "#e2e8f0", "#fee2e2"]
    edges = ["#2563eb", "#c2410c", "#c2410c", "#475569", "#dc2626"]
    for (x, y, width, height, title, detail), fill, edge in zip(
        trace_boxes, colors, edges, strict=True
    ):
        rect = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.04,rounding_size=0.08",
            facecolor=fill,
            edgecolor=edge,
            linewidth=1.0,
        )
        ax_right.add_patch(rect)
        ax_right.text(
            x + width / 2,
            y + 0.50,
            title,
            ha="center",
            va="center",
            fontsize=9.5,
            weight="bold",
            color=edge,
        )
        ax_right.text(
            x + width / 2,
            y + 0.23,
            detail,
            ha="center",
            va="center",
            fontsize=8.0,
            color="#334155",
        )
    arrows = [
        ((2.64, 4.84), (2.91, 4.84)),
        ((5.44, 4.84), (5.86, 4.84)),
        ((7.18, 4.40), (7.18, 3.50)),
        ((5.44, 3.05), (5.86, 3.05)),
    ]
    for start, end in arrows:
        ax_right.add_patch(
            FancyArrowPatch(
                start,
                end,
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.1,
                color="#64748b",
            )
        )
    ax_right.text(
        1.65,
        1.55,
        "All individual calls can be valid.",
        ha="left",
        va="center",
        fontsize=9,
        color="#334155",
    )
    ax_right.text(
        5.1,
        1.15,
        "The required dependency path is not.",
        ha="left",
        va="center",
        fontsize=9,
        color="#b91c1c",
        weight="bold",
    )
    fig.savefig(FIGURES / "figure1_wrong_tool.png")
    fig.savefig(FIGURES / "figure1_wrong_tool.svg")
    fig.savefig(FIGURES / "figure1_wrong_tool.pdf")
    plt.close(fig)

    # Story figure: end-to-end A3D pipeline.
    fig, ax = plt.subplots(figsize=(9.2, 4.2))
    ax.set_xlim(0, 14)
    ax.set_ylim(0, 7)
    ax.axis("off")

    def pipeline_box(
        x: float,
        y: float,
        width: float,
        height: float,
        title: str,
        body: str,
        accent: str,
        fill: str,
    ) -> None:
        patch = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            facecolor=fill,
            edgecolor=accent,
            linewidth=1.1,
        )
        ax.add_patch(patch)
        ax.text(
            x + width / 2,
            y + height * 0.68,
            title,
            ha="center",
            va="center",
            fontsize=9.2,
            weight="bold",
            color=accent,
        )
        ax.text(
            x + width / 2,
            y + height * 0.32,
            body,
            ha="center",
            va="center",
            fontsize=7.8,
            color="#334155",
            linespacing=1.25,
        )

    ax.text(
        0.0,
        6.65,
        "A3D: from a fixed task to auditable evidence",
        ha="left",
        va="center",
        fontsize=12.5,
        weight="bold",
    )
    top_y = 4.45
    boxes = [
        (0.25, "Task", "graph seed + target\n+ required closure", "#1d4ed8", "#dbeafe"),
        (3.0, "Tool conditions", "same count;\nchange tool usability", "#2563eb", "#eff6ff"),
        (5.75, "Execution", "model calls + returns\n+ exact outcome", "#475569", "#f1f5f9"),
        (8.5, "Trace evidence", "first decisive error\n+ category", "#0f766e", "#ccfbf1"),
        (11.25, "Effects", "paired loss\n+ ASP / EAP", "#7e22ce", "#f3e8ff"),
    ]
    for x, title, body, accent, fill in boxes:
        pipeline_box(x, top_y, 2.25, 1.35, title, body, accent, fill)
    for x in (2.54, 5.29, 8.04, 10.79):
        ax.add_patch(
            FancyArrowPatch(
                (x, top_y + 0.68),
                (x + 0.40, top_y + 0.68),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color="#64748b",
            )
        )

    ax.text(
        0.25,
        3.65,
        "Checks at each stage",
        ha="left",
        va="center",
        fontsize=10,
        weight="bold",
    )
    bottom_boxes = [
        (0.25, "Invariant audit", "same closure, prompt,\norder and target", "#1d4ed8", "#dbeafe"),
        (3.0, "Tool-usability\ncontrol", "same tool count;\nchange tool usability", "#2563eb", "#eff6ff"),
        (5.75, "Saved evidence", "calls, arguments,\nreturns and errors", "#475569", "#f1f5f9"),
        (8.5, "Graph pairing", "compare conditions\non the same task", "#0f766e", "#ccfbf1"),
        (11.25, "Decision", "paired result +\nlocalized failure", "#7e22ce", "#f3e8ff"),
    ]
    for x, title, body, accent, fill in bottom_boxes:
        pipeline_box(x, 1.65, 2.25, 1.35, title, body, accent, fill)
    for x in (2.54, 5.29, 8.04, 10.79):
        ax.add_patch(
            FancyArrowPatch(
                (x, 2.32),
                (x + 0.40, 2.32),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.2,
                color="#64748b",
            )
        )
    ax.text(
        7.0,
        0.72,
        "The output separates a performance change from its first decisive error.",
        ha="center",
        va="center",
        fontsize=9.2,
        color="#334155",
    )
    fig.savefig(FIGURES / "figure2_a3d_framework.png")
    fig.savefig(FIGURES / "figure2_a3d_framework.svg")
    fig.savefig(FIGURES / "figure2_a3d_framework.pdf")
    plt.close(fig)

    # Main effect.
    main_effects = read_csv(MAIN / "paired_effect_summary.csv").set_index(
        "model_alias"
    )
    fig, (ax_left, ax_right) = plt.subplots(
        1,
        2,
        figsize=(7.6, 3.8),
        gridspec_kw={"width_ratios": [1.42, 1.0]},
        constrained_layout=True,
    )
    y = np.arange(len(MODEL_ORDER))
    for yi, model in enumerate(MODEL_ORDER):
        baseline = float(
            main_metrics[
                (main_metrics["model_alias"] == model)
                & (main_metrics["distractor_count"] == 0)
            ]["accuracy"].iloc[0]
        )
        compatible = float(
            main_metrics[
                (main_metrics["model_alias"] == model)
                & (main_metrics["distractor_count"] == 5)
            ]["accuracy"].iloc[0]
        )
        color = COLORS[model]
        ax_left.plot(
            [baseline, compatible],
            [yi, yi],
            color=color,
            linewidth=4.0,
            alpha=0.38,
            solid_capstyle="round",
            zorder=1,
        )
        ax_left.scatter(
            baseline,
            yi,
            marker="o",
            s=78,
            facecolor="white",
            edgecolor=color,
            linewidth=2.0,
            zorder=3,
        )
        ax_left.scatter(
            compatible,
            yi,
            marker="D",
            s=78,
            facecolor=color,
            edgecolor="white",
            linewidth=1.0,
            zorder=3,
        )
        ax_left.text(
            1.03,
            yi,
            f"{round(baseline * 20)}/20 → {round(compatible * 20)}/20",
            ha="left",
            va="center",
            color=color,
            fontsize=10,
        )

        delta = 100 * float(main_effects.loc[model, "delta"])
        low = 100 * float(main_effects.loc[model, "delta_ci_low"])
        high = 100 * float(main_effects.loc[model, "delta_ci_high"])
        ax_right.hlines(yi, low, high, color=color, linewidth=1.8, zorder=2)
        ax_right.plot(
            delta,
            yi,
            marker="o",
            markersize=7,
            color=color,
            markeredgecolor="white",
            markeredgewidth=1.0,
            linestyle="None",
            zorder=3,
        )
        ax_right.text(
            high + 2.0,
            yi,
            f"{delta:+.1f}",
            ha="left",
            va="center",
            color=color,
            fontsize=10,
        )

    ax_left.set_xlim(0, 1.38)
    ax_left.set_ylim(len(MODEL_ORDER) - 0.5, -0.5)
    ax_left.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_left.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax_left.set_xlabel("Exact success rate (%)")
    ax_left.set_yticks(y, [MODEL_LABELS[m] for m in MODEL_ORDER])
    ax_left.set_title("A  Exact success rate", loc="left", pad=12)
    ax_left.grid(axis="y", color="#dbe3ed", linewidth=0.8)
    ax_left.set_axisbelow(True)
    ax_left.tick_params(length=0, pad=7)
    for side in ("top", "right", "left"):
        ax_left.spines[side].set_visible(False)

    ax_right.axvline(0, color="#94a3b8", linewidth=1)
    ax_right.set_xlim(-75, 25)
    ax_right.set_ylim(len(MODEL_ORDER) - 0.5, -0.5)
    ax_right.set_xlabel("Paired difference (percentage points)")
    ax_right.set_title("B  Paired change with 95% CI", loc="left", pad=12)
    ax_right.grid(axis="y", color="#dbe3ed", linewidth=0.8)
    ax_right.set_axisbelow(True)
    ax_right.tick_params(length=0, pad=7, labelleft=False)
    for side in ("top", "right", "left"):
        ax_right.spines[side].set_visible(False)
    ax_left.legend(
        handles=[
            plt.Line2D(
                [],
                [],
                marker="o",
                linestyle="None",
                markerfacecolor="white",
                markeredgecolor="#5b6678",
                markeredgewidth=2.0,
                markersize=7,
                label="No extra tools",
            ),
            plt.Line2D(
                [],
                [],
                marker="D",
                linestyle="None",
                markerfacecolor="#5b6678",
                markeredgecolor="white",
                markersize=7,
                label="5 executable goal-irrelevant",
            ),
        ],
        loc="lower left",
        bbox_to_anchor=(0.0, -0.27),
        ncol=2,
        frameon=False,
        fontsize=8,
    )
    fig.savefig(FIGURES / "figure1_main20_effect.png")
    fig.savefig(FIGURES / "figure1_main20_effect.svg")
    fig.savefig(FIGURES / "figure1_main20_effect.pdf")
    plt.close(fig)

    # Framework overview.
    fig, ax = plt.subplots(figsize=(8.2, 3.6))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 5)
    ax.axis("off")

    def box(x, y, w, h, title, body, color):
        patch = FancyBboxPatch(
            (x, y),
            w,
            h,
            boxstyle="round,pad=0.06,rounding_size=0.12",
            linewidth=0.8,
            edgecolor="#333333",
            facecolor=color,
        )
        ax.add_patch(patch)
        ax.text(x + w / 2, y + h - 0.28, title, ha="center", va="top", weight="bold", fontsize=9)
        ax.text(
            x + w / 2,
            y + h / 2 - 0.1,
            body,
            ha="center",
            va="center",
            fontsize=7.5,
            linespacing=1.25,
        )

    def arrow(x1, y1, x2, y2):
        ax.add_patch(
            FancyArrowPatch(
                (x1, y1),
                (x2, y2),
                arrowstyle="-|>",
                mutation_scale=12,
                linewidth=1.1,
                color="#555555",
            )
        )

    box(
        0.25,
        2.55,
        2.0,
        1.55,
        "Required graph",
        "Core DAG + target truth\n+ inputs + prompt",
        "#dbeafe",
    )
    box(
        2.65,
        2.55,
        2.25,
        1.55,
        "Tool-usability conditions",
        "0 / disconnected /\nincompatible / goal-irrelevant",
        "#fef3c7",
    )
    box(
        5.35,
        2.55,
        2.15,
        1.55,
        "Saved trajectory",
        "calls, arguments,\nreturns, final answer",
        "#dcfce7",
    )
    box(
        7.95,
        2.55,
        1.8,
        1.55,
        "A3D report",
        "first decisive error\n+ checks",
        "#fce7f3",
    )
    box(
        2.65,
        0.45,
        2.25,
        1.35,
        "Invariant audit",
        "required fingerprint\n+ exact target truth",
        "#e0f2fe",
    )
    box(
        5.35,
        0.45,
        2.15,
        1.35,
        "Trace evidence",
        "recall / precision /\norder / argument / stop",
        "#ecfccb",
    )
    arrow(2.25, 3.35, 2.65, 3.35)
    arrow(4.9, 3.35, 5.35, 3.35)
    arrow(7.5, 3.35, 7.95, 3.35)
    arrow(3.75, 2.55, 3.75, 1.8)
    arrow(4.9, 1.12, 5.35, 1.12)
    arrow(6.42, 1.8, 6.42, 2.55)
    ax.text(
        5.0,
        4.55,
        "A3D matched diagnosis",
        ha="center",
        va="center",
        fontsize=12,
        weight="bold",
    )
    fig.savefig(FIGURES / "figure0_a3d_framework.png")
    fig.savefig(FIGURES / "figure0_a3d_framework.svg")
    fig.savefig(FIGURES / "figure0_a3d_framework.pdf")
    plt.close(fig)

    # Dose response.
    fig, ax = plt.subplots(figsize=(6.3, 3.4))
    for model in MODEL_ORDER:
        frame = dose_conditions[dose_conditions["model_alias"] == model].sort_values(
            "distractor_count"
        )
        ax.plot(
            frame["distractor_count"],
            frame["accuracy"],
            marker="o",
            linewidth=1.8,
            color=COLORS[model],
            label=MODEL_LABELS[model],
        )
    ax.set_xticks([0, 2, 5])
    ax.set_xlabel("Executable goal-irrelevant tools")
    ax.set_ylabel("Exact success rate")
    ax.set_ylim(-0.03, 1.05)
    ax.yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax.legend(
        frameon=False,
        ncol=2,
        fontsize=8,
        loc="lower center",
        bbox_to_anchor=(0.5, 1.01),
    )
    fig.subplots_adjust(top=0.82)
    fig.savefig(FIGURES / "figure2_dose_response.png")
    fig.savefig(FIGURES / "figure2_dose_response.svg")
    fig.savefig(FIGURES / "figure2_dose_response.pdf")
    plt.close(fig)

    # Tool-usability controls.
    modes = [
        "none",
        "disconnected",
        "connected_incompatible",
        "connected_compatible",
    ]
    columns = [
        "0\n4 tools",
        "Disconnected\n4 + 5",
        "Incompatible\n4 + 5",
        "Compatible\n4 + 5",
    ]
    matrix = np.array(
        [
            [
                float(
                    control_metrics[
                        (control_metrics["model_alias"] == model)
                        & (control_metrics["distractor_mode"] == mode)
                    ]["accuracy"].iloc[0]
                )
                for mode in modes
            ]
            for model in MODEL_ORDER
        ]
    )
    cmap = LinearSegmentedColormap.from_list(
        "success_scale",
        ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#3182bd"],
    )
    fig, ax = plt.subplots(figsize=(7.6, 3.9), constrained_layout=True)
    image = ax.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax.set_xticks(np.arange(len(columns)), columns)
    ax.set_yticks(np.arange(len(MODEL_ORDER)), [MODEL_LABELS[m] for m in MODEL_ORDER])
    ax.tick_params(length=0)
    ax.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax.set_yticks(np.arange(-0.5, len(MODEL_ORDER), 1), minor=True)
    ax.grid(which="minor", color="white", linewidth=2.5)
    ax.tick_params(which="minor", bottom=False, left=False)
    for spine in ax.spines.values():
        spine.set_visible(False)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            count = round(value * 15)
            color = "white" if value >= 0.67 else "#172033"
            ax.text(
                col,
                row,
                f"{count}/15\n{100 * value:.1f}%",
                ha="center",
                va="center",
                color=color,
                fontsize=10,
                fontweight="bold" if col == 3 else "normal",
            )
    for row, model in enumerate(MODEL_ORDER):
        change = 100 * (matrix[row, 3] - matrix[row, 0])
        ax.text(
            3.68,
            row,
            f"{change:+.1f} pp",
            ha="left",
            va="center",
            color="#b42318",
            fontsize=10,
            fontweight="bold" if change <= -25 else "normal",
        )
    ax.set_xlim(-0.5, 4.15)
    ax.text(
        4.0,
        -0.78,
        "Pair change\nvs 0 extra tools",
        ha="left",
        va="center",
        fontsize=10,
        color="#5b6678",
    )
    colorbar = fig.colorbar(image, ax=ax, fraction=0.025, pad=0.02)
    colorbar.set_label("Exact success rate", fontsize=10)
    colorbar.ax.yaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(1.0)
    )
    fig.savefig(FIGURES / "figure3_affordance_controls.png")
    fig.savefig(FIGURES / "figure3_affordance_controls.svg")
    fig.savefig(FIGURES / "figure3_affordance_controls.pdf")
    plt.close(fig)

    # Core evidence: the paired loss and the equal-count control comparison.
    fig, (ax_main, ax_control) = plt.subplots(
        1,
        2,
        figsize=(8.8, 4.2),
        gridspec_kw={"width_ratios": [0.82, 1.18]},
        constrained_layout=True,
    )
    y = np.arange(len(MODEL_ORDER))
    for yi, model in enumerate(MODEL_ORDER):
        frame = main_metrics[main_metrics["model_alias"] == model]
        baseline = float(
            frame[frame["distractor_count"] == 0]["accuracy"].iloc[0]
        )
        compatible = float(
            frame[frame["distractor_count"] == 5]["accuracy"].iloc[0]
        )
        color = COLORS[model]
        ax_main.plot(
            [baseline, compatible],
            [yi, yi],
            color=color,
            linewidth=2.0,
            alpha=0.45,
            solid_capstyle="round",
        )
        ax_main.scatter(
            baseline,
            yi,
            marker="o",
            s=45,
            facecolor="white",
            edgecolor=color,
            linewidth=1.3,
            zorder=3,
        )
        ax_main.scatter(
            compatible,
            yi,
            marker="D",
            s=40,
            facecolor=color,
            edgecolor="white",
            linewidth=0.8,
            zorder=3,
        )
        ax_main.text(
            1.03,
            yi,
            f"{round(baseline * 20)} -> {round(compatible * 20)}",
            ha="left",
            va="center",
            fontsize=8.0,
            color=color,
        )
    ax_main.set_xlim(0, 1.35)
    ax_main.set_ylim(len(MODEL_ORDER) - 0.5, -0.5)
    ax_main.set_xticks([0, 0.25, 0.5, 0.75, 1.0])
    ax_main.xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    ax_main.set_xlabel("Exact success rate")
    ax_main.set_yticks(
        y,
        [MODEL_LABELS[model].replace(" ", "\n", 1) for model in MODEL_ORDER],
        fontsize=8,
    )
    ax_main.set_title("A  Paired exact success", loc="left", pad=10)
    ax_main.grid(axis="y", color="#e2e8f0", linewidth=0.7)
    ax_main.set_axisbelow(True)

    modes = [
        "none",
        "disconnected",
        "connected_incompatible",
        "connected_compatible",
    ]
    columns = ["0", "D5", "I5", "E5"]
    matrix = np.array(
        [
            [
                float(
                    control_metrics[
                        (control_metrics["model_alias"] == model)
                        & (control_metrics["distractor_mode"] == mode)
                    ]["accuracy"].iloc[0]
                )
                for mode in modes
            ]
            for model in MODEL_ORDER
        ]
    )
    cmap = LinearSegmentedColormap.from_list(
        "core_success_scale",
        ["#f7fbff", "#deebf7", "#c6dbef", "#9ecae1", "#6baed6", "#3182bd"],
    )
    image = ax_control.imshow(matrix, cmap=cmap, vmin=0, vmax=1, aspect="auto")
    ax_control.set_xticks(np.arange(len(columns)), columns)
    ax_control.set_yticks(
        np.arange(len(MODEL_ORDER)),
        [MODEL_LABELS[model] for model in MODEL_ORDER],
        fontsize=8,
    )
    ax_control.tick_params(length=0)
    ax_control.set_xticks(np.arange(-0.5, len(columns), 1), minor=True)
    ax_control.set_yticks(np.arange(-0.5, len(MODEL_ORDER), 1), minor=True)
    ax_control.grid(which="minor", color="white", linewidth=2.0)
    ax_control.tick_params(which="minor", bottom=False, left=False)
    for spine in ax_control.spines.values():
        spine.set_visible(False)
    for row in range(matrix.shape[0]):
        for col in range(matrix.shape[1]):
            value = matrix[row, col]
            ax_control.text(
                col,
                row,
                f"{round(value * 15)}/15",
                ha="center",
                va="center",
                fontsize=8,
                color="white" if value >= 0.55 else "#172033",
                fontweight="bold" if col == 3 else "normal",
            )
    ax_control.set_title("B  Same tool count, changed tool usability", loc="left", pad=10)
    colorbar = fig.colorbar(image, ax=ax_control, fraction=0.028, pad=0.015)
    colorbar.ax.yaxis.set_major_formatter(
        matplotlib.ticker.PercentFormatter(1.0)
    )
    colorbar.ax.tick_params(labelsize=8)
    fig.savefig(FIGURES / "figure3_core_evidence.png")
    fig.savefig(FIGURES / "figure3_core_evidence.svg")
    fig.savefig(FIGURES / "figure3_core_evidence.pdf")
    plt.close(fig)

    # ASP forest plot and baseline-vs-penalty profile.
    asp_by_model = asp.set_index("model_alias")
    baseline = control_metrics[
        control_metrics["distractor_mode"] == "none"
    ].set_index("model_alias")
    fig, axes = plt.subplots(1, 2, figsize=(8.8, 3.8))
    positions = np.arange(len(MODEL_ORDER))
    annotation_offsets = {
        "deepseek-flash": (4, 4),
        "kimi-k3.1-pro": (-18, 6),
        "glm-5.3-flashx": (4, 4),
        "minimax-m3.1-flash": (4, 4),
        "qwen3.8-max": (4, -10),
    }
    for index, model in enumerate(MODEL_ORDER):
        item = asp_by_model.loc[model]
        value = 100.0 * float(item["asp"])
        lower = value - 100.0 * float(item["asp_ci_low"])
        upper = 100.0 * float(item["asp_ci_high"]) - value
        axes[0].errorbar(
            value,
            index,
            xerr=np.array([[lower], [upper]]),
            fmt="o",
            color=COLORS[model],
            capsize=3,
        )
        base = float(baseline.loc[model, "accuracy"])
        axes[1].errorbar(
            base,
            value,
            yerr=np.array([[lower], [upper]]),
            fmt="o",
            color=COLORS[model],
            capsize=3,
        )
        axes[1].annotate(
            TABLE_MODEL_LABELS[model],
            (base, value),
            xytext=annotation_offsets[model],
            textcoords="offset points",
            fontsize=8.5,
        )
    axes[0].axvline(0, color="#64748b", linewidth=1)
    axes[0].set_yticks(positions, [MODEL_LABELS[m] for m in MODEL_ORDER])
    axes[0].set_xlabel("ASP (percentage points)")
    axes[0].set_xlim(-10, 80)
    axes[0].invert_yaxis()
    axes[1].axhline(0, color="#64748b", linewidth=1)
    axes[1].set_xlim(0.15, 1.05)
    axes[1].set_ylim(-10, 80)
    axes[1].set_xlabel("Exact success rate without extra tools")
    axes[1].set_ylabel("ASP (percentage points)")
    axes[1].xaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    fig.tight_layout()
    fig.savefig(FIGURES / "figure4_asp.png")
    fig.savefig(FIGURES / "figure4_asp.svg")
    fig.savefig(FIGURES / "figure4_asp.pdf")
    plt.close(fig)

    # Schema-matched sealed-twin probe.
    eap_by_model = eap.set_index("model_alias")
    fig, axes = plt.subplots(1, 2, figsize=(8.6, 3.6))
    x = np.arange(len(MODEL_ORDER))
    width = 0.36
    executable = [
        float(eap_by_model.loc[model, "executable_correct"])
        / float(eap_by_model.loc[model, "graphs"])
        for model in MODEL_ORDER
    ]
    sealed = [
        float(eap_by_model.loc[model, "sealed_correct"])
        / float(eap_by_model.loc[model, "graphs"])
        for model in MODEL_ORDER
    ]
    axes[0].bar(
        x - width / 2,
        executable,
        width,
        label="Executable",
        color="#d9e8f5",
        edgecolor="#333333",
        linewidth=0.5,
    )
    axes[0].bar(
        x + width / 2,
        sealed,
        width,
        label="Sealed",
        color="#4c78a8",
        edgecolor="#333333",
        linewidth=0.5,
    )
    axes[0].set_xticks(
        x,
        [
            "DeepSeek-V4.1-\nFlash",
            "Kimi K3.1\nPro",
            "GLM-5.3-\nFlashx",
            "MiniMax-\nM3.1-Flash",
            "Qwen3.8-\nMax",
        ],
        rotation=25,
        ha="right",
        fontsize=8,
    )
    axes[0].set_ylim(0, 1.08)
    axes[0].set_ylabel("Exact success rate")
    axes[0].yaxis.set_major_formatter(matplotlib.ticker.PercentFormatter(1.0))
    axes[0].legend(frameon=False, loc="upper right")

    positions = np.arange(len(MODEL_ORDER))
    for index, model in enumerate(MODEL_ORDER):
        item = eap_by_model.loc[model]
        value = 100.0 * float(item["eap"])
        lower = value - 100.0 * float(item["eap_ci_low"])
        upper = 100.0 * float(item["eap_ci_high"]) - value
        axes[1].errorbar(
            value,
            index,
            xerr=np.array([[lower], [upper]]),
            fmt="o",
            color=COLORS[model],
            capsize=3,
        )
    axes[1].axvline(0, color="#64748b", linewidth=1)
    axes[1].set_yticks(positions, [MODEL_LABELS[m] for m in MODEL_ORDER])
    axes[1].invert_yaxis()
    axes[1].set_xlim(-35, 85)
    axes[1].set_xlabel("EAP (percentage points)")
    fig.tight_layout()
    fig.savefig(FIGURES / "figure5_sealed.png")
    fig.savefig(FIGURES / "figure5_sealed.svg")
    fig.savefig(FIGURES / "figure5_sealed.pdf")
    plt.close(fig)

    # Error distribution by model and condition.
    categories = [
        "correct",
        "wrong_argument_value",
        "premature_dependency",
        "irrelevant_function",
        "premature_final",
        "technical_or_budget",
    ]
    labels = [
        "Correct",
        "Wrong argument value",
        "Premature dependency",
        "Irrelevant function call",
        "Premature final answer",
        "Technical/budget failure",
    ]
    colors = ["#4c78a8", "#f58518", "#e45756", "#72b7b2", "#b279a2", "#9d755d"]
    baseline_correct = {
        model: int(
            (
                trials[
                    (trials["model_alias"] == model)
                    & (trials["distractor_count"] == 0)
                ]["primary_category"]
                == "correct"
            ).sum()
        )
        for model in MODEL_ORDER
    }
    compatible_correct = {
        model: int(
            (
                trials[
                    (trials["model_alias"] == model)
                    & (trials["distractor_count"] == 5)
                ]["primary_category"]
                == "correct"
            ).sum()
        )
        for model in MODEL_ORDER
    }
    ranked_models = sorted(
        MODEL_ORDER,
        key=lambda model: (
            -baseline_correct[model],
            -compatible_correct[model],
            MODEL_LABELS[model],
        ),
    )
    fig, axs = plt.subplots(
        1,
        2,
        figsize=(8.4, 4.8),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
        constrained_layout=True,
    )
    for ax, count, title, panel in zip(
        axs,
        (0, 5),
        ("Zero extra tools", "Five executable goal-irrelevant tools"),
        ("A", "B"),
        strict=True,
    ):
        frame = trials[trials["distractor_count"] == count]
        for yi, model in enumerate(ranked_models):
            model_rows = frame[frame["model_alias"] == model]
            left = 0
            for category, label, color in zip(
                categories,
                labels,
                colors,
                strict=True,
            ):
                value = int((model_rows["primary_category"] == category).sum())
                if value == 0:
                    continue
                ax.barh(
                    yi,
                    value,
                    left=left,
                    height=0.62,
                    color=color,
                    edgecolor="white",
                    linewidth=0.7,
                )
                ax.text(
                    left + value / 2,
                    yi,
                    str(value),
                    ha="center",
                    va="center",
                    color="white",
                    fontsize=9,
                )
                left += value
        ax.set_xlim(0, 20)
        ax.set_xticks([0, 5, 10, 15, 20])
        ax.set_ylim(len(ranked_models) - 0.5, -0.5)
        ax.set_xlabel("Trials per model")
        ax.set_title(title, loc="left", pad=14)
        ax.grid(axis="x", color="#dbe3ed", linewidth=0.8)
        ax.set_axisbelow(True)
        ax.tick_params(length=0, pad=7)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.text(
            -0.12,
            1.04,
            panel,
            transform=ax.transAxes,
            ha="left",
            va="bottom",
            fontsize=12,
            weight="bold",
            color="#172033",
        )
    axs[0].set_yticks(
        np.arange(len(ranked_models)),
        [
            f"#{rank}  {MODEL_LABELS[model]}"
            for rank, model in enumerate(ranked_models, start=1)
        ],
    )
    handles = [
        plt.Rectangle((0, 0), 1, 1, facecolor=color, edgecolor="white")
        for color in colors
    ]
    fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, -0.16),
        ncol=2,
        frameon=False,
        fontsize=9,
    )
    fig.savefig(FIGURES / "figure6_errors.png")
    fig.savefig(FIGURES / "figure6_errors.svg")
    fig.savefig(FIGURES / "figure6_errors.pdf")
    plt.close(fig)


def write_fact_sheet(
    main_metrics: pd.DataFrame,
    effects: pd.DataFrame,
    dose_conditions: pd.DataFrame,
    control_metrics: pd.DataFrame,
    trials: pd.DataFrame,
    gating: pd.DataFrame,
    reminder: pd.DataFrame,
    heterogeneity: dict[str, object],
    gating_transitions: pd.DataFrame,
    trace_summary: pd.DataFrame,
    asp: pd.DataFrame,
    eap: pd.DataFrame,
) -> None:
    facts: dict[str, object] = {
        "generated_at": "2026-11-01",
        "api_calls_made_by_this_script": 0,
        "study_level": {},
        "main20": {},
        "heterogeneity": heterogeneity,
        "dose15": {},
        "affordance15": {},
        "interventions": {},
        "error_taxonomy": {},
        "trace_sensitivity": trace_summary.to_dict(orient="records"),
        "gating_error_transitions": gating_transitions.to_dict(orient="records"),
        "asp": {},
        "eap": {},
    }
    facts["study_level"] = build_study_level_summary(trials)
    repeatability_path = SUPPLEMENT / "repeatability_summary.json"
    if repeatability_path.exists():
        facts["repeatability"] = json.loads(
            repeatability_path.read_text(encoding="utf-8")
        )
    for row in asp.itertuples(index=False):
        facts["asp"][row.model_alias] = {
            "graphs": int(row.graphs),
            "asp": float(row.asp),
            "asp_ci_low": float(row.asp_ci_low),
            "asp_ci_high": float(row.asp_ci_high),
            "sign_flip_p": float(row.sign_flip_p),
            "holm_p": float(row.holm_p),
            "asp_solvable": float(row.asp_solvable),
            "asp_solvable_ci_low": float(row.asp_solvable_ci_low),
            "asp_solvable_ci_high": float(row.asp_solvable_ci_high),
            "asp_solvable_holm_p": float(row.asp_solvable_holm_p),
        }
    for row in eap.itertuples(index=False):
        facts["eap"][row.model_alias] = {
            "graphs": int(row.graphs),
            "executable_correct": int(row.executable_correct),
            "sealed_correct": int(row.sealed_correct),
            "eap": float(row.eap),
            "eap_ci_low": float(row.eap_ci_low),
            "eap_ci_high": float(row.eap_ci_high),
            "sign_flip_p": float(row.sign_flip_p),
            "holm_p": float(row.holm_p),
            "positive_graphs": int(row.positive_graphs),
            "zero_graphs": int(row.zero_graphs),
            "negative_graphs": int(row.negative_graphs),
        }
    for model in MODEL_ORDER:
        base = main_metrics[
            (main_metrics["model_alias"] == model)
            & (main_metrics["distractor_count"] == 0)
        ].iloc[0]
        dist = main_metrics[
            (main_metrics["model_alias"] == model)
            & (main_metrics["distractor_count"] == 5)
        ].iloc[0]
        effect = effects[effects["model_alias"] == model].iloc[0]
        facts["main20"][model] = {
            "baseline_correct": int(base["correct"]),
            "baseline_trials": int(base["trials"]),
            "distractor_correct": int(dist["correct"]),
            "distractor_trials": int(dist["trials"]),
            "delta": float(effect["delta"]),
            "delta_ci_low": float(effect["delta_ci_low"]),
            "delta_ci_high": float(effect["delta_ci_high"]),
            "mcnemar_exact_p": float(effect["mcnemar_exact_p"]),
            "holm_p": float(effect["holm_p"]),
        }

    for model in MODEL_ORDER:
        facts["dose15"][model] = {
            str(count): {
                "correct": int(dose_conditions[
                    (dose_conditions["model_alias"] == model)
                    & (dose_conditions["distractor_count"] == count)
                ].iloc[0]["correct"]),
                "trials": int(dose_conditions[
                    (dose_conditions["model_alias"] == model)
                    & (dose_conditions["distractor_count"] == count)
                ].iloc[0]["trials"]),
            }
            for count in (0, 2, 5)
        }
        facts["affordance15"][model] = {
            mode: {
                "correct": int(control_metrics[
                    (control_metrics["model_alias"] == model)
                    & (control_metrics["distractor_mode"] == mode)
                ].iloc[0]["correct"]),
                "trials": int(control_metrics[
                    (control_metrics["model_alias"] == model)
                    & (control_metrics["distractor_mode"] == mode)
                ].iloc[0]["trials"]),
            }
            for mode in (
                "none",
                "disconnected",
                "connected_incompatible",
                "connected_compatible",
            )
        }
        gating_row = gating[gating["model_alias"] == model].iloc[0]
        reminder_row = reminder[reminder["model"] == model].iloc[0]
        facts["interventions"][model] = {
            "gating_reference_correct": round(
                float(gating_row["reference_accuracy"]) * int(gating_row["pairs"])
            ),
            "gating_treatment_correct": round(
                float(gating_row["treatment_accuracy"]) * int(gating_row["pairs"])
            ),
            "gating_pairs": int(gating_row["pairs"]),
            "gating_delta": float(gating_row["delta"]),
            "reminder_baseline_correct": int(reminder_row["baseline_correct"]),
            "reminder_treated_correct": int(reminder_row["reminder_correct"]),
            "reminder_pairs": 4,
            "reminder_wrong_to_right": int(reminder_row["wrong_to_right"]),
            "reminder_right_to_wrong": int(reminder_row["right_to_wrong"]),
        }

    facts["error_taxonomy"] = (
        trials.groupby(["distractor_count", "primary_category"])
        .size()
        .unstack(fill_value=0)
        .to_dict(orient="index")
    )
    (PAPER / "generated_facts.json").write_text(
        json.dumps(facts, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )

    # LaTeX macros keep critical numbers tied to the generated fact sheet.
    macros: list[str] = [
        "% Generated by paper/scripts/build_paper_assets.py. Do not edit by hand.",
        f"\\newcommand{{\\MainGraphClusters}}{{{int(heterogeneity['graph_clusters'])}}}",
        f"\\newcommand{{\\InteractionP}}{{{float(heterogeneity['omnibus_permutation_p']):.3f}}}",
        f"\\newcommand{{\\InteractionEffectRangePP}}{{{100.0 * float(heterogeneity['effect_range']):.0f}}}",
    ]
    study = facts["study_level"]
    study_main = study["main"]
    study_asp = study["asp"]
    macros.extend(
        [
            f"\\newcommand{{\\StudyModels}}{{{study_main['models']}}}",
            f"\\newcommand{{\\StudyMainPairs}}{{{study_main['model_graph_pairs']}}}",
            f"\\newcommand{{\\StudyMainBaseCorrect}}{{{study_main['baseline_correct']}}}",
            f"\\newcommand{{\\StudyMainDistCorrect}}{{{study_main['distractor_correct']}}}",
            f"\\newcommand{{\\StudyMainDeltaPP}}{{{pct_pp(study_main['estimate'])}}}",
            f"\\newcommand{{\\StudyMainCILowPP}}{{{pct_pp(study_main['ci_low'])}}}",
            f"\\newcommand{{\\StudyMainCIHighPP}}{{{pct_pp(study_main['ci_high'])}}}",
            f"\\newcommand{{\\StudyMainSignFlipP}}{{{study_main['sign_flip_p']:.4f}}}",
            f"\\newcommand{{\\StudyBaselineOnly}}{{{study_main['baseline_only']}}}",
            f"\\newcommand{{\\StudyDistractorOnly}}{{{study_main['distractor_only']}}}",
            f"\\newcommand{{\\StudyMainPositiveGraphs}}{{{study_main['positive_graphs']}}}",
            f"\\newcommand{{\\StudyMainZeroGraphs}}{{{study_main['zero_graphs']}}}",
            f"\\newcommand{{\\StudyMainNegativeGraphs}}{{{study_main['negative_graphs']}}}",
            f"\\newcommand{{\\StudyASPP}}{{{pct_pp(study_asp['estimate'])}}}",
            f"\\newcommand{{\\StudyASPCILowPP}}{{{pct_pp(study_asp['ci_low'])}}}",
            f"\\newcommand{{\\StudyASPCIHighPP}}{{{pct_pp(study_asp['ci_high'])}}}",
            f"\\newcommand{{\\StudyASPSignFlipP}}{{{study_asp['sign_flip_p']:.4f}}}",
            f"\\newcommand{{\\StudyASPPositiveGraphs}}{{{study_asp['positive_graphs']}}}",
            f"\\newcommand{{\\StudyASPZeroGraphs}}{{{study_asp['zero_graphs']}}}",
            f"\\newcommand{{\\StudyASPNegativeGraphs}}{{{study_asp['negative_graphs']}}}",
        ]
    )
    asp_values = [fact["asp"] for fact in facts["asp"].values()]
    mini = facts["asp"]["minimax-m3.1-flash"]
    macros.extend(
        [
            f"\\newcommand{{\\ASPRangeLowPP}}{{{pct_pp(min(asp_values))}}}",
            f"\\newcommand{{\\ASPRangeHighPP}}{{{pct_pp(max(asp_values))}}}",
            f"\\newcommand{{\\MiniMaxASPP}}{{{pct_pp(mini['asp'])}}}",
            f"\\newcommand{{\\MiniMaxASPLowPP}}{{{pct_pp(mini['asp_ci_low'])}}}",
            f"\\newcommand{{\\MiniMaxASPHighPP}}{{{pct_pp(mini['asp_ci_high'])}}}",
            f"\\newcommand{{\\MiniMaxASPHolmP}}{{{mini['holm_p']:.4f}}}",
        ]
    )
    macro_names = {
        "deepseek-flash": "DeepSeek",
        "kimi-k3.1-pro": "Kimi",
        "glm-5.3-flashx": "GLM",
        "minimax-m3.1-flash": "MiniMax",
        "qwen3.8-max": "Qwen",
    }
    for model, name in macro_names.items():
        item = facts["main20"][model]
        macros.extend(
            [
                f"\\newcommand{{\\{name}BaseCorrect}}{{{item['baseline_correct']}}}",
                f"\\newcommand{{\\{name}BaseTrials}}{{{item['baseline_trials']}}}",
                f"\\newcommand{{\\{name}DistCorrect}}{{{item['distractor_correct']}}}",
                f"\\newcommand{{\\{name}DistTrials}}{{{item['distractor_trials']}}}",
                f"\\newcommand{{\\{name}DeltaPP}}{{{pct_pp(item['delta'])}}}",
                f"\\newcommand{{\\{name}DeltaLowPP}}{{{pct_pp(item['delta_ci_low'])}}}",
                f"\\newcommand{{\\{name}DeltaHighPP}}{{{pct_pp(item['delta_ci_high'])}}}",
                f"\\newcommand{{\\{name}McNemarP}}{{{item['mcnemar_exact_p']:.4f}}}",
                f"\\newcommand{{\\{name}HolmP}}{{{item['holm_p']:.4f}}}",
            ]
        )
    (MANUSCRIPT / "generated_numbers.tex").write_text(
        "\n".join(macros) + "\n", encoding="utf-8"
    )

    lines = [
        "# Generated analysis summary",
        "",
        "This file is generated from the canonical formal result files. No model API calls are made.",
        "",
        f"- Main graphs: {heterogeneity['graph_clusters']}",
        f"- Model-interaction permutation p: {float(heterogeneity['omnibus_permutation_p']):.3f}",
        f"- Main technical/budget failures: {int((trials['outcome'] == 'technical_failure').sum())}",
        "",
        "## Main experiment",
        "",
        md_table(
            ["Model", "0", "5", "Delta", "95% CI", "McNemar p", "Holm p"],
            [
                [
                    MODEL_LABELS[model],
                    f"{facts['main20'][model]['baseline_correct']}/{facts['main20'][model]['baseline_trials']}",
                    f"{facts['main20'][model]['distractor_correct']}/{facts['main20'][model]['distractor_trials']}",
                    f"{pct_pp(facts['main20'][model]['delta'])} pp",
                    f"[{pct_pp(facts['main20'][model]['delta_ci_low'])}, {pct_pp(facts['main20'][model]['delta_ci_high'])}]",
                    f"{facts['main20'][model]['mcnemar_exact_p']:.4f}",
                    f"{facts['main20'][model]['holm_p']:.4f}",
                ]
                for model in MODEL_ORDER
            ],
        ),
        "## Trace sensitivity",
        "",
        md_table(
            list(trace_summary.columns),
            trace_summary.itertuples(index=False, name=None),
        ),
        "## Repeatability audit",
        "",
    ]
    if "repeatability" in facts:
        repeatability = facts["repeatability"]
        lines.extend(
            [
                f"- Correctness agreement: {repeatability['same_correctness']}",
                f"- Correctness kappa: {float(repeatability['correctness_kappa']):.3f}",
                f"- Primary-category agreement: {repeatability['same_primary_category']}",
                f"- Final-answer signature agreement: {repeatability['same_final_answer_signature']}",
                f"- Model interactions: {repeatability['requests']}",
                f"- Technical/budget failures: {repeatability['technical_failures']}",
            ]
        )
    else:
        lines.append("Repeatability summary is not available in this build.")
    lines.extend(
        [
            "",
            "## Tool-usability controls",
            "",
            md_table(
                [
                    "Model",
                    "0",
                    "Disconnected 5",
                    "Connected-incompatible 5",
                    "Executable goal-irrelevant 5",
                ],
                [
                    [
                        TABLE_MODEL_LABELS[model],
                        *[
                            (
                                f"{facts['affordance15'][model][mode]['correct']}/"
                                f"{facts['affordance15'][model][mode]['trials']}"
                            )
                            for mode in (
                                "none",
                                "disconnected",
                                "connected_incompatible",
                                "connected_compatible",
                            )
                        ],
                    ]
                    for model in MODEL_ORDER
                ],
            ),
            (
                "All four conditions use the same 15 required graphs. "
                "Executable goal-irrelevant tools are lowest or tied for "
                "lowest for every model."
            ),
        ]
    )
    lines.extend(
        [
            "",
            "## ASP paired loss",
            "",
            md_table(
                ["Model", "ASP", "95% CI", "Sign-flip p", "Holm p", "+/0/-"],
                [
                    [
                        TABLE_MODEL_LABELS[model],
                        f"{pct_pp(facts['asp'][model]['asp'])} pp",
                        (
                            f"[{pct_pp(facts['asp'][model]['asp_ci_low'])}, "
                            f"{pct_pp(facts['asp'][model]['asp_ci_high'])}]"
                        ),
                        f"{facts['asp'][model]['sign_flip_p']:.4f}",
                        f"{facts['asp'][model]['holm_p']:.4f}",
                        (
                            f"{int(asp.set_index('model_alias').loc[model, 'positive_graphs'])}/"
                            f"{int(asp.set_index('model_alias').loc[model, 'zero_graphs'])}/"
                            f"{int(asp.set_index('model_alias').loc[model, 'negative_graphs'])}"
                        ),
                    ]
                    for model in MODEL_ORDER
                ],
            ),
            (
                "ASP is the mean of the disconnected-minus-goal-irrelevant and "
                "connected-incompatible-minus-goal-irrelevant contrasts on each graph."
            ),
            "",
            "## Schema-matched sealed probe",
            "",
            md_table(
                ["Model", "Executable", "Sealed", "EAP", "95% CI", "Holm p", "+/0/-"],
                [
                    [
                        TABLE_MODEL_LABELS[model],
                        (
                            f"{facts['eap'][model]['executable_correct']}/"
                            f"{facts['eap'][model]['graphs']}"
                        ),
                        (
                            f"{facts['eap'][model]['sealed_correct']}/"
                            f"{facts['eap'][model]['graphs']}"
                        ),
                        f"{pct_pp(facts['eap'][model]['eap'])} pp",
                        (
                            f"[{pct_pp(facts['eap'][model]['eap_ci_low'])}, "
                            f"{pct_pp(facts['eap'][model]['eap_ci_high'])}]"
                        ),
                        f"{facts['eap'][model]['holm_p']:.4f}",
                        (
                            f"{facts['eap'][model]['positive_graphs']}/"
                            f"{facts['eap'][model]['zero_graphs']}/"
                            f"{facts['eap'][model]['negative_graphs']}"
                        ),
                    ]
                    for model in MODEL_ORDER
                ],
            ),
        ]
    )
    lines.extend(
        [
        "",
        "## Reproducibility note",
        "",
        (
            "`build_paper_assets.py` is deterministic and reads only local result "
            "summaries. The source tables remain in `results/`; this script does "
            "not copy or mutate them."
        ),
        ]
    )
    (SUPPLEMENT / "derived_analysis.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )


def write_manifest() -> None:
    manifest = {
        "generated_at": "2026-11-01",
        "source_root": "results",
        "inputs": [
            str(MAIN.relative_to(ROOT)),
            str(DOSE.relative_to(ROOT)),
            str(CONTROL.relative_to(ROOT)),
            str(GATING.relative_to(ROOT)),
            str(REMINDER.relative_to(ROOT)),
            str(SEALED_PROBE.relative_to(ROOT)),
        ],
        "outputs": {
            "tables": sorted(path.name for path in TABLES.glob("*.tex")),
            "figures": sorted(path.name for path in FIGURES.glob("*") if path.is_file()),
            "supplement": sorted(path.name for path in SUPPLEMENT.glob("*") if path.is_file()),
        },
        "api_calls": 0,
        "notes": [
            "All numerical table values come from existing result summaries.",
            "Holm-adjusted p-values and paired reminder transitions are newly derived from existing trial rows.",
            "No raw provider response or result directory is copied into paper/.",
        ],
    }
    (PAPER / "asset_manifest.json").write_text(
        json.dumps(manifest, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )


def main() -> None:
    ensure_dirs()
    build_method_overview()
    main_metrics, main_effects = build_main_table()
    build_trace_table(main_metrics)
    dose_conditions = build_dose_table()
    control_metrics = build_control_table()
    asp = build_asp_tables()
    eap = build_eap_table()
    build_experiment_table()
    build_case_table()
    build_repeatability_tables()
    trials, trace_summary = build_error_tables()
    gating, reminder = build_intervention_table(main_metrics)
    gating_transitions = build_gating_error_transitions()
    heterogeneity = json.loads(
        (MAIN / "heterogeneity" / "heterogeneity_summary.json").read_text(
            encoding="utf-8"
        )
    )
    build_figures(
        main_metrics,
        dose_conditions,
        control_metrics,
        trials,
        asp,
        eap,
    )
    write_fact_sheet(
        main_metrics,
        main_effects,
        dose_conditions,
        control_metrics,
        trials,
        gating,
        reminder,
        heterogeneity,
        gating_transitions,
        trace_summary,
        asp,
        eap,
    )
    (TABLES / "README.md").write_text(
        "# Paper tables\n\n"
        "Generated by `paper/scripts/build_paper_assets.py` from formal result summaries.\n"
        "Each table is emitted as CSV, Markdown, and LaTeX. Do not edit numbers by hand.\n",
        encoding="utf-8",
    )
    (FIGURES / "README.md").write_text(
        "# Paper figures\n\n"
        "Generated from the canonical result summaries by "
        "`paper/scripts/build_paper_assets.py`. PDF/SVG versions are suitable for "
        "LaTeX; PNG versions are previews. The conceptual method overview is "
        "generated by `paper/scripts/build_method_overview_v2.py` and invoked "
        "from the asset builder.\n",
        encoding="utf-8",
    )
    write_manifest()


if __name__ == "__main__":
    main()
