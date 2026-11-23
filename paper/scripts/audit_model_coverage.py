#!/usr/bin/env python3
"""Audit which model panel backs each paper result, figure, and table."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"

ORIGINAL_FOUR = {
    "deepseek-flash",
    "kimi-k3.1-pro",
    "glm-5.3-flashx",
    "qwen3.8-max",
}
FIVE = ORIGINAL_FOUR | {"minimax-m3.1-flash"}
ORIGINAL_LABELS = {
    "DeepSeek-V4.1-Flash",
    "Kimi",
    "GLM",
    "Qwen",
}
FIVE_LABELS = ORIGINAL_LABELS | {"MiniMax"}


def models(path: Path, column: str) -> set[str]:
    return set(pd.read_csv(path)[column].astype(str))


def main() -> None:
    main_dir = ROOT / "results/funcbench-main20-five-model-analysis-20261101"
    checks = {
        "main_metrics_five_models": models(
            main_dir / "metric_summary.csv", "model_alias"
        )
        == FIVE,
        "main_effects_five_models": models(
            main_dir / "paired_effect_summary.csv", "model_alias"
        )
        == FIVE,
        "main_trials_200": len(pd.read_csv(main_dir / "trial_failures.csv")) == 200,
        "main_table_five_models": models(
            PAPER / "tables/main20_effect.csv", "Model"
        )
        == {"DeepSeek-V4.1-Flash", "Kimi", "GLM", "MiniMax", "Qwen"},
        "trace_table_five_models": models(
            PAPER / "tables/trace_metrics.csv", "Model"
        )
        == {"DeepSeek-V4.1-Flash", "Kimi", "GLM", "MiniMax", "Qwen"},
        "valid_trace_table_five_models": models(
            PAPER / "tables/valid_trace_summary.csv", "Model"
        )
        == {"DeepSeek-V4.1-Flash", "Kimi", "GLM", "MiniMax", "Qwen"},
        "diagnostic_cases_five_models": models(
            PAPER / "tables/diagnostic_cases.csv", "Model"
        )
        == FIVE_LABELS,
        "dose_five_models": models(
            PAPER / "tables/dose_response.csv", "Model"
        )
        == FIVE_LABELS,
        "affordance_five_models": models(
            PAPER / "tables/affordance_controls.csv", "Model"
        )
        == FIVE_LABELS,
        "interventions_five_models": models(
            PAPER / "tables/interventions.csv", "Model"
        )
        == FIVE_LABELS,
        "repeatability_five_models": models(
            PAPER / "tables/repeatability_conditions.csv", "Model"
        )
        == FIVE_LABELS,
        "asp_five_models": models(PAPER / "tables/asp.csv", "Model")
        == FIVE_LABELS,
        "asp_solvable_five_models": models(
            PAPER / "tables/asp_solvable.csv", "Model"
        )
        == FIVE_LABELS,
        "eap_five_models": models(PAPER / "tables/eap.csv", "Model")
        == FIVE_LABELS,
    }

    design = pd.read_csv(PAPER / "tables/experiment_design.csv")
    checks["design_table_has_models_column"] = "Models" in design.columns
    if "Models" in design.columns:
        main_models = int(
            design.loc[design["Study"] == "Main", "Models"].iloc[0]
        )
        auxiliary_models = set(
            design.loc[design["Study"] != "Main", "Models"].astype(int)
        )
        checks["design_table_main_five"] = main_models == 5
        checks["design_table_auxiliary_five"] = auxiliary_models == {5}

    failed = sorted(key for key, passed in checks.items() if not passed)
    payload = {
        "passed": not failed,
        "failed_checks": failed,
        "coverage": {
            "main_paired_effect": sorted(FIVE),
            "trace_and_error_analysis": sorted(FIVE),
            "dose_response": sorted(FIVE),
            "affordance_controls": sorted(FIVE),
            "gating_and_reminder": sorted(FIVE),
            "repeatability": sorted(FIVE),
            "affordance_specific_penalty": sorted(FIVE),
            "sealed_twin_probe": sorted(FIVE),
        },
        "checks": checks,
    }
    (PAPER / "model_coverage_audit.json").write_text(
        json.dumps(payload, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Model coverage audit",
        "",
        f"Status: **{'PASS' if not failed else 'FAIL'}**",
        "",
        "| Analysis | Model panel |",
        "|---|---|",
        "| Main paired effect, trace metrics, error taxonomy | Five models, including MiniMax |",
        "| Dose response | Five models, including MiniMax |",
        "| Tool-usability controls | Five models, including MiniMax |",
        "| Prerequisite hiding and reminder interventions | Five models, including MiniMax |",
        "| Repeatability | Five models, including MiniMax |",
        "| ASP paired loss | Five models, including MiniMax |",
        "| Schema-matched sealed probe | Five models, including MiniMax |",
        "",
    ]
    if failed:
        lines.extend(["Failed checks:", "", *[f"- `{item}`" for item in failed], ""])
    else:
        lines.append("All expected model-coverage checks passed.")
    (PAPER / "model_coverage_audit.md").write_text(
        "\n".join(lines) + "\n",
        encoding="utf-8",
    )
    if failed:
        raise SystemExit(1)
    print(
        "VALID: every main and auxiliary analysis contains all five models."
    )


if __name__ == "__main__":
    main()
