#!/usr/bin/env python3
"""Validate generated paper assets against canonical result summaries."""

from __future__ import annotations

import json
import re
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"


def main() -> None:
    facts = json.loads((PAPER / "generated_facts.json").read_text(encoding="utf-8"))
    coverage = json.loads(
        (PAPER / "model_coverage_audit.json").read_text(encoding="utf-8")
    )
    assert coverage["passed"], coverage["failed_checks"]

    main = pd.read_csv(
        ROOT
        / "results/funcbench-main20-five-model-analysis-20261101/metric_summary.csv"
    )
    effects = pd.read_csv(
        ROOT
        / "results/funcbench-main20-five-model-analysis-20261101/paired_effect_summary.csv"
    )
    assert len(effects) == 5
    assert set(effects["pairs"]) == {20}
    assert set(main.groupby(["model_alias", "distractor_count"])["trials"].sum()) == {20}

    for row in effects.itertuples(index=False):
        item = facts["main20"][row.model_alias]
        assert item["delta_ci_low"] == row.delta_ci_low
        assert item["delta_ci_high"] == row.delta_ci_high
        assert item["mcnemar_exact_p"] == row.mcnemar_exact_p
        assert item["baseline_correct"] + item["distractor_correct"] == (
            row.both_correct + row.baseline_only + row.distractor_only + row.both_correct
        )

    trials = pd.read_csv(
        ROOT
        / "results/funcbench-main20-five-model-analysis-20261101/trial_failures.csv"
    )
    assert len(trials) == 200
    assert trials.groupby(["model_alias", "distractor_count"]).size().eq(20).all()
    assert int((trials["outcome"] == "technical_failure").sum()) == 1
    assert int((trials["primary_category"] == "correct").sum()) == 118

    control = pd.read_csv(
        ROOT
        / "results/funcbench-control-five-model-analysis-20261101/metric_summary.csv"
    )
    assert set(control["trials"]) == {15}
    assert control.groupby("model_alias")["distractor_mode"].nunique().eq(4).all()
    assert control["model_alias"].nunique() == 5
    control_trials = pd.read_csv(
        ROOT
        / "results/funcbench-control-five-model-analysis-20261101/trial_failures.csv"
    )
    assert len(control_trials) == 300
    assert set(control_trials["graph_index"]) == set(range(15))
    assert control_trials.groupby(
        ["model_alias", "distractor_mode"]
    ).size().eq(15).all()
    assert control_trials.groupby("graph_index")[
        "required_graph_fingerprint"
    ].nunique().eq(1).all()
    assert control_trials.groupby("graph_index")["target_value"].nunique().eq(1).all()

    asp = pd.read_csv(
        ROOT
        / "results/funcbench-control-five-model-analysis-20261101/asp/asp_summary.csv"
    )
    assert len(asp) == 5
    assert set(asp["model_alias"]) == set(main["model_alias"])
    assert asp["asp"].gt(0).all()
    assert asp["holm_p"].between(0, 1).all()
    control_trials["correct_bool"] = control_trials["correct"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    for asp_row in asp.itertuples(index=False):
        model_rows = control_trials[
            control_trials["model_alias"] == asp_row.model_alias
        ]
        pivot = model_rows.pivot(
            index="graph_seed",
            columns="distractor_mode",
            values="correct_bool",
        )
        scores = (
            pivot["disconnected"].astype(int)
            + pivot["connected_incompatible"].astype(int)
            - 2 * pivot["connected_compatible"].astype(int)
        ) / 2
        assert abs(float(scores.mean()) - float(asp_row.asp)) < 1e-12
        assert int((scores > 0).sum()) == int(asp_row.positive_graphs)

    sealed_dir = ROOT / "results/funcbench-sealed-probe-analysis-20261101"
    eap = pd.read_csv(sealed_dir / "eap/eap_summary.csv")
    eap_payload = json.loads(
        (sealed_dir / "eap/eap_summary.json").read_text(encoding="utf-8")
    )
    assert len(eap) == 5
    assert eap_payload["schema_match_graphs"] == 10
    assert set(eap["model_alias"]) == set(main["model_alias"])
    assert set(eap["graphs"]) == {10}
    sealed_trials = pd.read_csv(sealed_dir / "trial_failures.csv")
    sealed_trials["correct_bool"] = sealed_trials["correct"].astype(str).str.lower().isin(
        {"true", "1", "yes"}
    )
    sealed_trials = sealed_trials[sealed_trials["distractor_count"] == 5]
    for eap_row in eap.itertuples(index=False):
        model_rows = sealed_trials[
            sealed_trials["model_alias"] == eap_row.model_alias
        ]
        pivot = model_rows.pivot(
            index="graph_seed",
            columns="distractor_mode",
            values="correct_bool",
        )
        scores = (
            pivot["sealed_compatible"].astype(int)
            - pivot["connected_compatible"].astype(int)
        )
        assert abs(float(scores.mean()) - float(eap_row.eap)) < 1e-12

    dose = pd.read_csv(
        ROOT
        / "results/funcbench-dose-five-model-20261101/dose_condition_summary.csv"
    )
    assert set(dose["trials"]) == {15}
    assert dose.groupby("model_alias")["distractor_count"].nunique().eq(3).all()
    assert dose["model_alias"].nunique() == 5

    gating = pd.read_csv(
        ROOT
        / "results/funcbench-gating-five-model-analysis-20261101/"
        "intervention/intervention_effect_summary.csv"
    )
    assert len(gating) == 5
    assert set(gating["pairs"]) == {8}

    reminder = pd.read_csv(
        ROOT
        / "results/funcbench-reminder-five-model-analysis-20261101/"
        "trial_failures.csv"
    )
    assert reminder["model_alias"].nunique() == 5
    assert len(reminder) == 40

    repeatability = pd.read_csv(
        PAPER / "tables/repeatability_conditions_full.csv"
    )
    assert len(repeatability) == 5
    assert set(repeatability["Model"]) == {
        "DeepSeek",
        "Kimi",
        "GLM",
        "MiniMax",
        "Qwen",
    }

    generated_numbers = (
        PAPER / "manuscript/generated_numbers.tex"
    ).read_text(encoding="utf-8")
    assert "$-$" not in generated_numbers
    generated_macros = set(
        re.findall(r"\\newcommand\{\\([^}]+)\}", generated_numbers)
    )
    assert {
        "MainGraphClusters",
        "InteractionP",
        "StudyMainDeltaPP",
        "StudyMainCILowPP",
        "StudyMainCIHighPP",
        "StudyMainSignFlipP",
        "StudyASPP",
        "StudyASPCILowPP",
        "StudyASPCIHighPP",
        "StudyASPSignFlipP",
    } <= generated_macros

    bib = (PAPER / "manuscript/references.bib").read_text(encoding="utf-8")
    bib_keys = set(re.findall(r"@\w+\{([^,]+),", bib))
    for manuscript_name in ("main.tex", "main_applied_intelligence.tex"):
        tex = (PAPER / "manuscript" / manuscript_name).read_text(encoding="utf-8")
        inputs = re.findall(r"\\input\{([^}]+)\}", tex)
        assert inputs
        for relative in inputs:
            assert (PAPER / "manuscript" / relative).resolve().exists(), relative

        figure_paths = re.findall(
            r"\\includegraphics(?:\[[^]]*\])?\{([^}]+)\}", tex
        )
        assert figure_paths
        for relative in figure_paths:
            candidate = (PAPER / "manuscript" / relative).resolve()
            figure_candidate = (PAPER / "figures" / relative).resolve()
            assert candidate.exists() or figure_candidate.exists(), relative

        cited_keys: set[str] = set()
        for citation_group in re.findall(r"\\citep\{([^}]*)\}", tex):
            cited_keys.update(
                key.strip() for key in citation_group.split(",") if key.strip()
            )
        assert cited_keys <= bib_keys, (
            f"Missing bibliography keys in {manuscript_name}: "
            f"{cited_keys - bib_keys}"
        )

        tex_sources = [tex]
        for relative in inputs:
            tex_sources.append(
                (PAPER / "manuscript" / relative).read_text(encoding="utf-8")
            )
        all_tex = "\n".join(tex_sources)
        environment_stack: list[str] = []
        for event, environment in re.findall(
            r"\\(begin|end)\{([^}]+)\}", all_tex
        ):
            if event == "begin":
                environment_stack.append(environment)
            else:
                assert environment_stack, f"Unexpected end of {environment}"
                assert environment_stack.pop() == environment, environment
        assert not environment_stack, f"Unclosed environments: {environment_stack}"

    applied_tex = (
        PAPER / "manuscript/main_applied_intelligence.tex"
    ).read_text(encoding="utf-8")
    assert "MiniMax-M3.1-Flash" in applied_tex
    assert "four-model panel" not in applied_tex
    assert "160 main-experiment" not in applied_tex
    assert "160 main trials" not in applied_tex
    abstract_match = re.search(
        r"\\abstract\{(.*?)\}\n\n\\keywords", applied_tex, re.DOTALL
    )
    assert abstract_match
    abstract_words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", abstract_match.group(1))
    assert 150 <= len(abstract_words) <= 250, len(abstract_words)
    assert (PAPER / "manuscript/sn-jnl.cls").exists()
    assert (PAPER / "manuscript/sn-basic.bst").exists()

    print(
        "VALID: five-model main, dose, control, gating, reminder, repeatability, "
        "ASP, and sealed-twin analyses; 200 main trials; generated numbers, "
        "table/figure inputs, citations, "
        "LaTeX environment balance, model coverage, and Applied Intelligence "
        "abstract length."
    )


if __name__ == "__main__":
    main()

