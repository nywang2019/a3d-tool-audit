# Reproducibility and validation record

## Scope

The pre-submission upgrade included new disconnected/incompatible controls, a
repeatability audit, and a formal MiniMax-M3.1-Flash 20-graph comparison. Paper
assets are then generated locally from the resulting saved analyses.

## Canonical sources

- Main experiment:
  `results/funcbench-main20-five-model-analysis-20261101/`
- MiniMax formal run:
  `results/funcbench-minimax-main20-20261101T105025Z/`
- Dose analysis:
  `results/funcbench-dose-five-model-20261101/`
- Tool-usability controls:
  `results/funcbench-control-five-model-analysis-20261101/`
- Prerequisite hiding:
  `results/funcbench-gating-five-model-analysis-20261101/`
- Reminder:
  `results/funcbench-reminder-five-model-analysis-20261101/`
- Repeatability:
  `results/funcbench-repeatability8-five-model-20261101/`
- Sealed twin:
  `results/funcbench-sealed-probe-analysis-20261101/`
- Pair audit:
  `results/funcbench-paired-main-v2-20261030T122222Z/pair_audit.json`

## Frozen generator

- Repository: `vendor/FuncBenchGen`
- Commit: `0729e2567dfaee1e53262ba9bfd5ba6a4b4e4343`
- License: BSD-3-Clause

The local implementation sorts set-derived structure before graph construction
and tests identical fingerprints under different `PYTHONHASHSEED` values. The
invalid mixed-process run is documented in `docs/invalid_runs.md` and is not used
for reported effects.

## API configuration

- Temperature: 1.0
- Top-p: 0.95
- Maximum output tokens: 8192
- Reasoning effort: low where supported
- Timeout: 300 seconds
- Retries: at most two
- Proxy policy: `trust_env=false`
- Run dates: 2026-10-29, 2026-10-30, and 2026-11-01
- MiniMax thinking: cannot be disabled; lowest effort (`low`) used
- MiniMax compatibility: returned `reasoning_content` preserved across tool turns

## Table and figure generation

Run from the project root:

```powershell
.\.venv\Scripts\python.exe paper\scripts\analyze_asp.py
.\.venv\Scripts\python.exe paper\scripts\analyze_sealed_probe.py
.\.venv\Scripts\python.exe paper\scripts\build_paper_assets.py
```

The analysis scripts compute graph-level ASP/EAP scores, bootstrap intervals,
exact sign-flip tests, and Holm corrections. The asset builder then:

- makes no network or model API request;
- reads only formal derived result CSV/JSON and paired trial summaries;
- emits CSV, Markdown, and LaTeX versions of each table;
- emits PNG, SVG, and PDF versions of each figure;
- writes `paper/generated_facts.json`;
- writes the LaTeX numeric macros used in the manuscript.

## Statistical checks

- Exact success is the primary outcome.
- Paired graph bootstrap intervals are read from the canonical effect files.
- Exact McNemar p-values are read from the canonical effect files.
- Holm correction across the five main-model McNemar tests is derived in the
  paper-generation script.
- Cross-model interaction is taken from the canonical graph-cluster permutation
  summary.
- No sigmoid or continuous breakpoint is fit.

## Additional analyses generated here

- End-to-end success after excluding technical/budget failures.
- Counts of complete structural traces and incorrect outcomes among complete
  traces.
- Reminder success transitions reconstructed by graph fingerprint.
- Descriptive prerequisite-hiding primary-category transitions.

These are derived only from existing trial records and are labeled as
supplementary/exploratory where appropriate.

## Prior validation evidence

The current project suite records 48 passing pytest tests and successful graph
fingerprint, target-value, tool-order, and completeness audits for the expanded
formal runs. This preparation task did not rerun the full test suite until the
verification step described in the final execution report.

Systematic human review of error categories is explicitly outside the first
submission. Automated labels are therefore reported as heuristic diagnostics,
not human-validated ground truth.

