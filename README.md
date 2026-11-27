# A3D Tool Audit

This repository accompanies:

> Nianyi Wang. *The Wrong Tool Looks Right: Auditing Executable
> Goal-Irrelevant Tool Use in Multi-Step LLM Agents.*

Release version: `1.0.0` (manuscript `v0.13.30`).

A3D is a trace-level audit for multi-step LLM tool use. It holds the required
dependency task fixed, varies the operational availability of an equal number of
extra tools, and derives a first decisive error from saved tool calls.

## What is included

- Python implementation of paired FuncBenchGen execution, graph invariants,
  trace diagnostics, statistical summaries, and figure generation.
- Experimental configurations for the main, dose, matched-control, and
  sealed-probe studies reported in the current manuscript.
- Sanitized trial-level CSVs and derived analysis summaries for the five-model
  panel.
- Final manuscript PDF, LaTeX source, figures, tables, and supplementary
  documentation.
- A three-clause BSD notice for the vendored FuncBenchGen dependency.

The release does not contain API keys, provider credentials, authentication
headers, personal request metadata, or unprocessed provider response dumps.

## Data

`data/trials/` contains convenience copies of the sanitized trial records, and
`results/` preserves the directory names expected by the analysis scripts. The
aggregate study directories are:

- `main20`: 20-graph main comparison.
- `control15`: equal-count disconnected, connected-incompatible, and executable
  goal-irrelevant controls.
- `dose15`: 0/2/5 executable goal-irrelevant dose study.
- `sealed10`: executable-versus-sealed schema-matched probe.

Historical extension configs and result directories remain in the repository
for provenance, but prerequisite-hiding, reminder, and repeatability studies
are not part of the current manuscript's empirical scope.

Sanitized data and derived results in `data/` and `results/` are licensed under
CC BY 4.0. Code is licensed under the MIT License. The vendored dependencies
retain their original licenses; see `THIRD_PARTY_NOTICES.md`.

## Installation

Python 3.12 or newer is required.

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -e ".[dev,funcbench]"
```

Configure provider API keys through environment variables named in the
configuration files. Never commit a populated `.env` file.

## Reproduction overview

```powershell
.\.venv\Scripts\python.exe -m pytest
.\.venv\Scripts\python.exe paper\scripts\validate_paper_assets.py
```

Because the analysis scripts use workspace-relative paths, run commands from
the repository root. Regenerating model outputs requires valid provider
credentials and may incur API charges. Reproduction of the published tables,
figures, and asset checks from the released records does not require new model
calls.

## Citation

Use the metadata in `CITATION.cff`. No archival DOI has been assigned.

## Maintenance

Nianyi Wang is responsible for maintaining this repository and the versioned
release. Questions: `livingsailor@gmail.com`.

## AI-use disclosure

OpenAI Codex assisted with adversarial questioning, consistency checks between
text and research artifacts, literature-positioning checks, and preparation of
this reproducibility repository. It did not generate the empirical results,
execute the reported model trials, substitute for independent peer review, or
assume responsibility for the work.

