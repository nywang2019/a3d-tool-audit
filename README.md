# A3D Tool Audit

This repository accompanies:

> Nianyi Wang. *The Wrong Tool Looks Right: Auditing Executable
> Goal-Irrelevant Tools in Multi-Step Tool Use.*

Release version: `1.0.0` (manuscript `v0.13.7`).

A3D is a trace-level audit for multi-step LLM tool use. It holds the required
dependency task fixed, varies the operational availability of an equal number of
extra tools, and derives a first decisive error from saved tool calls.

## What is included

- Python implementation of paired FuncBenchGen execution, graph invariants,
  trace diagnostics, statistical summaries, and figure generation.
- Experimental configurations for the main, dose, matched-control,
  prerequisite-hiding, reminder, repeatability, and sealed-probe studies.
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
- `gating8`: prerequisite-hiding intervention.
- `reminder4`: state-reminder intervention.
- `repeatability8`: fresh repeat run on eight graph indices.
- `sealed10`: executable-versus-sealed schema-matched probe.

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

OpenAI Codex assisted with manuscript restructuring and adversarial questioning,
consistency checks between text and research artifacts, literature-positioning
checks, LaTeX/build verification, and preparation of this reproducibility
repository. It did not generate the empirical results, execute the reported
model trials, or assume responsibility for the work. Nianyi Wang directed the
work, verified the AI-assisted outputs, and accepts full responsibility for the
manuscript and released materials.

