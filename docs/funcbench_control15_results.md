# FuncBenchGen 15-graph tool-usability controls

Date: 2026-11-01

## Scope

This analysis expands the tool-usability controls from graphs 0-7 to
graphs 0-14 without regenerating the original cells.

The four conditions are:

- no extra tools;
- five disconnected tools;
- five connected but type-incompatible tools;
- five executable goal-irrelevant tools.

The executable goal-irrelevant and no-extra-tool cells for graphs 8-14 come from the
formal 20-graph main analysis. The new disconnected and incompatible calls were
run on the same required graphs with the locked generator and stable tool order.

## New runs

| Run | Trials | Model requests | Correct | Technical failures |
|---|---:|---:|---:|---:|
| `funcbench-control-disconnected-extension15-20261101T085735Z` | 28 | 139 | 19/28 | 0 |
| `funcbench-control-incompatible-extension15-20261101T090905Z` | 28 | 141 | 13/28 | 1 |
| `funcbench-minimax-control-disconnected-20261101T113958Z` | 15 | 89 | 13/15 | 0 |
| `funcbench-minimax-control-incompatible-20261101T114402Z` | 15 | 76 | 11/15 | 0 |

The extension audits correctly report missing same-run baseline cells because
these are extension-only runs. Their graph fingerprints and target values were
checked during the explicit merge with the existing main and control analyses.
No duplicate model/graph/condition cell was accepted.

## Combined results

Each cell is correct/trials on the same 15 graph indices.

| Model | 0 | Disconnected 5 | Connected-incompatible 5 | Executable goal-irrelevant 5 |
|---|---:|---:|---:|---:|
| DeepSeek-V4.1-Flash | 15/15 | 14/15 | 12/15 | 10/15 |
| Kimi K3.1 Pro | 11/15 | 10/15 | 7/15 | 5/15 |
| GLM-5.3-Flashx | 4/15 | 8/15 | 4/15 | 3/15 |
| MiniMax-M3.1-Flash | 12/15 | 13/15 | 11/15 | 5/15 |
| Qwen3.8-Max | 12/15 | 14/15 | 8/15 | 8/15 |

The executable goal-irrelevant condition is lowest for DeepSeek-V4.1-Flash, Kimi, and GLM
and ties the connected-incompatible condition for Qwen. Five disconnected tools
have a higher exact success rate than executable goal-irrelevant tools for every
model.
DeepSeek-V4.1-Flash has one technical failure in each connected condition; every other
model-condition cell has zero technical failures.

## Interpretation boundary

The result makes a pure tool-count explanation less plausible across all five
models. It does not
identify one tool-usability property as a single causal factor because
connectivity, type compatibility, and runtime execution remain bundled in the
conditions. The sample
also remains limited to one dependency family and five model aliases.

## Canonical files

- merged metrics:
  `results/funcbench-control-five-model-analysis-20261101/metric_summary.csv`
- merged trials:
  `results/funcbench-control-five-model-analysis-20261101/trial_failures.csv`
- merge summary:
  `results/funcbench-control-five-model-analysis-20261101/merge_summary.json`
- disconnected extension analysis:
  `results/funcbench-control-disconnected-extension15-20261101T085735Z/error_analysis/`
- incompatible extension analysis:
  `results/funcbench-control-incompatible-extension15-20261101T090905Z/error_analysis/`

## Reproduction

The five-model main panel supplies the zero-extra-tool and executable
goal-irrelevant cells. The four per-model control analyses and both
MiniMax control analyses are merged into that panel:

```powershell
.\.venv\Scripts\python.exe -m llm_eval.cli funcbench-merge `
  --analysis results\funcbench-main15-five-model-20261101 `
  --analysis results\funcbench-control-disconnected-20261030T133754Z\error_analysis `
  --analysis results\funcbench-control-incompatible-20261030T134725Z\error_analysis `
  --analysis results\funcbench-control-disconnected-extension15-20261101T085735Z\error_analysis `
  --analysis results\funcbench-control-incompatible-extension15-20261101T090905Z\error_analysis `
  --analysis results\funcbench-minimax-control-disconnected-20261101T113958Z\error_analysis `
  --analysis results\funcbench-minimax-control-incompatible-20261101T114402Z\error_analysis `
  --output results\funcbench-control-five-model-analysis-20261101 `
  --name affordance_controls15_five_models
```

The merged analysis contains 300 rows and 15 trials in every model/mode cell.
The internal merge name `affordance_controls15_five_models` is retained as a
legacy reproducibility identifier.
