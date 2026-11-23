# MiniMax M3.1 Flash strict-paired evaluation

Date: 2026-11-01

## Model and endpoint

- Alias: `minimax-m3.1-flash`
- Model ID: `MiniMax-M3.1-Flash`
- Endpoint: `https://api.minimaxi.com/v1`
- Protocol: OpenAI-compatible chat completions
- Temperature: 1.0
- Top-p: 0.95
- Maximum output: 8192 tokens
- Thinking effort: `low`
- Direct connection: `trust_env=false`

MiniMax does not allow thinking to be disabled for this model. The lowest
advertised effort is `low`. The experiment therefore records this as a model
configuration difference rather than claiming disabled reasoning.

## Compatibility audit

The project was changed to preserve `reasoning_content` in the assistant
message across tool turns. This follows MiniMax's requirement for interleaved
thinking and multi-turn tool use. A regression test covers the message-history
path.

A four-trial smoke test on graphs 0 and 1 was run before the formal experiment
and excluded from the paper:

- 22 HTTP attempts, zero technical failures;
- one correct outcome, two wrong-argument outcomes, one premature-dependency
  outcome;
- two graph pairs passed required-fingerprint, target-truth, and core-tool-order
  audit.

## Formal run

Run directory:

`results/funcbench-minimax-main20-20261101T105025Z`

Result:

| Condition | Correct | Trials | Rate |
|---|---:|---:|---:|
| 0 extra tools | 16 | 20 | 80.0% |
| 5 executable goal-irrelevant tools | 8 | 20 | 40.0% |
| Total | 24 | 40 | 60.0% |

Paired effect:

- delta: -40.0 percentage points;
- graph bootstrap 95% CI: [-65.0, -15.0];
- exact McNemar p: 0.0215;
- Holm-adjusted p across five models: 0.1074.

The interval excludes zero, but the effect does not survive the five-model Holm
multiplicity check. It should be described as a repeatable directional decline
in this sample, not as a universally significant model effect.

## Runtime and failures

- Successful model responses: 212;
- HTTP attempts including retries: 283;
- rate-limit retries: 71;
- final technical failures: 0;
- input tokens: 317,056;
- output tokens: 34,343;
- reasoning tokens: 25,459;
- elapsed model time: 577.9 seconds.

The 20 graph pairs passed `funcbench-audit`: identical required-graph
fingerprints and target values, different full tool-condition fingerprints, and a
five-tool toolkit delta.

## Error categories

| Category | Count |
|---|---:|
| Correct | 24 |
| Wrong argument value | 3 |
| Premature dependency | 6 |
| Irrelevant function | 7 |
| Premature final | 0 |
| Technical/budget | 0 |

With zero extra tools, MiniMax had 16 correct trials, two wrong-argument trials,
and two premature-dependency trials. With five extra tools, MiniMax had eight
correct trials, one wrong-argument trial, four premature-dependency trials, and
seven irrelevant-function trials.

## Five-model aggregate

Merged analysis:

`results/funcbench-main20-five-model-analysis-20261101`

The full five-model table is:

| Model | 0 | 5 compatible | Delta |
|---|---:|---:|---:|
| DeepSeek-V4.1-Flash | 20/20 | 14/20 | -30.0 pp |
| Kimi K3.1 Pro | 13/20 | 7/20 | -30.0 pp |
| GLM-5.3-Flashx | 7/20 | 5/20 | -10.0 pp |
| MiniMax-M3.1-Flash | 16/20 | 8/20 | -40.0 pp |
| Qwen3.8-Max | 17/20 | 11/20 | -30.0 pp |

The five-model graph-cluster permutation interaction test gives p=0.488 with a
30-percentage-point effect range. The data do not establish statistically
heterogeneous model-specific effects.

## Auxiliary five-model panels

After the main comparison, MiniMax was added to every auxiliary experiment so
that all figures and tables use the same five-model panel.

| Auxiliary analysis | MiniMax result |
|---|---|
| Dose: 0 / 2 / 5 compatible | 12/15, 13/15, 5/15 |
| Tool usability: 0 / disconnected / incompatible / goal-irrelevant | 12/15, 13/15, 11/15, 5/15 |
| Prerequisite gating | 3/8 ungated to 1/8 gated |
| State reminder | 2/4 baseline to 2/4 reminder |
| Repeatability: original 0 / repeat 0 / original 5 / repeat 5 | 7/8, 7/8, 3/8, 4/8 |

The merged five-model repeatability panel preserves 67/80 correctness labels
(Cohen's kappa 0.673). MiniMax contributed 77 new auxiliary trials in addition
to the 40 main trials in this report. These are condition-level attempts with
reused baseline cells, not additional independent graphs.

## Reproduction

```powershell
.\.venv\Scripts\python.exe -m llm_eval.cli funcbench-audit `
  --run results\funcbench-minimax-main20-20261101T105025Z

.\.venv\Scripts\python.exe -m llm_eval.cli funcbench-errors `
  --run results\funcbench-minimax-main20-20261101T105025Z `
  --output results\funcbench-minimax-main20-20261101T105025Z\error_analysis

.\.venv\Scripts\python.exe -m llm_eval.cli funcbench-merge `
  --analysis results\funcbench-main20-analysis-20261030 `
  --analysis results\funcbench-minimax-main20-20261101T105025Z\error_analysis `
  --output results\funcbench-main20-five-model-analysis-20261101 `
  --name main20_five_models

.\.venv\Scripts\python.exe -m llm_eval.cli funcbench-heterogeneity `
  --analysis results\funcbench-main20-five-model-analysis-20261101 `
  --output results\funcbench-main20-five-model-analysis-20261101\heterogeneity
```
