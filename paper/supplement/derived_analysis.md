# Generated analysis summary

This file is generated from the canonical formal result files. No model API calls are made.

- Main graphs: 20
- Model-interaction permutation p: 0.488
- Main technical/budget failures: 1

## Main experiment

| Model | 0 | 5 | Delta | 95% CI | McNemar p | Holm p |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek-V4.1-Flash | 20/20 | 14/20 | -30.0 pp | [-50.0, -10.0] | 0.0312 | 0.1250 |
| Kimi K3.1 Pro | 13/20 | 7/20 | -30.0 pp | [-55.0, -5.0] | 0.0703 | 0.2109 |
| GLM-5.3-Flashx | 7/20 | 5/20 | -10.0 pp | [-35.0, +15.0] | 0.6875 | 0.6875 |
| MiniMax-M3.1-Flash | 16/20 | 8/20 | -40.0 pp | [-65.0, -15.0] | 0.0215 | 0.1074 |
| Qwen3.8-Max | 17/20 | 11/20 | -30.0 pp | [-60.0, -5.0] | 0.1094 | 0.2188 |

## Trace sensitivity

| model | extra_tools | attempts | technical_failures | valid_success | complete_trace | wrong_complete_trace |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek-V4.1-Flash | 0 | 20 | 0 | 20/20 | 20/20 | 0 |
| DeepSeek-V4.1-Flash | 5 | 20 | 1 | 14/19 | 10/20 | 0 |
| Kimi | 0 | 20 | 0 | 13/20 | 13/20 | 0 |
| Kimi | 5 | 20 | 0 | 7/20 | 4/20 | 0 |
| GLM | 0 | 20 | 0 | 7/20 | 7/20 | 0 |
| GLM | 5 | 20 | 0 | 5/20 | 2/20 | 0 |
| MiniMax | 0 | 20 | 0 | 16/20 | 16/20 | 0 |
| MiniMax | 5 | 20 | 0 | 8/20 | 3/20 | 0 |
| Qwen | 0 | 20 | 0 | 17/20 | 17/20 | 0 |
| Qwen | 5 | 20 | 0 | 11/20 | 7/20 | 0 |

## Repeatability audit

- Correctness agreement: 67/80
- Correctness kappa: 0.673
- Primary-category agreement: 56/80
- Final-answer signature agreement: 49/80
- Model interactions: 447
- Technical/budget failures: 0

## Tool-usability controls

| Model | 0 | Disconnected 5 | Connected-incompatible 5 | Executable goal-irrelevant 5 |
|---|---:|---:|---:|---:|
| DeepSeek-V4.1-Flash | 15/15 | 14/15 | 12/15 | 10/15 |
| Kimi | 11/15 | 10/15 | 7/15 | 5/15 |
| GLM | 4/15 | 8/15 | 4/15 | 3/15 |
| MiniMax | 12/15 | 13/15 | 11/15 | 5/15 |
| Qwen | 12/15 | 14/15 | 8/15 | 8/15 |

All four conditions use the same 15 required graphs. Executable goal-irrelevant tools are lowest or tied for lowest for every model.

## ASP paired loss

| Model | ASP | 95% CI | Sign-flip p | Holm p | +/0/- |
|---|---:|---:|---:|---:|---:|
| DeepSeek-V4.1-Flash | +20.0 pp | [-3.3, +43.3] | 0.2188 | 0.5625 | 5/8/2 |
| Kimi | +23.3 pp | [+3.3, +43.3] | 0.0938 | 0.3750 | 6/8/1 |
| GLM | +20.0 pp | [-6.7, +43.3] | 0.2461 | 0.5625 | 7/6/2 |
| MiniMax | +46.7 pp | [+23.3, +70.0] | 0.0068 | 0.0342 | 10/4/1 |
| Qwen | +20.0 pp | [-3.3, +43.3] | 0.1875 | 0.5625 | 6/7/2 |

ASP is the mean of the disconnected-minus-goal-irrelevant and connected-incompatible-minus-goal-irrelevant contrasts on each graph.

## Schema-matched sealed probe

| Model | Executable | Sealed | EAP | 95% CI | Holm p | +/0/- |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek-V4.1-Flash | 7/10 | 9/10 | +20.0 pp | [+0.0, +50.0] | 1.0000 | 2/8/0 |
| Kimi | 4/10 | 4/10 | +0.0 pp | [-30.0, +30.0] | 1.0000 | 1/8/1 |
| GLM | 1/10 | 5/10 | +40.0 pp | [+10.0, +70.0] | 0.6250 | 4/6/0 |
| MiniMax | 3/10 | 7/10 | +40.0 pp | [+10.0, +70.0] | 0.6250 | 4/6/0 |
| Qwen | 6/10 | 8/10 | +20.0 pp | [-20.0, +60.0] | 1.0000 | 3/6/1 |


## Reproducibility note

`build_paper_assets.py` is deterministic and reads only local result summaries. The source tables remain in `results/`; this script does not copy or mutate them.