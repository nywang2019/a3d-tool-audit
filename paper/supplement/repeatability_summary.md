# Eight-graph repeatability audit

- Runs: `results\funcbench-paired-repeatability8-20261101T081739Z`, `results\funcbench-minimax-repeatability8-20261101T115312Z`
- Trials: 80
- Same correctness: 67/80
- Correctness kappa: 0.673
- Same primary category: 56/80
- Same final-answer signature: 49/80
- Technical/budget failures: 0
- Model interactions: 447
- Input/output/reasoning tokens: 489,718 / 55,384 / 36,776

## Agreement by model

| Model | Original | Repeat | Same correctness | Kappa | Same category | Repeat 5-0 effect |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek | 13/16 | 12/16 | 15/16 | 0.818 | 14/16 | -87.5, -12.5 pp |
| Kimi | 8/16 | 6/16 | 14/16 | 0.750 | 13/16 | -87.5, -12.5 pp |
| GLM | 4/16 | 4/16 | 14/16 | 0.667 | 9/16 | +0.0, +0.0 pp |
| MiniMax | 10/16 | 11/16 | 11/16 | 0.310 | 10/16 | -87.5, +12.5 pp |
| Qwen | 10/16 | 9/16 | 13/16 | 0.613 | 10/16 | -75.0, +12.5 pp |

## Condition counts

| Model | Original 0 | Repeat 0 | Original 5 | Repeat 5 | Original delta | Repeat delta |
|---|---:|---:|---:|---:|---:|---:|
| DeepSeek | 8/8 | 8/8 | 5/8 | 4/8 | -37.5 pp | -50.0 pp |
| Kimi | 5/8 | 5/8 | 3/8 | 1/8 | -25.0 pp | -50.0 pp |
| GLM | 3/8 | 2/8 | 1/8 | 2/8 | -25.0 pp | +0.0 pp |
| MiniMax | 7/8 | 7/8 | 3/8 | 4/8 | -50.0 pp | -37.5 pp |
| Qwen | 6/8 | 6/8 | 4/8 | 3/8 | -25.0 pp | -37.5 pp |

The repeat run uses the same graph indices and generator seed as the
original run, but is a fresh model sample. It is therefore a
sampling-repeatability audit, not an independent new graph sample.
