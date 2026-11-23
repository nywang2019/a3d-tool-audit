# Prerequisite-hiding error transitions

Only trials whose primary category changed after graph-matched prerequisite hiding are shown.

| Model | Unhidden category | Hidden category | Count |
|---|---:|---:|---:|
| DeepSeek-V4.1-Flash | correct | irrelevant_function | 9 |
| DeepSeek-V4.1-Flash | correct | premature_final | 2 |
| DeepSeek-V4.1-Flash | premature_dependency | irrelevant_function | 1 |
| Kimi K3.1 Pro | correct | irrelevant_function | 3 |
| Kimi K3.1 Pro | premature_dependency | irrelevant_function | 2 |
| Kimi K3.1 Pro | premature_dependency | wrong_argument_value | 2 |
| Kimi K3.1 Pro | correct | wrong_argument_value | 1 |
| Kimi K3.1 Pro | wrong_argument_value | irrelevant_function | 1 |
| Kimi K3.1 Pro | premature_final | wrong_argument_value | 1 |
| GLM-5.3-Flashx | premature_dependency | wrong_argument_value | 4 |
| GLM-5.3-Flashx | wrong_argument_value | irrelevant_function | 2 |
| GLM-5.3-Flashx | correct | premature_final | 2 |
| GLM-5.3-Flashx | wrong_argument_value | premature_final | 1 |
| GLM-5.3-Flashx | correct | wrong_argument_value | 1 |
| GLM-5.3-Flashx | correct | irrelevant_function | 1 |
| GLM-5.3-Flashx | premature_final | irrelevant_function | 1 |
| MiniMax-M3.1-Flash | correct | wrong_argument_value | 5 |
| MiniMax-M3.1-Flash | irrelevant_function | wrong_argument_value | 3 |
| MiniMax-M3.1-Flash | correct | irrelevant_function | 2 |
| MiniMax-M3.1-Flash | correct | premature_final | 1 |
| MiniMax-M3.1-Flash | premature_dependency | premature_final | 1 |
| Qwen3.8-Max | correct | premature_final | 4 |
| Qwen3.8-Max | correct | irrelevant_function | 4 |
| Qwen3.8-Max | premature_dependency | wrong_argument_value | 1 |
| Qwen3.8-Max | wrong_argument_value | correct | 1 |
| Qwen3.8-Max | correct | wrong_argument_value | 1 |
| Qwen3.8-Max | premature_dependency | irrelevant_function | 1 |
