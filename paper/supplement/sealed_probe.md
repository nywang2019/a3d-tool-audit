# Schema-matched sealed-twin probe and EAP

## Purpose

The sealed probe tests whether the paired loss depends on tools actually running
rather than merely appearing in the JSON tool list.

For 10 shared required graphs, the executable and sealed conditions have:

- identical prompt and input variables;
- identical target and required-graph fingerprints;
- identical tool names, descriptions, schemas, arguments, and order;
- identical full graph fingerprints;
- the same five executable goal-irrelevant tools.

The only difference is runtime behavior. In the executable condition, an
executable goal-irrelevant tool runs normally. In the sealed condition, the
model receives:

`Tool invocation failed: this tool is currently sealed.`

## Definition

EAP is a historical metric name for the paired loss between the sealed and
executable conditions. For model \(m\) and graph \(g\):

\[
\operatorname{EAP}_m
=100\,\mathbb{E}_g[Y_{\text{sealed}}-Y_{\text{executable}}].
\]

A positive value means that sealing improved the exact success rate.

## Results

| Model | Executable | Sealed | EAP |
|---|---:|---:|---:|
| DeepSeek | 7/10 | 9/10 | +20.0 pp |
| Kimi | 4/10 | 4/10 | 0.0 pp |
| GLM | 1/10 | 5/10 | +40.0 pp |
| MiniMax | 3/10 | 7/10 | +40.0 pp |
| Qwen | 6/10 | 8/10 | +20.0 pp |

No model effect survives Holm correction with 10 graphs. The probe is therefore
exploratory construct-validity evidence, not confirmation that executable
execution is the sole causal factor.

## Verification

- 10/10 graph-seed pairs matched on prompt, tool list, full fingerprint,
  required fingerprint, and target value;
- 50 sealed trials and zero final technical/budget failures;
- all five models were run under the same protocol.
