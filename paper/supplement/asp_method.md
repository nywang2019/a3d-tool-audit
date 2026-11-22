# ASP as a paired-loss summary

## Construct

ASP is a historical metric name for the average within-graph paired loss. The
comparison condition contains five executable goal-irrelevant tools: they are
connected, type-compatible, and able to run, but lie outside the required tool
chain. The two controls contain five disconnected tools and five
connected-incompatible tools on the same required graph.

For model \(m\) and graph \(g\), let \(Y_D,Y_I,Y_{GI}\in\{0,1\}\) denote exact
success outcomes with five disconnected tools, five connected-incompatible
tools, and five executable goal-irrelevant tools. The graph score and model
score are:

\[
a_{m,g}=\frac{Y_D+Y_I}{2}-Y_{GI},
\qquad
\operatorname{ASP}_m=100\,
\mathbb{E}_g[a_{m,g}].
\]

Units are percentage points. A positive score means that the executable
goal-irrelevant condition has a lower exact success rate than the average of the
two controls. The components
\(Y_D-Y_{GI}\) and \(Y_I-Y_{GI}\) remain visible so the metric does not conceal how
the two contrasts contribute.

## Inference

- The graph is the resampling unit.
- Confidence intervals use a deterministic 4,000-sample graph bootstrap.
- The two-sided sign-flip test uses the exact distribution of graph-level scores
  under sign symmetry.
- Holm correction is applied across five models.
- `ASP_solvable` repeats the calculation only on graphs solved without extra
  tools and is explicitly a sensitivity analysis.

## Interpretation limits

ASP is a descriptive paired loss, not a causal mediation estimate. The three
tool-usability properties and associated graph structure remain bundled across
the three non-baseline conditions. A positive ASP therefore supports a
condition-specific paired loss in this design; it does not identify a single
internal mechanism. Equal weighting of the two control contrasts is a
pre-specified design choice.

The result is also bounded to five model aliases, one function-dependency
family, depth three, and 15 shared control graphs.
