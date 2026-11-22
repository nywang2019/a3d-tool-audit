# A3D diagnostic protocol

## Input

One saved FuncBenchGen trial containing:

- trial identifier, graph seed, depth, and extra-tool count;
- target variable and program-computed truth;
- ordered tool calls;
- function names, arguments, returned values, and call failures;
- final answer and outcome label.

The completed run configuration supplies the graph-generation settings needed to
reconstruct the required dependency graph.

## Output

The A3D report contains:

- final-correctness status;
- primary rule-derived diagnosis;
- risk level;
- first decisive error and its supporting call, where available;
- irrelevant and missing required functions;
- trace metrics;
- natural-language interpretation;
- recommended checks.

## Classification rules

The implementation applies these categories in the current priority order:

1. `correct` when the trial is marked correct;
2. `technical_or_budget` when there is no usable completed model trajectory;
3. `premature_final` when the trial outcome is a shortcut;
4. `unknown_function` when a called function is absent from the graph;
5. `premature_dependency` when a call failure reports that a variable is not
   yet known;
6. `wrong_argument_value` when a call failure reports an incorrect value;
7. `irrelevant_function` when a called graph node lies outside the required
   target closure;
8. `incomplete_chain` when required functions are missing;
9. `final_answer_error` when no earlier rule applies.

These are deterministic heuristics. They are not human-annotated ground truth,
and the present submission makes no inter-rater-agreement claim.

## Diagnostic metrics

- Required recall: unique required functions called divided by required
  functions.
- Selection precision: unique required functions called divided by all unique
  called functions.
- Dependency-order score: one minus the fraction of observed required edges
  called in reverse order.
- Argument score: one minus invalid-call count divided by total calls.
- Trace fidelity: equal-weight mean of the four components.

Exact success remains the primary outcome. Trace metrics are explanatory
diagnostics.

## Running the component

The executable name `affordance-diagnose` is a legacy interface identifier. It
does not define a separate tool property.

```powershell
.\.venv\Scripts\python.exe -m llm_eval.cli affordance-diagnose `
  --run results\funcbench-paired-main-v2-20261030T122222Z `
  --model deepseek-flash `
  --trial-id d3-x5-g5-r0 `
  --format markdown `
  --output paper\supplement\case_studies\deepseek_irrelevant.md
```

The command performs no model or network call.

