# A3D diagnosis: d3-x5-g0-r0

- Model: `glm-5.3-flashx`
- Final correct: `False`
- Primary diagnosis: `premature_final`
- Risk: `high`
- First decisive call: none
- Required recall: `0.25`
- Selection precision: `0.3333333333333333`
- Trace fidelity: `0.6458333333333333`

## Interpretation

The model terminated before producing the target variable, leaving the dependency chain incomplete.

## Recommended checks

- Require the target variable to exist before accepting a final answer.
- Score dependency completion separately from answer formatting.

## Scope

Rule-derived diagnosis from saved trace evidence; not human-validated ground truth.
