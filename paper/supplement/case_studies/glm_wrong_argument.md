# A3D diagnosis: d3-x5-g3-r0

- Model: `glm-5.3-flashx`
- Final correct: `False`
- Primary diagnosis: `wrong_argument_value`
- Risk: `high`
- First decisive call: call 7: func_ndq with arguments {"ihd":244,"soch":240}
- Required recall: `1.0`
- Selection precision: `0.5714285714285714`
- Trace fidelity: `0.8571428571428571`

## Interpretation

The function choice could be valid, but the call supplied an incorrect argument value and produced an invalid intermediate state.

## Recommended checks

- Validate arguments against the current variable state.
- Trace the producer of every non-input argument value.

## Scope

Rule-derived diagnosis from saved trace evidence; not human-validated ground truth.
