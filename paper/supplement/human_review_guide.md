# Human trace review guide

## Purpose

Validate the automated primary-error labels used in the paper and estimate how
often a reviewer would incorrectly flag a successful trace.

Recommended minimum:

- 32 non-technical/error cases, stratified across models and conditions;
- 8 correct control cases;
- 40 total cases;
- two independent reviewers who do not see the automated label.

If resources are tight, review 30 error cases and 6 correct controls. Do not
review only failures.

## Estimated time

Recommended 40-case design:

| Activity | Time |
|---|---:|
| Read rubric and inspect three worked examples | 30-45 minutes |
| Joint calibration on five non-evaluation cases | 45-60 minutes |
| Independent review | 2-3.5 hours per reviewer |
| Compare and adjudicate disagreements | 45-75 minutes |
| Compute kappa, confusion matrix, and write short summary | 30-60 minutes |

Total team effort: approximately **6-9 person-hours**.  
Calendar time: **1-2 working days** if the two reviewers can work in parallel.

A 30-case minimum is approximately 5-7 person-hours.

## What each reviewer receives

One record per case, with:

- a randomized review ID;
- the task prompt;
- the target variable and expected value;
- the ordered function calls, arguments, returned values, and tool errors;
- the final model answer.

Hide these fields during independent review:

- model name;
- extra-tool count and condition;
- trial outcome;
- automated primary category;
- automated trace scores;
- token counts;
- the other reviewer's labels.

Keep a separate private key mapping review ID to the original trial.

## Step 1: Freeze the sample before labeling

Select a stratified sample from:

`results/funcbench-main20-analysis-20261030/trial_failures.csv`

Recommended allocation:

- 8 error cases per model where available;
- mix zero- and five-extra-tool failures;
- include every rare category represented in the main run, including the one
  technical/budget failure;
- add 2 correct controls per model;
- randomize the order and assign anonymous review IDs.

Do not replace difficult cases after seeing reviewer disagreement.

## Step 2: Train on examples

Choose five traces that are not part of the 40-case evaluation set.

Both reviewers independently label them, then discuss:

- where the first decisive error occurred;
- whether the trace recovered;
- why a later symptom is not the primary category;
- when the case should be labeled `unclear`.

Revise only the rubric wording, not the substantive target claims.

## Step 3: Label independently

Use `human_review_template.csv`.

First decide final correctness:

- `correct`;
- `incorrect`;
- `unclear`.

Then decide the primary action error:

- `none`;
- `wrong_argument_value`;
- `premature_dependency`;
- `irrelevant_function`;
- `premature_final`;
- `technical`;
- `other`;
- `unclear`.

Additional evidence fields:

- `first_error_order`: call number of the first decisive error;
- `recovered`: whether the model later repaired the error;
- `evidence_quote`: a short excerpt or call number supporting the decision;
- `confidence`: 1 = low, 2 = medium, 3 = high;
- `notes`: concise explanation.

## Step 4: Apply one decision rule consistently

Use the **earliest decisive error** as the primary category.

### Wrong argument value

A call uses a known value that is wrong for the function's required argument.

### Premature dependency

A call requires a variable that has not yet been produced.

### Irrelevant function

The chosen function is not in the required dependency set and is used even
though the goal-relevant action was available or needed.

### Premature final

The model stops and answers before computing the target.

### Technical

No usable model response was returned, for example an empty response or
provider-level failure.

Do not label a tool error as `technical` when the model caused it by supplying
an invalid or unavailable dependency. That is a model/tool-use error.

### Correct with recovery

If the final answer is correct after one or more intermediate errors:

- set `final_outcome=correct`;
- record the primary action error;
- set `recovered=true`.

The paper should distinguish final success from clean execution.

### Multiple errors

If several errors are present, choose the earliest error that caused an
invalid state or diverted the plan. Missing a later function is a symptom, not
the primary category, when the wrong branch already made the target
unreachable.

## Step 5: Compare without changing labels

Before discussion, export both reviewers' files and calculate:

- percent agreement on final correctness;
- percent agreement on primary error;
- Cohen's kappa for both fields;
- a confusion matrix for primary errors;
- confidence and unclear counts.

Reference interpretation for Cohen's kappa:

- below 0.40: weak;
- 0.41-0.60: moderate;
- 0.61-0.80: substantial;
- above 0.80: strong.

A practical target is at least **0.60**, preferably **0.70** for the main
primary-error categories.

Standard-library calculation:

```python
def cohen_kappa(a, b):
    assert len(a) == len(b) and a
    n = len(a)
    observed = sum(x == y for x, y in zip(a, b)) / n
    labels = set(a) | set(b)
    expected = sum(
        (a.count(label) / n) * (b.count(label) / n)
        for label in labels
    )
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0
```

If kappa is low:

- inspect which categories are confused;
- revise examples and decision rules;
- relabel the full sample only once under a frozen revised rubric;
- report the revision and never hide the first disagreement result.

## Step 6: Adjudicate disagreements

Meet for 45-75 minutes:

1. reviewers explain their earliest-error evidence;
2. the coordinator shows the required dependency sequence;
3. assign an adjudicated label;
4. record whether the disagreement came from ambiguous tasks, rubric
   interpretation, or reviewer error;
5. do not inspect aggregate model performance while adjudicating.

Report both the original agreement and the adjudicated labels.

## Step 7: Update the paper

Add a small table:

| Field | Result |
|---|---:|
| Reviewed traces | 40 |
| Correct controls | 8 |
| Error traces | 32 |
| Final-correctness kappa | actual value |
| Primary-error kappa | actual value |
| Unclear before adjudication | actual count |
| Adjudicated label changes | actual count |

Then state:

- automated labels are heuristic or validated according to the result;
- reviewers were blinded to model/condition;
- disagreements were resolved without using model-level aggregate outcomes;
- remaining uncertain cases stay reported as uncertain.

## Recommended division of labor

- Reviewer A and Reviewer B: one independent labeling session each.
- Coordinator: samples cases, removes metadata, computes agreement, and records
  adjudication.
- A third person can adjudicate if the two reviewers cannot resolve a case, but
  do not let the coordinator silently overwrite labels.

