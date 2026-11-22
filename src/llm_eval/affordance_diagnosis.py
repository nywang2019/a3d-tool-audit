"""Affordance-aware diagnosis for saved multi-step tool-use trajectories.

The diagnostic layer consumes a saved trial and deterministic classifier output.
It does not call a model or mutate the source run.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .funcbench import prepare_condition
from .funcbench_errors import classify_trial


class DiagnosisError(ValueError):
    """Raised when a saved trial cannot be diagnosed."""


def _condition_from_trial(trial: dict[str, Any], experiment: dict[str, Any]):
    return prepare_condition(
        depth=int(trial["depth"]),
        distractor_count=int(trial["distractor_count"]),
        graph_seed=int(trial["graph_seed"]),
        core_nodes=int(experiment["core_nodes"]),
        trial_seed=0,
        use_types=bool(experiment.get("use_types", True)),
        use_subtypes=bool(experiment.get("use_subtypes", True)),
        num_supertypes=int(experiment.get("num_supertypes", 5)),
        show_variable_names=bool(experiment.get("show_variable_names", False)),
        rename_variables=bool(experiment.get("rename_variables", False)),
        num_noise_inputs=int(experiment.get("num_noise_inputs", 0)),
        repeat_known_variable_values=bool(
            trial.get("repeat_known_variable_values", False)
        ),
        paired_core_graph=bool(experiment.get("paired_core_graph", False)),
        stable_tool_order=bool(
            experiment.get(
                "stable_tool_order",
                experiment.get("paired_core_graph", False),
            )
        ),
        distractor_mode=str(
            experiment.get("distractor_mode", "connected_compatible")
        ),
    )


def _first_failed_call(
    calls: list[dict[str, Any]],
    primary_category: str,
    irrelevant_functions: list[str],
) -> dict[str, Any] | None:
    for call in calls:
        failure = str(call.get("failure") or "").lower()
        if primary_category == "premature_dependency" and "not yet known" in failure:
            return call
        if primary_category == "wrong_argument_value" and "incorrect" in failure:
            return call
        if primary_category == "unknown_function" and "func not in graph" in failure:
            return call
    if primary_category == "irrelevant_function":
        for call in calls:
            if str(call.get("function")) in set(irrelevant_functions):
                return call
    return None


def diagnose_from_classification(
    trial: dict[str, Any],
    classification: dict[str, Any],
) -> dict[str, Any]:
    """Build an actionable diagnosis from a trial and classifier output."""
    category = str(classification.get("primary_category", "final_answer_error"))
    calls = list(trial.get("calls", []))
    irrelevant = list(classification.get("irrelevant_functions", []))
    missing = list(classification.get("missing_required_functions", []))
    first_call = _first_failed_call(calls, category, irrelevant)

    narratives = {
        "correct": (
            "The trajectory reached the target with the required dependency "
            "closure and no observed decisive action failure."
        ),
        "irrelevant_function": (
            "An executable distractor displaced the goal-relevant plan. The "
            "model entered a tool branch outside the target dependency set."
        ),
        "premature_dependency": (
            "The model attempted to consume a variable before its producer had "
            "run, so the required computation state was unavailable."
        ),
        "wrong_argument_value": (
            "The function choice could be valid, but the call supplied an "
            "incorrect argument value and produced an invalid intermediate state."
        ),
        "premature_final": (
            "The model terminated before producing the target variable, leaving "
            "the dependency chain incomplete."
        ),
        "unknown_function": (
            "The model selected a function that was not available in the "
            "current tool graph."
        ),
        "technical_or_budget": (
            "The run did not yield a usable completed trajectory because of a "
            "technical or budget failure; this must not be interpreted as a model "
            "reasoning error."
        ),
        "incomplete_chain": (
            "At least one required function was never called, so the dependency "
            "closure could not produce the target."
        ),
        "final_answer_error": (
            "The observed calls form the required dependency set, but the final "
            "answer does not match the program-computed target."
        ),
        "classification_error": (
            "The saved trial could not be classified. Inspect the original trace "
            "and classifier error before drawing a conclusion."
        ),
    }
    recommendations = {
        "correct": ["No repair required."],
        "irrelevant_function": [
            "Compare every selected function with the target dependency closure.",
            "Check whether a type-compatible distractor is becoming executable.",
            "Do not hide the full candidate set; validate goal relevance before use.",
        ],
        "premature_dependency": [
            "Track producer variables and permit a call only after all inputs exist.",
            "Preserve the full tool list and validate argument provenance.",
        ],
        "wrong_argument_value": [
            "Validate arguments against the current variable state.",
            "Trace the producer of every non-input argument value.",
        ],
        "premature_final": [
            "Require the target variable to exist before accepting a final answer.",
            "Score dependency completion separately from answer formatting.",
        ],
        "unknown_function": [
            "Validate function identifiers against the exposed tool schema.",
        ],
        "technical_or_budget": [
            "Record the response failure and rerun only if the protocol permits.",
            "Do not count the attempt as a model reasoning failure.",
        ],
        "incomplete_chain": [
            "Identify the first missing required function and its unmet inputs.",
            "Check whether an irrelevant branch displaced the required action.",
        ],
        "final_answer_error": [
            "Recompute the target from the trace and compare intermediate values.",
            "Separate execution success from final-response extraction.",
        ],
        "classification_error": [
            "Inspect the classifier exception and raw saved response.",
        ],
    }
    risk = (
        "none"
        if category == "correct"
        else "high"
        if category
        in {
            "irrelevant_function",
            "premature_dependency",
            "wrong_argument_value",
            "premature_final",
        }
        else "critical"
        if category in {"technical_or_budget", "classification_error"}
        else "medium"
    )
    first_order = first_call.get("order") if first_call else None
    report = {
        "schema_version": 1,
        "trial_id": trial.get("trial_id"),
        "model_alias": trial.get("model_alias"),
        "target_variable": trial.get("target_variable"),
        "target_value": trial.get("target_value"),
        "final_correct": bool(trial.get("correct")),
        "outcome": trial.get("outcome"),
        "primary_category": category,
        "risk_level": risk,
        "first_decisive_call_order": first_order,
        "first_decisive_call": first_call,
        "irrelevant_functions": irrelevant,
        "missing_required_functions": missing,
        "call_failures": list(classification.get("call_failures", [])),
        "required_recall": classification.get("required_recall"),
        "selection_precision": classification.get("selection_precision"),
        "dependency_order_score": classification.get("dependency_order_score"),
        "argument_score": classification.get("argument_score"),
        "dependency_trace_fidelity": classification.get(
            "dependency_trace_fidelity"
        ),
        "narrative": narratives.get(category, narratives["classification_error"]),
        "recommended_checks": recommendations.get(
            category, recommendations["classification_error"]
        ),
        "diagnostic_scope": (
            "Rule-derived diagnosis from saved trace evidence; not human-validated "
            "ground truth."
        ),
    }
    return report


def diagnose_saved_trial(
    run_root: Path | str,
    *,
    model_alias: str,
    trial_id: str,
) -> dict[str, Any]:
    """Load and diagnose one trial from a completed FuncBenchGen run."""
    root = Path(run_root)
    config_path = root / "config.snapshot.json"
    trials_path = root / model_alias / "trials.json"
    if not config_path.exists():
        raise DiagnosisError(f"Missing config snapshot: {config_path}")
    if not trials_path.exists():
        raise DiagnosisError(f"Missing model trials file: {trials_path}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    trials = json.loads(trials_path.read_text(encoding="utf-8"))
    trial = next(
        (row for row in trials if str(row.get("trial_id")) == trial_id),
        None,
    )
    if trial is None:
        raise DiagnosisError(
            f"Trial {trial_id!r} not found for model {model_alias!r}"
        )
    condition = _condition_from_trial(trial, config["experiment"])
    classification = classify_trial(trial, condition)
    return diagnose_from_classification(trial, classification)


def render_markdown(report: dict[str, Any]) -> str:
    """Render a diagnosis report for a paper appendix or artifact demo."""
    first = report.get("first_decisive_call")
    first_text = "none"
    if first:
        first_text = (
            f"call {first.get('order')}: {first.get('function')} "
            f"with arguments {first.get('arguments')}"
        )
    checks = "\n".join(
        f"- {item}" for item in report.get("recommended_checks", [])
    )
    return "\n".join(
        [
            f"# Affordance-aware diagnosis: {report['trial_id']}",
            "",
            f"- Model: `{report['model_alias']}`",
            f"- Final correct: `{report['final_correct']}`",
            f"- Primary diagnosis: `{report['primary_category']}`",
            f"- Risk: `{report['risk_level']}`",
            f"- First decisive call: {first_text}",
            f"- Required recall: `{report['required_recall']}`",
            f"- Selection precision: `{report['selection_precision']}`",
            f"- Trace fidelity: `{report['dependency_trace_fidelity']}`",
            "",
            "## Interpretation",
            "",
            report["narrative"],
            "",
            "## Recommended checks",
            "",
            checks,
            "",
            "## Scope",
            "",
            report["diagnostic_scope"],
            "",
        ]
    )

