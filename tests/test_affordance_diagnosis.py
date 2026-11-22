from pathlib import Path

import pytest

from llm_eval.affordance_diagnosis import (
    diagnose_from_classification,
    diagnose_saved_trial,
    render_markdown,
)


def _trial(calls):
    return {
        "trial_id": "diag-1",
        "model_alias": "model-a",
        "target_variable": "target",
        "target_value": 10,
        "correct": False,
        "outcome": "model_error",
        "calls": calls,
    }


def test_correct_trial_needs_no_repair():
    report = diagnose_from_classification(
        _trial([]),
        {
            "primary_category": "correct",
            "required_recall": 1.0,
            "selection_precision": 1.0,
            "dependency_order_score": 1.0,
            "argument_score": 1.0,
            "dependency_trace_fidelity": 1.0,
        },
    )
    assert report["risk_level"] == "none"
    assert report["recommended_checks"] == ["No repair required."]
    assert "reached the target" in report["narrative"]


def test_premature_dependency_uses_first_failed_call():
    calls = [
        {
            "order": 1,
            "function": "core_a",
            "arguments": "{}",
            "failure": None,
        },
        {
            "order": 2,
            "function": "core_b",
            "arguments": '{"missing": 1}',
            "failure": "Value of variable is not yet known",
        },
    ]
    report = diagnose_from_classification(
        _trial(calls),
        {
            "primary_category": "premature_dependency",
            "call_failures": ["Value of variable is not yet known"],
        },
    )
    assert report["risk_level"] == "high"
    assert report["first_decisive_call_order"] == 2
    assert "before its producer" in report["narrative"]


def test_irrelevant_function_uses_earliest_distractor_call():
    calls = [
        {"order": 1, "function": "extra_a", "arguments": "{}", "failure": None},
        {"order": 2, "function": "core_a", "arguments": "{}", "failure": None},
    ]
    report = diagnose_from_classification(
        _trial(calls),
        {
            "primary_category": "irrelevant_function",
            "irrelevant_functions": ["extra_a"],
        },
    )
    assert report["first_decisive_call_order"] == 1
    assert "executable distractor" in report["narrative"]


def test_markdown_marks_rule_derived_scope():
    report = diagnose_from_classification(
        _trial([]),
        {"primary_category": "incomplete_chain", "missing_required_functions": ["x"]},
    )
    markdown = render_markdown(report)
    assert "not human-validated" in markdown
    assert "incomplete_chain" in markdown


def test_saved_trial_integration():
    run_root = (
        Path(__file__).resolve().parents[1]
        / "results"
        / "funcbench-paired-main-v2-20261030T122222Z"
    )
    if not run_root.exists():
        pytest.skip("Canonical result run is not present")
    report = diagnose_saved_trial(
        run_root,
        model_alias="deepseek-flash",
        trial_id="d3-x0-g0-r0",
    )
    assert report["primary_category"] == "correct"
    assert report["final_correct"] is True
    assert report["first_decisive_call_order"] is None

