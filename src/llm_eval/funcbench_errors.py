"""Classify FuncBenchGen failures from saved call traces without new API calls."""

from __future__ import annotations

import json
import math
import random
from collections import Counter
from pathlib import Path
from typing import Any

import networkx as nx
import pandas as pd

from .funcbench import prepare_condition
from .io_utils import atomic_write_json, write_csv


def _wilson_interval(successes: int, total: int) -> tuple[float, float]:
    if total <= 0:
        return (math.nan, math.nan)
    z = 1.959963984540054
    proportion = successes / total
    denominator = 1 + z * z / total
    centre = (proportion + z * z / (2 * total)) / denominator
    margin = (
        z
        * math.sqrt(
            proportion * (1 - proportion) / total + z * z / (4 * total * total)
        )
        / denominator
    )
    return (max(0.0, centre - margin), min(1.0, centre + margin))


def _mcnemar_exact_p(baseline_only: int, distractor_only: int) -> float:
    discordant = baseline_only + distractor_only
    if discordant == 0:
        return 1.0
    tail = sum(
        math.comb(discordant, index)
        for index in range(min(baseline_only, distractor_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def _write_metric_summaries(frame: pd.DataFrame, output: Path) -> None:
    metrics = [
        "required_recall",
        "selection_precision",
        "dependency_order_score",
        "argument_score",
        "dependency_trace_fidelity",
    ]
    present_metrics = [column for column in metrics if column in frame.columns]
    grouping_columns = ["run", "model_alias", "distractor_mode", "distractor_count"]
    if "tool_gating" in frame.columns:
        grouping_columns.append("tool_gating")
    grouped = (
        frame.groupby(grouping_columns, dropna=False)
        .agg(
            trials=("trial_id", "count"),
            correct=("correct", "sum"),
            technical_failures=(
                "outcome",
                lambda values: sum(
                    value == "technical_failure" for value in values
                ),
            ),
            **{metric: (metric, "mean") for metric in present_metrics},
        )
        .reset_index()
    )
    grouped["accuracy"] = grouped["correct"] / grouped["trials"]
    intervals = [
        _wilson_interval(int(correct), int(total))
        for correct, total in zip(grouped["correct"], grouped["trials"], strict=True)
    ]
    grouped["accuracy_ci_low"] = [interval[0] for interval in intervals]
    grouped["accuracy_ci_high"] = [interval[1] for interval in intervals]
    write_csv(grouped, output / "metric_summary.csv")

    paired = frame.copy()
    paired["graph_index"] = paired["trial_id"].astype(str).str.extract(r"-g(\d+)-")[0]
    paired = paired.dropna(subset=["graph_index"])
    paired["graph_index"] = paired["graph_index"].astype(int)
    baseline = paired.loc[paired["distractor_count"] == 0, [
        "run",
        "model_alias",
        "graph_index",
        "correct",
        "paired_core_graph",
    ]].rename(columns={"correct": "baseline_correct"})
    treatment = paired.loc[paired["distractor_count"] != 0, [
        "run",
        "model_alias",
        "distractor_mode",
        "graph_index",
        "distractor_count",
        "correct",
        "tool_gating",
    ]].rename(columns={"correct": "distractor_correct"})
    matched = baseline.merge(
        treatment,
        on=["run", "model_alias", "graph_index"],
        how="inner",
    )
    effect_rows = []
    for keys, group in matched.groupby(
        [
            "run",
            "model_alias",
            "distractor_mode",
            "distractor_count",
            "tool_gating",
            "paired_core_graph",
        ],
        dropna=False,
    ):
        (
            run_name,
            model_alias,
            distractor_mode,
            distractor_count,
            tool_gating,
            paired_core_graph,
        ) = keys
        baseline_correct = group["baseline_correct"].astype(bool)
        distractor_correct = group["distractor_correct"].astype(bool)
        both_wrong = int((~baseline_correct & ~distractor_correct).sum())
        baseline_only = int((baseline_correct & ~distractor_correct).sum())
        distractor_only = int((~baseline_correct & distractor_correct).sum())
        both_correct = int((baseline_correct & distractor_correct).sum())
        pair_differences = (
            distractor_correct.astype(int) - baseline_correct.astype(int)
        ).tolist()
        rng = random.Random(20261030)
        bootstrap_deltas = []
        for _ in range(4000):
            sample = [
                pair_differences[rng.randrange(len(pair_differences))]
                for _ in range(len(pair_differences))
            ]
            bootstrap_deltas.append(sum(sample) / len(sample))
        bootstrap_deltas.sort()
        ci_low = bootstrap_deltas[int(0.025 * (len(bootstrap_deltas) - 1))]
        ci_high = bootstrap_deltas[int(0.975 * (len(bootstrap_deltas) - 1))]
        effect_rows.append(
            {
                "run": run_name,
                "model_alias": model_alias,
                "distractor_mode": distractor_mode,
                "distractor_count": distractor_count,
                "tool_gating": bool(tool_gating),
                "paired_core_graph": bool(paired_core_graph),
                "pairs": len(group),
                "baseline_accuracy": float(baseline_correct.mean()),
                "distractor_accuracy": float(distractor_correct.mean()),
                "delta": float(distractor_correct.mean() - baseline_correct.mean()),
                "both_wrong": both_wrong,
                "baseline_only": baseline_only,
                "distractor_only": distractor_only,
                "both_correct": both_correct,
                "delta_ci_low": ci_low,
                "delta_ci_high": ci_high,
                "mcnemar_exact_p": _mcnemar_exact_p(
                    baseline_only,
                    distractor_only,
                ),
            }
        )
    write_csv(pd.DataFrame(effect_rows), output / "paired_effect_summary.csv")

    complexity_columns = [
        "run",
        "depth",
        "distractor_count",
        "distractor_mode",
        "graph_seed",
        "graph_fingerprint",
        "required_graph_fingerprint",
        "required_depth",
        "required_function_count",
        "required_edge_count",
        "max_required_fanout",
        "input_variable_count",
        "distractor_node_count",
        "num_tools",
        "tool_gating",
    ]
    present_complexity = [
        column for column in complexity_columns if column in frame.columns
    ]
    complexity = frame[present_complexity].drop_duplicates().sort_values(
        ["run", "depth", "distractor_count", "graph_seed"],
        kind="stable",
    )
    write_csv(complexity, output / "condition_complexity.csv")


def _required_nodes(condition) -> set[str]:
    tree = condition.function_tree
    target_node = getattr(tree, "desired_target_node", None)
    if not target_node:
        return set(tree.graph.nodes)
    return set(nx.ancestors(tree.graph, target_node)) | {target_node}


def classify_trial(trial: dict[str, Any], condition) -> dict[str, Any]:
    """Assign one primary failure category and diagnostic flags."""
    outcome = str(trial.get("outcome", "technical_failure"))
    calls = trial.get("calls", [])
    call_names = [str(call.get("function")) for call in calls if call.get("function")]
    required = _required_nodes(condition)
    graph = condition.function_tree.graph
    node_types = {
        name: graph.nodes[name].get("node_type", "unknown") for name in graph.nodes
    }
    unknown_functions = [name for name in call_names if name not in graph.nodes]
    irrelevant_functions = [
        name for name in call_names if name in graph.nodes and name not in required
    ]
    missing_required = sorted(required - set(call_names))
    call_failures = [str(value) for value in trial.get("failures", [])]
    unique_calls = list(dict.fromkeys(call_names))
    relevant_calls = [name for name in unique_calls if name in required]
    call_positions = {name: index for index, name in enumerate(unique_calls)}
    required_edges = [
        (parent, child)
        for parent, child in graph.edges()
        if parent in required and child in required
    ]
    required_graph = graph.subgraph(required)
    required_depth = (
        nx.dag_longest_path_length(required_graph)
        if required_graph.number_of_nodes()
        else 0
    )
    max_required_fanout = max(
        (graph.out_degree(node) for node in required),
        default=0,
    )
    order_violations = sum(
        call_positions.get(parent, -1) > call_positions.get(child, -1)
        for parent, child in required_edges
        if parent in call_positions and child in call_positions
    )
    invalid_call_count = sum(
        any(
            marker in failure.lower()
            for marker in ("incorrect", "not yet known", "func not in graph")
        )
        for failure in call_failures
    )
    selection_recall = len(relevant_calls) / max(1, len(required))
    selection_precision = len(relevant_calls) / max(1, len(unique_calls))
    order_score = (
        1.0 - order_violations / len(required_edges) if required_edges else 1.0
    )
    argument_score = (
        1.0 - invalid_call_count / max(1, len(call_names)) if call_names else 0.0
    )
    trace_fidelity = (
        selection_recall + selection_precision + order_score + argument_score
    ) / 4
    if "correct" in outcome:
        category = "correct"
    elif outcome in {"technical_failure", "budget_exhausted"}:
        category = "technical_or_budget"
    elif outcome == "shortcut":
        category = "premature_final"
    elif unknown_functions:
        category = "unknown_function"
    elif any("not yet known" in value.lower() for value in call_failures):
        category = "premature_dependency"
    elif any("incorrect" in value.lower() for value in call_failures):
        category = "wrong_argument_value"
    elif irrelevant_functions:
        category = "irrelevant_function"
    elif missing_required:
        category = "incomplete_chain"
    else:
        category = "final_answer_error"
    return {
        "primary_category": category,
        "unknown_functions": unknown_functions,
        "irrelevant_functions": irrelevant_functions,
        "missing_required_functions": missing_required,
        "call_failures": call_failures,
        "required_function_count": len(required),
        "required_edge_count": len(required_edges),
        "required_depth": required_depth,
        "max_required_fanout": max_required_fanout,
        "input_variable_count": len(condition.input_vars),
        "num_tools": len(condition.tools),
        "distractor_node_count": sum(
            data.get("node_type") == "extra"
            for _, data in graph.nodes(data=True)
        ),
        "called_function_count": len(call_names),
        "called_types": [node_types.get(name, "unknown") for name in call_names],
        "required_recall": selection_recall,
        "selection_precision": selection_precision,
        "dependency_order_score": order_score,
        "argument_score": argument_score,
        "dependency_trace_fidelity": trace_fidelity,
        "irrelevant_call_count": len(irrelevant_functions),
        "invalid_call_count": invalid_call_count,
        "exact_dependency_set": set(unique_calls) == required,
    }


def analyze_funcbench_runs(
    run_roots: list[Path | str],
    *,
    output_dir: Path | str,
) -> dict[str, Any]:
    """Classify every saved trial under the supplied completed run roots."""
    rows: list[dict[str, Any]] = []
    for raw_root in run_roots:
        run_root = Path(raw_root)
        config_path = run_root / "config.snapshot.json"
        if not config_path.exists():
            raise FileNotFoundError(f"Missing config snapshot: {config_path}")
        config = json.loads(config_path.read_text(encoding="utf-8"))
        experiment = config["experiment"]
        for model_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
            trials_path = model_dir / "trials.json"
            if not trials_path.exists():
                continue
            trials = json.loads(trials_path.read_text(encoding="utf-8"))
            for trial in trials:
                try:
                    condition = prepare_condition(
                        depth=int(trial["depth"]),
                        distractor_count=int(trial["distractor_count"]),
                        graph_seed=int(trial["graph_seed"]),
                        core_nodes=int(experiment["core_nodes"]),
                        trial_seed=0,
                        use_types=bool(experiment.get("use_types", True)),
                        use_subtypes=bool(experiment.get("use_subtypes", True)),
                        num_supertypes=int(experiment.get("num_supertypes", 5)),
                        show_variable_names=bool(
                            experiment.get("show_variable_names", False)
                        ),
                        rename_variables=bool(
                            experiment.get("rename_variables", False)
                        ),
                        num_noise_inputs=int(experiment.get("num_noise_inputs", 0)),
                        repeat_known_variable_values=bool(
                            trial.get("repeat_known_variable_values", False)
                        ),
                        paired_core_graph=bool(
                            experiment.get("paired_core_graph", False)
                        ),
                        stable_tool_order=bool(
                            experiment.get(
                                "stable_tool_order",
                                experiment.get("paired_core_graph", False),
                            )
                        ),
                        distractor_mode=str(
                            experiment.get(
                                "distractor_mode",
                                "connected_compatible",
                            )
                        ),
                    )
                    classification = classify_trial(trial, condition)
                except Exception as exc:  # noqa: BLE001 - preserve audit output
                    classification = {
                        "primary_category": "classification_error",
                        "classification_error": f"{type(exc).__name__}: {exc}",
                    }
                rows.append(
                    {
                        "run": run_root.name,
                        "model_alias": model_dir.name,
                        "trial_id": trial.get("trial_id"),
                        "depth": trial.get("depth"),
                        "distractor_count": trial.get("distractor_count"),
                        "distractor_mode": (
                            "none"
                            if int(trial.get("distractor_count", 0)) == 0
                            else trial.get("distractor_mode")
                            or experiment.get(
                                "distractor_mode",
                                "connected_compatible",
                            )
                        ),
                        "graph_seed": trial.get("graph_seed"),
                        "graph_fingerprint": trial.get("graph_fingerprint"),
                        "required_graph_fingerprint": trial.get(
                            "required_graph_fingerprint"
                        ),
                        "target_value": trial.get("target_value"),
                        "repeat_known_variable_values": trial.get(
                            "repeat_known_variable_values"
                        ),
                        "paired_core_graph": bool(
                            experiment.get("paired_core_graph", False)
                        ),
                        "tool_gating": bool(trial.get("tool_gating", False)),
                        "correct": bool(trial.get("correct")),
                        "outcome": trial.get("outcome"),
                        **classification,
                    }
                )
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    frame = pd.DataFrame(rows)
    write_csv(frame, output / "trial_failures.csv")
    _write_metric_summaries(frame, output)
    summary = (
        frame.groupby(["run", "model_alias", "primary_category"])
        .size()
        .rename("count")
        .reset_index()
    )
    write_csv(summary, output / "failure_summary.csv")
    payload = {
        "runs": [str(Path(path)) for path in run_roots],
        "trials": len(rows),
        "correct": int(frame["correct"].sum()) if not frame.empty else 0,
        "categories": dict(Counter(frame["primary_category"])),
    }
    atomic_write_json(output / "failure_summary.json", payload)
    return payload
