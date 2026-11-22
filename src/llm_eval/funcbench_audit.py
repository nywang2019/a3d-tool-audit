"""Audit strict paired FuncBenchGen runs without model calls."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from .io_utils import atomic_write_json

TRIAL_PATTERN = re.compile(
    r"^d(?P<depth>\d+)-x(?P<distractor>\d+)-g(?P<graph>\d+)-r(?P<repeat>[01])$"
)


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def audit_paired_funcbench_run(run_root: Path | str) -> dict[str, Any]:
    """Verify graph pairing and trial completeness inside one run."""
    root = Path(run_root)
    config = _load_json(root / "config.snapshot.json")
    manifest = _load_json(root / "run_manifest.json")
    errors: list[str] = []

    condition_by_id = {
        str(condition["trial_id"]): condition
        for condition in manifest.get("conditions", [])
    }
    groups: dict[tuple[int, int, int], dict[int, dict[str, Any]]] = {}
    for condition in condition_by_id.values():
        match = TRIAL_PATTERN.fullmatch(str(condition["trial_id"]))
        if match is None:
            errors.append(f"Unparseable condition id: {condition['trial_id']}")
            continue
        key = (
            int(match["depth"]),
            int(match["graph"]),
            int(match["repeat"]),
        )
        groups.setdefault(key, {})[int(match["distractor"])] = condition

    pair_rows = []
    for (depth, graph_index, repeat), conditions in sorted(groups.items()):
        if 0 not in conditions:
            errors.append(f"Missing distractor=0 condition for {depth}/{graph_index}")
            continue
        baseline = conditions[0]
        for distractor, treatment in sorted(conditions.items()):
            if distractor == 0:
                continue
            same_required = (
                baseline["required_graph_fingerprint"]
                == treatment["required_graph_fingerprint"]
            )
            same_target = baseline["target_value"] == treatment["target_value"]
            different_full = (
                baseline["graph_fingerprint"] != treatment["graph_fingerprint"]
            )
            toolkit_delta = int(treatment["num_tools"]) - int(baseline["num_tools"])
            baseline_tools = list(baseline.get("tool_names", []))
            treatment_tools = list(treatment.get("tool_names", []))
            treatment_iterator = iter(treatment_tools)
            order_preserved = all(
                any(candidate == tool for candidate in treatment_iterator)
                for tool in baseline_tools
            )
            pair_rows.append(
                {
                    "depth": depth,
                    "graph_index": graph_index,
                    "repeat": bool(repeat),
                    "distractor_count": distractor,
                    "same_required_fingerprint": same_required,
                    "same_target_value": same_target,
                    "different_full_fingerprint": different_full,
                    "toolkit_delta": toolkit_delta,
                    "core_tool_order_preserved": order_preserved,
                }
            )
            if not same_required:
                errors.append(
                    f"Required fingerprint mismatch for {depth}/{graph_index}"
                )
            if not same_target:
                errors.append(f"Target mismatch for {depth}/{graph_index}")
            if not different_full:
                errors.append(
                    f"Full graph did not change for {depth}/{graph_index}/x{distractor}"
                )
            if toolkit_delta != distractor:
                errors.append(
                    f"Toolkit delta {toolkit_delta} != {distractor} "
                    f"for {depth}/{graph_index}"
                )
            if not order_preserved:
                errors.append(
                    f"Core tool order changed for {depth}/{graph_index}"
                )

    enabled_aliases = [
        str(model["alias"])
        for model in config.get("models", [])
        if model.get("enabled")
    ]
    expected_ids = set(condition_by_id)
    model_rows = {}
    trials_by_model: dict[str, dict[str, dict[str, Any]]] = {}
    for alias in enabled_aliases:
        trials_path = root / alias / "trials.json"
        if not trials_path.exists():
            errors.append(f"Missing trial file for {alias}")
            continue
        trials = _load_json(trials_path)
        trials_by_model[alias] = {
            str(trial.get("trial_id")): trial for trial in trials
        }
        trial_ids = [str(trial.get("trial_id")) for trial in trials]
        missing = sorted(expected_ids - set(trial_ids))
        unexpected = sorted(set(trial_ids) - expected_ids)
        duplicates = sorted(
            trial_id for trial_id in set(trial_ids) if trial_ids.count(trial_id) > 1
        )
        if missing:
            errors.append(f"{alias} missing trials: {missing}")
        if unexpected:
            errors.append(f"{alias} unexpected trials: {unexpected}")
        if duplicates:
            errors.append(f"{alias} duplicate trials: {duplicates}")
        for trial in trials:
            trial_id = str(trial.get("trial_id"))
            condition = condition_by_id.get(trial_id)
            if condition is None:
                continue
            match = TRIAL_PATTERN.fullmatch(trial_id)
            if match is not None:
                expected_fields = {
                    "depth": int(match["depth"]),
                    "distractor_count": int(match["distractor"]),
                    "repeat_known_variable_values": bool(int(match["repeat"])),
                }
                for field_name, expected_value in expected_fields.items():
                    if field_name in trial and trial[field_name] != expected_value:
                        errors.append(
                            f"{field_name} mismatch in {alias}/{trial_id}"
                        )
            if (
                "graph_seed" in condition
                and trial.get("graph_seed") != condition["graph_seed"]
            ):
                errors.append(f"Graph seed mismatch in {alias}/{trial_id}")
            if trial.get("graph_fingerprint") != condition["graph_fingerprint"]:
                errors.append(f"Graph fingerprint mismatch in {alias}/{trial_id}")
            if int(trial.get("target_value", -1)) != int(condition["target_value"]):
                errors.append(f"Target mismatch in {alias}/{trial_id}")
            if (
                "target_variable" in condition
                and trial.get("target_variable") != condition["target_variable"]
            ):
                errors.append(f"Target variable mismatch in {alias}/{trial_id}")
            if "prompt" in condition and trial.get("prompt") != condition["prompt"]:
                errors.append(f"Prompt mismatch in {alias}/{trial_id}")
            stored_required = trial.get("required_graph_fingerprint")
            if stored_required and (
                stored_required != condition["required_graph_fingerprint"]
            ):
                errors.append(
                    f"Required fingerprint mismatch in {alias}/{trial_id}"
                )
        model_rows[alias] = {
            "trials": len(trials),
            "unique_trials": len(set(trial_ids)),
            "missing_trials": len(missing),
            "unexpected_trials": len(unexpected),
            "duplicate_trials": len(duplicates),
        }

    for alias, trials_by_id in trials_by_model.items():
        for (depth, graph_index, repeat), conditions in groups.items():
            if 0 not in conditions:
                continue
            baseline_id = str(conditions[0]["trial_id"])
            baseline_trial = trials_by_id.get(baseline_id)
            if baseline_trial is None:
                continue
            for distractor, treatment in conditions.items():
                if distractor == 0:
                    continue
                treatment_trial = trials_by_id.get(str(treatment["trial_id"]))
                if treatment_trial is None:
                    continue
                if baseline_trial.get("prompt") != treatment_trial.get("prompt"):
                    errors.append(
                        "Prompt mismatch across paired conditions for "
                        f"{alias}/{depth}/{graph_index}"
                    )

    result = {
        "run_root": str(root),
        "generation_protocol": manifest.get("generation_protocol", {}),
        "passed": not errors,
        "errors": errors,
        "condition_count": len(condition_by_id),
        "pair_count": len(pair_rows),
        "models": model_rows,
        "pairs": pair_rows,
    }
    atomic_write_json(root / "pair_audit.json", result)
    return result
