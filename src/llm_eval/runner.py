"""Configuration validation and API-only orchestration around BeyondBench."""

from __future__ import annotations

import os
import platform
import re
import subprocess
import sys
from contextlib import contextmanager
from copy import deepcopy
from datetime import UTC, datetime
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path
from typing import Any

import yaml
from beyondbench import EvaluationEngine
from beyondbench.core.base_task import BaseTask
from beyondbench.core.task_registry import TaskRegistry

from .adapters import OpenAICompatibleAdapter
from .analysis import analyse_results
from .io_utils import atomic_write_json


class ConfigError(ValueError):
    """Raised when an experiment configuration violates the protocol."""


HARD_TASK_PARAMETER_FIELDS = (
    "board_sizes",
    "num_disks_list",
    "num_variables_list",
    "sat_types_list",
    "clause_ratios_list",
    "graph_types",
    "num_vertices_list",
    "grid_sizes",
    "difficulty_levels",
    "word_lengths",
    "puzzle_types",
    "num_equations_list",
    "constraint_types",
    "matrix_counts_list",
    "dimension_patterns",
    "num_projects_list",
    "technique_levels",
)


@contextmanager
def bounded_base_task_retries(max_retries: int):
    """Apply a run-level retry bound even when a task omits the parameter."""
    original_init = BaseTask.__init__

    def bounded_init(self, *args, **kwargs):
        kwargs.setdefault("max_retries", max_retries)
        original_init(self, *args, **kwargs)

    BaseTask.__init__ = bounded_init
    try:
        yield
    finally:
        BaseTask.__init__ = original_init


def load_config(path: Path | str) -> dict[str, Any]:
    """Load a YAML configuration and reject an empty document."""
    config_path = Path(path)
    with config_path.open("r", encoding="utf-8") as handle:
        config = yaml.safe_load(handle)
    if not isinstance(config, dict):
        raise ConfigError("The YAML configuration must be a mapping")
    return config


def _enabled_models(config: dict[str, Any]) -> list[dict[str, Any]]:
    models = config.get("models", [])
    if not isinstance(models, list) or not models:
        raise ConfigError("models must be a non-empty list")
    enabled = [dict(model) for model in models if model.get("enabled", False)]
    if not enabled:
        raise ConfigError("At least one model must have enabled: true")
    aliases = [str(model.get("alias", "")).strip() for model in enabled]
    if any(not alias for alias in aliases):
        raise ConfigError("Every enabled model needs a non-empty alias")
    if len(set(aliases)) != len(aliases):
        raise ConfigError("Enabled model aliases must be unique")
    return enabled


def validate_config(
    config: dict[str, Any],
    *,
    require_keys: bool = True,
) -> dict[str, Any]:
    """Validate protocol invariants, task names, keys, and budget feasibility."""
    experiment = config.get("experiment")
    if not isinstance(experiment, dict):
        raise ConfigError("experiment must be a mapping")
    required_experiment = {
        "name",
        "suite",
        "tasks",
        "datapoints_per_task",
        "list_sizes",
        "folds",
        "seed",
        "temperature",
        "top_p",
        "max_tokens",
    }
    missing = required_experiment - set(experiment)
    if missing:
        raise ConfigError(f"experiment is missing fields: {sorted(missing)}")
    tasks = experiment["tasks"]
    if not isinstance(tasks, list) or not tasks:
        raise ConfigError("experiment.tasks must contain at least one task name")

    registry = TaskRegistry()
    known = set(registry.get_tasks_for_suite("all"))
    unknown = sorted(set(tasks) - known)
    if unknown:
        raise ConfigError(f"Unknown BeyondBench tasks: {unknown}")
    if experiment["suite"] not in {"easy", "medium", "hard", "all"}:
        raise ConfigError("experiment.suite must be easy, medium, hard, or all")

    positive_ints = {
        "datapoints_per_task": experiment["datapoints_per_task"],
        "folds": experiment["folds"],
        "max_tokens": experiment["max_tokens"],
    }
    for name, value in positive_ints.items():
        if not isinstance(value, int) or value <= 0:
            raise ConfigError(f"experiment.{name} must be a positive integer")
    if not isinstance(experiment["list_sizes"], list) or not experiment["list_sizes"]:
        raise ConfigError("experiment.list_sizes must be a non-empty list")
    if any(
        not isinstance(value, int) or value <= 0 for value in experiment["list_sizes"]
    ):
        raise ConfigError("experiment.list_sizes values must be positive integers")
    if not 0.0 <= float(experiment["temperature"]) <= 2.0:
        raise ConfigError("experiment.temperature must be between 0 and 2")
    if not 0.0 < float(experiment["top_p"]) <= 1.0:
        raise ConfigError("experiment.top_p must be in (0, 1]")

    enabled = _enabled_models(config)
    for model in enabled:
        for field in ("base_url", "model", "api_key_env"):
            if not str(model.get(field, "")).strip():
                raise ConfigError(f"Model {model.get('alias')} is missing {field}")
        if "REPLACE_WITH" in str(model["model"]):
            raise ConfigError(
                f"Model {model.get('alias')} still has a placeholder model ID"
            )
        if require_keys and not os.environ.get(str(model["api_key_env"])):
            raise ConfigError(
                f"Environment variable {model['api_key_env']} is not set for {model['alias']}"
            )

    calls_per_model = (
        len(tasks)
        * int(experiment["datapoints_per_task"])
        * len(experiment["list_sizes"])
        * int(experiment["folds"])
    )
    budget = config.get("budget", {})
    max_per_model = int(budget.get("max_requests_per_model", calls_per_model))
    max_total = int(budget.get("max_total_requests", calls_per_model * len(enabled)))
    if calls_per_model > max_per_model:
        raise ConfigError(
            f"Planned {calls_per_model} requests/model exceeds budget {max_per_model}"
        )
    if calls_per_model * len(enabled) > max_total:
        raise ConfigError(
            f"Planned {calls_per_model * len(enabled)} total requests exceeds {max_total}"
        )

    return {
        "valid": True,
        "enabled_models": [model["alias"] for model in enabled],
        "n_tasks": len(tasks),
        "planned_requests_per_model": calls_per_model,
        "planned_total_requests": calls_per_model * len(enabled),
        "seed": int(experiment["seed"]),
        "task_registry": registry.get_suite_stats(),
    }


def plan_config(config: dict[str, Any]) -> dict[str, Any]:
    """Describe budget and reproducibility settings without needing API keys."""
    experiment = config.get("experiment", {})
    models = config.get("models", [])
    enabled = [model for model in models if model.get("enabled", False)]
    task_count = len(experiment.get("tasks", []))
    samples = int(experiment.get("datapoints_per_task", 0))
    sizes = len(experiment.get("list_sizes", []))
    folds = int(experiment.get("folds", 0))
    calls_per_model = task_count * samples * sizes * folds
    return {
        "experiment": experiment.get("name"),
        "suite": experiment.get("suite"),
        "seed": experiment.get("seed"),
        "tasks": experiment.get("tasks", []),
        "datapoints_per_task": samples,
        "list_sizes": experiment.get("list_sizes", []),
        "enabled_models": [model.get("alias") for model in enabled],
        "planned_requests_per_model": calls_per_model,
        "planned_total_requests": calls_per_model * len(enabled),
        "budget": config.get("budget", {}),
        "note": (
            "Request count is the number of logical model prompts. Retries consume "
            "additional attempts from max_requests_per_model."
        ),
    }


def _redacted_snapshot(config: dict[str, Any]) -> dict[str, Any]:
    """Return a config snapshot that contains variable names, never secret values."""
    secret_key = re.compile(
        r"^(api[_-]?key|authorization|password|secret|credential)$",
        re.IGNORECASE,
    )

    def scrub(value: Any, key: str = "") -> Any:
        if key == "api_key_env":
            return value
        if key and secret_key.match(key):
            return "[REDACTED]"
        if isinstance(value, dict):
            return {str(name): scrub(item, str(name)) for name, item in value.items()}
        if isinstance(value, list):
            return [scrub(item) for item in value]
        return deepcopy(value)

    return scrub(config)


def environment_snapshot(
    vendor_dir: Path | str = "vendor/BeyondBench",
) -> dict[str, Any]:
    """Capture software and source revisions needed to reproduce a run."""
    vendor = Path(vendor_dir)
    try:
        beyondbench_version = version("beyondbench")
    except PackageNotFoundError:
        beyondbench_version = None
    commit = None
    if (vendor / ".git").exists():
        try:
            commit = subprocess.run(
                ["git", "-C", str(vendor), "rev-parse", "HEAD"],
                check=True,
                capture_output=True,
                text=True,
                timeout=10,
            ).stdout.strip()
        except (OSError, subprocess.SubprocessError):
            commit = None
    return {
        "python": sys.version,
        "platform": platform.platform(),
        "beyondbench_version": beyondbench_version,
        "beyondbench_commit": commit,
        "beyondbench_vendor_dir": str(vendor.resolve()),
    }


def run_experiment(
    config: dict[str, Any],
    *,
    results_dir: Path | str = "results",
    static_scores_path: Path | str | None = None,
    model_aliases: set[str] | None = None,
) -> dict[str, Any]:
    """Run enabled models sequentially, preserving the same task instances."""
    validation = validate_config(config, require_keys=True)
    experiment = config["experiment"]
    enabled = _enabled_models(config)
    if model_aliases:
        enabled = [model for model in enabled if model["alias"] in model_aliases]
        if not enabled:
            raise ConfigError("No enabled model matched --model")

    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    run_root = Path(results_dir) / f"{experiment['name']}-{stamp}"
    run_root.mkdir(parents=True, exist_ok=False)
    atomic_write_json(run_root / "config.snapshot.json", _redacted_snapshot(config))
    atomic_write_json(run_root / "validation.json", validation)
    atomic_write_json(run_root / "environment.json", environment_snapshot())

    completed: list[str] = []
    failures: dict[str, str] = {}
    for model_config in enabled:
        alias = str(model_config["alias"])
        model_dir = run_root / alias
        model_dir.mkdir(parents=True, exist_ok=False)
        try:
            adapter = OpenAICompatibleAdapter(
                model_id=str(model_config["model"]),
                base_url=str(model_config["base_url"]),
                api_key_env=str(model_config["api_key_env"]),
                timeout_seconds=float(model_config.get("timeout_seconds", 120)),
                max_retries=int(model_config.get("max_retries", 4)),
                temperature=float(experiment["temperature"]),
                max_tokens=int(experiment["max_tokens"]),
                extra_body=dict(model_config.get("extra_body", {})),
                max_requests=int(
                    config.get("budget", {}).get("max_requests_per_model", 0)
                )
                or None,
                raw_log_dir=model_dir / "raw",
                trust_env=bool(model_config.get("trust_env", False)),
            )
            engine = EvaluationEngine(
                model_handler=adapter,
                output_dir=str(model_dir),
                store_details=True,
                max_retries=int(model_config.get("max_retries", 4)),
            )
            evaluation_options: dict[str, Any] = {}
            for field in HARD_TASK_PARAMETER_FIELDS:
                if field in experiment:
                    evaluation_options[field] = experiment[field]
            with bounded_base_task_retries(int(model_config.get("max_retries", 4))):
                engine.run_evaluation(
                    suite=str(experiment["suite"]),
                    tasks=list(experiment["tasks"]),
                    datapoints=int(experiment["datapoints_per_task"]),
                    folds=int(experiment["folds"]),
                    list_sizes=list(experiment["list_sizes"]),
                    seed=int(experiment["seed"]),
                    temperature=float(experiment["temperature"]),
                    top_p=float(experiment["top_p"]),
                    max_tokens=int(experiment["max_tokens"]),
                    range_min=int(experiment.get("range_min", 1)),
                    range_max=int(experiment.get("range_max", 100)),
                    skip_warmup=True,
                    stream=False,
                    **evaluation_options,
                )
            if not any(model_dir.rglob("detailed_results_fold_*.json")):
                raise RuntimeError(
                    "BeyondBench completed without saving detailed instance results"
                )
            completed.append(alias)
        except Exception as exc:  # noqa: BLE001 - one model failure must not discard other runs
            failures[alias] = f"{type(exc).__name__}: {exc}"
            (model_dir / "run_failure.txt").write_text(
                failures[alias] + "\n", encoding="utf-8"
            )

    analysis = None
    if completed:
        try:
            analysis = analyse_results(
                run_root,
                static_scores_path=static_scores_path,
                output_dir=run_root / "analysis",
                seed=int(experiment["seed"]),
            )
        except Exception as exc:  # noqa: BLE001 - preserve dynamic results if plotting fails
            failures["analysis"] = f"{type(exc).__name__}: {exc}"

    manifest = {
        "run_root": str(run_root),
        "completed_models": completed,
        "failures": failures,
        "analysis_completed": analysis is not None,
    }
    atomic_write_json(run_root / "run_manifest.json", manifest)
    return manifest
