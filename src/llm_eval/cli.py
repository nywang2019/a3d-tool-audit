"""Command-line entry points for planning, running, and analysing experiments."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import click
from beyondbench.core.task_registry import TaskRegistry
from dotenv import load_dotenv

from .adapters import OpenAICompatibleAdapter
from .affordance_diagnosis import (
    DiagnosisError,
    diagnose_saved_trial,
    render_markdown,
)
from .analysis import analyse_results
from .audit import replay_model, scan_secret_leaks, write_replay_report
from .funcbench import plan_funcbench_calibration, run_funcbench_calibration
from .funcbench_audit import audit_paired_funcbench_run
from .funcbench_dose import analyze_funcbench_distractor_dose
from .funcbench_errors import analyze_funcbench_runs
from .funcbench_heterogeneity import analyze_distractor_heterogeneity
from .funcbench_intervention import compare_funcbench_intervention
from .funcbench_merge import merge_funcbench_analyses
from .funcbench_plots import (
    plot_control_comparison,
    plot_funcbench_summary,
    plot_intervention_effect,
)
from .runner import (
    ConfigError,
    load_config,
    plan_config,
    run_experiment,
    validate_config,
)
from .status import compact_status, load_project_manifest


@click.group()
def cli() -> None:
    """API-only dynamic reasoning evaluation toolkit."""
    load_dotenv()


@cli.command("status")
@click.option("--full", is_flag=True, help="Print the full project manifest.")
def status(full: bool) -> None:
    """Print compact project state for low-context continuation."""
    manifest = load_project_manifest()
    payload = manifest if full else compact_status(manifest)
    click.echo(json.dumps(payload, ensure_ascii=True, indent=2))


@cli.command("list-tasks")
@click.option(
    "--suite", default="all", type=click.Choice(["easy", "medium", "hard", "all"])
)
def list_tasks(suite: str) -> None:
    """List registered BeyondBench tasks."""
    registry = TaskRegistry()
    click.echo(
        json.dumps(registry.get_available_tasks(suite), ensure_ascii=False, indent=2)
    )


@cli.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
def plan(config_path: Path) -> None:
    """Show call plan without requiring API keys or making requests."""
    click.echo(
        json.dumps(plan_config(load_config(config_path)), ensure_ascii=False, indent=2)
    )


@cli.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option("--require-keys/--no-require-keys", default=True)
def validate(config_path: Path, require_keys: bool) -> None:
    """Validate task names, protocol settings, keys, and budgets."""
    config = load_config(config_path)
    try:
        payload = validate_config(config, require_keys=require_keys)
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@cli.command()
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option("--results-dir", default="results", type=click.Path(path_type=Path))
@click.option("--static-scores", type=click.Path(exists=True, path_type=Path))
@click.option("--model", "model_aliases", multiple=True)
def run(
    config_path: Path,
    results_dir: Path,
    static_scores: Path | None,
    model_aliases: tuple[str, ...],
) -> None:
    """Run enabled models and write raw, detailed, and aggregate results."""
    load_dotenv()
    try:
        payload = run_experiment(
            load_config(config_path),
            results_dir=results_dir,
            static_scores_path=static_scores,
            model_aliases=set(model_aliases) if model_aliases else None,
        )
    except ConfigError as exc:
        raise click.ClickException(str(exc)) from exc
    click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    if payload["failures"]:
        raise click.exceptions.Exit(1)


@cli.command()
@click.option(
    "--results",
    "results_root",
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option("--static-scores", type=click.Path(exists=True, path_type=Path))
@click.option("--output", "output_dir", type=click.Path(path_type=Path))
@click.option("--bootstrap", "n_bootstrap", default=2_000, type=click.IntRange(min=200))
@click.option("--seed", default=20261029, type=int)
def analyze(
    results_root: Path,
    static_scores: Path | None,
    output_dir: Path | None,
    n_bootstrap: int,
    seed: int,
) -> None:
    """Aggregate detailed outputs and test rank divergence."""
    payload = analyse_results(
        results_root,
        static_scores_path=static_scores,
        output_dir=output_dir,
        n_bootstrap=n_bootstrap,
        seed=seed,
    )
    click.echo(json.dumps(payload, ensure_ascii=False, indent=2, default=str))


@cli.command("funcbench-plan")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
def funcbench_plan(config_path: Path) -> None:
    """Estimate the FuncBenchGen calibration budget without API calls."""
    click.echo(
        json.dumps(
            plan_funcbench_calibration(load_config(config_path)),
            ensure_ascii=False,
            indent=2,
        )
    )


@cli.command("funcbench-run")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option("--results-dir", default="results", type=click.Path(path_type=Path))
@click.option("--model", "model_aliases", multiple=True)
@click.option(
    "--resume-from",
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
def funcbench_run(
    config_path: Path,
    results_dir: Path,
    model_aliases: tuple[str, ...],
    resume_from: Path | None,
) -> None:
    """Run a bounded low-cost FuncBenchGen calibration."""
    payload = run_funcbench_calibration(
        load_config(config_path),
        results_dir=results_dir,
        model_aliases=set(model_aliases) if model_aliases else None,
        resume_from=resume_from,
    )
    click.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@cli.command("funcbench-errors")
@click.option(
    "--run",
    "run_roots",
    multiple=True,
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option("--output", "output_dir", required=True, type=click.Path(path_type=Path))
def funcbench_errors(run_roots: tuple[Path, ...], output_dir: Path) -> None:
    """Classify saved FuncBenchGen failures without new API calls."""
    payload = analyze_funcbench_runs(list(run_roots), output_dir=output_dir)
    click.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@cli.command("affordance-diagnose")
@click.option(
    "--run",
    "run_root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--model", "model_alias", required=True)
@click.option("--trial-id", required=True)
@click.option(
    "--format",
    "output_format",
    type=click.Choice(["json", "markdown"]),
    default="json",
    show_default=True,
)
@click.option("--output", "output_path", type=click.Path(path_type=Path))
def affordance_diagnose(
    run_root: Path,
    model_alias: str,
    trial_id: str,
    output_format: str,
    output_path: Path | None,
) -> None:
    """Diagnose one saved tool-use trajectory without new API calls."""
    try:
        report = diagnose_saved_trial(
            run_root,
            model_alias=model_alias,
            trial_id=trial_id,
        )
    except DiagnosisError as exc:
        raise click.ClickException(str(exc)) from exc
    rendered = (
        render_markdown(report)
        if output_format == "markdown"
        else json.dumps(report, ensure_ascii=False, indent=2) + "\n"
    )
    if output_path:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(rendered, encoding="utf-8")
        click.echo(str(output_path))
    else:
        click.echo(rendered)


@cli.command("funcbench-audit")
@click.option(
    "--run",
    "run_root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
def funcbench_audit(run_root: Path) -> None:
    """Verify strict graph pairing and trial completeness without API calls."""
    payload = audit_paired_funcbench_run(run_root)
    click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["passed"]:
        raise click.exceptions.Exit(1)


@cli.command("funcbench-plot")
@click.option(
    "--analysis",
    "analysis_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--output", "output_dir", type=click.Path(path_type=Path))
def funcbench_plot(analysis_dir: Path, output_dir: Path | None) -> None:
    """Create static accuracy and paired-effect figures from saved summaries."""
    payload = plot_funcbench_summary(analysis_dir, output_dir=output_dir)
    click.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@cli.command("funcbench-control-plot")
@click.option(
    "--analysis",
    "analysis_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--output", "output_dir", type=click.Path(path_type=Path))
def funcbench_control_plot(
    analysis_dir: Path,
    output_dir: Path | None,
) -> None:
    """Plot disconnected, incompatible and compatible distractor controls."""
    payload = plot_control_comparison(analysis_dir, output_dir=output_dir)
    click.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@cli.command("funcbench-intervention-plot")
@click.option(
    "--analysis",
    "analysis_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--output", "output_dir", type=click.Path(path_type=Path))
def funcbench_intervention_plot(
    analysis_dir: Path,
    output_dir: Path | None,
) -> None:
    """Plot paired success change for a tool-use intervention."""
    payload = plot_intervention_effect(analysis_dir, output_dir=output_dir)
    click.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@cli.command("funcbench-dose")
@click.option(
    "--baseline-analysis",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option(
    "--dose-analysis",
    "dose_analyses",
    multiple=True,
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--output", "output_dir", required=True, type=click.Path(path_type=Path))
def funcbench_dose(
    baseline_analysis: Path,
    dose_analyses: tuple[Path, ...],
    output_dir: Path,
) -> None:
    """Merge a separate distractor-dose run with the strict baseline."""
    payload = analyze_funcbench_distractor_dose(
        baseline_analysis,
        list(dose_analyses),
        output_dir=output_dir,
    )
    click.echo(json.dumps(payload, ensure_ascii=False, indent=2))
    if not payload["passed"]:
        raise click.exceptions.Exit(1)


@cli.command("funcbench-merge")
@click.option(
    "--analysis",
    "analysis_dirs",
    multiple=True,
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--output", "output_dir", required=True, type=click.Path(path_type=Path))
@click.option("--name", "run_name", required=True)
@click.option("--graph-index", "graph_indices", multiple=True, type=int)
@click.option("--model", "model_aliases", multiple=True)
def funcbench_merge(
    analysis_dirs: tuple[Path, ...],
    output_dir: Path,
    run_name: str,
    graph_indices: tuple[int, ...],
    model_aliases: tuple[str, ...],
) -> None:
    """Merge compatible analyses into one paper-level trial table."""
    payload = merge_funcbench_analyses(
        list(analysis_dirs),
        output_dir=output_dir,
        run_name=run_name,
        graph_indices=list(graph_indices) if graph_indices else None,
        model_aliases=list(model_aliases) if model_aliases else None,
    )
    click.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@cli.command("funcbench-intervention")
@click.option(
    "--analysis",
    "analysis_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--output", "output_dir", required=True, type=click.Path(path_type=Path))
@click.option("--distractor-count", default=5, type=click.IntRange(min=1))
@click.option("--distractor-mode", default="connected_compatible")
def funcbench_intervention(
    analysis_dir: Path,
    output_dir: Path,
    distractor_count: int,
    distractor_mode: str,
) -> None:
    """Compare ungated and prerequisite-gated trials on the same graphs."""
    payload = compare_funcbench_intervention(
        analysis_dir,
        output_dir=output_dir,
        distractor_count=distractor_count,
        distractor_mode=distractor_mode,
    )
    click.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@cli.command("funcbench-heterogeneity")
@click.option(
    "--analysis",
    "analysis_dir",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--output", "output_dir", required=True, type=click.Path(path_type=Path))
@click.option("--distractor-count", default=5, type=click.IntRange(min=1))
@click.option("--distractor-mode", default="connected_compatible")
def funcbench_heterogeneity(
    analysis_dir: Path,
    output_dir: Path,
    distractor_count: int,
    distractor_mode: str,
) -> None:
    """Cluster-bootstrap model effect differences and permutation omnibus test."""
    payload = analyze_distractor_heterogeneity(
        analysis_dir,
        output_dir=output_dir,
        distractor_count=distractor_count,
        distractor_mode=distractor_mode,
    )
    click.echo(json.dumps(payload, ensure_ascii=False, indent=2))


@cli.command("secret-scan")
@click.option(
    "--run",
    "run_root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
def secret_scan(run_root: Path) -> None:
    """Scan a completed run for configured API key values."""
    config = json.loads(
        (run_root / "config.snapshot.json").read_text(encoding="utf-8")
    )
    key_envs = [
        str(model["api_key_env"])
        for model in config.get("models", [])
        if model.get("enabled")
    ]
    payload = scan_secret_leaks(run_root, api_key_envs=key_envs)
    click.echo(json.dumps(payload, ensure_ascii=True, indent=2))
    if not payload["passed"]:
        raise click.exceptions.Exit(1)


@cli.command("audit-replay")
@click.option(
    "--config",
    "config_path",
    required=True,
    type=click.Path(exists=True, path_type=Path),
)
@click.option(
    "--run",
    "run_root",
    required=True,
    type=click.Path(exists=True, file_okay=False, path_type=Path),
)
@click.option("--model", "model_aliases", multiple=True)
@click.option("--sample-size", default=10, type=click.IntRange(min=1, max=100))
def audit_replay(
    config_path: Path,
    run_root: Path,
    model_aliases: tuple[str, ...],
    sample_size: int,
) -> None:
    """Replay saved prompts to audit temperature-zero output stability."""
    load_dotenv()
    config = load_config(config_path)
    experiment = config["experiment"]
    selected = set(model_aliases)
    models = [
        model
        for model in config.get("models", [])
        if model.get("enabled") and (not selected or model["alias"] in selected)
    ]
    if not models:
        raise click.ClickException("No matching enabled models in configuration")

    report: dict[str, Any] = {"run_root": str(run_root), "models": {}}
    for model_config in models:
        alias = str(model_config["alias"])
        adapter = OpenAICompatibleAdapter(
            model_id=str(model_config["model"]),
            base_url=str(model_config["base_url"]),
            api_key_env=str(model_config["api_key_env"]),
            timeout_seconds=float(model_config.get("timeout_seconds", 120)),
            max_retries=int(model_config.get("max_retries", 4)),
            temperature=float(experiment["temperature"]),
            max_tokens=int(experiment["max_tokens"]),
            extra_body=dict(model_config.get("extra_body", {})),
            max_requests=sample_size
            + int(model_config.get("max_retries", 4)) * sample_size,
            raw_log_dir=run_root / alias / "replay_raw",
            trust_env=bool(model_config.get("trust_env", True)),
        )
        report["models"][alias] = replay_model(
            adapter=adapter,
            model_dir=run_root / alias,
            sample_size=sample_size,
            temperature=float(experiment["temperature"]),
            top_p=float(experiment["top_p"]),
            max_tokens=int(experiment["max_tokens"]),
        )
        report["models"][alias]["adapter_statistics"] = adapter.get_statistics()

    report["secret_scan"] = scan_secret_leaks(
        run_root,
        api_key_envs=[str(model["api_key_env"]) for model in models],
    )
    destination = run_root / "replay_audit.json"
    write_replay_report(destination, report)
    summary = {
        "report": str(destination),
        "models": {
            alias: {
                key: model_report.get(key)
                for key in (
                    "sample_size",
                    "normalised_text_equal_rate",
                    "final_answer_equal_rate",
                )
            }
            for alias, model_report in report["models"].items()
        },
        "secret_scan": report["secret_scan"],
    }
    click.echo(json.dumps(summary, ensure_ascii=True, indent=2))


def main() -> None:
    cli()


if __name__ == "__main__":
    main()
