#!/usr/bin/env python3
"""Run the repeatability audit while maintaining a live HTML progress panel."""

from __future__ import annotations

import argparse
import html
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import yaml

ROOT = Path(__file__).resolve().parents[2]
MODEL_LABELS = {
    "deepseek-flash": "DeepSeek-V4.1-Flash",
    "kimi-k3.1-pro": "Kimi K3.1 Pro",
    "glm-5.3-flashx": "GLM-5.3-Flashx",
    "minimax-m3.1-flash": "MiniMax-M3.1-Flash",
    "qwen3.8-max": "Qwen3.8-Max",
}


def now_iso() -> str:
    return datetime.now(UTC).isoformat(timespec="seconds")


def find_run_root(results_dir: Path, experiment_name: str, before: set[Path]) -> Path | None:
    candidates = [
        path
        for path in results_dir.glob(f"{experiment_name}-*")
        if path.is_dir() and path not in before
    ]
    return max(candidates, key=lambda path: path.stat().st_mtime, default=None)


def load_trials(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (json.JSONDecodeError, OSError):
        return []


def collect_progress(
    run_root: Path | None,
    *,
    aliases: list[str],
    expected_per_model: int,
    status: str,
    started: float,
    return_code: int | None,
) -> dict[str, Any]:
    models: dict[str, dict[str, Any]] = {}
    recent: list[dict[str, Any]] = []
    total_completed = 0
    total_requests = 0
    total_correct = 0
    technical_failures = 0

    if run_root:
        for alias in aliases:
            model_dir = run_root / alias
            trials = load_trials(model_dir / "trials.json")
            raw_requests = (
                len(list((model_dir / "raw").glob("*.json")))
                if (model_dir / "raw").exists()
                else 0
            )
            correct = sum(bool(row.get("correct")) for row in trials)
            technical = sum(
                row.get("outcome") == "technical_failure" for row in trials
            )
            models[alias] = {
                "completed": len(trials),
                "expected": expected_per_model,
                "correct": correct,
                "technical_failures": technical,
                "requests": raw_requests,
                "zero_correct": sum(
                    bool(row.get("correct"))
                    for row in trials
                    if int(row.get("distractor_count", -1)) == 0
                ),
                "five_correct": sum(
                    bool(row.get("correct"))
                    for row in trials
                    if int(row.get("distractor_count", -1)) == 5
                ),
            }
            total_completed += len(trials)
            total_requests += raw_requests
            total_correct += correct
            technical_failures += technical
            recent.extend(
                {
                    "model": alias,
                    "trial_id": row.get("trial_id"),
                    "distractors": row.get("distractor_count"),
                    "correct": bool(row.get("correct")),
                    "outcome": row.get("outcome"),
                    "calls": row.get("num_calls"),
                }
                for row in trials[-3:]
            )

    elapsed = max(0.0, time.monotonic() - started)
    expected_total = len(aliases) * expected_per_model
    rate = total_completed / elapsed if elapsed > 0 else 0.0
    eta = (
        (expected_total - total_completed) / rate
        if rate > 0 and total_completed < expected_total
        else 0.0
    )
    return {
        "status": status,
        "return_code": return_code,
        "updated_at": now_iso(),
        "run_root": str(run_root) if run_root else None,
        "completed": total_completed,
        "expected": expected_total,
        "correct": total_correct,
        "technical_failures": technical_failures,
        "requests": total_requests,
        "elapsed_seconds": round(elapsed, 1),
        "eta_seconds": round(eta, 1),
        "models": models,
        "recent": recent[-8:],
    }


def pct(completed: int, expected: int) -> int:
    return round(100 * completed / expected) if expected else 0


def fmt_duration(seconds: float) -> str:
    seconds = max(0, round(seconds))
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    if hours:
        return f"{hours}h {minutes:02d}m"
    if minutes:
        return f"{minutes}m {seconds:02d}s"
    return f"{seconds}s"


def render_dashboard(progress: dict[str, Any]) -> str:
    completed = int(progress["completed"])
    expected = int(progress["expected"])
    progress_pct = pct(completed, expected)
    correct = int(progress["correct"])
    accuracy = 100 * correct / completed if completed else 0.0
    state = str(progress["status"])
    reload_script = (
        "<script>setTimeout(() => location.reload(), 4000);</script>"
        if state in {"launching", "running"}
        else ""
    )
    model_rows = []
    for alias, item in progress["models"].items():
        model_rows.append(
            "<tr>"
            f"<td>{html.escape(MODEL_LABELS[alias])}</td>"
            f"<td class='text-end tabular-nums'>{item['completed']}/{item['expected']}</td>"
            f"<td class='text-end tabular-nums'>{item['zero_correct']}/8</td>"
            f"<td class='text-end tabular-nums'>{item['five_correct']}/8</td>"
            f"<td class='text-end tabular-nums'>{item['requests']}</td>"
            "</tr>"
        )
    recent_rows = []
    for row in reversed(progress["recent"]):
        recent_rows.append(
            "<tr>"
            f"<td>{html.escape(MODEL_LABELS.get(row['model'], row['model']))}</td>"
            f"<td>{html.escape(str(row['trial_id']))}</td>"
            f"<td class='text-end'>D{row['distractors']}</td>"
            f"<td>{'pass' if row['correct'] else html.escape(str(row['outcome']))}</td>"
            f"<td class='text-end tabular-nums'>{row['calls']}</td>"
            "</tr>"
        )
    return f"""
<style>
#repeatability-progress {{
  color: var(--foreground, #1f2937);
  background: var(--background, #ffffff);
  font-family: ui-sans-serif, system-ui, sans-serif;
  padding: 16px;
}}
#repeatability-progress h1 {{ font-size: 20px; margin: 0 0 12px; }}
#repeatability-progress h2 {{ font-size: 16px; margin: 18px 0 8px; }}
#repeatability-progress .viz-row {{ display: flex; gap: 16px; flex-wrap: wrap; margin: 10px 0; }}
#repeatability-progress .progress {{
  background: var(--muted, #e5e7eb);
  border-radius: 999px;
  height: 10px;
  overflow: hidden;
  margin: 10px 0 16px;
}}
#repeatability-progress .progress-bar {{
  background: var(--primary, #2563eb);
  height: 100%;
  transition: width 0.4s ease;
}}
#repeatability-progress .viz-grid {{
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(180px, 1fr));
  gap: 12px;
}}
#repeatability-progress .card {{
  border: 1px solid var(--border, #d1d5db);
  border-radius: 8px;
  color: var(--card-foreground, #1f2937);
  background: var(--card, #f9fafb);
  padding: 12px;
}}
#repeatability-progress .viz-stat-value {{ font-size: 20px; font-weight: 500; margin: 4px 0; }}
#repeatability-progress .text-muted {{ color: var(--muted-foreground, #6b7280); }}
#repeatability-progress .text-small {{ font-size: 12px; }}
#repeatability-progress .text-end {{ text-align: end; }}
#repeatability-progress .tabular-nums {{ font-variant-numeric: tabular-nums; }}
#repeatability-progress table {{
  border-collapse: collapse;
  width: 100%;
}}
#repeatability-progress th,
#repeatability-progress td {{
  border-bottom: 1px solid var(--border, #d1d5db);
  padding: 6px 8px;
  text-align: start;
}}
</style>
<div id="repeatability-progress">
  <h1>Repeatability audit — 8 graphs</h1>
  <div class="viz-row" aria-live="polite">
    <span><strong>{state.upper()}</strong></span>
    <span>{completed} / {expected} trials</span>
    <span>{progress['requests']} model interactions</span>
    <span>{fmt_duration(progress['elapsed_seconds'])} elapsed</span>
    <span>{fmt_duration(progress['eta_seconds'])} ETA</span>
  </div>
  <div class="progress" role="progressbar" aria-label="Trial progress"
       aria-valuenow="{progress_pct}" aria-valuemin="0" aria-valuemax="100">
    <div class="progress-bar" style="width:{progress_pct}%"></div>
  </div>
  <div class="viz-grid">
    <div class="card viz-stat">
      <div class="text-muted">Completed</div>
      <div class="viz-stat-value">{completed}/{expected}</div>
      <div>{progress_pct}%</div>
    </div>
    <div class="card viz-stat">
      <div class="text-muted">Observed success</div>
      <div class="viz-stat-value">{accuracy:.1f}%</div>
      <div>{correct} correct</div>
    </div>
    <div class="card viz-stat">
      <div class="text-muted">Technical failures</div>
      <div class="viz-stat-value">{progress['technical_failures']}</div>
      <div>retained as failures</div>
    </div>
  </div>
  <h2>Models</h2>
  <div class="table-responsive">
    <table class="table table-sm">
      <thead><tr><th>Model</th><th class="text-end">Trials</th><th class="text-end">0 distractor</th><th class="text-end">5 distractors</th><th class="text-end">Requests</th></tr></thead>
      <tbody>{''.join(model_rows)}</tbody>
    </table>
  </div>
  <h2>Latest results</h2>
  <div class="table-responsive">
    <table class="table table-sm">
      <thead><tr><th>Model</th><th>Trial</th><th class="text-end">Condition</th><th>Outcome</th><th class="text-end">Calls</th></tr></thead>
      <tbody>{''.join(recent_rows)}</tbody>
    </table>
  </div>
  <p class="text-small text-muted">Updated {html.escape(progress['updated_at'])}. Source: atomically written live run summaries.</p>
  {reload_script}
</div>
""".strip()


def write_dashboard(path: Path, progress: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_dashboard(progress), encoding="utf-8")
    path.with_suffix(".json").write_text(
        json.dumps(progress, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--config",
        type=Path,
        default=ROOT / "configs/funcbench_repeatability8.yaml",
    )
    parser.add_argument(
        "--dashboard",
        type=Path,
        required=True,
    )
    parser.add_argument("--results-dir", type=Path, default=ROOT / "results")
    args = parser.parse_args()

    config = yaml.safe_load(args.config.read_text(encoding="utf-8"))
    experiment = config["experiment"]
    aliases = [model["alias"] for model in config["models"] if model.get("enabled")]
    expected_per_model = (
        len(experiment["graph_indices"])
        * len(experiment["distractor_counts"])
        * len(experiment.get("repeat_known_variable_values_modes", [False]))
    )
    before = {
        path
        for path in args.results_dir.glob(f"{experiment['name']}-*")
        if path.is_dir()
    }
    started = time.monotonic()
    progress = collect_progress(
        None,
        aliases=aliases,
        expected_per_model=expected_per_model,
        status="launching",
        started=started,
        return_code=None,
    )
    write_dashboard(args.dashboard, progress)

    log_path = args.dashboard.with_suffix(".run.log")
    command = [
        sys.executable,
        "-m",
        "llm_eval.cli",
        "funcbench-run",
        "--config",
        str(args.config),
        "--results-dir",
        str(args.results_dir),
    ]
    with log_path.open("w", encoding="utf-8") as log:
        process = subprocess.Popen(
            command,
            cwd=ROOT,
            stdout=log,
            stderr=subprocess.STDOUT,
            text=True,
        )
        run_root: Path | None = None
        while process.poll() is None:
            run_root = run_root or find_run_root(
                args.results_dir, experiment["name"], before
            )
            progress = collect_progress(
                run_root,
                aliases=aliases,
                expected_per_model=expected_per_model,
                status="running" if run_root else "launching",
                started=started,
                return_code=None,
            )
            write_dashboard(args.dashboard, progress)
            time.sleep(2)

        return_code = process.returncode
    run_root = run_root or find_run_root(
        args.results_dir, experiment["name"], before
    )
    progress = collect_progress(
        run_root,
        aliases=aliases,
        expected_per_model=expected_per_model,
        status="completed" if return_code == 0 else "failed",
        started=started,
        return_code=return_code,
    )
    write_dashboard(args.dashboard, progress)
    print(json.dumps(progress, indent=2, sort_keys=True))
    return int(return_code or 0)


if __name__ == "__main__":
    raise SystemExit(main())

