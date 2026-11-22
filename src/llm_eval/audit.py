"""Repeatability and secret-leak audits for completed runs."""

from __future__ import annotations

import ast
import hashlib
import json
import os
import re
from collections import defaultdict
from pathlib import Path
from typing import Any

from .adapters import OpenAICompatibleAdapter
from .io_utils import atomic_write_json


def _last_boxed(text: str) -> str | None:
    matches = re.findall(r"\\boxed\{([^{}]+)\}", text)
    return matches[-1].strip() if matches else None


def _normalise_text(text: str) -> str:
    return " ".join(text.split())


def canonical_answer_text(text: str) -> str | None:
    """Extract and canonicalise the final answer across common task formats."""
    if not text or not text.strip():
        return None
    boxed = _last_boxed(text)
    if boxed is not None:
        candidate = boxed
    else:
        answer_blocks = re.findall(r"<answer>\s*(.*?)\s*</answer>", text, re.DOTALL)
        if answer_blocks:
            candidate = answer_blocks[-1]
        else:
            lines = [line.strip() for line in text.splitlines() if line.strip()]
            candidate = lines[-1] if lines else text.strip()
    try:
        parsed = ast.literal_eval(candidate.strip())
    except (SyntaxError, ValueError):
        try:
            return str(float(candidate.strip()))
        except ValueError:
            return _normalise_text(candidate).casefold()
    return json.dumps(
        parsed,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        default=str,
    )


def select_replay_records(
    model_dir: Path | str,
    *,
    sample_size: int = 10,
) -> list[dict[str, Any]]:
    """Select hash-ordered records with round-robin task coverage."""
    root = Path(model_dir)
    by_task: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for path in sorted(root.rglob("detailed_results_fold_*.json")):
        records = json.loads(path.read_text(encoding="utf-8"))
        parts = path.parts
        task = (
            parts[parts.index("task_results") + 1]
            if "task_results" in parts
            else path.parent.name
        )
        for record in records:
            if record.get("generation_success") and record.get("prompt"):
                record = dict(record)
                record["task"] = task
                by_task[task].append(record)
    if not by_task:
        raise ValueError(f"No replayable records found under {root}")
    for task_records in by_task.values():
        task_records.sort(
            key=lambda item: hashlib.sha256(
                str(item["prompt"]).encode("utf-8")
            ).hexdigest()
        )
    selected: list[dict[str, Any]] = []
    tasks = sorted(by_task)
    index = 0
    while len(selected) < min(
        sample_size, sum(len(items) for items in by_task.values())
    ):
        task = tasks[index % len(tasks)]
        offset = index // len(tasks)
        if offset < len(by_task[task]):
            selected.append(by_task[task][offset])
        index += 1
    return selected


def replay_model(
    *,
    adapter: OpenAICompatibleAdapter,
    model_dir: Path | str,
    sample_size: int = 10,
    temperature: float = 0.0,
    top_p: float = 1.0,
    max_tokens: int = 4096,
) -> dict[str, Any]:
    """Call previously generated prompts again and compare stability."""
    records = select_replay_records(model_dir, sample_size=sample_size)
    output_rows: list[dict[str, Any]] = []
    for record in records:
        replay_text = adapter.generate(
            [record["prompt"]],
            temperature=temperature,
            top_p=top_p,
            max_tokens=max_tokens,
        )[0]
        original_text = str(record.get("model_response", ""))
        original_answer = canonical_answer_text(original_text)
        replay_answer = canonical_answer_text(replay_text)
        output_rows.append(
            {
                "task": record["task"],
                "prompt_sha256": hashlib.sha256(
                    record["prompt"].encode("utf-8")
                ).hexdigest(),
                "original_response": original_text,
                "replay_response": replay_text,
                "normalised_text_equal": _normalise_text(original_text)
                == _normalise_text(replay_text),
                "original_answer": original_answer,
                "replay_answer": replay_answer,
                "final_answer_equal": original_answer is not None
                and replay_answer is not None
                and original_answer == replay_answer,
            }
        )
    n = len(output_rows)
    return {
        "sample_size": n,
        "normalised_text_equal_rate": (
            sum(row["normalised_text_equal"] for row in output_rows) / n if n else None
        ),
        "final_answer_equal_rate": (
            sum(row["final_answer_equal"] for row in output_rows) / n if n else None
        ),
        "records": output_rows,
    }


def scan_secret_leaks(
    results_root: Path | str,
    *,
    api_key_envs: list[str],
) -> dict[str, Any]:
    """Check result text files without returning any secret value."""
    secrets = [os.environ[name] for name in api_key_envs if os.environ.get(name)]
    findings: list[str] = []
    checked = 0
    for path in Path(results_root).rglob("*"):
        if not path.is_file() or path.suffix.lower() not in {
            ".json",
            ".jsonl",
            ".txt",
            ".log",
            ".csv",
        }:
            continue
        checked += 1
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        if any(secret in text for secret in secrets):
            findings.append(str(path))
    return {
        "checked_files": checked,
        "secret_values_checked": len(secrets),
        "files_with_secret_material": findings,
        "passed": not findings,
    }


def write_replay_report(destination: Path | str, payload: dict[str, Any]) -> None:
    """Persist a replay report without including API credentials."""
    atomic_write_json(destination, payload)
