#!/usr/bin/env python3
"""Compare the eight-graph repeatability run with the original paired sample."""

from __future__ import annotations

import json
import math
import random
import re
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

ROOT = Path(__file__).resolve().parents[2]
ORIGINAL = (
    ROOT
    / "results/funcbench-main20-five-model-analysis-20261101/trial_failures.csv"
)
ORIGINAL_RUNS = [
    ROOT / "results/funcbench-paired-main-v2-20261030T122222Z",
    ROOT / "results/funcbench-minimax-main20-20261101T105025Z",
]
REPEAT_RUNS = [
    ROOT / "results/funcbench-paired-repeatability8-20261101T081739Z",
    ROOT / "results/funcbench-minimax-repeatability8-20261101T115312Z",
]
REPEAT = (
    ROOT
    / "results/funcbench-repeatability8-five-model-20261101/trial_failures.csv"
)
OUTPUT = ROOT / "paper/supplement"
TABLE = ROOT / "paper/tables"
MODEL_LABELS = {
    "deepseek-flash": "DeepSeek-V4.1-Flash",
    "kimi-k3.1-pro": "Kimi K3.1 Pro",
    "glm-5.3-flashx": "GLM-5.3-Flashx",
    "minimax-m3.1-flash": "MiniMax-M3.1-Flash",
    "qwen3.8-max": "Qwen3.8-Max",
}
BOOTSTRAP_SEEDS = {
    "deepseek-flash": 20261101,
    "kimi-k3.1-pro": 20261102,
    "glm-5.3-flashx": 20261103,
    "minimax-m3.1-flash": 20261105,
    "qwen3.8-max": 20261104,
}


def as_bool(series: pd.Series) -> pd.Series:
    return series.astype(str).str.lower().isin({"true", "1", "yes"})


def cohen_kappa(left: list[Any], right: list[Any]) -> float:
    if not left or len(left) != len(right):
        return math.nan
    observed = sum(a == b for a, b in zip(left, right, strict=True)) / len(left)
    left_counts = Counter(left)
    right_counts = Counter(right)
    expected = sum(
        (left_counts[label] / len(left)) * (right_counts[label] / len(right))
        for label in set(left_counts) | set(right_counts)
    )
    return (observed - expected) / (1 - expected) if expected < 1 else 1.0


def mcnemar_exact(left_only: int, right_only: int) -> float:
    discordant = left_only + right_only
    if not discordant:
        return 1.0
    tail = sum(
        math.comb(discordant, i)
        for i in range(min(left_only, right_only) + 1)
    ) / (2**discordant)
    return min(1.0, 2 * tail)


def bootstrap_delta(rows: list[tuple[bool, bool]], seed: int) -> tuple[float, float]:
    if not rows:
        return math.nan, math.nan
    rng = random.Random(seed)
    values = []
    for _ in range(2_000):
        sample = [rows[rng.randrange(len(rows))] for _ in rows]
        values.append(
            sum(right for _, right in sample) / len(sample)
            - sum(left for left, _ in sample) / len(sample)
        )
    values.sort()
    return values[50], values[1949]


def answer_signature(value: object) -> str:
    text = str(value or "")
    numbers = re.findall(r"[-+]?\d+(?:\.\d+)?", text.replace(",", ""))
    if numbers:
        number = numbers[-1]
        try:
            return f"number:{float(number):.12g}"
        except ValueError:
            pass
    return "text:" + re.sub(r"\s+", " ", text).strip().lower()


def load_trial_map(run_root: Path) -> dict[tuple[str, str], dict[str, Any]]:
    rows: dict[tuple[str, str], dict[str, Any]] = {}
    for model_dir in sorted(path for path in run_root.iterdir() if path.is_dir()):
        path = model_dir / "trials.json"
        if not path.exists():
            continue
        for row in json.loads(path.read_text(encoding="utf-8")):
            rows[(model_dir.name, str(row["trial_id"]))] = row
    return rows


def main() -> None:
    original = pd.read_csv(ORIGINAL)
    original = original[original["trial_id"].str.match(r"d3-x[05]-g[0-7]-r0")]
    repeat = pd.read_csv(REPEAT)
    for frame in (original, repeat):
        frame["correct"] = as_bool(frame["correct"])

    left = original.set_index(["model_alias", "trial_id"]).sort_index()
    right = repeat.set_index(["model_alias", "trial_id"]).sort_index()
    common = left.index.intersection(right.index)
    left = left.loc[common]
    right = right.loc[common]
    assert len(left) == len(right) == 80
    assert (left["distractor_count"] == right["distractor_count"]).all()

    paired_rows: list[list[object]] = []
    agreement_rows: list[list[object]] = []
    for model, model_label in MODEL_LABELS.items():
        model_index = [
            idx for idx in common if idx[0] == model
        ]
        left_model = left.loc[model_index]
        right_model = right.loc[model_index]
        counts: dict[int, dict[str, int]] = {}
        for count in (0, 5):
            pair_index = [
                idx
                for idx in model_index
                if int(left.loc[idx, "distractor_count"]) == count
            ]
            original_count = int(left.loc[pair_index, "correct"].sum())
            repeat_count = int(right.loc[pair_index, "correct"].sum())
            counts[count] = {
                "original": original_count,
                "repeat": repeat_count,
                "pooled": original_count + repeat_count,
            }
        original_delta = (counts[5]["original"] - counts[0]["original"]) / 8
        repeat_delta = (counts[5]["repeat"] - counts[0]["repeat"]) / 8
        pooled_delta = (counts[5]["pooled"] - counts[0]["pooled"]) / 16
        paired_rows.append(
            [
                model_label,
                counts[0]["original"],
                counts[0]["repeat"],
                counts[0]["pooled"],
                counts[5]["original"],
                counts[5]["repeat"],
                counts[5]["pooled"],
                f"{100 * original_delta:+.1f}",
                f"{100 * repeat_delta:+.1f}",
                f"{100 * pooled_delta:+.1f}",
            ]
        )

        bootstrap_input = []
        for graph_index in range(8):
            base_id = f"d3-x0-g{graph_index}-r0"
            dist_id = f"d3-x5-g{graph_index}-r0"
            bootstrap_input.append(
                (
                    bool(right.loc[(model, base_id), "correct"]),
                    bool(right.loc[(model, dist_id), "correct"]),
                )
            )
        ci_low, ci_high = bootstrap_delta(
            bootstrap_input,
            BOOTSTRAP_SEEDS[model],
        )
        repeat_only = int((~left_model["correct"] & right_model["correct"]).sum())
        original_only = int((left_model["correct"] & ~right_model["correct"]).sum())
        agreement_rows.append(
            [
                model_label,
                len(model_index),
                f"{int(left_model['correct'].sum())}/{len(model_index)}",
                f"{int(right_model['correct'].sum())}/{len(model_index)}",
                f"{int((left_model['correct'] == right_model['correct']).sum())}/{len(model_index)}",
                f"{cohen_kappa(left_model['correct'].tolist(), right_model['correct'].tolist()):.3f}",
                f"{int((left_model['primary_category'] == right_model['primary_category']).sum())}/{len(model_index)}",
                original_only,
                repeat_only,
                mcnemar_exact(original_only, repeat_only),
                f"{100 * ci_low:+.1f}, {100 * ci_high:+.1f}",
            ]
        )

    TABLE.mkdir(parents=True, exist_ok=True)
    summary = pd.DataFrame(
        paired_rows,
        columns=[
            "Model",
            "Original 0",
            "Repeat 0",
            "Pooled 0",
            "Original 5",
            "Repeat 5",
            "Pooled 5",
            "Original delta pp",
            "Repeat delta pp",
            "Pooled delta pp",
        ],
    )
    summary.to_csv(TABLE / "repeatability_conditions_full.csv", index=False)

    agreement = pd.DataFrame(
        agreement_rows,
        columns=[
            "Model",
            "Cells",
            "Original correct",
            "Repeat correct",
            "Same correctness",
            "Correctness kappa",
            "Same primary category",
            "Original-only correct",
            "Repeat-only correct",
            "McNemar p",
            "Repeat 5-0 bootstrap 95% CI pp",
        ],
    )
    agreement.to_csv(TABLE / "repeatability_agreement_full.csv", index=False)

    trial_map = {
        key: value
        for run_root in REPEAT_RUNS
        for key, value in load_trial_map(run_root).items()
    }
    original_map = {
        key: value
        for run_root in ORIGINAL_RUNS
        for key, value in load_trial_map(run_root).items()
    }
    answer_pairs = []
    for key in common:
        if key in trial_map and key in original_map:
            answer_pairs.append(
                (
                    answer_signature(original_map[key].get("final_content")),
                    answer_signature(trial_map[key].get("final_content")),
                )
            )
    same_answers = sum(a == b for a, b in answer_pairs)

    request_count = sum(
        len(list((run_root / model / "raw").glob("*.json")))
        for run_root in REPEAT_RUNS
        for model in MODEL_LABELS
        if (run_root / model / "raw").exists()
    )
    input_tokens = sum(
        int(row.get("input_tokens", 0)) for row in trial_map.values()
    )
    output_tokens = sum(
        int(row.get("output_tokens", 0)) for row in trial_map.values()
    )
    reasoning_tokens = sum(
        int(row.get("reasoning_tokens", 0)) for row in trial_map.values()
    )
    overall = {
        "run_roots": [str(path.relative_to(ROOT)) for path in REPEAT_RUNS],
        "trials": 80,
        "same_correctness": f"{int((left['correct'] == right['correct']).sum())}/80",
        "correctness_kappa": cohen_kappa(
            left["correct"].tolist(), right["correct"].tolist()
        ),
        "same_primary_category": f"{int((left['primary_category'] == right['primary_category']).sum())}/80",
        "same_final_answer_signature": f"{same_answers}/{len(answer_pairs)}",
        "technical_failures": int((right["outcome"] == "technical_failure").sum()),
        "requests": request_count,
        "input_tokens": input_tokens,
        "output_tokens": output_tokens,
        "reasoning_tokens": reasoning_tokens,
    }
    (OUTPUT / "repeatability_summary.json").write_text(
        json.dumps(overall, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )

    lines = [
        "# Eight-graph repeatability audit",
        "",
        f"- Runs: {', '.join(f'`{path}`' for path in overall['run_roots'])}",
        f"- Trials: {overall['trials']}",
        f"- Same correctness: {overall['same_correctness']}",
        f"- Correctness kappa: {overall['correctness_kappa']:.3f}",
        f"- Same primary category: {overall['same_primary_category']}",
        f"- Same final-answer signature: {overall['same_final_answer_signature']}",
        f"- Technical failures: {overall['technical_failures']}",
        f"- Model interactions: {overall['requests']}",
        f"- Input/output/reasoning tokens: {input_tokens:,} / {output_tokens:,} / {reasoning_tokens:,}",
        "",
        "## Agreement by model",
        "",
        "| Model | Original | Repeat | Same correctness | Kappa | Same category | Repeat 5-0 effect |",
        "|---|---:|---:|---:|---:|---:|---:|",
    ]
    for row in agreement_rows:
        lines.append(
            f"| {row[0]} | {row[2]} | {row[3]} | {row[4]} | {row[5]} | "
            f"{row[6]} | {row[10]} pp |"
        )
    lines.extend(
        [
            "",
            "## Condition counts",
            "",
            "| Model | Original 0 | Repeat 0 | Original 5 | Repeat 5 | Original delta | Repeat delta |",
            "|---|---:|---:|---:|---:|---:|---:|",
        ]
    )
    for row in paired_rows:
        lines.append(
            f"| {row[0]} | {row[1]}/8 | {row[2]}/8 | {row[4]}/8 | "
            f"{row[5]}/8 | {row[7]} pp | {row[8]} pp |"
        )
    lines.extend(
        [
            "",
            "The repeat run uses the same graph indices and generator seed as the",
            "original run, but is a fresh model sample. It is therefore a",
            "sampling-repeatability audit, not an independent new graph sample.",
        ]
    )
    (OUTPUT / "repeatability_summary.md").write_text(
        "\n".join(lines) + "\n", encoding="utf-8"
    )
    print(json.dumps(overall, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()

