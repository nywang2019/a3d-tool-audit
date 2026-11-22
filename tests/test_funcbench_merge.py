from __future__ import annotations

from pathlib import Path

import pandas as pd

from llm_eval.funcbench_merge import merge_funcbench_analyses


def _row(*, run: str, distractor: int, correct: bool) -> dict[str, object]:
    return {
        "run": run,
        "model_alias": "model-a",
        "trial_id": f"d3-x{distractor}-g0-r0",
        "depth": 3,
        "distractor_count": distractor,
        "distractor_mode": (
            "none" if distractor == 0 else "connected_compatible"
        ),
        "graph_seed": 100,
        "graph_fingerprint": f"full-{distractor}",
        "required_graph_fingerprint": "core",
        "target_value": 42,
        "correct": correct,
        "outcome": "correct" if correct else "model_error",
        "paired_core_graph": True,
        "tool_gating": False,
        "required_recall": 1.0,
        "selection_precision": 1.0 if correct else 0.5,
        "dependency_order_score": 1.0,
        "argument_score": 1.0 if correct else 0.5,
        "dependency_trace_fidelity": 1.0 if correct else 0.75,
        "num_tools": distractor,
        "input_variable_count": 1,
        "required_edge_count": 3,
        "max_required_fanout": 2,
        "required_depth": 3,
        "distractor_node_count": distractor,
    }


def test_merge_analyses_rebuilds_paper_level_paired_table(
    tmp_path: Path,
) -> None:
    baseline_dir = tmp_path / "baseline"
    treatment_dir = tmp_path / "treatment"
    baseline_dir.mkdir()
    treatment_dir.mkdir()
    pd.DataFrame(
        [_row(run="baseline", distractor=0, correct=True)]
    ).to_csv(baseline_dir / "trial_failures.csv", index=False)
    pd.DataFrame(
        [_row(run="treatment", distractor=5, correct=False)]
    ).to_csv(treatment_dir / "trial_failures.csv", index=False)

    output = tmp_path / "merged"
    payload = merge_funcbench_analyses(
        [baseline_dir, treatment_dir],
        output_dir=output,
        run_name="paper-main",
    )

    assert payload["trials"] == 2
    effects = pd.read_csv(output / "paired_effect_summary.csv")
    assert effects.loc[0, "delta"] == -1.0


def test_merge_infers_missing_graph_index_when_another_frame_has_it(
    tmp_path: Path,
) -> None:
    indexed_dir = tmp_path / "indexed"
    inferred_dir = tmp_path / "inferred"
    indexed_dir.mkdir()
    inferred_dir.mkdir()

    indexed = pd.DataFrame(
        [_row(run="indexed", distractor=0, correct=True)]
    )
    indexed["graph_index"] = 0
    indexed.to_csv(indexed_dir / "trial_failures.csv", index=False)

    treatment = _row(run="inferred", distractor=5, correct=False)
    treatment["distractor_mode"] = "disconnected"
    pd.DataFrame([treatment]).to_csv(
        inferred_dir / "trial_failures.csv",
        index=False,
    )

    output = tmp_path / "merged"
    payload = merge_funcbench_analyses(
        [indexed_dir, inferred_dir],
        output_dir=output,
        run_name="mixed-index",
    )

    assert payload["trials"] == 2
    merged = pd.read_csv(output / "trial_failures.csv")
    assert merged["graph_index"].tolist() == [0, 0]


def test_merge_can_filter_to_selected_models(tmp_path: Path) -> None:
    source = tmp_path / "source"
    source.mkdir()
    first = _row(run="source", distractor=0, correct=True)
    second = {**first, "model_alias": "model-b", "trial_id": "d3-x0-g1-r0"}
    pd.DataFrame([first, second]).to_csv(
        source / "trial_failures.csv",
        index=False,
    )

    output = tmp_path / "merged"
    payload = merge_funcbench_analyses(
        [source],
        output_dir=output,
        run_name="model-filter",
        model_aliases=["model-b"],
    )

    assert payload["models"] == ["model-b"]
    merged = pd.read_csv(output / "trial_failures.csv")
    assert set(merged["model_alias"]) == {"model-b"}


def test_merge_allows_same_cell_with_reminder_treatment(tmp_path: Path) -> None:
    source = tmp_path / "reminder"
    source.mkdir()
    baseline = _row(run="reminder", distractor=5, correct=True)
    treatment = {
        **baseline,
        "trial_id": "d3-x5-g0-r1",
        "repeat_known_variable_values": True,
        "correct": False,
    }
    baseline["repeat_known_variable_values"] = False
    pd.DataFrame([baseline, treatment]).to_csv(
        source / "trial_failures.csv",
        index=False,
    )

    output = tmp_path / "merged"
    payload = merge_funcbench_analyses(
        [source],
        output_dir=output,
        run_name="reminder-pair",
    )

    assert payload["trials"] == 2
