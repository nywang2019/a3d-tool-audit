from __future__ import annotations

import json
from pathlib import Path

from llm_eval.funcbench_audit import audit_paired_funcbench_run


def _write(path: Path, payload: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload), encoding="utf-8")


def test_audit_paired_run_validates_pairing_and_completeness(
    tmp_path: Path,
) -> None:
    _write(
        tmp_path / "config.snapshot.json",
        {
            "models": [{"alias": "model-a", "enabled": True}],
        },
    )
    baseline = {
        "trial_id": "d3-x0-g0-r0",
        "graph_fingerprint": "full-0",
        "required_graph_fingerprint": "core-0",
        "target_value": 42,
        "num_tools": 4,
    }
    treatment = {
        "trial_id": "d3-x5-g0-r0",
        "graph_fingerprint": "full-5",
        "required_graph_fingerprint": "core-0",
        "target_value": 42,
        "num_tools": 9,
    }
    _write(
        tmp_path / "run_manifest.json",
        {
            "generation_protocol": {"version": "test"},
            "conditions": [baseline, treatment],
        },
    )
    _write(
        tmp_path / "model-a" / "trials.json",
        [
            {
                "trial_id": "d3-x0-g0-r0",
                "graph_fingerprint": "full-0",
                "required_graph_fingerprint": "core-0",
                "target_value": 42,
            },
            {
                "trial_id": "d3-x5-g0-r0",
                "graph_fingerprint": "full-5",
                "required_graph_fingerprint": "core-0",
                "target_value": 42,
            },
        ],
    )

    result = audit_paired_funcbench_run(tmp_path)

    assert result["passed"]
    assert result["pair_count"] == 1
    assert result["models"]["model-a"]["trials"] == 2
    assert (tmp_path / "pair_audit.json").exists()
