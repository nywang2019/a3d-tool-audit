from __future__ import annotations

import os
from pathlib import Path

import pytest

from llm_eval.io_utils import atomic_write_json


def test_atomic_write_json_retries_transient_permission_error(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    target = tmp_path / "result.json"
    real_replace = os.replace
    attempts = 0

    def flaky_replace(source: Path, destination: Path) -> None:
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise PermissionError("temporarily locked")
        real_replace(source, destination)

    monkeypatch.setattr("llm_eval.io_utils.os.replace", flaky_replace)
    atomic_write_json(target, {"ok": True})

    assert attempts == 2
    assert target.read_text(encoding="utf-8").strip() == '{\n  "ok": true\n}'
