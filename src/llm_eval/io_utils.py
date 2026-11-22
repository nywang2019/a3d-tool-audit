"""Small deterministic I/O helpers used by runners and tests."""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any

import pandas as pd


def atomic_write_json(path: Path | str, payload: Any) -> None:
    """Write JSON through a sibling temporary file, then replace atomically."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = target.with_suffix(target.suffix + ".tmp")
    temporary.write_text(
        json.dumps(payload, ensure_ascii=False, indent=2, default=str) + "\n",
        encoding="utf-8",
    )
    last_error: PermissionError | None = None
    for attempt in range(10):
        try:
            os.replace(temporary, target)
            return
        except PermissionError as exc:
            # Windows search indexing, antivirus, or editors may briefly hold a
            # target open between the temporary write and replacement.
            last_error = exc
            time.sleep(min(0.05 * (2**attempt), 0.5))
    if last_error is not None:
        raise last_error


def write_csv(frame: pd.DataFrame, path: Path | str) -> None:
    """Write a UTF-8 CSV while preserving the caller's row order."""
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(target, index=False, encoding="utf-8-sig")
