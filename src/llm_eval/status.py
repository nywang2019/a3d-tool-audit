"""Compact project-state loading for low-context continuation."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def load_project_manifest(root: Path | str = PROJECT_ROOT) -> dict[str, Any]:
    """Read the machine-readable project manifest."""
    path = Path(root) / "project_manifest.json"
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise TypeError("project_manifest.json must contain an object")
    return payload


def compact_status(manifest: dict[str, Any]) -> dict[str, Any]:
    """Return only fields needed to resume work."""
    return {
        "updated_at": manifest.get("updated_at"),
        "status": manifest.get("status"),
        "research_question": manifest.get("research_question"),
        "primary_framework": manifest.get("primary_framework"),
        "models": [model.get("alias") for model in manifest.get("models", [])],
        "network_policy": manifest.get("network_policy"),
        "confirmed_findings": manifest.get("confirmed_findings", []),
        "pending_decisions": manifest.get("pending_decisions", []),
        "context_state": manifest.get("context_state"),
    }
