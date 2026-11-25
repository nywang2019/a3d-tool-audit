#!/usr/bin/env python3
"""Assemble an editable Springer submission without copying result datasets."""

from __future__ import annotations

import re
import shutil
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
PAPER = ROOT / "paper"
MANUSCRIPT = PAPER / "manuscript"
TABLES = PAPER / "tables"
FIGURES = PAPER / "figures"
SUBMISSION = PAPER / "submission"


def expand_inputs(source: str) -> str:
    def replace(match: re.Match[str]) -> str:
        relative = match.group(1)
        if relative == "generated_numbers.tex":
            path = MANUSCRIPT / relative
        elif relative.startswith("../tables/"):
            path = TABLES / relative.removeprefix("../tables/")
        else:
            raise ValueError(f"Unexpected input: {relative}")
        return path.read_text(encoding="utf-8").rstrip() + "\n"

    return re.sub(r"\\input\{([^}]+)\}", replace, source)


def main() -> None:
    SUBMISSION.mkdir(parents=True, exist_ok=True)

    source = (MANUSCRIPT / "main_applied_intelligence.tex").read_text(
        encoding="utf-8"
    )
    source = source.replace(
        r"\graphicspath{{../figures/}}",
        r"\graphicspath{{./}}",
    )
    source = expand_inputs(source)
    (SUBMISSION / "main.tex").write_text(source, encoding="utf-8")

    for name in ("sn-jnl.cls", "sn-basic.bst"):
        shutil.copy2(MANUSCRIPT / name, SUBMISSION / name)
    shutil.copy2(MANUSCRIPT / "references.bib", SUBMISSION / "references.bib")

    figure_names = sorted(
        set(re.findall(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}", source))
    )
    referenced = {Path(name).name for name in figure_names}
    for stale in SUBMISSION.glob("figure*.pdf"):
        if stale.name not in referenced:
            stale.unlink()
    for name in sorted(referenced):
        source_figure = FIGURES / name
        if not source_figure.exists():
            raise FileNotFoundError(source_figure)
        shutil.copy2(source_figure, SUBMISSION / name)

    manifest = SUBMISSION / "package_manifest.txt"
    files = sorted(
        str(path.relative_to(SUBMISSION)).replace("\\", "/")
        for path in SUBMISSION.rglob("*")
        if path.is_file()
    )
    manifest.write_text("\n".join(files) + "\n", encoding="utf-8")
    print(
        f"Assembled {SUBMISSION / 'main.tex'} with "
        f"{len(files)} submission-package files."
    )


if __name__ == "__main__":
    main()

