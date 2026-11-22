"""Extract PDF page text for local source audits."""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import fitz


def main() -> None:
    sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser()
    parser.add_argument("pdf", type=Path)
    parser.add_argument("--term", action="append", default=[])
    parser.add_argument("--context", type=int, default=1200)
    args = parser.parse_args()
    document = fitz.open(args.pdf)
    for page_number, page in enumerate(document, start=1):
        text = page.get_text()
        if not args.term:
            print(f"===== PAGE {page_number} =====\n{text}")
            continue
        matches = [
            match
            for term in args.term
            for match in re.finditer(re.escape(term), text, flags=re.IGNORECASE)
        ]
        if matches:
            print(f"===== PAGE {page_number} =====")
            for match in matches:
                start = max(0, match.start() - args.context)
                end = min(len(text), match.end() + args.context)
                print(text[start:end])
                print("---")


if __name__ == "__main__":
    main()
