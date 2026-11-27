#!/usr/bin/env python3
"""Audit every reachable Git commit for private paths and credential patterns."""

from __future__ import annotations

import subprocess
import sys

from check_public_release import PRIVATE_PATH_PATTERNS, ROOT, SECRET_PATTERNS


def git(*args: str) -> str:
    return subprocess.check_output(["git", *args], cwd=ROOT, text=True)


def main() -> int:
    historical_names = sorted(
        set(git("log", "--all", "--name-only", "--pretty=format:").splitlines())
    )
    private_paths = [
        path
        for path in historical_names
        if path and any(pattern.search(path) for pattern in PRIVATE_PATH_PATTERNS)
    ]

    commits = git("rev-list", "--all").splitlines()
    secret_hits: list[str] = []
    for commit in commits:
        for pattern in SECRET_PATTERNS:
            result = subprocess.run(
                [
                    "git",
                    "grep",
                    "-I",
                    "-n",
                    "-P",
                    "-e",
                    pattern.pattern.decode(),
                    commit,
                ],
                cwd=ROOT,
                capture_output=True,
                text=True,
            )
            if result.returncode == 0:
                secret_hits.append(result.stdout.strip())
            elif result.returncode not in (1,):
                raise RuntimeError(result.stderr.strip())

    print("PUBLIC HISTORY AUDIT")
    print(f"commits={len(commits)}")
    print(f"historical_private_path_hits={len(private_paths)}")
    for item in private_paths:
        print(f"  PRIVATE_PATH {item}")
    print(f"historical_secret_hits={len(secret_hits)}")
    for item in secret_hits:
        print(f"  SECRET_PATTERN {item}")

    failed = bool(private_paths or secret_hits)
    print("RESULT=" + ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
