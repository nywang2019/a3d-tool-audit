#!/usr/bin/env python3
"""Fail when tracked files contain private submission or credential material."""

from __future__ import annotations

import re
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

PRIVATE_PATH_PATTERNS = [
    re.compile(r"(^|/)\.env($|\.)"),
    re.compile(r"(^|/)(secrets?|credentials?)(/|$)", re.IGNORECASE),
    re.compile(r"(^|/)(__pycache__|\.pytest_cache|\.ruff_cache|\.venv)(/|$)"),
    re.compile(r"(^|/)(tmp|artifacts)(/|$)"),
    re.compile(r"(^|/)results/.+/(raw|replay_raw)(/|$)"),
    re.compile(r"^paper/submission/author_inputs_needed\.md$"),
    re.compile(r"^paper/submission/cover_letter[^/]*$"),
    re.compile(r"^paper/submission/portal_text[^/]*$"),
    re.compile(r"^paper/submission/submission_(bundle|checklist|fields)[^/]*$"),
    re.compile(r"^paper/submission/upload-ready[^/]*/"),
    re.compile(r"\.(pem|key|p12|pfx|log)$", re.IGNORECASE),
]

SECRET_PATTERNS = [
    re.compile(rb"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----"),
    re.compile(rb"\bAKIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bASIA[0-9A-Z]{16}\b"),
    re.compile(rb"\bsk-[A-Za-z0-9_-]{20,}\b"),
    re.compile(rb"\bgh[pousr]_[A-Za-z0-9]{20,}\b"),
    re.compile(rb"\bxox[baprs]-[A-Za-z0-9-]{20,}\b"),
    re.compile(rb"\bAIza[0-9A-Za-z_-]{30,}\b"),
    re.compile(rb"(?i)\bAuthorization:\s*Bearer\s+[A-Za-z0-9._-]{20,}"),
    re.compile(
        rb"(?i)\b(?:OPENAI|DEEPSEEK|DASHSCOPE|MOONSHOT|ZHIPUAI|MINIMAX|QWEN|ANTHROPIC)"
        rb"[A-Z0-9_]*KEY\s*[:=]\s*[A-Za-z0-9._-]{12,}"
    ),
]

EMAIL_PATTERN = re.compile(
    rb"[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}"
)
PUBLIC_EMAIL_ALLOWLIST = {
    b"livingsailor@gmail.com",
}
RESERVED_EMAIL_DOMAINS = {
    b"example.com",
    b"example.org",
    b"example.net",
    b"example.invalid",
}

BINARY_SUFFIXES = {
    ".pdf",
    ".zip",
    ".png",
    ".jpg",
    ".jpeg",
    ".gif",
    ".webp",
    ".ico",
    ".woff",
    ".woff2",
}


def tracked_files() -> list[Path]:
    output = subprocess.check_output(
        ["git", "ls-files", "-z"], cwd=ROOT
    ).decode("utf-8")
    return [ROOT / item for item in output.split("\0") if item]


def is_placeholder_email(email: bytes) -> bool:
    domain = email.rsplit(b"@", 1)[-1].lower()
    return (
        domain in RESERVED_EMAIL_DOMAINS
        or domain.endswith(b".example")
        or domain.endswith(b".test")
        or domain.endswith(b".invalid")
    )


def main() -> int:
    files = tracked_files()
    private_paths: list[str] = []
    secret_hits: list[str] = []
    public_emails: set[str] = set()
    unexpected_emails: set[str] = set()

    for path in files:
        relative = path.relative_to(ROOT).as_posix()
        if any(pattern.search(relative) for pattern in PRIVATE_PATH_PATTERNS):
            private_paths.append(relative)
            continue
        if path.suffix.lower() in BINARY_SUFFIXES or not path.is_file():
            continue
        data = path.read_bytes()
        for pattern in SECRET_PATTERNS:
            if pattern.search(data):
                secret_hits.append(f"{relative}: {pattern.pattern.decode(errors='replace')}")
                break
        for email in EMAIL_PATTERN.findall(data):
            if email in PUBLIC_EMAIL_ALLOWLIST:
                public_emails.add(email.decode())
            elif is_placeholder_email(email):
                continue
            else:
                unexpected_emails.add(email.decode())

    print("PUBLIC RELEASE AUDIT")
    print(f"tracked_files={len(files)}")
    print(f"private_path_hits={len(private_paths)}")
    for item in private_paths:
        print(f"  PRIVATE_PATH {item}")
    print(f"secret_pattern_hits={len(secret_hits)}")
    for item in secret_hits:
        print(f"  SECRET_PATTERN {item}")
    print(f"allowlisted_public_emails={len(public_emails)}")
    for item in sorted(public_emails):
        print(f"  PUBLIC_EMAIL {item}")
    print(f"unexpected_email_hits={len(unexpected_emails)}")
    for item in sorted(unexpected_emails):
        print(f"  REVIEW_EMAIL {item}")

    failed = bool(private_paths or secret_hits or unexpected_emails)
    print("RESULT=" + ("FAIL" if failed else "PASS"))
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
