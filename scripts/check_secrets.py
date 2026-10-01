"""Fail CI if a committed file looks like it contains a credential.

This exists because it already happened once in this project: a live Groq API
key was placed in `.env.example`, which the application never reads and which
`.gitignore` does NOT exclude. It would have been committed. A linter would not
have caught it and neither would a test.

Deliberately narrow: high-confidence provider key shapes plus an explicit check
that `.env.example` carries no values. Broad entropy scanning produces noise,
and a noisy secret scanner gets disabled.
"""

from __future__ import annotations

import re
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

SKIP_DIRS = {
    ".git",
    ".venv",
    "node_modules",
    "__pycache__",
    ".next",
    "artifacts",
    "data",
    ".pytest_cache",
    ".ruff_cache",
    ".tmp",
}
SKIP_SUFFIX = {".pyc", ".lock", ".png", ".jpg", ".ico", ".woff", ".woff2"}
# `.env` holds real credentials BY DESIGN and is gitignored, so it is never
# committed and must not be scanned. Scanning it would make the gate fail on
# every developer machine that has actually been configured.
SKIP_FILES = {".env", ".env.local"}

PATTERNS: list[tuple[str, re.Pattern[str]]] = [
    ("Groq API key", re.compile(r"\bgsk_[A-Za-z0-9]{40,}")),
    ("OpenAI API key", re.compile(r"\bsk-[A-Za-z0-9_\-]{32,}")),
    ("Anthropic API key", re.compile(r"\bsk-ant-[A-Za-z0-9_\-]{32,}")),
    ("AWS access key id", re.compile(r"\bAKIA[0-9A-Z]{16}\b")),
    ("GitHub token", re.compile(r"\bgh[pousr]_[A-Za-z0-9]{36,}")),
    ("Google API key", re.compile(r"\bAIza[0-9A-Za-z_\-]{35}\b")),
    ("Private key block", re.compile(r"-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----")),
    ("Slack token", re.compile(r"\bxox[abprs]-[0-9A-Za-z\-]{10,}")),
]

# Values that are obviously placeholders, not secrets.
ALLOW = re.compile(r"(change-me|placeholder|example|your[-_]?key|dev-only|xxx+|\.\.\.)", re.I)


def walk():
    for p in ROOT.rglob("*"):
        if not p.is_file() or p.suffix in SKIP_SUFFIX:
            continue
        if any(part in SKIP_DIRS for part in p.parts):
            continue
        if p.name in SKIP_FILES:
            continue
        yield p


def main() -> int:
    findings: list[str] = []

    for path in walk():
        try:
            text = path.read_text(encoding="utf-8", errors="ignore")
        except OSError:
            continue
        for label, pattern in PATTERNS:
            for m in pattern.finditer(text):
                line_no = text[: m.start()].count("\n") + 1
                line = text.splitlines()[line_no - 1] if line_no <= len(text.splitlines()) else ""
                if ALLOW.search(line):
                    continue
                rel = path.relative_to(ROOT)
                findings.append(f"{rel}:{line_no}  {label}  (value redacted)")

    # .env.example is a TEMPLATE. Any assignment with a value is suspect,
    # because a real key ends up committed -- this is the exact mistake that
    # motivated this script.
    example = ROOT / ".env.example"
    if example.exists():
        for i, line in enumerate(example.read_text(encoding="utf-8").splitlines(), 1):
            line = line.strip()
            if not line or line.startswith("#") or "=" not in line:
                continue
            key, _, value = line.partition("=")
            value = value.strip()
            if not value or ALLOW.search(value):
                continue
            # Numeric settings are configuration, not credentials --
            # OPSPILOT_DAILY_TOKEN_BUDGET=200000 contains "TOKEN" but is a limit.
            if re.fullmatch(r"[0-9]+(\.[0-9]+)?", value):
                continue
            # Sizing/limit knobs are configuration even when non-numeric.
            if re.search(r"(BUDGET|LIMIT|TTL|TIMEOUT|COUNT|SIZE|MAX|MIN)", key, re.I):
                continue
            # Non-secret defaults are fine; flag only credential-shaped names.
            if re.search(r"(KEY|TOKEN|SECRET|PASSWORD|CREDENTIAL)", key, re.I):
                findings.append(
                    f".env.example:{i}  {key.strip()} has a value; templates must be empty"
                )

    if findings:
        print("POSSIBLE SECRETS COMMITTED:\n")
        for f in sorted(set(findings)):
            print(f"  {f}")
        print(
            "\nRemove the value, rotate the credential, and put real values in .env "
            "(which is gitignored)."
        )
        return 1

    print("No committed secrets detected.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
