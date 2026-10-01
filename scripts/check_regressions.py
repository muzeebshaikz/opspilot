"""Fail CI if measured retrieval or security quality regresses.

Tests prove the code runs. This proves it still *works well*, which is the thing
that actually degrades silently in a RAG system: a chunking tweak, a tokenisation
change or a reranker weight can leave every test green while quality drops.

Thresholds are set slightly below the CURRENTLY MEASURED values, so they catch
regression without failing on noise. They are floors to be raised as the system
improves -- not targets to be lowered when something breaks.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
REPORTS = ROOT / "artifacts" / "reports"

# (report, json path, floor, note) -- floors sit ~0.02-0.03 under measured.
THRESHOLDS: list[tuple[str, list, float, str]] = [
    (
        "retrieval_experiments.json",
        ["experiments", 3, "overall", "ndcg"],
        0.48,
        "E4 hybrid+reranker nDCG@10 (measured 0.511)",
    ),
    (
        "retrieval_experiments.json",
        ["experiments", 3, "overall", "recall"],
        0.79,
        "E4 recall@10 (measured 0.829)",
    ),
    (
        "retrieval_experiments.json",
        ["experiments", 1, "overall", "ndcg"],
        0.38,
        "E2 BM25 nDCG@10 (measured 0.406)",
    ),
    (
        "abstention_calibration.json",
        ["chosen", "balanced_accuracy"],
        0.72,
        "abstention balanced accuracy (measured 0.754)",
    ),
    (
        "adversarial.json",
        ["detection", "recall"],
        0.70,
        "injection detection recall (measured 0.781 regex-only)",
    ),
]

# Metrics where a HIGHER number is worse. These are the safety ceilings.
CEILINGS: list[tuple[str, list, float, str]] = [
    (
        "abstention_policy.json",
        ["configured", "confident_wrong"],
        0.15,
        "confident-wrong rate (answered with no evidence)",
    ),
    (
        "abstention_policy.json",
        ["configured", "hard_refusal"],
        0.25,
        "hard refusal on answerable questions",
    ),
]

# These must be exactly right, not merely above a floor.
EXACT: list[tuple[str, list, object, str]] = [
    ("adversarial.json", ["authorization_probes"], None, "every authorization probe must pass"),
]


def dig(obj, path: list):
    for key in path:
        obj = obj[key]
    return obj


def main() -> int:
    failures: list[str] = []
    checked = 0

    for report, path, floor, note in THRESHOLDS:
        p = REPORTS / report
        if not p.exists():
            failures.append(f"MISSING REPORT {report} -- run the evaluation stage first")
            continue
        try:
            value = float(dig(json.loads(p.read_text(encoding="utf-8")), path))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            failures.append(f"{report}:{'.'.join(map(str, path))} unreadable ({exc})")
            continue
        checked += 1
        status = "ok  " if value >= floor else "FAIL"
        print(f"  {status} {note}: {value:.3f} >= {floor:.3f}")
        if value < floor:
            failures.append(f"{note}: {value:.3f} below floor {floor:.3f}")

    for report, path, ceiling, note in CEILINGS:
        p = REPORTS / report
        if not p.exists():
            failures.append(f"MISSING REPORT {report}")
            continue
        try:
            value = float(dig(json.loads(p.read_text(encoding="utf-8")), path))
        except (KeyError, IndexError, TypeError, ValueError) as exc:
            failures.append(f"{report}:{'.'.join(map(str, path))} unreadable ({exc})")
            continue
        checked += 1
        ok = value <= ceiling
        print(f"  {'ok  ' if ok else 'FAIL'} {note}: {value:.3f} <= {ceiling:.3f}")
        if not ok:
            failures.append(f"{note}: {value:.3f} exceeds ceiling {ceiling:.3f}")

    for report, path, _, note in EXACT:
        p = REPORTS / report
        if not p.exists():
            failures.append(f"MISSING REPORT {report}")
            continue
        probes = dig(json.loads(p.read_text(encoding="utf-8")), path)
        failed = [x["probe"] for x in probes if not x.get("passed")]
        checked += 1
        print(
            f"  {'ok  ' if not failed else 'FAIL'} {note}: "
            f"{len(probes) - len(failed)}/{len(probes)} passed"
        )
        if failed:
            failures.append(f"authorization probes failed: {failed}")

    print()
    if failures:
        print(f"REGRESSION: {len(failures)} check(s) failed")
        for f in failures:
            print(f"  - {f}")
        return 1
    print(f"No regressions ({checked} checks passed).")
    return 0


if __name__ == "__main__":
    sys.exit(main())
