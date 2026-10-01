"""Run the adversarial / prompt-injection evaluation."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opspilot.config import REPO_ROOT, get_settings
from opspilot.eval.adversarial import (
    build_cases,
    run_authorization_probes,
    run_detection,
    run_sanitisation_checks,
    write_report,
)

s = get_settings()
cases = build_cases()
use_neural = "--neural" in sys.argv
detector = None
if use_neural:
    from opspilot.llm.providers import build_provider
    from opspilot.security import InjectionDetector

    detector = InjectionDetector(provider=build_provider(s), enable_neural=True)
detection = run_detection(cases, detector=detector, pace_s=2.2 if use_neural else 0.0)
probes = run_authorization_probes()
sanit = run_sanitisation_checks()
path = write_report(
    detection,
    probes,
    sanit,
    REPO_ROOT / "artifacts" / "reports",
    {
        "cases": len(cases),
        "seed": s.seed,
        "detector": "regex+prompt-guard-2-86m" if use_neural else "regex-only",
        "neural_calls": getattr(detector, "neural_calls", 0),
        "neural_failures": getattr(detector, "neural_failures", 0),
    },
)
print(
    f"cases={len(cases)}  recall={detection.recall:.1%}  "
    f"precision={detection.precision:.1%}  fpr={detection.false_positive_rate:.1%}"
)
passed = sum(1 for p in probes if p["passed"])
print(f"authorization probes: {passed}/{len(probes)} passed")
for p in probes:
    if not p["passed"]:
        print(f"  FAILED: {p['probe']} -> {p['actual']}")
print(f"report: {path}")
