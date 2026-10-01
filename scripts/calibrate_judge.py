"""Recompute judge calibration from saved cases. Costs zero API quota."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opspilot.config import REPO_ROOT
from opspilot.eval.generation import calibrate
from opspilot.eval.judge import JudgeScore, summarise

cases = json.loads((REPO_ROOT / "evaluation" / "judged_cases.json").read_text(encoding="utf-8"))
human = json.loads((REPO_ROOT / "evaluation" / "human_labels.json").read_text(encoding="utf-8"))

scores: dict[str, list] = {}
for row in cases:
    for metric, val in (row.get("judge") or {}).items():
        if val is None:
            continue
        scores.setdefault(metric, []).append(JudgeScore(metric, row["query_id"], float(val)))

kappas = calibrate(scores, human)
labelled = sum(1 for k in human if not k.startswith("_"))
print(f"cases={len(cases)}  human-labelled={labelled}\n")
summaries = [summarise(m, s, *kappas.get(m, (None, 0, False))) for m, s in scores.items()]
for s in summaries:
    k = f"{s.kappa:.3f}" if s.kappa is not None else "undefined"
    print(
        f"  {s.metric:<20} mean={s.mean:.3f} [{s.ci_low:.3f},{s.ci_high:.3f}] "
        f"n={s.n:<3} kappa={k:<9} {s.label}"
    )

out = REPO_ROOT / "artifacts" / "reports" / "judge_calibration.json"
out.write_text(
    json.dumps(
        {
            "human_labels": labelled,
            "annotator": human.get("_annotator"),
            "metrics": [s.to_json() for s in summaries],
        },
        indent=2,
    ),
    encoding="utf-8",
)
print(f"\nwritten: {out}")
