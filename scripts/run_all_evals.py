"""Consolidated evaluation report across every layer.

    python scripts/run_all_evals.py [--base http://127.0.0.1:8200]

Runs, in order:
  1. Unit + integration tests          (zero API calls)
  2. Retrieval experiments E1-E4       (zero API calls)
  3. Abstention calibration sweep      (zero API calls)
  4. Adversarial / injection + authz   (zero API calls)
  5. MCP layer checks                  (zero API calls)
  6. End-to-end demo scenarios         (live, if a server is running)
  7. Generation judge summary          (reads the last judged run)

Then writes artifacts/reports/EVALUATION_SUMMARY.md tying them together.

Only step 6 needs a running server and only steps 6-7 spend API budget, which is
why the bulk of the evidence in this project costs nothing to reproduce.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
import time
from datetime import UTC, datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
PY = str(ROOT / ".venv" / "Scripts" / "python.exe")
REPORTS = ROOT / "artifacts" / "reports"


def run(cmd: list[str], label: str, timeout: int = 1800) -> tuple[bool, str]:
    print(f"\n=== {label} ===", flush=True)
    t0 = time.perf_counter()
    try:
        r = subprocess.run(
            cmd,
            capture_output=True,
            text=True,
            timeout=timeout,
            cwd=ROOT,
            encoding="utf-8",
            errors="replace",
            env={**__import__("os").environ, "PYTHONIOENCODING": "utf-8"},
        )
        out = (r.stdout or "") + (r.stderr or "")
        ok = r.returncode == 0
    except subprocess.TimeoutExpired:
        return False, "TIMEOUT"
    tail = "\n".join(ln for ln in out.strip().splitlines() if ln.strip())[-1500:]
    print(tail[-700:], flush=True)
    print(f"--- {'OK' if ok else 'FAILED'} in {time.perf_counter() - t0:.0f}s", flush=True)
    return ok, tail


def load(name: str) -> dict:
    p = REPORTS / name
    if not p.exists():
        return {}
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        return {}


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--base", default="http://127.0.0.1:8200")
    ap.add_argument("--skip-live", action="store_true")
    args = ap.parse_args()

    results: dict[str, tuple[bool, str]] = {}
    results["tests"] = run([PY, "-m", "pytest", "-q"], "1. Unit + integration tests")
    results["retrieval"] = run([PY, "-m", "scripts.build_index"], "2. Retrieval experiments")
    results["abstention"] = run(
        [PY, "scripts/calibrate_abstention.py"], "3. Abstention calibration"
    )
    results["adversarial"] = run(
        [PY, "scripts/run_adversarial.py"], "4. Adversarial + authorization probes"
    )
    results["mcp"] = run([PY, "scripts/test_mcp.py"], "5. MCP layer")
    if not args.skip_live:
        results["e2e"] = run(
            [PY, "scripts/demo_scenarios.py", "--base", args.base],
            "6. End-to-end demo scenarios (live)",
            timeout=2400,
        )
    results["judge"] = run([PY, "scripts/calibrate_judge.py"], "7. Judge calibration")

    retrieval = load("retrieval_experiments.json")
    abst = load("abstention_calibration.json")
    adv = load("adversarial.json")
    gen = load("generation_eval.json")
    jud = load("judge_calibration.json")

    L: list[str] = []
    A = L.append
    A("# OpsPilot AI — Consolidated Evaluation Summary")
    A("")
    A(f"Generated `{datetime.now(UTC).isoformat()}`")
    A("")
    A("> Every figure below was produced by running the code in this repository.")
    A("> Anything unmeasured is labelled `Not yet measured.` (spec §38).")
    A("")

    A("## 0. Suite status")
    A("")
    A("| # | Stage | API calls | Result |")
    A("| --- | --- | --- | --- |")
    labels = {
        "tests": ("Unit + integration", "none"),
        "retrieval": ("Retrieval experiments E1-E4", "none"),
        "abstention": ("Abstention calibration", "none"),
        "adversarial": ("Adversarial + authz probes", "none"),
        "mcp": ("MCP layer", "none"),
        "e2e": ("End-to-end demo scenarios", "live"),
        "judge": ("Judge calibration", "none (replays saved)"),
    }
    for i, (key, (name, cost)) in enumerate(labels.items(), 1):
        if key not in results:
            A(f"| {i} | {name} | {cost} | skipped |")
            continue
        ok, _ = results[key]
        A(f"| {i} | {name} | {cost} | {'PASS' if ok else '**FAIL**'} |")
    A("")
    A("Five of seven stages need **no model access at all**. That is deliberate:")
    A("under an 8,000 token/minute ceiling, rigour has to live where it is free.")
    A("")

    # ---- retrieval ----
    if retrieval:
        A("## 1. Retrieval (spec §19, §20)")
        A("")
        ctx = retrieval.get("context", {})
        A(
            f"Corpus `{ctx.get('documents')}` docs / `{ctx.get('chunks')}` chunks · "
            f"embedder `{ctx.get('embedder')}` · `{ctx.get('queries_scored')}` queries scored"
        )
        A("")
        A("| Experiment | Recall@10 | Precision@10 | MRR | nDCG@10 | p95 |")
        A("| --- | ---: | ---: | ---: | ---: | ---: |")
        for e in retrieval.get("experiments", []):
            o = e["overall"]
            A(
                f"| {e['experiment_id']} {e['label']} | {o['recall']:.3f} | "
                f"{o['precision']:.3f} | {o['mrr']:.3f} | **{o['ndcg']:.3f}** | "
                f"{e['latency_ms']['p95']:.1f} ms |"
            )
        A("")
        A("### Per difficulty class (nDCG@10)")
        A("")
        classes = sorted(
            {c for e in retrieval.get("experiments", []) for c in e.get("per_difficulty", {})}
        )
        A("| Experiment | " + " | ".join(classes) + " |")
        A("| --- | " + " | ".join("---:" for _ in classes) + " |")
        for e in retrieval.get("experiments", []):
            row = [
                f"{e['per_difficulty'][c]['ndcg']:.3f}" if c in e["per_difficulty"] else "-"
                for c in classes
            ]
            A(f"| {e['experiment_id']} | " + " | ".join(row) + " |")
        A("")
        A("### Significance (paired bootstrap, 10,000 resamples)")
        A("")
        A("| A | B | Δ nDCG | 95% CI | Verdict |")
        A("| --- | --- | ---: | --- | --- |")
        for c in retrieval.get("comparisons", []):
            A(
                f"| {c['a']} | {c['b']} | {c['delta']:+.4f} | "
                f"[{c['ci_low']:+.4f}, {c['ci_high']:+.4f}] | "
                f"{'**' + c['verdict'] + '**' if c['significant'] else c['verdict']} |"
            )
        A("")

    # ---- abstention ----
    if abst:
        ch = abst.get("chosen", {})
        A("## 2. Abstention (spec §10)")
        A("")
        A("Threshold chosen from a measured sweep, not guessed. Policy: cap the")
        A("safety error first, then minimise the usability error.")
        A("")
        A("| Quantity | Value |")
        A("| --- | ---: |")
        A(f"| chosen threshold | {ch.get('threshold')} |")
        A(
            f"| hallucination-risk rate (answered with no evidence) | "
            f"{ch.get('hallucination_risk_rate', 0):.1%} |"
        )
        A(
            f"| false-abstention rate (refused with evidence) | "
            f"{ch.get('false_abstention_rate', 0):.1%} |"
        )
        A(f"| balanced accuracy | {ch.get('balanced_accuracy', 0):.3f} |")
        A("")

    # ---- security ----
    if adv:
        d = adv.get("detection", {})
        probes = adv.get("authorization_probes", [])
        passed = sum(1 for p in probes if p.get("passed"))
        A("## 3. Security (spec §15, §17, §18)")
        A("")
        A("| Layer | Result |")
        A("| --- | --- |")
        A(f"| **Authorization probes** (the strong layer) | **{passed}/{len(probes)} pass** |")
        A(f"| Injection detection recall | {d.get('recall', 0):.1%} |")
        A(f"| Injection detection precision | {d.get('precision', 0):.1%} |")
        A(f"| False positives on benign look-alikes | {d.get('false_positive_rate', 0):.1%} |")
        A("")
        A("Detection is the **weak** layer and is reported as such. The security")
        A("claim rests on the authorization probes, which hold even when detection")
        A("misses entirely.")
        A("")

    # ---- generation ----
    A("## 4. Generation quality (spec §19)")
    A("")
    if gen:
        gctx = gen.get("context", {})
        A(
            f"Generator `{gctx.get('generator')}` · judge `{gctx.get('judge')}` · "
            f"independent families: `{gctx.get('judge_independent_of_generator')}` · "
            f"sample n=`{gctx.get('sample_size')}`"
        )
        A("")
        A("| Metric | Mean | 95% CI | n | Cohen's kappa | Status |")
        A("| --- | ---: | --- | ---: | ---: | --- |")
        src = jud.get("metrics") or gen.get("metrics", [])
        for m in src:
            k = f"{m['kappa']:.2f}" if m.get("kappa") is not None else "—"
            A(
                f"| {m['metric']} | {m['mean']:.3f} | "
                f"[{m['ci_low']:.3f}, {m['ci_high']:.3f}] | {m['n']} | {k} | {m['label']} |"
            )
        A("")
        A("**How to read `UNTESTED`:** the human labels for that metric contained")
        A("only one class, so chance agreement is 1.0 and kappa is mathematically")
        A("undefined. Reporting 1.00 there would look like perfect validation while")
        A("actually meaning the judge was never challenged.")
        A("")
        if jud.get("annotator"):
            A(f"> **Annotator caveat.** {jud['annotator']}")
            A("")
    else:
        A("`Not yet measured.` Run `python scripts/run_generation_eval.py --limit 40`.")
        A("")

    # ---- what is still unmeasured ----
    A("## 5. Still unmeasured")
    A("")
    A("| Claim | Why not measured |")
    A("| --- | --- |")
    A(
        "| Neural embeddings improve paraphrase retrieval | Model files blocked by the corporate proxy |"
    )
    A("| Cross-encoder reranker quality and CPU latency | Same block |")
    A("| End-to-end latency under concurrent load | No load test run |")
    A("| Docker Compose / Kubernetes behaviour | Never executed — no Docker on this machine |")
    A("| Cache hit rate in real usage | Mechanism unit-tested only |")
    A("| Context recall | Needs reference answers beyond gold spans |")
    A("")

    out = REPORTS / "EVALUATION_SUMMARY.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L), encoding="utf-8")

    print("\n" + "=" * 74)
    failed = [k for k, (ok, _) in results.items() if not ok]
    for k, (ok, _) in results.items():
        print(f"  {'PASS' if ok else 'FAIL'}  {labels[k][0]}")
    print("=" * 74)
    print(f"summary: {out}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
