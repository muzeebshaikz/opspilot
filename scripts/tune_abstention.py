"""Measure the three-tier evidence policy against the two-tier baseline.

Baseline (single threshold 1.5): 37.0% of ANSWERABLE questions refused, to hold
confident-wrong answers at 12.1%.

The three-tier policy keeps the SUFFICIENT bar exactly where it was -- so the
confident-wrong rate cannot move -- and converts the marginal band into a hedged
"related material" answer instead of a flat refusal.

Zero LLM calls.
"""

from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opspilot.config import REPO_ROOT, get_settings
from opspilot.core import Role
from opspilot.data.corpus import generate_documents
from opspilot.data.evalset import build_queries
from opspilot.eval.abstention import _is_answerable
from opspilot.eval.evidence import EvidencePolicy, EvidenceTier, compute_signal
from opspilot.rag.chunking import chunk_corpus
from opspilot.rag.embeddings import HashingEmbedder
from opspilot.rag.retrieval import KnowledgeIndex, LexicalReranker, RetrievalFilter

BASELINE = {"hard_refusal": 0.370, "confident_wrong": 0.121, "balanced_accuracy": 0.754}


def main() -> int:
    s = get_settings()
    docs = generate_documents(seed=s.seed)
    queries = build_queries(docs)
    chunks = chunk_corpus(docs, "structure")
    index = KnowledgeIndex(chunks, HashingEmbedder(dim=s.hashing_dim)).build(use_qdrant=False)
    rr = LexicalReranker(bm25=index.bm25)

    rows = []
    for q in queries:
        hits = index.retrieve(
            q.text,
            arm="hybrid",
            flt=RetrievalFilter(role=Role(q.role_scope), exclude_deprecated=True),
            candidate_k=30,
            final_k=6,
            reranker=rr,
        )
        rows.append((q, compute_signal(q.text, hits, index.bm25), _is_answerable(q)))

    n_ans = sum(1 for _, _, a in rows if a)
    n_un = len(rows) - n_ans
    print(f"queries={len(rows)}  answerable={n_ans}  unanswerable={n_un}\n")

    def score(policy: EvidencePolicy) -> dict:
        c: dict = defaultdict(int)
        per_class: dict = defaultdict(lambda: defaultdict(int))
        for q, sig, answerable in rows:
            tier, _ = policy.classify(sig)
            c[(answerable, tier)] += 1
            if answerable:
                per_class[q.difficulty][tier.value] += 1
        return {
            "hard_refusal": c[(True, EvidenceTier.INSUFFICIENT)] / n_ans,
            "partial": c[(True, EvidenceTier.PARTIAL)] / n_ans,
            "answered": c[(True, EvidenceTier.SUFFICIENT)] / n_ans,
            "confident_wrong": c[(False, EvidenceTier.SUFFICIENT)] / n_un,
            "partial_on_unanswerable": c[(False, EvidenceTier.PARTIAL)] / n_un,
            "correct_refusal": c[(False, EvidenceTier.INSUFFICIENT)] / n_un,
            "per_class": {k: dict(v) for k, v in per_class.items()},
        }

    print("=== sweep (SUFFICIENT bar held at 1.5 throughout) ===")
    print(
        f"{'partial_top':>12} {'max_oov':>8} {'hard refuse':>12} {'partial':>9} "
        f"{'confident wrong':>16}"
    )
    grid = []
    for pt in (0.6, 0.7, 0.85, 1.0, 1.2):
        for oov in (0.25, 0.34, 0.45):
            r = score(EvidencePolicy(min_top=1.5, partial_top=pt, max_oov_mass=oov))
            grid.append(
                {
                    "partial_top": pt,
                    "max_oov_mass": oov,
                    **{k: v for k, v in r.items() if k != "per_class"},
                }
            )
            print(
                f"{pt:>12} {oov:>8} {r['hard_refusal']:>11.1%} {r['partial']:>9.1%} "
                f"{r['confident_wrong']:>16.1%}"
            )

    chosen_policy = EvidencePolicy(
        min_top=s.min_evidence_score,
        partial_top=s.partial_evidence_score,
        max_oov_mass=s.max_oov_mass,
    )
    chosen = score(chosen_policy)

    print(
        f"\n=== configured policy (min_top={chosen_policy.min_top}, "
        f"partial_top={chosen_policy.partial_top}, "
        f"max_oov={chosen_policy.max_oov_mass}) ==="
    )
    print(
        f"  hard refusal      {chosen['hard_refusal']:>6.1%}   "
        f"(baseline {BASELINE['hard_refusal']:.1%})"
    )
    print(f"  partial answer    {chosen['partial']:>6.1%}   (baseline n/a)")
    print(f"  full answer       {chosen['answered']:>6.1%}")
    print(
        f"  confident wrong   {chosen['confident_wrong']:>6.1%}   "
        f"(baseline {BASELINE['confident_wrong']:.1%})"
    )

    print("\n=== answerable queries by class ===")
    print(f"{'class':>6} {'answered':>9} {'partial':>9} {'refused':>9}")
    for cls in sorted(chosen["per_class"]):
        d = chosen["per_class"][cls]
        print(
            f"{cls:>6} {d.get('sufficient', 0):>9} {d.get('partial', 0):>9} "
            f"{d.get('insufficient', 0):>9}"
        )

    out = REPO_ROOT / "artifacts" / "reports" / "abstention_policy.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "baseline_two_tier": BASELINE,
                "configured": {
                    "min_top": chosen_policy.min_top,
                    "partial_top": chosen_policy.partial_top,
                    "max_oov_mass": chosen_policy.max_oov_mass,
                    **chosen,
                },
                "sweep": grid,
                "note": (
                    "The SUFFICIENT bar is unchanged from the baseline, so the "
                    "confident-wrong rate is unchanged by construction. The gain is "
                    "converting flat refusals into hedged, cited, related-material "
                    "answers."
                ),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
