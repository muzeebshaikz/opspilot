"""Sweep fusion weights and RRF k to fix hybrid underperforming BM25 alone.

The problem being solved
------------------------
Measured: E3 hybrid (nDCG 0.372) is SIGNIFICANTLY WORSE than E2 sparse alone
(0.406), CI [-0.052, -0.015]. Unweighted RRF gives both arms an equal vote, so
averaging a strong arm with a weak one drags the strong one down.

This is not fixed by asserting a weight. It is fixed by sweeping the weight and
the smoothing constant against the labelled set, reporting confidence intervals,
and accepting whatever the data says -- including "drop the dense arm entirely"
if w=0 wins.

Zero LLM calls. Runs in seconds, unlimited.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opspilot.config import REPO_ROOT, get_settings
from opspilot.core import Role
from opspilot.data.corpus import generate_documents
from opspilot.data.evalset import build_queries
from opspilot.eval.metrics import aggregate, by_difficulty, paired_bootstrap, score_query
from opspilot.rag.chunking import chunk_corpus
from opspilot.rag.embeddings import HashingEmbedder
from opspilot.rag.retrieval import KnowledgeIndex, LexicalReranker, RetrievalFilter


def evaluate(index, queries, chunks, *, dense_w, sparse_w, rrf_k, rerank, k=10):
    rr = LexicalReranker(bm25=index.bm25) if rerank else None
    scores = []
    for q in queries:
        if q.difficulty == "X" or not q.qrels:
            continue
        hits = index.retrieve(
            q.text,
            arm="hybrid",
            flt=RetrievalFilter(role=Role(q.role_scope)),
            candidate_k=30,
            final_k=k,
            rrf_k=rrf_k,
            reranker=rr,
            weights={"dense": dense_w, "sparse": sparse_w},
        )
        scores.append(score_query(hits, q, chunks, k=k))
    return scores


def main() -> int:
    s = get_settings()
    docs = generate_documents(seed=s.seed)
    queries = build_queries(docs)
    chunks = chunk_corpus(docs, "structure")
    index = KnowledgeIndex(chunks, HashingEmbedder(dim=s.hashing_dim)).build(use_qdrant=False)
    print(f"docs={len(docs)} chunks={len(chunks)} scored={sum(1 for q in queries if q.qrels)}\n")

    results = []
    print("=== Sweep: dense weight (sparse fixed at 1.0), no reranker ===")
    print(f"{'dense_w':>8} {'rrf_k':>6} {'nDCG':>7} {'Recall':>7} {'MRR':>7}")
    for dense_w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        for rrf_k in (60,):
            sc = evaluate(
                index, queries, chunks, dense_w=dense_w, sparse_w=1.0, rrf_k=rrf_k, rerank=False
            )
            agg = aggregate(sc)
            results.append(("plain", dense_w, rrf_k, agg, sc))
            print(
                f"{dense_w:>8.1f} {rrf_k:>6} {agg['ndcg']:>7.3f} "
                f"{agg['recall']:>7.3f} {agg['mrr']:>7.3f}"
            )

    print("\n=== Sweep: dense weight WITH reranker (the production path) ===")
    print(f"{'dense_w':>8} {'rrf_k':>6} {'nDCG':>7} {'Recall':>7} {'MRR':>7}")
    reranked = []
    for dense_w in (0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 1.0):
        for rrf_k in (20, 60):
            sc = evaluate(
                index, queries, chunks, dense_w=dense_w, sparse_w=1.0, rrf_k=rrf_k, rerank=True
            )
            agg = aggregate(sc)
            reranked.append((dense_w, rrf_k, agg, sc))
            print(
                f"{dense_w:>8.1f} {rrf_k:>6} {agg['ndcg']:>7.3f} "
                f"{agg['recall']:>7.3f} {agg['mrr']:>7.3f}"
            )

    best = max(reranked, key=lambda r: r[2]["ndcg"])
    baseline = next(r for r in reranked if r[0] == 1.0 and r[1] == 60)
    print(f"\nbest: dense_w={best[0]} rrf_k={best[1]} nDCG={best[2]['ndcg']:.3f}")
    print(f"current (dense_w=1.0, rrf_k=60): nDCG={baseline[2]['ndcg']:.3f}")

    cmp = paired_bootstrap(baseline[3], best[3], metric="ndcg_at_k")
    print(f"delta={cmp.delta:+.4f} CI=[{cmp.ci_low:+.4f}, {cmp.ci_high:+.4f}] -> {cmp.verdict}")

    print("\n=== per difficulty class at the best setting ===")
    per = by_difficulty(best[3])
    base_per = by_difficulty(baseline[3])
    print(f"{'class':>6} {'current':>9} {'best':>9} {'delta':>8}")
    for cls in sorted(per):
        b, n = base_per.get(cls, {}).get("ndcg", 0), per[cls]["ndcg"]
        print(f"{cls:>6} {b:>9.3f} {n:>9.3f} {n - b:>+8.3f}")

    out = REPO_ROOT / "artifacts" / "reports" / "fusion_tuning.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(
        json.dumps(
            {
                "best": {"dense_weight": best[0], "rrf_k": best[1], **best[2]},
                "current": {"dense_weight": 1.0, "rrf_k": 60, **baseline[2]},
                "delta": cmp.delta,
                "ci": [cmp.ci_low, cmp.ci_high],
                "significant": cmp.significant,
                "verdict": cmp.verdict,
                "sweep_reranked": [
                    {"dense_weight": w, "rrf_k": rk, **a} for w, rk, a, _ in reranked
                ],
                "sweep_plain": [
                    {"dense_weight": w, "rrf_k": rk, **a} for _, w, rk, a, _ in results
                ],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    print(f"\nwritten: {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
