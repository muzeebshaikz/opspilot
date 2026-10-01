"""Retrieval metrics with span-based relevance and paired bootstrap CIs.

Two things here are deliberate and matter:

1. Relevance is decided by SPAN OVERLAP between a retrieved chunk and a gold
   document span, not by chunk-id equality. Labels therefore survive a change of
   chunking strategy (audit G2).

2. Every comparison between two arms reports a paired bootstrap 95% CI. House
   rule from the audit (G3): an improvement whose CI crosses zero is reported as
   "no measurable difference", never as a win.

None of these metrics require an LLM, so they run unlimited and offline -- which
is what makes rigorous measurement affordable under the Groq free-tier ceiling.
"""

from __future__ import annotations

import math
import random
from dataclasses import dataclass

from opspilot.data.evalset import EvalQuery
from opspilot.rag.chunking import Chunk
from opspilot.rag.retrieval import Hit


def grade_of(chunk: Chunk, query: EvalQuery) -> int:
    """Highest grade among gold spans this chunk overlaps."""
    best = 0
    for q in query.qrels:
        if chunk.document_id != q.document_id:
            continue
        if chunk.overlaps(q.char_start, q.char_end):
            best = max(best, q.grade)
    return best


def relevant_chunk_count(chunks: list[Chunk], query: EvalQuery) -> int:
    return sum(1 for c in chunks if grade_of(c, query) > 0)


def ideal_grades(chunks: list[Chunk], query: EvalQuery, k: int) -> list[int]:
    grades = sorted((grade_of(c, query) for c in chunks), reverse=True)
    return grades[:k]


@dataclass(slots=True)
class QueryScores:
    query_id: str
    difficulty: str
    recall_at_k: float
    precision_at_k: float
    reciprocal_rank: float
    ndcg_at_k: float
    retrieved: int


def score_query(
    hits: list[Hit], query: EvalQuery, all_chunks: list[Chunk], k: int = 10
) -> QueryScores:
    top = hits[:k]
    grades = [grade_of(h.chunk, query) for h in top]

    hit_docs = {h.chunk.document_id for h, g in zip(top, grades, strict=True) if g > 0}
    gold_docs = {q.document_id for q in query.qrels if q.grade > 0}

    # Recall is measured over gold DOCUMENTS, not chunks: retrieving two chunks
    # of the same document should not count as covering two answers.
    recall = (len(hit_docs & gold_docs) / len(gold_docs)) if gold_docs else 0.0
    precision = (sum(1 for g in grades if g > 0) / len(top)) if top else 0.0

    rr = 0.0
    for i, g in enumerate(grades, 1):
        if g > 0:
            rr = 1.0 / i
            break

    dcg = sum((2**g - 1) / math.log2(i + 1) for i, g in enumerate(grades, 1))
    idcg = sum(
        (2**g - 1) / math.log2(i + 1) for i, g in enumerate(ideal_grades(all_chunks, query, k), 1)
    )
    ndcg = (dcg / idcg) if idcg > 0 else 0.0

    return QueryScores(
        query_id=query.query_id,
        difficulty=query.difficulty,
        recall_at_k=recall,
        precision_at_k=precision,
        reciprocal_rank=rr,
        ndcg_at_k=ndcg,
        retrieved=len(top),
    )


def aggregate(scores: list[QueryScores]) -> dict[str, float]:
    if not scores:
        return {"recall": 0.0, "precision": 0.0, "mrr": 0.0, "ndcg": 0.0, "n": 0}
    n = len(scores)
    return {
        "recall": sum(s.recall_at_k for s in scores) / n,
        "precision": sum(s.precision_at_k for s in scores) / n,
        "mrr": sum(s.reciprocal_rank for s in scores) / n,
        "ndcg": sum(s.ndcg_at_k for s in scores) / n,
        "n": n,
    }


def by_difficulty(scores: list[QueryScores]) -> dict[str, dict[str, float]]:
    groups: dict[str, list[QueryScores]] = {}
    for s in scores:
        groups.setdefault(s.difficulty, []).append(s)
    return {d: aggregate(v) for d, v in sorted(groups.items())}


# --------------------------------------------------------------------------
# significance
# --------------------------------------------------------------------------
@dataclass(slots=True)
class Comparison:
    metric: str
    mean_a: float
    mean_b: float
    delta: float
    ci_low: float
    ci_high: float
    n: int

    @property
    def significant(self) -> bool:
        """CI must not straddle zero (audit G3)."""
        return (self.ci_low > 0 and self.ci_high > 0) or (self.ci_low < 0 and self.ci_high < 0)

    @property
    def verdict(self) -> str:
        if not self.significant:
            return "no measurable difference"
        return "B better" if self.delta > 0 else "A better"


def paired_bootstrap(
    a: list[QueryScores],
    b: list[QueryScores],
    metric: str = "ndcg_at_k",
    resamples: int = 10_000,
    seed: int = 20260929,
    alpha: float = 0.05,
) -> Comparison:
    """Paired bootstrap over per-query deltas. `a` and `b` must be aligned."""
    by_a = {s.query_id: s for s in a}
    ids = [s.query_id for s in b if s.query_id in by_a]
    deltas = [
        getattr(next(s for s in b if s.query_id == i), metric) - getattr(by_a[i], metric)
        for i in ids
    ]
    n = len(deltas)
    if n == 0:
        return Comparison(metric, 0, 0, 0, 0, 0, 0)

    rng = random.Random(seed)
    means = []
    for _ in range(resamples):
        means.append(sum(deltas[rng.randrange(n)] for _ in range(n)) / n)
    means.sort()
    lo = means[int((alpha / 2) * resamples)]
    hi = means[min(resamples - 1, int((1 - alpha / 2) * resamples))]
    mean_a = sum(getattr(by_a[i], metric) for i in ids) / n
    mean_b = mean_a + sum(deltas) / n
    return Comparison(metric, mean_a, mean_b, sum(deltas) / n, lo, hi, n)
