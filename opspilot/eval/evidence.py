"""Composite evidence signal, to cut false abstention.

The problem
-----------
The original gate was a single number: max reranker score >= 1.5. Measured, that
refused 37% of ANSWERABLE questions in order to hold confident-wrong answers to
12%. One in three legitimate questions bounced.

A single threshold on a single score cannot do better, because two very different
situations produce the same middling score:

  (a) the corpus genuinely has nothing -- "tape backup retrieval";
  (b) the corpus has the answer but the weak dense arm scored it low --
      "internal DNS names not resolving over VPN".

Refusing (a) is correct. Refusing (b) is the 37%.

The fix
-------
Separate them with signals that differ between the two cases:

  top          best reranker score (the original signal)
  oov_mass     share of the query's IDF weight sitting on terms that appear in
               NO document. High for (a), low for (b). This is the strongest
               single discriminator -- if the user names something the corpus has
               never heard of, that is near-proof we cannot answer.
  support      how many retrieved passages are close to the top score. Real
               topic coverage produces several; a lucky single match does not.
  agreement    do the top passages come from the same document or category?
               Scattered hits across unrelated documents mean nothing matched.

Thresholds are swept over the labelled set, never assumed.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum

from opspilot.rag.retrieval import Hit
from opspilot.rag.sparse import BM25Index, tokenize


@dataclass(slots=True)
class EvidenceSignal:
    top: float
    oov_mass: float
    support: int
    agreement: float
    n_hits: int

    def as_dict(self) -> dict:
        return {
            "top": round(self.top, 3),
            "oov_mass": round(self.oov_mass, 3),
            "support": self.support,
            "agreement": round(self.agreement, 3),
            "n_hits": self.n_hits,
        }


def compute_signal(query: str, hits: list[Hit], bm25: BM25Index) -> EvidenceSignal:
    if not hits:
        return EvidenceSignal(0.0, 1.0, 0, 0.0, 0)

    scores = [h.rerank_score if h.rerank_score is not None else h.score for h in hits]
    top = max(scores)

    # Share of query IDF mass on terms the corpus has never seen.
    terms = set(tokenize(query))
    if terms:
        import math

        n = max(1, bm25.n_docs)
        max_idf = math.log(1.0 + (n - 1 + 0.5) / 1.5)
        weights = {
            t: (max_idf if bm25.doc_freq.get(t, 0) == 0 else max(0.1, bm25.idf(t))) for t in terms
        }
        total = sum(weights.values()) or 1.0
        oov = sum(w for t, w in weights.items() if bm25.doc_freq.get(t, 0) == 0)
        oov_mass = oov / total
    else:
        oov_mass = 1.0

    support = sum(1 for s in scores if s >= 0.55 * top) if top > 0 else 0

    docs = [h.chunk.document_id for h in hits[:4]]
    cats = [h.chunk.category for h in hits[:4]]
    agreement = (
        max((docs.count(d) / len(docs) for d in set(docs)), default=0.0) * 0.5
        + max((cats.count(c) / len(cats) for c in set(cats)), default=0.0) * 0.5
    )

    return EvidenceSignal(top, oov_mass, support, agreement, len(hits))


class EvidenceTier(StrEnum):
    """Three tiers, not two.

    A binary answer/refuse gate has to put every marginal case on one side. The
    measured cost of putting them on the refuse side was 37% of answerable
    questions bounced; putting them on the answer side raises confident-wrong
    answers above the safety cap. Neither is acceptable, because the premise is
    wrong: "I found related material but not a direct answer" is a genuinely
    different thing to say, and it is useful.

    PARTIAL answers from the evidence but frames it as related-not-definitive
    and shows the sources, so the user can judge. It is not a confident answer
    and is not counted as one.
    """

    SUFFICIENT = "sufficient"
    PARTIAL = "partial"
    INSUFFICIENT = "insufficient"


@dataclass(slots=True)
class EvidencePolicy:
    """Decision rule over the composite signal.

    The OOV veto runs first and applies to every tier: if the question is mostly
    about words the corpus has never seen, no amount of retrieval score should
    produce an answer. That is what makes the lower PARTIAL floor safe.
    """

    min_top: float = 1.5
    partial_top: float = 0.85
    max_oov_mass: float = 0.34
    min_support: int = 1

    def classify(self, sig: EvidenceSignal) -> tuple[EvidenceTier, str]:
        if sig.n_hits == 0:
            return EvidenceTier.INSUFFICIENT, "no documents retrieved"
        if sig.oov_mass > self.max_oov_mass:
            return EvidenceTier.INSUFFICIENT, (
                f"{sig.oov_mass:.0%} of the question is about terms that appear in no document"
            )
        if sig.top >= self.min_top and sig.support >= self.min_support:
            return EvidenceTier.SUFFICIENT, "sufficient"
        if sig.top >= self.partial_top:
            return EvidenceTier.PARTIAL, (
                f"best evidence score {sig.top:.2f} is related but not conclusive"
            )
        return EvidenceTier.INSUFFICIENT, f"best evidence score {sig.top:.2f} is too weak"

    # Kept so existing call sites and tests keep working.
    def sufficient(self, sig: EvidenceSignal) -> tuple[bool, str]:
        tier, reason = self.classify(sig)
        return tier is EvidenceTier.SUFFICIENT, reason
