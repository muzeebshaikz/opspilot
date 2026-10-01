"""Hybrid retrieval: dense + sparse -> RRF -> rerank -> context.

Why RRF is implemented here in Python when Qdrant can fuse server-side
---------------------------------------------------------------------
Qdrant's local mode *does* support `FusionQuery(RRF)` (verified by spike). We
still fuse in Python because the evaluation needs the per-arm ranked lists
intact: "BM25 beat dense on L-class by X" is impossible to say if the arms were
already collapsed inside the database. Server-side fusion is the production
path; Python fusion is the measurement path. Both are exercised.
"""

from __future__ import annotations

import contextlib
import pickle
from dataclasses import dataclass
from pathlib import Path
from typing import ClassVar, Literal

import numpy as np

from opspilot.core import Role
from opspilot.rag.chunking import Chunk
from opspilot.rag.embeddings import Embedder
from opspilot.rag.sparse import BM25Index, tokenize

Arm = Literal["dense", "sparse", "hybrid"]


@dataclass(slots=True)
class Hit:
    chunk: Chunk
    score: float
    rank: int = 0
    arm: str = ""
    dense_rank: int | None = None
    sparse_rank: int | None = None
    rerank_score: float | None = None

    @property
    def citation(self) -> dict:
        c = self.chunk
        return {
            "document_id": c.document_id,
            "title": c.title,
            "section": c.heading,
            "version": c.version,
            "category": c.category,
            "product": c.product,
            "effective_date": c.effective_date,
            "status": c.status,
            "score": round(self.rerank_score if self.rerank_score is not None else self.score, 4),
        }


@dataclass(slots=True)
class RetrievalFilter:
    """Metadata filters (spec §9). `role` is applied unconditionally -- it is a
    security control, not a preference, and is enforced here AND at the service
    layer."""

    role: Role = Role.EMPLOYEE
    category: str | None = None
    product: str | None = None
    platform: str | None = None
    exclude_deprecated: bool = False

    def allows(self, c: Chunk) -> bool:
        if not self.role.can_see(c.access_level):
            return False
        if self.category and c.category != self.category:
            return False
        if self.product and c.product != self.product:
            return False
        if self.platform and c.platform not in (None, self.platform):
            return False
        return not (self.exclude_deprecated and c.status == "deprecated")


def reciprocal_rank_fusion(
    ranked_lists: dict[str, list[int]],
    k: int = 60,
    weights: dict[str, float] | None = None,
) -> list[tuple[int, float, dict[str, int]]]:
    """Standard RRF: score(d) = sum_arms w / (k + rank(d)).

    `k` damps the influence of the very top of each list; a larger k makes the
    fusion more democratic. Returns (index, score, {arm: rank}).
    """
    weights = weights or {}
    scores: dict[int, float] = {}
    ranks: dict[int, dict[str, int]] = {}
    for arm, ids in ranked_lists.items():
        w = weights.get(arm, 1.0)
        for rank, idx in enumerate(ids, start=1):
            scores[idx] = scores.get(idx, 0.0) + w / (k + rank)
            ranks.setdefault(idx, {})[arm] = rank
    fused = sorted(scores.items(), key=lambda kv: -kv[1])
    return [(i, s, ranks[i]) for i, s in fused]


# --------------------------------------------------------------------------
# rerankers
# --------------------------------------------------------------------------
class LexicalReranker:
    """Feature-based reranker. No model required.

    It is NOT a second BM25 pass -- that would add nothing. It scores signals
    that fusion structurally cannot see:
      * exact identifier / phrase presence (drives L-class precision)
      * heading-vs-query alignment
      * document status: current > superseded > deprecated (drives V-class)
      * product-name agreement with the query (drives D-class distractors)
    """

    name = "lexical-v2"

    _STATUS_BOOST: ClassVar[dict[str, float]] = {
        "current": 0.25,
        "superseded": -0.30,
        "deprecated": -0.55,
    }
    _FAMILY_BOOST: ClassVar[dict[str, float]] = {
        "canonical": 0.10,
        "errorcode": 0.08,
        "runbook": 0.05,
        "faq": 0.02,
        "distractor": 0.0,
        "version": -0.05,
        "overview": -0.04,
    }

    def __init__(self, bm25: BM25Index | None = None):
        """`bm25` supplies IDF so coverage can be weighted by term informativeness.

        Without it, matching "procedure" counts the same as matching
        ERR_VPN_809, which makes the score useless as an evidence signal: an
        out-of-domain question full of common words scores as highly as a
        genuine hit. Measured directly -- see artifacts/reports/abstention_calibration.md.
        """
        self.bm25 = bm25

    def _importance(self, term: str) -> float:
        """Out-of-vocabulary terms are treated as MAXIMALLY informative.

        This is the key inversion. A query term that appears nowhere in the
        corpus ("tape", "CICS", "Novell") is the strongest possible evidence
        that we cannot answer -- so it must carry weight in the denominator and
        never be satisfied in the numerator.
        """
        if self.bm25 is None:
            return 1.0
        if self.bm25.doc_freq.get(term, 0) == 0:
            return self._max_idf()
        return max(0.1, self.bm25.idf(term))

    def _max_idf(self) -> float:
        n = getattr(self.bm25, "n_docs", 0) or 1
        import math

        return math.log(1.0 + (n - 1 + 0.5) / 1.5)

    def score(self, query: str, chunk: Chunk) -> float:
        q_tokens = set(tokenize(query))
        if not q_tokens:
            return 0.0
        text_l = chunk.text.lower()
        c_tokens = set(tokenize(chunk.text))
        weights = {t: self._importance(t) for t in q_tokens}
        total_w = sum(weights.values()) or 1.0
        coverage = sum(w for t, w in weights.items() if t in c_tokens) / total_w

        # exact identifier match is a very strong signal
        ident = [t for t in q_tokens if any(ch.isdigit() for ch in t) and len(t) > 3]
        ident_hit = sum(1 for t in ident if t in text_l) / len(ident) if ident else 0.0

        head_tokens = set(tokenize(chunk.heading or ""))
        head = len(q_tokens & head_tokens) / len(q_tokens)

        title_tokens = set(tokenize(chunk.title or ""))
        title = len(q_tokens & title_tokens) / len(q_tokens)

        # product agreement: if the query names a product, prefer that product
        product_tokens = set(tokenize(chunk.product or ""))
        product = 0.0
        if product_tokens:
            overlap = len(q_tokens & product_tokens) / len(product_tokens)
            product = overlap - (0.35 if overlap == 0 and chunk.family == "distractor" else 0.0)

        return (
            0.90 * coverage
            + 0.85 * ident_hit
            + 0.35 * head
            + 0.30 * title
            + 0.40 * product
            + self._STATUS_BOOST.get(chunk.status, 0.0)
            + self._FAMILY_BOOST.get(chunk.family, 0.0)
        )

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        for h in hits:
            h.rerank_score = self.score(query, h.chunk)
        out = sorted(hits, key=lambda h: -(h.rerank_score or 0.0))[:top_k]
        for i, h in enumerate(out, 1):
            h.rank = i
        return out


class CrossEncoderReranker:
    """ONNX cross-encoder (ms-marco-MiniLM class).

    Requires model files. On CPU the p95 for 30 pairs is the least certain
    number in the latency budget -- measure before committing (audit §8.3).
    """

    def __init__(self, model: str = "Xenova/ms-marco-MiniLM-L-6-v2", cache_dir: str | None = None):
        from fastembed.rerank.cross_encoder import TextCrossEncoder

        self.name = f"cross-encoder:{model}"
        self._model = TextCrossEncoder(model_name=model, cache_dir=cache_dir)

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        if not hits:
            return []
        scores = list(self._model.rerank(query, [h.chunk.text for h in hits]))
        for h, s in zip(hits, scores, strict=True):
            h.rerank_score = float(s)
        out = sorted(hits, key=lambda h: -(h.rerank_score or 0.0))[:top_k]
        for i, h in enumerate(out, 1):
            h.rank = i
        return out


class NullReranker:
    name = "none"

    def rerank(self, query: str, hits: list[Hit], top_k: int) -> list[Hit]:
        out = hits[:top_k]
        for i, h in enumerate(out, 1):
            h.rank = i
        return out


# --------------------------------------------------------------------------
# the index
# --------------------------------------------------------------------------
class KnowledgeIndex:
    """Dense vectors in Qdrant (local mode, no server), BM25 in process.

    Dense search runs through Qdrant so the production code path is the one
    being measured; swapping `qdrant_path` for `qdrant_url` moves to a server
    with no other change.
    """

    def __init__(
        self,
        chunks: list[Chunk],
        embedder: Embedder,
        collection: str = "opspilot_knowledge",
        qdrant_path: str | None = None,
        qdrant_url: str | None = None,
    ):
        self.chunks = chunks
        self.embedder = embedder
        self.collection = collection
        self._by_id = {c.chunk_id: i for i, c in enumerate(chunks)}
        self.bm25 = BM25Index().fit([self._index_text(c) for c in chunks])
        self._dense: np.ndarray | None = None
        self._client = None
        self._qdrant_path = qdrant_path
        self._qdrant_url = qdrant_url

    @staticmethod
    def _index_text(c: Chunk) -> str:
        """Title and heading are prepended so lexical matching can use them."""
        return f"{c.title}\n{c.heading}\n{c.product}\n{c.text}"

    # -- build ------------------------------------------------------------
    def build(self, *, use_qdrant: bool = True, batch: int = 256) -> KnowledgeIndex:
        texts = [self._index_text(c) for c in self.chunks]
        vecs = []
        for i in range(0, len(texts), batch):
            vecs.append(self.embedder.encode(texts[i : i + batch]))
        self._dense = np.vstack(vecs) if vecs else np.zeros((0, self.embedder.dim), np.float32)
        if use_qdrant:
            self._build_qdrant()
        return self

    def _build_qdrant(self) -> None:
        from qdrant_client import QdrantClient, models

        client = (
            QdrantClient(url=self._qdrant_url)
            if self._qdrant_url
            else QdrantClient(path=self._qdrant_path)
        )
        if client.collection_exists(self.collection):
            client.delete_collection(self.collection)
        client.create_collection(
            self.collection,
            vectors_config={
                "dense": models.VectorParams(
                    size=self.embedder.dim, distance=models.Distance.COSINE
                )
            },
            sparse_vectors_config={"sparse": models.SparseVectorParams()},
        )
        points = []
        for i, c in enumerate(self.chunks):
            idx, val = self.bm25.sparse_vector(self._index_text(c))
            points.append(
                models.PointStruct(
                    id=i,
                    vector={
                        "dense": self._dense[i].tolist(),
                        "sparse": models.SparseVector(indices=idx, values=val),
                    },
                    payload={
                        "chunk_id": c.chunk_id,
                        "document_id": c.document_id,
                        "category": c.category,
                        "product": c.product,
                        "version": c.version,
                        "access_level": c.access_level,
                        "status": c.status,
                        "family": c.family,
                        "platform": c.platform or "",
                    },
                )
            )
        for i in range(0, len(points), 512):
            client.upsert(self.collection, points[i : i + 512])
        self._client = client

    def close(self) -> None:
        if self._client is not None:
            with contextlib.suppress(Exception):
                self._client.close()
            self._client = None

    # -- arms -------------------------------------------------------------
    def _allowed(self, flt: RetrievalFilter) -> list[int]:
        return [i for i, c in enumerate(self.chunks) if flt.allows(c)]

    def dense_search(self, query: str, top_k: int, flt: RetrievalFilter) -> list[int]:
        if self._dense is None or not len(self._dense):
            return []
        q = self.embedder.encode([query], is_query=True)[0]
        allowed = self._allowed(flt)
        if not allowed:
            return []
        sub = self._dense[allowed]
        sims = sub @ q
        order = np.argsort(-sims)[:top_k]
        return [allowed[i] for i in order]

    def dense_search_qdrant(self, query: str, top_k: int, flt: RetrievalFilter) -> list[int]:
        """Production path: same query through Qdrant with server-side filters."""
        from qdrant_client import models

        if self._client is None:
            return self.dense_search(query, top_k, flt)
        q = self.embedder.encode([query], is_query=True)[0].tolist()
        must = [
            models.FieldCondition(
                key="access_level",
                match=models.MatchAny(any=[r.value for r in Role if flt.role.can_see(r.value)]),
            )
        ]
        if flt.category:
            must.append(
                models.FieldCondition(key="category", match=models.MatchValue(value=flt.category))
            )
        if flt.product:
            must.append(
                models.FieldCondition(key="product", match=models.MatchValue(value=flt.product))
            )
        res = self._client.query_points(
            self.collection,
            query=q,
            using="dense",
            query_filter=models.Filter(must=must),
            limit=top_k,
        )
        return [int(p.id) for p in res.points]

    def sparse_search(self, query: str, top_k: int, flt: RetrievalFilter) -> list[int]:
        allowed = set(self._allowed(flt))
        return [i for i, _ in self.bm25.search(query, top_k=top_k, allowed=allowed)]

    # -- orchestration ----------------------------------------------------
    def retrieve(
        self,
        query: str,
        *,
        arm: Arm = "hybrid",
        flt: RetrievalFilter | None = None,
        candidate_k: int = 30,
        final_k: int = 6,
        rrf_k: int = 60,
        reranker=None,
        use_qdrant_dense: bool = False,
        weights: dict[str, float] | None = None,
    ) -> list[Hit]:
        flt = flt or RetrievalFilter()
        lists: dict[str, list[int]] = {}
        if arm in ("dense", "hybrid"):
            lists["dense"] = (
                self.dense_search_qdrant(query, candidate_k, flt)
                if use_qdrant_dense
                else self.dense_search(query, candidate_k, flt)
            )
        if arm in ("sparse", "hybrid"):
            lists["sparse"] = self.sparse_search(query, candidate_k, flt)

        if arm == "hybrid":
            # Weighted RRF. Equal votes assume both arms are equally good --
            # measured, they are not, and averaging a weak arm into a strong one
            # made hybrid WORSE than sparse alone. Weights are swept, not assumed
            # (scripts/tune_fusion.py).
            fused = reciprocal_rank_fusion(lists, k=rrf_k, weights=weights)
            hits = [
                Hit(
                    chunk=self.chunks[i],
                    score=s,
                    arm="hybrid",
                    dense_rank=r.get("dense"),
                    sparse_rank=r.get("sparse"),
                )
                for i, s, r in fused[:candidate_k]
            ]
        else:
            ids = lists[arm]
            hits = [
                Hit(
                    chunk=self.chunks[i],
                    score=1.0 / (rrf_k + rank),
                    arm=arm,
                    **({"dense_rank": rank} if arm == "dense" else {"sparse_rank": rank}),
                )
                for rank, i in enumerate(ids, 1)
            ][:candidate_k]

        if reranker is not None:
            hits = reranker.rerank(query, hits, final_k)
        else:
            hits = hits[:final_k]
            for i, h in enumerate(hits, 1):
                h.rank = i
        return hits

    # -- persistence -------------------------------------------------------
    def save_sidecar(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("wb") as fh:
            pickle.dump(
                {"chunks": self.chunks, "dense": self._dense, "embedder": self.embedder.name}, fh
            )
