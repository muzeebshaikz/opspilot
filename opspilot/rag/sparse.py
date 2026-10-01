"""Sparse lexical retrieval (BM25).

Implemented directly rather than pulled from a library for three reasons:
  1. we need the IDF weights to emit Qdrant sparse vectors;
  2. the per-term score contributions are needed to explain *why* a lexical
     match won, which is the point of the L-class analysis;
  3. it makes the tokenisation policy explicit -- and tokenisation is exactly
     where lexical retrieval of identifiers like ERR_VPN_809 is won or lost.

Tokenisation policy
-------------------
Identifiers are the whole reason sparse retrieval is in this system, so they are
preserved intact AND decomposed:
    "ERR_VPN_809"  ->  ["err_vpn_809", "err", "vpn", "809"]
This lets both an exact-code query and a partial "vpn 809" query match.
"""

from __future__ import annotations

import math
import re
from collections import Counter
from dataclasses import dataclass, field

_TOKEN = re.compile(r"[A-Za-z0-9_]+(?:\.[A-Za-z0-9_]+)*")
_SPLIT_ID = re.compile(r"[_\.]")
_STOP = frozenset(
    [
        "a",
        "an",
        "the",
        "and",
        "or",
        "but",
        "if",
        "then",
        "than",
        "of",
        "for",
        "to",
        "in",
        "on",
        "at",
        "by",
        "with",
        "from",
        "as",
        "is",
        "are",
        "was",
        "were",
        "be",
        "been",
        "this",
        "that",
        "these",
        "those",
        "it",
        "its",
        "i",
        "you",
        "your",
        "my",
        "me",
        "we",
        "our",
        "they",
        "them",
        "he",
        "she",
        "do",
        "does",
        "did",
        "how",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "can",
        "could",
        "should",
        "would",
        "will",
        "shall",
        "may",
        "might",
        "must",
        "not",
        "no",
        "yes",
    ]
)


def tokenize(text: str, *, keep_stopwords: bool = False) -> list[str]:
    out: list[str] = []
    for raw in _TOKEN.findall(text):
        low = raw.lower()
        if not keep_stopwords and low in _STOP:
            continue
        out.append(low)
        if _SPLIT_ID.search(low):
            parts = [p for p in _SPLIT_ID.split(low) if p]
            if len(parts) > 1:
                out.extend(parts)
    return out


@dataclass(slots=True)
class BM25Index:
    k1: float = 1.2
    b: float = 0.75
    doc_tokens: list[list[str]] = field(default_factory=list)
    doc_freq: Counter = field(default_factory=Counter)
    doc_len: list[int] = field(default_factory=list)
    avgdl: float = 0.0
    n_docs: int = 0
    _postings: dict[str, list[tuple[int, int]]] = field(default_factory=dict)
    _vocab: dict[str, int] = field(default_factory=dict)

    def fit(self, texts: list[str]) -> BM25Index:
        self.doc_tokens = [tokenize(t) for t in texts]
        self.doc_len = [len(t) for t in self.doc_tokens]
        self.n_docs = len(texts)
        self.avgdl = (sum(self.doc_len) / self.n_docs) if self.n_docs else 0.0
        self.doc_freq = Counter()
        self._postings = {}
        for i, toks in enumerate(self.doc_tokens):
            tf = Counter(toks)
            for term, f in tf.items():
                self.doc_freq[term] += 1
                self._postings.setdefault(term, []).append((i, f))
        self._vocab = {t: i for i, t in enumerate(sorted(self._postings))}
        return self

    def idf(self, term: str) -> float:
        df = self.doc_freq.get(term, 0)
        if df == 0:
            return 0.0
        # Robertson/Sparck-Jones idf with the +1 guard against negative values
        return math.log(1.0 + (self.n_docs - df + 0.5) / (df + 0.5))

    def search(
        self, query: str, top_k: int = 30, allowed: set[int] | None = None
    ) -> list[tuple[int, float]]:
        q_terms = tokenize(query)
        if not q_terms:
            return []
        scores: dict[int, float] = {}
        for term in set(q_terms):
            postings = self._postings.get(term)
            if not postings:
                continue
            idf = self.idf(term)
            for doc_i, f in postings:
                if allowed is not None and doc_i not in allowed:
                    continue
                dl = self.doc_len[doc_i] or 1
                denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
                scores[doc_i] = scores.get(doc_i, 0.0) + idf * (f * (self.k1 + 1)) / denom
        return sorted(scores.items(), key=lambda kv: -kv[1])[:top_k]

    # -- Qdrant sparse-vector emission ------------------------------------
    def sparse_vector(self, text: str) -> tuple[list[int], list[float]]:
        """BM25-weighted sparse vector for a document."""
        toks = tokenize(text)
        if not toks:
            return [], []
        tf = Counter(toks)
        dl = len(toks) or 1
        idx, val = [], []
        for term, f in tf.items():
            j = self._vocab.get(term)
            if j is None:
                continue
            denom = f + self.k1 * (1 - self.b + self.b * dl / (self.avgdl or 1))
            idx.append(j)
            val.append(self.idf(term) * (f * (self.k1 + 1)) / denom)
        return idx, val

    def sparse_query_vector(self, text: str) -> tuple[list[int], list[float]]:
        """Query side carries idf only; document side carries the saturation."""
        idx, val = [], []
        for term in set(tokenize(text)):
            j = self._vocab.get(term)
            if j is None:
                continue
            idx.append(j)
            val.append(1.0)
        return idx, val
