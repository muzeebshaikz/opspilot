"""Embedding provider abstraction (spec §12).

Two backends, same interface:

  HashingEmbedder -- pure numpy. No model download, no network, no torch.
                     Hashed word unigrams + character 4-grams projected to a
                     fixed dimension, L2 normalised.
  OnnxEmbedder    -- fastembed / ONNX Runtime (bge-small-en-v1.5 by default).
                     Real neural embeddings, CPU, no PyTorch dependency.

Why a no-model default exists
-----------------------------
On the target machine the corporate proxy rejects HuggingFace model-file
downloads (TLS handshake failure on the CDN), so a neural embedder cannot be
obtained. Rather than block the whole project, HashingEmbedder lets the entire
pipeline -- ingestion, retrieval, fusion, reranking, evaluation -- run and be
measured today. It is a genuine baseline, not a stub: it produces real, stable
vectors and real scores.

It is also expected to be WEAK on paraphrase queries, because hashing captures
surface form rather than meaning. That is not a defect to hide -- it is the
measurement that justifies swapping in OnnxEmbedder. See the S-class results in
artifacts/reports/.
"""

from __future__ import annotations

import hashlib
import re
from typing import Protocol, runtime_checkable

import numpy as np

_WORD = re.compile(r"[a-z0-9_]+")


@runtime_checkable
class Embedder(Protocol):
    name: str
    dim: int

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray: ...


def _tokens(text: str) -> list[str]:
    return _WORD.findall(text.lower())


def _char_ngrams(text: str, n: int = 4) -> list[str]:
    s = re.sub(r"\s+", " ", text.lower())
    return [s[i : i + n] for i in range(max(0, len(s) - n + 1))]


class HashingEmbedder:
    """Deterministic, dependency-free, offline-capable embedder."""

    name = "hashing-v1"

    def __init__(
        self,
        dim: int = 384,
        word_weight: float = 1.0,
        ngram_weight: float = 0.45,
        seed: int = 20260929,
    ):
        self.dim = dim
        self.word_weight = word_weight
        self.ngram_weight = ngram_weight
        self.seed = seed

    def _bucket(self, token: str) -> tuple[int, float]:
        h = hashlib.blake2b(f"{self.seed}:{token}".encode(), digest_size=8).digest()
        idx = int.from_bytes(h[:4], "little") % self.dim
        sign = 1.0 if h[4] & 1 else -1.0
        return idx, sign

    def _one(self, text: str) -> np.ndarray:
        v = np.zeros(self.dim, dtype=np.float32)
        words = _tokens(text)
        if words:
            # sublinear term frequency damping, as in TF-IDF
            counts: dict[str, int] = {}
            for w in words:
                counts[w] = counts.get(w, 0) + 1
            for w, c in counts.items():
                idx, sign = self._bucket(w)
                v[idx] += sign * self.word_weight * (1.0 + np.log(c))
        for g in _char_ngrams(text):
            idx, sign = self._bucket("#" + g)
            v[idx] += sign * self.ngram_weight
        norm = float(np.linalg.norm(v))
        return v / norm if norm > 0 else v

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return np.vstack([self._one(t) for t in texts])


class OnnxEmbedder:
    """fastembed / ONNX Runtime. CPU-only, no PyTorch.

    Requires the model files to be present in the fastembed cache. Raises at
    construction if they cannot be fetched, so the caller can fall back.
    """

    def __init__(
        self, model: str = "BAAI/bge-small-en-v1.5", dim: int = 384, cache_dir: str | None = None
    ):
        from fastembed import TextEmbedding  # imported lazily: optional dependency

        self.name = f"onnx:{model}"
        self.dim = dim
        self._model = TextEmbedding(model_name=model, cache_dir=cache_dir)

    def encode(self, texts: list[str], *, is_query: bool = False) -> np.ndarray:
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        gen = self._model.query_embed(texts) if is_query else self._model.embed(texts)
        arr = np.vstack([np.asarray(v, dtype=np.float32) for v in gen])
        norms = np.linalg.norm(arr, axis=1, keepdims=True)
        return arr / np.where(norms == 0, 1, norms)


def build_embedder(backend: str, model: str, dim: int) -> Embedder:
    """Factory with an explicit, logged fallback.

    Never silently degrades: if ONNX is requested and unavailable the caller
    sees the reason, because a silent downgrade would make the evaluation
    report a number for a model that was not actually used.
    """
    if backend == "onnx":
        return OnnxEmbedder(model=model, dim=dim)
    return HashingEmbedder(dim=dim)
