"""Lazily build the retrieval index inside a knowledge MCP server process.

Each MCP server is a separate OS process, so the knowledge server cannot share
the API process's in-memory index. It builds its own on first use. This is a
real cost of the process boundary and is measured rather than hidden: the first
`search_knowledge` call over MCP pays index construction.

Production answer: point both processes at a Qdrant *server* so the index is
shared and neither process builds anything. That is a URL change
(`OPSPILOT_QDRANT_URL`), not a code change.
"""

from __future__ import annotations

from functools import lru_cache
from typing import Any

from opspilot.config import get_settings
from opspilot.core import get_logger, log_event
from opspilot.tools import set_index_provider

log = get_logger("mcp.index")


@lru_cache(maxsize=1)
def _build_index() -> Any:
    import time

    from opspilot.data.corpus import generate_documents
    from opspilot.rag.chunking import chunk_corpus
    from opspilot.rag.embeddings import HashingEmbedder, build_embedder
    from opspilot.rag.retrieval import KnowledgeIndex

    s = get_settings()
    t0 = time.perf_counter()
    docs = generate_documents(seed=s.seed)
    chunks = chunk_corpus(docs, "structure")
    backend = s.embedding_backend.value
    try:
        dim = s.hashing_dim if backend == "hashing" else s.embedding_dim
        embedder = build_embedder(backend, s.embedding_model, dim)
    except Exception as exc:
        log_event(log, "embedder fallback in MCP process", error=str(exc)[:120])
        embedder = HashingEmbedder(dim=s.hashing_dim)
    index = KnowledgeIndex(chunks, embedder).build(use_qdrant=False)
    log_event(
        log,
        "mcp knowledge index built",
        chunks=len(chunks),
        seconds=round(time.perf_counter() - t0, 2),
    )
    return index


def install_index_provider() -> None:
    set_index_provider(_build_index)
