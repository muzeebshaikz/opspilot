"""Sweep the evidence threshold and pick an operating point from measured data."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opspilot.config import REPO_ROOT, get_settings
from opspilot.data.corpus import generate_documents
from opspilot.data.evalset import build_queries
from opspilot.eval.abstention import choose_threshold, sweep, write_report
from opspilot.rag.chunking import chunk_corpus
from opspilot.rag.embeddings import HashingEmbedder
from opspilot.rag.retrieval import KnowledgeIndex

s = get_settings()
docs = generate_documents(seed=s.seed)
queries = build_queries(docs)
chunks = chunk_corpus(docs, "structure")
index = KnowledgeIndex(chunks, HashingEmbedder(dim=s.hashing_dim)).build(use_qdrant=False)

points = sweep(index, queries)
chosen = choose_threshold(points, max_hallucination_risk=0.15)
path = write_report(
    points,
    chosen,
    REPO_ROOT / "artifacts" / "reports",
    {
        "documents": len(docs),
        "chunks": len(chunks),
        "queries": len(queries),
        "embedder": "hashing-v1",
        "reranker": "lexical-v1",
        "seed": s.seed,
    },
)
print(f"chosen threshold = {chosen.threshold}")
print(f"  false-abstention   {chosen.false_abstention_rate:.1%}  (n={chosen.n_answerable})")
print(f"  hallucination-risk {chosen.hallucination_risk_rate:.1%}  (n={chosen.n_unanswerable})")
print(f"  balanced accuracy  {chosen.balanced_accuracy:.3f}")
print(f"report: {path}")
