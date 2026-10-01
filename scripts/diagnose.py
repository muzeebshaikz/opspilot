"""Inspect what retrieval actually returns for a few queries, per difficulty class."""

from __future__ import annotations

import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opspilot.config import get_settings
from opspilot.core import Role
from opspilot.data.corpus import generate_documents
from opspilot.data.evalset import build_queries
from opspilot.eval.metrics import grade_of, relevant_chunk_count, score_query
from opspilot.rag.chunking import chunk_corpus
from opspilot.rag.embeddings import HashingEmbedder
from opspilot.rag.retrieval import KnowledgeIndex, LexicalReranker, RetrievalFilter

s = get_settings()
docs = generate_documents(seed=s.seed)
queries = build_queries(docs)
chunks = chunk_corpus(docs, "structure")
index = KnowledgeIndex(chunks, HashingEmbedder(dim=s.hashing_dim)).build(use_qdrant=False)

by_diff = defaultdict(list)
for q in queries:
    by_diff[q.difficulty].append(q)

print(f"chunks={len(chunks)} docs={len(docs)} queries={len(queries)}\n")

for diff in ("B", "L", "S", "V", "D", "M", "P"):
    qs = by_diff.get(diff, [])
    if not qs:
        continue
    q = qs[0]
    print("=" * 96)
    print(f"[{diff}] {q.text}")
    print(f"     gold docs: {[r.document_id for r in q.qrels if r.grade > 0]}")
    print(f"     relevant chunks in corpus: {relevant_chunk_count(chunks, q)}")
    for arm, rr in (
        ("dense", None),
        ("sparse", None),
        ("hybrid", None),
        ("hybrid", LexicalReranker(bm25=index.bm25)),
    ):
        hits = index.retrieve(
            q.text,
            arm=arm,
            flt=RetrievalFilter(role=Role(q.role_scope)),
            candidate_k=30,
            final_k=10,
            reranker=rr,
        )
        sc = score_query(hits, q, chunks, k=10)
        tag = f"{arm}+rerank" if rr else arm
        top = [
            (h.chunk.document_id[:9], h.chunk.heading[:18], grade_of(h.chunk, q)) for h in hits[:4]
        ]
        print(
            f"  {tag:<16} nDCG={sc.ndcg_at_k:.3f} R={sc.recall_at_k:.2f} "
            f"MRR={sc.reciprocal_rank:.2f} | {top}"
        )

# where are the gold chunks?
print("\n" + "=" * 96)
q = by_diff["B"][0]
gold_doc = next(r.document_id for r in q.qrels if r.grade > 0)
print(f"chunks belonging to gold doc {gold_doc} for query: {q.text}")
for c in chunks:
    if c.document_id == gold_doc:
        print(
            f"  {c.chunk_id} [{c.char_start:>5}-{c.char_end:<5}] "
            f"head={c.heading[:28]:<28} grade={grade_of(c, q)} len={len(c.text)}"
        )
print("gold spans:", [(r.document_id, r.char_start, r.char_end, r.grade) for r in q.qrels])
