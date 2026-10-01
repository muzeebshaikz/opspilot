"""Generate the synthetic corpus, chunk it, build the index, run experiments.

    python -m scripts.build_index [--chunker structure|fixed] [--embedder hashing|onnx]

Everything is deterministic given the seed, so re-running reproduces the corpus
hash and the dataset hash recorded in the report.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opspilot.config import DATA_DIR, REPO_ROOT, get_settings
from opspilot.core import configure_logging, get_logger, log_event
from opspilot.data.corpus import generate_documents, write_documents
from opspilot.data.evalset import build_queries, write_queries
from opspilot.eval.experiments import (
    DEFAULT_EXPERIMENTS,
    compare_all,
    run_suite,
    write_report,
)
from opspilot.rag.chunking import chunk_corpus
from opspilot.rag.embeddings import HashingEmbedder, build_embedder
from opspilot.rag.retrieval import KnowledgeIndex

log = get_logger("build")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--chunker", default="structure", choices=["structure", "fixed"])
    ap.add_argument("--embedder", default=None, choices=["hashing", "onnx"])
    ap.add_argument("--no-qdrant", action="store_true")
    ap.add_argument("--k", type=int, default=10)
    args = ap.parse_args()

    settings = get_settings()
    configure_logging(settings.log_level)
    backend = args.embedder or settings.embedding_backend.value

    t0 = time.perf_counter()

    # ---- 1. corpus ------------------------------------------------------
    docs = generate_documents(seed=settings.seed)
    manifest = write_documents(DATA_DIR / "generated" / "documents", docs)
    log_event(
        log,
        "corpus generated",
        documents=len(docs),
        corpus_hash=manifest["corpus_hash"][:12],
        chars=manifest["total_chars"],
    )

    # ---- 2. evaluation set ----------------------------------------------
    queries = build_queries(docs)
    qmanifest = write_queries(DATA_DIR / "eval", queries)
    log_event(
        log,
        "eval set built",
        queries=len(queries),
        dataset_hash=qmanifest["dataset_hash"][:12],
        by_difficulty=qmanifest["by_difficulty"],
    )

    # ---- 3. chunking -----------------------------------------------------
    chunks = chunk_corpus(docs, strategy=args.chunker)
    sizes = [len(c.text) for c in chunks]
    log_event(
        log,
        "chunked",
        strategy=args.chunker,
        chunks=len(chunks),
        mean_chars=round(sum(sizes) / len(sizes), 1),
        max_chars=max(sizes),
        min_chars=min(sizes),
    )

    # ---- 4. embedder ------------------------------------------------------
    try:
        dim = settings.hashing_dim if backend == "hashing" else settings.embedding_dim
        embedder = build_embedder(backend, settings.embedding_model, dim)
    except Exception as exc:  # model files unavailable -> explicit, logged fallback
        log_event(
            log,
            "embedder unavailable, falling back to hashing",
            requested=backend,
            error=str(exc)[:200],
        )
        backend = "hashing (fallback)"
        embedder = HashingEmbedder(dim=settings.hashing_dim)

    # ---- 5. index ---------------------------------------------------------
    index = KnowledgeIndex(
        chunks,
        embedder,
        collection=settings.collection_name,
        qdrant_path=settings.qdrant_path,
        qdrant_url=settings.qdrant_url,
    )
    t_idx = time.perf_counter()
    index.build(use_qdrant=not args.no_qdrant)
    index_s = time.perf_counter() - t_idx
    log_event(
        log,
        "index built",
        seconds=round(index_s, 2),
        embedder=embedder.name,
        qdrant=not args.no_qdrant,
    )

    # ---- 6. experiments ---------------------------------------------------
    results = run_suite(index, queries, DEFAULT_EXPERIMENTS, k=args.k)
    comparisons = compare_all(results)

    context = {
        "corpus_hash": manifest["corpus_hash"][:16],
        "dataset_hash": qmanifest["dataset_hash"][:16],
        "documents": manifest["document_count"],
        "chunks": len(chunks),
        "chunker": args.chunker,
        "embedder": embedder.name,
        "embedding_backend": backend,
        "queries_scored": sum(1 for q in queries if q.qrels),
        "k": args.k,
        "index_build_seconds": round(index_s, 2),
        "seed": settings.seed,
    }
    report = write_report(results, comparisons, REPO_ROOT / "artifacts" / "reports", context)
    index.close()

    print("\n" + "=" * 78)
    for r in results:
        o = r.overall
        print(
            f"{r.config.experiment_id} {r.config.label:<24} "
            f"nDCG@{args.k}={o['ndcg']:.3f}  Recall={o['recall']:.3f}  "
            f"MRR={o['mrr']:.3f}  p95={r.latency_ms['p95']:.1f}ms"
        )
    print("=" * 78)
    print(f"report: {report}")
    print(f"total:  {time.perf_counter() - t0:.1f}s")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
