"""Run the sampled, judged generation evaluation.

    python scripts/run_generation_eval.py --limit 8          # smoke run
    python scripts/run_generation_eval.py --limit 40         # real run (~25 min)

Pacing is driven by the live token budget, not a fixed sleep, so the run adapts
to whatever the account's real limits turn out to be.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from opspilot.config import REPO_ROOT, get_settings
from opspilot.core import configure_logging
from opspilot.data.corpus import generate_documents
from opspilot.data.evalset import build_queries
from opspilot.eval.generation import (
    calibrate,
    generate_cases,
    judge_cases,
    save_cases,
    write_report,
)
from opspilot.eval.judge import Judge, summarise
from opspilot.llm.providers import build_provider
from opspilot.rag.chunking import chunk_corpus
from opspilot.rag.embeddings import HashingEmbedder, build_embedder
from opspilot.rag.retrieval import KnowledgeIndex

HUMAN_LABELS = REPO_ROOT / "evaluation" / "human_labels.json"


def make_pacer(provider, settings, quiet: bool = False):
    """Wait until the in-process token budget has headroom for another call."""
    budget = getattr(getattr(provider, "inner", None), "budget", None)

    def pace() -> None:
        if budget is None:
            return
        need = 3000
        for _ in range(30):
            snap = budget.snapshot()
            if (
                snap["tokens_this_minute"] + need <= snap["tpm_limit"]
                and snap["requests_this_minute"] < 25
            ):
                return
            if not quiet:
                print(
                    f"    … budget wait ({snap['tokens_this_minute']}/"
                    f"{snap['tpm_limit']} tpm, {snap['requests_this_minute']} rpm)",
                    flush=True,
                )
            time.sleep(10)

    return pace


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=8, help="number of queries to sample")
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--no-pace", action="store_true")
    args = ap.parse_args()

    settings = get_settings()
    configure_logging("WARNING")

    docs = generate_documents(seed=settings.seed)
    queries = build_queries(docs)
    chunks = chunk_corpus(docs, "structure")
    backend = settings.embedding_backend.value
    try:
        dim = settings.hashing_dim if backend == "hashing" else settings.embedding_dim
        embedder = build_embedder(backend, settings.embedding_model, dim)
    except Exception:
        embedder = HashingEmbedder(dim=settings.hashing_dim)
    index = KnowledgeIndex(chunks, embedder).build(use_qdrant=False)

    # Stratified sample: keep answerable and unanswerable both represented, or
    # the abstention columns in the report are meaningless.
    rng = random.Random(args.seed)
    answerable = [q for q in queries if q.qrels and q.difficulty in "BSLVDPM"]
    unanswerable = [q for q in queries if q.difficulty == "X"]
    n_un = max(2, args.limit // 4)
    sample = rng.sample(answerable, min(args.limit - n_un, len(answerable))) + rng.sample(
        unanswerable, min(n_un, len(unanswerable))
    )
    rng.shuffle(sample)

    provider = build_provider(settings)
    pace = (lambda: None) if args.no_pace else make_pacer(provider, settings)

    print(
        f"provider={provider.name} judge={settings.model_judge} "
        f"generator={settings.model_generator}"
    )
    print(f"sample={len(sample)} (answerable={len(sample) - n_un}, unanswerable~{n_un})\n")

    t0 = time.perf_counter()
    print("generating answers…", flush=True)
    cases = generate_cases(index, provider, sample, pace_fn=pace)

    print("judging…", flush=True)
    judge = Judge(provider)
    scores = judge_cases(judge, cases, pace_fn=pace)

    cases_path = save_cases(cases, scores, REPO_ROOT / "evaluation" / "judged_cases.json")
    print(f"cases saved for labelling: {cases_path}")

    human = {}
    if HUMAN_LABELS.exists():
        human = json.loads(HUMAN_LABELS.read_text(encoding="utf-8"))
    kappas = calibrate(scores, human)

    summaries = [summarise(m, s, *kappas.get(m, (None, 0))) for m, s in scores.items()]
    elapsed = time.perf_counter() - t0

    path = write_report(
        summaries,
        cases,
        REPO_ROOT / "artifacts" / "reports",
        {
            "generator": settings.model_generator,
            "judge": settings.model_judge,
            "judge_independent_of_generator": settings.model_judge != settings.model_generator,
            "embedder": embedder.name,
            "sample_size": len(cases),
            "human_labels_available": len(human),
            "judge_calls": judge.calls,
            "judge_failures": judge.failures,
            "wall_clock_seconds": round(elapsed, 1),
            "seed": settings.seed,
        },
    )

    print()
    for s in summaries:
        k = f"kappa={s.kappa:.2f}" if s.kappa is not None else "kappa=—"
        print(
            f"  {s.metric:<20} {s.mean:.3f}  [{s.ci_low:.3f},{s.ci_high:.3f}]  "
            f"n={s.n:<3} fail={s.failures}  {k}  {s.label}"
        )
    print(f"\njudge calls={judge.calls} failures={judge.failures} wall={elapsed:.0f}s")
    print(f"report: {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
