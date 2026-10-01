"""Reproducible retrieval experiments (spec §20).

Experiments 1-4 require ZERO LLM calls, so they run unlimited and offline. This
is deliberate: under the Groq free-tier ceiling (~55 grounded answers/day) the
only way to do rigorous, repeatable measurement is to put the rigour where it
costs nothing. Experiment 5 (query rewrite) is the one that needs a model and is
therefore run on a sampled subset.
"""

from __future__ import annotations

import json
import platform
import time
from dataclasses import dataclass, field
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

from opspilot.core import Role
from opspilot.data.evalset import EvalQuery
from opspilot.eval.metrics import (
    Comparison,
    QueryScores,
    aggregate,
    by_difficulty,
    paired_bootstrap,
    score_query,
)
from opspilot.rag.retrieval import (
    KnowledgeIndex,
    LexicalReranker,
    NullReranker,
    RetrievalFilter,
)


@dataclass(slots=True)
class ExperimentConfig:
    experiment_id: str
    label: str
    arm: str  # dense | sparse | hybrid
    reranker: str = "none"  # none | lexical | cross_encoder
    candidate_k: int = 30
    final_k: int = 10
    rrf_k: int = 60


@dataclass(slots=True)
class ExperimentResult:
    config: ExperimentConfig
    overall: dict[str, float]
    per_difficulty: dict[str, dict[str, float]]
    scores: list[QueryScores] = field(default_factory=list)
    latency_ms: dict[str, float] = field(default_factory=dict)


DEFAULT_EXPERIMENTS = [
    ExperimentConfig("E1", "Dense only", arm="dense"),
    ExperimentConfig("E2", "Sparse only (BM25)", arm="sparse"),
    ExperimentConfig("E3", "Hybrid (RRF)", arm="hybrid"),
    ExperimentConfig("E4", "Hybrid + reranker", arm="hybrid", reranker="lexical"),
]


def _make_reranker(name: str, index=None):
    if name == "lexical":
        return LexicalReranker(bm25=getattr(index, "bm25", None))
    if name == "cross_encoder":
        from opspilot.rag.retrieval import CrossEncoderReranker

        return CrossEncoderReranker()
    return NullReranker()


def _percentile(values: list[float], p: float) -> float:
    if not values:
        return 0.0
    s = sorted(values)
    i = min(len(s) - 1, round((p / 100.0) * (len(s) - 1)))
    return s[i]


def run_experiment(
    index: KnowledgeIndex, queries: list[EvalQuery], cfg: ExperimentConfig, k: int = 10
) -> ExperimentResult:
    reranker = _make_reranker(cfg.reranker, index)
    scores: list[QueryScores] = []
    lat: list[float] = []
    for q in queries:
        if q.difficulty == "X" or not q.qrels:
            continue  # unanswerable queries are scored by the abstention harness
        flt = RetrievalFilter(role=Role(q.role_scope))
        t0 = time.perf_counter()
        hits = index.retrieve(
            q.text,
            arm=cfg.arm,
            flt=flt,
            candidate_k=cfg.candidate_k,
            final_k=cfg.final_k,
            rrf_k=cfg.rrf_k,
            reranker=reranker if cfg.reranker != "none" else None,
        )
        lat.append((time.perf_counter() - t0) * 1000)
        scores.append(score_query(hits, q, index.chunks, k=k))
    return ExperimentResult(
        config=cfg,
        overall=aggregate(scores),
        per_difficulty=by_difficulty(scores),
        scores=scores,
        latency_ms={
            "mean": sum(lat) / len(lat) if lat else 0.0,
            "p50": _percentile(lat, 50),
            "p95": _percentile(lat, 95),
            "p99": _percentile(lat, 99),
        },
    )


def run_suite(
    index: KnowledgeIndex,
    queries: list[EvalQuery],
    configs: list[ExperimentConfig] | None = None,
    k: int = 10,
) -> list[ExperimentResult]:
    return [run_experiment(index, queries, c, k=k) for c in (configs or DEFAULT_EXPERIMENTS)]


def compare_all(
    results: list[ExperimentResult], metric: str = "ndcg_at_k"
) -> list[tuple[str, str, Comparison]]:
    out = []
    for i in range(len(results)):
        for j in range(i + 1, len(results)):
            a, b = results[i], results[j]
            out.append(
                (
                    a.config.experiment_id,
                    b.config.experiment_id,
                    paired_bootstrap(a.scores, b.scores, metric=metric),
                )
            )
    return out


# --------------------------------------------------------------------------
# reporting
# --------------------------------------------------------------------------
DIFFICULTY_LEGEND = {
    "B": "baseline canonical phrasing (tier1, optimistic)",
    "L": "lexical-anchored (error codes) - BM25 should win",
    "S": "semantic paraphrase (disjoint vocabulary) - dense should win",
    "V": "version/recency - stale revisions are wrong answers",
    "P": "platform variant",
    "D": "near-duplicate distractor (different product)",
    "M": "multi-hop (two documents required)",
    "A": "access-gated (role-scoped)",
}


def write_report(
    results: list[ExperimentResult], comparisons, out_dir: Path, context: dict[str, Any]
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).isoformat()

    payload = {
        "generated_at": stamp,
        "context": context,
        "experiments": [
            {
                "experiment_id": r.config.experiment_id,
                "label": r.config.label,
                "arm": r.config.arm,
                "reranker": r.config.reranker,
                "overall": r.overall,
                "per_difficulty": r.per_difficulty,
                "latency_ms": r.latency_ms,
            }
            for r in results
        ],
        "comparisons": [
            {
                "a": a,
                "b": b,
                "metric": c.metric,
                "delta": c.delta,
                "ci_low": c.ci_low,
                "ci_high": c.ci_high,
                "significant": c.significant,
                "verdict": c.verdict,
            }
            for a, b, c in comparisons
        ],
    }
    (out_dir / "retrieval_experiments.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    lines: list[str] = []
    A = lines.append
    A("# Retrieval Experiment Report")
    A("")
    A(f"Generated: `{stamp}`")
    A("")
    A("> Every number below was produced by an actual run of the code in this")
    A("> repository. Nothing here is estimated or illustrative (spec §38).")
    A("")
    A("## Run context")
    A("")
    A("| Key | Value |")
    A("| --- | --- |")
    for key, value in context.items():
        A(f"| {key} | `{value}` |")
    A(f"| python | `{platform.python_version()}` |")
    A(f"| platform | `{platform.platform()}` |")
    A("")

    A("## Overall (k=10)")
    A("")
    A("| Experiment | Arm | Reranker | Recall@10 | Precision@10 | MRR | nDCG@10 | p95 ms | n |")
    A("| --- | --- | --- | ---: | ---: | ---: | ---: | ---: | ---: |")
    for r in results:
        o = r.overall
        A(
            f"| {r.config.experiment_id} {r.config.label} | {r.config.arm} | "
            f"{r.config.reranker} | {o['recall']:.3f} | {o['precision']:.3f} | "
            f"{o['mrr']:.3f} | {o['ndcg']:.3f} | {r.latency_ms['p95']:.1f} | {int(o['n'])} |"
        )
    A("")

    A("## nDCG@10 by difficulty class")
    A("")
    A("This is the table that makes the experiment falsifiable. An aggregate")
    A("number can hide a technique that wins on one class and loses on another.")
    A("")
    classes = sorted({d for r in results for d in r.per_difficulty})
    A("| Experiment | " + " | ".join(classes) + " |")
    A("| --- | " + " | ".join("---:" for _ in classes) + " |")
    for r in results:
        row = [
            f"{r.per_difficulty.get(c, {}).get('ndcg', float('nan')):.3f}"
            if c in r.per_difficulty
            else "-"
            for c in classes
        ]
        A(f"| {r.config.experiment_id} | " + " | ".join(row) + " |")
    A("")
    A("Legend:")
    A("")
    for cls in classes:
        A(f"- **{cls}** — {DIFFICULTY_LEGEND.get(cls, 'unclassified')}")
    A("")

    A("## Pairwise significance (paired bootstrap, 10,000 resamples, 95% CI)")
    A("")
    A("House rule (audit G3): an improvement whose CI crosses zero is reported as")
    A("*no measurable difference*, never as a win.")
    A("")
    A("| A | B | Δ nDCG@10 (B−A) | 95% CI | Verdict |")
    A("| --- | --- | ---: | --- | --- |")
    for a, b, c in comparisons:
        A(
            f"| {a} | {b} | {c.delta:+.4f} | [{c.ci_low:+.4f}, {c.ci_high:+.4f}] | "
            f"{'**' + c.verdict + '**' if c.significant else c.verdict} |"
        )
    A("")

    path = out_dir / "retrieval_experiments.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
