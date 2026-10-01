"""End-to-end generation evaluation: answer, judge, calibrate, report.

Pipeline per sampled query:
    retrieve -> generate -> judge (faithfulness, relevance, context precision)

Budget shape (this is why the code looks the way it does)
---------------------------------------------------------
Measured Groq free tier: 8,000 tokens/minute, ~2,400 tokens per grounded answer.
Each judged query costs 1 generation + 3 judge calls. At ~3 model calls/minute of
budget, a 60-query judged run is roughly 80 calls and takes ~25 minutes of
wall-clock pacing. That is affordable once a day; a full 173-query run with claim
decomposition is not. So:

  * the run is SAMPLED and the sample size is printed on every number;
  * pacing is driven by the live /metrics token budget, not a fixed sleep;
  * a --limit flag exists so a smoke run costs a handful of calls.

Calibration
-----------
`human_labels.json` maps query_id -> {"faithful": 0|1, "relevant": 0|1}. Those
are hand-judged. The harness computes Cohen's kappa between the judge's
binarised verdict and the human label, and marks every metric UNRELIABLE when
kappa falls below the floor. No human labels -> metrics are reported
UNVALIDATED rather than presented as trustworthy.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from opspilot.config import get_settings
from opspilot.core import Role, get_logger, log_event
from opspilot.data.evalset import EvalQuery
from opspilot.eval.judge import Judge, MetricSummary, cohens_kappa
from opspilot.llm.providers import LLMProvider
from opspilot.rag.generator import GroundedGenerator
from opspilot.rag.retrieval import KnowledgeIndex, LexicalReranker, RetrievalFilter

log = get_logger("eval.generation")

ANSWERABLE_CLASSES = frozenset("BSLVDPM")


@dataclass(slots=True)
class GeneratedCase:
    query_id: str
    question: str
    difficulty: str
    answerable: bool
    answer: str
    abstained: bool
    evidence: list[str] = field(default_factory=list)
    citations: int = 0
    reference: str = ""
    usage: dict[str, Any] = field(default_factory=dict)
    latency_ms: float = 0.0


def _reference_answer(q: EvalQuery, index: KnowledgeIndex) -> str:
    """Gold text = the labelled spans themselves.

    Not a written-out ideal answer: that would be another generated artifact to
    trust. The graded spans are the ground truth we already committed to.
    """
    parts = []
    for qrel in q.qrels:
        if qrel.grade < 2:
            continue
        for c in index.chunks:
            if c.document_id == qrel.document_id and c.overlaps(qrel.char_start, qrel.char_end):
                parts.append(c.text)
                break
    return "\n\n".join(parts[:3])


def generate_cases(
    index: KnowledgeIndex, provider: LLMProvider, queries: list[EvalQuery], *, pace_fn=None
) -> list[GeneratedCase]:
    gen = GroundedGenerator(provider)
    reranker = LexicalReranker(bm25=index.bm25)
    settings = get_settings()
    out: list[GeneratedCase] = []

    for q in queries:
        if pace_fn:
            pace_fn()
        flt = RetrievalFilter(role=Role(q.role_scope), exclude_deprecated=True)
        t0 = time.perf_counter()
        hits = index.retrieve(
            q.text,
            arm="hybrid",
            flt=flt,
            candidate_k=settings.candidate_k,
            final_k=settings.final_k,
            rrf_k=settings.rrf_k,
            reranker=reranker,
        )
        result = gen.generate(q.text, hits)
        elapsed = (time.perf_counter() - t0) * 1000
        out.append(
            GeneratedCase(
                query_id=q.query_id,
                question=q.text,
                difficulty=q.difficulty,
                answerable=(q.difficulty in ANSWERABLE_CLASSES and bool(q.qrels)),
                answer=result.text,
                abstained=result.abstained,
                evidence=[h.chunk.text for h in hits],
                citations=len(result.citations),
                reference=_reference_answer(q, index),
                usage=result.usage,
                latency_ms=elapsed,
            )
        )
        log_event(
            log, "generated", query_id=q.query_id, abstained=result.abstained, ms=round(elapsed, 1)
        )
    return out


def judge_cases(judge: Judge, cases: list[GeneratedCase], *, pace_fn=None) -> dict[str, list]:
    scores: dict[str, list] = {"faithfulness": [], "answer_relevance": [], "context_precision": []}
    for c in cases:
        # An abstention has no claims to check, so faithfulness is vacuous.
        # Scoring it 1.0 would inflate the mean with free marks; it is skipped
        # and the n reflects that.
        if not c.abstained and c.evidence:
            if pace_fn:
                pace_fn()
            scores["faithfulness"].append(judge.faithfulness(c.query_id, c.answer, c.evidence))
        if pace_fn:
            pace_fn()
        scores["answer_relevance"].append(
            judge.answer_relevance(c.query_id, c.question, c.answer, c.answerable)
        )
        if c.evidence:
            if pace_fn:
                pace_fn()
            scores["context_precision"].append(
                judge.context_precision(c.query_id, c.question, c.evidence)
            )
    return scores


def calibrate(
    scores: dict[str, list], human_labels: dict[str, dict], threshold: float = 0.7
) -> dict[str, tuple[float | None, int, bool]]:
    """Cohen's kappa between the judge's binarised verdict and human labels."""
    mapping = {"faithfulness": "faithful", "answer_relevance": "relevant"}
    out: dict[str, tuple[float | None, int, bool]] = {}
    for metric, key in mapping.items():
        pairs = [
            (1 if s.score >= threshold else 0, int(human_labels[s.query_id][key]))
            for s in scores.get(metric, [])
            if s.ok
            and isinstance(human_labels.get(s.query_id), dict)
            and key in human_labels[s.query_id]
        ]
        if not pairs:
            out[metric] = (None, 0, False)
            continue
        j, h = zip(*pairs, strict=True)
        k = cohens_kappa(list(j), list(h))
        out[metric] = (k, len(pairs), k is None)
    out.setdefault("context_precision", (None, 0, False))
    return out


def save_cases(cases: list[GeneratedCase], scores: dict[str, list], path: Path) -> Path:
    """Persist answers + judge verdicts so human labelling costs no further quota."""
    path.parent.mkdir(parents=True, exist_ok=True)
    judge_by_metric = {m: {s.query_id: s.score for s in v} for m, v in scores.items()}
    rows = []
    for c in cases:
        rows.append(
            {
                "query_id": c.query_id,
                "difficulty": c.difficulty,
                "question": c.question,
                "answerable": c.answerable,
                "abstained": c.abstained,
                "citations": c.citations,
                "answer": c.answer,
                "evidence_preview": [e[:400] for e in c.evidence[:3]],
                "judge": {m: judge_by_metric.get(m, {}).get(c.query_id) for m in scores},
            }
        )
    path.write_text(json.dumps(rows, indent=2, ensure_ascii=False), encoding="utf-8")
    return path


def write_report(
    summaries: list[MetricSummary], cases: list[GeneratedCase], out_dir: Path, context: dict
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = {
        "context": context,
        "metrics": [s.to_json() for s in summaries],
        "cases": [
            {
                "query_id": c.query_id,
                "difficulty": c.difficulty,
                "abstained": c.abstained,
                "citations": c.citations,
                "answerable": c.answerable,
                "latency_ms": round(c.latency_ms, 1),
            }
            for c in cases
        ],
    }
    (out_dir / "generation_eval.json").write_text(
        json.dumps(payload, indent=2, default=str), encoding="utf-8"
    )

    n = len(cases)
    abstained = sum(1 for c in cases if c.abstained)
    answerable = sum(1 for c in cases if c.answerable)
    false_abstain = sum(1 for c in cases if c.answerable and c.abstained)
    correct_abstain = sum(1 for c in cases if not c.answerable and c.abstained)
    unanswerable = n - answerable

    L = [
        "# Generation Evaluation (LLM-as-judge)",
        "",
        "> **Sampled run.** Every number below carries its own `n`. Under the",
        "> measured Groq free tier a full-set judged evaluation is not affordable,",
        "> so this is a sample and says so (spec §38).",
        "",
        "## Run context",
        "",
        "| Key | Value |",
        "| --- | --- |",
    ]
    for k, v in context.items():
        L.append(f"| {k} | `{v}` |")

    L += [
        "",
        "## Judge metrics",
        "",
        "| Metric | Mean | 95% CI | n | failed | Cohen's kappa | Status |",
        "| --- | ---: | --- | ---: | ---: | ---: | --- |",
    ]
    for s in summaries:
        k = f"{s.kappa:.2f} (n={s.kappa_n})" if s.kappa is not None else "—"
        L.append(
            f"| {s.metric} | {s.mean:.3f} | [{s.ci_low:.3f}, {s.ci_high:.3f}] | "
            f"{s.n} | {s.failures} | {k} | {s.label} |"
        )

    L += [
        "",
        "### How to read the Status column",
        "",
        f"- **reliable** — judge/human agreement above kappa {0.40:.2f}.",
        "- **UNRELIABLE** — the judge disagrees with a human too often for the",
        "  mean to carry information. The number is printed, not trusted.",
        "- **UNVALIDATED** — no human labels exist for this metric yet. Add them",
        "  to `evaluation/human_labels.json`.",
        "",
        "## Abstention behaviour on this sample",
        "",
        "| Quantity | Value |",
        "| --- | ---: |",
        f"| cases | {n} |",
        f"| answerable | {answerable} |",
        f"| unanswerable | {unanswerable} |",
        f"| abstained (total) | {abstained} |",
        f"| **false abstention** (answerable but refused) | "
        f"{false_abstain}/{answerable}"
        f"{f' = {false_abstain / answerable:.1%}' if answerable else ''} |",
        f"| **correct abstention** (unanswerable, refused) | "
        f"{correct_abstain}/{unanswerable}"
        f"{f' = {correct_abstain / unanswerable:.1%}' if unanswerable else ''} |",
        "",
        "## Known limitations of these numbers",
        "",
        "1. **Sentence-level grounding, not claim decomposition.** Decomposing",
        "   each answer into atomic claims multiplies judge calls per answer and",
        "   is unaffordable at 8,000 tokens/minute. Faithfulness here is",
        "   therefore coarser than a RAGAS-style claim-level score.",
        "2. **Faithfulness skips abstentions** — an abstention has no claims, and",
        "   scoring it 1.0 would inflate the mean with free marks.",
        "3. **Reference answers are the labelled gold spans**, not hand-written",
        "   ideal answers, so context recall measures span coverage.",
        "4. **The judge shares a provider with the generator** (both on Groq),",
        "   though a different model family. Full independence would mean a",
        "   different vendor.",
    ]
    path = out_dir / "generation_eval.md"
    path.write_text("\n".join(L), encoding="utf-8")
    return path
