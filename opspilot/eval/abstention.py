"""Abstention calibration (audit G6).

Spec §10 requires the system to abstain when evidence is insufficient, and §19
lists an out-of-domain test set. On its own that is trivially gameable: a system
that abstains on everything scores perfectly on the unanswerable set.

Abstention is therefore scored as a confusion matrix over two PAIRED sets:

  answerable   (B/S/L/V/D/P/M classes)  -> answering is correct
  unanswerable (X class, and A-class    -> abstaining is correct
                queries whose gold
                document the asking
                role may not see)

and the two error rates traded off explicitly:

  false_abstention_rate      refused although evidence existed  (usability harm)
  hallucination_risk_rate    answered with no supporting evidence (safety harm)

The operating threshold is then CHOSEN FROM THE SWEEP, not guessed. Every number
produced here is retrieval-only: no LLM call, no quota.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from opspilot.core import Role
from opspilot.data.evalset import EvalQuery
from opspilot.rag.retrieval import KnowledgeIndex, LexicalReranker, RetrievalFilter

ANSWERABLE = frozenset("BSLVDPM")


@dataclass(slots=True)
class AbstentionPoint:
    threshold: float
    true_abstain: int  # unanswerable, abstained  (correct)
    false_abstain: int  # answerable, abstained    (usability failure)
    true_answer: int  # answerable, answered     (correct)
    false_answer: int  # unanswerable, answered   (safety failure)

    @property
    def n_unanswerable(self) -> int:
        return self.true_abstain + self.false_answer

    @property
    def n_answerable(self) -> int:
        return self.true_answer + self.false_abstain

    @property
    def false_abstention_rate(self) -> float:
        return self.false_abstain / self.n_answerable if self.n_answerable else 0.0

    @property
    def hallucination_risk_rate(self) -> float:
        return self.false_answer / self.n_unanswerable if self.n_unanswerable else 0.0

    @property
    def balanced_accuracy(self) -> float:
        tpr = self.true_answer / self.n_answerable if self.n_answerable else 0.0
        tnr = self.true_abstain / self.n_unanswerable if self.n_unanswerable else 0.0
        return (tpr + tnr) / 2

    def to_json(self) -> dict:
        return {
            "threshold": round(self.threshold, 3),
            "true_abstain": self.true_abstain,
            "false_abstain": self.false_abstain,
            "true_answer": self.true_answer,
            "false_answer": self.false_answer,
            "false_abstention_rate": round(self.false_abstention_rate, 4),
            "hallucination_risk_rate": round(self.hallucination_risk_rate, 4),
            "balanced_accuracy": round(self.balanced_accuracy, 4),
        }


def _is_answerable(q: EvalQuery) -> bool:
    if q.difficulty == "X":
        return False
    if q.difficulty == "A" and not q.qrels:
        return False  # answer exists but this role may not see it
    return q.difficulty in ANSWERABLE or bool(q.qrels)


def top_evidence_scores(
    index: KnowledgeIndex, queries: list[EvalQuery], candidate_k: int = 30, final_k: int = 6
) -> dict[str, float]:
    """Best reranker score per query. Computed once, swept many times."""
    reranker = LexicalReranker(bm25=index.bm25)
    out: dict[str, float] = {}
    for q in queries:
        hits = index.retrieve(
            q.text,
            arm="hybrid",
            flt=RetrievalFilter(role=Role(q.role_scope), exclude_deprecated=True),
            candidate_k=candidate_k,
            final_k=final_k,
            reranker=reranker,
        )
        out[q.query_id] = max((h.rerank_score or 0.0) for h in hits) if hits else 0.0
    return out


def sweep(
    index: KnowledgeIndex, queries: list[EvalQuery], thresholds: list[float] | None = None
) -> list[AbstentionPoint]:
    scores = top_evidence_scores(index, queries)
    labels = {q.query_id: _is_answerable(q) for q in queries}
    grid = thresholds or [round(x * 0.02, 3) for x in range(0, 101)]
    points: list[AbstentionPoint] = []
    for t in grid:
        p = AbstentionPoint(t, 0, 0, 0, 0)
        for qid, score in scores.items():
            answered = score >= t
            answerable = labels[qid]
            if answerable and answered:
                p.true_answer += 1
            elif answerable and not answered:
                p.false_abstain += 1
            elif not answerable and answered:
                p.false_answer += 1
            else:
                p.true_abstain += 1
        points.append(p)
    return points


def choose_threshold(
    points: list[AbstentionPoint], max_hallucination_risk: float = 0.15
) -> AbstentionPoint:
    """Pick the operating point.

    Policy: cap the SAFETY error first, then minimise the usability error within
    that cap. Answering confidently with no evidence is the worse failure for an
    IT copilot that people act on, so it gets the hard constraint.
    """
    feasible = [p for p in points if p.hallucination_risk_rate <= max_hallucination_risk]
    if not feasible:
        return max(points, key=lambda p: p.balanced_accuracy)
    return min(feasible, key=lambda p: (p.false_abstention_rate, -p.balanced_accuracy))


def write_report(
    points: list[AbstentionPoint], chosen: AbstentionPoint, out_dir: Path, context: dict
) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "abstention_calibration.json").write_text(
        json.dumps(
            {
                "context": context,
                "chosen": chosen.to_json(),
                "sweep": [p.to_json() for p in points],
            },
            indent=2,
        ),
        encoding="utf-8",
    )

    lines = [
        "# Abstention Calibration",
        "",
        "Measured, retrieval-only (no LLM calls). Two error types are traded off",
        "explicitly rather than reporting a single abstention 'accuracy'.",
        "",
        "| Key | Value |",
        "| --- | --- |",
    ]
    for k, v in context.items():
        lines.append(f"| {k} | `{v}` |")
    lines += [
        "",
        f"**Chosen operating point: threshold = {chosen.threshold}**",
        "",
        f"- false-abstention rate (refused with evidence available): "
        f"**{chosen.false_abstention_rate:.1%}**",
        f"- hallucination-risk rate (answered with no evidence): "
        f"**{chosen.hallucination_risk_rate:.1%}**",
        f"- balanced accuracy: **{chosen.balanced_accuracy:.3f}**",
        f"- n answerable = {chosen.n_answerable}, n unanswerable = {chosen.n_unanswerable}",
        "",
        "Selection policy: cap hallucination risk first, then minimise false",
        "abstention within that cap. Answering with no evidence is the worse",
        "failure for a copilot whose output people act on.",
        "",
        "## Sweep",
        "",
        "| threshold | false-abstention | hallucination-risk | balanced acc |",
        "| ---: | ---: | ---: | ---: |",
    ]
    for p in points:
        mark = " **<-- chosen**" if p.threshold == chosen.threshold else ""
        lines.append(
            f"| {p.threshold:.2f} | {p.false_abstention_rate:.3f} | "
            f"{p.hallucination_risk_rate:.3f} | {p.balanced_accuracy:.3f}{mark} |"
        )
    path = out_dir / "abstention_calibration.md"
    path.write_text("\n".join(lines), encoding="utf-8")
    return path
