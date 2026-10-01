"""LLM-as-judge harness for generation quality (spec §19).

Metrics: faithfulness, answer relevance, context precision, context recall.

Three rules make these numbers worth printing, and without them the scores are
decoration:

1. **The judge is a different model family from the generator.** Generator is
   `openai/gpt-oss-120b`; judge is `qwen/qwen3.8-27b`. A model grading its own
   output is self-evaluation, and a different *size* of the same family is only
   marginally better (audit G5).

2. **Every judge metric is published with Cohen's kappa** against human labels.
   A faithfulness score of 0.91 at kappa 0.34 means the judge and a human agree
   barely more than chance, so the 0.91 is meaningless. The harness refuses to
   present a metric as reliable when kappa is below the stated floor.

3. **The sample size is stated and the CI is reported.** Under the measured Groq
   free tier (8,000 tokens/minute) a full-set judged evaluation is not
   affordable, so this runs on a SAMPLE and says so. Claiming a full-set number
   from a sampled run would be exactly the fabrication spec §38 forbids.

Claim decomposition is deliberately avoided: it multiplies judge calls per answer
by the number of claims, which at ~3 answers/minute of budget would make a
60-question evaluation take hours. Sentence-level grounding is scored instead,
in a single call per answer, and the weaker granularity is disclosed.
"""

from __future__ import annotations

import json
import re
import statistics
import time
from dataclasses import asdict, dataclass, field
from typing import Any

from opspilot.config import get_settings
from opspilot.core import get_logger
from opspilot.llm.providers import LLMMessage, LLMProvider

log = get_logger("eval.judge")

KAPPA_FLOOR = 0.40  # below this, a judge metric is reported as UNRELIABLE


# --------------------------------------------------------------------------
# prompts -- each returns strict JSON and scores ONE dimension
# --------------------------------------------------------------------------
FAITHFULNESS_PROMPT = """You are grading whether an ANSWER is supported by EVIDENCE.

Rules:
- Judge ONLY support by the evidence. A claim can be true in the world and still
  unsupported here; that counts as unsupported.
- Ignore style, tone and helpfulness.
- Generic hedging ("contact IT if this persists") is not a factual claim; skip it.

Return JSON only:
{"supported": <int>, "unsupported": <int>, "score": <0.0-1.0>,
 "unsupported_claims": ["..."]}

score = supported / (supported + unsupported), or 1.0 if there are no claims."""

RELEVANCE_PROMPT = """You are grading whether an ANSWER addresses the QUESTION.

Ignore correctness and whether it is well-supported; grade only whether it
answers what was asked.

1.0 = fully answers it
0.5 = partially answers, or answers a related but different question
0.0 = does not address it

A refusal or abstention scores 0.0 for relevance ONLY if the question was
answerable; if the question could not be answered from the evidence, an
abstention is fully relevant and scores 1.0.

Return JSON only: {"score": <0.0-1.0>, "reason": "<one sentence>"}"""

CONTEXT_PRECISION_PROMPT = """You are grading RETRIEVED PASSAGES against a QUESTION.

For each numbered passage decide whether it is useful for answering the question.

Return JSON only:
{"useful": [<passage numbers>], "not_useful": [<passage numbers>],
 "score": <0.0-1.0>}

score = count(useful) / total passages."""

CONTEXT_RECALL_PROMPT = """You are grading whether RETRIEVED PASSAGES contain
everything needed for the REFERENCE ANSWER.

Return JSON only:
{"covered": <int>, "missing": <int>, "score": <0.0-1.0>,
 "missing_points": ["..."]}

score = covered / (covered + missing)."""


@dataclass(slots=True)
class JudgeScore:
    metric: str
    query_id: str
    score: float
    raw: dict[str, Any] = field(default_factory=dict)
    ok: bool = True
    error: str | None = None


@dataclass(slots=True)
class MetricSummary:
    metric: str
    n: int
    mean: float
    ci_low: float
    ci_high: float
    failures: int
    kappa: float | None = None
    kappa_n: int = 0

    @property
    def reliable(self) -> bool:
        return self.kappa is not None and self.kappa >= KAPPA_FLOOR

    degenerate: bool = False

    @property
    def label(self) -> str:
        if self.degenerate:
            return f"UNTESTED ({self.kappa_n} labels, no negative examples)"
        if self.kappa is None:
            return "UNVALIDATED (no human labels)"
        return "reliable" if self.reliable else f"UNRELIABLE (kappa={self.kappa:.2f})"

    def to_json(self) -> dict:
        return {**asdict(self), "reliable": self.reliable, "label": self.label}


def _extract_json(text: str) -> dict:
    """Judges wrap JSON in prose or fences often enough to handle it here."""
    if not text:
        return {}
    fenced = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.S)
    candidate = fenced.group(1) if fenced else None
    if candidate is None:
        start, end = text.find("{"), text.rfind("}")
        candidate = text[start : end + 1] if start >= 0 and end > start else None
    if not candidate:
        return {}
    try:
        return json.loads(candidate)
    except json.JSONDecodeError:
        return {}


class Judge:
    """Single-call-per-dimension judge with budget-aware pacing."""

    def __init__(self, provider: LLMProvider, model: str | None = None, pace_s: float = 0.0):
        self.provider = provider
        self.settings = get_settings()
        self.model = model or self.settings.model_judge
        self.pace_s = pace_s
        self.calls = 0
        self.failures = 0

    def _ask(self, system: str, user: str, metric: str, query_id: str) -> JudgeScore:
        if self.pace_s:
            time.sleep(self.pace_s)
        try:
            self.calls += 1
            resp = self.provider.complete(
                [LLMMessage("system", system), LLMMessage("user", user)],
                model=self.model,
                temperature=0.0,
                max_tokens=700,
            )
            data = _extract_json(resp.text)
            if "score" not in data:
                self.failures += 1
                return JudgeScore(
                    metric,
                    query_id,
                    0.0,
                    {},
                    ok=False,
                    error=f"unparseable: {(resp.text or '')[:120]}",
                )
            return JudgeScore(metric, query_id, max(0.0, min(1.0, float(data["score"]))), data)
        except Exception as exc:
            self.failures += 1
            return JudgeScore(metric, query_id, 0.0, {}, ok=False, error=str(exc)[:160])

    # -- dimensions --------------------------------------------------------
    def faithfulness(self, query_id: str, answer: str, evidence: list[str]) -> JudgeScore:
        ev = "\n\n".join(f"[{i}] {e[:900]}" for i, e in enumerate(evidence, 1))
        return self._ask(
            FAITHFULNESS_PROMPT, f"EVIDENCE:\n{ev}\n\nANSWER:\n{answer}", "faithfulness", query_id
        )

    def answer_relevance(
        self, query_id: str, question: str, answer: str, answerable: bool
    ) -> JudgeScore:
        return self._ask(
            RELEVANCE_PROMPT,
            f"QUESTION:\n{question}\n\nANSWER:\n{answer}\n\n"
            f"WAS THIS ANSWERABLE FROM THE KNOWLEDGE BASE? "
            f"{'yes' if answerable else 'no'}",
            "answer_relevance",
            query_id,
        )

    def context_precision(self, query_id: str, question: str, evidence: list[str]) -> JudgeScore:
        ev = "\n\n".join(f"[{i}] {e[:600]}" for i, e in enumerate(evidence, 1))
        return self._ask(
            CONTEXT_PRECISION_PROMPT,
            f"QUESTION:\n{question}\n\nRETRIEVED PASSAGES:\n{ev}",
            "context_precision",
            query_id,
        )

    def context_recall(self, query_id: str, reference: str, evidence: list[str]) -> JudgeScore:
        ev = "\n\n".join(f"[{i}] {e[:600]}" for i, e in enumerate(evidence, 1))
        return self._ask(
            CONTEXT_RECALL_PROMPT,
            f"REFERENCE ANSWER:\n{reference}\n\nRETRIEVED PASSAGES:\n{ev}",
            "context_recall",
            query_id,
        )


# --------------------------------------------------------------------------
# statistics
# --------------------------------------------------------------------------
def bootstrap_ci(
    values: list[float], resamples: int = 5000, seed: int = 20260930, alpha: float = 0.05
) -> tuple[float, float]:
    import random

    if not values:
        return (0.0, 0.0)
    rng = random.Random(seed)
    n = len(values)
    means = sorted(sum(values[rng.randrange(n)] for _ in range(n)) / n for _ in range(resamples))
    return (
        means[int((alpha / 2) * resamples)],
        means[min(resamples - 1, int((1 - alpha / 2) * resamples))],
    )


def cohens_kappa(judge: list[int], human: list[int]) -> float | None:
    """Agreement on binary labels, corrected for chance. None when undefined.

    Raw agreement is misleading when one class dominates: a judge that always
    says "faithful" agrees 90% of the time on a 90%-faithful set while carrying
    no information. Kappa subtracts that expected agreement.

    Returns None for the DEGENERATE case where either rater used only one label.
    Chance agreement is then 1.0 and kappa is mathematically undefined. Reporting
    1.0 there would be the worst kind of fake metric: it looks like perfect
    validation when it actually means "this sample contained no negative
    examples, so the judge was never tested".
    """
    if not judge or len(judge) != len(human):
        return None
    n = len(judge)
    if len(set(judge)) < 2 or len(set(human)) < 2:
        return None  # no variance -> undefined, not perfect
    observed = sum(1 for a, b in zip(judge, human, strict=True) if a == b) / n
    pj, ph = sum(judge) / n, sum(human) / n
    expected = pj * ph + (1 - pj) * (1 - ph)
    if expected >= 1.0:
        return None
    return (observed - expected) / (1 - expected)


def summarise(
    metric: str,
    scores: list[JudgeScore],
    kappa: float | None = None,
    kappa_n: int = 0,
    degenerate: bool = False,
) -> MetricSummary:
    good = [s.score for s in scores if s.ok]
    lo, hi = bootstrap_ci(good)
    return MetricSummary(
        metric=metric,
        n=len(good),
        mean=statistics.fmean(good) if good else 0.0,
        ci_low=lo,
        ci_high=hi,
        failures=sum(1 for s in scores if not s.ok),
        kappa=kappa,
        kappa_n=kappa_n,
        degenerate=degenerate,
    )
