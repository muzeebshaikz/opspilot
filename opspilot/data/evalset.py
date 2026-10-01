"""Evaluation query + qrel generation.

Gold labels are SPAN-based: (document_id, char_start, char_end). A retrieved
chunk counts as a hit when it overlaps the gold span past a threshold, so labels
survive re-chunking (audit G2).

Every query carries a difficulty tag. Reporting metrics per tag is what makes the
retrieval experiment falsifiable instead of a single aggregate that hides
everything (audit G7).

Provenance tiers
----------------
tier1  constructive  -- query derived from the document. Optimistic by
                        construction; useful for regression, never for headline
                        numbers.
tier2  independent   -- query written from the topic's *user* vocabulary, which
                        is disjoint from the document vocabulary by design.
tier3  human         -- reserved for hand-verified labels (not generated here).
"""
from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field
from pathlib import Path

from opspilot.core import Role
from opspilot.data.corpus import Document
from opspilot.data.topics import ABSENT_TOPICS, TOPICS


@dataclass(slots=True)
class Qrel:
    document_id: str
    char_start: int
    char_end: int
    grade: int  # 0 irrelevant, 1 partial, 2 full


@dataclass(slots=True)
class EvalQuery:
    query_id: str
    text: str
    difficulty: str          # L S V P D M A X
    tier: str                # tier1 | tier2 | tier3
    role_scope: str          # role asking the question
    qrels: list[Qrel] = field(default_factory=list)
    note: str = ""

    def to_json(self) -> dict:
        d = asdict(self)
        d["qrels"] = [asdict(q) for q in self.qrels]
        return d


def _qid(*parts: str) -> str:
    return "Q-" + hashlib.sha256("::".join(parts).encode()).hexdigest()[:10].upper()


def _find(docs: list[Document], topic_key: str, family: str,
          platform: str | None = None, version: str | None = None) -> Document | None:
    for d in docs:
        if d.topic_key != topic_key or d.family != family:
            continue
        if platform is not None and d.platform != platform:
            continue
        if version is not None and d.version != version:
            continue
        return d
    return None


def _span(doc: Document, kind: str) -> tuple[int, int]:
    s = doc.section_span(kind)
    if s is None:
        return (0, len(doc.content))
    return s


def _role_for(topic_access_level: str) -> str:
    """A query's asking role must be able to SEE its own gold documents.

    Labelling an `employee` query with a gold document classified `IT_admin`
    makes the task impossible and silently drives recall to zero -- the label,
    not the retriever, is wrong. Audit G4: qrels are role-scoped.
    """
    return topic_access_level


def _doc_level(doc: Document, *, grade: int = 1) -> list[Qrel]:
    """Partial credit for retrieving the correct document at all.

    Passage-level grading alone is too harsh: a chunk from the right document
    but the wrong section is genuinely useful to a RAG pipeline (it establishes
    the right document, and the context builder can expand). Grade 2 stays
    reserved for the passage that actually answers the question.
    """
    return [Qrel(doc.document_id, 0, len(doc.content), grade)]


def _overview_qrels(docs: list[Document], t, grade: int = 1) -> list[Qrel]:
    """The category handbook restates every topic. Retrieving its section for
    this topic is partially relevant, and leaving it unlabelled would score a
    correct retrieval as a miss."""
    for d in docs:
        if d.family != "overview" or d.category != t.category:
            continue
        for s in d.sections:
            if s.heading == t.title:
                return [Qrel(d.document_id, s.start, s.end, grade)]
    return []


def build_queries(docs: list[Document]) -> list[EvalQuery]:
    qs: list[EvalQuery] = []

    for t in TOPICS:
        canonical = _find(docs, t.key, "canonical")
        if canonical is None:
            continue
        role = _role_for(t.access_level)
        base = _doc_level(canonical) + _overview_qrels(docs, t)

        # ---- L: lexical-anchored. Exact identifiers. BM25 should win. -------
        err = _find(docs, t.key, "errorcode")
        for code in t.error_codes:
            target = err or canonical
            qs.append(EvalQuery(
                query_id=_qid(t.key, "L", code), text=f"what does {code} mean and how do I fix it",
                difficulty="L", tier="tier2", role_scope=role,
                qrels=[Qrel(target.document_id, *_span(target, "codes"), 2),
                       *_doc_level(target),
                       Qrel(canonical.document_id, *_span(canonical, "resolution"), 1)] + base,
                note="rare identifier; dense embeddings tend to smear these",
            ))

        # ---- S: semantic paraphrase. Disjoint vocabulary. Dense should win. -
        qs.append(EvalQuery(
            query_id=_qid(t.key, "S"), text=t.paraphrase_query,
            difficulty="S", tier="tier2", role_scope=role,
            qrels=[Qrel(canonical.document_id, *_span(canonical, "cause"), 2),
                   Qrel(canonical.document_id, *_span(canonical, "resolution"), 2),
                   Qrel(canonical.document_id, *_span(canonical, "symptoms"), 1)] + base,
            note="user vocabulary is disjoint from document vocabulary by design",
        ))

        # ---- baseline canonical phrasing (tier1, constructive) --------------
        qs.append(EvalQuery(
            query_id=_qid(t.key, "B"), text=t.canonical_query,
            difficulty="B", tier="tier1", role_scope=role,
            qrels=[Qrel(canonical.document_id, *_span(canonical, "resolution"), 2)] + base,
            note="optimistic by construction",
        ))

        # ---- V: version / recency. Stale revisions are wrong answers. -------
        v21 = _find(docs, t.key, "version", version="2.1")
        v10 = _find(docs, t.key, "version", version="1.0")
        if v21 is not None:
            stale = [Qrel(v21.document_id, *_span(v21, "resolution"), 0)]
            if v10 is not None:
                stale.append(Qrel(v10.document_id, *_span(v10, "resolution"), 0))
            qs.append(EvalQuery(
                query_id=_qid(t.key, "V"),
                text=f"what is the current supported fix for {t.title.lower()}",
                difficulty="V", tier="tier2", role_scope=role,
                qrels=[Qrel(canonical.document_id, *_span(canonical, "resolution"), 2),
                       *_doc_level(canonical), *stale],
                note="v1.0/v2.1 contain a deliberately wrong step; retrieving them is a real failure",
            ))

        # ---- P: platform variant --------------------------------------------
        for plat in t.platforms:
            pd = _find(docs, t.key, "platform", platform=plat)
            if pd is None:
                continue
            qs.append(EvalQuery(
                query_id=_qid(t.key, "P", plat),
                text=f"{t.canonical_query} on {plat}",
                difficulty="P", tier="tier2", role_scope=role,
                qrels=[Qrel(pd.document_id, *_span(pd, "resolution"), 2),
                       *_doc_level(pd),
                       Qrel(canonical.document_id, *_span(canonical, "resolution"), 1)],
                note="metadata-aware retrieval should prefer the matching platform",
            ))

        # ---- D: distractor near-duplicate -----------------------------------
        dd = _find(docs, t.key, "distractor")
        if dd is not None:
            qs.append(EvalQuery(
                query_id=_qid(t.key, "D"),
                text=f"{t.canonical_query} for {t.product}",
                difficulty="D", tier="tier2", role_scope=role,
                qrels=[Qrel(canonical.document_id, *_span(canonical, "resolution"), 2),
                       *_doc_level(canonical),
                       Qrel(dd.document_id, 0, len(dd.content), 0)],
                note=f"{dd.product} doc is near-identical; only a cross-encoder should separate them",
            ))

        # ---- A: access-gated -------------------------------------------------
        pol = _find(docs, t.key, "policy")
        if pol is not None and t.access_level == "employee":
            qs.append(EvalQuery(
                query_id=_qid(t.key, "A"),
                text=f"what is the internal escalation contact for {t.product}",
                difficulty="A", tier="tier2", role_scope="employee",
                qrels=[],  # employee must NOT retrieve it -> correct outcome is abstention
                note=f"answer exists in {pol.document_id} at {pol.access_level}; "
                     "employee-scoped retrieval must return nothing",
            ))
            qs.append(EvalQuery(
                query_id=_qid(t.key, "A", "admin"),
                text=f"what is the internal escalation contact for {t.product}",
                difficulty="A", tier="tier2", role_scope="IT_admin",
                qrels=[Qrel(pol.document_id, *_span(pol, "contacts"), 2)],
                note="same query, elevated role, answer is now legitimately reachable",
            ))

        # ---- M: multi-hop ----------------------------------------------------
        if t.prerequisite_of:
            other = _find(docs, t.prerequisite_of, "canonical")
            if other is not None:
                # asking role must be able to see BOTH documents
                m_role = max((role, other.access_level), key=lambda r: Role(r).rank)
                qs.append(EvalQuery(
                    query_id=_qid(t.key, "M"),
                    text=f"{t.paraphrase_query} - and what do I need in place first",
                    difficulty="M", tier="tier2", role_scope=m_role,
                    qrels=[Qrel(canonical.document_id, *_span(canonical, "resolution"), 2),
                           Qrel(other.document_id, *_span(other, "resolution"), 2),
                           *_doc_level(canonical), *_doc_level(other)],
                    note="requires two documents; tests that recall@k does real work",
                ))

    # ---- X: unanswerable ---------------------------------------------------
    for topic in ABSENT_TOPICS:
        qs.append(EvalQuery(
            query_id=_qid("X", topic), text=f"what is the procedure for {topic}",
            difficulty="X", tier="tier2", role_scope="employee", qrels=[],
            note="deliberately undocumented; correct outcome is abstention",
        ))

    qs.sort(key=lambda q: q.query_id)
    return qs


def write_queries(out_dir: Path, queries: list[EvalQuery]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    payload = [q.to_json() for q in queries]
    (out_dir / "queries.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True).encode()).hexdigest()
    counts: dict[str, int] = {}
    for q in queries:
        counts[q.difficulty] = counts.get(q.difficulty, 0) + 1
    manifest = {
        "dataset_hash": digest,
        "query_count": len(queries),
        "by_difficulty": counts,
        "by_tier": {t: sum(1 for q in queries if q.tier == t) for t in ("tier1", "tier2", "tier3")},
        "labelled_positives": sum(len([r for r in q.qrels if r.grade > 0]) for q in queries),
    }
    (out_dir / "dataset_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
