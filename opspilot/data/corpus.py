"""Deterministic synthetic knowledge-base generator.

Design notes
------------
* Fully deterministic (seeded). `make data` reproduces byte-for-byte, which is
  what makes "dataset version" in the experiment record meaningful (audit G17).
* No LLM is used. Structure comes from templates so that structure-aware
  chunking is testable; the topic table supplies the semantics.
* Every document records its sections WITH character offsets. Evaluation
  gold labels key to (document_id, char_start, char_end) rather than to a
  chunk id, so labels survive re-chunking (audit G2).
"""
from __future__ import annotations

import hashlib
import json
import random
from dataclasses import asdict, dataclass, field
from datetime import date, timedelta
from pathlib import Path

from opspilot.data.topics import ABSENT_TOPICS, CATEGORIES, TOPICS, Topic

BASE_DATE = date(2024, 1, 15)


@dataclass(slots=True)
class Section:
    heading: str
    body: str
    start: int
    end: int
    kind: str = "prose"


@dataclass(slots=True)
class Document:
    document_id: str
    title: str
    department: str
    category: str
    product: str
    version: str
    effective_date: str
    access_level: str
    family: str          # canonical | errorcode | version | platform | distractor | runbook | faq | policy | known_issues | overview
    topic_key: str
    status: str          # current | superseded | deprecated
    platform: str | None
    content: str
    sections: list[Section] = field(default_factory=list)

    def section_span(self, kind: str) -> tuple[int, int] | None:
        for s in self.sections:
            if s.kind == kind:
                return (s.start, s.end)
        return None

    def to_json(self) -> dict:
        d = asdict(self)
        d["sections"] = [asdict(s) for s in self.sections]
        return d


class _Builder:
    """Accumulates markdown while tracking exact character offsets per section."""

    def __init__(self, title: str):
        self.parts: list[str] = []
        self.sections: list[Section] = []
        self._cursor = 0
        self._append_raw(f"# {title}\n\n")

    def _append_raw(self, text: str) -> None:
        self.parts.append(text)
        self._cursor += len(text)

    def add(self, heading: str, body: str, kind: str = "prose") -> None:
        head = f"## {heading}\n\n"
        self._append_raw(head)
        start = self._cursor
        block = body.rstrip() + "\n\n"
        self._append_raw(block)
        self.sections.append(
            Section(heading=heading, body=body.rstrip(), start=start,
                    end=start + len(body.rstrip()), kind=kind)
        )

    def build(self) -> tuple[str, list[Section]]:
        return "".join(self.parts), self.sections


def _fmt_steps(steps: tuple[str, ...]) -> str:
    return "\n".join(f"{i}. {s}" for i, s in enumerate(steps, 1))


def _doc_id(*parts: str) -> str:
    h = hashlib.sha256("::".join(parts).encode()).hexdigest()[:10]
    return f"KB-{h.upper()}"


def _eff(rng: random.Random, offset_days: int) -> str:
    return (BASE_DATE + timedelta(days=offset_days + rng.randint(0, 25))).isoformat()


# --------------------------------------------------------------------------
# document families
# --------------------------------------------------------------------------
def _background(t: Topic) -> str:
    return (
        f"{t.product} is operated by the {t.department} function and is in scope for all "
        f"Acme Technologies staff using {', '.join(t.platforms)} endpoints. The behaviour "
        f"described here has been observed across multiple hardware generations and is not "
        f"specific to any single device model.\n\n"
        f"Users most commonly describe this condition informally as "
        f"{', '.join(t.user_terms[:3])}. Service desk agents should treat those descriptions "
        f"as pointing at this article even though the underlying mechanism is "
        f"{t.doc_terms[0]} related."
    )


def _analysis(t: Topic) -> str:
    terms = ", ".join(t.doc_terms)
    return (
        f"{t.cause}\n\n"
        f"The relevant subsystems are: {terms}. When the condition is present, the control "
        f"path fails before the data path, which is why connectivity can appear healthy in "
        f"a simple reachability test while the user-visible symptom persists.\n\n"
        f"Diagnosis should confirm the mechanism before applying the remediation, because the "
        f"same surface symptom can be produced by unrelated causes that the steps below will "
        f"not address."
    )


def _config_reference(t: Topic) -> str:
    rows = ["| Setting | Default | Supported range | Notes |",
            "| --- | --- | --- | --- |"]
    for i, term in enumerate(t.doc_terms[:4]):
        key = term.lower().replace(" ", "_")
        rows.append(f"| `{key}` | `{100 * (i + 1)}` | `{50 * (i + 1)}`-`{200 * (i + 1)}` | "
                    f"Applies to {t.product}. Changing this requires a service restart. |")
    return "\n".join(rows)


def _related(t: Topic) -> str:
    return (
        f"- Escalate to the {t.department} queue if the symptom persists after remediation.\n"
        f"- Confirm the endpoint is enrolled before troubleshooting {t.product}.\n"
        f"- Check the active incident board; a platform-wide event will present identically."
    )


def _canonical(t: Topic, rng: random.Random) -> Document:
    b = _Builder(t.title)
    b.add("Summary", f"{t.symptom} This document describes the supported remediation for "
                     f"{t.product} in the {t.department} estate.", kind="summary")
    b.add("Background", _background(t), kind="background")
    b.add("Symptoms", f"{t.symptom} Affected users typically observe this on "
                      f"{', '.join(t.platforms)}. Reported informally as: "
                      f"{'; '.join(t.user_terms)}.", kind="symptoms")
    b.add("Root Cause", _analysis(t), kind="cause")
    b.add("Resolution", _fmt_steps(t.fix_steps), kind="resolution")
    b.add("Configuration Reference", _config_reference(t), kind="config")
    if t.error_codes:
        b.add("Related Error Codes", ", ".join(t.error_codes), kind="codes")
    b.add("Related Articles", _related(t), kind="related")
    b.add("Escalation", "If the issue persists after these steps, raise a ticket with the "
                        f"{t.department} queue and attach the diagnostic bundle.", kind="escalation")
    content, sections = b.build()
    return Document(
        document_id=_doc_id(t.key, "canonical"), title=t.title, department=t.department,
        category=t.category, product=t.product, version="3.2",
        effective_date=_eff(rng, 600), access_level=t.access_level, family="canonical",
        topic_key=t.key, status="current", platform=None, content=content, sections=sections,
    )


def _errorcode(t: Topic, rng: random.Random) -> Document | None:
    if not t.error_codes:
        return None
    title = f"{t.product} Error Code Reference"
    b = _Builder(title)
    b.add("Scope", f"Authoritative lookup for diagnostic codes emitted by {t.product}.", kind="summary")
    rows = ["| Code | Meaning | Severity | First action |", "| --- | --- | --- | --- |"]
    for i, code in enumerate(t.error_codes):
        rows.append(f"| `{code}` | {t.symptom} | {'High' if i == 0 else 'Medium'} | "
                    f"{t.fix_steps[0]} |")
    b.add("Code Table", "\n".join(rows), kind="codes")
    for code in t.error_codes:
        b.add(f"Code {code}", (
            f"`{code}` is emitted by {t.product} when {t.cause.lower()} "
            f"The code is logged once per failed attempt and is not rate limited, so a "
            f"repeated appearance of `{code}` in the endpoint log is expected during an "
            f"active fault rather than an indication of a second problem.\n\n"
            f"Remediation for `{code}`:\n\n{_fmt_steps(t.fix_steps)}"
        ), kind="codedetail")
    b.add("Diagnostics", f"Collect logs with `acme-diag collect --product "
                         f"{t.product.lower().replace(' ', '-')}` before escalating. The bundle "
                         f"includes the {t.doc_terms[0]} trace, which is required by the "
                         f"{t.department} queue for any escalation.", kind="diagnostics")
    content, sections = b.build()
    return Document(
        document_id=_doc_id(t.key, "errorcode"), title=title, department=t.department,
        category=t.category, product=t.product, version="3.2",
        effective_date=_eff(rng, 620), access_level=t.access_level, family="errorcode",
        topic_key=t.key, status="current", platform=None, content=content, sections=sections,
    )


def _version_history(t: Topic, rng: random.Random) -> list[Document]:
    """v1.0 deprecated + v2.1 superseded. The current answer is the canonical doc.

    These exist to create V-class queries: the right answer is present in three
    documents and only recency/version metadata distinguishes them.
    """
    out: list[Document] = []
    for version, status, offset, drift in (("1.0", "deprecated", 0, 2), ("2.1", "superseded", 300, 1)):
        title = f"{t.title} (v{version})"
        b = _Builder(title)
        b.add("Notice", f"This revision is {status}. Refer to version 3.2 for current guidance.",
              kind="notice")
        b.add("Symptoms", t.symptom, kind="symptoms")
        b.add("Root Cause", t.cause, kind="cause")
        # Older revisions carry a subtly WRONG step -- this is what makes
        # retrieving the stale version an actual failure rather than a cosmetic one.
        stale = list(t.fix_steps)
        stale[0] = stale[0].replace("1400", "1500").replace("14 characters", "8 characters")
        stale.append("Contact the legacy support line on extension 4402.")
        b.add("Resolution", _fmt_steps(tuple(stale[: len(stale) - drift + 1])), kind="resolution")
        content, sections = b.build()
        out.append(Document(
            document_id=_doc_id(t.key, "version", version), title=title, department=t.department,
            category=t.category, product=t.product, version=version,
            effective_date=_eff(rng, offset), access_level=t.access_level, family="version",
            topic_key=t.key, status=status, platform=None, content=content, sections=sections,
        ))
    return out


def _platform_variants(t: Topic, rng: random.Random) -> list[Document]:
    out: list[Document] = []
    specifics = {
        "Windows": ("Run the commands from an elevated PowerShell session.",
                    "Check Event Viewer under Applications and Services Logs."),
        "macOS": ("Run the commands from Terminal with `sudo`.",
                  "Check unified logging with `log show --last 30m`."),
        "Linux": ("Run the commands as root or via `sudo`.",
                  "Inspect `journalctl -u acme-agent --since '-30 min'`."),
    }
    for plat in t.platforms:
        title = f"{t.title} on {plat}"
        b = _Builder(title)
        how, logs = specifics[plat]
        b.add("Applies To", f"{plat} endpoints running {t.product}. This article does not "
                            f"apply to other operating systems; see the corresponding "
                            f"{plat}-specific guidance only.", kind="summary")
        b.add("Symptoms", f"{t.symptom} On {plat} the condition additionally surfaces in the "
                          f"platform log with a {t.doc_terms[0]} warning.", kind="symptoms")
        b.add("Resolution", _fmt_steps(t.fix_steps) + f"\n\n{how}", kind="resolution")
        b.add("Platform Notes", (
            f"{plat} applies its own policy layer on top of {t.product}. Where the two "
            f"disagree the platform policy wins, so a change that appears to apply cleanly "
            f"may be reverted at next policy refresh. Confirm persistence after a reboot."
        ), kind="notes")
        b.add("Log Collection", logs, kind="diagnostics")
        content, sections = b.build()
        out.append(Document(
            document_id=_doc_id(t.key, "platform", plat), title=title, department=t.department,
            category=t.category, product=t.product, version="3.2",
            effective_date=_eff(rng, 590), access_level=t.access_level, family="platform",
            topic_key=t.key, status="current", platform=plat, content=content, sections=sections,
        ))
    return out


def _distractor(t: Topic, rng: random.Random) -> Document | None:
    """Near-duplicate for a DIFFERENT product. D-class: fusion cannot separate
    these, a cross-encoder can."""
    if not t.distractor_product:
        return None
    title = t.title.replace(t.product, t.distractor_product)
    if title == t.title:
        title = f"{t.title} ({t.distractor_product})"
    b = _Builder(title)
    b.add("Applies To", f"This document applies ONLY to {t.distractor_product}. "
                        f"For {t.product} see the corresponding {t.product} guide.", kind="summary")
    b.add("Symptoms", t.symptom, kind="symptoms")
    b.add("Root Cause", t.cause.replace("corporate", "laboratory"), kind="cause")
    wrong = tuple(s.replace("1400", "9000").replace("acmevpn", "labvpn") for s in t.fix_steps)
    b.add("Resolution", _fmt_steps(wrong), kind="resolution")
    content, sections = b.build()
    return Document(
        document_id=_doc_id(t.key, "distractor"), title=title, department=t.department,
        category=t.category, product=t.distractor_product, version="3.2",
        effective_date=_eff(rng, 595), access_level=t.access_level, family="distractor",
        topic_key=t.key, status="current", platform=None, content=content, sections=sections,
    )


def _runbook(t: Topic, rng: random.Random) -> Document:
    title = f"Runbook: {t.title}"
    b = _Builder(title)
    b.add("Preconditions", "- Requester identity verified\n- Device enrolled in management\n"
                           f"- {t.product} installed at a supported version", kind="preconditions")
    b.add("Procedure", _fmt_steps(t.fix_steps), kind="resolution")
    b.add("Verification", f"Confirm the symptom is cleared: {t.symptom.lower()} should no longer occur.",
          kind="verification")
    b.add("Rollback", "Revert the configuration change and restore the previous profile from backup.",
          kind="rollback")
    b.add("Owning Team", f"{t.department} Operations", kind="ownership")
    content, sections = b.build()
    return Document(
        document_id=_doc_id(t.key, "runbook"), title=title, department=t.department,
        category=t.category, product=t.product, version="3.2",
        effective_date=_eff(rng, 610), access_level="IT_agent", family="runbook",
        topic_key=t.key, status="current", platform=None, content=content, sections=sections,
    )


def _faq(t: Topic, rng: random.Random) -> Document:
    """Deliberately shorter than one chunk -- chunking edge case."""
    title = f"FAQ: {t.title}"
    b = _Builder(title)
    b.add("Question", t.canonical_query.capitalize() + "?", kind="question")
    b.add("Answer", t.fix_steps[0], kind="resolution")
    content, sections = b.build()
    return Document(
        document_id=_doc_id(t.key, "faq"), title=title, department=t.department,
        category=t.category, product=t.product, version="3.2",
        effective_date=_eff(rng, 615), access_level=t.access_level, family="faq",
        topic_key=t.key, status="current", platform=None, content=content, sections=sections,
    )


def _known_issues(t: Topic, rng: random.Random) -> Document:
    title = f"Known Issues: {t.product}"
    b = _Builder(title)
    b.add("Open Issues", f"- {t.symptom} Tracked internally; workaround below.\n"
                         f"- Intermittent latency on first connection after wake from sleep.",
          kind="issues")
    b.add("Workarounds", _fmt_steps(t.fix_steps[:2]), kind="resolution")
    content, sections = b.build()
    return Document(
        document_id=_doc_id(t.key, "known"), title=title, department=t.department,
        category=t.category, product=t.product, version="3.2",
        effective_date=_eff(rng, 605), access_level=t.access_level, family="known_issues",
        topic_key=t.key, status="current", platform=None, content=content, sections=sections,
    )


def _policy(t: Topic, rng: random.Random) -> Document:
    """Access-graded policy doc -- A-class source."""
    level = {"employee": "IT_agent", "IT_agent": "IT_admin", "IT_admin": "IT_admin"}[t.access_level]
    title = f"Internal Policy: {t.title}"
    b = _Builder(title)
    b.add("Classification", f"Internal -- {level} and above.", kind="classification")
    b.add("Policy Statement",
          f"Remediation of {t.product} issues follows the approved procedure. Deviations require "
          f"{t.department} lead approval and must be recorded in the change log.", kind="policy")
    b.add("Internal Escalation Contacts",
          "Primary: ops-oncall@acme.internal\nSecondary: +1-555-0142 (24x7 bridge)", kind="contacts")
    content, sections = b.build()
    return Document(
        document_id=_doc_id(t.key, "policy"), title=title, department=t.department,
        category=t.category, product=t.product, version="2.0",
        effective_date=_eff(rng, 560), access_level=level, family="policy",
        topic_key=t.key, status="current", platform=None, content=content, sections=sections,
    )


def _overview(category: str, topics: list[Topic], rng: random.Random) -> Document:
    """Long document -- forces hierarchy detection in chunking."""
    title = f"{category.title()} Support Handbook"
    b = _Builder(title)
    b.add("Introduction", f"Consolidated reference for all {category} topics supported by Acme "
                          "Technologies. Each section links to the authoritative guide.", kind="summary")
    for t in topics:
        b.add(t.title, f"{t.symptom}\n\n{t.cause}\n\n{_fmt_steps(t.fix_steps)}", kind="section")
    b.add("Cross References", "\n".join(f"- {a}" for a in ABSENT_TOPICS[:4]) +
          "\n\n(Topics above are out of scope for this handbook.)", kind="xref")
    content, sections = b.build()
    return Document(
        document_id=_doc_id("overview", category), title=title,
        department=topics[0].department if topics else "IT Operations",
        category=category, product="Acme Platform", version="3.2",
        effective_date=_eff(rng, 630), access_level="employee", family="overview",
        topic_key=f"__overview_{category}", status="current", platform=None,
        content=content, sections=sections,
    )


# --------------------------------------------------------------------------
# entry point
# --------------------------------------------------------------------------
def generate_documents(seed: int = 20260929) -> list[Document]:
    rng = random.Random(seed)
    docs: list[Document] = []
    for t in TOPICS:
        docs.append(_canonical(t, rng))
        if (d := _errorcode(t, rng)):
            docs.append(d)
        docs.extend(_version_history(t, rng))
        docs.extend(_platform_variants(t, rng))
        if (d := _distractor(t, rng)):
            docs.append(d)
        docs.append(_runbook(t, rng))
        docs.append(_faq(t, rng))
        docs.append(_known_issues(t, rng))
        docs.append(_policy(t, rng))
    by_cat: dict[str, list[Topic]] = {}
    for t in TOPICS:
        by_cat.setdefault(t.category, []).append(t)
    for cat in CATEGORIES:
        if by_cat.get(cat):
            docs.append(_overview(cat, by_cat[cat], rng))
    docs.sort(key=lambda d: d.document_id)
    return docs


def write_documents(out_dir: Path, docs: list[Document]) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    for cat in {d.category for d in docs}:
        (out_dir / cat).mkdir(exist_ok=True)
    manifest_rows = []
    for d in docs:
        path = out_dir / d.category / f"{d.document_id}.json"
        path.write_text(json.dumps(d.to_json(), indent=2), encoding="utf-8")
        manifest_rows.append({
            "document_id": d.document_id, "family": d.family, "category": d.category,
            "access_level": d.access_level, "status": d.status, "chars": len(d.content),
        })
    digest = hashlib.sha256(
        json.dumps(sorted(manifest_rows, key=lambda r: r["document_id"]), sort_keys=True).encode()
    ).hexdigest()
    families: dict[str, int] = {}
    for r in manifest_rows:
        families[r["family"]] = families.get(r["family"], 0) + 1
    manifest = {
        "corpus_hash": digest,
        "document_count": len(docs),
        "families": families,
        "categories": sorted({d.category for d in docs}),
        "absent_topics": list(ABSENT_TOPICS),
        "total_chars": sum(len(d.content) for d in docs),
    }
    (out_dir.parent / "corpus_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8")
    return manifest
