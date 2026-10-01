"""Structure-aware chunking.

Spec §8 explicitly forbids "chunk_size=500 / overlap=100" as the only strategy.
Two strategies are implemented so chunking is a measurable variable rather than
an assumption:

  FixedChunker       -- the naive baseline, kept as the control arm
  StructureChunker   -- respects markdown section boundaries, keeps numbered
                        procedures intact, and packs small sections together

Every chunk carries `char_start` / `char_end` into the ORIGINAL document, which
is what lets span-based gold labels survive a strategy change (audit G2).
"""

from __future__ import annotations

import hashlib
import re
from abc import ABC, abstractmethod
from dataclasses import dataclass

from opspilot.data.corpus import Document

_HEADING = re.compile(r"^(#{1,3})\s+(.*)$", re.MULTILINE)
_NUMBERED = re.compile(r"^\d+\.\s", re.MULTILINE)


@dataclass(slots=True)
class Chunk:
    chunk_id: str
    document_id: str
    text: str
    char_start: int
    char_end: int
    ordinal: int
    heading: str
    # denormalised metadata -- retrieval filters on these (spec §8, §9)
    title: str = ""
    category: str = ""
    product: str = ""
    version: str = ""
    effective_date: str = ""
    access_level: str = "employee"
    family: str = ""
    status: str = "current"
    platform: str | None = None

    def overlaps(
        self, start: int, end: int, *, cover_gold: float = 0.5, cover_chunk: float = 0.8
    ) -> bool:
        """Span-overlap hit test used by the evaluation harness.

        A chunk is a hit if it covers at least `cover_gold` of the gold span,
        OR the gold span covers at least `cover_chunk` of the chunk.
        """
        lo, hi = max(self.char_start, start), min(self.char_end, end)
        inter = max(0, hi - lo)
        if inter == 0:
            return False
        gold_len = max(1, end - start)
        chunk_len = max(1, self.char_end - self.char_start)
        return (inter / gold_len) >= cover_gold or (inter / chunk_len) >= cover_chunk


def _chunk_id(document_id: str, strategy: str, start: int, end: int) -> str:
    """Stable WITHIN a strategy. Changing strategy intentionally changes ids --
    which is exactly why gold labels must not key to them."""
    h = hashlib.sha256(f"{document_id}|{strategy}|{start}|{end}".encode()).hexdigest()[:12]
    return f"C-{h}"


class Chunker(ABC):
    name: str = "base"

    @abstractmethod
    def split(self, doc: Document) -> list[tuple[int, int, str]]:
        """Return (start, end, heading) spans over doc.content."""

    def chunk(self, doc: Document) -> list[Chunk]:
        out: list[Chunk] = []
        for i, (start, end, heading) in enumerate(self.split(doc)):
            text = doc.content[start:end].strip()
            if not text:
                continue
            out.append(
                Chunk(
                    chunk_id=_chunk_id(doc.document_id, self.name, start, end),
                    document_id=doc.document_id,
                    text=text,
                    char_start=start,
                    char_end=end,
                    ordinal=i,
                    heading=heading,
                    title=doc.title,
                    category=doc.category,
                    product=doc.product,
                    version=doc.version,
                    effective_date=doc.effective_date,
                    access_level=doc.access_level,
                    family=doc.family,
                    status=doc.status,
                    platform=doc.platform,
                )
            )
        return out


class FixedChunker(Chunker):
    """The control arm. Deliberately naive: it will cut numbered procedures in
    half, and the evaluation should show that."""

    name = "fixed"

    def __init__(self, size: int = 500, overlap: int = 100):
        self.size, self.overlap = size, overlap

    def split(self, doc: Document) -> list[tuple[int, int, str]]:
        spans, pos, n = [], 0, len(doc.content)
        step = max(1, self.size - self.overlap)
        while pos < n:
            spans.append((pos, min(pos + self.size, n), doc.title))
            pos += step
        return spans


class StructureChunker(Chunker):
    """Splits on markdown headings, then packs and only hard-splits oversized
    sections -- never inside a numbered procedure if it can be avoided."""

    name = "structure"

    def __init__(self, target: int = 900, max_size: int = 1400, min_size: int = 220):
        self.target, self.max_size, self.min_size = target, max_size, min_size

    def _sections(self, content: str) -> list[tuple[int, int, str]]:
        marks = [(m.start(), m.group(2).strip()) for m in _HEADING.finditer(content)]
        if not marks:
            return [(0, len(content), "")]
        out = []
        for i, (pos, heading) in enumerate(marks):
            end = marks[i + 1][0] if i + 1 < len(marks) else len(content)
            out.append((pos, end, heading))
        return out

    def split(self, doc: Document) -> list[tuple[int, int, str]]:
        content = doc.content
        spans: list[tuple[int, int, str]] = []
        buf_start: int | None = None
        buf_end = 0
        buf_head = ""

        def flush() -> None:
            nonlocal buf_start, buf_end, buf_head
            if buf_start is not None and buf_end > buf_start:
                spans.append((buf_start, buf_end, buf_head))
            buf_start, buf_end, buf_head = None, 0, ""

        for start, end, heading in self._sections(content):
            length = end - start
            if length > self.max_size:
                flush()
                spans.extend(self._hard_split(content, start, end, heading))
                continue
            if buf_start is None:
                buf_start, buf_end, buf_head = start, end, heading
            elif (end - buf_start) <= self.target:
                buf_end, buf_head = end, buf_head or heading
            else:
                flush()
                buf_start, buf_end, buf_head = start, end, heading
        flush()
        return spans or [(0, len(content), doc.title)]

    def _hard_split(
        self, content: str, start: int, end: int, heading: str
    ) -> list[tuple[int, int, str]]:
        """Split an oversized section, preferring paragraph boundaries and
        avoiding cutting between numbered steps."""
        body = content[start:end]
        breaks = [m.start() for m in re.finditer(r"\n\n", body)]
        step_starts = {m.start() for m in _NUMBERED.finditer(body)}
        out, cur = [], 0
        while cur < len(body):
            hard = min(cur + self.max_size, len(body))
            if hard >= len(body):
                out.append((start + cur, end, heading))
                break
            candidates = [
                b for b in breaks if cur + self.min_size < b <= hard and (b + 2) not in step_starts
            ]
            cut = candidates[-1] if candidates else hard
            out.append((start + cur, start + cut, heading))
            cur = cut
        return out


CHUNKERS: dict[str, type[Chunker]] = {
    "fixed": FixedChunker,
    "structure": StructureChunker,
}


def chunk_corpus(docs: list[Document], strategy: str = "structure") -> list[Chunk]:
    chunker = CHUNKERS[strategy]()
    out: list[Chunk] = []
    for d in docs:
        out.extend(chunker.chunk(d))
    return out
