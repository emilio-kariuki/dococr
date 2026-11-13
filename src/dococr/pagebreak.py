"""Reading a value across a page break: page furniture, and the bottom of one page joined to the
top of the next.

Pages are read one at a time (`pipeline`). A sentence value (`until`) whose label is near the
bottom of a page and whose end text is near the top of the next, or a `multiline` block that
continues on the next page, is not on either page alone. Two things stand between the two halves:

- **Page furniture** (`find_furniture`): running headers and footers, page numbers, initials
  boxes. A line counts as furniture when it is one of the `EDGE_ROWS` outermost rows of text at
  the top or bottom of a page, inside the outer `EDGE_BAND` of the page's height, and
  - it is a page number: a bare number, or one with "page" / "p." / "pg", or "n of m", "n/m",
    "- n -" (a bare number must be repeated at that edge of another page, or equal the page's
    position in the document), or contains "page n" ("Sale Agreement - Page 2 of 3"), or
  - its text, with digits ignored, is repeated at the same edge of another page, at about the
    same height (similarity `REPEAT_SIMILARITY`, height within `REPEAT_DRIFT` of the page), or
  - it holds no letters or digits (a rule of underscores), or
  - it contains one of the document type's `page_furniture` texts.
  Rows are peeled from the edge inwards; the first row that is not furniture stops the peeling.
- **The gap**: the bottom margin of one page and the top margin of the next.

`join_pair` takes the last `WINDOW_ROWS` rows of a page and the first `WINDOW_ROWS` rows of the
next, both without furniture, places the second right under the first at the first page's own
line spacing, and runs the matcher on that strip for the `until` and `multiline` fields only. A
reading is kept only when its label, value or end text lie on both pages; readings inside one
page are left to the per-page result. Form-style values (beside or below a label) are never read
across a break.
"""

from __future__ import annotations

import re
import statistics
from dataclasses import dataclass, field
from typing import Optional

from rapidfuzz.distance import Levenshtein

from dococr.extractors.base import PageLines
from dococr.models import DocType, FieldSpec
from dococr.ocr import Box, OcrLine
from dococr.ocr.matcher import MAX_FLOW_LINES, match_fields

EDGE_BAND = 0.15  # furniture lies in the top or bottom 15% of the page
EDGE_ROWS = 3  # ... among the three outermost rows of text there
REPEAT_SIMILARITY = 0.8  # text repeated on another page: at least this similar, digits ignored
REPEAT_DRIFT = 0.05  # ... and at a height within this share of the page height
WINDOW_ROWS = MAX_FLOW_LINES + 2  # rows of each page read together across the break
HEIGHT_RATIO = 1.6  # pages whose line heights differ more than this are not joined (OCR boxes
# are about 1.4 times as tall as text-layer boxes of the same type, so a text page joins an OCR page)

# Page numbers: "3", "- 3 -", "[3]", "Page 3", "p. 3", "Pg 3", "3 of 10", "Page 3 of 10", "3/10".
_PAGE_NO = re.compile(
    r"^[\W_]*(?P<word>(?:page|pg|p)\.?\s*)?(?P<n>\d{1,4})(?P<of>\s*(?:of|/|out\s+of)\s*\d{1,4})?[\W_]*$",
    re.IGNORECASE)
_PAGE_IN = re.compile(r"\bpage\s*\d{1,4}(?:\s*(?:of|/)\s*\d{1,4})?\b", re.IGNORECASE)


def _canon(text: str) -> str:
    """Comparison form: lower case, digits as '#', letters and digits only."""
    return re.sub(r"[^a-z#]", "", re.sub(r"\d+", "#", text.lower()))


def _rows(lines: list[OcrLine], keep: Optional[list[int]] = None) -> list[list[int]]:
    """Line indices grouped into rows by vertical overlap, top to bottom, each left to right."""
    idx = list(range(len(lines))) if keep is None else list(keep)
    idx.sort(key=lambda i: (lines[i].box[1] + lines[i].box[3]) / 2)
    rows: list[list[int]] = []
    bands: list[tuple[float, float]] = []
    for i in idx:
        t, b = lines[i].box[1], lines[i].box[3]
        best, best_ov = -1, 0.0
        for k, (rt, rb) in enumerate(bands):
            ov = (min(b, rb) - max(t, rt)) / max(1e-6, min(b - t, rb - rt))
            if ov >= 0.5 and ov > best_ov:
                best, best_ov = k, ov
        if best < 0:
            rows.append([i])
            bands.append((t, b))
        else:
            rows[best].append(i)
            members = rows[best]
            bands[best] = (sum(lines[k].box[1] for k in members) / len(members),
                           sum(lines[k].box[3] for k in members) / len(members))
    order = sorted(range(len(rows)), key=lambda k: bands[k][0])
    return [sorted(rows[k], key=lambda i: lines[i].box[0]) for k in order]


@dataclass
class _Edge:
    page: int
    side: str  # "top" | "bottom"
    rows: list[list[int]]  # outermost first


def _edges(page: int, pl: PageLines) -> list[_Edge]:
    h = float(pl.size[1]) or 1.0
    rows = _rows(pl.lines)

    def cy(row: list[int]) -> float:
        return sum((pl.lines[i].box[1] + pl.lines[i].box[3]) / 2 for i in row) / len(row)

    top: list[list[int]] = []
    for row in rows[:EDGE_ROWS]:
        if cy(row) > EDGE_BAND * h:
            break
        top.append(row)
    bottom: list[list[int]] = []
    for row in rows[::-1][:EDGE_ROWS]:
        if cy(row) < (1 - EDGE_BAND) * h or row in top:
            break
        bottom.append(row)
    return [_Edge(page, "top", top), _Edge(page, "bottom", bottom)]


def find_furniture(pages: list[Optional[PageLines]], extra: list[str] = ()) -> list[set[int]]:
    """For each page, the indices of its lines that are page furniture (see the module
    docstring). A page without lines (None) gets an empty set."""
    out: list[set[int]] = [set() for _ in pages]
    edges = [e for p, pl in enumerate(pages) if pl is not None for e in _edges(p, pl)]
    extra_canon = [c for c in (_canon(t) for t in extra) if len(c) >= 3]

    def text(page: int, i: int) -> str:
        return pages[page].lines[i].text

    def rel_y(page: int, i: int) -> float:
        box = pages[page].lines[i].box
        return (box[1] + box[3]) / 2 / (float(pages[page].size[1]) or 1.0)

    # every candidate line: (page, side, index) -> canonical text
    cands = {(e.page, e.side, i): _canon(text(e.page, i)) for e in edges for row in e.rows for i in row}

    def repeated(page: int, side: str, i: int) -> bool:
        c = cands[(page, side, i)]
        if not c:
            return False
        for (q, s, j), d in cands.items():
            if q == page or s != side or not d:
                continue
            if abs(rel_y(page, i) - rel_y(q, j)) > REPEAT_DRIFT:
                continue
            if Levenshtein.normalized_similarity(c, d) >= REPEAT_SIMILARITY:
                return True
        return False

    def is_furniture(page: int, side: str, i: int) -> bool:
        t = text(page, i).strip()
        if not re.search(r"[A-Za-z0-9]", t):
            return True
        m = _PAGE_NO.match(t)
        if m and (m.group("word") or m.group("of")):
            return True
        if _PAGE_IN.search(t):
            return True
        if m and (repeated(page, side, i) or int(m.group("n")) == page + 1):
            return True
        if not m and repeated(page, side, i):
            return True
        c = cands[(page, side, i)]
        return any(x in c for x in extra_canon) or any(
            Levenshtein.normalized_similarity(c, x) >= REPEAT_SIMILARITY for x in extra_canon)

    for e in edges:
        for row in e.rows:  # outermost first
            if all(is_furniture(e.page, e.side, i) for i in row):
                out[e.page].update(row)
            else:
                break
    return out


@dataclass
class Joined:
    """A reading that crosses the break between two pages (0: the first, 1: the second)."""
    name: str
    raw: str
    confidence: Optional[float]
    warnings: list[str]
    start: int  # page (0 or 1) where the value starts
    parts: list[tuple[int, Box]] = field(default_factory=list)  # (page, box in its working image)
    clash: Optional[str] = None  # another field's reading on one of the pages has these words


def _covers(a: Box, b: Box) -> bool:
    """Do two word boxes overlap by more than half of the smaller?"""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return False
    return w * h > 0.5 * max(1e-6, min((a[2] - a[0]) * (a[3] - a[1]), (b[2] - b[0]) * (b[3] - b[1])))


def _union(boxes: list[Box]) -> Box:
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def join_pair(a: PageLines, b: PageLines, furniture_a: set[int], furniture_b: set[int],
              doc_type: DocType, fields: list[FieldSpec]) -> dict[str, Joined]:
    """Read the `until` and `multiline` fields among `fields` across the break from page `a` to
    page `b`. Returns the readings that cross it, by field name."""
    specs = [f for f in fields if f.until_texts() or f.multiline]
    if not specs or not a.lines or not b.lines:
        return {}
    body_a = [i for i in range(len(a.lines)) if i not in furniture_a]
    body_b = [i for i in range(len(b.lines)) if i not in furniture_b]
    rows_a, rows_b = _rows(a.lines, body_a), _rows(b.lines, body_b)
    if not rows_a or not rows_b:
        return {}
    tail = [i for row in rows_a[-WINDOW_ROWS:] for i in row]
    head = [i for row in rows_b[:WINDOW_ROWS] for i in row]
    ha = statistics.median(a.lines[i].height for i in tail)
    hb = statistics.median(b.lines[i].height for i in head)
    if not (ha > 0 and hb > 0) or max(ha, hb) / min(ha, hb) > HEIGHT_RATIO:
        return {}

    # The first page's own spacing between rows, to place the second page's text under it.
    gaps = []
    for r1, r2 in zip(rows_a[-WINDOW_ROWS:], rows_a[-WINDOW_ROWS:][1:]):
        g = min(a.lines[i].box[1] for i in r2) - max(a.lines[i].box[3] for i in r1)
        if 0 < g < 2 * ha:
            gaps.append(g)
    gap = min(max(statistics.median(gaps), 0.2 * ha), ha) if gaps else 0.4 * ha
    bottom = max(a.lines[i].box[3] for i in tail)
    top = min(b.lines[i].box[1] for i in head)
    dy = bottom + gap - top
    # Line the pages up on their left margins (a scan may be shifted on the glass).
    dx = min(a.lines[i].box[0] for i in body_a) - min(b.lines[i].box[0] for i in body_b)

    lines: list[OcrLine] = []
    origin: list[tuple[int, int]] = []
    for i in tail:
        lines.append(a.lines[i])
        origin.append((0, i))
    for i in head:
        ln = b.lines[i]
        l, t, r, bt = ln.box
        lines.append(OcrLine(ln.text, ln.confidence, (l + dx, t + dy, r + dx, bt + dy)))
        origin.append((1, i))
    width = max(a.size[0], b.size[0] + max(0.0, dx))
    matches, _ = match_fields(lines, doc_type, specs, page_size=(width, bottom + gap + b.size[1]))

    out: dict[str, Joined] = {}
    for m in matches:
        if m.raw is None or not m.parts:
            continue
        touched = {origin[li][0] for li, _ in m.parts} | {origin[li][0] for li in m.anchor_lines}
        if touched != {0, 1}:
            continue
        pieces: dict[int, list[Box]] = {}
        order: list[int] = []
        clash = None
        for li, box in m.parts:
            page = origin[li][0]
            if page == 1:
                box = (box[0] - dx, box[1] - dy, box[2] - dx, box[3] - dy)
            if page not in pieces:
                pieces[page] = []
                order.append(page)
            pieces[page].append(box)
            pl = a if page == 0 else b
            clash = clash or next((other for other, boxes in pl.claimed.items()
                                   if other != m.name and any(_covers(box, x) for x in boxes)), None)
        warnings = list(m.warnings)
        spec = next(s for s in specs if s.name == m.name)
        if spec.multiline and not spec.until_texts() and len(order) > 1:
            warnings.append("the value continues at the top of the next page; check where it ends")
        out[m.name] = Joined(m.name, m.raw, m.confidence, warnings, order[0],
                             [(p, _union(pieces[p])) for p in order], clash)
    return out
