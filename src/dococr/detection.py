"""Which document type is this? Scores every candidate type's YAML against a page's lines.

The page is read once (`Extractor.read_page`); this module only compares text. Everything it
looks for comes from the candidates' YAML files; it holds no document knowledge.

Evidence
--------
Each candidate type contributes *phrases*: the `label` and `aliases` of its fields, the label and
end texts of its `until` fields, its `other_labels`, its `identifying_text` and its `title`.
A phrase is compared in the matcher's canonical form, so two types that list the same words
share one phrase. Phrases are searched on the page together, for all candidates at once:

- as row labels (`matcher.find_label_hits`, the matcher's own label search: edit budget by word
  length, a short single word exact and at the start of an OCR box, labels stacked on two or
  three lines). Where two phrases claim the same printed words, the longer wins, so "Code"
  printed inside "Permit Code" is evidence only for types that list the longer label.
- in reading order (`matcher.find_flow_hits`), so a phrase may cross a line break: the labels
  and end texts of `until` fields, `other_labels`, `identifying_text` and `title`. Only phrases
  of two or more words or one word of six or more characters are searched this way, because
  a short word ("of", "between") is found in any running text.

Each phrase has a *weight*, its distinctiveness among the candidates:
log((N + 1) / n) / log(N + 1), where N is the number of candidate types and n the number of
them that list the phrase. A phrase only one candidate lists weighs 1; one that every
candidate lists weighs little (0.10 among five).

Score
-----
A type's labels are grouped into *units*: one per field (found when any of its labels or end
texts is found; credited with the weight of the most distinctive one found, and counted in
the total with the weight of its most distinctive one), and one per other label. Identifying
text and the title are units too, but they only count when found: a heading may be cut off or
unreadable, and its absence says little.

- coverage = credit of the units found / (total weight of the units + credit of the
  identifying units found). A share, so a type does not win by listing more labels.
- precision = weight of the phrases found that this type lists / weight of all phrases found
  for any candidate. Text on the page that a type cannot explain counts against it; this is
  what separates two types that share most of their labels.
- score = harmonic mean of coverage and precision.

Refusal
-------
No type is chosen (`Detection.doc_type` is None) when the best score is under `MIN_SCORE`, when
the best type has fewer than `MIN_FOUND` units found (identifying ones included), or when
another type scores more than `MAX_RUNNER_UP` of the best, unless the best dominates it: every
phrase found for the other type is listed by the best too, and the best has found evidence the
other does not list (`_dominates`). Two types with the same labels therefore stay unknown, and
two that differ in one label are told apart when that label is printed. The thresholds were chosen from measurements on synthetic pages
and real documents; see `docs/ARCHITECTURE.md`.

Several pages
-------------
The evidence of the first `DETECT_PAGES` pages is pooled (a phrase found on any of them counts
once) and the type is decided once for the document.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterable, Optional, Sequence, Union

from dococr.doctypes import find_doc_types, load_doc_type
from dococr.models import Detection, DetectionCandidate, DocType, Evidence, slugify

# Thresholds, chosen from measurements (see the module docstring).
MIN_SCORE = 0.38  # the best type's score must reach this
MIN_FOUND = 3  # ... with at least this many of its labels or texts found
MAX_RUNNER_UP = 0.80  # ... and the second best may score at most this share of the best

DETECT_PAGES = 5  # evidence is taken from the first pages of a document, at most this many
TOP = 5  # candidates reported in a Detection

MAX_CANDIDATES_ENV = "DOCOCR_MAX_CANDIDATES"
DEFAULT_MAX_CANDIDATES = 50

AUTO = "auto"


class UnknownDocumentError(ValueError):
    """The document is none of the candidate types, or the page cannot tell which. `detection`
    holds the ranking and the evidence."""

    def __init__(self, detection: Detection):
        self.detection = detection
        super().__init__(f"the document type could not be detected: {detection.reason}")


def max_candidates() -> int:
    """The most candidate types one detection scores (DOCOCR_MAX_CANDIDATES, default 50)."""
    try:
        n = int(os.environ.get(MAX_CANDIDATES_ENV, DEFAULT_MAX_CANDIDATES))
    except ValueError:
        return DEFAULT_MAX_CANDIDATES
    return n if n >= 1 else DEFAULT_MAX_CANDIDATES


def is_auto(doc_type) -> bool:
    return doc_type is None or (isinstance(doc_type, str) and doc_type.strip().lower() == AUTO)


def candidate_types(
    candidates: Optional[Sequence[Union[str, Path, DocType]]] = None,
    dirs: Optional[Iterable[Union[str, Path]]] = None,
    limit: Optional[int] = None,
) -> tuple[list[DocType], list[str]]:
    """The types to choose among, and warnings. By default every type in the search folders
    (`doctypes.find_doc_types`); otherwise the named ones. More than `limit` (default
    `max_candidates()`) is a ValueError, never a silent cut: a cut could leave out the right
    type and choose a wrong one."""
    limit = max_candidates() if limit is None else limit
    warnings: list[str] = []
    if candidates:
        docs: list[DocType] = []
        seen: set[str] = set()
        if len(candidates) > limit:
            raise ValueError(f"{len(candidates)} candidate document types were given; the limit is "
                             f"{limit} ({MAX_CANDIDATES_ENV})")
        for item in candidates:
            if is_auto(item):
                raise ValueError("'auto' is not a document type")
            doc = load_doc_type(item, dirs)
            if not doc.fields:
                raise ValueError(f"unknown document type '{item}'")
            if doc.name not in seen:
                seen.add(doc.name)
                docs.append(doc)
    else:
        docs, skipped = find_doc_types(dirs)
        docs = [d for d in docs if d.fields]
        if skipped:
            warnings.append(f"{len(skipped)} document type file(s) could not be read and were not "
                            "considered")
    if len(docs) > limit:
        raise ValueError(f"{len(docs)} document types were found, more than the {limit} that one "
                         f"detection may compare ({MAX_CANDIDATES_ENV}); name the candidates")
    if not docs:
        raise ValueError("there are no document types to choose from")
    return docs, warnings


# ---------------------------------------------------------------------------- profiles

@dataclass
class _Phrase:
    key: tuple[str, ...]  # canonical words
    text: str
    kind: str  # Evidence.kind
    field: Optional[str]
    row: bool  # searched as a row label
    flow: bool  # searched in reading order


@dataclass
class _Unit:
    phrases: list[_Phrase]
    ident: bool  # identifying text or title: counts only when found


@dataclass
class _Profile:
    doc: DocType
    units: list[_Unit]
    keys: set = field(default_factory=set)


def _profile(doc: DocType) -> _Profile:
    from dococr.ocr.matcher import _label_words

    def distinctive(words: tuple[str, ...]) -> bool:
        return len(words) >= 2 or (len(words) == 1 and len(words[0]) >= 6)

    owned: set[tuple[str, ...]] = set()
    units: list[_Unit] = []

    def unit(items: list[tuple[str, str, Optional[str], bool, bool]], ident: bool) -> None:
        phrases = []
        for text, kind, fname, row, flow in items:
            key = _label_words(text)
            if not key or key in owned:
                continue
            flow = flow and distinctive(key)
            if not (row or flow):
                continue
            owned.add(key)
            phrases.append(_Phrase(key, text.strip(), kind, fname, row, flow))
        if phrases:
            units.append(_Unit(phrases, ident))

    # Identifying text first, so a phrase listed both there and among other labels counts as
    # identifying (never against the type when missing).
    for text in doc.identifying_text:
        unit([(text, "identifying_text", None, True, True)], True)
    if slugify(doc.title) != doc.name:  # a title that only repeats the name says nothing more
        unit([(doc.title, "title", None, True, True)], True)
    for spec in doc.fields:
        if spec.until_texts():
            unit([(t, "label", spec.name, False, True) for t in [*spec.labels(), *spec.until_texts()]],
                 False)
        else:
            unit([(t, "label", spec.name, True, False) for t in spec.labels()], False)
    for text in doc.other_labels:
        unit([(text, "other_label", None, True, True)], False)
    prof = _Profile(doc, units)
    prof.keys = {p.key for u in units for p in u.phrases}
    return prof


# ---------------------------------------------------------------------------- search

@dataclass
class _Found:
    key: tuple[str, ...]
    page: int


def _pieces(word: str, edits: int) -> list[str]:
    """`word` cut into `edits + 1` pieces: a reading of it with at most `edits` edits contains
    at least one of them unchanged."""
    n = edits + 1
    step = len(word) / n
    return [word[round(i * step):round((i + 1) * step)] for i in range(n)]


def _could_occur(variant, text: str) -> bool:
    """Can the matcher possibly find `variant` in a page whose canonical words, run together,
    are `text`? Each word must keep one of its pieces exact under the most edits any of the
    matcher's comparisons allows it (its own budget, and one for a label compared as one joined
    string). A conservative filter: it only skips phrases that cannot match, so the search
    costs time for the phrases the page could hold, not for every phrase of every candidate."""
    from dococr.ocr.matcher import _allowed_edits

    joined_budget = 0 if len(variant.joined) <= 5 else 1
    for w in variant.words:
        edits = max(_allowed_edits(len(w)), joined_budget)
        if edits >= len(w):
            continue
        if not any(piece and piece in text for piece in _pieces(w, edits)):
            return False
    return True


def _find_on_page(lines: list, size: tuple[float, float], row_vocab: list, flow_vocab: list,
                  glue: tuple[frozenset, frozenset]) -> set[tuple[str, ...]]:
    """Keys of the phrases found on one page."""
    from dococr.ocr.matcher import (
        Layout,
        canon,
        find_flow_hits,
        find_label_hits,
        group_rows,
        tokenize,
    )

    if not lines:
        return set()
    text = "".join(canon(tok) for ln in lines for tok in ln.text.split())
    row_vocab = [v for v in row_vocab if _could_occur(v, text)]
    flow_vocab = [v for v in flow_vocab if _could_occur(v, text)]
    words = tokenize(lines, row_vocab, glue)
    rows = group_rows(lines, words)
    found: set[tuple[str, ...]] = set()
    claimed: dict[int, tuple[str, ...]] = {}  # id(word) -> the row label that holds it
    for hit in find_label_hits(rows, row_vocab):
        for r, k in hit.cells():
            claimed[id(rows[r][k])] = hit.variant.words
        if hit.fragment_of is None:
            found.add(hit.variant.words)
    if flow_vocab:
        page_w = size[0] if size and size[0] else max((ln.box[2] for ln in lines), default=1.0)
        for hit in find_flow_hits(Layout(rows, page_w), flow_vocab, 0):
            key = hit.variant.words
            # Inside a longer row label of something else, as the row search already decided.
            if all(id(w) in claimed and len("".join(claimed[id(w)])) > len("".join(key))
                   for w in hit.words):
                continue
            found.add(key)
    return found


def _vocabularies(profiles: list[_Profile]):
    from dococr.ocr.matcher import Variant, _anchor_glue

    row: dict[tuple[str, ...], Variant] = {}
    flow: dict[tuple[str, ...], Variant] = {}
    heads: set[str] = set()
    tails: set[str] = set()
    for prof in profiles:
        h, t = _anchor_glue(prof.doc, prof.doc.fields)
        heads |= h
        tails |= t
        for u in prof.units:
            for p in u.phrases:
                tag = " ".join(p.key)
                if p.row and p.key not in row:
                    row[p.key] = Variant(p.key, "".join(p.key), p.text, {tag})
                if p.flow and p.key not in flow:
                    flow[p.key] = Variant(p.key, "".join(p.key), p.text, {tag})
    return list(row.values()), list(flow.values()), (frozenset(heads), frozenset(tails))


# ---------------------------------------------------------------------------- scoring

def _dominates(a: str, b: str, listed: dict, seen: dict) -> bool:
    """Does the page favour type `a` over type `b` outright? It does when every phrase found
    for `b` is one `a` lists too, and something found for `a` is not listed by `b`: then `a`
    explains all the evidence for `b` and more. This separates two types that share most of
    their labels when the label that differs is printed, however close their scores."""
    return seen[b] <= listed[a] and bool(seen[a] - listed[b])


def detect_lines(pages: Sequence, docs: Sequence[DocType], warnings: Sequence[str] = ()) -> Detection:
    """Choose among `docs` for a document whose pages were read into `pages`
    (`extractors.base.PageRead`, or anything with `lines` and `size`). Only the first
    `DETECT_PAGES` pages are used."""
    profiles = [_profile(d) for d in docs]
    row_vocab, flow_vocab, glue = _vocabularies(profiles)
    used = list(pages)[:DETECT_PAGES]
    first_page: dict[tuple[str, ...], int] = {}
    for n, page in enumerate(used, start=1):
        for key in _find_on_page(page.lines, page.size, row_vocab, flow_vocab, glue):
            first_page.setdefault(key, n)

    n_types = len(profiles)
    df: dict[tuple[str, ...], int] = {}
    for prof in profiles:
        for key in prof.keys:
            df[key] = df.get(key, 0) + 1
    norm = math.log(n_types + 1)

    def weight(key) -> float:
        return math.log((n_types + 1) / df[key]) / norm

    found_mass = sum(weight(k) for k in first_page if k in df)

    ranked: list[DetectionCandidate] = []
    listed: dict[str, set] = {}  # type -> keys of the phrases it lists
    seen: dict[str, set] = {}  # type -> keys of its phrases found on the page
    units_found: dict[str, int] = {}  # type -> units found, identifying ones included
    for prof in profiles:
        credit = total = ident = 0.0
        found = count = 0
        evidence: list[Evidence] = []
        for u in prof.units:
            hits = [p for p in u.phrases if p.key in first_page]
            if not u.ident:
                count += 1
                total += max(weight(p.key) for p in u.phrases)
            if not hits:
                continue
            best = max(hits, key=lambda p: weight(p.key))
            if u.ident:
                ident += weight(best.key)
            else:
                credit += weight(best.key)
                found += 1
            evidence.append(Evidence(text=best.text, kind=best.kind, field=best.field,
                                     weight=round(weight(best.key), 3), page=first_page[best.key]))
        coverage = (credit + ident) / (total + ident) if total + ident > 0 else 0.0
        mine = sum(weight(k) for k in first_page if k in prof.keys)
        precision = mine / found_mass if found_mass > 0 else 0.0
        score = 2 * coverage * precision / (coverage + precision) if coverage + precision > 0 else 0.0
        evidence.sort(key=lambda e: -e.weight)
        ranked.append(DetectionCandidate(
            doc_type=prof.doc.name, title=prof.doc.title, score=round(score, 3),
            coverage=round(coverage, 3), precision=round(precision, 3),
            found=found, total=count, evidence=evidence))
        listed[prof.doc.name] = prof.keys
        seen[prof.doc.name] = {k for k in first_page if k in prof.keys}
        units_found[prof.doc.name] = found + sum(
            1 for e in evidence if e.kind in ("identifying_text", "title"))
    ranked.sort(key=lambda c: (-c.score, c.doc_type))
    best = ranked[0]
    second = ranked[1].score if len(ranked) > 1 else 0.0
    result = Detection(score=best.score, runner_up=second, candidates=ranked[:TOP],
                       considered=n_types, pages=len(used), warnings=list(warnings))
    if not any(p.lines for p in used):
        result.reason = "no text was found on the page"
    elif best.score < MIN_SCORE:
        result.reason = (f"no candidate matches well enough (best: '{best.doc_type}' scores "
                         f"{best.score:.2f}, below {MIN_SCORE:.2f})")
    elif units_found[best.doc_type] < MIN_FOUND:
        result.reason = (f"too little evidence ('{best.doc_type}': {units_found[best.doc_type]} of its "
                         f"labels or texts found, at least {MIN_FOUND} are needed)")
    elif close := [c for c in ranked[1:] if c.score > MAX_RUNNER_UP * best.score
                   and not _dominates(best.doc_type, c.doc_type, listed, seen)]:
        result.reason = (f"'{best.doc_type}' ({best.score:.2f}) and '{close[0].doc_type}' "
                         f"({close[0].score:.2f}) match about equally well")
    else:
        result.doc_type = best.doc_type
        result.reason = (f"'{best.doc_type}' scores {best.score:.2f} with {best.found} labels or texts "
                         f"found" + (f"; the next, '{ranked[1].doc_type}', {second:.2f}"
                                     if len(ranked) > 1 else ""))
    return result
