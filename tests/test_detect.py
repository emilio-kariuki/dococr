"""Detecting the document type (`dococr.detection`), with hand-built OCR lines and made-up types.
No OCR model. Every type, label and value here is invented."""

from __future__ import annotations

import pytest
from PIL import Image

from dococr import (
    DocType,
    Extractor,
    ExtractorOutput,
    FieldResult,
    FieldSpec,
    UnknownDocumentError,
    detect,
    extract,
)
from dococr import detection as D
from dococr.extractors.base import PageRead
from dococr.extractors.local import LocalExtractor
from dococr.ocr import OcrLine


def line(y: float, x: float, text: str, conf: float = 0.95) -> OcrLine:
    return OcrLine(text, conf, (x, y, x + 13 * len(text), y + 24))


def form(*rows: tuple[str, str], title: str = "") -> list[OcrLine]:
    """A form page: an optional title, then one 'label   value' row per pair."""
    out = [line(20, 300, title)] if title else []
    for k, (label, value) in enumerate(rows):
        y = 80 + 50 * k
        out += [line(y, 40, label), line(y, 420, value)]
    return out


class Page:
    def __init__(self, lines, size=(1000, 800)):
        self.lines, self.size = lines, size


def doc(name: str, labels: list[str], other=(), ident=(), title: str = "") -> DocType:
    return DocType(name=name, title=title or name, other_labels=list(other),
                   identifying_text=list(ident),
                   fields=[FieldSpec(name=f"f{i}", label=lab) for i, lab in enumerate(labels)])


# Two invented forms that share two common labels.
PERMIT = doc("permit", ["Permit Code", "Holder Name", "Date of Birth", "Grazing Zone", "Herd Size",
                        "Valid From"])
TICKET = doc("ticket", ["Ticket Serial", "Holder Name", "Date of Birth", "Seat Row", "Gate Letter",
                        "Valid From"])


def permit_page():
    return form(("PERMIT CODE", "PX-55102"), ("HOLDER NAME", "ZANELE OKAFOR"),
                ("DATE OF BIRTH", "03.04.1991"), ("GRAZING ZONE", "NORTH 4"), ("HERD SIZE", "37"))


def test_picks_the_type_whose_labels_are_printed():
    d = D.detect_lines([Page(permit_page())], [PERMIT, TICKET])
    assert d.doc_type == "permit", d.reason
    assert [c.doc_type for c in d.candidates] == ["permit", "ticket"]
    best = d.candidates[0]
    assert {e.text for e in best.evidence} >= {"Permit Code", "Grazing Zone", "Herd Size"}
    assert best.found == 5 and best.total == 6


def test_shared_labels_weigh_less_than_distinctive_ones():
    d = D.detect_lines([Page(permit_page())], [PERMIT, TICKET])
    weights = {e.text: e.weight for e in d.candidates[0].evidence}
    assert weights["Permit Code"] == 1.0
    assert weights["Date of Birth"] < 0.5
    assert weights["Holder Name"] == weights["Date of Birth"]
    # The other type found the shared labels only, so it scores low.
    ticket = d.candidates[1]
    assert {e.text for e in ticket.evidence} == {"Holder Name", "Date of Birth"}
    assert ticket.score < D.MIN_SCORE


def test_a_type_does_not_win_by_listing_more_labels():
    small = doc("small", ["Permit Code", "Grazing Zone", "Herd Size", "Holder Name"])
    big = doc("big", ["Permit Code", "Grazing Zone", "Herd Size", "Holder Name"]
              + [f"Extra Label {w}" for w in "abcdefghijklmnop"])
    d = D.detect_lines([Page(permit_page())], [big, small])
    assert d.doc_type == "small", d.reason
    by = {c.doc_type: c for c in d.candidates}
    assert by["small"].coverage == 1.0 and by["big"].coverage < 0.3


def test_unrelated_page_is_unknown():
    page = form(("INVOICE NO", "88"), ("AMOUNT DUE", "1,200.00"), ("DATE OF BIRTH", "01.01.1990"))
    d = D.detect_lines([Page(page)], [PERMIT, TICKET])
    assert d.doc_type is None
    assert "no candidate matches well enough" in d.reason
    assert d.candidates and d.candidates[0].score < D.MIN_SCORE


def test_too_little_evidence_is_unknown():
    tiny = doc("tiny", ["Permit Code", "Grazing Zone"])
    d = D.detect_lines([Page(form(("PERMIT CODE", "PX-1"), ("GRAZING ZONE", "W")))], [tiny])
    assert d.candidates[0].score == 1.0
    assert d.doc_type is None and "too little evidence" in d.reason


def test_two_types_that_print_the_same_labels_are_unknown():
    twin = PERMIT.model_copy(update={"name": "permit_copy", "title": "permit copy"})
    d = D.detect_lines([Page(permit_page())], [PERMIT, twin])
    assert d.doc_type is None
    assert "about equally" in d.reason
    assert {c.doc_type for c in d.candidates[:2]} == {"permit", "permit_copy"}


def test_one_differing_label_separates_two_near_identical_types():
    shared = [f"Shared Item {c}" for c in "ABCDEFGHJKLMNP"]
    a = doc("route_a", [*shared, "North Gate Pass"])
    b = doc("route_b", [*shared, "South Gate Pass"])
    page = [ln for k, lab in enumerate([*shared, "North Gate Pass"])
            for ln in (line(40 + 40 * k, 40, lab.upper()), line(40 + 40 * k, 500, f"V{k}"))]
    d = D.detect_lines([Page(page, (1000, 1000))], [a, b])
    # Both explain the shared labels; only one explains the printed gate label, and the other
    # is charged for text it cannot explain.
    by = {c.doc_type: c for c in d.candidates}
    assert by["route_a"].precision == 1.0 and by["route_b"].precision < 1.0
    assert by["route_a"].score > by["route_b"].score
    # Close scores, but route_a explains everything found for route_b and more: chosen.
    assert by["route_b"].score > D.MAX_RUNNER_UP * by["route_a"].score
    assert d.doc_type == "route_a", d.reason
    # Without the differing label printed, neither explains more: unknown.
    d = D.detect_lines([Page(page[:-2], (1000, 1000))], [a, b])
    assert d.doc_type is None and "about equally" in d.reason


def test_identifying_text_counts_when_found_and_not_when_missing():
    labels = [f.label for f in PERMIT.fields]
    with_head = doc("permit", labels, ident=["Pastoral Grazing Permit"])
    plain = doc("permit", labels)
    page = permit_page()
    missing = D.detect_lines([Page(page)], [with_head, TICKET]).candidates[0]
    without = D.detect_lines([Page(page)], [plain, TICKET]).candidates[0]
    assert missing.score == without.score  # the heading was not printed: no penalty
    headed = D.detect_lines([Page([line(20, 300, "PASTORAL GRAZING PERMIT"), *page])],
                            [with_head, TICKET]).candidates[0]
    assert headed.score > missing.score
    assert any(e.kind == "identifying_text" for e in headed.evidence)


def test_identifying_text_across_a_line_break():
    t = doc("lease", ["Lessor Name", "Lessee Name", "Monthly Rent"],
            ident=["Agreement for the Lease of Grazing Land"])
    lines = [line(20, 40, "THIS IS AN AGREEMENT FOR THE LEASE"), line(50, 40, "OF GRAZING LAND MADE TODAY"),
             *form(("LESSOR NAME", "A"), ("LESSEE NAME", "B"), ("MONTHLY RENT", "5"))]
    d = D.detect_lines([Page(lines)], [t, PERMIT])
    assert any(e.kind == "identifying_text" for e in d.candidates[0].evidence)


def test_short_words_of_sentence_labels_are_not_evidence():
    prose = DocType(name="deed", fields=[FieldSpec(name="party", label="of", until="and")])
    page = [line(20, 40, "a field of maize and of beans and of millet")]
    d = D.detect_lines([Page(page)], [prose, PERMIT])
    assert all(c.found == 0 for c in d.candidates)
    assert d.doc_type is None


def test_evidence_is_pooled_over_pages_and_bounded():
    page1 = form(("PERMIT CODE", "PX-1"), ("GRAZING ZONE", "W"))
    page2 = form(("HERD SIZE", "12"), ("VALID FROM", "01.02.2020"))
    d = D.detect_lines([Page(page1), Page(page2)], [PERMIT, TICKET])
    assert d.doc_type == "permit" and d.pages == 2
    assert {e.page for e in d.candidates[0].evidence} == {1, 2}
    many = [Page([])] * D.DETECT_PAGES + [Page(permit_page())]
    late = D.detect_lines(many, [PERMIT, TICKET])
    assert late.doc_type is None and late.pages == D.DETECT_PAGES


def test_blank_page_is_unknown():
    d = D.detect_lines([Page([])], [PERMIT, TICKET])
    assert d.doc_type is None and d.reason == "no text was found on the page"


# ---------------------------------------------------------------------------- candidates and cap

def write_types(folder, n, start=0):
    for i in range(start, start + n):
        (folder / f"t{i}.yaml").write_text(
            f"name: t{i}\nfields:\n  - name: code\n    label: Code {i}\n", encoding="utf-8")


def test_candidate_cap_refuses_a_large_folder(tmp_path, monkeypatch):
    write_types(tmp_path, 12)
    monkeypatch.setenv("DOCOCR_DOCTYPES_DIR", str(tmp_path))
    monkeypatch.setenv(D.MAX_CANDIDATES_ENV, "10")
    with pytest.raises(ValueError, match="more than the 10"):
        D.candidate_types()
    monkeypatch.setenv(D.MAX_CANDIDATES_ENV, "100")
    docs, _ = D.candidate_types()
    assert len(docs) >= 12


def test_named_candidates(tmp_path, monkeypatch):
    write_types(tmp_path, 3)
    monkeypatch.setenv("DOCOCR_DOCTYPES_DIR", str(tmp_path))
    docs, _ = D.candidate_types(["t0", "t2", "t0"])
    assert [d.name for d in docs] == ["t0", "t2"]
    with pytest.raises(ValueError, match="unknown document type"):
        D.candidate_types(["t0", "no_such_type"])
    with pytest.raises(ValueError, match="not a document type"):
        D.candidate_types(["auto"])
    with pytest.raises(ValueError, match="limit is 2"):
        D.candidate_types(["t0", "t1", "t2"], limit=2)


def test_broken_files_are_reported(tmp_path, monkeypatch):
    write_types(tmp_path, 2)
    (tmp_path / "broken.yaml").write_text("name: [unclosed\n", encoding="utf-8")
    monkeypatch.setenv("DOCOCR_DOCTYPES_DIR", str(tmp_path))
    docs, warnings = D.candidate_types()
    assert {"t0", "t1"} <= {d.name for d in docs}
    assert warnings and "could not be read" in warnings[0]


def test_cost_is_bounded_by_the_cap():
    """Scoring 50 types against a dense page stays far below the cost of reading it."""
    import time

    types = [doc(f"type{i}", [f"Label {i} {w}" for w in ("alpha", "beta", "gamma", "delta", "epsilon",
                                                        "zeta", "eta", "theta", "iota", "kappa")],
                 other=[f"Other {i} {w}" for w in ("one", "two", "three", "four", "five")],
                 ident=[f"Heading number {i} of the series"])
             for i in range(D.DEFAULT_MAX_CANDIDATES)]
    page = [line(10 + 30 * k, 40, " ".join(f"word{k}{j}" for j in range(12))) for k in range(30)]
    started = time.perf_counter()
    D.detect_lines([Page(page, (2000, 1200))], types)
    assert time.perf_counter() - started < 3.0


# ---------------------------------------------------------------------------- pipeline

class Reader(LocalExtractor):
    """The real matcher on given lines: `read_page` returns the next prepared page."""

    name = "reader"

    def __init__(self, pages):
        super().__init__(engine=object())
        self.pages = list(pages)
        self.reads = 0

    def read_page(self, image, *, text_layer=None):
        self.reads += 1
        return PageRead(self.pages[(self.reads - 1) % len(self.pages)], (1000, 800))


@pytest.fixture
def image():
    return Image.new("RGB", (200, 100), "white")


def test_auto_extraction_reads_each_page_once(image):
    reader = Reader([permit_page()])
    result = extract(image, candidates=[PERMIT, TICKET], extractor=reader)
    assert reader.reads == 1
    assert result.doc_type == "permit"
    assert result.detection is not None and result.detection.doc_type == "permit"
    assert result.values()["f0"] == "PX-55102"
    assert result.values()["f2"] == "1991-04-03" or result.values()["f2"] == "03.04.1991"
    explicit = extract(image, PERMIT, extractor=Reader([permit_page()]))
    assert explicit.detection is None
    assert explicit.values() == result.values()


def test_auto_is_the_default_and_none_means_auto(image):
    for doc_type in ("auto", "AUTO", None):
        result = extract(image, doc_type, candidates=[PERMIT, TICKET], extractor=Reader([permit_page()]))
        assert result.doc_type == "permit"


def test_unknown_document_raises_with_the_ranking(image):
    reader = Reader([form(("INVOICE NO", "88"), ("AMOUNT DUE", "12"))])
    with pytest.raises(UnknownDocumentError) as err:
        extract(image, "auto", candidates=[PERMIT, TICKET], extractor=reader)
    assert isinstance(err.value, ValueError)
    assert err.value.detection.doc_type is None
    assert {c.doc_type for c in err.value.detection.candidates} == {"permit", "ticket"}


def test_detect_reads_only_the_first_pages(image):
    reader = Reader([permit_page()])
    d = detect([image] * 6, [PERMIT, TICKET], extractor=reader)
    assert d.doc_type == "permit" and reader.reads == D.DETECT_PAGES and d.pages == D.DETECT_PAGES


def test_auto_extraction_of_several_pages_reads_each_once(image):
    reader = Reader([form(("PERMIT CODE", "PX-1"), ("GRAZING ZONE", "W"), ("HOLDER NAME", "Z O")),
                     form(("HERD SIZE", "12"), ("VALID FROM", "01.02.2020"))])
    result = extract([image, image], candidates=[PERMIT, TICKET], extractor=reader)
    assert reader.reads == 2 and result.pages == 2
    assert result.fields["f4"].page == 2 and result.values()["f4"] == "12"


class NoLines(Extractor):
    name = "nolines"

    def __init__(self):
        self.calls = 0

    def extract(self, image, doc_type, fields):
        self.calls += 1
        return ExtractorOutput(fields=[FieldResult(name=f.name) for f in fields])


def test_an_extractor_without_lines_cannot_detect(image):
    ex = NoLines()
    with pytest.raises(ValueError, match="cannot detect the document type"):
        extract(image, candidates=[PERMIT], extractor=ex)
    with pytest.raises(ValueError, match="cannot detect the document type"):
        detect(image, [PERMIT], extractor=ex)
    assert ex.calls == 0
    assert not ex.supports_reading and Reader([[]]).supports_reading
    # Naming the type still works.
    assert extract(image, PERMIT, extractor=ex).doc_type == "permit"


def test_detection_holds_no_document_knowledge():
    """The detector's code names no shipped type and none of their labels."""
    import inspect
    import re

    from dococr import list_doc_types, load_doc_type

    source = inspect.getsource(D).lower()
    for name in list_doc_types():
        t = load_doc_type(name)
        for text in [t.name, *t.aliases, *t.identifying_text, *t.other_labels,
                     *(lab for f in t.fields for lab in f.labels())]:
            if " " in text.strip() or "_" in text:
                assert not re.search(rf"\b{re.escape(text.lower())}\b", source), text
