"""Values read across a page break: page furniture, joining the bottom of one page to the top of
the next, the result shape, and what must not change. Fast: hand-built OCR lines and a fake
engine; the digital PDF fixture needs no OCR."""

from __future__ import annotations

import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pdfs  # noqa: E402
import synth_pdf  # noqa: E402

from dococr import DocType, FieldSpec, extract  # noqa: E402
from dococr.extractors.base import PageLines  # noqa: E402
from dococr.extractors.local import LocalExtractor  # noqa: E402
from dococr.ocr import OcrEngine, OcrLine  # noqa: E402
from dococr.pagebreak import find_furniture, join_pair  # noqa: E402

W, H = 1200, 1700
LH = 30  # text height
PITCH = 44


def L(text: str, x: float, y: float, conf: float = 0.97, h: float = LH) -> OcrLine:
    return OcrLine(text, conf, (x, y, x + 15 * len(text), y + h))


def page(*lines: OcrLine) -> PageLines:
    return PageLines(list(lines), (W, H))


def para(texts, y0, x=100):
    return [L(t, x, y0 + k * PITCH) for k, t in enumerate(texts)]


class PageEngine(OcrEngine):
    """Returns the lines of the page it is shown, told apart by its solid colour."""

    def __init__(self, pages):
        self.pages = pages

    def read(self, image):
        return list(self.pages[pdfs.page_index(Image.fromarray(image))])


def run(pages, doc, fields=None, **kw):
    images = [Image.new("RGB", (W, H), pdfs.COLOURS[i]) for i in range(len(pages))]
    return extract(images, doc, fields, extractor=LocalExtractor(engine=PageEngine(pages)), **kw)


def header(n, of=2):
    return [L("SALE AGREEMENT", 100, 40, h=20), L(f"Page {n} of {of}", 950, 1630, h=20)]


AGREEMENT = DocType(name="agreement", fields=[
    FieldSpec(name="vendor_address", label="of", until="(the Vendor)"),
    FieldSpec(name="completion_date", label="on or before the", until="by bank transfer", type="date"),
    FieldSpec(name="witness", label="in the presence of", until="Advocate", type="name"),
])


# ---- furniture ---------------------------------------------------------------------------------

def test_page_numbers_and_running_header_are_furniture():
    p1 = page(L("SALE AGREEMENT", 100, 40), *para(["body one", "body two"], 300), L("Page 1 of 3", 550, 1640))
    p2 = page(L("SALE AGREEMENT", 100, 42), *para(["more body", "and more"], 300), L("- 2 -", 580, 1640))
    p3 = page(L("SALE AGREEMENT", 100, 41), *para(["last body"], 300), L("3", 1100, 1645))
    f = find_furniture([p1, p2, p3])
    assert f[0] == {0, 3} and f[1] == {0, 3} and f[2] == {0, 2}


def test_repeated_footer_with_different_numbers_is_furniture():
    pages = [page(*para(["body text here"], 300), L(f"Vendor initials ____ Ref 00{n}", 100, 1600))
             for n in range(1, 3)]
    assert find_furniture(pages) == [{1}, {1}]


def test_body_text_at_the_edge_is_not_furniture():
    p1 = page(*para(["The Vendor agrees to sell the land", "and the Purchaser agrees to buy it"], 1500))
    p2 = page(*para(["situated at Plot No. 1234 Nakuru.", "The said land is registered"], 60))
    assert find_furniture([p1, p2]) == [set(), set()]


def test_a_number_alone_must_be_the_page_number_or_repeated():
    p1 = page(*para(["body"], 300), L("00100", 550, 1640))  # a postal code at the bottom
    p2 = page(*para(["body"], 300))
    assert find_furniture([p1, p2]) == [set(), set()]
    p1 = page(*para(["body"], 300), L("1", 550, 1640))
    assert find_furniture([p1, p2]) == [{1}, set()]


def test_peeling_stops_at_the_first_row_that_is_not_furniture():
    # the page number is above a line of body text that runs to the edge: nothing is taken
    p1 = page(*para(["body"], 300), L("Page 1 of 2", 550, 1560), L("closing words of the body", 100, 1640))
    assert find_furniture([p1, page(L("x", 100, 300))])[0] == set()


def test_a_rule_of_underscores_at_the_edge_is_furniture():
    p1 = page(*para(["body"], 300), L("________________", 100, 1640))
    assert find_furniture([p1])[0] == {1}


def test_yaml_page_furniture_names_text_the_detector_cannot_know():
    p1 = page(*para(["body"], 300), L("Continued overleaf", 900, 1640))
    p2 = page(*para(["body"], 300))
    assert find_furniture([p1, p2]) == [set(), set()]
    assert find_furniture([p1, p2], ["continued overleaf"]) == [{1}, set()]


# ---- joining -----------------------------------------------------------------------------------

def two_page_agreement():
    p1 = header(1) + para([
        "THIS AGREEMENT is made between JOHN KAMAU of",
        "P.O. Box 4410-00506 Nairobi (the Vendor) and the",
        "balance shall be paid on or before the Twelfth",
    ], 1400)
    p2 = header(2) + para([
        "day of March 2024 by bank transfer. SIGNED in the",
        "presence of MARY ATIENO Advocate.",
    ], 200)
    return p1, p2


def test_sentence_value_across_a_page_break_skips_furniture():
    p1, p2 = two_page_agreement()
    res = run([p1, p2], AGREEMENT)
    f = res.fields["completion_date"]
    assert f.value == "2024-03-12"
    assert f.page == 1 and f.raw == "Twelfth day of March 2024"
    assert [p.page for p in f.parts] == [1, 2]
    assert f.parts[0].bbox == f.bbox
    assert f.parts[1].bbox[1] == 200  # the top of page 2's first body line
    # values inside one page are as before
    assert res.fields["vendor_address"].value == "P.O. Box 4410-00506 Nairobi"
    assert res.fields["vendor_address"].parts is None
    assert res.fields["witness"].value == "MARY ATIENO" and res.fields["witness"].page == 2


def test_result_json_has_parts():
    p1, p2 = two_page_agreement()
    data = run([p1, p2], AGREEMENT).model_dump(mode="json")
    parts = data["fields"]["completion_date"]["parts"]
    assert [p["page"] for p in parts] == [1, 2] and len(parts[0]["bbox"]) == 4
    assert data["fields"]["witness"]["parts"] is None


def test_without_the_furniture_skip_the_page_number_would_be_read():
    # the same pages with the page number and header inside the window: joined around them
    p1, p2 = two_page_agreement()
    pl1, pl2 = page(*p1), page(*p2)
    joined = join_pair(pl1, pl2, set(), set(), AGREEMENT, AGREEMENT.fields)
    assert "completion_date" not in joined or joined["completion_date"].raw != "Twelfth day of March 2024"
    f = find_furniture([pl1, pl2])
    joined = join_pair(pl1, pl2, f[0], f[1], AGREEMENT, AGREEMENT.fields)
    assert joined["completion_date"].raw == "Twelfth day of March 2024"


def test_label_at_the_foot_of_a_page_and_value_on_the_next():
    p1 = header(1) + para(["The balance is payable on or before the"], 1500)
    p2 = header(2) + para(["First day of May 2025 by bank transfer."], 200)
    f = run([p1, p2], AGREEMENT, ["completion_date"]).fields["completion_date"]
    assert f.value == "2025-05-01" and f.page == 2 and f.parts is None


def test_end_text_at_the_top_of_the_next_page():
    p1 = header(1) + para(["The balance is payable on or before the First day of May 2025"], 1500)
    p2 = header(2) + para(["by bank transfer and not otherwise."], 200)
    f = run([p1, p2], AGREEMENT, ["completion_date"]).fields["completion_date"]
    assert f.value == "2025-05-01" and f.page == 1 and f.parts is None
    assert f.warnings == []  # the fallback's "end text not found" is replaced


def test_multiline_block_continues_on_the_next_page():
    doc = DocType(name="form", fields=[
        FieldSpec(name="postal_address", label="Postal Address", multiline=True),
        FieldSpec(name="county", label="County"),
    ])
    p1 = [L("Postal Address", 100, 1500), L("P.O. BOX 123", 400, 1500), L("MOI AVENUE", 400, 1544),
          L("Page 1 of 2", 550, 1640, h=20)]
    p2 = [L("NAIROBI", 400, 150), L("County", 100, 220), L("NAKURU", 400, 220), L("Page 2 of 2", 550, 1640, h=20)]
    res = run([p1, p2], doc)
    f = res.fields["postal_address"]
    assert f.value == "P.O. BOX 123 MOI AVENUE NAIROBI"
    assert [p.page for p in f.parts] == [1, 2]
    assert any("next page" in w for w in f.warnings)
    assert res.fields["county"].value == "NAKURU"


def test_form_values_are_not_read_across_a_break():
    doc = DocType(name="form", fields=[FieldSpec(name="occupation", label="Occupation"),
                                       FieldSpec(name="county", label="County")])
    p1 = [L("Occupation", 100, 1500), L("Page 1 of 2", 550, 1640, h=20)]  # value missing on page 1
    p2 = [L("TEACHER", 100, 150), L("County", 100, 300), L("NAKURU", 400, 300), L("Page 2 of 2", 550, 1640, h=20)]
    res = run([p1, p2], doc)
    assert res.fields["occupation"].value is None
    # and a label on page 2 does not take a value from page 1
    p1 = [L("NAKURU", 400, 1560), L("Page 1 of 2", 550, 1640, h=20)]
    p2 = [L("County", 100, 150), L("Page 2 of 2", 550, 1640, h=20)]
    assert run([p1, p2], doc).fields["county"].value is None


def test_a_joined_value_does_not_take_another_fields_words():
    doc = DocType(name="t", fields=[
        FieldSpec(name="price", label="the sum of", until="(KES"),
        FieldSpec(name="total", label="Sum"),  # a form field whose value is on page 2
    ])
    p1 = para(["The Purchaser shall pay the sum of Kenya"], 1500) + [L("Page 1 of 2", 550, 1640, h=20)]
    p2 = [L("Sum", 100, 200), L("Ten Thousand (KES 10)", 200, 200), L("Page 2 of 2", 550, 1640, h=20)]
    res = run([p1, p2], doc)
    assert res.fields["total"].value == "Ten Thousand (KES 10)"
    assert res.fields["price"].value is None
    assert any("read as field 'total'" in w for w in res.fields["price"].warnings)


def test_per_page_merge_rule_still_applies_within_pages():
    p1 = para(["SIGNED in the presence of MARY ATIENO Advocate."], 300)
    p2 = para(["SIGNED in the presence of PETER OTIENO Advocate."], 300, x=100)
    p2 = [OcrLine(ln.text, 0.99, ln.box) for ln in p2]
    f = run([p1, p2], AGREEMENT, ["witness"]).fields["witness"]
    assert f.value == "PETER OTIENO" and f.page == 2 and f.parts is None
    assert "page 1 read a different value than page 2" in f.warnings[0]


def test_pages_of_very_different_text_sizes_are_not_joined():
    p1 = header(1) + para(["The balance is payable on or before the Twelfth"], 1500)
    p2 = header(2) + [L("day of March 2024 by bank transfer.", 100, 200, h=80)]
    f = run([p1, p2], AGREEMENT, ["completion_date"]).fields["completion_date"]
    assert f.parts is None


def test_single_image_is_unchanged_by_the_page_break_code():
    p1, _ = two_page_agreement()
    res = run([p1], AGREEMENT)
    assert res.fields["completion_date"].parts is None
    assert res.fields["vendor_address"].value == "P.O. Box 4410-00506 Nairobi"


# ---- the digital fixture -----------------------------------------------------------------------

class NoOcr(OcrEngine):
    def read(self, image):
        raise AssertionError("OCR was run")


@pytest.mark.parametrize("seed", [3, 4, 6])  # 0, 1, 2, 5: in test_pdf_text.py
def test_digital_agreement_values_crossing_breaks(seed):
    doc = synth_pdf.make_pdf_document(seed)
    res = extract(synth_pdf.build(doc, "digital"), synth_pdf.doc_type_ref(),
                  extractor=LocalExtractor(engine=NoOcr()))
    for name in doc.crossing:
        f = res.fields[name]
        assert f.value == doc.truth[name], name
        assert f.parts is None or len({p.page for p in f.parts}) == 2
