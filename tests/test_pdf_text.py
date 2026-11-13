"""Reading a PDF's own text layer: lines, the trust decision per page, rotation and cropping, the
`pdf_text` setting, and a whole digital document read without OCR. Fast: no OCR model (an engine
that fails when called proves that none was needed)."""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pytest
from PIL import Image, ImageDraw

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pdfs  # noqa: E402
import synth  # noqa: E402
import synth_pdf  # noqa: E402
from pdfs import PdfPage, Picture, Rule, Text  # noqa: E402

from dococr import Extractor, ExtractorOutput, FieldResult, extract  # noqa: E402
from dococr.extractors.local import LocalExtractor  # noqa: E402
from dococr.imaging import RENDER_LONG_SIDE, open_pages  # noqa: E402
from dococr.ocr import OcrEngine, OcrLine  # noqa: E402
from dococr.pdftext import CONF_INVISIBLE, build_layer  # noqa: E402

SCALE = RENDER_LONG_SIDE / 842  # A4 portrait rendered to 2000 px high


class NoOcr(OcrEngine):
    """An OCR engine that must not be used."""

    def read(self, image):
        raise AssertionError("OCR was run")

    def detect(self, image):
        raise AssertionError("OCR was run")

    def warm_up(self):
        raise AssertionError("OCR was loaded")


class CropEngine(OcrEngine):
    """Answers with fixed lines (in the coordinates of whatever image it is given) and records
    the sizes of the images it read."""

    def __init__(self, lines):
        self.lines = lines
        self.shapes = []

    def read(self, image):
        self.shapes.append(image.shape)
        return list(self.lines)


def layer_of(data: bytes, page: int = 0, mode: str = "auto"):
    with open_pages(data) as pages:
        p = list(pages.with_text())[page]
        return build_layer(p.text, p.image, mode), p


def px(x_pt: float, y_pt: float, height_pt: float = 842) -> tuple[float, float]:
    """A point on an unrotated, uncropped page in rendered pixels (y down)."""
    return x_pt * SCALE, (height_pt - y_pt) * SCALE


# ---- lines --------------------------------------------------------------------------------------

def test_lines_one_per_visual_line_split_at_wide_gaps():
    page = PdfPage(items=[
        Text(72, 700, "Full Name", 11), Text(250, 700, "JOSEPHINE WANJIRU", 11),  # a form row
        Text(72, 680, "The purchase price is Kenya", 11),
        Text(72 + pdfs.text_width("The purchase price is Kenya ", 11), 680, "Shillings Five", 11),  # one line, two runs
        Text(72, 660, "regis-", 11),  # a hyphen at the end of the line
        Text(72, 645, "tered owner", 11),
    ])
    layer, _ = layer_of(pdfs.text_pdf([page]))
    assert layer.kind == "text" and layer.regions == []
    assert [ln.text for ln in layer.lines] == [
        "Full Name", "JOSEPHINE WANJIRU", "The purchase price is Kenya Shillings Five", "regis-", "tered owner"]
    assert all(ln.confidence == 1.0 for ln in layer.lines)
    # boxes are in the rendered page's pixels; the baseline sits inside the box
    x, y = px(72, 700)
    box = layer.lines[0].box
    assert abs(box[0] - x) < 2 and box[1] < y < box[3]
    assert box[3] - box[1] == pytest.approx(11 * 1.164 * SCALE, rel=0.05)  # ascent + descent
    assert layer.lines[1].box[0] == pytest.approx(250 * SCALE, abs=2)


def test_words_separated_without_space_characters():
    # a writer that places each word on its own, with no space characters between them
    items, x = [], 72.0
    for word in ["made", "on", "the", "First"]:
        items.append(Text(x, 700, word, 11))
        x += pdfs.text_width(word + " ", 11)
    layer, _ = layer_of(pdfs.text_pdf([PdfPage(items=items)]))
    assert [ln.text for ln in layer.lines] == ["made on the First"]


def test_text_drawn_twice_is_read_once():
    page = PdfPage(items=[Text(72, 700, "BOLD", 11), Text(72.15, 700, "BOLD", 11)])
    layer, _ = layer_of(pdfs.text_pdf([page]))
    assert [ln.text for ln in layer.lines] == ["BOLD"]


def test_reading_order_is_by_position_not_by_drawing_order():
    page = PdfPage(items=[Text(72, 600, "third", 11), Text(72, 700, "first", 11), Text(300, 700, "second", 11)])
    layer, _ = layer_of(pdfs.text_pdf([page]))
    assert [ln.text for ln in layer.lines] == ["first", "second", "third"]


# ---- what is trusted -----------------------------------------------------------------------------

def test_invisible_layer_is_ignored_under_auto_and_used_under_always():
    img = Image.new("RGB", (400, 560), "white")
    ImageDraw.Draw(img).text((40, 100), "SCANNED TEXT", fill="black")
    page = PdfPage(items=[Picture(0, 0, 595, 842, img), Text(72, 700, "SCANNED TEXT", 11, mode=3)])
    data = pdfs.text_pdf([page])
    layer, _ = layer_of(data)
    assert layer.kind == "ocr" and "invisible" in layer.notes[0]
    layer, _ = layer_of(data, mode="always")
    assert layer.kind == "text" and [ln.text for ln in layer.lines] == ["SCANNED TEXT"]
    assert layer.lines[0].confidence == CONF_INVISIBLE
    assert any("invisible" in n for n in layer.notes)


@pytest.mark.parametrize("font", ["pua", "nomap"])
def test_font_without_unicode_map_is_not_trusted(font):
    page = PdfPage(items=[Text(72, 700 - 15 * k, "The Vendor agrees to sell the land", 11, font=font) for k in range(6)])
    layer, _ = layer_of(pdfs.text_pdf([page]))
    assert layer.kind == "ocr" and "Unicode" in layer.notes[0]


def test_one_line_without_unicode_map_is_read_by_ocr_in_its_area():
    page = PdfPage(items=[Text(72, 700 - 15 * k, f"Clause {k} of the agreement reads well", 11) for k in range(8)]
                   + [Text(72, 500, "Garbled heading", 11, font="pua")])
    layer, _ = layer_of(pdfs.text_pdf([page]))
    assert layer.kind == "hybrid" and len(layer.regions) == 1
    assert len(layer.lines) == 8 and all("Garbled" not in ln.text for ln in layer.lines)
    l, t, r, b = layer.regions[0]
    x, y = px(72, 500)
    assert l <= x and t <= y <= b


def test_a_lone_symbol_character_is_dropped():
    # a bullet from a symbol font comes out as a private-use character, standing alone
    page = PdfPage(items=[Text(72, 700, "·", 11, font="pua"), Text(90, 700, "First clause", 11)])
    layer, _ = layer_of(pdfs.text_pdf([page]))
    assert layer.kind == "text" and [ln.text for ln in layer.lines] == ["First clause"]


def test_text_that_does_not_show_is_dropped():
    page = PdfPage(items=[
        Text(72, 700, "visible clause", 11),
        Text(72, 680, "white on white", 11, gray=1.0),
        Text(72, 660, "redacted badly", 11), Rule(70, 655, 200, 16, gray=0.0),  # a black box over it
    ])
    layer, _ = layer_of(pdfs.text_pdf([page]))
    assert [ln.text for ln in layer.lines] == ["visible clause"]
    assert any("do not show" in n for n in layer.notes)


def test_text_under_a_picture_is_dropped_and_the_picture_read_by_ocr():
    img = Image.new("RGB", (300, 100), (240, 240, 240))
    page = PdfPage(items=[Text(72, 700, "visible clause", 11), Text(72, 600, "hidden under", 11),
                          Picture(60, 560, 300, 80, img)])
    layer, _ = layer_of(pdfs.text_pdf([page]))
    assert [ln.text for ln in layer.lines] == ["visible clause"]
    assert layer.kind == "hybrid" and len(layer.regions) == 1


def test_small_picture_is_not_read_by_ocr():
    img = Image.new("RGB", (40, 40), (200, 0, 0))
    page = PdfPage(items=[Text(72, 700, "Letter body", 11), Picture(500, 780, 30, 30, img)])  # a logo
    layer, _ = layer_of(pdfs.text_pdf([page]))
    assert layer.kind == "text"


def test_rotated_page_with_upright_text_maps_correctly():
    # /Rotate 90 turns the MediaBox clockwise; text drawn at 90 degrees then shows upright
    page = PdfPage(width=842, height=595, rotate=90, items=[Text(842 - 700, 72, "Turned page text", 11, angle=90)])
    layer, p = layer_of(pdfs.text_pdf([page]))
    assert p.image.size[1] > p.image.size[0]  # shows portrait
    assert layer.kind == "text" and [ln.text for ln in layer.lines] == ["Turned page text"]
    x, y = px(72, 700)
    l, t, r, b = layer.lines[0].box
    assert abs(l - x) < 2 and t < y < b and r - l > 4 * (b - t)


def test_page_whose_text_shows_sideways_falls_back_to_ocr():
    page = PdfPage(rotate=90, items=[Text(72, 700 - 15 * k, "Sideways text on the page", 11) for k in range(5)])
    layer, _ = layer_of(pdfs.text_pdf([page]))
    assert layer.kind == "ocr" and "angle" in layer.notes[0]


def test_a_little_text_at_an_angle_is_left_to_ocr():
    page = PdfPage(items=[Text(72, 700 - 15 * k, f"Clause {k} of the agreement reads well", 11) for k in range(8)]
                   + [Text(560, 300, "MARGIN NOTE", 9, angle=90)])
    layer, _ = layer_of(pdfs.text_pdf([page]))
    assert layer.kind == "hybrid" and len(layer.lines) == 8 and len(layer.regions) == 1


def test_cropped_page_maps_through_the_crop_box():
    page = PdfPage(width=700, height=1000, crop=(50, 100, 645, 942), items=[Text(122, 800, "Cropped text", 11)])
    layer, p = layer_of(pdfs.text_pdf([page]))
    assert abs(p.image.size[1] - RENDER_LONG_SIDE) <= 1
    x, y = (122 - 50) * SCALE, (942 - 800) * SCALE
    l, t, r, b = layer.lines[0].box
    assert abs(l - x) < 2 and t < y < b


def test_text_outside_the_crop_box_is_ignored():
    page = PdfPage(width=700, height=1000, crop=(50, 100, 645, 942),
                   items=[Text(122, 800, "Inside", 11), Text(122, 960, "Outside", 11)])
    layer, _ = layer_of(pdfs.text_pdf([page]))
    assert [ln.text for ln in layer.lines] == ["Inside"]


def test_image_only_pdf_has_no_text_layer_and_no_note():
    layer, _ = layer_of(pdfs.image_pdf(1))
    assert layer.kind == "ocr" and layer.notes == []


# ---- through the pipeline -----------------------------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2, 5])
def test_digital_agreement_is_read_without_ocr(seed):
    doc = synth_pdf.make_pdf_document(seed)
    res = extract(synth_pdf.build(doc, "digital"), synth_pdf.doc_type_ref(),
                  extractor=LocalExtractor(engine=NoOcr()))
    assert res.pages == len(doc.pages) >= 2
    wrong = {n: (res.fields[n].value, v) for n, v in doc.truth.items() if res.fields[n].value != v}
    assert wrong == {}
    assert all("text layer" in w for w in res.warnings)


@pytest.mark.parametrize("form", ["rotated", "cropped"])
def test_rotated_and_cropped_digital_pages_are_read_without_ocr(form):
    doc = synth_pdf.make_pdf_document(1)
    res = extract(synth_pdf.build(doc, form), synth_pdf.doc_type_ref(), extractor=LocalExtractor(engine=NoOcr()))
    assert res.values() == doc.truth


def test_boxes_of_text_layer_values_are_on_the_rendered_page():
    doc = synth_pdf.make_pdf_document(0)
    data = synth_pdf.build(doc, "digital")
    res = extract(data, synth_pdf.doc_type_ref(), ["vendor_name"], extractor=LocalExtractor(engine=NoOcr()))
    f = res.fields["vendor_name"]
    with open_pages(data) as pages:
        image = list(pages)[f.page - 1]
    l, t, r, b = f.bbox
    crop = np.asarray(image.convert("L").crop((l, t, r, b)))
    assert crop.min() < 80  # ink inside the box
    assert 0 <= l < r <= image.width and 0 <= t < b <= image.height


def test_never_reads_every_page_by_ocr():
    seen = []

    class Recorder(Extractor):
        name = "recorder"
        supports_text_layer = True

        def extract(self, image, doc_type, fields, text_layer=None):
            seen.append(text_layer)
            return ExtractorOutput(fields=[FieldResult(name=f.name) for f in fields])

    data = synth_pdf.build(synth_pdf.make_pdf_document(0), "digital")
    res = extract(data, synth_pdf.doc_type_ref(), extractor=Recorder(), pdf_text="never")
    assert seen and all(t is None for t in seen) and res.warnings == []
    seen.clear()
    extract(data, synth_pdf.doc_type_ref(), extractor=Recorder())
    assert seen and all(t is not None and t.kind == "text" for t in seen)


def test_extractor_without_text_layer_support_gets_images_only():
    calls = []

    class Old(Extractor):
        name = "old"

        def extract(self, image, doc_type, fields):
            calls.append(image.size)
            return ExtractorOutput(fields=[FieldResult(name=f.name) for f in fields])

    data = synth_pdf.build(synth_pdf.make_pdf_document(0), "digital")
    res = extract(data, synth_pdf.doc_type_ref(), extractor=Old())
    assert len(calls) == res.pages and res.warnings == []


def test_pdf_text_setting_is_checked():
    with pytest.raises(ValueError, match="pdf_text"):
        extract(pdfs.png(), "logbook", ["make"], extractor=LocalExtractor(engine=NoOcr()), pdf_text="sometimes")


def test_invisible_layer_with_errors_wrong_under_always():
    # the reason `auto` does not trust a scanner's layer: its errors come back looking certain
    doc = synth_pdf.make_pdf_document(1)
    res = extract(synth_pdf.build(doc, "ocr_layer"), synth_pdf.doc_type_ref(), pdf_text="always",
                  extractor=LocalExtractor(engine=NoOcr()))
    wrong = [n for n, v in doc.truth.items() if res.fields[n].value != v]
    assert wrong  # the layer's errors reach the values
    assert all(res.fields[n].confidence is None or res.fields[n].confidence <= CONF_INVISIBLE for n in doc.truth)


def test_mixed_page_reads_the_picture_by_ocr_and_keeps_the_text_layer():
    # a picture pasted on a text page: the fake engine "reads" one line inside it
    img = Image.new("RGB", (600, 120), (250, 250, 250))
    page = PdfPage(items=[Text(72, 700, "Clause one of the agreement", 11), Text(72, 500, "Clause two", 11),
                          Picture(72, 560, 300, 60, img)])
    data = pdfs.text_pdf([page])
    eng = CropEngine([OcrLine("in the presence of JAMES OTIENO, Advocate", 0.9, (10, 20, 500, 50))])
    from dococr import DocType, FieldSpec

    doc = DocType(name="t", fields=[FieldSpec(name="witness", label="in the presence of", until=",")])
    res = extract(data, doc, extractor=LocalExtractor(engine=eng), include_raw_text=True)
    assert len(eng.shapes) == 1  # only the picture's area was read by OCR
    h, w = eng.shapes[0][:2]
    assert w / h > 3  # the picture's shape (300 x 60 points), not the page's
    assert res.fields["witness"].value == "JAMES OTIENO"
    assert "hybrid" not in res.warnings[0] and "by OCR in 1 area" in res.warnings[0]
    assert res.raw_text.count("Clause one") == 1


def test_digital_fixture_crossing_fields_are_recorded():
    doc = synth_pdf.make_pdf_document(1)
    assert len(doc.pages) == 3 and doc.crossing
    assert synth.FIXTURE_DIR.joinpath("synth_contract.yaml").is_file()


def test_ocr_engine_is_not_even_imported_for_a_digital_pdf(tmp_path):
    import subprocess

    p = tmp_path / "agreement.pdf"
    p.write_bytes(synth_pdf.build(synth_pdf.make_pdf_document(2), "digital"))
    code = (
        "import sys\n"
        "from dococr import extract\n"
        f"res = extract({str(p)!r}, {str(synth_pdf.doc_type_ref())!r})\n"
        "assert res.values()['vendor_name'], res\n"
        "assert 'rapidocr' not in sys.modules and 'onnxruntime' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)


def test_hyphens_at_line_ends_reach_the_matcher_as_printed():
    from dococr import DocType, FieldSpec

    page = PdfPage(items=[Text(72, 700, "Kenya Shillings Twenty-", 11), Text(72, 686, "Five Thousand only, and the", 11),
                          Text(72, 672, "land is regis­", 11), Text(72, 658, "tered in the name of JOHN.", 11)])
    data = pdfs.text_pdf([page])
    layer, _ = layer_of(data)
    assert [ln.text for ln in layer.lines][0] == "Kenya Shillings Twenty-"
    assert layer.lines[2].text == "land is regis-"  # a soft hyphen at the line end shows as "-"
    doc = DocType(name="t", fields=[FieldSpec(name="amount", label="Kenya Shillings", until="only", type="integer"),
                                    FieldSpec(name="state", label="land is", until="in the name")])
    res = extract(data, doc, extractor=LocalExtractor(engine=NoOcr()))
    assert res.fields["amount"].value == 25000
    assert res.fields["state"].value == "registered"


@pytest.mark.parametrize("rotate", [180, 270])
def test_other_page_rotations_with_upright_text_map_correctly(rotate):
    X, Y = 72, 700  # where the text starts on the page as it shows (portrait, y up)
    if rotate == 180:
        page = PdfPage(rotate=180, items=[Text(595 - X, 842 - Y, "Turned page text", 11, angle=180)])
    else:  # /Rotate 270 shows the landscape MediaBox turned counter-clockwise
        page = PdfPage(width=842, height=595, rotate=270, items=[Text(Y, 595 - X, "Turned page text", 11, angle=-90)])
    layer, p = layer_of(pdfs.text_pdf([page]))
    assert p.image.size[1] > p.image.size[0]
    assert layer.kind == "text" and [ln.text for ln in layer.lines] == ["Turned page text"]
    x, y = px(X, Y)
    l, t, r, b = layer.lines[0].box
    assert abs(l - x) < 2 and t < y < b and r - l > 4 * (b - t)


def test_stamp_and_form_field_annotations_are_read_by_ocr():
    from pdfs import Annotation

    page = PdfPage(items=[Text(72, 700, "Certified copy of the agreement", 11),
                          Annotation(300, 400, 200, 80, "Stamp"), Annotation(72, 300, 200, 20, "Widget"),
                          Annotation(72, 200, 100, 20, "Link")])  # a link holds no text
    layer, _ = layer_of(pdfs.text_pdf([page]))
    assert layer.kind == "hybrid" and len(layer.regions) == 2
