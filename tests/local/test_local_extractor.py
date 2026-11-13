"""LocalExtractor wiring and preprocessing with a fake OCR engine. Fast: no model."""

from __future__ import annotations

import pytest
from PIL import Image

from dococr import ExtractorError, extract, load_doc_type
from dococr.extractors.local import LocalExtractor
from dococr.ocr import OcrEngine, OcrLine
from dococr.preprocess import prepare


class FakeEngine(OcrEngine):
    """Returns fixed lines (in working-image coordinates) and optional detector quads."""

    def __init__(self, lines, quads=None, upside_down=0.0):
        self.lines, self.quads, self.upside_down = lines, quads or [], upside_down
        self.read_shapes = []

    def read(self, image):
        self.read_shapes.append(image.shape)
        return list(self.lines)

    def detect(self, image):
        return self.quads

    def classify_upside_down(self, crops):
        return self.upside_down


def box_quad(l, t, r, b):
    return [[l, t], [r, t], [r, b], [l, b]]


def test_extractor_contract_and_bbox():
    img = Image.new("RGB", (1200, 800), "white")
    lines = [OcrLine("SURNAME", 0.99, (40, 100, 160, 130)), OcrLine("DOE", 0.97, (300, 100, 350, 130))]
    quads = [box_quad(40, 100, 160, 130), box_quad(300, 100, 350, 130)]
    ex = LocalExtractor(engine=FakeEngine(lines, quads))
    assert ex.name == "local"
    res = extract(img, "driving_license", ["surname", "date_of_expiry", "made_up"], extractor=ex,
                  include_raw_text=True)
    assert list(res.fields) == ["surname", "date_of_expiry", "made_up"]
    assert res.values()["surname"] == "DOE"
    assert res.fields["surname"].bbox == (300, 100, 350, 130)
    assert res.missing == ["date_of_expiry", "made_up"]
    assert "SURNAME" in res.raw_text


def test_no_text_found():
    img = Image.new("RGB", (1200, 800), "white")
    res = extract(img, "national_id", ["id_number"], extractor=LocalExtractor(engine=FakeEngine([])))
    assert res.missing == ["id_number"]
    assert res.warnings


def test_image_not_mutated():
    img = Image.new("RGB", (1200, 800), "white")
    before = img.tobytes()
    LocalExtractor(engine=FakeEngine([])).extract(img, load_doc_type("passport"), [])
    assert img.tobytes() == before


def test_small_image_is_upscaled_and_boxes_map_back():
    img = Image.new("RGB", (400, 250), "white")
    quads = [box_quad(20, 20, 120, 30), box_quad(150, 20, 250, 30)]  # 10 px tall text
    work, tr = prepare(img, FakeEngine([], quads))
    assert tr.scale > 2 and work.shape[1] == round(400 * tr.scale)
    assert tr.box_to_original((20 * tr.scale, 20 * tr.scale, 120 * tr.scale, 30 * tr.scale)) == (20, 20, 120, 30)


def test_huge_image_is_capped():
    img = Image.new("RGB", (5000, 3000), "white")
    work, tr = prepare(img, FakeEngine([], [box_quad(100, 100, 1000, 160)]))
    assert max(work.shape[:2]) <= 2000


def test_quarter_turn_detected_from_vertical_boxes():
    # Page rotated: text lines are tall, thin boxes.
    img = Image.new("RGB", (600, 1000), "white")
    quads = [box_quad(100, 100, 130, 500), box_quad(200, 100, 230, 600), box_quad(300, 100, 330, 400)]
    work, tr = prepare(img, FakeEngine([], quads))
    assert tr.rotation in (90, 270)
    assert work.shape[1] > work.shape[0]  # now landscape
    # a box in the working image maps back inside the original image
    l, t, r, b = tr.box_to_original((10, 10, 200, 40))
    assert 0 <= l < r <= 600 and 0 <= t < b <= 1000 and (b - t) > (r - l)


def test_upside_down_detected_by_classifier():
    img = Image.new("RGB", (1200, 800), "white")
    quads = [box_quad(100, 100, 600, 130), box_quad(100, 200, 500, 230)]
    work, tr = prepare(img, FakeEngine([], quads, upside_down=0.9))
    assert tr.rotation == 180
    # top-left corner of the working image is the bottom-right of the original
    assert tr.box_to_original((0, 0, 10, 10)) == (1190, 790, 1200, 800)


def test_skew_detected():
    import math

    img = Image.new("RGB", (1200, 800), "white")
    a = math.radians(4)
    quads = []
    for y in (100, 200, 300):
        x0, x1 = 100, 600
        quads.append([[x0, y], [x1, y + (x1 - x0) * math.tan(a)],
                      [x1, y + 30 + (x1 - x0) * math.tan(a)], [x0, y + 30]])
    work, tr = prepare(img, FakeEngine([], quads))
    assert abs(tr.skew - 4) < 0.5


def test_missing_rapidocr_raises_extractor_error(monkeypatch):
    import builtins

    import dococr.ocr as ocr_pkg
    from dococr.ocr import rapid

    real_import = builtins.__import__

    def fake_import(name, *args, **kwargs):
        if name == "rapidocr" or name.startswith("rapidocr."):
            raise ImportError("no rapidocr")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", fake_import)
    monkeypatch.setattr(ocr_pkg, "_default", None)
    with pytest.raises(ExtractorError):
        rapid.RapidOcrEngine()


def test_wide_lines_are_cut_down_before_the_orientation_classifier():
    """A full-width line of running text is far wider than the classifier's input shape, and
    squeezed into it the letters are unreadable, so the upside-down verdict was close to a coin
    flip on prose pages. The classifier must only ever see a legible slice of a line."""
    from dococr import preprocess

    seen = []

    class Recording(FakeEngine):
        def classify_upside_down(self, crops):
            seen.extend(c.shape for c in crops)
            return 0.0

    img = Image.new("RGB", (1200, 800), "white")
    prose_line = box_quad(100, 100, 1100, 130)  # 1000 x 30: 33 times wider than tall
    short_line = box_quad(100, 200, 220, 230)  # 120 x 30: already narrow enough
    prepare(img, Recording([], [prose_line, short_line]))

    assert len(seen) == 2
    for height, width, _ in seen:
        assert width <= preprocess.CLASSIFIER_ASPECT * height + 2
    assert sorted(width for _, width, _ in seen)[0] >= 110  # the short line is not cut
