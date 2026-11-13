"""Single-pass OCR: the quads found by `prepare`'s detector pass are recognised directly instead
of detecting a second time on the working image. Fast: fake engines, no model."""

from __future__ import annotations

import numpy as np
import pytest
from PIL import Image, ImageDraw

from dococr import extract
from dococr import preprocess
from dococr.extractors.local import LocalExtractor
from dococr.ocr import OcrEngine, OcrLine
from dococr.preprocess import prepare


class SpyEngine(OcrEngine):
    """Records which OCR entry points the extractor uses."""

    def __init__(self, quads, lines=(), upside_down=0.0):
        self.quads, self.lines, self.upside_down = quads, list(lines), upside_down
        self.calls: list[str] = []
        self.recognized_quads = None

    def read(self, image):
        self.calls.append("read")
        return list(self.lines)

    def detect(self, image):
        self.calls.append("detect")
        return self.quads

    def recognize(self, image, quads):
        self.calls.append("recognize")
        self.recognized_quads = [np.asarray(q, dtype=float) for q in quads]
        return list(self.lines)

    def classify_upside_down(self, crops):
        self.calls.append("classify")
        return self.upside_down


def box_quad(l, t, r, b):
    return [[l, t], [r, t], [r, b], [l, b]]


QUADS = [box_quad(40, 100, 300, 130), box_quad(400, 100, 600, 130), box_quad(40, 200, 400, 230)]


def inked_page(size=(1000, 600), quads=QUADS) -> Image.Image:
    """A white page with a dark bar at each quad: enough contrast that no CLAHE step runs."""
    img = Image.new("RGB", size, "white")
    d = ImageDraw.Draw(img)
    for q in quads:
        xs, ys = [p[0] for p in q], [p[1] for p in q]
        d.rectangle((min(xs), min(ys), max(xs), max(ys)), fill="black")
    return img


def test_untransformed_page_is_detected_once_and_quads_are_passed_unchanged():
    img = inked_page()
    lines = [OcrLine("SURNAME", 0.99, (40, 100, 300, 130)), OcrLine("DOE", 0.97, (400, 100, 600, 130))]
    eng = SpyEngine(QUADS, lines)
    res = extract(img, "driving_license", ["surname"], extractor=LocalExtractor(eng))
    assert eng.calls.count("detect") == 1
    assert "read" not in eng.calls and eng.calls.count("recognize") == 1
    assert [q.tolist() for q in eng.recognized_quads] == [np.asarray(q, float).tolist() for q in QUADS]
    assert res.values()["surname"] == "DOE"
    assert res.fields["surname"].bbox == (400, 100, 600, 130)


def test_no_detected_text_falls_back_to_a_full_read():
    img = Image.new("RGB", (1000, 600), "white")
    eng = SpyEngine([])
    LocalExtractor(eng).extract(img, preprocess_doc_type(), [])
    assert "recognize" not in eng.calls and eng.calls.count("read") == 1


def preprocess_doc_type():
    from dococr import load_doc_type

    return load_doc_type("passport")


def test_default_engine_recognize_is_a_full_read():
    class OnlyRead(OcrEngine):
        def read(self, image):
            return [OcrLine("X", 1.0, (0, 0, 1, 1))]

    assert OnlyRead().recognize(np.zeros((4, 4, 3), np.uint8), [box_quad(0, 0, 2, 2)])[0].text == "X"


# --- box mapping and crop orientation on transformed pages -----------------------------------


def _page_with_marked_bar() -> tuple[Image.Image, tuple[int, int, int, int]]:
    """An upright page with one dark 'text line' whose left end carries a white notch, so the
    orientation of a crop of it can be told."""
    img = Image.new("RGB", (900, 500), "white")
    d = ImageDraw.Draw(img)
    bar = (100, 200, 500, 240)
    d.rectangle(bar, fill="black")
    d.rectangle((110, 210, 140, 230), fill="white")  # notch near the left end
    d.rectangle((300, 140, 700, 170), fill=(60, 60, 60))  # a second line so skew logic has two
    return img, bar


def _crop_working(work: np.ndarray, quad) -> np.ndarray:
    from rapidocr.utils.process_img import get_rotate_crop_image

    from dococr.ocr.rapid import _order_and_clip

    (box,) = _order_and_clip([quad], work.shape[1], work.shape[0])
    return get_rotate_crop_image(work, box.copy())


def _notch_on_the_left(crop: np.ndarray) -> bool:
    gray = crop.mean(axis=2)
    h, w = gray.shape
    assert w > 3 * h, crop.shape  # a horizontal line, not a tall one
    mid = gray[h // 3: 2 * h // 3]
    return mid[:, : w // 4].mean() > mid[:, 3 * w // 4:].mean() + 20


@pytest.mark.parametrize("turn,upside_down", [(90, 0.9), (270, 0.0), (180, 0.9)])
def test_quads_from_a_turned_page_map_to_upright_crops(monkeypatch, turn, upside_down):
    """A quad detected on a sideways or upside-down page must give an upright crop of the same
    line in the working image, with its corners in the working frame."""
    monkeypatch.setattr(preprocess, "_reuse_detection", lambda *a: True)
    upright, bar = _page_with_marked_bar()
    page = upright.rotate(turn, expand=True)
    w0, h0 = upright.size

    def turn_point(x, y):  # PIL rotate(turn, expand=True) on an upright point
        return {90: (y, w0 - x), 180: (w0 - x, h0 - y), 270: (h0 - y, x)}[turn]

    def turned_quad(l, t, r, b):
        pts = [turn_point(x, y) for x, y in ((l, t), (r, t), (r, b), (l, b))]
        xs, ys = [p[0] for p in pts], [p[1] for p in pts]
        return box_quad(min(xs), min(ys), max(xs), max(ys))  # detector-style corner order

    quads = [turned_quad(*bar), turned_quad(300, 140, 700, 170)]
    work, tr = prepare(page, SpyEngine(quads, upside_down=upside_down))
    assert tr.line_quads is not None
    assert tr.rotation == (360 - turn) % 360
    quad = tr.line_quads[0]
    # the bar's quad lands where the bar is in the working image (up to a pixel of turn offset)
    l, t = quad[:, 0].min(), quad[:, 1].min()
    r, b = quad[:, 0].max(), quad[:, 1].max()
    assert abs(l - bar[0]) <= 2 and abs(t - bar[1]) <= 2 and abs(r - bar[2]) <= 2 and abs(b - bar[3]) <= 2
    crop = _crop_working(work, quad)
    assert _notch_on_the_left(crop)
    # and it maps back to the bar's place in the image the extractor was given
    expected = turned_quad(*bar)
    got = tr.box_to_original((l, t, r, b))
    assert all(abs(g - e) <= 2 for g, e in zip(got, (expected[0][0], expected[0][1], expected[2][0], expected[2][1])))


def test_quads_follow_deskew_and_upscale(monkeypatch):
    import math

    monkeypatch.setattr(preprocess, "_reuse_detection", lambda *a: True)
    img = Image.new("RGB", (600, 400), "white")
    a = math.radians(4)
    quads = []
    for y in (60, 120, 180):  # 12 px tall, tilted 4 degrees: upscaled and deskewed
        x0, x1 = 50, 450
        dy = (x1 - x0) * math.tan(a)
        quads.append([[x0, y], [x1, y + dy], [x1, y + 12 + dy], [x0, y + 12]])
    work, tr = prepare(img, SpyEngine(quads))
    assert tr.scale > 2 and abs(tr.skew - 4) < 0.5
    for q_orig, q_work in zip(quads, tr.line_quads):
        q_work = np.asarray(q_work)
        # levelled: top edge horizontal; scaled: height grows by the scale factor
        top = q_work[np.argsort(q_work[:, 1])[:2]]
        assert abs(top[0, 1] - top[1, 1]) < 1.5
        assert abs((q_work[:, 1].max() - q_work[:, 1].min()) - 12 * tr.scale) < 0.2 * 12 * tr.scale
        # every working-frame corner maps back onto the original corner
        inv = np.linalg.inv(tr.matrix)
        back = (inv @ np.c_[q_work, np.ones(4)].T).T[:, :2]
        assert np.allclose(back, np.asarray(q_orig, float), atol=1e-6)


# --- warm_up ----------------------------------------------------------------------------------


def test_warm_up_defaults_to_nothing_and_extractor_delegates():
    class Counting(SpyEngine):
        warmed = 0

        def warm_up(self):
            self.warmed += 1

    assert OcrEngine.warm_up(SpyEngine([])) is None
    eng = Counting([])
    LocalExtractor(eng).warm_up()
    assert eng.warmed == 1


# --- when the single pass is used ---------------------------------------------------------------


class FlooredEngine(SpyEngine):
    """A detector that, like RapidOCR's default, enlarges inputs to a 736 px short side."""

    def detector_resolution(self, width, height):
        return max(1.0, 736 / min(width, height))


SMALL_TEXT = [box_quad(20, 20, 220, 30), box_quad(250, 20, 380, 30), box_quad(20, 60, 300, 70)]


def test_upscaled_page_reuses_quads_when_the_detector_saw_it_as_finely():
    """A small page is enlarged by the detector to a 736 px short side anyway, so detecting
    again on the upscaled page would see the text no finer: the quads are reused, scaled."""
    img = inked_page((400, 250), SMALL_TEXT)
    eng = FlooredEngine(SMALL_TEXT)
    work, tr = prepare(img, eng)
    assert tr.scale > 2
    assert preprocess.detector_gain(eng, (400, 250), (400, 250), (work.shape[1], work.shape[0]),
                                    tr.scale) == pytest.approx(1.0, abs=0.02)
    assert tr.line_quads is not None
    assert np.allclose(tr.line_quads[0], np.asarray(SMALL_TEXT[0], float) * tr.scale)


def test_upscaled_page_detects_again_when_the_second_pass_would_see_finer_text():
    """A detector that runs at the input's own resolution would see the upscaled text larger:
    the first pass's quads are not trusted and the working image is read in full."""
    img = inked_page((400, 250), SMALL_TEXT)
    eng = SpyEngine(SMALL_TEXT)
    work, tr = prepare(img, eng)
    assert tr.scale > 2 and tr.line_quads is None
    LocalExtractor(eng).extract(img, preprocess_doc_type(), [])
    assert eng.calls[-1] == "read" and "recognize" not in eng.calls


def test_small_text_on_a_large_page_detects_again():
    """Short side over the detector's floor, small text: upscaling makes the second pass see
    the text finer, so it runs."""
    quads = [box_quad(100, 100, 300, 110), box_quad(100, 140, 350, 150)]
    img = inked_page((1000, 800), quads)
    eng = FlooredEngine(quads)
    work, tr = prepare(img, eng)
    assert tr.scale == pytest.approx(2.0) and tr.line_quads is None


def test_large_page_is_reduced_once_and_quads_are_reused_exactly():
    quads = [box_quad(400, 400, 2400, 520), box_quad(400, 800, 2000, 920)]
    img = inked_page((4000, 2500), quads)
    eng = FlooredEngine([np.asarray(q, float) / 2 for q in quads])  # detected on the half-size copy
    work, tr = prepare(img, eng)
    assert work.shape[:2] == (1250, 2000) and tr.scale == pytest.approx(0.5)
    assert tr.line_quads is not None
    assert np.allclose(tr.line_quads[0], np.asarray(quads[0], float) / 2)
    assert tr.box_to_original((200, 200, 1200, 260)) == (400, 400, 2400, 520)


def test_rapid_detector_resolution_follows_its_settings():
    from types import SimpleNamespace

    from dococr.ocr.rapid import RapidOcrEngine

    engine = RapidOcrEngine()
    engine._ocr = SimpleNamespace(
        max_side_len=2000,
        cfg=SimpleNamespace(Global=SimpleNamespace(use_preprocess_img=True),
                            Det=SimpleNamespace(limit_side_len=736, limit_type="min")))
    assert engine.detector_resolution(1000, 368) == pytest.approx(2.0)
    assert engine.detector_resolution(1000, 800) == 1.0
    assert engine.detector_resolution(4000, 1000) == pytest.approx(0.5 * 736 / 500)
    engine._ocr.cfg.Det.limit_type = "max"
    assert engine.detector_resolution(1472, 500) == pytest.approx(0.5)
