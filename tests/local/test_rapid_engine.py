"""RapidOcrEngine behaviour that does not need the real model."""

import threading
import time
from types import SimpleNamespace

import numpy as np
import pytest

from dococr.ocr.rapid import RapidOcrEngine


class RecordingOcr:
    """Stands in for rapidocr.RapidOCR and records how many callers are inside it at once."""

    def __init__(self):
        self.inside = 0
        self.most_inside = 0
        self._guard = threading.Lock()

    def __call__(self, image, **flags):
        with self._guard:
            self.inside += 1
            self.most_inside = max(self.most_inside, self.inside)
        time.sleep(0.01)
        with self._guard:
            self.inside -= 1
        return SimpleNamespace(boxes=None, txts=None, scores=None, cls_res=None)


def test_calls_into_rapidocr_are_serialised():
    """RapidOCR keeps per-call flags on the shared instance, so the HTTP service's worker
    threads must never be inside it together."""
    engine = RapidOcrEngine()
    engine._ocr = RecordingOcr()
    image = np.zeros((20, 20, 3), dtype=np.uint8)

    def work():
        engine.read(image)
        engine.detect(image)

    threads = [threading.Thread(target=work) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert engine._ocr.most_inside == 1


def test_missing_model_file_is_an_error_not_a_download(monkeypatch, tmp_path):
    import rapidocr

    from dococr import ExtractorError
    from dococr.ocr import rapid

    monkeypatch.setattr(rapidocr, "__file__", str(tmp_path / "__init__.py"))
    with pytest.raises(ExtractorError, match="never downloads"):
        rapid.RapidOcrEngine()


class FakeComponents(RecordingOcr):
    """RecordingOcr plus the parts of RapidOCR 3.10 that `recognize`, `classify_upside_down` and
    `warm_up` use: the lazily loaded models, `text_score` and the size settings. Every model call
    counts as being inside RapidOCR."""

    def __init__(self, rec_txts=None, cls_labels=None):
        super().__init__()
        self.text_score = 0.5
        self.min_side_len, self.max_side_len = 30, 2000
        self.min_height, self.width_height_ratio = 30, 8
        self.cfg = SimpleNamespace(Global=SimpleNamespace(use_preprocess_img=True,
                                                          use_vertical_padding=True))
        self.rec_txts = rec_txts
        self.cls_labels = cls_labels
        self.loaded: list[str] = []
        self.rec_inputs: list = []
        self.cls_inputs: list = []

    def _enter(self):
        with self._guard:
            self.inside += 1
            self.most_inside = max(self.most_inside, self.inside)
        time.sleep(0.005)
        with self._guard:
            self.inside -= 1

    def _load_det_model(self):
        self.loaded.append("det")
        return lambda img: self._enter()

    def _load_cls_model(self):
        self.loaded.append("cls")

        def cls(imgs):
            self._enter()
            self.cls_inputs.append(list(imgs))
            labels = self.cls_labels or [("0", 0.99)] * len(imgs)
            return SimpleNamespace(cls_res=labels[: len(imgs)])

        return cls

    def _load_rec_model(self):
        self.loaded.append("rec")

        def rec(inp):
            self._enter()
            self.rec_inputs.append(inp.img)
            txts = self.rec_txts or [(f"line{i}", 0.9) for i in range(len(inp.img))]
            txts = txts[: len(inp.img)]
            return SimpleNamespace(txts=tuple(t for t, _ in txts), scores=tuple(s for _, s in txts))

        return rec


def quad(l, t, r, b):
    return [[l, t], [r, t], [r, b], [l, b]]


def page(w=400, h=200):
    return np.full((h, w, 3), 255, dtype=np.uint8)


def test_every_entry_point_is_serialised():
    engine = RapidOcrEngine()
    engine._ocr = FakeComponents()
    image = page()
    crops = [np.zeros((20, 80, 3), np.uint8)] * 3

    def work():
        engine.read(image)
        engine.detect(image)
        engine.recognize(image, [quad(10, 10, 200, 40)])
        engine.classify_upside_down(crops)
        engine.warm_up()

    threads = [threading.Thread(target=work) for _ in range(6)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()
    assert engine._ocr.most_inside == 1


def test_upside_down_check_is_one_batched_classifier_call():
    engine = RapidOcrEngine()
    fake = FakeComponents(cls_labels=[("180", 0.9), ("0", 0.6), ("180", 1.0)])
    engine._ocr = fake
    crops = [np.zeros((20, 80, 3), np.uint8), np.zeros((0, 0, 3), np.uint8),
             np.zeros((24, 100, 3), np.uint8), np.zeros((40, 120, 3), np.uint8)]
    frac = engine.classify_upside_down(crops)
    assert len(fake.cls_inputs) == 1 and len(fake.cls_inputs[0]) == 3  # empty crop skipped
    # crops under RapidOCR's min_side_len (30) get the same resize its own call applies
    assert min(fake.cls_inputs[0][0].shape[:2]) >= 30
    assert frac == pytest.approx((0.9 + 1.0) / (0.9 + 0.6 + 1.0))
    assert engine.classify_upside_down([]) is None


def test_recognize_orders_crops_filters_and_reports_boxes():
    engine = RapidOcrEngine()
    fake = FakeComponents(rec_txts=[("first", 0.95), (" ", 0.99), ("weak", 0.3), ("last", 0.8)])
    engine._ocr = fake
    image = page(600, 300)
    quads = [quad(300, 200, 500, 230),  # bottom row
             [[20, 20], [200, 20], [200, 50], [20, 50]][::-1],  # top-left, corners reversed
             quad(250, 22, 400, 52),  # top row, right of the first
             quad(20, 120, 220, 150),  # middle row
             quad(10, 120, 12, 150)]  # too thin: dropped like DBPostProcess does
    lines = engine.recognize(image, quads)
    assert len(fake.rec_inputs) == 1 and len(fake.rec_inputs[0]) == 4
    assert [c.shape[:2] for c in fake.rec_inputs[0]] == [(30, 180), (30, 150), (30, 200), (30, 200)]
    # reading order top-left first; empty text and score under text_score dropped
    assert [(ln.text, ln.box) for ln in lines] == [("first", (20.0, 20.0, 200.0, 50.0)),
                                                    ("last", (300.0, 200.0, 500.0, 230.0))]


def test_recognize_turns_tall_crops_like_rapidocr():
    engine = RapidOcrEngine()
    fake = FakeComponents()
    engine._ocr = fake
    engine.recognize(page(), [quad(50, 20, 70, 120)])
    assert fake.rec_inputs[0][0].shape[:2] == (20, 100)


@pytest.mark.parametrize("shape", [(100, 2100), (25, 600), (60, 600)])
def test_recognize_falls_back_to_a_full_read_when_rapidocr_would_alter_the_image(shape):
    """Over max_side_len, under min_side_len, or a strip wider than 8:1: RapidOCR resizes or pads
    the image before cropping, so the crops would differ; a full read is used instead."""
    engine = RapidOcrEngine()
    fake = FakeComponents()
    engine._ocr = fake
    calls = []
    engine.read = lambda image: calls.append(image.shape) or []
    engine.recognize(page(shape[1], shape[0]), [quad(5, 5, 50, 20)])
    assert calls == [(shape[0], shape[1], 3)] and fake.rec_inputs == []


def test_recognize_with_no_quads_does_nothing():
    engine = RapidOcrEngine()
    engine._ocr = FakeComponents()
    assert engine.recognize(page(), []) == []
    assert engine._ocr.rec_inputs == []


def test_warm_up_loads_and_runs_all_three_models():
    engine = RapidOcrEngine()
    fake = FakeComponents()
    engine._ocr = fake
    engine.warm_up()
    assert sorted(fake.loaded) == ["cls", "det", "rec"]
    assert len(fake.cls_inputs) == 1 and len(fake.rec_inputs) == 1
