"""End to end through the real RapidOCR model on synthetic documents. Slow."""

from __future__ import annotations

import time

import pytest

from dococr import extract
from synth import DOC_TYPES, FIXTURE_TYPES, VARIANTS, doc_type_ref, make_document, variants_for

pytestmark = pytest.mark.slow


def test_driving_license_default_backend():
    doc = make_document("driving_license", "clean", 0)
    res = extract(doc.image, "driving_license", ["license_number", "date_of_expiry"])
    assert res.backend == "local"
    assert res.values() == {k: doc.truth[k] for k in ("license_number", "date_of_expiry")}
    for name in ("license_number", "date_of_expiry"):
        f = res.fields[name]
        assert f.confidence and f.confidence > 0.8
        l, t, r, b = f.bbox
        assert 0 <= l < r <= doc.image.width and 0 <= t < b <= doc.image.height


def test_bbox_points_at_value_on_rotated_page():
    upright = make_document("driving_license", "clean", 0)
    rotated = make_document("driving_license", "rot90", 0)
    a = extract(upright.image, "driving_license", ["surname"]).fields["surname"]
    b = extract(rotated.image, "driving_license", ["surname"]).fields["surname"]
    assert a.value == b.value == upright.truth["surname"]
    # PIL rotate(90, expand) maps (x, y) -> (y, W - x): the box must move accordingly
    W = upright.image.width
    l, t, r, bt = a.bbox
    exp = (t, W - r, bt, W - l)
    assert all(abs(p - q) <= 12 for p, q in zip(b.bbox, exp)), (b.bbox, exp)


@pytest.mark.parametrize("variant", list(VARIANTS))
def test_synthetic_accuracy_per_variant(variant):
    """Per-variant floor across the four doc types (seed 0). Full table: tests/local/eval_synth.py."""
    ok = total = 0
    misses = []
    for dtp in DOC_TYPES:
        doc = make_document(dtp, variant, 0)
        res = extract(doc.image, dtp, doc.fields)
        for name, expected in doc.truth.items():
            total += 1
            if res.values()[name] == expected:
                ok += 1
            else:
                misses.append((dtp, name, expected, res.values()[name]))
    assert ok / total >= 0.9, misses


def test_adhoc_field_end_to_end():
    from PIL import Image, ImageDraw
    from synth import font

    img = Image.new("RGB", (1000, 300), "white")
    d = ImageDraw.Draw(img)
    f = font("sans", 26)
    d.text((40, 40), "COUNTY OF RESIDENCE", font=f, fill="black")
    d.text((450, 40), "NAKURU", font=f, fill="black")
    d.text((40, 110), "DATE OF EXPIRY", font=f, fill="black")
    d.text((450, 110), "12-MAR-2031", font=f, fill="black")
    res = extract(img, "driving_license", ["county_of_residence", "date_of_expiry"])
    assert res.values() == {"county_of_residence": "NAKURU", "date_of_expiry": "2031-03-12"}


def test_warm_latency_is_reasonable():
    doc = make_document("national_id", "clean", 1)
    extract(doc.image, "national_id", ["id_number"])  # warm
    t = time.perf_counter()
    extract(doc.image, "national_id", ["id_number"])
    # generous: the CI box may be shared; this only guards against pathological regressions
    assert time.perf_counter() - t < 60


def test_runs_with_the_network_cut_off(monkeypatch):
    """Offline guarantee: a fresh engine loads its models and reads a page while every attempt
    to open a network connection or resolve a host name fails the test."""
    import socket

    from dococr.extractors.local import LocalExtractor
    from dococr.ocr.rapid import RapidOcrEngine

    def blocked(*args, **kwargs):
        raise AssertionError(f"network access attempted: {args[:2]}")

    monkeypatch.setattr(socket.socket, "connect", blocked)
    monkeypatch.setattr(socket.socket, "connect_ex", blocked)
    monkeypatch.setattr(socket, "getaddrinfo", blocked)
    monkeypatch.setattr(socket, "create_connection", blocked)

    doc = make_document("national_id", "clean", 0)
    res = extract(doc.image, "national_id", ["id_number"], extractor=LocalExtractor(RapidOcrEngine()))
    assert res.values() == {"id_number": doc.truth["id_number"]}


@pytest.mark.parametrize("doc_type", list(DOC_TYPES))
def test_single_pass_equals_two_pass_on_an_untransformed_page(doc_type):
    """On a page that needs no rotation, deskew, scaling or contrast step, recognising the
    quads of the preprocessing detector pass gives exactly what detecting again would."""
    from dococr.ocr import get_default_engine
    from dococr.preprocess import prepare

    engine = get_default_engine()
    doc = make_document(doc_type, "clean", 0)
    work, transform = prepare(doc.image, engine)
    assert transform.line_quads is not None, "expected the single pass on a clean page"
    assert work.shape[:2] == (doc.image.height, doc.image.width)
    single = engine.recognize(work, transform.line_quads)
    double = engine.read(work)
    assert len(single) > 5
    assert [(ln.text, ln.confidence, ln.box) for ln in single] == \
        [(ln.text, ln.confidence, ln.box) for ln in double]


def test_batched_upside_down_check_equals_one_call_per_crop():
    import numpy as np

    from dococr.ocr import get_default_engine

    engine = get_default_engine()
    doc = make_document("logbook", "rot180", 0)
    image = np.asarray(doc.image.convert("RGB"))
    crops = []
    for q in engine.detect(image)[:7]:
        q = np.asarray(q)
        crops.append(image[int(q[:, 1].min()):int(q[:, 1].max()), int(q[:, 0].min()):int(q[:, 0].max())])
    flipped = total = 0.0
    for crop in crops:  # the previous implementation: RapidOCR's public call, one crop at a time
        label, score = engine._call(crop, use_det=False, use_cls=True, use_rec=False).cls_res[0]
        total += float(score)
        flipped += float(score) if str(label) == "180" else 0.0
    assert engine.classify_upside_down(crops) == pytest.approx(flipped / total, abs=1e-6)
    assert flipped / total > 0.6


def test_warm_up_loads_all_models():
    from dococr.extractors.local import LocalExtractor
    from dococr.ocr.rapid import RapidOcrEngine

    engine = RapidOcrEngine()
    LocalExtractor(engine).warm_up()
    ocr = engine._ocr
    assert ocr.text_det is not None and ocr.text_cls is not None and ocr.text_rec is not None


# ---------------------------------------------------------------------------- multi-line and prose fixtures

# synth_ruled: blur_lowres loses spaces inside values ("K.M.MUTUA"), 19 of 22 on seeds 0 and 1.
FIXTURE_FLOOR = {"synth_form": 0.9, "synth_agreement": 0.8, "synth_contract": 0.8, "synth_glued": 0.8,
                 "synth_ruled": 0.85}


@pytest.mark.parametrize("doc_type,variant", [(t, v) for t in FIXTURE_TYPES for v in variants_for(t)])
def test_fixture_accuracy_per_variant(doc_type, variant):
    """Floors for the test-fixture types with `multiline` and `until` fields (seeds 0 and 1).
    Full table: tests/local/eval_synth.py --types synth_form,synth_agreement."""
    ok = total = 0
    misses = []
    for seed in (0, 1):
        doc = make_document(doc_type, variant, seed)
        res = extract(doc.image, doc_type_ref(doc_type), doc.fields)
        for name, expected in doc.truth.items():
            total += 1
            if res.values()[name] == expected:
                ok += 1
            else:
                misses.append((seed, name, expected, res.values()[name], res.fields[name].warnings))
    assert ok / total >= FIXTURE_FLOOR[doc_type], misses


def test_prose_bbox_follows_a_turned_page():
    """A multi-line prose value's box is in the caller's image coordinates on a turned page."""
    doc = make_document("synth_agreement", "clean", 0)
    ref = doc_type_ref("synth_agreement")
    a = extract(doc.image, ref, ["vendor_name", "price_in_words"])
    b = extract(doc.image.rotate(90, expand=True), ref, ["vendor_name", "price_in_words"])
    W = doc.image.width
    for name in ("vendor_name", "price_in_words"):
        fa, fb = a.fields[name], b.fields[name]
        assert fa.value == fb.value == doc.truth[name]
        l, t, r, bt = fa.bbox
        assert all(abs(p - q) <= 12 for p, q in zip(fb.bbox, (t, W - r, bt, W - l))), (name, fa.bbox, fb.bbox)
