"""Regression tests for the attacks found in the security review. None needs the OCR model."""

import io
import struct
import zlib

import numpy as np
import pytest
from fastapi.testclient import TestClient
from PIL import Image

import pdfs
from dococr import Extractor, ExtractorError, ExtractorOutput, FieldResult, SourceError, extract
from dococr.imaging import IMAGE_FORMATS, MAX_MEGAPIXELS_ENV, open_pages
from dococr.models import FieldSpec
from dococr.normalize import MAX_TYPED_LENGTH, normalize_value


class Fake(Extractor):
    name = "fake"

    def extract(self, image, doc_type, fields):
        return ExtractorOutput(fields=[FieldResult(name=f.name) for f in fields])


def blank_png(width: int, height: int) -> bytes:
    """A 1-bit PNG of any size in a few kilobytes: the shape of a decompression bomb."""
    raw = (b"\x00" + b"\x00" * ((width + 7) // 8)) * height

    def chunk(kind: bytes, data: bytes) -> bytes:
        return struct.pack(">I", len(data)) + kind + data + struct.pack(">I", zlib.crc32(kind + data))

    header = struct.pack(">IIBBBBB", width, height, 1, 0, 0, 0, 0)
    return b"\x89PNG\r\n\x1a\n" + chunk(b"IHDR", header) + chunk(b"IDAT", zlib.compress(raw, 9)) + chunk(b"IEND", b"")


# ---- images that are tiny on the wire and enormous in memory ----------------------------------

def test_a_huge_blank_image_is_refused_before_it_is_decoded(monkeypatch):
    bomb = blank_png(13300, 13300)  # 21 KB; 176 megapixels
    assert len(bomb) < 40_000
    decoded = []
    monkeypatch.setattr(Image.Image, "convert", lambda self, *a, **k: decoded.append(1))
    with pytest.raises(SourceError, match="megapixels"):
        extract(bomb, "x", ["a"], extractor=Fake())
    assert decoded == []


def test_the_megapixel_limit_can_be_changed(monkeypatch):
    image = blank_png(3000, 3000)  # 9 megapixels
    extract(image, "x", ["a"], extractor=Fake())
    monkeypatch.setenv(MAX_MEGAPIXELS_ENV, "5")
    with pytest.raises(SourceError, match="megapixels"):
        extract(image, "x", ["a"], extractor=Fake())


@pytest.mark.parametrize("size", [(3, 3000), (3000, 3), (20, 20000), (8, 60000), (1, 1), (10, 10)])
def test_a_strip_or_speck_is_refused(size):
    with pytest.raises(SourceError, match="too small or too narrow"):
        extract(blank_png(*size), "x", ["a"], extractor=Fake())


@pytest.mark.parametrize("size", [(1410, 393), (400, 3000), (64, 64)])
def test_ordinary_shapes_are_accepted(size):
    assert extract(blank_png(*size), "x", ["a"], extractor=Fake()).pages == 1


def test_a_thin_pdf_page_is_refused():
    with pytest.raises(SourceError, match="too narrow"):
        with open_pages(pdfs.raw_pdf([(1, 100000)])) as pages:
            list(pages)


def test_the_detector_is_never_given_an_image_it_would_blow_up():
    """The detector enlarges the short side to a fixed size. Whatever reaches the engine (a
    crop of a PDF page, a custom caller), a strip must come back empty, not allocate."""
    from dococr.ocr.rapid import RapidOcrEngine

    called = []

    class Recording:
        cfg = None

        def __call__(self, image, **flags):
            called.append(image.shape)

    engine = RapidOcrEngine()
    engine._ocr = Recording()
    strip = np.zeros((3000, 3, 3), dtype=np.uint8)
    assert engine.read(strip) == [] and engine.detect(strip) == []
    assert called == []


def test_only_document_image_formats_are_decoded():
    assert set(IMAGE_FORMATS) <= {"PNG", "JPEG", "MPO", "TIFF", "WEBP", "GIF", "BMP"}
    assert "PNG" in IMAGE_FORMATS and "JPEG" in IMAGE_FORMATS
    buffer = io.BytesIO()
    Image.new("RGB", (64, 64), "white").save(buffer, "PPM")  # a format outside the list
    with pytest.raises(OSError):
        extract(buffer.getvalue(), "x", ["a"], extractor=Fake())


# ---- text that costs too much to read ---------------------------------------------------------

def test_a_page_with_an_absurd_amount_of_text_is_not_read_from_its_text_layer():
    from dococr.pdftext import MAX_PAGE_CHARS, PdfChar, PdfText, build_layer

    chars = [PdfChar(ch="a", box=(i % 900, (i // 900) * 2, i % 900 + 1, (i // 900) * 2 + 1.5))
             for i in range(MAX_PAGE_CHARS + 1)]
    layer = build_layer(PdfText(size=(1000, 1400), chars=chars), None, "auto")
    assert layer.kind == "ocr" and layer.lines == []
    assert any("over the limit" in note for note in layer.notes)


@pytest.mark.parametrize("kind", ["date", "integer", "number"])
def test_an_absurdly_long_value_is_not_parsed_as_a_typed_value(kind):
    value, warnings = normalize_value("onehundredand" * 5000, FieldSpec(name="f", type=kind))
    assert value is None and "too long" in warnings[0]
    assert len("one hundred and fifty thousand") < MAX_TYPED_LENGTH


@pytest.mark.parametrize("pattern", ["(a+)+$", r"(\d*)*x", "(?:ab*)+c*(x+)+", r"((a|b)+)*"])
def test_a_pattern_that_can_take_exponential_time_is_refused(pattern):
    with pytest.raises(ValueError, match="nested repeat|repeats a group"):
        FieldSpec(name="f", pattern=pattern)


@pytest.mark.parametrize("pattern", [r"[A-Z]{2}\d{6}", r"K[A-Z]{2}\s?\d{3}\s?[A-Z]?", r"(A|B|AB|O)\s?[+-]?(VE)?",
                                     r"(19|20)\d{2}", r"(\d{3}\s?){2,4}", r"GM-\d+", r"[\d,]{5,}", "("])
def test_ordinary_patterns_are_accepted(pattern):
    FieldSpec(name="f", pattern=pattern)


def test_shipped_document_types_still_load():
    from dococr import list_doc_types, load_doc_type

    assert all(load_doc_type(name).fields for name in list_doc_types())


# ---- the HTTP service -------------------------------------------------------------------------

@pytest.fixture
def client(monkeypatch):
    from dococr.api import app

    monkeypatch.setattr("dococr.pipeline.get_extractor", lambda: Fake())
    return TestClient(app)


def post(client, data=None, content=None):
    form = {"doc_type": "x", "fields": "a", **(data or {})}
    return client.post("/extract", data=form, files={"file": ("a.png", content or blank_png(64, 64), "image/png")})


def test_service_refuses_image_bombs_and_strips(client):
    assert post(client, content=blank_png(13300, 13300)).status_code == 422
    assert post(client, content=blank_png(3, 3000)).status_code == 422
    assert post(client).status_code == 200


def test_service_bounds_the_fields_parameter(client):
    assert post(client, {"fields": "[" * 100000}).status_code == 422  # was an unhandled error
    assert post(client, {"fields": ",".join(f"f{i}" for i in range(500))}).status_code == 422
    assert post(client, {"fields": "x" * 500}).status_code == 422
    assert post(client, {"fields": "a," * 30000}).status_code == 422
    assert post(client, {"doc_type": "d" * 5000}).status_code == 422


def test_service_does_not_show_engine_internals(monkeypatch):
    from dococr.api import app

    class Bad(Fake):
        def extract(self, *a):
            raise ExtractorError("RapidOCR failed: error: OpenCV(5.0.0) /io/opencv/modules/imgproc/src/resize.cpp:4223")

    monkeypatch.setattr("dococr.pipeline.get_extractor", lambda: Bad())
    r = post(TestClient(app))
    assert r.status_code == 503 and "opencv" not in r.text.lower() and "/io/" not in r.text


def test_api_description_pages_are_off_by_default(client):
    for path in ("/docs", "/redoc", "/openapi.json"):
        assert client.get(path).status_code == 404


def test_service_limits_concurrent_extractions(monkeypatch):
    import threading

    from dococr import api

    inside = []
    most = []
    gate = threading.Event()

    class Slow(Fake):
        def extract(self, image, doc_type, fields):
            inside.append(1)
            most.append(len(inside))
            gate.wait(5)
            inside.pop()
            return super().extract(image, doc_type, fields)

    monkeypatch.setattr("dococr.pipeline.get_extractor", lambda: Slow())
    monkeypatch.setattr(api, "_extract_slots", threading.BoundedSemaphore(2))
    monkeypatch.setattr(api, "BUSY_WAIT_SECONDS", 0.3)
    client = TestClient(api.app)
    codes = []
    threads = [threading.Thread(target=lambda: codes.append(post(client).status_code)) for _ in range(6)]
    for t in threads:
        t.start()
    threading.Timer(1.0, gate.set).start()
    for t in threads:
        t.join()
    assert max(most) <= 2
    assert sorted(set(codes)) == [200, 503]  # two ran, the rest were told to come back
