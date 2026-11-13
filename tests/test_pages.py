"""PDF and multi-page input: sources, merging, limits, bad files. Fake extractor, no OCR model."""

import io
import sys
from pathlib import Path

import pytest
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
import pdfs  # noqa: E402

from dococr import Extractor, ExtractorOutput, FieldResult, PageLimitError, SourceError, extract  # noqa: E402
from dococr.imaging import RENDER_LONG_SIDE, open_pages  # noqa: E402

FIELDS = ["registration_number", "make", "colour"]


class Scripted(Extractor):
    """Answers per page, where the page is told apart by its solid colour (see pdfs.page_index).

    script[page_index] = {field: (raw, confidence)} ; extras: per-page warnings and text.
    """

    name = "fake"

    def __init__(self, script, warnings=None, text=None):
        self.script = script
        self.page_warnings = warnings or {}
        self.text = text or {}
        self.calls = []

    def extract(self, image, doc_type, fields):
        k = pdfs.page_index(image)
        self.calls.append((k, image.size))
        answers = self.script.get(k, {})
        out = []
        for f in fields:
            raw, conf = answers.get(f.name, (None, None))
            out.append(FieldResult(name=f.name, raw=raw, confidence=conf, bbox=(1, 2, 3, 4) if raw else None))
        return ExtractorOutput(fields=out, raw_text=self.text.get(k), warnings=self.page_warnings.get(k, []))


def run(source, script, **kw):
    ex = Scripted(script, kw.pop("warnings", None), kw.pop("text", None))
    return extract(source, "logbook", FIELDS, extractor=ex, **kw), ex


TWO = {0: {"registration_number": ("KDA123A", 0.9)}, 1: {"make": ("Toyota", 0.8)}}


# ---- sources -------------------------------------------------------------------------------

def test_single_image_unchanged():
    res, _ = run(pdfs.png(), {0: {"registration_number": ("KDA123A", 0.9)}}, include_raw_text=True,
                 text={0: "TEXT"}, warnings={0: ["rotated 90 degrees"]})
    assert res.pages == 1
    f = res.fields["registration_number"]
    assert (f.value, f.page, f.bbox) == ("KDA123A", 1, (1, 2, 3, 4))
    assert res.fields["make"].page is None and res.fields["make"].value is None
    assert res.missing == ["make", "colour"]
    assert res.warnings == ["rotated 90 degrees"]  # not prefixed with a page number
    assert res.raw_text == "TEXT"  # no separator for one page


def test_pdf_bytes_pages_in_order():
    res, ex = run(pdfs.image_pdf(3), {0: {}, 1: {}, 2: {}})
    assert res.pages == 3
    assert [k for k, _ in ex.calls] == [0, 1, 2]


def test_pdf_found_by_content_not_name(tmp_path):
    p = tmp_path / "scan.dat"  # wrong extension on purpose
    p.write_bytes(pdfs.image_pdf(2))
    res, ex = run(p, TWO)
    assert res.pages == 2 and res.fields["make"].page == 2
    # and a PNG named .pdf is still an image
    q = tmp_path / "photo.pdf"
    q.write_bytes(pdfs.png())
    res, _ = run(str(q), {0: {}})
    assert res.pages == 1


def test_pdf_with_junk_before_header():
    res, _ = run(b"\n\n junk \n" + pdfs.raw_pdf([(100, 100)]), {})
    assert res.pages == 1


def test_multipage_tiff_bytes_and_path(tmp_path):
    data = pdfs.tiff(3)
    res, ex = run(data, TWO)
    assert res.pages == 3 and [k for k, _ in ex.calls] == [0, 1, 2]
    p = tmp_path / "scan.tif"
    p.write_bytes(data)
    res, _ = run(p, TWO)
    assert res.pages == 3 and res.fields["make"].page == 2


def test_single_frame_tiff_and_png_are_one_page():
    buf = io.BytesIO()
    Image.new("RGB", (30, 30), pdfs.RED).save(buf, "TIFF")
    assert run(buf.getvalue(), {})[0].pages == 1
    assert run(pdfs.png(), {})[0].pages == 1


def test_list_of_mixed_sources_and_nesting(tmp_path):
    path = tmp_path / "p.png"
    path.write_bytes(pdfs.png(pdfs.BLUE))
    source = [pdfs.image_pdf(2), [Image.new("RGB", (20, 20), pdfs.GREEN), str(path)]]
    res, ex = run(source, {})
    assert res.pages == 4
    assert [k for k, _ in ex.calls] == [0, 1, 1, 2]
    res, _ = run((pdfs.png(), pdfs.png()), {})
    assert res.pages == 2


def test_pil_image_source_is_one_page():
    assert run(Image.new("RGB", (20, 20), pdfs.RED), {})[0].pages == 1


def test_empty_list_and_odd_type():
    with pytest.raises(SourceError, match="no pages"):
        run([], {})
    with pytest.raises(SourceError):
        run([42], {})


def test_pdf_render_size_is_capped():
    for box in [(595, 842), (14400, 14400), (14400, 20), (30, 30)]:
        with open_pages(pdfs.raw_pdf([box])) as pages:
            (image,) = list(pages)
        assert RENDER_LONG_SIDE <= max(image.size) <= RENDER_LONG_SIDE + 1, (box, image.size)
        assert image.mode == "RGB"


def test_enormous_page_does_not_allocate_enormous_image():
    with open_pages(pdfs.raw_pdf([(14400, 14400)])) as pages:
        (image,) = list(pages)
    assert image.width * image.height <= (RENDER_LONG_SIDE + 1) ** 2


# ---- merge rule ----------------------------------------------------------------------------

def test_fields_from_each_page_with_page_numbers():
    res, _ = run(pdfs.image_pdf(2), TWO)
    assert res.pages == 2
    a, b, c = (res.fields[n] for n in FIELDS)
    assert (a.value, a.page, a.warnings) == ("KDA123A", 1, [])
    assert (b.value, b.page, b.warnings) == ("Toyota", 2, [])
    assert (c.value, c.page, c.confidence, c.bbox) == (None, None, None, None)
    assert res.missing == ["colour"]


def test_highest_confidence_wins_and_conflict_warns():
    script = {0: {"make": ("Toyota", 0.7)}, 1: {"make": ("Toyoto", 0.95)}}
    res, _ = run(pdfs.image_pdf(2), script)
    f = res.fields["make"]
    assert (f.value, f.page, f.confidence) == ("Toyoto", 2, 0.95)
    assert len(f.warnings) == 1 and "page 1" in f.warnings[0] and "page 2" in f.warnings[0]
    assert "Toyota" not in f.warnings[0]  # the warning names pages, not values


def test_tie_goes_to_earlier_page():
    script = {0: {"make": ("Alpha", 0.8)}, 1: {"make": ("Beta", 0.8)}, 2: {"make": ("Gamma", 0.8)}}
    res, _ = run(pdfs.image_pdf(3), script)
    f = res.fields["make"]
    assert (f.value, f.page) == ("Alpha", 1)
    assert "pages 2, 3" in f.warnings[0]


def test_same_value_on_several_pages_is_no_warning():
    script = {0: {"make": ("Toyota", 0.7)}, 1: {"make": ("  Toyota ", 0.9)}}
    res, _ = run(pdfs.image_pdf(2), script)
    f = res.fields["make"]
    assert (f.value, f.page, f.warnings) == ("Toyota", 2, [])


def test_values_compared_after_normalising():
    # typed as dates in a user type would be better; identifiers are upper-cased and unspaced
    script = {0: {"registration_number": ("kda 123a", 0.7)}, 1: {"registration_number": ("KDA123A", 0.6)}}
    ex = Scripted(script)
    from dococr import FieldSpec

    spec = FieldSpec(name="registration_number", type="identifier")
    res = extract(pdfs.image_pdf(2), "logbook", [spec], extractor=ex)
    assert res.fields["registration_number"].warnings == []


def test_unknown_confidence_loses_to_known():
    script = {0: {"make": ("Alpha", None)}, 1: {"make": ("Beta", 0.1)}}
    res, _ = run(pdfs.image_pdf(2), script)
    assert res.fields["make"].value == "Beta" and res.fields["make"].page == 2


def test_page_level_warnings_name_the_page_only_when_several():
    res, _ = run(pdfs.image_pdf(2), {}, warnings={1: ["rotated 90 degrees"]})
    assert res.warnings == ["page 2: rotated 90 degrees"]


def test_missing_field_keeps_page_warnings_with_page():
    class W(Scripted):
        def extract(self, image, doc_type, fields):
            out = super().extract(image, doc_type, fields)
            out.fields[0].warnings.append("label found, no value")
            return out

    res = extract(pdfs.image_pdf(2), "logbook", FIELDS, extractor=W({}))
    assert res.fields["registration_number"].warnings == [
        "page 1: label found, no value", "page 2: label found, no value"]


def test_raw_text_covers_all_pages():
    res, _ = run(pdfs.image_pdf(2), {}, include_raw_text=True, text={0: "ONE", 1: "TWO"})
    assert res.raw_text == "--- page 1 ---\nONE\n--- page 2 ---\nTWO"
    res, _ = run(pdfs.image_pdf(2), {}, text={0: "ONE"})
    assert res.raw_text is None  # not asked for


def test_result_json_shape():
    res, _ = run(pdfs.image_pdf(2), {0: {"make": ("A", 0.9)}, 1: {"make": ("B", 0.5)}})
    data = res.model_dump(mode="json")
    assert data["pages"] == 2 and data["fields"]["make"]["page"] == 1
    assert data["fields"]["colour"]["page"] is None


# ---- limits --------------------------------------------------------------------------------

def test_page_limit_is_an_error_before_any_ocr():
    ex = Scripted({})
    with pytest.raises(PageLimitError, match="3 pages; the limit is 2"):
        extract(pdfs.image_pdf(3), "logbook", FIELDS, extractor=ex, max_pages=2)
    assert ex.calls == []


def test_page_limit_exact_is_allowed_and_default_is_set():
    assert run(pdfs.image_pdf(2), {}, max_pages=2)[0].pages == 2
    from dococr.imaging import DEFAULT_MAX_PAGES

    assert 1 < DEFAULT_MAX_PAGES <= 100
    with pytest.raises(PageLimitError):
        run(pdfs.pages(DEFAULT_MAX_PAGES + 1), {})


def test_page_limit_counts_a_list_across_sources():
    with pytest.raises(PageLimitError):
        run([pdfs.image_pdf(2), pdfs.tiff(2)], {}, max_pages=3)


def test_page_limit_stops_opening_a_long_list():
    with pytest.raises(PageLimitError, match="more than 2"):
        run([pdfs.png()] * 3 + [b"not an image"], {}, max_pages=2)  # the bad third part is never opened


def test_huge_pdf_is_refused_without_rendering(monkeypatch):
    import pypdfium2

    rendered = []
    real = pypdfium2.PdfPage.render
    monkeypatch.setattr(pypdfium2.PdfPage, "render", lambda self, *a, **k: rendered.append(1) or real(self, *a, **k))
    with pytest.raises(PageLimitError):
        run(pdfs.raw_pdf([(100, 100)] * 50), {}, max_pages=5)
    assert rendered == []


def test_max_pages_must_be_positive():
    with pytest.raises(ValueError):
        run(pdfs.png(), {}, max_pages=0)


# ---- bad PDFs ------------------------------------------------------------------------------

@pytest.mark.parametrize(
    "data",
    [b"%PDF-1.4\nthis is not a pdf", pdfs.image_pdf(1)[:150], b"%PDF-", pdfs.raw_pdf([])],
    ids=["garbage", "truncated", "header-only", "no-pages"],
)
def test_damaged_pdf_is_a_short_error(data):
    with pytest.raises(SourceError) as info:
        run(data, {})
    msg = str(info.value)
    assert "PDF" in msg and len(msg) < 100 and "pdfium" not in msg.lower()
    assert "encrypted" not in msg  # an earlier encrypted file must not colour this one


def test_encrypted_pdf_says_so():
    with pytest.raises(SourceError, match="encrypted"):
        run(pdfs.raw_pdf([(100, 100)], encrypt=True), {})
    # a no-pages PDF read afterwards is not blamed on a password (PDFium's stale last error)
    with pytest.raises(SourceError) as info:
        run(pdfs.raw_pdf([]), {})
    assert "encrypted" not in str(info.value)


def test_encrypted_pdf_from_path(tmp_path):
    p = tmp_path / "locked.pdf"
    p.write_bytes(pdfs.raw_pdf([(100, 100)], encrypt=True))
    with pytest.raises(SourceError, match="encrypted"):
        run(p, {})


def test_damaged_pdf_inside_a_list_is_still_an_error():
    with pytest.raises(SourceError):
        run([pdfs.png(), b"%PDF-1.4\njunk"], {})


# ---- start-up ------------------------------------------------------------------------------

def test_pdf_library_not_imported_by_import_or_types_command():
    import subprocess

    code = (
        "import sys\n"
        "import dococr\n"
        "from typer.testing import CliRunner\n"
        "from dococr.cli import app\n"
        "r = CliRunner().invoke(app, ['types'])\n"
        "assert r.exit_code == 0, r.output\n"
        "assert 'pypdfium2' not in sys.modules\n"
        "from dococr.imaging import load_image\n"
        "assert 'pypdfium2' not in sys.modules\n"
    )
    subprocess.run([sys.executable, "-c", code], check=True, timeout=120)
