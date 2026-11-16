"""HTTP service tests with a fake extractor injected through the pipeline."""

import io
import sys

import pytest
from fastapi.testclient import TestClient
from PIL import Image

from dococr import Extractor, ExtractorError, ExtractorOutput, FieldResult
from dococr.api import app


class Fake(Extractor):
    name = "fake"

    def extract(self, image, doc_type, fields):
        return ExtractorOutput(
            fields=[FieldResult(name=f.name, raw="KDA 123A" if f.name == "registration_number" else None)
                    for f in fields]
        )


@pytest.fixture
def client(monkeypatch):
    monkeypatch.setattr("dococr.pipeline.get_extractor", lambda: Fake())
    return TestClient(app)


def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 40), "white").save(buf, "PNG")
    return buf.getvalue()


def post(client, data=None, content=None, **extra):
    form = {"doc_type": "logbook", "fields": "registration_number,make", **(data or {})}
    return client.post("/extract", data=form, files={"file": ("a.png", png() if content is None else content, "image/png")}, **extra)


def test_health_and_doc_types(client):
    assert client.get("/health").json() == {"status": "ok"}
    names = [d["name"] for d in client.get("/doc-types").json()]
    assert "logbook" in names
    doc = client.get("/doc-types/logbook").json()
    assert any(f["name"] == "registration_number" for f in doc["fields"])
    assert client.get("/doc-types/nope").status_code == 404
    assert client.get("/doc-types/x.yaml").status_code == 404


def test_extract(client):
    r = post(client)
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["backend"] == "fake"
    assert body["fields"]["registration_number"]["raw"] == "KDA 123A"
    assert body["missing"] == ["make"]


def test_backend_cannot_be_chosen(monkeypatch):
    """A `backend` form field is ignored: there is one engine and it is offline."""
    seen = []
    monkeypatch.setattr("dococr.pipeline.get_extractor", lambda: seen.append(1) or Fake())
    r = post(TestClient(app), {"backend": "llm"})
    assert r.status_code == 200 and r.json()["backend"] == "fake" and seen == [1]


def test_unreadable_image_422(client):
    r = post(client, content=b"not an image")
    assert r.status_code == 422 and "readable image" in r.json()["detail"]


def test_empty_file_422(client):
    assert post(client, content=b"").status_code == 422


def test_no_preset_no_fields_422(client):
    r = post(client, {"doc_type": "invoice", "fields": ""})
    assert r.status_code == 422 and "no YAML file" in r.json()["detail"]


def test_doc_type_path_rejected(client):
    assert post(client, {"doc_type": "/etc/x.yaml"}).status_code == 422


def test_upload_size_limit(client, monkeypatch):
    monkeypatch.setenv("DOCOCR_MAX_UPLOAD_MB", "0.00002")  # ~20 bytes
    assert post(client).status_code == 413


def test_extractor_error_503(monkeypatch):
    class Bad(Fake):
        def extract(self, *a):
            raise ExtractorError("OCR model file missing")

    monkeypatch.setattr("dococr.pipeline.get_extractor", lambda: Bad())
    r = post(TestClient(app))
    # the engine's own message stays in the server log; the client gets a plain statement
    assert r.status_code == 503 and "model file" not in r.text
    assert r.json()["detail"] == "the OCR engine could not process this document"


def test_upload_not_written_to_disk(client, monkeypatch):
    """A phone-photo-sized upload must stay in memory. Starlette's default is to move file
    parts over 1 MB into a temp file; that file is unlinked at once, so a directory listing
    cannot see it — watch the spool's rollover-to-disk instead."""
    import os
    import tempfile

    rolled = []
    real = tempfile.SpooledTemporaryFile.rollover
    monkeypatch.setattr(
        tempfile.SpooledTemporaryFile, "rollover", lambda self: rolled.append(1) or real(self)
    )
    buf = io.BytesIO()
    Image.frombytes("RGB", (1000, 1000), os.urandom(3_000_000)).save(buf, "PNG")
    assert len(buf.getvalue()) > 2 * 1024 * 1024
    assert post(client, content=buf.getvalue()).status_code == 200
    assert rolled == []


def test_oversized_upload_refused_before_it_is_read(client, monkeypatch):
    monkeypatch.setenv("DOCOCR_MAX_UPLOAD_MB", "1")
    r = post(client, content=b"x" * (2 * 1024 * 1024))
    assert r.status_code == 413 and "1 MB" in r.json()["detail"]


def test_file_just_over_limit_413(client, monkeypatch):
    # Under the whole-body cap (limit + form overhead) but over the per-file limit.
    monkeypatch.setenv("DOCOCR_MAX_UPLOAD_MB", "1")
    assert post(client, content=b"x" * (1024 * 1024 + 100)).status_code == 413


def _user_dir(tmp_path, monkeypatch):
    d = tmp_path / "types"
    d.mkdir()
    (d / "widget.yaml").write_text(
        'name: gizmo\ntitle: Gizmo sheet\naliases: ["gadget"]\nfields:\n  - name: registration_number\n  - name: make\n'
    )
    (d / "broken.yaml").write_text("name: [unclosed\n")
    monkeypatch.setenv("DOCOCR_DOCTYPES_DIR", str(d))


def test_user_type_served_by_name_and_alias(client, tmp_path, monkeypatch):
    _user_dir(tmp_path, monkeypatch)
    assert {"name": "gizmo", "title": "Gizmo sheet"} in client.get("/doc-types").json()
    assert client.get("/doc-types/gadget").json()["name"] == "gizmo"
    r = post(client, {"doc_type": "gadget", "fields": ""})
    assert r.status_code == 200 and r.json()["doc_type"] == "gizmo"
    assert set(r.json()["fields"]) == {"registration_number", "make"}


def test_broken_user_file_is_named_not_fatal(client, tmp_path, monkeypatch):
    _user_dir(tmp_path, monkeypatch)
    assert client.get("/doc-types").status_code == 200
    r = client.get("/doc-types/broken")
    assert r.status_code == 422 and "broken.yaml" in r.json()["detail"]


def test_path_still_rejected_with_user_folder(client, tmp_path, monkeypatch):
    _user_dir(tmp_path, monkeypatch)
    r = post(client, {"doc_type": str(tmp_path / "types" / "widget.yaml")})
    assert r.status_code == 422 and "not a file path" in r.json()["detail"]


# ---- start-up warming -------------------------------------------------------------------------

def _count_warm_ups(monkeypatch):
    calls = []
    monkeypatch.setattr("dococr.pipeline.warm_up", lambda: calls.append(1))
    return calls


def test_plain_test_client_does_not_warm(monkeypatch):
    calls = _count_warm_ups(monkeypatch)
    TestClient(app).get("/health")
    assert calls == []


def test_service_warms_at_startup(monkeypatch):
    monkeypatch.delenv("DOCOCR_WARMUP", raising=False)
    calls = _count_warm_ups(monkeypatch)
    with TestClient(app) as c:
        assert calls == [1]  # already done before the first request
        assert c.get("/health").status_code == 200


@pytest.mark.parametrize("value", ["0", "false", "off", "No"])
def test_warming_can_be_turned_off(monkeypatch, value):
    monkeypatch.setenv("DOCOCR_WARMUP", value)
    calls = _count_warm_ups(monkeypatch)
    with TestClient(app):
        pass
    assert calls == []


def test_failed_warm_up_stops_startup(monkeypatch):
    monkeypatch.delenv("DOCOCR_WARMUP", raising=False)

    def boom():
        raise ExtractorError("models missing")

    monkeypatch.setattr("dococr.pipeline.warm_up", boom)
    with pytest.raises(ExtractorError, match="models missing"):
        with TestClient(app):
            pass


# ---- PDFs and several pages ------------------------------------------------------------------

class PerPage(Extractor):
    name = "fake"

    def extract(self, image, doc_type, fields):
        import pdfs

        k = pdfs.page_index(image)
        answers = {0: {"registration_number": "KDA 123A"}, 1: {"make": "Toyota"}}.get(k, {})
        return ExtractorOutput(
            fields=[FieldResult(name=f.name, raw=answers.get(f.name), confidence=0.9 if f.name in answers else None)
                    for f in fields],
            raw_text=f"text {k}",
        )


@pytest.fixture
def pages_client(monkeypatch):
    monkeypatch.setattr("dococr.pipeline.get_extractor", lambda: PerPage())
    return TestClient(app)


def post_files(client, files, **form):
    data = {"doc_type": "logbook", "fields": "registration_number,make", **form}
    return client.post("/extract", data=data, files=[("file", f) for f in files])


def test_pdf_upload(pages_client):
    import pdfs

    r = post_files(pages_client, [("scan.pdf", pdfs.image_pdf(2), "application/pdf")])
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["pages"] == 2
    assert body["fields"]["registration_number"]["page"] == 1
    assert body["fields"]["make"]["page"] == 2 and body["fields"]["make"]["value"] == "Toyota"


def test_pdf_recognised_by_content_not_name_or_type(pages_client):
    import pdfs

    r = post_files(pages_client, [("scan.png", pdfs.image_pdf(2), "image/png")])
    assert r.status_code == 200 and r.json()["pages"] == 2


def test_several_parts_are_the_pages_of_one_document(pages_client):
    import pdfs

    r = post_files(
        pages_client,
        [("a.png", pdfs.png(pdfs.RED), "image/png"), ("b.png", pdfs.png(pdfs.GREEN), "image/png")],
        include_raw_text="true",
    )
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["pages"] == 2 and body["fields"]["make"]["page"] == 2
    assert body["raw_text"] == "--- page 1 ---\ntext 0\n--- page 2 ---\ntext 1"


def test_single_part_has_one_page(pages_client):
    r = post(pages_client)
    assert r.json()["pages"] == 1 and r.json()["fields"]["registration_number"]["page"] == 1


def test_page_limit_from_environment(pages_client, monkeypatch):
    import pdfs

    monkeypatch.setenv("DOCOCR_MAX_PAGES", "2")
    r = post_files(pages_client, [("s.pdf", pdfs.image_pdf(3), "application/pdf")])
    assert r.status_code == 422 and "limit is 2" in r.json()["detail"]
    parts = [("p.png", pdfs.png(), "image/png")] * 3
    r = post_files(pages_client, parts)
    assert r.status_code == 422 and "limit" in r.json()["detail"]
    assert post_files(pages_client, parts[:2]).status_code == 200


def test_bad_environment_value_falls_back_to_default(monkeypatch):
    from dococr.api import max_pages
    from dococr.imaging import DEFAULT_MAX_PAGES

    for value in ("abc", "0", "-3"):
        monkeypatch.setenv("DOCOCR_MAX_PAGES", value)
        assert max_pages() == DEFAULT_MAX_PAGES


@pytest.mark.parametrize(
    "data", [b"%PDF-1.4\njunk", None, "empty"], ids=["damaged", "encrypted", "no-pages"]
)
def test_bad_pdfs_are_422_with_a_short_message(pages_client, data):
    import pdfs

    if data is None:
        data = pdfs.raw_pdf([(100, 100)], encrypt=True)
    elif data == "empty":
        data = pdfs.raw_pdf([])
    r = post_files(pages_client, [("s.pdf", data, "application/pdf")])
    assert r.status_code == 422
    assert "PDF" in r.json()["detail"] and len(r.json()["detail"]) < 100


def test_size_limit_is_for_the_whole_request(pages_client, monkeypatch):
    import pdfs

    monkeypatch.setenv("DOCOCR_MAX_UPLOAD_MB", "0.01")  # 10 KB in all
    part = ("p.png", pdfs.png(size=(40, 40)), "image/png")
    big = ("q.bin", b"x" * 6000, "image/png")
    assert post_files(pages_client, [big]).status_code == 422  # under the limit, just not an image
    r = post_files(pages_client, [big, big])  # each part under the limit, the sum over it
    assert r.status_code == 413 and "limit" in r.json()["detail"]
    assert post_files(pages_client, [part, part]).status_code == 200


_watching: list = []
_watch_writes: list = []


def _audit(event, args):
    # Python-level opens for writing while a test is watching (PDFium's own C-level file use is
    # checked by strace, see the report; this catches our code and Pillow).
    if _watching and event == "open" and isinstance(args[1], str) and set(args[1] or "") & set("wax+"):
        _watch_writes.append(args[0])
    if _watching and event == "tempfile.mkstemp":
        _watch_writes.append(args[0])


sys.addaudithook(_audit)


def test_pdf_upload_not_written_to_disk(pages_client, monkeypatch):
    import os
    import tempfile

    import pdfs

    rolled = []
    real = tempfile.SpooledTemporaryFile.rollover
    monkeypatch.setattr(
        tempfile.SpooledTemporaryFile, "rollover", lambda self: rolled.append(1) or real(self)
    )
    _watch_writes.clear()
    _watching.append(1)
    noisy = [Image.frombytes("RGB", (1000, 1000), os.urandom(3_000_000)) for _ in range(2)]
    buf = io.BytesIO()
    noisy[0].save(buf, "PDF", save_all=True, append_images=noisy[1:], quality=100)
    assert len(buf.getvalue()) > 2 * 1024 * 1024
    r = post_files(pages_client, [("s.pdf", buf.getvalue(), "application/pdf")])
    assert r.status_code == 200, r.text
    _watching.clear()
    assert rolled == [] and _watch_writes == []


def test_pdf_contents_are_not_logged(pages_client, caplog):
    import logging

    import pdfs

    caplog.set_level(logging.DEBUG)
    r = post_files(pages_client, [("s.pdf", pdfs.image_pdf(2), "application/pdf")])
    assert r.status_code == 200
    text = " ".join(rec.getMessage() for rec in caplog.records)
    assert "KDA123A" not in text and "KDA 123A" not in text and "Toyota" not in text


def test_pdf_text_setting_from_the_environment(client, monkeypatch):
    seen = {}
    import dococr.pipeline as pipeline

    real = pipeline.extract

    def spy(*a, **k):
        seen.update(k)
        return real(*a, **k)

    monkeypatch.setattr(pipeline, "extract", spy)
    assert post(client).status_code == 200 and seen["pdf_text"] == "auto"
    monkeypatch.setenv("DOCOCR_PDF_TEXT", "never")
    assert post(client).status_code == 200 and seen["pdf_text"] == "never"
    monkeypatch.setenv("DOCOCR_PDF_TEXT", "Always ")
    assert post(client).status_code == 200 and seen["pdf_text"] == "always"
    monkeypatch.setenv("DOCOCR_PDF_TEXT", "sometimes")
    r = post(client)
    assert r.status_code == 500 and "DOCOCR_PDF_TEXT" in r.json()["detail"]
