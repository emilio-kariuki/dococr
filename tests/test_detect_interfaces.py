"""Type detection through the CLI and the HTTP service, with a fake reader and invented types.
No OCR model."""

from __future__ import annotations

import io
import json

import pytest
from fastapi.testclient import TestClient
from PIL import Image
from typer.testing import CliRunner

from dococr import Extractor, ExtractorOutput, FieldResult
from dococr import detection as D
from dococr.api import app as api_app
from dococr.cli import app as cli_app
from dococr.extractors.base import PageRead
from dococr.extractors.local import LocalExtractor
from dococr.ocr import OcrLine

PERMIT = """name: permit
title: Pastoral grazing permit
identifying_text: ["Pastoral Grazing Permit"]
fields:
  - {name: permit_code, label: Permit Code, type: identifier}
  - {name: holder_name, label: Holder Name, type: name}
  - {name: grazing_zone, label: Grazing Zone}
  - {name: herd_size, label: Herd Size, type: integer}
  - {name: date_of_birth, label: Date of Birth, type: date}
"""
TICKET = """name: ticket
title: Stadium ticket
fields:
  - {name: ticket_serial, label: Ticket Serial}
  - {name: holder_name, label: Holder Name, type: name}
  - {name: seat_row, label: Seat Row}
  - {name: gate_letter, label: Gate Letter}
  - {name: date_of_birth, label: Date of Birth, type: date}
"""


def line(y, x, text):
    return OcrLine(text, 0.95, (x, y, x + 13 * len(text), y + 24))


PERMIT_PAGE = [line(20, 300, "PASTORAL GRAZING PERMIT"),
               line(80, 40, "PERMIT CODE"), line(80, 420, "PX-55102"),
               line(130, 40, "HOLDER NAME"), line(130, 420, "ZANELE OKAFOR"),
               line(180, 40, "GRAZING ZONE"), line(180, 420, "NORTH 4"),
               line(230, 40, "HERD SIZE"), line(230, 420, "37")]
OTHER_PAGE = [line(80, 40, "INVOICE NO"), line(80, 420, "88"),
              line(130, 40, "AMOUNT DUE"), line(130, 420, "1,200.00")]


class Reader(LocalExtractor):
    name = "reader"

    def __init__(self, lines):
        super().__init__(engine=object())
        self.lines = lines
        self.reads = 0

    def read_page(self, image, *, text_layer=None):
        self.reads += 1
        return PageRead(self.lines, (1000, 800))


class NoLines(Extractor):
    name = "nolines"

    def extract(self, image, doc_type, fields):
        return ExtractorOutput(fields=[FieldResult(name=f.name) for f in fields])


@pytest.fixture
def types_dir(tmp_path, monkeypatch):
    folder = tmp_path / "types"
    folder.mkdir()
    (folder / "permit.yaml").write_text(PERMIT, encoding="utf-8")
    (folder / "ticket.yaml").write_text(TICKET, encoding="utf-8")
    monkeypatch.setenv("DOCOCR_DOCTYPES_DIR", str(folder))
    return folder


@pytest.fixture
def reader(monkeypatch):
    holder = {"lines": PERMIT_PAGE}
    made = []

    def getter():
        r = Reader(holder["lines"])
        made.append(r)
        return r

    monkeypatch.setattr("dococr.pipeline.get_extractor", getter)
    holder["made"] = made
    return holder


@pytest.fixture
def img(tmp_path):
    p = tmp_path / "doc.png"
    Image.new("RGB", (50, 50), "white").save(p)
    return str(p)


runner = CliRunner()
CANDS = ["--candidates", "permit,ticket"]


# ---------------------------------------------------------------------------- CLI

def test_cli_extract_without_type_detects_it(types_dir, reader, img):
    r = runner.invoke(cli_app, ["extract", img, *CANDS])
    assert r.exit_code == 0, r.output
    body = json.loads(r.output)
    assert body["doc_type"] == "permit"
    assert body["detection"]["doc_type"] == "permit"
    assert body["fields"]["permit_code"]["value"] == "PX-55102"
    assert reader["made"][0].reads == 1


def test_cli_type_auto_and_table(types_dir, reader, img):
    r = runner.invoke(cli_app, ["extract", img, "--type", "auto", "--format", "table", *CANDS])
    assert r.exit_code == 0, r.output
    assert "[permit]" in r.output and "detected: permit" in r.output
    assert "Permit Code" in r.output


def test_cli_extract_unknown_document_fails_with_the_ranking(types_dir, reader, img):
    reader["lines"] = OTHER_PAGE
    r = runner.invoke(cli_app, ["extract", img, *CANDS])
    assert r.exit_code == 1
    assert "could not be detected" in r.output
    assert "detected: unknown" in r.output and "permit" in r.output


def test_cli_detect_command(types_dir, reader, img):
    r = runner.invoke(cli_app, ["detect", img, *CANDS])
    assert r.exit_code == 0, r.output
    assert "detected: permit" in r.output
    assert "ticket" in r.output
    r = runner.invoke(cli_app, ["detect", img, "--format", "json", *CANDS])
    body = json.loads(r.output)
    assert body["doc_type"] == "permit"
    assert body["candidates"][0]["evidence"]
    assert body["considered"] == 2


def test_cli_detect_unknown_exits_zero(types_dir, reader, img):
    reader["lines"] = OTHER_PAGE
    r = runner.invoke(cli_app, ["detect", img, "--format", "json", *CANDS])
    assert r.exit_code == 0, r.output
    assert json.loads(r.output)["doc_type"] is None


def test_cli_detect_among_all_types_found(types_dir, reader, img):
    r = runner.invoke(cli_app, ["detect", img, "--format", "json"])
    assert r.exit_code == 0, r.output
    body = json.loads(r.output)
    assert body["doc_type"] == "permit"
    assert body["considered"] >= 2


def test_cli_candidate_cap(types_dir, reader, img, monkeypatch):
    monkeypatch.setenv(D.MAX_CANDIDATES_ENV, "1")
    r = runner.invoke(cli_app, ["detect", img])
    assert r.exit_code == 1 and "DOCOCR_MAX_CANDIDATES" in r.output


def test_cli_extractor_without_lines(types_dir, img, monkeypatch):
    monkeypatch.setattr("dococr.pipeline.get_extractor", lambda: NoLines())
    r = runner.invoke(cli_app, ["extract", img])
    assert r.exit_code == 1 and "cannot detect the document type" in r.output


# ---------------------------------------------------------------------------- HTTP

def png() -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", (40, 40), "white").save(buf, "PNG")
    return buf.getvalue()


@pytest.fixture
def client(types_dir, reader):
    return TestClient(api_app)


def post(client, path, data=None):
    return client.post(path, data=data or {}, files={"file": ("a.png", png(), "image/png")})


def test_api_extract_without_doc_type(client):
    r = post(client, "/extract", {"candidates": "permit,ticket"})
    assert r.status_code == 200, r.text
    body = r.json()
    assert body["doc_type"] == "permit" and body["detection"]["doc_type"] == "permit"
    r = post(client, "/extract", {"doc_type": "auto", "fields": "herd_size"})
    assert r.status_code == 200, r.text
    assert r.json()["fields"]["herd_size"]["value"] == 37


def test_api_extract_named_type_has_no_detection(client):
    r = post(client, "/extract", {"doc_type": "permit"})
    assert r.status_code == 200, r.text
    assert r.json()["detection"] is None


def test_api_unknown_document_is_422_with_the_ranking(client, reader):
    reader["lines"] = OTHER_PAGE
    r = post(client, "/extract", {"candidates": "permit,ticket"})
    assert r.status_code == 422
    detail = r.json()["detail"]
    assert "could not be detected" in detail["message"]
    assert detail["detection"]["doc_type"] is None
    assert {c["doc_type"] for c in detail["detection"]["candidates"]} == {"permit", "ticket"}


def test_api_detect(client, reader):
    r = post(client, "/detect", {"candidates": '["permit", "ticket"]'})
    assert r.status_code == 200, r.text
    assert r.json()["doc_type"] == "permit"
    reader["lines"] = OTHER_PAGE
    r = post(client, "/detect")
    assert r.status_code == 200 and r.json()["doc_type"] is None


def test_api_candidates_are_names_only_and_capped(client, monkeypatch):
    r = post(client, "/detect", {"candidates": "permit,/etc/x.yaml"})
    assert r.status_code == 422 and "not file paths" in r.text
    r = post(client, "/detect", {"candidates": "permit,nope"})
    assert r.status_code == 422 and "unknown document type" in r.text
    monkeypatch.setenv(D.MAX_CANDIDATES_ENV, "1")
    r = post(client, "/detect", {"candidates": "permit,ticket"})
    assert r.status_code == 422 and "more than 1" in r.text
    r = post(client, "/detect")  # every type found: more than the cap
    assert r.status_code == 422 and "DOCOCR_MAX_CANDIDATES" in r.text
    r = post(client, "/detect", {"candidates": "[1, 2]"})
    assert r.status_code == 422


def test_api_detect_keeps_the_upload_limits(client, monkeypatch):
    monkeypatch.setenv("DOCOCR_MAX_PAGES", "1")
    files = [("file", ("a.png", png(), "image/png")), ("file", ("b.png", png(), "image/png"))]
    r = client.post("/detect", files=files)
    assert r.status_code == 422 and "limit is 1" in r.text
    monkeypatch.setenv("DOCOCR_MAX_UPLOAD_MB", "0.00001")
    r = post(client, "/detect")
    assert r.status_code == 413
