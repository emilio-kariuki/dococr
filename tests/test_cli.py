"""CLI tests with a fake extractor injected through the pipeline."""

import json
import os

import pytest
from PIL import Image
from typer.testing import CliRunner

from dococr import Extractor, ExtractorError, ExtractorOutput, FieldResult
from dococr.cli import app

runner = CliRunner()


class Fake(Extractor):
    name = "fake"

    def extract(self, image, doc_type, fields):
        return ExtractorOutput(
            fields=[FieldResult(name=f.name, raw="ABC 1" if f.name == "registration_number" else None)
                    for f in fields]
        )


class Broken(Extractor):
    name = "broken"

    def extract(self, image, doc_type, fields):
        raise ExtractorError("OCR model file missing")


@pytest.fixture
def img(tmp_path):
    p = tmp_path / "doc.png"
    Image.new("RGB", (50, 50), "white").save(p)
    return str(p)


@pytest.fixture
def fake(monkeypatch):
    seen = []

    def getter():
        seen.append("local")
        return Fake()

    monkeypatch.setattr("dococr.pipeline.get_extractor", getter)
    return seen


def test_types_and_fields():
    r = runner.invoke(app, ["types"])
    assert r.exit_code == 0 and "logbook" in r.output and "passport" in r.output
    r = runner.invoke(app, ["fields", "logbook"])
    assert r.exit_code == 0 and "registration_number" in r.output


def test_fields_unknown_type():
    r = runner.invoke(app, ["fields", "nope"])
    assert r.exit_code == 1 and r.output.startswith("error:")


def test_extract_json_default_backend_local(img, fake):
    r = runner.invoke(app, ["extract", img, "--type", "logbook", "--fields", "registration_number,make"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert data["backend"] == "fake" and fake == ["local"]
    assert data["fields"]["registration_number"]["raw"] == "ABC 1"
    assert data["missing"] == ["make"]


def test_no_backend_option(img, fake):
    """There is one engine and it is offline; a cloud backend cannot be selected."""
    r = runner.invoke(app, ["extract", img, "-t", "logbook", "-f", "make", "--backend", "llm"])
    assert r.exit_code == 2 and fake == []


def test_several_images_give_list(img, fake):
    r = runner.invoke(app, ["extract", img, img, "-t", "logbook", "-f", "make"])
    assert isinstance(json.loads(r.output), list) and len(json.loads(r.output)) == 2


def test_table_format(img, fake):
    r = runner.invoke(app, ["extract", img, "-t", "logbook", "-f", "registration_number,make", "--format", "table"])
    assert r.exit_code == 0 and "(missing)" in r.output and "registration_number" in r.output


def test_doc_type_file_and_adhoc_fields(img, fake, tmp_path):
    f = tmp_path / "invoice.yaml"
    f.write_text("name: invoice\nfields:\n  - name: total\n")
    r = runner.invoke(app, ["extract", img, "--doc-type-file", str(f)])
    assert r.exit_code == 0 and json.loads(r.output)["doc_type"] == "invoice"


@pytest.mark.parametrize(
    "args",
    [
        ["extract", "/no/such.png", "-t", "logbook"],
        ["extract", "IMG", "-t", "invoice"],  # no preset, no fields
        ["extract", "IMG"],  # no type
    ],
)
def test_bad_input_is_short_message(img, fake, args):
    args = [img if a == "IMG" else a for a in args]
    r = runner.invoke(app, args)
    assert r.exit_code != 0
    assert r.exit_code in (1, 2) and "Traceback" not in r.output


def test_unreadable_image(tmp_path, fake):
    p = tmp_path / "bad.jpg"
    p.write_text("not an image")
    r = runner.invoke(app, ["extract", str(p), "-t", "logbook", "-f", "make"])
    assert r.exit_code == 1 and r.output.startswith("error:") and "Traceback" not in r.output


def test_extractor_error_message(img, monkeypatch):
    monkeypatch.setattr("dococr.pipeline.get_extractor", lambda: Broken())
    r = runner.invoke(app, ["extract", img, "-t", "logbook", "-f", "make"])
    assert r.exit_code == 1 and "error: OCR model file missing" in r.output


def test_missing_type_yaml_is_short_message(img, fake):
    for args in (["extract", img, "--type", "nofile.yaml"], ["fields", "nofile.yaml"]):
        r = runner.invoke(app, args)
        assert r.exit_code == 1 and r.output.startswith("error:") and "Traceback" not in r.output


def test_table_shows_page_text_when_asked(img, monkeypatch):
    class WithText(Fake):
        def extract(self, image, doc_type, fields):
            out = super().extract(image, doc_type, fields)
            return out.model_copy(update={"raw_text": "MAKE TOYOTA"})

    monkeypatch.setattr("dococr.pipeline.get_extractor", lambda: WithText())
    args = ["extract", img, "-t", "logbook", "-f", "make", "--format", "table"]
    assert "MAKE TOYOTA" in runner.invoke(app, [*args, "--raw-text"]).output
    assert "MAKE TOYOTA" not in runner.invoke(app, args).output


def _user_types(tmp_path):
    d = tmp_path / "mytypes"
    d.mkdir()
    (d / "widget.yaml").write_text(
        'name: gizmo\ntitle: Gizmo sheet\naliases: ["gadget"]\nfields:\n  - name: serial\n    label: Serial\n'
    )
    (d / "broken.yaml").write_text("name: [unclosed\n")
    return d


def test_types_dir_lists_fields_and_aliases(tmp_path):
    d = _user_types(tmp_path)
    r = runner.invoke(app, ["types", "--types-dir", str(d)])
    assert r.exit_code == 0 and "gizmo\tGizmo sheet" in r.stdout and "logbook" in r.stdout
    assert "skipped" in r.stderr and "broken.yaml" in r.stderr and "skipped" not in r.stdout
    r = runner.invoke(app, ["fields", "gadget", "--types-dir", str(d)])
    assert r.exit_code == 0 and r.stdout.startswith("serial\t")


def test_types_dir_ahead_of_environment(tmp_path, monkeypatch):
    env = tmp_path / "env"
    env.mkdir()
    (env / "widget.yaml").write_text("name: gizmo\nfields:\n  - name: from_env\n")
    d = _user_types(tmp_path)
    monkeypatch.setenv("DOCOCR_DOCTYPES_DIR", str(env))
    assert runner.invoke(app, ["fields", "gizmo"]).stdout.startswith("from_env")
    assert runner.invoke(app, ["fields", "gizmo", "--types-dir", str(d)]).stdout.startswith("serial")


def test_extract_user_type_by_name(img, fake, tmp_path):
    d = _user_types(tmp_path)
    r = runner.invoke(app, ["extract", img, "-t", "gadget", "--types-dir", str(d)])
    assert r.exit_code == 0, r.output
    assert json.loads(r.stdout)["doc_type"] == "gizmo"


def test_broken_asked_for_file_is_named(img, fake, tmp_path):
    d = _user_types(tmp_path)
    r = runner.invoke(app, ["fields", "broken", "--types-dir", str(d)])
    assert r.exit_code == 1 and "broken.yaml" in r.output and "Traceback" not in r.output


def test_types_dir_must_exist(tmp_path):
    r = runner.invoke(app, ["types", "--types-dir", str(tmp_path / "nope")])
    assert r.exit_code == 1 and "nope" in r.output


# ---- serve: start-up warming ------------------------------------------------------------------

def _serve(monkeypatch, args, warm):
    import uvicorn

    ran = []
    monkeypatch.setattr(uvicorn, "run", lambda *a, **k: ran.append(1))
    monkeypatch.setattr("dococr.pipeline.warm_up", warm)
    monkeypatch.delenv("DOCOCR_WARMUP", raising=False)
    return CliRunner().invoke(app, ["serve", *args]), ran


def test_serve_warms_before_listening(monkeypatch):
    calls = []
    result, ran = _serve(monkeypatch, [], lambda: calls.append(1))
    assert result.exit_code == 0 and calls == [1] and ran == [1]


def test_serve_no_warm_up(monkeypatch):
    calls = []
    result, ran = _serve(monkeypatch, ["--no-warm-up"], lambda: calls.append(1))
    assert result.exit_code == 0 and calls == [] and ran == [1]
    assert os.environ["DOCOCR_WARMUP"] == "0"  # the app's start-up hook honours it too
    monkeypatch.delenv("DOCOCR_WARMUP")


def test_serve_failed_warm_up_is_a_plain_message(monkeypatch):
    def boom():
        raise ExtractorError("models missing")

    result, ran = _serve(monkeypatch, [], boom)
    assert result.exit_code == 1 and ran == []
    assert "error: cannot start the service: models missing" in result.output


# ---- PDFs and several pages ------------------------------------------------------------------

class PerPage(Extractor):
    """Page one (red) has the registration number, page two (green) the make."""

    name = "fake"

    def extract(self, image, doc_type, fields):
        import pdfs

        k = pdfs.page_index(image)
        answers = {0: {"registration_number": "KDA 1A"}, 1: {"make": "Toyota"}}[k]
        return ExtractorOutput(
            fields=[FieldResult(name=f.name, raw=answers.get(f.name), confidence=0.9 if f.name in answers else None)
                    for f in fields]
        )


@pytest.fixture
def two_pages(tmp_path, monkeypatch):
    import pdfs

    monkeypatch.setattr("dococr.pipeline.get_extractor", lambda: PerPage())
    paths = []
    for i, colour in enumerate((pdfs.RED, pdfs.GREEN)):
        p = tmp_path / f"page{i}.png"
        Image.new("RGB", (60, 40), colour).save(p)
        paths.append(str(p))
    return paths


def test_pdf_is_one_document(tmp_path, monkeypatch):
    import pdfs

    monkeypatch.setattr("dococr.pipeline.get_extractor", lambda: PerPage())
    p = tmp_path / "contract.pdf"
    p.write_bytes(pdfs.image_pdf(2))
    r = runner.invoke(app, ["extract", str(p), "-t", "logbook", "-f", "registration_number,make"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert isinstance(data, dict) and data["pages"] == 2
    assert data["fields"]["registration_number"]["page"] == 1
    assert data["fields"]["make"]["page"] == 2


def test_several_paths_stay_separate_documents(two_pages):
    r = runner.invoke(app, ["extract", *two_pages, "-t", "logbook", "-f", "registration_number,make"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert isinstance(data, list) and [d["pages"] for d in data] == [1, 1]
    assert data[0]["fields"]["make"]["value"] is None and data[1]["fields"]["make"]["value"] == "Toyota"


def test_one_document_option_merges_the_files(two_pages):
    r = runner.invoke(app, ["extract", *two_pages, "--one-document", "-t", "logbook", "-f", "registration_number,make"])
    assert r.exit_code == 0, r.output
    data = json.loads(r.output)
    assert isinstance(data, dict) and data["pages"] == 2
    assert data["fields"]["registration_number"]["value"] == "KDA1A"
    assert data["fields"]["make"]["page"] == 2


def test_table_shows_page_only_when_several(two_pages):
    args = ["-t", "logbook", "-f", "registration_number,make", "--format", "table"]
    r = runner.invoke(app, ["extract", *two_pages, "--one-document", *args])
    assert r.exit_code == 0, r.output
    lines = {ln.split()[0]: ln for ln in r.output.splitlines() if ln.strip().startswith(("registration", "make"))}
    assert " p1 " in lines["registration_number"] and " p2 " in lines["make"]
    r = runner.invoke(app, ["extract", two_pages[0], *args])
    assert " p1" not in r.output


def test_max_pages_option(tmp_path, monkeypatch):
    import pdfs

    monkeypatch.setattr("dococr.pipeline.get_extractor", lambda: PerPage())
    p = tmp_path / "long.pdf"
    p.write_bytes(pdfs.image_pdf(3))
    r = runner.invoke(app, ["extract", str(p), "-t", "logbook", "-f", "make", "--max-pages", "2"])
    assert r.exit_code == 1 and "limit is 2" in r.output and "Traceback" not in r.output
    assert str(p) in r.output


def test_bad_pdf_names_the_file(tmp_path, fake):
    p = tmp_path / "bad.pdf"
    p.write_bytes(b"%PDF-1.4\nnot really")
    r = runner.invoke(app, ["extract", str(p), "-t", "logbook", "-f", "make"])
    assert r.exit_code == 1 and r.output.startswith("error:") and "bad.pdf" in r.output
    assert "Traceback" not in r.output


def test_pdf_text_option_reaches_the_pipeline(monkeypatch, img):
    seen = {}
    real = __import__("dococr.pipeline", fromlist=["extract"]).extract

    def spy(*a, **k):
        seen.update(k)
        return real(*a, extractor=Fake(), **k)

    monkeypatch.setattr("dococr.pipeline.extract", spy)
    r = runner.invoke(app, ["extract", img, "--type", "logbook"])
    assert r.exit_code == 0, r.output
    assert seen["pdf_text"] == "auto"
    r = runner.invoke(app, ["extract", img, "--type", "logbook", "--pdf-text", "never"])
    assert r.exit_code == 0 and seen["pdf_text"] == "never"
    r = runner.invoke(app, ["extract", img, "--type", "logbook", "--pdf-text", "sometimes"])
    assert r.exit_code != 0


def test_table_shows_the_pages_of_a_value_across_a_break(tmp_path, monkeypatch):
    import sys as _sys
    from pathlib import Path as _P

    _sys.path.insert(0, str(_P(__file__).resolve().parent))
    import synth_pdf

    from dococr.extractors.local import LocalExtractor

    doc = synth_pdf.make_pdf_document(1)
    p = tmp_path / "agreement.pdf"
    p.write_bytes(synth_pdf.build(doc, "digital"))
    monkeypatch.setattr("dococr.pipeline.get_extractor", lambda: LocalExtractor())
    r = runner.invoke(app, ["extract", str(p), "--type", str(synth_pdf.doc_type_ref()), "--format", "table"])
    assert r.exit_code == 0, r.output
    assert "p1+2" in r.output and "read from the PDF's text layer" in r.output
    r = runner.invoke(app, ["extract", str(p), "--type", str(synth_pdf.doc_type_ref())])
    data = json.loads(r.output)
    assert [part["page"] for part in data["fields"]["land_location"]["parts"]] == [1, 2]
