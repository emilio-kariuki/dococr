"""Batch processing with a fake reader (no OCR model): folder walking, outputs, resume,
failures, crashed workers, ordering, exit codes. The slow test with the real model is in
tests/local/test_batch_end_to_end.py."""

from __future__ import annotations

import csv
import io
import json
import os
import sys
from pathlib import Path

import pytest
from PIL import Image
from typer.testing import CliRunner

sys.path.insert(0, str(Path(__file__).resolve().parent))
import batch_fakes as F  # noqa: E402
import pdfs  # noqa: E402

import dococr  # noqa: E402
from dococr import batchio  # noqa: E402
from dococr.batching import document_extensions, find_documents  # noqa: E402
from dococr.cli import app  # noqa: E402

runner = CliRunner()


def png(path: Path, width: int = 300, height: int = 100) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.new("RGB", (width, height), "white").save(path)
    return path


@pytest.fixture
def types_dir(tmp_path):
    folder = tmp_path / "types"
    folder.mkdir()
    (folder / "permit.yaml").write_text(F.PERMIT, encoding="utf-8")
    (folder / "ticket.yaml").write_text(F.TICKET, encoding="utf-8")
    return folder


@pytest.fixture
def reader(monkeypatch):
    """The in-process extractor the CLI gets with --workers 0."""
    monkeypatch.setattr("dococr.pipeline.get_extractor", lambda: F.FakeReader())


def cli(*args):
    return runner.invoke(app, ["batch", *map(str, args)])


def names(paths, root):
    return [str(p.relative_to(root)) for p in paths]


# ------------------------------------------------------------------------ folder walking

def test_folder_walk_rules(tmp_path):
    root = tmp_path / "in"
    for rel in ["b.png", "a.JPG", "c.pdf", "notes.txt", ".hidden.png", "sub/d.png",
                "sub/deep/e.tif", ".secret/f.png", "sub/.h/g.png", "z.png"]:
        png(root / rel) if not rel.endswith((".pdf", ".txt")) else (root / rel).parent.mkdir(
            parents=True, exist_ok=True) or (root / rel).write_bytes(b"x")
    assert names(find_documents([root]), root) == ["a.JPG", "b.png", "c.pdf", "z.png"]
    assert names(find_documents([root], recursive=True), root) == [
        "a.JPG", "b.png", "c.pdf", "sub/d.png", "sub/deep/e.tif", "z.png"]


def test_walk_does_not_follow_symlinked_folders(tmp_path):
    outside = tmp_path / "outside"
    png(outside / "x.png")
    root = tmp_path / "in"
    png(root / "a.png")
    (root / "link").symlink_to(outside, target_is_directory=True)
    (root / "loop").symlink_to(root, target_is_directory=True)  # would never end if followed
    link_file = root / "linked.png"
    link_file.symlink_to(outside / "x.png")
    (root / "broken.png").symlink_to(root / "nothing")
    assert names(find_documents([root], recursive=True), root) == ["a.png", "linked.png"]
    # A symlinked folder that was named is walked: it was asked for.
    assert [p.name for p in find_documents([root / "link"])] == ["x.png"]


def test_named_files_are_taken_as_given_in_order_and_once(tmp_path):
    a, b = png(tmp_path / "b.png"), png(tmp_path / "a.png")
    odd = tmp_path / "scan.dat"
    odd.write_bytes(b"x")
    hidden = png(tmp_path / ".hid.png")
    got = find_documents([b, odd, a, hidden, b, tmp_path / "a.png"])
    assert got == [b, odd, a, hidden]  # named: no extension or hidden filter, no repeats
    assert all(p.is_absolute() for p in got)


def test_missing_input_raises(tmp_path, types_dir):
    with pytest.raises(FileNotFoundError):
        find_documents([tmp_path / "nope"])
    with pytest.raises(FileNotFoundError):
        dococr.batch([tmp_path / "nope"], "permit", dirs=[types_dir])


def test_extensions_follow_the_image_decoders():
    exts = document_extensions()
    assert {".png", ".jpg", ".jpeg", ".tif", ".tiff", ".webp", ".bmp", ".gif", ".pdf"} <= exts
    assert ".txt" not in exts and ".psd" not in exts and ".eps" not in exts


# ------------------------------------------------------------------------ the Python API

def run(sources, doc_type="auto", fields=None, **kw):
    kw.setdefault("workers", 0)
    kw.setdefault("extractor_factory", F.make)
    return list(dococr.batch(sources, doc_type, fields, **kw))


def test_a_bad_file_among_good_ones(tmp_path, types_dir):
    good1 = png(tmp_path / "a.png", height=101)
    bad = tmp_path / "b.png"
    bad.write_text("this is not an image")
    empty = tmp_path / "c.png"
    empty.write_bytes(b"")
    locked = tmp_path / "d.pdf"
    locked.write_bytes(pdfs.raw_pdf([(200, 200)], encrypt=True))
    pdf3 = tmp_path / "e.pdf"
    pdf3.write_bytes(pdfs.image_pdf(3))
    good2 = png(tmp_path / "f.png", height=102)
    gone = tmp_path / "g.png"  # vanishes after being listed: a bad file, not a crash of the batch
    items = run([good1, bad, empty, locked, pdf3, good2], "permit", dirs=[types_dir], max_pages=2)
    assert [i.status for i in items] == ["ok", "bad_input", "bad_input", "bad_input", "bad_input", "ok"]
    assert items[0].result.fields["permit_code"].value == "PX-101"
    assert items[5].result.fields["permit_code"].value == "PX-102"
    assert "encrypted" in items[3].error and "limit" in items[4].error
    assert all(i.error and "\n" not in i.error for i in items[1:5])
    assert all(i.result is None for i in items[1:5])
    assert [i.index for i in items] == list(range(6))
    assert gone.exists() is False


def test_unknown_type_is_a_row_not_an_exception(tmp_path, types_dir):
    items = run([png(tmp_path / "a.png"), png(tmp_path / "b.png", width=F.OTHER)],
                "auto", dirs=[types_dir], candidates=["permit", "ticket"])
    assert [i.status for i in items] == ["ok", "unknown_type"]
    assert items[0].result.doc_type == "permit" and items[0].result.detection.doc_type == "permit"
    assert items[1].detection is not None and items[1].detection.doc_type is None
    assert "could not be detected" in items[1].error


def test_engine_error_is_recorded(tmp_path, types_dir):
    items = run([png(tmp_path / "a.png", width=F.ENGINE), png(tmp_path / "b.png")],
                "permit", dirs=[types_dir])
    assert [i.status for i in items] == ["engine_error", "ok"]
    assert items[0].error == "the fake engine failed"


def test_options_reach_extract(tmp_path, types_dir):
    p = png(tmp_path / "a.png")
    item = run([p], "permit", ["permit_code", "herd_size"], dirs=[types_dir])[0]
    assert list(item.result.fields) == ["permit_code", "herd_size"]
    assert item.result.fields["herd_size"].value == 37
    assert run([p], "permit", dirs=[types_dir], include_raw_text=True)[0].result.raw_text is not None
    assert run([p], "permit", dirs=[types_dir])[0].result.raw_text is None


def test_whole_run_problems_raise_when_called(tmp_path, types_dir):
    p = png(tmp_path / "a.png")
    with pytest.raises(ValueError):
        dococr.batch([p], "permit", dirs=[tmp_path / "no-such-folder"])
    with pytest.raises(ValueError):
        dococr.batch([p], "permit", dirs=[types_dir], pdf_text="sometimes")
    with pytest.raises(ValueError):
        dococr.batch([p], "permit", dirs=[types_dir], workers=-1)
    with pytest.raises(ValueError):
        dococr.batch([p], "permit", dirs=[types_dir], max_pages=0)


def test_no_documents_yields_nothing(tmp_path, types_dir):
    assert run([], "permit", dirs=[types_dir]) == []


# ------------------------------------------------------------------------ workers

def test_workers_keep_input_order_whatever_finishes_first(tmp_path, types_dir):
    # The first documents are slow (the fake waits a second), the later ones are not.
    slow = [png(tmp_path / f"{i:02d}.png", height=SLOW + i) for i, SLOW in zip(range(2), [F.SLEEP_HEIGHT] * 2)]
    fast = [png(tmp_path / f"{i:02d}.png", height=200 + i) for i in range(2, 8)]
    files = slow + fast
    items = list(dococr.batch(files, "permit", dirs=[types_dir], workers=3,
                              extractor_factory=F.make))
    assert [i.source.name for i in items] == [p.name for p in files]
    assert [i.index for i in items] == list(range(8))
    assert all(i.ok for i in items)
    assert [i.result.fields["permit_code"].value for i in items][-1] == "PX-207"


def test_unordered_gives_each_document_once_as_it_finishes(tmp_path, types_dir):
    files = [png(tmp_path / "slow.png", height=F.SLEEP_HEIGHT)] + [
        png(tmp_path / f"f{i}.png", height=200 + i) for i in range(5)]
    items = list(dococr.batch(files, "permit", dirs=[types_dir], workers=2, ordered=False,
                              extractor_factory=F.make))
    assert sorted(i.index for i in items) == list(range(6))
    assert items[0].index != 0 or len(items) == 1  # the slow one did not hold the others back
    assert all(i.ok for i in items)


def test_a_crashed_worker_is_recorded_against_its_document(tmp_path, types_dir):
    files = [png(tmp_path / "a.png", height=101), png(tmp_path / "b.png", width=F.CRASH),
             png(tmp_path / "c.png", height=103), png(tmp_path / "d.png", width=F.CRASH),
             png(tmp_path / "e.png", height=105), png(tmp_path / "f.png", height=106)]
    for workers in (1, 2):
        items = list(dococr.batch(files, "permit", dirs=[types_dir], workers=workers,
                                  extractor_factory=F.make))
        assert [i.status for i in items] == ["ok", "crashed", "ok", "crashed", "ok", "ok"], workers
        assert "exited with code 7" in items[1].error
        assert items[5].result.fields["permit_code"].value == "PX-106"


def test_a_document_that_hangs_is_stopped(tmp_path, types_dir):
    files = [png(tmp_path / "a.png", width=F.HANG), png(tmp_path / "b.png", height=102)]
    items = list(dococr.batch(files, "permit", dirs=[types_dir], workers=1, timeout=2.5,
                              extractor_factory=F.make))
    assert [i.status for i in items] == ["timeout", "ok"]
    assert "2.5 seconds" in items[0].error


def test_an_engine_that_cannot_start_stops_the_batch(tmp_path, types_dir):
    p = png(tmp_path / "a.png")
    with pytest.raises(dococr.ExtractorError, match="cannot start"):
        list(dococr.batch([p], "permit", dirs=[types_dir], workers=1,
                          extractor_factory=F.fail_to_start))
    with pytest.raises(dococr.ExtractorError, match="keep dying"):
        list(dococr.batch([p], "permit", dirs=[types_dir], workers=1,
                          extractor_factory=F.die_at_start))


def test_closing_the_generator_stops_the_workers(tmp_path, types_dir):
    files = [png(tmp_path / f"{i}.png", height=100 + i) for i in range(6)]
    gen = dococr.batch(files, "permit", dirs=[types_dir], workers=2, extractor_factory=F.make)
    next(gen)
    import multiprocessing

    assert multiprocessing.active_children()
    gen.close()
    assert not multiprocessing.active_children()


# ------------------------------------------------------------------------ CSV

def record_for(tmp_path, types_dir, width=300, doc_type="permit", **kw):
    item = run([png(tmp_path / "x.png", width=width, height=111)], doc_type, dirs=[types_dir], **kw)[0]
    return item, batchio.record(item)


def test_csv_header_for_a_named_type_and_for_auto(types_dir):
    from dococr.batching import result_fields

    named = result_fields("permit", None, dirs=[types_dir])
    assert named == ["permit_code", "holder_name", "grazing_zone", "herd_size"]
    header = batchio.csv_header(named)
    assert header[:10] == batchio.FIXED_COLUMNS
    assert header[10:13] == ["permit_code", "permit_code_confidence", "permit_code_warnings"]
    assert len(header) == 10 + 3 * 4
    union = result_fields("auto", None, candidates=["permit", "ticket"], dirs=[types_dir])
    assert union == ["permit_code", "holder_name", "grazing_zone", "herd_size",
                     "ticket_serial", "seat_row", "gate_letter"]  # holder_name once
    assert result_fields("auto", ["holder_name"], candidates=["permit", "ticket"],
                         dirs=[types_dir]) == ["holder_name"]
    assert result_fields("permit", ["herd_size", "extra thing"], dirs=[types_dir]) == [
        "herd_size", "extra_thing"]


def test_a_field_named_like_a_fixed_column_gets_its_own_column():
    header = batchio.csv_header(["status", "file_confidence", "file"])
    assert len(set(header)) == len(header)
    assert "status_field" in header


def test_csv_row_flattening(tmp_path, types_dir):
    item, rec = record_for(tmp_path, types_dir)
    cols = ["permit_code", "herd_size", "nonexistent"]
    row = batchio.csv_row(rec, cols)
    head = dict(zip(batchio.FIXED_COLUMNS, row))
    assert head["status"] == "ok" and head["doc_type"] == "permit" and head["pages"] == 1
    assert head["missing_fields"] == ""
    assert row[10:13] == ["PX-111", pytest.approx(0.95, abs=0.1), ""]
    assert row[13] == 37 and isinstance(row[13], int)  # a number stays a number
    assert row[16:19] == ["", "", ""]


def test_csv_missing_fields_and_failures(tmp_path, types_dir):
    item, rec = record_for(tmp_path, types_dir, width=F.OTHER)
    row = dict(zip(batchio.FIXED_COLUMNS, batchio.csv_row(rec, ["permit_code"])))
    assert row["status"] == "ok" and row["missing_fields"].split(";") == [
        "permit_code", "holder_name", "grazing_zone", "herd_size"]
    bad = tmp_path / "bad.png"
    bad.write_text("x")
    item = run([bad], "permit", dirs=[types_dir])[0]
    row = dict(zip(batchio.FIXED_COLUMNS, batchio.csv_row(batchio.record(item), ["permit_code"])))
    assert row["status"] == "bad_input" and row["error"] and row["doc_type"] == ""
    unknown = run([png(tmp_path / "u.png", width=F.OTHER)], "auto", dirs=[types_dir],
                  candidates=["permit", "ticket"])[0]
    row = dict(zip(batchio.FIXED_COLUMNS, batchio.csv_row(batchio.record(unknown), ["permit_code"])))
    assert row["status"] == "unknown_type" and row["doc_type"] == "" and row["detection_score"] != ""


@pytest.mark.parametrize("text", ["=1+1", "+1", "-1+2", "@SUM(A1)", "\tx", "\rx"])
def test_neutralise_formula_starts(text):
    assert batchio.neutralise(text) == "'" + text


@pytest.mark.parametrize("text", ["", "abc", "1=1", "'=1", " =1", "NORTH 4"])
def test_neutralise_leaves_other_text(text):
    assert batchio.neutralise(text) == text


def test_csv_injection_in_values_errors_and_file_names(tmp_path, types_dir):
    item, rec = record_for(tmp_path, types_dir, width=F.FORMULA)
    assert item.result.fields["grazing_zone"].value == "@SUM(1)"
    row = batchio.csv_row(rec, ["grazing_zone", "herd_size"])
    assert row[10] == "'@SUM(1)"
    # JSON is not altered.
    assert rec["result"]["fields"]["grazing_zone"]["value"] == "@SUM(1)"
    # A negative number is a number, not a formula.
    number = {"file": "/f", "status": "ok", "error": "", "elapsed_ms": 0, "size_bytes": 1,
              "mtime_ns": 1, "result": {"doc_type": "t", "fields": {"n": {"value": -5}}}}
    assert batchio.csv_row(number, ["n"])[10] == -5
    # An error message and a warning are text from outside too.
    rec = {"file": "=cmd.png", "status": "bad_input", "error": "@x", "elapsed_ms": 0,
           "size_bytes": 1, "mtime_ns": 2, "result": {"doc_type": "t", "fields": {
               "a": {"value": "ok", "confidence": 0.5, "warnings": ["-w1", "w2"]}}}}
    row = batchio.csv_row(rec, ["a"])
    assert row[0] == "'=cmd.png" and row[2] == "'@x" and row[12] == "'-w1 | w2"


def test_csv_written_with_the_csv_module_survives_commas_quotes_and_newlines(tmp_path):
    out = tmp_path / "o.csv"
    writer, _ = batchio.open_writer(out, "csv", ["a"])
    rec = {"file": "/x,y.png", "status": "ok", "error": "", "elapsed_ms": 1, "size_bytes": 1,
           "mtime_ns": 1, "result": {"doc_type": "t", "pages": 1, "missing": [], "fields": {
               "a": {"value": 'line one\nline "two", three', "confidence": 1.0, "warnings": []}}}}
    writer.write(rec)
    writer.close()
    rows = list(csv.reader(out.open(newline="", encoding="utf-8")))
    assert rows[1][0] == "/x,y.png" and rows[1][10] == 'line one\nline "two", three'


# ------------------------------------------------------------------------ the command

def make_inputs(tmp_path, n=3, **kw):
    folder = tmp_path / "in"
    for i in range(n):
        png(folder / f"{i}.png", height=100 + i)
    return folder


def read_jsonl(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines()]


def base(types_dir, out, *extra):
    return ["-o", out, "-t", "permit", "--types-dir", types_dir, "--workers", "0", "--quiet", *extra]


def test_jsonl_output_and_exit_zero(tmp_path, types_dir, reader):
    folder, out = make_inputs(tmp_path), tmp_path / "o.jsonl"
    r = cli(folder, *base(types_dir, out))
    assert r.exit_code == 0, r.output
    rows = read_jsonl(out)
    assert [Path(x["file"]).name for x in rows] == ["0.png", "1.png", "2.png"]
    assert rows[0]["status"] == "ok" and rows[0]["result"]["fields"]["permit_code"]["value"] == "PX-100"
    assert rows[0]["size_bytes"] == (folder / "0.png").stat().st_size
    assert "finished: 3 documents" in r.stderr and "3 ok" in r.stderr
    assert r.stdout == ""


def test_csv_output_with_a_failure_gives_exit_three(tmp_path, types_dir, reader):
    folder, out = make_inputs(tmp_path, 2), tmp_path / "o.csv"
    (folder / "z.png").write_text("not an image")
    r = cli(folder, *base(types_dir, out))
    assert r.exit_code == 3, r.output
    rows = list(csv.DictReader(out.open(newline="", encoding="utf-8")))
    assert [x["status"] for x in rows] == ["ok", "ok", "bad_input"]
    assert rows[0]["permit_code"] == "PX-100" and rows[0]["herd_size"] == "37"
    assert rows[2]["error"] and rows[2]["permit_code"] == ""
    assert "1 bad_input" in r.stderr and "failed: " in r.stderr


def test_json_output_is_one_array(tmp_path, types_dir, reader):
    folder, out = make_inputs(tmp_path), tmp_path / "o.json"
    assert cli(folder, *base(types_dir, out)).exit_code == 0
    data = json.loads(out.read_text(encoding="utf-8"))
    assert isinstance(data, list) and len(data) == 3
    assert not [p for p in tmp_path.iterdir() if p.name.endswith(".tmp")]


def test_format_flag_overrides_the_extension_and_unknown_extension_is_refused(tmp_path, types_dir, reader):
    folder = make_inputs(tmp_path, 1)
    out = tmp_path / "o.txt"
    r = cli(folder, *base(types_dir, out))
    assert r.exit_code == 1 and "format" in r.stderr and not out.exists()
    assert cli(folder, *base(types_dir, out, "--format", "jsonl")).exit_code == 0
    assert read_jsonl(out)[0]["status"] == "ok"


def test_formula_in_a_value_is_neutralised_in_csv_only(tmp_path, types_dir, reader):
    folder = tmp_path / "in"
    png(folder / "a.png", width=F.FORMULA)
    csv_out, json_out = tmp_path / "o.csv", tmp_path / "o.jsonl"
    assert cli(folder, *base(types_dir, csv_out)).exit_code == 0
    assert cli(folder, *base(types_dir, json_out)).exit_code == 0
    row = next(csv.DictReader(csv_out.open(newline="", encoding="utf-8")))
    assert row["grazing_zone"] == "'@SUM(1)"
    assert read_jsonl(json_out)[0]["result"]["fields"]["grazing_zone"]["value"] == "@SUM(1)"


def test_auto_detection_csv_has_the_union_of_columns(tmp_path, types_dir, reader):
    folder = tmp_path / "in"
    png(folder / "a.png")
    png(folder / "b.png", width=F.OTHER)
    out = tmp_path / "o.csv"
    r = cli(folder, "-o", out, "--types-dir", types_dir, "--candidates", "permit,ticket",
            "--workers", "0", "--quiet")
    assert r.exit_code == 3
    with out.open(newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    assert "ticket_serial" in rows[0] and "permit_code" in rows[0]
    assert [x["status"] for x in rows] == ["ok", "unknown_type"]
    assert rows[0]["doc_type"] == "permit" and rows[0]["detection_score"]
    assert rows[1]["doc_type"] == "" and "ticket_serial" in rows[1]


def test_fields_option(tmp_path, types_dir, reader):
    folder, out = make_inputs(tmp_path, 1), tmp_path / "o.csv"
    assert cli(folder, *base(types_dir, out, "--fields", "herd_size")).exit_code == 0
    header = out.read_text(encoding="utf-8").splitlines()[0].split(",")
    assert header[10:] == ["herd_size", "herd_size_confidence", "herd_size_warnings"]


def test_recursive_is_opt_in(tmp_path, types_dir, reader):
    folder = make_inputs(tmp_path, 1)
    png(folder / "sub" / "x.png")
    out = tmp_path / "o.jsonl"
    cli(folder, *base(types_dir, out))
    assert len(read_jsonl(out)) == 1
    cli(folder, *base(types_dir, out, "--overwrite", "--recursive"))
    assert len(read_jsonl(out)) == 2


def test_usage_errors_exit_one(tmp_path, types_dir, reader):
    folder = make_inputs(tmp_path, 1)
    out = tmp_path / "o.jsonl"
    assert cli(tmp_path / "missing", *base(types_dir, out)).exit_code == 1
    empty = tmp_path / "empty"
    empty.mkdir()
    r = cli(empty, *base(types_dir, out))
    assert r.exit_code == 1 and "no documents found" in r.stderr
    assert cli(folder, "-o", out, "-t", "nosuchtype", "-f", "", "--types-dir", types_dir,
               "--workers", "0").exit_code in (0, 1)  # an unknown name is an ad-hoc type with no fields
    assert cli(folder, *base(types_dir, tmp_path / "nofolder" / "o.jsonl")).exit_code == 1
    assert cli(folder, *base(types_dir, out, "--types-dir", tmp_path / "nope")).exit_code == 1
    assert cli(folder, *base(types_dir, out, "--overwrite", "--resume")).exit_code == 1
    assert not out.exists()


def test_never_overwrites_silently(tmp_path, types_dir, reader):
    folder = make_inputs(tmp_path, 1)
    for name in ("o.jsonl", "o.csv", "o.json"):
        out = tmp_path / name
        out.write_text("precious", encoding="utf-8")
        r = cli(folder, *base(types_dir, out))
        assert r.exit_code == 1 and "already exists" in r.stderr
        assert out.read_text(encoding="utf-8") == "precious"
        assert cli(folder, *base(types_dir, out, "--overwrite")).exit_code == 0
        assert out.read_text(encoding="utf-8") != "precious"


def test_json_is_written_through_a_temporary_file_and_renamed(tmp_path, types_dir, reader, monkeypatch):
    folder, out = make_inputs(tmp_path, 2), tmp_path / "o.json"
    out.write_text('["old"]', encoding="utf-8")
    seen = {}
    real_replace = os.replace

    def spy(src, dst):
        seen["src"], seen["dst"] = Path(src), Path(dst)
        seen["partial"] = out.read_text(encoding="utf-8")  # the target is untouched until now
        assert seen["src"].parent == out.parent and seen["src"].exists()
        real_replace(src, dst)

    monkeypatch.setattr(os, "replace", spy)
    assert cli(folder, *base(types_dir, out, "--overwrite")).exit_code == 0
    assert seen["dst"] == out and seen["partial"] == '["old"]'
    assert len(json.loads(out.read_text(encoding="utf-8"))) == 2
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []


def test_a_failure_while_writing_json_leaves_the_old_file_and_no_temporary(tmp_path, types_dir, reader, monkeypatch):
    folder, out = make_inputs(tmp_path, 1), tmp_path / "o.json"
    out.write_text('["old"]', encoding="utf-8")

    def broken(*a, **k):
        raise OSError("disk full")

    monkeypatch.setattr(os, "replace", broken)
    r = cli(folder, *base(types_dir, out, "--overwrite"))
    assert r.exit_code == 1 and "disk full" in r.stderr
    assert out.read_text(encoding="utf-8") == '["old"]'
    assert [p.name for p in tmp_path.iterdir() if p.suffix == ".tmp"] == []


def test_json_without_overwrite_refuses_a_file_that_appeared_meanwhile(tmp_path):
    out = tmp_path / "o.json"
    writer, _ = batchio.open_writer(out, "json", [])
    out.write_text("someone else's", encoding="utf-8")
    with pytest.raises(batchio.OutputError, match="appeared"):
        writer.close(True)
    assert out.read_text(encoding="utf-8") == "someone else's"
    assert [p.name for p in tmp_path.iterdir()] == ["o.json"]


def test_an_interrupted_json_run_writes_nothing(tmp_path):
    out = tmp_path / "o.json"
    writer, _ = batchio.open_writer(out, "json", [], overwrite=False)
    writer.write({"file": "x"})
    writer.close(False)
    assert list(tmp_path.iterdir()) == []


# ------------------------------------------------------------------------ resume

def test_resume_skips_recorded_documents_and_redoes_changed_ones(tmp_path, types_dir, reader):
    folder, out = make_inputs(tmp_path, 3), tmp_path / "o.jsonl"
    assert cli(folder, *base(types_dir, out)).exit_code == 0
    assert len(read_jsonl(out)) == 3
    # nothing changed: nothing is read
    r = cli(folder, *base(types_dir, out, "--resume"))
    assert r.exit_code == 0 and len(read_jsonl(out)) == 3 and "3 already recorded" in r.stderr
    # one changed (new size), one new
    png(folder / "1.png", height=150)
    png(folder / "9.png", height=109)
    r = cli(folder, *base(types_dir, out, "--resume"))
    assert r.exit_code == 0, r.output
    rows = read_jsonl(out)
    assert [Path(x["file"]).name for x in rows] == ["0.png", "1.png", "2.png", "1.png", "9.png"]
    assert rows[3]["result"]["fields"]["permit_code"]["value"] == "PX-150"


def test_resume_notices_a_changed_modification_time(tmp_path, types_dir, reader):
    folder, out = make_inputs(tmp_path, 2), tmp_path / "o.csv"
    assert cli(folder, *base(types_dir, out)).exit_code == 0
    p = folder / "0.png"
    st = p.stat()
    os.utime(p, ns=(st.st_atime_ns, st.st_mtime_ns + 5_000_000_000))
    assert cli(folder, *base(types_dir, out, "--resume")).exit_code == 0
    rows = list(csv.DictReader(out.open(newline="", encoding="utf-8")))
    assert [Path(x["file"]).name for x in rows] == ["0.png", "1.png", "0.png"]


@pytest.mark.parametrize("name", ["o.jsonl", "o.csv"])
def test_resume_drops_an_incomplete_last_record(tmp_path, types_dir, reader, name):
    folder, out = make_inputs(tmp_path, 3), tmp_path / name
    assert cli(folder, *base(types_dir, out)).exit_code == 0
    data = out.read_bytes()
    out.write_bytes(data[: len(data) - 40])  # cut inside the last record
    r = cli(folder, *base(types_dir, out, "--resume"))
    assert r.exit_code == 0 and "incomplete" in r.stderr
    if name.endswith("jsonl"):
        assert [Path(x["file"]).name for x in read_jsonl(out)] == ["0.png", "1.png", "2.png"]
    else:
        rows = list(csv.DictReader(out.open(newline="", encoding="utf-8")))
        assert [Path(x["file"]).name for x in rows] == ["0.png", "1.png", "2.png"]


def test_resume_with_no_file_starts_fresh(tmp_path, types_dir, reader):
    folder, out = make_inputs(tmp_path, 2), tmp_path / "o.csv"
    assert cli(folder, *base(types_dir, out, "--resume")).exit_code == 0
    assert len(list(csv.DictReader(out.open(newline="", encoding="utf-8")))) == 2


def test_resume_refuses_what_it_cannot_continue(tmp_path, types_dir, reader):
    folder = make_inputs(tmp_path, 1)
    out = tmp_path / "o.csv"
    assert cli(folder, *base(types_dir, out)).exit_code == 0
    r = cli(folder, *base(types_dir, out, "--resume", "--fields", "herd_size"))
    assert r.exit_code == 1 and "columns differ" in r.stderr
    j = tmp_path / "o.json"
    r = cli(folder, *base(types_dir, j, "--resume"))
    assert r.exit_code == 1 and "cannot be resumed" in r.stderr and not j.exists()
    other = tmp_path / "other.jsonl"
    other.write_text('{"not": "ours"}\n', encoding="utf-8")
    r = cli(folder, *base(types_dir, other, "--resume"))
    assert r.exit_code == 1 and other.read_text(encoding="utf-8") == '{"not": "ours"}\n'


def test_resume_does_not_treat_a_crash_as_special(tmp_path, types_dir, reader):
    """A failed document is recorded, so a resume does not read it again."""
    folder, out = make_inputs(tmp_path, 1), tmp_path / "o.jsonl"
    (folder / "bad.png").write_text("x")
    assert cli(folder, *base(types_dir, out)).exit_code == 3
    r = cli(folder, *base(types_dir, out, "--resume"))
    assert r.exit_code == 0 and len(read_jsonl(out)) == 2


# ------------------------------------------------------------------------ progress

def test_progress_line_has_counts_rate_and_eta_and_no_values(tmp_path, types_dir, reader):
    folder, out = make_inputs(tmp_path, 3), tmp_path / "o.jsonl"
    r = cli(folder, "-o", out, "-t", "permit", "--types-dir", types_dir, "--workers", "0")
    assert r.exit_code == 0
    assert "[3/3] failed 0" in r.stderr and "docs/s" in r.stderr and "ETA" in r.stderr
    for secret in ("PX-100", "ZANELE", "OKAFOR", "NORTH"):
        assert secret not in r.stderr + r.stdout
    assert "--quiet" not in r.stderr


def test_progress_estimates_the_time_left():
    stream = io.StringIO()
    p = batchio.Progress(10, stream, enabled=True)
    p.started -= 10  # ten seconds ago
    item = type("I", (), {"ok": True, "status": "ok", "source": Path("x"), "error": ""})()
    for _ in range(5):
        p.update(item)
    assert "[5/10]" in p.line() and "ETA 10s" in p.line()
    quiet = io.StringIO()
    q = batchio.Progress(2, quiet, enabled=False)
    q.update(item)
    assert quiet.getvalue() == ""


# ------------------------------------------------------------------------ the pool through the command

def test_default_workers_is_modest(monkeypatch):
    from dococr import batching

    monkeypatch.delenv(batching.WORKERS_ENV, raising=False)
    assert 1 <= batching.default_workers() <= 2
    monkeypatch.setenv(batching.WORKERS_ENV, "3")
    assert batching.default_workers() == 3
    monkeypatch.setattr(batching, "_available_mb", lambda: 1500)
    monkeypatch.delenv(batching.WORKERS_ENV)
    assert batching.default_workers() == 1


def test_several_workers_get_two_ocr_threads_each_unless_told(monkeypatch, tmp_path, types_dir):
    from dococr import batching

    seen = []

    def fake_run(jobs, count, opts, factory, threads, timeout, ordered):
        seen.append((count, threads))
        return iter(())

    monkeypatch.setattr(batching, "_run_workers", fake_run)
    p = png(tmp_path / "a.png")
    monkeypatch.delenv("DOCOCR_OCR_THREADS", raising=False)
    for kwargs in ({"workers": 1}, {"workers": 3}, {"workers": 3, "ocr_threads": 1}):
        list(dococr.batch([p], "permit", dirs=[types_dir], **kwargs))
    monkeypatch.setenv("DOCOCR_OCR_THREADS", "3")
    list(dococr.batch([p], "permit", dirs=[types_dir], workers=3))
    assert seen == [(1, None), (1, 2), (1, 1), (1, None)]


def test_interrupt_keeps_what_was_written_and_exits_130(tmp_path, types_dir, monkeypatch):
    class Stops(F.FakeReader):
        calls = 0

        def read_page(self, image, *, text_layer=None):
            Stops.calls += 1
            if Stops.calls == 2:
                raise KeyboardInterrupt
            return super().read_page(image, text_layer=text_layer)

    monkeypatch.setattr("dococr.pipeline.get_extractor", lambda: Stops())
    folder = make_inputs(tmp_path, 3)
    out = tmp_path / "o.jsonl"
    r = cli(folder, *base(types_dir, out))
    assert r.exit_code == 130 and "interrupted: 1 of 3" in r.stderr
    assert len(read_jsonl(out)) == 1
    Stops.calls = 0
    j = tmp_path / "o.json"
    r = cli(folder, *base(types_dir, j))
    assert r.exit_code == 130 and not j.exists() and "nothing was written" in r.stderr
    Stops.calls = 5  # no more interruptions
    r = cli(folder, *base(types_dir, out, "--resume"))  # picks up where it stopped
    assert r.exit_code == 0 and "1 already recorded" in r.stderr
    assert [Path(x["file"]).name for x in read_jsonl(out)] == ["0.png", "1.png", "2.png"]
