"""`dococr batch` through the real OCR model with two worker processes, on synthetic pages."""

from __future__ import annotations

import json
from pathlib import Path

import pytest
from synth import make_document
from typer.testing import CliRunner

from dococr import extract, get_extractor
from dococr.cli import app

pytestmark = pytest.mark.slow

PAGES = [("logbook", "photo", 1), ("national_id", "clean", 2), ("passport", "photo", 3),
         ("driving_license", "clean", 4), ("logbook", "clean", 5)]


def test_batch_with_two_workers_equals_one_by_one(tmp_path):
    folder = tmp_path / "in"
    folder.mkdir()
    files = []
    for n, (doc_type, variant, seed) in enumerate(PAGES):
        path = folder / f"{n}_{doc_type}.jpg"
        make_document(doc_type, variant, seed).image.save(path)
        files.append(path)
    (folder / "9_broken.jpg").write_text("not an image")
    out = tmp_path / "out.jsonl"

    r = CliRunner().invoke(app, ["batch", str(folder), "-o", str(out), "--workers", "2", "--quiet"])
    assert r.exit_code == 3, r.output  # one document failed, the batch finished
    rows = [json.loads(line) for line in out.read_text(encoding="utf-8").splitlines()]
    assert [Path(x["file"]).name for x in rows] == [p.name for p in files] + ["9_broken.jpg"]
    assert [x["status"] for x in rows] == ["ok"] * len(files) + ["bad_input"]

    engine = get_extractor()
    for path, row in zip(files, rows):
        alone = extract(path, "auto", extractor=engine)
        assert row["result"]["doc_type"] == alone.doc_type == path.stem.split("_", 1)[1]
        assert {k: f["value"] for k, f in row["result"]["fields"].items()} == alone.values()
