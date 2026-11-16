"""Document type detection through the real OCR model, on synthetic pages."""

from __future__ import annotations

import pytest
from synth import DOC_TYPES, FIXTURE_TYPES, doc_type_ref, make_document

from dococr import (
    UnknownDocumentError,
    detect,
    extract,
    get_extractor,
    list_doc_types,
    load_doc_type,
)
from dococr.extractors.local import LocalExtractor

pytestmark = pytest.mark.slow


def shipped():
    return [load_doc_type(n) for n in list_doc_types() if n not in FIXTURE_TYPES]


def everything():
    return shipped() + [load_doc_type(doc_type_ref(n)) for n in FIXTURE_TYPES]


@pytest.fixture
def reads(monkeypatch):
    count = {"n": 0}
    original = LocalExtractor.read_page

    def counting(self, image, **kw):
        count["n"] += 1
        return original(self, image, **kw)

    monkeypatch.setattr(LocalExtractor, "read_page", counting)
    return count


@pytest.mark.parametrize("doc_type,variant", [
    ("logbook", "clean"), ("national_id", "photo"), ("passport", "rot90"),
    ("driving_license", "skew"), ("synth_form", "value_below"), ("synth_contract", "clean"),
    ("synth_ruled", "clean"),
])
def test_auto_extraction_detects_and_equals_named_extraction(doc_type, variant, reads):
    doc = make_document(doc_type, variant, 1)
    ex = get_extractor()
    auto = extract(doc.image, "auto", doc.fields, candidates=everything(), extractor=ex)
    assert reads["n"] == 1  # the page is read once, for detection and extraction both
    assert auto.doc_type == load_doc_type(doc_type_ref(doc_type)).name
    assert auto.detection is not None and auto.detection.doc_type == auto.doc_type
    named = extract(doc.image, doc_type_ref(doc_type), doc.fields, extractor=ex)
    assert auto.values() == named.values()
    assert {k: f.warnings for k, f in auto.fields.items()} == {k: f.warnings for k, f in named.fields.items()}


def test_a_fixture_page_is_none_of_the_shipped_types():
    doc = make_document("synth_agreement", "clean", 0)
    with pytest.raises(UnknownDocumentError) as err:
        extract(doc.image, candidates=shipped())
    assert err.value.detection.doc_type is None
    assert detect(doc.image, shipped()).doc_type is None


@pytest.mark.parametrize("doc_type", DOC_TYPES)
def test_detect_among_the_shipped_types(doc_type):
    d = detect(make_document(doc_type, "clean", 2).image)
    assert d.doc_type == doc_type, d.reason
