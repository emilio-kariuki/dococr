"""Contract tests: the pipeline's behaviour with a fake backend. No OCR model, no network."""

import pytest
from PIL import Image

from dococr import Extractor, ExtractorOutput, FieldResult, extract, list_doc_types, load_doc_type
from dococr.doctypes import resolve_fields


class FakeExtractor(Extractor):
    name = "fake"

    def __init__(self, answers):
        self.answers = answers
        self.seen_fields = None

    def extract(self, image, doc_type, fields):
        self.seen_fields = fields
        return ExtractorOutput(
            fields=[FieldResult(name=f.name, raw=self.answers.get(f.name), confidence=0.9) for f in fields],
            raw_text="page text",
        )


@pytest.fixture
def image():
    return Image.new("RGB", (200, 100), "white")


def test_presets_load():
    assert {"driving_license", "logbook", "national_id", "passport"} <= set(list_doc_types())
    for name in list_doc_types():
        doc = load_doc_type(name)
        assert doc.name == name and doc.fields
        assert len({f.name for f in doc.fields}) == len(doc.fields)


def test_doc_type_synonyms():
    assert load_doc_type("Driving Licence").name == "driving_license"
    assert load_doc_type("log book").name == "logbook"


def test_requested_fields_only(image):
    fake = FakeExtractor({"id_number": " 12345678 ", "full_name": "JANE  DOE"})
    result = extract(image, "national_id", ["ID Number", "full_name", "date_of_birth"], extractor=fake)
    assert [f.name for f in fake.seen_fields] == ["id_number", "full_name", "date_of_birth"]
    assert result.backend == "fake"
    assert result.fields["id_number"].raw == "12345678"
    assert result.values()["full_name"] == "JANE DOE"
    assert result.missing == ["date_of_birth"]
    assert result.fields["date_of_birth"].value is None
    assert result.raw_text is None


def test_all_preset_fields_by_default(image):
    fake = FakeExtractor({})
    result = extract(image, "logbook", extractor=fake, include_raw_text=True)
    assert set(result.fields) == {f.name for f in load_doc_type("logbook").fields}
    assert result.raw_text == "page text"


def test_unknown_doc_type_needs_fields(image):
    fake = FakeExtractor({"invoice_total": "1,200.00"})
    with pytest.raises(ValueError):
        extract(image, "invoice", extractor=fake)
    result = extract(image, "invoice", ["Invoice Total"], extractor=fake)
    assert result.doc_type == "invoice"
    assert result.fields["invoice_total"].raw == "1,200.00"


def test_adhoc_field_gets_a_label():
    doc = load_doc_type("national_id")
    (spec,) = resolve_fields(doc, ["county_of_residence"])
    assert spec.label == "County Of Residence"
    assert spec.labels() == ["County Of Residence"]


# ---- warm_up ----------------------------------------------------------------------------------

def test_warm_up_calls_the_extractors_method(monkeypatch):
    import dococr
    from dococr import pipeline

    class Warm:
        name = "w"
        calls = 0

        def warm_up(self):
            Warm.calls += 1

    engine = Warm()
    monkeypatch.setattr(pipeline, "get_extractor", lambda: engine)
    assert dococr.warm_up() is engine and Warm.calls == 1


def test_warm_up_is_harmless_without_the_method(monkeypatch):
    import dococr
    from dococr import pipeline

    class Plain:
        name = "p"

    engine = Plain()
    monkeypatch.setattr(pipeline, "get_extractor", lambda: engine)
    assert dococr.warm_up() is engine


def test_warm_up_failure_is_an_extractor_error(monkeypatch):
    import dococr
    from dococr import ExtractorError, pipeline

    class Broken:
        name = "b"

        def warm_up(self):
            raise OSError("cannot open model")

    monkeypatch.setattr(pipeline, "get_extractor", lambda: Broken())
    with pytest.raises(ExtractorError, match="cannot open model"):
        dococr.warm_up()


def test_typed_field_that_does_not_parse_keeps_raw_but_has_no_value(image):
    from dococr import DocType

    doc = DocType(name="t", fields=[{"name": "shares", "type": "integer"},
                                    {"name": "closing", "type": "date"},
                                    {"name": "note"}])
    fake = FakeExtractor({"shares": "Not applicable", "closing": "to be agreed", "note": "Not applicable"})
    result = extract(image, doc, extractor=fake)
    shares, closing = result.fields["shares"], result.fields["closing"]
    assert shares.raw == "Not applicable" and shares.value is None and "no number found" in shares.warnings
    assert closing.raw == "to be agreed" and closing.value is None and "could not parse date" in closing.warnings
    assert result.fields["note"].value == "Not applicable"  # a string is never unparsed
    assert result.missing == ["shares", "closing"]  # `missing` is every field without a value


def test_a_handwritten_field_is_always_flagged(image):
    """Handwriting is misread often and confidently, so the confidence cannot be the guard: a
    field the YAML marks as handwritten carries a warning whenever it has a value."""
    from dococr import DocType, FieldSpec
    from dococr.pipeline import HANDWRITTEN_WARNING

    doc = DocType(name="form", fields=[
        FieldSpec(name="applicant", handwritten=True),
        FieldSpec(name="reference"),
        FieldSpec(name="witness", handwritten=True),
    ])
    fake = FakeExtractor({"applicant": "JANE DOE", "reference": "AB123"})
    result = extract(image, doc, extractor=fake)
    assert result.fields["applicant"].value == "JANE DOE"
    assert HANDWRITTEN_WARNING in result.fields["applicant"].warnings
    assert result.fields["applicant"].confidence == 0.9  # high confidence does not remove it
    assert result.fields["reference"].warnings == []
    assert HANDWRITTEN_WARNING not in result.fields["witness"].warnings  # nothing found, nothing to flag
    assert result.missing == ["witness"]
