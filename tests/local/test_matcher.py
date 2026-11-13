"""Field matcher on hand-built OCR lines. Fast: no OCR model."""

from __future__ import annotations

import pytest

from dococr import load_doc_type
from dococr.doctypes import resolve_fields
from dococr.models import DocType, FieldSpec
from dococr.normalize import normalize_value
from dococr.ocr import OcrLine
from dococr.ocr.matcher import build_vocabulary, canon, match_fields

CHAR_W = 12
LINE_H = 24


def L(text: str, x: float, y: float, conf: float = 0.98, w: float | None = None, h: float = LINE_H) -> OcrLine:
    return OcrLine(text, conf, (x, y, x + (w if w is not None else CHAR_W * len(text)), y + h))


def run(lines, doc_type, fields):
    doc = load_doc_type(doc_type) if isinstance(doc_type, str) else doc_type
    specs = resolve_fields(doc, fields)
    matches, text = match_fields(lines, doc, specs)
    return {m.name: m for m in matches}, text


def raws(result):
    return {k: v.raw for k, v in result.items()}


# ---------------------------------------------------------------------------- basic layouts

def test_label_left_value_right_rows():
    lines = [
        L("REPUBLIC OF KENYA", 300, 10),
        L("LICENCE NO", 40, 100), L("ABC123456", 300, 98),
        L("SURNAME", 40, 140), L("WAFULA", 300, 138),
        L("DATE OF EXPIRY", 40, 180), L("28 Dec 2018", 300, 178),
    ]
    res, text = run(lines, "driving_license", ["license_number", "surname", "date_of_expiry"])
    assert raws(res) == {"license_number": "ABC123456", "surname": "WAFULA", "date_of_expiry": "28 Dec 2018"}
    assert res["license_number"].confidence > 0.9
    assert res["license_number"].box == (300, 98, 300 + 9 * CHAR_W, 98 + LINE_H)
    assert "LICENCE NO   ABC123456" in text


def test_label_colon_value_in_one_box():
    lines = [L("Licence No: ABC123456", 40, 100), L("Surname:WAFULA", 40, 140), L("Sex : F", 40, 180)]
    res, _ = run(lines, "driving_license", ["license_number", "surname", "sex"])
    assert raws(res) == {"license_number": "ABC123456", "surname": "WAFULA", "sex": "F"}


def test_label_merged_with_value_without_separator():
    lines = [L("DateofBirth12.03.1990", 40, 100), L("SURNAME DOE", 40, 140)]
    res, _ = run(lines, "driving_license", ["date_of_birth", "surname"])
    assert raws(res) == {"date_of_birth": "12.03.1990", "surname": "DOE"}


def test_label_split_across_boxes():
    lines = [L("DATE OF", 40, 100), L("EXPIRY", 40 + 8 * CHAR_W, 100), L("01.02.2030", 300, 100)]
    res, _ = run(lines, "driving_license", ["date_of_expiry"])
    assert res["date_of_expiry"].raw == "01.02.2030"


def test_label_with_ocr_noise():
    lines = [
        L("LlCENCE N0", 40, 100), L("ABC123456", 300, 100),
        L("Sumame", 40, 140), L("DOE", 300, 140),  # 'rn' read as 'm'
        L("Dale of Expiry", 40, 180), L("01.02.2030", 300, 180),
    ]
    res, _ = run(lines, "driving_license", ["license_number", "surname", "date_of_expiry"])
    assert raws(res) == {"license_number": "ABC123456", "surname": "DOE", "date_of_expiry": "01.02.2030"}


def test_value_below_label_grid():
    lines = [
        L("Passport No", 40, 100, h=16), L("Surname", 370, 100, h=16), L("Given Names", 700, 100, h=16),
        L("BK4057958", 40, 125), L("WAFULA", 370, 125), L("PETER SAMUEL", 700, 125),
        L("Date of Birth", 40, 180, h=16), L("Sex", 370, 180, h=16), L("Place of Birth", 700, 180, h=16),
        L("12 OCT 1974", 40, 205), L("F", 370, 205), L("KISUMU", 700, 205),
    ]
    res, _ = run(lines, "passport", ["passport_number", "surname", "given_names", "date_of_birth", "sex", "place_of_birth"])
    assert raws(res) == {
        "passport_number": "BK4057958", "surname": "WAFULA", "given_names": "PETER SAMUEL",
        "date_of_birth": "12 OCT 1974", "sex": "F", "place_of_birth": "KISUMU",
    }


def test_value_row_merged_into_one_box_is_split_by_columns():
    # OCR merged the three values of a row into one box; columns come from the labels above.
    lines = [
        L("Date of Birth", 40, 100, h=16), L("Sex", 300, 100, h=16), L("Nationality", 420, 100, h=16),
        OcrLine("12.10.1974  F  KENYAN", 0.97, (40, 125, 40 + 50 * CHAR_W, 125 + LINE_H)),
    ]
    # proportional word positions: '12.10.1974' 40..160, 'F' 184..196, 'KENYAN' 220..292 (x scale 2.4)
    res, _ = run(lines, "passport", ["date_of_birth", "sex", "nationality"])
    assert res["date_of_birth"].raw == "12.10.1974"


def test_two_pairs_on_one_row_stop_at_next_label():
    lines = [L("SEX", 40, 100), L("M", 200, 100), L("BLOOD GROUP", 400, 100), L("O+", 600, 100)]
    res, _ = run(lines, "driving_license", ["sex", "blood_group"])
    assert raws(res) == {"sex": "M", "blood_group": "O+"}


def test_two_pairs_in_one_box():
    lines = [L("Sex: M Blood Group: O+", 40, 100)]
    res, _ = run(lines, "driving_license", ["sex", "blood_group"])
    assert raws(res) == {"sex": "M", "blood_group": "O+"}


def test_unknown_label_after_colon_stops_value():
    lines = [L("Surname: DOE Foo Bar: 12", 40, 100)]
    res, _ = run(lines, "driving_license", ["surname"])
    # 'Foo Bar' is not a known label; the colon still ends the surname at a word boundary
    assert res["surname"].raw.startswith("DOE")
    assert "12" not in res["surname"].raw


# ---------------------------------------------------------------------------- confusable labels

def test_date_of_issue_vs_place_of_issue():
    lines = [L("PLACE OF ISSUE", 40, 100), L("NAIROBI", 300, 100),
             L("DATE OF ISSUE", 40, 140), L("01.02.2020", 300, 140)]
    res, _ = run(lines, "national_id", ["date_of_issue", "place_of_issue"])
    assert raws(res) == {"date_of_issue": "01.02.2020", "place_of_issue": "NAIROBI"}


def test_place_of_issue_alone_is_not_date_of_issue():
    lines = [L("PLACE OF ISSUE", 40, 100), L("NAIROBI", 300, 100)]
    # driving_license has no place_of_issue field; its other_labels list "Place of Issue",
    # which competes for the words and stops them reading as "Date of Issue".
    res, _ = run(lines, "driving_license", ["date_of_issue"])
    assert res["date_of_issue"].raw is None


def test_id_no_vs_national_id_no():
    doc = DocType(name="custom", fields=[
        FieldSpec(name="id_no", label="ID No"),
        FieldSpec(name="national_id_no", label="National ID No"),
    ])
    lines = [L("NATIONAL ID NO", 40, 100), L("11111111", 300, 100),
             L("ID NO", 40, 140), L("22222222", 300, 140)]
    res, _ = run(lines, doc, ["id_no", "national_id_no"])
    assert raws(res) == {"id_no": "22222222", "national_id_no": "11111111"}
    only_national = [L("NATIONAL ID NO", 40, 100), L("11111111", 300, 100)]
    res, _ = run(only_national, doc, ["id_no"])
    assert res["id_no"].raw is None


def test_registration_no_vs_date_of_registration():
    lines = [L("DATE OF REGISTRATION", 40, 100), L("01.02.2015", 400, 100),
             L("REGISTRATION NO", 40, 140), L("KDA 123A", 400, 140)]
    res, _ = run(lines, "logbook", ["registration_number", "date_of_registration"])
    assert raws(res) == {"registration_number": "KDA 123A", "date_of_registration": "01.02.2015"}
    res, _ = run(lines[:2], "logbook", ["registration_number"])
    assert res["registration_number"].raw is None


def test_title_word_does_not_become_a_label():
    lines = [L("CERTIFICATE OF REGISTRATION", 300, 20), L("KDA 123A", 300, 50)]
    res, _ = run(lines, "logbook", ["registration_number"])
    # No 'Registration No' label: only the format fallback may find the plate, flagged.
    assert res["registration_number"].raw == "KDA 123A"
    assert res["registration_number"].confidence <= 0.5
    assert any("format" in w for w in res["registration_number"].warnings)


@pytest.mark.parametrize("text", ["MALE", "Male", "MAKER", "Mako"])
def test_short_label_make_is_strict(text):
    lines = [L(text, 40, 100), L("TOYOTA", 300, 100)]
    res, _ = run(lines, "logbook", ["make"])
    assert res["make"].raw is None


def test_short_label_must_start_a_box():
    lines = [L("Valid for one Year only", 40, 100), L("2019", 400, 100),
             L("Keep your PIN secret", 40, 140), L("A123456789Z", 400, 140)]
    res, _ = run(lines, "logbook", ["year_of_manufacture", "owner_pin"])
    # The mid-sentence 'Year'/'PIN' are not labels. Only the format fallback finds values,
    # and it says so with a low confidence.
    for name in ("year_of_manufacture", "owner_pin"):
        m = res[name]
        assert m.confidence <= 0.5
        assert any("located by its format" in w for w in m.warnings)


def test_short_labels_exact_still_match_with_lookalikes():
    lines = [L("P1N", 40, 100), L("A123456789Z", 300, 100), L("5EX", 40, 140), L("F", 300, 140)]
    res, _ = run(lines, DocType(name="x", fields=[FieldSpec(name="pin", label="PIN"),
                                                    FieldSpec(name="sex", label="Sex")]), ["pin", "sex"])
    assert raws(res) == {"pin": "A123456789Z", "sex": "F"}


def test_type_does_not_match_tyre():
    lines = [L("TYRE", 40, 100), L("RADIAL", 300, 100)]
    res, _ = run(lines, "logbook", ["body_type"])
    assert res["body_type"].raw is None


def test_body_type_alias_longest_wins_over_type():
    lines = [L("BODY TYPE", 40, 100), L("SALOON", 300, 100), L("TYPE APPROVAL", 40, 140), L("X1", 300, 140)]
    res, _ = run(lines, "logbook", ["body_type"])
    assert res["body_type"].raw == "SALOON"


# ---------------------------------------------------------------------------- ad-hoc fields

def test_adhoc_field_label_from_name():
    lines = [L("COUNTY OF RESIDENCE", 40, 100), L("NAKURU", 400, 100), L("SURNAME", 40, 140), L("DOE", 400, 140)]
    res, _ = run(lines, "national_id", ["county_of_residence"])
    assert res["county_of_residence"].raw == "NAKURU"


def test_adhoc_field_gets_no_word_variants():
    # An ad-hoc field's label is derived from its name and nothing else: 'engine_number' looks
    # for "Engine Number", not "Engine No."; 'color' looks for "Color", not "Colour".
    doc = DocType(name="blank")
    lines = [L("Engine No.", 40, 100), L("1NZ-1234567", 300, 100), L("Colour", 40, 140), L("RED", 300, 140)]
    res, _ = run(lines, doc, ["engine_number", "color"])
    assert raws(res) == {"engine_number": None, "color": None}
    lines = [L("Engine Number", 40, 100), L("1NZ-1234567", 300, 100), L("Color", 40, 140), L("RED", 300, 140)]
    res, _ = run(lines, doc, ["engine_number", "color"])
    assert raws(res) == {"engine_number": "1NZ-1234567", "color": "RED"}


def test_adhoc_field_named_like_a_date_is_a_string():
    # The type comes only from the spec: an ad-hoc 'expiry_date' is a string, so the value is
    # not trimmed to the date inside it. A declared date field is.
    doc = DocType(name="blank")
    lines = [L("EXPIRY DATE 01.02.2030 RENEWABLE", 40, 100)]
    res, _ = run(lines, doc, ["expiry_date"])
    assert res["expiry_date"].raw == "01.02.2030 RENEWABLE"
    dated = DocType(name="dated", fields=[FieldSpec(name="expiry_date", type="date")])
    res, _ = run(lines, dated, ["expiry_date"])
    assert res["expiry_date"].raw == "01.02.2030"


def test_adhoc_and_preset_field_with_same_label_share_value():
    lines = [L("LICENCE NUMBER", 40, 100), L("ABC123456", 300, 100)]
    res, _ = run(lines, "driving_license", ["license_number", "licence_number"])
    assert res["license_number"].raw == res["licence_number"].raw == "ABC123456"


# ---------------------------------------------------------------------------- vocabulary from YAML only

def test_vocabulary_is_labels_aliases_and_other_labels_only():
    doc = DocType(name="x", other_labels=["Signature"], fields=[
        FieldSpec(name="licence", label="Licence No", aliases=["Lic No"]),
    ])
    vocab = build_vocabulary(doc, doc.fields)
    assert sorted(v.text for v in vocab) == ["Lic No", "Licence No", "Signature"]
    assert {v.text: v.owners for v in vocab}["Signature"] == set()
    # no word variants: "Licence Number" is not "Licence No" unless the YAML lists it
    lines = [L("LICENCE NUMBER", 40, 100), L("ABC123", 300, 100)]
    res, _ = run(lines, doc, ["licence"])
    assert res["licence"].raw is None


def test_other_labels_stop_a_value():
    fields = [FieldSpec(name="surname", type="name")]
    lines = [L("SURNAME DOE SIGNATURE JD", 40, 100)]
    with_stop = DocType(name="a", fields=fields, other_labels=["Signature"])
    res, _ = run(lines, with_stop, ["surname"])
    assert res["surname"].raw == "DOE"
    # An other_label produces no value of its own. Asking for it by name makes it an ad-hoc
    # field, which then reads the value beside it like any other field.
    res, _ = run(lines, with_stop, ["surname", "signature"])
    assert raws(res) == {"surname": "DOE", "signature": "JD"}


def test_other_labels_bound_a_value_below():
    doc = DocType(name="a", fields=[FieldSpec(name="surname", type="name")], other_labels=["Signature"])
    lines = [L("Surname", 40, 100, h=16), L("Signature", 400, 100, h=16),
             OcrLine("DOE  JD", 0.97, (40, 125, 40 + 60 * CHAR_W, 125 + LINE_H))]
    res, _ = run(lines, doc, ["surname"])
    assert res["surname"].raw == "DOE"
    # Without it, "Signature" is plain text: it is read as the value to the right of "Surname".
    res, _ = run(lines, doc.model_copy(update={"other_labels": []}), ["surname"])
    assert res["surname"].raw == "Signature"


def test_doc_type_without_other_labels_has_no_builtin_stop_labels():
    # Labels typical of identity documents are not known to the code: without other_labels,
    # "SIGNATURE" is just text and runs on into the value.
    doc = DocType(name="a", fields=[FieldSpec(name="surname", type="name")])
    assert [v.text for v in build_vocabulary(doc, doc.fields)] == ["Surname"]
    res, _ = run([L("SURNAME DOE SIGNATURE JD", 40, 100)], doc, ["surname"])
    assert res["surname"].raw == "DOE SIGNATURE JD"


def test_preset_other_labels_stop_a_value():
    lines = [L("FULL NAMES JOHN DOE HOLDER'S SIGN JD", 40, 100)]
    res, _ = run(lines, "national_id", ["full_name"])
    assert res["full_name"].raw == "JOHN DOE"


# ---------------------------------------------------------------------------- fallbacks / weak matches

def test_pattern_fallback_single_candidate():
    lines = [L("OWNER DETAILS", 40, 100), L("A123456789Z", 300, 140)]
    res, _ = run(lines, "logbook", ["owner_pin"])
    m = res["owner_pin"]
    assert m.raw == "A123456789Z"
    assert m.confidence <= 0.5
    assert any("format" in w for w in m.warnings)


def test_pattern_fallback_ambiguous_returns_none():
    lines = [L("A123456789Z", 300, 100), L("P987654321B", 300, 140)]
    res, _ = run(lines, "logbook", ["owner_pin"])
    assert res["owner_pin"].raw is None
    assert res["owner_pin"].warnings


def test_label_found_without_value():
    lines = [L("DATE OF EXPIRY", 40, 100), L("SURNAME", 40, 140), L("DOE", 300, 140)]
    res, _ = run(lines, "driving_license", ["date_of_expiry"])
    assert res["date_of_expiry"].raw is None
    assert any("no value" in w for w in res["date_of_expiry"].warnings)


def test_weak_match_is_rejected_or_flagged():
    # label read loosely, value is not a date, OCR unsure -> not returned confidently
    lines = [L("Dafe of Expiry", 40, 100, conf=0.7), L("NAIROBI", 300, 100, conf=0.7)]
    res, _ = run(lines, "driving_license", ["date_of_expiry"])
    m = res["date_of_expiry"]
    assert m.raw is None or (m.confidence < 0.6 and m.warnings)


def test_date_prefers_candidate_that_parses():
    # label-right text is not a date but the line below the label is
    lines = [L("DATE OF EXPIRY", 40, 100, h=16), L("(see reverse)", 300, 100), L("01.02.2030", 40, 122)]
    res, _ = run(lines, "driving_license", ["date_of_expiry"])
    assert res["date_of_expiry"].raw == "01.02.2030"


def test_two_fields_cannot_take_the_same_value():
    lines = [L("Surname", 40, 100, h=16), L("Given Names", 400, 100, h=16), L("JANE", 300, 125, w=120)]
    res, _ = run(lines, "passport", ["surname", "given_names"])
    got = [n for n in ("surname", "given_names") if res[n].raw == "JANE"]
    assert len(got) == 1
    other = "given_names" if got == ["surname"] else "surname"
    assert res[other].raw is None


def test_date_trimmed_out_of_longer_text():
    lines = [L("DATE OF BIRTH 12.03.1990 SEXX M", 40, 100)]
    res, _ = run(lines, "driving_license", ["date_of_birth"])
    assert res["date_of_birth"].raw == "12.03.1990"


def test_no_lines():
    res, text = run([], "driving_license", ["surname"])
    assert res["surname"].raw is None and text == ""


def test_results_in_request_order_with_all_fields():
    lines = [L("SURNAME", 40, 100), L("DOE", 300, 100)]
    doc = load_doc_type("driving_license")
    specs = resolve_fields(doc, ["date_of_expiry", "surname", "foo_bar"])
    matches, _ = match_fields(lines, doc, specs)
    assert [m.name for m in matches] == ["date_of_expiry", "surname", "foo_bar"]


def test_canon_folds_lookalikes():
    assert canon("L1cence N0.") == canon("Licence No")
    assert canon("Sumame") == canon("Surname")


def test_end_to_end_normalised_values_from_matcher():
    lines = [L("DATE OF EXPIRY", 40, 100), L("12-MAR-2031", 300, 100)]
    doc = load_doc_type("driving_license")
    (spec,) = resolve_fields(doc, ["date_of_expiry"])
    (m,), _ = match_fields(lines, doc, [spec])
    assert normalize_value(m.raw, spec)[0] == "2031-03-12"


def test_pattern_fallback_warning_is_truthful_when_label_was_found():
    """Label present with nothing beside it, value found elsewhere by its format: the warning
    must not claim the label was missing."""
    from dococr.doctypes import load_doc_type, resolve_fields

    doc = load_doc_type("logbook")
    lines = [
        OcrLine("PIN", 0.99, (50, 100, 110, 130)),
        OcrLine("A123456789Z", 0.98, (600, 400, 820, 430)),
    ]
    (m,), _ = match_fields(lines, doc, resolve_fields(doc, ["owner_pin"]), page_size=(1000, 600))
    assert m.raw == "A123456789Z" and m.confidence <= 0.5
    assert not any("label not found" in w for w in m.warnings)
    assert any("located by its format" in w for w in m.warnings)
