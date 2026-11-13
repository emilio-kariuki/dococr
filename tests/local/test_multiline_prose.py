"""`multiline` and `until` on hand-built OCR lines. Fast: no OCR model.

Every label, end text and value format here is test data in a `DocType` built in the test;
none of it is known to the matcher.
"""

from __future__ import annotations

import time

import pytest

from dococr.models import DocType, FieldSpec
from dococr.normalize import normalize_value
from dococr.ocr import OcrLine
from dococr.ocr import matcher
from dococr.ocr.matcher import build_vocabulary, match_fields

CHAR_W = 12
LINE_H = 24


def L(text: str, x: float, y: float, conf: float = 0.98, h: float = LINE_H) -> OcrLine:
    return OcrLine(text, conf, (x, y, x + CHAR_W * len(text), y + h))


def run(lines, doc: DocType, names=None):
    names = names or [f.name for f in doc.fields]
    specs = [doc.field(n) or FieldSpec(name=n) for n in names]
    matches, _ = match_fields(lines, doc, specs, page_size=(1400, 1000))
    return {m.name: m for m in matches}


def raws(res):
    return {k: v.raw for k, v in res.items()}


# ---------------------------------------------------------------------------- the model

def test_fieldspec_defaults_leave_fields_single_line():
    spec = FieldSpec(name="x")
    assert spec.multiline is False and spec.until is None and spec.until_texts() == []
    assert FieldSpec(name="x", until="  next  ").until_texts() == ["next"]
    assert FieldSpec(name="x", until=["a b", "", "c"]).until_texts() == ["a b", "c"]


def test_until_field_labels_are_not_row_labels():
    doc = DocType(name="t", other_labels=["Signature"], fields=[
        FieldSpec(name="name", label="Name"),
        FieldSpec(name="buyer", label="sold to", until="for the sum"),
    ])
    assert sorted(v.text for v in build_vocabulary(doc, doc.fields)) == ["Name", "Signature"]


# ---------------------------------------------------------------------------- multiline: form blocks

FORM = DocType(name="form", fields=[
    FieldSpec(name="member", label="Member No"),
    FieldSpec(name="address", label="Postal Address", multiline=True),
    FieldSpec(name="name", label="Full Name", type="name", multiline=True),
    FieldSpec(name="phone", label="Telephone"),
    FieldSpec(name="county", label="County"),
])
PITCH = 30  # line pitch inside a block; form rows are 42 apart


def test_address_block_right_of_label_ends_at_next_label():
    lines = [
        L("POSTAL ADDRESS", 40, 100), L("P.O. BOX 123-00100", 300, 98),
        L("KIMATHI STREET", 300, 98 + PITCH, conf=0.90), L("NAIROBI", 300, 98 + 2 * PITCH),
        L("TELEPHONE", 40, 100 + 2 * PITCH + 42), L("0722000111", 300, 98 + 2 * PITCH + 42),
    ]
    res = run(lines, FORM, ["address", "phone"])
    assert raws(res) == {"address": "P.O. BOX 123-00100 KIMATHI STREET NAIROBI", "phone": "0722000111"}
    m = res["address"]
    # the box covers all three lines; the confidence is that of the weakest line
    assert m.box == (300, 98, 300 + CHAR_W * len("P.O. BOX 123-00100"), 98 + 2 * PITCH + LINE_H)
    assert m.confidence == pytest.approx(0.90 * 0.9, abs=0.01)  # weakest line x string fit
    assert not m.warnings


def test_without_multiline_only_the_first_line_is_read():
    lines = [L("POSTAL ADDRESS", 40, 100), L("P.O. BOX 123", 300, 98), L("NAIROBI", 300, 98 + PITCH)]
    single = DocType(name="s", fields=[FieldSpec(name="address", label="Postal Address")])
    assert run(lines, single)["address"].raw == "P.O. BOX 123"
    assert run(lines, FORM, ["address"])["address"].raw == "P.O. BOX 123 NAIROBI"


def test_multiline_field_with_one_line_reads_as_before():
    lines = [L("POSTAL ADDRESS", 40, 100), L("P.O. BOX 123", 300, 98),
             L("COUNTY", 40, 142), L("NAKURU", 300, 140)]
    single = DocType(name="s", fields=[FieldSpec(name="address", label="Postal Address"),
                                        FieldSpec(name="county", label="County")])
    a, b = run(lines, FORM, ["address", "county"]), run(lines, single)
    assert raws(a) == raws(b) == {"address": "P.O. BOX 123", "county": "NAKURU"}
    assert a["address"].confidence == b["address"].confidence
    assert a["address"].box == b["address"].box


def test_block_ends_at_an_unlisted_label_in_the_label_column():
    # "OCCUPATION" is not in the YAML. Its value sits in the value column under the address,
    # but the label column is not empty on that row, so it is the next row of the form.
    lines = [L("POSTAL ADDRESS", 40, 100), L("P.O. BOX 123", 300, 98), L("NAIROBI", 300, 98 + PITCH),
             L("OCCUPATION", 40, 100 + PITCH + 42), L("TEACHER", 300, 98 + PITCH + 42)]
    assert run(lines, FORM, ["address"])["address"].raw == "P.O. BOX 123 NAIROBI"


def test_block_ends_at_a_blank_gap():
    lines = [L("POSTAL ADDRESS", 40, 100), L("P.O. BOX 123", 300, 98), L("NAIROBI", 300, 98 + PITCH),
             L("PLEASE PRINT CLEARLY", 300, 98 + PITCH + 70)]  # 46 px of blank space
    assert run(lines, FORM, ["address"])["address"].raw == "P.O. BOX 123 NAIROBI"


def test_block_ignores_smaller_text_and_lines_that_do_not_align():
    small = [L("POSTAL ADDRESS", 40, 100), L("P.O. BOX 123", 300, 98),
             L("(include postal code)", 300, 98 + PITCH, h=14)]
    assert run(small, FORM, ["address"])["address"].raw == "P.O. BOX 123"
    indented = [L("POSTAL ADDRESS", 40, 100), L("P.O. BOX 123", 300, 98), L("NAIROBI", 500, 98 + PITCH)]
    assert run(indented, FORM, ["address"])["address"].raw == "P.O. BOX 123"


def test_runaway_block_is_cut_at_the_cap_with_a_warning():
    lines = [L("POSTAL ADDRESS", 40, 100)] + [L(f"LINE {k}", 300, 98 + k * PITCH) for k in range(8)]
    m = run(lines, FORM, ["address"])["address"]
    assert m.raw == " ".join(f"LINE {k}" for k in range(matcher.MAX_BLOCK_LINES))
    assert any("may continue" in w for w in m.warnings)


def test_block_below_its_label_in_a_grid():
    lines = [
        L("POSTAL ADDRESS", 40, 100, h=16), L("TELEPHONE", 500, 100, h=16),
        L("P.O. BOX 123", 40, 122), L("0722000111", 500, 122),
        L("NAIROBI", 40, 122 + PITCH),
        L("COUNTY", 40, 210, h=16), L("MEMBER NO", 500, 210, h=16),
        L("NAKURU", 40, 232), L("M123456", 500, 232),
    ]
    res = run(lines, FORM, ["address", "phone", "county", "member"])
    assert raws(res) == {"address": "P.O. BOX 123 NAIROBI", "phone": "0722000111",
                         "county": "NAKURU", "member": "M123456"}


def test_block_stays_in_its_column_on_a_two_column_page():
    lines = [
        L("POSTAL ADDRESS", 40, 100), L("P.O. BOX 123", 260, 98),
        L("TELEPHONE", 740, 100), L("0722000111", 960, 98),
        L("NAIROBI", 260, 98 + PITCH), L("MOMBASA ROAD", 960, 98 + PITCH),  # right column: not ours
        L("COUNTY", 40, 100 + PITCH + 42), L("NAKURU", 260, 98 + PITCH + 42),
    ]
    res = run(lines, FORM, ["address", "county"])
    assert raws(res) == {"address": "P.O. BOX 123 NAIROBI", "county": "NAKURU"}


def test_block_after_a_colon_in_the_same_box():
    lines = [L("Full Name: GRACE FAITH", 40, 100), L("WANJIKU NJOROGE", 40 + 11 * CHAR_W, 100 + PITCH),
             L("Telephone: 0722000111", 40, 100 + PITCH + 42)]
    res = run(lines, FORM, ["name", "phone"])
    assert raws(res) == {"name": "GRACE FAITH WANJIKU NJOROGE", "phone": "0722000111"}


def test_block_line_below_is_not_offered_as_a_value_of_its_own():
    # The second address line must not become a competing 'below' candidate for the field.
    lines = [L("POSTAL ADDRESS", 40, 100, conf=0.80), L("P.O. BOX 123", 300, 98, conf=0.80),
             L("NAIROBI", 300, 98 + PITCH, conf=0.99)]
    assert run(lines, FORM, ["address"])["address"].raw == "P.O. BOX 123 NAIROBI"


# ---------------------------------------------------------------------------- until: values in prose

AGREEMENT = DocType(name="agreement", fields=[
    FieldSpec(name="date", label="made on the", until="between", type="date"),
    FieldSpec(name="vendor", label="between", until="of", type="name"),
    FieldSpec(name="vendor_address", label="of", until="(hereinafter called the Vendor)"),
    FieldSpec(name="purchaser", label="the Vendor) and", until="of", type="name"),
    FieldSpec(name="purchaser_address", label="of", until="(hereinafter called the Purchaser)"),
    FieldSpec(name="reg", label="registration number", until="at a price", type="identifier",
              pattern=r"K[A-Z]{2}\s?\d{3}\s?[A-Z]?"),
    FieldSpec(name="price", label="(KES", until="payable", type="integer"),
])


def para(*texts: str, x: float = 40, y: float = 100, pitch: float = 34, conf: float = 0.98):
    return [L(t, x, y + k * pitch, conf=conf) for k, t in enumerate(texts)]


PAGE = para(
    "THIS AGREEMENT is made on the 5th day of March 2024 between JOHN MWANGI",
    "KAMAU of P.O. Box 123 Nairobi (hereinafter called the Vendor) and GRACE",
    "WANJIKU OTIENO of P.O. Box 4455 Mombasa (hereinafter called the Purchaser)",
    "for the sale of the motor vehicle registration number KDA 123A at a price",
    "of Kenya Shillings Eight Hundred Thousand (KES 800,000) payable on signing.",
)


def test_prose_values_between_label_and_end_text():
    res = run(PAGE, AGREEMENT)
    assert raws(res) == {
        "date": "5th day of March 2024", "vendor": "JOHN MWANGI KAMAU",
        "vendor_address": "P.O. Box 123 Nairobi", "purchaser": "GRACE WANJIKU OTIENO",
        "purchaser_address": "P.O. Box 4455 Mombasa", "reg": "KDA 123A", "price": "800,000)",
    }
    assert all(not m.warnings for m in res.values())
    specs = {f.name: f for f in AGREEMENT.fields}
    assert normalize_value(res["date"].raw, specs["date"])[0] == "2024-03-05"
    assert normalize_value(res["price"].raw, specs["price"])[0] == 800000
    assert normalize_value(res["reg"].raw, specs["reg"])[0] == "KDA123A"


def test_value_across_a_line_break_has_a_box_over_both_lines():
    m = run(PAGE, AGREEMENT, ["vendor"])["vendor"]
    assert m.raw == "JOHN MWANGI KAMAU"  # single spaces, reading order
    l, t, _, b = m.box
    assert t == 100 and b == 134 + LINE_H  # both lines
    assert l == 40  # "KAMAU" starts the second line


def test_confidence_is_the_weakest_line():
    lines = list(PAGE)
    lines[1] = L(lines[1].text, *lines[1].box[:2], conf=0.7)
    m = run(lines, AGREEMENT, ["vendor"])["vendor"]
    assert m.confidence == pytest.approx(0.7, abs=0.01)


def test_end_text_and_label_straddle_line_breaks():
    lines = para(
        "The parties agree that the goods are sold to PETER",
        "OTIENO WAFULA for the",
        "sum stated below.",
    )
    doc = DocType(name="t", fields=[FieldSpec(name="buyer", label="sold to", until="for the sum", type="name")])
    assert run(lines, doc)["buyer"].raw == "PETER OTIENO WAFULA"
    lines = para("The goods are sold", "to PETER OTIENO for the sum stated.")
    assert run(lines, doc)["buyer"].raw == "PETER OTIENO"


def test_end_text_of_one_field_is_the_label_of_the_next():
    # "between" ends the date and starts the vendor; "of" ends the vendor and starts the address.
    res = run(PAGE, AGREEMENT, ["date", "vendor", "vendor_address"])
    assert raws(res) == {"date": "5th day of March 2024", "vendor": "JOHN MWANGI KAMAU",
                         "vendor_address": "P.O. Box 123 Nairobi"}


def test_repeated_short_label_takes_the_nearest_before_the_end_text():
    # "of" occurs five times before "(hereinafter called the Vendor)"; the value is after the last.
    res = run(PAGE, AGREEMENT, ["vendor_address"])
    assert res["vendor_address"].raw == "P.O. Box 123 Nairobi"


def test_short_anchor_mid_line_must_match_exactly():
    doc = DocType(name="t", fields=[FieldSpec(name="addr", label="of", until="(the Vendor)")])
    assert run(para("JOHN KAMAU of P.O. Box 1 Nairobi (the Vendor) and"), doc)["addr"].raw == "P.O. Box 1 Nairobi"
    # "or" is not "of" (no edits allowed in a two-letter anchor)
    assert run(para("JOHN KAMAU or P.O. Box 1 Nairobi (the Vendor) and"), doc)["addr"].raw is None


def test_ocr_noise_in_the_end_text_is_tolerated_but_other_wording_is_not():
    noisy = para("KAMAU of P.O. Box 123 Nairobi (hereinafler caIled the Vendor) and")
    assert run(noisy, AGREEMENT, ["vendor_address"])["vendor_address"].raw == "P.O. Box 123 Nairobi"
    worded = para("KAMAU of P.O. Box 123 Nairobi (hereinafter referred to as the Vendor) and")
    m = run(worded, AGREEMENT, ["vendor_address"])["vendor_address"]
    assert m.raw is None and any("not its end text" in w for w in m.warnings)


def test_end_text_alternatives_end_at_the_first_one_found():
    doc = DocType(name="t", fields=[FieldSpec(name="addr", label="of",
                                              until=["(the Seller)", "(hereinafter called the Seller)"])])
    lines = para("JOHN KAMAU of P.O. Box 1 Nairobi (hereinafter called the Seller) and")
    assert run(lines, doc)["addr"].raw == "P.O. Box 1 Nairobi"


def test_same_end_text_twice_with_different_values_is_ambiguous():
    doc = DocType(name="t", fields=[FieldSpec(name="party", label="between", until="(the Party)", type="name")])
    lines = para("made between JOHN KAMAU (the Party) and later",
                 "agreed between PETER OTIENO (the Party) again.")
    m = run(lines, doc)["party"]
    assert m.raw is None and any("more than once" in w for w in m.warnings)


def test_same_value_repeated_is_not_ambiguous():
    doc = DocType(name="t", fields=[FieldSpec(name="party", label="between", until="(the Party)", type="name")])
    lines = para("made between JOHN KAMAU (the Party) and later",
                 "agreed between JOHN KAMAU (the Party) again.")
    m = run(lines, doc)["party"]
    assert m.raw == "JOHN KAMAU" and m.box[1] == 100  # the first occurrence


def test_missing_end_text_gives_no_value_rather_than_the_rest_of_the_page():
    doc = DocType(name="t", fields=[FieldSpec(name="buyer", label="sold to", until="for the sum", type="name")])
    words = " ".join(f"WORD{k}" for k in range(60))
    lines = para("The goods are sold to PETER OTIENO and then " + words[:60], words[60:130], words[130:200])
    m = run(lines, doc)["buyer"]
    assert m.raw is None and any("not its end text" in w for w in m.warnings)


def test_end_text_beyond_the_word_cap_is_not_used():
    doc = DocType(name="t", fields=[FieldSpec(name="buyer", label="sold to", until="for the sum")])
    filler = " ".join(["word"] * (matcher.MAX_FLOW_WORDS + 5))
    lines = para(f"sold to {filler} for the sum")
    assert run(lines, doc)["buyer"].raw is None


def test_end_text_beyond_the_line_cap_is_not_used():
    doc = DocType(name="t", fields=[FieldSpec(name="buyer", label="sold to", until="for the sum")])
    lines = para("sold to A", *["B"] * matcher.MAX_FLOW_LINES, "for the sum")
    assert run(lines, doc)["buyer"].raw is None


def test_date_with_misread_end_text_is_cut_to_the_date_and_flagged():
    lines = para("THIS AGREEMENT is made on the 5th day of March 2024 b3tw33n JOHN KAMAU of Nairobi")
    doc = DocType(name="t", fields=[FieldSpec(name="date", label="made on the", until="between", type="date")])
    m = run(lines, doc)["date"]
    assert m.raw == "5th day of March 2024"
    assert any("not found" in w for w in m.warnings) and m.confidence < 0.8


def test_value_running_into_another_fields_anchor_is_dropped():
    # The vendor's end text "of" was misread as "ot"; read on to the next "of", the vendor
    # would swallow the address and "(hereinafter called the Vendor)", an anchor of another
    # field. Missing is better than that.
    lines = list(PAGE)
    lines[1] = L(lines[1].text.replace("KAMAU of", "KAMAU ot"), *lines[1].box[:2])
    res = run(lines, AGREEMENT, ["vendor", "vendor_address"])
    assert res["vendor"].raw is None
    assert any("run into" in w for w in res["vendor"].warnings)
    # The address's label "of" was also lost; the earlier "of" before it ("5th day of March")
    # would give a reading containing "between", another field's label: dropped as well.
    assert res["vendor_address"].raw is None


def test_other_labels_bound_prose_values():
    doc = DocType(name="t", fields=[FieldSpec(name="addr", label="of", until="the Vendor")])
    lines = para("JOHN KAMAU of P.O. Box 1 Nairobi the Vendor and PETER of the other part.",
                 "WHEREAS the Vendor owns it.")
    m = run(lines, doc)["addr"]
    assert m.raw is None and any("more than once" in w for w in m.warnings)
    bounded = doc.model_copy(update={"other_labels": ["the other part"]})
    assert run(lines, bounded)["addr"].raw == "P.O. Box 1 Nairobi"


def test_flow_does_not_cross_into_the_other_column():
    left = para("The goods are sold to PETER", "OTIENO for the sum agreed.", x=40)
    right = para("Notes for the clerk only", "for the sum of tax due.", x=760)
    doc = DocType(name="t", fields=[FieldSpec(name="buyer", label="sold to", until="for the sum", type="name")])
    assert run(left + right, doc)["buyer"].raw == "PETER OTIENO"


def test_paragraph_gap_beyond_double_spacing_ends_the_flow():
    doc = DocType(name="t", fields=[FieldSpec(name="buyer", label="sold to", until="for the sum", type="name")])
    double = para("The goods are sold to PETER", "OTIENO for the sum agreed.", pitch=58)
    assert run(double, doc)["buyer"].raw == "PETER OTIENO"
    far = para("The goods are sold to PETER", "OTIENO for the sum agreed.", pitch=110)
    assert run(far, doc)["buyer"].raw is None


def test_until_field_in_a_form_row_reads_to_the_right_and_below():
    doc = DocType(name="t", fields=[FieldSpec(name="addr", label="Address", until="Telephone")])
    lines = [L("ADDRESS", 40, 100), L("P.O. BOX 1", 300, 98), L("NAIROBI", 300, 128),
             L("TELEPHONE", 40, 170), L("0722000111", 300, 168)]
    assert run(lines, doc)["addr"].raw == "P.O. BOX 1 NAIROBI"


def test_prose_labels_do_not_cut_form_values():
    # "between" is the label of a prose field; it is not a row label, so a form value that
    # contains the word is read whole, exactly as without the prose field.
    form = [FieldSpec(name="remarks", label="Remarks")]
    lines = [L("REMARKS", 40, 100), L("SPLIT between HEIRS", 300, 98)]
    plain = DocType(name="p", fields=form)
    mixed = DocType(name="m", fields=form + [FieldSpec(name="party", label="between", until="of")])
    assert run(lines, plain, ["remarks"])["remarks"].raw == "SPLIT between HEIRS"
    assert run(lines, mixed, ["remarks"])["remarks"].raw == "SPLIT between HEIRS"


def test_end_text_without_letters_is_reported():
    # A single mark of END_MARKS is an end text (tests/local/test_prose_typeset.py); other
    # punctuation cannot be matched.
    for until in ("!", "--", ")."):
        doc = DocType(name="t", fields=[FieldSpec(name="x", label="sold to", until=until)])
        m = run(para("sold to PETER (x)! -- )."), doc)["x"]
        assert m.raw is None and any("no letters or digits" in w for w in m.warnings), until


def test_label_not_found_and_pattern_fallback_still_apply():
    doc = DocType(name="t", fields=[FieldSpec(name="reg", label="registration number", until="at a price",
                                              type="identifier", pattern=r"K[A-Z]{2}\s?\d{3}\s?[A-Z]?")])
    m = run(para("the motor vehicle with plate KDA 123A at a price"), doc)["reg"]
    assert m.raw == "KDA 123A" and m.confidence <= 0.5
    assert any("format" in w for w in m.warnings)


def test_matcher_stays_fast_on_a_full_page_of_prose():
    sentence = ("The Vendor of P.O. Box 123 Nairobi agrees with the Purchaser of the other part that "
                "the goods are sold to PETER OTIENO for the sum of money stated in this schedule ")
    words = (sentence * 30).split()
    lines, y = [], 100
    while words:
        take, words = words[:13], words[13:]
        lines.append(L(" ".join(take), 40, y))
        y += 34
    assert len(lines) > 40
    started = time.perf_counter()
    for _ in range(3):
        run(lines, AGREEMENT)
    per_page = (time.perf_counter() - started) / 3
    assert per_page < 0.5, per_page  # generous: a shared machine; measured figures in the report


def test_anchor_word_glued_to_the_value_is_split():
    doc = DocType(name="t", fields=[
        FieldSpec(name="chassis", label="chassis number", until="and has agreed", type="identifier"),
        FieldSpec(name="reg", label="registration number", until="chassis number", type="identifier"),
    ])
    lines = para("the vehicle registration number KDA 123Achassis numberJSX956-5326946 and has agreed")
    assert raws(run(lines, doc)) == {"chassis": "JSX956-5326946", "reg": "KDA 123A"}
    # A glued word is split only when the other part has a digit
    lines = para("registration numberplate unknown chassis number X1 and has agreed")
    assert run(lines, doc)["reg"].raw is None


def test_short_label_word_glued_to_digits_is_split():
    doc = DocType(name="t", fields=[FieldSpec(name="price", label="(KES", until="payable", type="integer")])
    assert run(para("the sum of (KES450,000) payable now"), doc)["price"].raw == "450,000)"
    # ... but not when letters follow it
    assert run(para("the sum of (KESHO 450,000) payable now"), doc)["price"].raw is None
