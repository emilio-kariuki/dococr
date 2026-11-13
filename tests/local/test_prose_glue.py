"""Prose anchors (labels and end texts of `until` fields) that OCR glued to the neighbouring
text, on hand-built OCR lines, and opening brackets as end marks. Fast: no OCR model.

Every label, end text and value here is test data in a `DocType` built in the test; none of it
is known to the matcher.
"""

from __future__ import annotations

from dococr.models import DocType, FieldSpec
from dococr.ocr import matcher
from test_multiline_prose import CHAR_W, para, raws, run


def field(**kw) -> DocType:
    return DocType(name="t", fields=[FieldSpec(name="v", **kw)])


def one(lines, doc):
    return run(lines, doc)["v"]


# ---------------------------------------------------------------------------- the label's last word glued to the value

PARCEL = field(label="known as Title Number", until="situated in", type="identifier")


def test_label_glued_to_a_value_in_capitals_starts_the_value_at_the_join():
    text = "of land known as Title NumberNAKURU/MUNICIPALITY BLOCK 12/345 situated in Nakuru."
    m = one(para(text), PARCEL)
    assert m.raw == "NAKURU/MUNICIPALITY BLOCK 12/345" and not m.warnings
    # the box covers the value's part of the glued word only (estimated from character offsets)
    assert abs(m.box[0] - (40 + CHAR_W * text.index("NAKURU"))) < 1


def test_label_glued_to_a_figure_and_a_lower_case_word_after_it():
    doc = field(label="completion date shall be the", until="day of", type="string")
    assert one(para("The completion date shall be the30th day of April."), doc).raw == "30th"
    doc = field(label="Kenya Shillings Two Million and", until="Thousand")
    assert one(para("Kenya Shillings Two Million andSeventeen Thousand only"), doc).raw == "Seventeen"


def test_label_glued_over_a_line_break_and_words_merged_inside_it():
    lines = para("of land known as TitleNum-", "berNAKURU/BLOCK 7 situated in Nakuru.")
    assert one(lines, PARCEL).raw == "NAKURU/BLOCK 7"
    lines = para("of land known asTitle", "NumberNAKURU/BLOCK 7 situated in Nakuru.")
    assert one(lines, PARCEL).raw == "NAKURU/BLOCK 7"


def test_one_word_label_glued_at_a_change_of_case_is_read_with_a_warning():
    doc = field(label="between", until=",", type="name")
    m = one(para("Two Thousand and Seventeen betweenGRACE OTIENO, of Nairobi"), doc)
    assert m.raw == "GRACE OTIENO"
    assert m.warnings == ["label 'between' was read inside 'betweenGRACE', where a space seems to be "
                          "missing; the value was taken to start at 'GRACE'"]


def test_label_glued_after_punctuation_or_an_opening_bracket():
    doc = field(label="(KES", until=")", type="integer")
    assert one(para("Fifty Thousand(KES 2,450,000) payable"), doc).raw == "2,450,000"
    doc = field(label="P.O. Box", until="Nairobi")
    assert one(para("of P.O.Box4410-00506 Nairobi (the Vendor)"), doc).raw == "4410-00506"


# ---------------------------------------------------------------------------- the label's first word glued to the text before

PRICE = field(label="purchase price is Kenya Shillings", until="(KES", type="integer")


def test_label_glued_to_a_capitalised_word_before_it_is_found():
    m = one(para("Thepurchase price is Kenya Shillings TWO MILLION (KES 2,000,000) only."), PRICE)
    assert m.raw == "TWO MILLION" and not m.warnings


def test_label_glued_on_both_sides():
    doc = field(label="title number", until="situated in", type="identifier")
    assert one(para("land known as aTitle NumberNAKURU/7 situated in Nakuru."), doc).raw == "NAKURU/7"


# ---------------------------------------------------------------------------- end texts

VENDOR = field(label="BETWEEN", until="of Post Office Box", type="name")


def test_end_text_over_a_line_break_with_its_last_word_glued_to_a_figure():
    lines = para("Twenty-Six BETWEEN JOSEPHINE WANJIRU MUTHONI of Post",
                 "Office Box4410-00506 Nairobi (the Vendor) and DANIEL")
    m = one(lines, VENDOR)
    assert m.raw == "JOSEPHINE WANJIRU MUTHONI" and not m.warnings


def test_end_text_with_words_merged_inside_and_glued_at_its_end():
    lines = para("BETWEEN JOSEPHINE MUTHONI ofPost OfficeBox4410 Nairobi")
    assert one(lines, VENDOR).raw == "JOSEPHINE MUTHONI"


def test_end_text_glued_to_the_end_of_the_value():
    m = one(para("BETWEEN JOSEPHINE MUTHONIof Post Office Box 4410 Nairobi"), VENDOR)
    assert m.raw == "JOSEPHINE MUTHONI" and not m.warnings
    assert m.box[2] <= 40 + CHAR_W * len("BETWEEN JOSEPHINE MUTHONI") + 1
    doc = field(label="Kenya Shillings", until="(KES", type="integer")
    assert one(para("Kenya Shillings Fifty Thousand(KES 50,000) only"), doc).raw == "Fifty Thousand"


def test_one_word_end_text_glued_to_the_value_is_read_with_a_warning():
    doc = field(label="made on the", until="between", type="date")
    m = one(para("made on the 25th day of November 2017 between JOHN"), doc)
    assert m.raw == "25th day of November 2017" and not m.warnings  # right-hand glue: no warning
    m = one(para("made on the 25th day of November 2017betweenGRACE OTIENO"), doc)
    assert m.raw == "25th day of November 2017"


def test_end_text_glued_at_both_ends_splits_the_word_between_two_values():
    doc = DocType(name="t", fields=[
        FieldSpec(name="seller", label="sold by", until="and", type="name"),
        FieldSpec(name="buyer", label="and", until="(the Buyer)", type="name"),
    ])
    res = run(para("The car is sold by JOHN KAMAUandGRACE OTIENO (the Buyer) today."), doc)
    assert raws(res) == {"seller": "JOHN KAMAU", "buyer": "GRACE OTIENO"}
    assert res["seller"].warnings == [
        "end text 'and' was read inside 'KAMAUandGRACE', where a space seems to be missing; "
        "the value was taken to end at 'KAMAU'"]
    assert res["buyer"].warnings == [
        "label 'and' was read inside 'KAMAUandGRACE', where a space seems to be missing; "
        "the value was taken to start at 'GRACE'"]


# ---------------------------------------------------------------------------- false anchors refused

def test_short_word_inside_a_longer_word_is_not_an_anchor():
    # no evidence at the join: "of" in "office", "and" in "android", "in" in "invoice"
    doc = field(label="of", until="(the Vendor)", type="name")
    assert one(para("the office JOHN KAMAU (the Vendor) today"), doc).raw is None
    doc = field(label="and", until="(the Buyer)", type="name")
    assert one(para("an android GRACE OTIENO (the Buyer) today"), doc).raw is None
    assert one(para("an Android GRACE OTIENO (the Buyer) today"), doc).raw is None
    doc = field(label="situated in", until=".")
    assert one(para("the land situated invoice Nakuru."), doc).raw is None


def test_compound_or_inflected_word_does_not_end_a_label_without_evidence():
    assert one(para("known as Title Numbering 12 and 13 situated in Nakuru."), PARCEL).raw is None
    doc = field(label="title number", until="situated in", type="identifier")
    assert one(para("the title numberplate 12 situated in Nakuru."), doc).raw is None
    # all capitals on both sides: nothing shows where the label ends
    doc = field(label="between", until=",", type="name")
    assert one(para("made BETWEENJOSEPHINE MUTHONI, of Nairobi"), doc).raw is None


def test_lower_case_end_text_glued_to_a_lower_case_value_is_not_split():
    doc = field(label="sold to", until="of Post Office Box", type="name")
    assert one(para("sold to John Kamauof Post Office Box 12"), doc).raw is None


def test_change_of_case_inside_a_name_is_not_a_short_anchor():
    doc = field(label="mc", until=",", type="name")
    assert one(para("the agent McDONALD KAMAU, of Nairobi"), doc).raw is None


def test_word_internal_marks_are_not_evidence():
    doc = field(label="(the Vendor) and", until=",", type="name")
    assert one(para("JOHN (the Vendor) and/or GRACE OTIENO, of Nairobi"), doc).raw is None


def test_weak_split_needs_a_long_exact_multi_word_label():
    # a capitalised word glued before the label's first word is accepted only when that word is
    # long and the rest of the label matches exactly
    doc = field(label="of the price", until="(KES")
    assert one(para("Thereof the price TWO (KES 2) only"), doc).raw is None
    assert one(para("Thepurchase prise is Kenya Shillings TWO (KES 2) only"), PRICE).raw is None


def test_split_evidence_is_read_at_the_join():
    ev = matcher._join_evidence
    assert ev("NumberNAKURU", 6) == "case"
    assert ev("Box4410", 3) == "digit"
    assert ev("Thousand(KES", 8) == "mark"
    assert ev("Nakuru,Kenya", 7) == "mark"
    assert ev("NAKURUThe", 6) == "case"
    assert ev("MUTHONIof", 7) == "case"
    assert ev("1,200", 2) is None and ev("12.5", 3) is None
    assert ev("and/or", 3) is None and ev("and/or", 4) is None
    assert ev("office", 2) is None and ev("BETWEENJOSEPHINE", 7) is None


def test_run_on_guard_takes_only_splits_with_evidence():
    anchors = matcher.Anchors([matcher._variant("chassis number")])
    words = matcher.tokenize(para("KDA 123A thechassis number"), [])
    assert matcher._contains_anchor(words, anchors) is None
    words = matcher.tokenize(para("KDA 123A theCHASSIS NUMBER"), [])
    assert matcher._contains_anchor(words, anchors) is not None


# ---------------------------------------------------------------------------- opening brackets as end marks

def test_opening_bracket_ends_the_value_before_it():
    doc = field(label="Kenya Shillings", until="(", type="integer")
    assert one(para("Kenya Shillings Two Million Four Hundred Thousand (KES 2,400,000) only"), doc).raw \
        == "Two Million Four Hundred Thousand"
    assert one(para("Kenya Shillings Fifty Thousand(KES 50,000) only"), doc).raw == "Fifty Thousand"
    assert one(para("Kenya Shillings Fifty Thousand ( KES 50,000) only"), doc).raw == "Fifty Thousand"
    doc = field(label="BETWEEN", until="(", type="name")
    assert one(para("BETWEEN GRACE OTIENO of NAIROBI (the Vendor) and"), doc).raw == "GRACE OTIENO of NAIROBI"
    doc = field(label="BETWEEN", until="[", type="name")
    assert one(para("BETWEEN GRACE OTIENO [the Vendor] and"), doc).raw == "GRACE OTIENO"


def test_opening_bracket_inside_a_word_or_at_the_start_of_the_value_does_not_end_it():
    doc = field(label="payable to the", until="(")
    assert one(para("payable to the Purchaser(s) of the land (the Buyers) now"), doc).raw \
        == "Purchaser(s) of the land"
    # the value cannot be empty: a bracket that opens it does not end it, and with ')' as an
    # end mark too, the ')' that closes that bracket does not end it either
    doc = field(label="the sum of", until=["(", ")"])
    assert one(para("the sum of (fifty) shillings (KES 50) only"), doc).raw == "(fifty) shillings"


def test_end_marks_list_opening_brackets():
    assert {"(", "["} <= matcher.END_MARKS
