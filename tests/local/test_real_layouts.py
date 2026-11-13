"""Layouts first seen on real documents, rebuilt with hand-made OCR lines and invented values.

Each test names the failure it guards against. No text here comes from a real document. Fast:
no OCR model.
"""

from __future__ import annotations

from dococr.models import DocType, FieldSpec
from test_multiline_prose import L, para, raws, run


def doc(*fields: FieldSpec, other_labels=()) -> DocType:
    return DocType(name="t", fields=list(fields), other_labels=list(other_labels))


# ---------------------------------------------------------------------------- a time in a form value

TIME = doc(FieldSpec(name="closing_time", label="Time of Closing"),
           FieldSpec(name="closing_date", label="Date of Closing"))


def test_time_beside_its_label_keeps_the_part_after_the_colon():
    lines = [L("Time of Closing", 40, 100), L("10:30 a.m. Local Time", 400, 100),
             L("Date of Closing", 40, 140), L("June 3, 2011", 400, 140)]
    assert raws(run(lines, TIME)) == {"closing_time": "10:30 a.m. Local Time", "closing_date": "June 3, 2011"}


def test_time_in_one_box_with_its_label_and_colon():
    assert run([L("Time of Closing: 7:45AM CET", 40, 100)], TIME, ["closing_time"])["closing_time"].raw \
        == "7:45AM CET"


def test_colon_after_a_label_still_ends_an_unknown_label_value():
    # "Label1: v1 Label2: v2" with Label2 not in the YAML still stops at the colon
    m = run([L("Time of Closing: noon Place: Hall 2", 40, 100)], TIME, ["closing_time"])["closing_time"]
    assert m.raw == "noon Place"


# ---------------------------------------------------------------------------- a one-letter word before an anchor

def test_one_letter_word_before_an_end_text_stays_in_the_value():
    d = doc(FieldSpec(name="series", label="Series of Shares Purchased", until="Price per Share"))
    lines = para("Series of Shares Purchased Fixed Rate Preference Shares, Series C Price per Share 25")
    assert run(lines, d)["series"].raw == "Fixed Rate Preference Shares, Series C"


def test_one_letter_word_before_a_row_label_stays_in_the_value():
    d = doc(FieldSpec(name="klass", label="Class"), FieldSpec(name="price", label="Price per Share"))
    res = run([L("Class Series C Price per Share 25", 40, 100)], d)
    assert raws(res) == {"klass": "Series C", "price": "25"}


def test_a_split_word_is_still_joined_into_its_label():
    d = doc(FieldSpec(name="price", label="Price per Share"))
    assert run([L("Pri ce per Share 25", 40, 100)], d)["price"].raw == "25"


# ---------------------------------------------------------------------------- a repeated short label

TITLE = doc(FieldSpec(name="title", label="FOR", until="SECTION 1. PARTIES"),
            other_labels=["SECTION 1. PARTIES"])


def test_short_label_repeated_inside_the_value_gives_no_value_rather_than_a_part():
    lines = para("COOPERATION AGREEMENT", "FOR", "ROTOR TESTING FOR LOW NOISE DRONES", "SECTION 1. PARTIES")
    m = run(lines, TITLE)["title"]
    assert m.raw is None
    assert any("occurs more than once" in w and "make the label longer" in w for w in m.warnings)


def test_short_label_once_still_reads_the_value():
    lines = para("COOPERATION AGREEMENT", "FOR", "ROTOR TESTING AT LOW NOISE", "SECTION 1. PARTIES")
    assert run(lines, TITLE)["title"].raw == "ROTOR TESTING AT LOW NOISE"


def test_repeated_label_whose_earlier_reading_crosses_another_anchor_takes_the_nearest():
    d = doc(FieldSpec(name="buyer", label="sold to", until="for the sum"),
            FieldSpec(name="where", label="situated at", until="and"))
    lines = para("goods sold to the depot situated at Elm Road and later sold to ANNA BELL for the sum")
    assert run(lines, d, ["buyer"])["buyer"].raw == "ANNA BELL"


# ---------------------------------------------------------------------------- one label the start of another

PARTNER = doc(
    FieldSpec(name="partner", label="agreement with", until=","),
    FieldSpec(name="address", label="agreement with Acme Rotors Ltd, located at", until="(the Partner)"),
)


def test_label_that_starts_another_fields_label_is_found_with_a_warning():
    lines = para("This is an agreement with Acme Rotors Ltd, located at 12 Mill Lane, Springfield",
                 "(the Partner) and the Agency.")
    res = run(lines, PARTNER)
    assert raws(res) == {"partner": "Acme Rotors Ltd", "address": "12 Mill Lane, Springfield"}
    assert any("only as the start of the longer label" in w for w in res["partner"].warnings)
    assert not res["address"].warnings


def test_label_found_on_its_own_elsewhere_is_used_without_a_warning():
    lines = para("This is an agreement with Acme Rotors Ltd, located at 12 Mill Lane, Springfield",
                 "(the Partner). A second agreement with Bolt Works, signed later.")
    m = run(lines, PARTNER, ["partner"])["partner"]
    assert m.raw == "Bolt Works" and not m.warnings


# ---------------------------------------------------------------------------- a form value wrapped back to the label's margin

def form(*specs: FieldSpec) -> DocType:
    return doc(*specs, other_labels=["Terms of the Sale"])


SALE = form(
    FieldSpec(name="klass", label="Class of Shares Purchased", multiline=True),
    FieldSpec(name="price", label="Price per Share"),
    FieldSpec(name="dates", label="Payment Dates", multiline=True),
    FieldSpec(name="office", label="Registered Office", multiline=True),
)


def rows(*texts: str, x: float = 40, y: float = 100, pitch: float = 34):
    return [L(t, x, y + k * pitch) for k, t in enumerate(texts)]


def test_value_wrapped_after_a_comma_to_the_label_margin_is_joined():
    lines = rows("Class of Shares Purchased: Fixed Rate Preference Shares,", "Series C", "Price per Share: $25")
    res = run(lines, SALE, ["klass", "price"])
    assert raws(res) == {"klass": "Fixed Rate Preference Shares, Series C", "price": "$25"}
    assert not res["klass"].warnings


def test_value_wrapped_after_and_or_into_a_lower_case_line_is_joined():
    lines = rows("Payment Dates: March 1, June 1, September 1 and", "December 1 of every year", "Price per Share: $25")
    assert run(lines, SALE, ["dates"])["dates"].raw == "March 1, June 1, September 1 and December 1 of every year"
    lines = rows("Payment Dates: March 1, June 1, September 1 and December 1", "of every year", "Price per Share: $25")
    assert run(lines, SALE, ["dates"])["dates"].raw == "March 1, June 1, September 1 and December 1 of every year"


def test_wrapped_value_with_label_and_value_in_separate_boxes():
    lines = [L("Class of Shares Purchased:", 40, 100), L("Fixed Rate Preference Shares,", 400, 100),
             L("Series C", 40, 134), L("Price per Share:", 40, 168), L("$25", 400, 168)]
    res = run(lines, SALE, ["klass", "price"])
    assert raws(res) == {"klass": "Fixed Rate Preference Shares, Series C", "price": "$25"}


def test_wrap_does_not_swallow_the_next_row_of_the_form():
    lines = rows("Class of Shares Purchased: Fixed Rate Preference Shares,", "Price per Share: $25")
    res = run(lines, SALE, ["klass", "price"])
    assert raws(res) == {"klass": "Fixed Rate Preference Shares,", "price": "$25"}
    assert any("next line starts another label" in w for w in res["klass"].warnings)


def test_wrap_does_not_take_an_unlisted_label_with_a_colon():
    lines = rows("Class of Shares Purchased: Fixed Rate Preference Shares,", "Contact: Desk 4")
    m = run(lines, SALE, ["klass"])["klass"]
    assert m.raw == "Fixed Rate Preference Shares," and any("may be incomplete" in w for w in m.warnings)


def test_line_at_the_margin_without_evidence_is_not_joined_but_flagged():
    lines = rows("Registered Office: Unit 4 Harbour Plaza", "Port Town 12345", "Price per Share: $25")
    m = run(lines, SALE, ["office"])["office"]
    assert m.raw == "Unit 4 Harbour Plaza"
    assert any("may continue it" in w for w in m.warnings)


def test_value_followed_by_a_heading_and_the_next_label_has_no_warning():
    lines = rows("Registered Office: Unit 4 Harbour Plaza", "Terms of the Sale", "Price per Share: $25")
    m = run(lines, SALE, ["office"])["office"]
    assert m.raw == "Unit 4 Harbour Plaza" and not m.warnings


def test_field_without_multiline_warns_when_its_value_is_visibly_cut():
    d = form(FieldSpec(name="dates", label="Payment Dates"), FieldSpec(name="price", label="Price per Share"))
    lines = rows("Payment Dates: March 1, June 1 and", "December 1", "Price per Share: $25")
    res = run(lines, d)
    assert res["dates"].raw == "March 1, June 1 and"
    assert any("cut short" in w and "multiline" in w for w in res["dates"].warnings)
    assert not res["price"].warnings
    lines = rows("Payment Dates: March 1, June 1 and December 1", "of every year", "Price per Share: $25")
    assert any("cut short" in w for w in run(lines, d)["dates"].warnings)
    lines = rows("Payment Dates: March 1 and June 1", "Price per Share: $25")
    assert not run(lines, d)["dates"].warnings


def test_value_ending_in_a_capital_letter_is_not_taken_for_an_open_word():
    d = form(FieldSpec(name="klass", label="Class of Shares Purchased"), FieldSpec(name="price", label="Price per Share"))
    lines = rows("Class of Shares Purchased: Series A", "Price per Share: $25")
    assert not run(lines, d)["klass"].warnings


def test_form_label_twice_with_different_values_keeps_the_best_but_warns():
    d = doc(FieldSpec(name="ref", label="Ref"))
    lines = [L("Ref", 40, 100), L("AB-1001", 300, 100), L("Ref", 40, 300), L("CD-2002", 300, 300)]
    m = run(lines, d)["ref"]
    assert m.raw in ("AB-1001", "CD-2002") and any("occurs more than once" in w for w in m.warnings)
    lines = [L("Ref", 40, 100), L("AB-1001", 300, 100), L("Ref", 40, 300), L("AB-1001", 300, 300)]
    assert not run(lines, d)["ref"].warnings  # the same value twice is not a conflict


def test_block_below_its_label_stops_at_the_next_label_and_value_in_one_box():
    d = form(FieldSpec(name="where", label="Place of Signing", multiline=True),
             FieldSpec(name="when", label="Time of Signing"))
    lines = [L("Place of Signing:", 40, 100), L("Harbour Hall, Room 2", 300, 134), L("14 Quay Street", 300, 166),
             L("Time of Signing: 10:00 AM local", 40, 198)]
    res = run(lines, d)
    assert raws(res) == {"where": "Harbour Hall, Room 2 14 Quay Street", "when": "10:00 AM local"}


def test_indented_continuation_is_joined_only_with_evidence():
    d = form(FieldSpec(name="company", label="Name of the Firm", multiline=True),
             FieldSpec(name="kind", label="Kind of Firm"))
    lines = [L("Name of the Firm:", 40, 100), L("NORTHWIND TRADERS OF KENT,", 400, 100), L("LTD.", 480, 134),
             L("Kind of Firm:", 40, 168), L("company", 400, 168)]
    res = run(lines, d)
    assert raws(res) == {"company": "NORTHWIND TRADERS OF KENT, LTD.", "kind": "company"}
    lines[1] = L("NORTHWIND TRADERS OF KENT", 400, 100)
    m = run(lines, d)["company"]
    assert m.raw == "NORTHWIND TRADERS OF KENT" and any("indented" in w for w in m.warnings)


def test_ampersand_in_a_value_below_its_label_is_kept():
    d = form(FieldSpec(name="where", label="Place of Signing", multiline=True))
    lines = [L("Place of Signing:", 40, 100), L("Hart, Lowe & Finch LLP", 300, 134), L("2 Quay Street", 300, 166)]
    assert run(lines, d)["where"].raw == "Hart, Lowe & Finch LLP 2 Quay Street"
