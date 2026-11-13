"""Typeset prose on hand-built OCR lines: words hyphenated across line breaks, punctuation
marks as end texts, and amounts in words. Fast: no OCR model.

Every label, end text and value here is test data in a `DocType` built in the test; none of it
is known to the matcher.
"""

from __future__ import annotations

import pytest

from dococr.models import DocType, FieldSpec
from dococr.normalize import normalize_value
from dococr.ocr import matcher
from test_multiline_prose import L, para, raws, run


def field(**kw) -> DocType:
    return DocType(name="t", fields=[FieldSpec(name="v", **kw)])


def one(lines, doc):
    return run(lines, doc)["v"]


def value(lines, doc):
    """The normalised value and every warning (matcher and normaliser)."""
    m = one(lines, doc)
    if m.raw is None:
        return None, m.warnings
    v, w = normalize_value(m.raw, doc.fields[0])
    return v, m.warnings + w


# ---------------------------------------------------------------------------- hyphenated line breaks: anchors

OWNER = field(label="registered in the name of", until="(the Owner)", type="name")


def test_label_broken_by_a_hyphen_is_found():
    lines = para("The land is regis-", "tered in the name of JOHN KAMAU (the Owner) today.")
    m = one(lines, OWNER)
    assert m.raw == "JOHN KAMAU" and not m.warnings


def test_label_broken_inside_its_first_word_and_value_on_the_same_line():
    lines = para("The land is registered in the na-", "me of JOHN KAMAU (the Owner) today.")
    assert one(lines, OWNER).raw == "JOHN KAMAU"


def test_end_text_broken_by_a_hyphen_is_found():
    doc = field(label="sold to", until="(hereinafter called the Buyer)", type="name")
    lines = para("The goods are sold to PETER OTIENO (herein-", "after called the Buyer) for cash.")
    assert one(lines, doc).raw == "PETER OTIENO"


def test_anchor_with_a_real_hyphen_broken_at_it_is_found():
    doc = field(label="the self-employed", until="(the Agent)", type="name")
    lines = para("appointed the self-", "employed PETER OTIENO (the Agent) today.")
    assert one(lines, doc).raw == "PETER OTIENO"


def test_broken_anchor_still_has_its_edit_budget_per_word():
    doc = field(label="hereinafter called the Purchaser", until="who accepts", type="name")
    lines = para("Notice: hereinafter cal-", "led the Purchaser PETER OTIENO who accepts.")
    assert one(lines, doc).raw == "PETER OTIENO"


# ---------------------------------------------------------------------------- hyphenated line breaks: values

USE = field(label="used for", until="purposes only")


def test_lower_case_word_broken_at_a_line_end_is_joined_with_a_warning():
    v, w = value(para("The land shall be used for agri-", "cultural purposes only."), USE)
    assert v == "agricultural"
    assert w == ["joined 'agri-' and 'cultural' across a line break as 'agricultural'; "
                 "the hyphen may be part of the word ('agri-cultural')"]


def test_the_page_settles_a_broken_word_without_a_warning():
    lines = para("The land shall be used for agri-", "cultural purposes only. No other",
                 "agricultural use is allowed.")
    assert value(lines, USE) == ("agricultural", [])


def test_the_yaml_settles_a_broken_word_without_a_warning():
    doc = DocType(name="t", other_labels=["agricultural land"], fields=USE.fields)
    assert value(para("The land shall be used for agri-", "cultural purposes only."), doc) == ("agricultural", [])


def test_a_compound_seen_with_its_hyphen_on_the_page_keeps_it():
    lines = para("The Agent, a self-employed man, is used for self-", "employed purposes only.")
    assert value(lines, USE) == ("self-employed", [])


def test_a_compound_broken_at_its_hyphen_without_evidence_is_joined_and_flagged():
    v, w = value(para("The land is used for self-", "employed purposes only."), USE)
    assert v == "selfemployed" and any("may be part of the word" in x for x in w)


def test_capitals_keep_the_hyphen_with_a_warning():
    doc = field(label="sold to", until="(the Buyer)", type="name")
    v, w = value(para("The goods are sold to MARY WANJIKU-", "KAMAU (the Buyer) for cash."), doc)
    assert v == "MARY WANJIKU-KAMAU"
    assert w == ["kept the hyphen in 'WANJIKU-KAMAU' at a line break; it may be one word broken "
                 "across the line ('WANJIKUKAMAU')"]


def test_capitals_settled_by_the_page():
    doc = field(label="sold to", until="(the Buyer)", type="name")
    joined = para("The goods are sold to JOHN NJORO-", "GE (the Buyer) for cash. Signed: JOHN NJOROGE.")
    assert value(joined, doc) == ("JOHN NJOROGE", [])
    kept = para("The goods are sold to MARY WANJIKU-", "KAMAU (the Buyer). Signed: MARY WANJIKU-KAMAU.")
    assert value(kept, doc) == ("MARY WANJIKU-KAMAU", [])


def test_lower_then_capital_is_a_real_hyphen():
    doc = field(label="sold to", until="(the Buyer)", type="name")
    assert value(para("The goods are sold to Mary-", "Anne Kamau (the Buyer)."), doc) == ("Mary-Anne Kamau", [])


@pytest.mark.parametrize("first,second,expected", [
    ("P.O. Box 1021-", "00100 Nairobi", "P.O. Box 1021-00100 Nairobi"),  # digits on both sides
    ("Plot LR-", "4521 Nakuru", "Plot LR-4521 Nakuru"),  # digits after
    ("Unit 12-", "B Nakuru", "Unit 12-B Nakuru"),  # a digit before
])
def test_a_hyphen_next_to_a_digit_is_never_a_line_break_hyphen(first, second, expected):
    doc = field(label="of", until="(the Buyer)")
    v, w = value(para(f"PETER of {first}", f"{second} (the Buyer) today."), doc)
    assert v == expected and not w


def test_soft_hyphen_is_removed_without_a_warning():
    v, w = value(para("The land shall be used for agri­", "cultural purposes only."), USE)
    assert v == "agricultural" and not w


def test_hyphen_inside_a_line_is_kept_as_printed():
    assert value(para("The land is used for self-employed purposes only."), USE) == ("self-employed", [])


def test_value_box_and_confidence_cover_both_pieces():
    lines = para("The land shall be used for agri-", "cultural purposes only.")
    lines[1] = L(lines[1].text, *lines[1].box[:2], conf=0.8)
    m = one(lines, USE)
    assert m.box[1] == 100 and m.box[3] == 134 + 24  # both lines
    assert m.confidence == pytest.approx(0.8 * 0.9, abs=0.01)  # weakest line x string fit


def test_broken_word_cannot_be_claimed_by_two_fields():
    doc = DocType(name="t", fields=[
        FieldSpec(name="a", label="used for", until="purposes only"),
        FieldSpec(name="b", label="shall be used for agri", until="purposes"),
    ])
    res = run(para("The land shall be used for agri-", "cultural purposes only."), doc)
    assert [res["a"].raw, res["b"].raw].count(None) >= 1


def test_multiline_block_joins_a_broken_word():
    doc = DocType(name="f", fields=[FieldSpec(name="name", label="Full Name", type="name", multiline=True),
                                    FieldSpec(name="phone", label="Telephone")])
    lines = [L("FULL NAME", 40, 100), L("GRACE WANJIKU-", 300, 98), L("KAMAU", 300, 128),
             L("TELEPHONE", 40, 170), L("0722000111", 300, 168)]
    m = run(lines, doc, ["name"])["name"]
    assert m.raw == "GRACE WANJIKU-KAMAU" and any("kept the hyphen" in w for w in m.warnings)


def test_fields_without_until_or_multiline_are_not_dehyphenated():
    doc = DocType(name="f", fields=[FieldSpec(name="name", label="Full Name")])
    lines = [L("FULL NAME", 40, 100), L("GRACE WANJIKU-", 300, 98), L("KAMAU", 300, 128)]
    assert run(lines, doc)["name"].raw == "GRACE WANJIKU-"


# ---------------------------------------------------------------------------- punctuation end texts

BUYER = field(label="sold to", until=",", type="name")


def test_value_ends_at_a_comma():
    m = one(para("The vehicle is sold to AMINA HASSAN OMAR, who accepts it as it is."), BUYER)
    assert m.raw == "AMINA HASSAN OMAR" and not m.warnings
    # the box ends before the comma
    assert m.box[2] < 40 + 12 * len("The vehicle is sold to AMINA HASSAN OMAR,")


def test_comma_as_a_separate_token():
    assert one(para("The vehicle is sold to AMINA HASSAN OMAR , who accepts it."), BUYER).raw == "AMINA HASSAN OMAR"


def test_value_ends_at_a_comma_on_the_next_line():
    lines = para("The vehicle is sold to AMINA HASSAN", "OMAR, who accepts it as it is.")
    assert one(lines, BUYER).raw == "AMINA HASSAN OMAR"


def test_initials_are_inside_a_comma_ended_value():
    assert one(para("The vehicle is sold to J. K. KAMAU, who accepts it."), BUYER).raw == "J. K. KAMAU"


def test_a_comma_inside_a_value_cuts_it():
    # Documented limit: a comma end text ends the value at the first comma. A value that
    # contains commas needs word end texts.
    doc = field(label="of", until=",")
    assert one(para("JOHN of P.O. Box 12, Nakuru, who sells."), doc).raw == "P.O. Box 12"


def test_thousands_separators_are_not_ends():
    doc = field(label="the sum of KES", until=",", type="integer")
    assert value(para("for the sum of KES 1,200,000, payable now."), doc) == (1200000, [])
    # OCR put a space after a thousands separator
    assert value(para("for the sum of KES 1, 200,000, payable now."), doc) == (1200000, [])


def test_dropped_comma_gives_no_value_rather_than_the_next_sentence():
    m = one(para("The vehicle is sold to AMINA HASSAN OMAR who accepts it. The Vendor, who"), BUYER)
    assert m.raw is None and any("before the end of the sentence" in w for w in m.warnings)


def test_dropped_comma_at_the_end_of_a_paragraph_gives_no_value():
    lines = para("The vehicle is sold to AMINA HASSAN", "OMAR who accepts it") + \
        para("Second paragraph, with a comma.", y=100 + 2 * 34 + 30)
    assert one(lines, BUYER).raw is None


def test_mixed_list_ends_at_whichever_comes_first():
    doc = field(label="sold to", until=[",", "who accepts"], type="name")
    assert one(para("sold to AMINA HASSAN OMAR who accepts, the vehicle."), doc).raw == "AMINA HASSAN OMAR"
    assert one(para("sold to AMINA HASSAN OMAR, who accepts the vehicle."), doc).raw == "AMINA HASSAN OMAR"


def test_semicolon_and_colon_ends():
    semi = field(label="registration number", until=";", type="identifier")
    assert one(para("registration number KDA 123A; chassis number X"), semi).raw == "KDA 123A"
    colon = field(label="the Vendor", until=":", type="name")
    assert one(para("signed by the Vendor JOHN KAMAU: witnessed"), colon).raw == "JOHN KAMAU"


def test_colon_inside_a_time_is_not_an_end():
    doc = field(label="at", until=":")
    m = one(para("signed at 10:30 hours on site: witnessed"), doc)
    assert m.raw == "10:30 hours on site"  # the colon is kept, not turned into a space


def test_closing_bracket_ends_a_value():
    doc = field(label="(KES", until=")", type="integer")
    assert value(para("the sum of (KES 850,000) payable on signing"), doc) == (850000, [])


def test_bracket_opened_inside_the_value_is_not_its_end():
    doc = field(label="sold to", until=")")
    assert one(para("(the goods sold to ACME (K) LIMITED) and"), doc).raw == "ACME (K) LIMITED"


# full stops

LAND = field(label="situated in", until=".")


def test_value_ends_at_a_full_stop():
    m = one(para("The land is situated in Nakuru County. The Vendor warrants that it is free."), LAND)
    assert m.raw == "Nakuru County" and not m.warnings


def test_full_stop_after_a_listed_abbreviation_before_a_figure_is_not_an_end():
    doc = field(label="situated at", until=".")
    m = one(para("The land situated at Plot No. 4521, Kenyatta Avenue, Nakuru. The Vendor warrants"), doc)
    assert m.raw == "Plot No. 4521, Kenyatta Avenue, Nakuru" and not m.warnings


def test_full_stops_inside_an_abbreviation_are_not_ends():
    doc = field(label="be sent to", until=".")
    m = one(para("Notices shall be sent to P.O. Box 4455-80100, Mombasa. The Vendor warrants"), doc)
    assert m.raw == "P.O. Box 4455-80100, Mombasa" and not m.warnings


def test_ocr_dropped_dot_in_an_abbreviation_is_read_past_with_a_warning():
    doc = field(label="be sent to", until=".")
    m = one(para("Notices shall be sent to PO. Box 4455-80100, Mombasa. The Vendor warrants"), doc)
    assert m.raw == "PO. Box 4455-80100, Mombasa"
    assert m.warnings == ["the value runs on past the full stop after 'PO'; check that it does not end there"]


def test_initials_and_titles_before_a_capital_are_not_ends():
    doc = field(label="in the presence of", until=".", type="name")
    assert one(para("SIGNED in the presence of J. K. KAMAU. The Vendor agrees."), doc).raw == "J. K. KAMAU"
    assert one(para("SIGNED in the presence of Mr. John Kamau. The Vendor agrees."), doc).raw == "Mr. John Kamau"


def test_abbreviation_followed_by_a_new_sentence_ends_the_value_with_a_warning():
    doc = field(label="in the name of", until=".", type="name")
    m = one(para("registered in the name of ACME HOLDINGS LTD. The Vendor warrants that it is free."), doc)
    assert m.raw == "ACME HOLDINGS LTD"
    assert m.warnings == ["the value ends at the full stop after 'LTD', which may be an abbreviation; "
                          "check where it ends"]


def test_abbreviation_followed_by_lower_case_is_not_an_end():
    doc = field(label="in the name of", until=".", type="name")
    m = one(para("registered in the name of ACME HOLDINGS Ltd. of Nairobi. The Vendor"), doc)
    assert m.raw == "ACME HOLDINGS Ltd. of Nairobi"


def test_decimal_and_thousands_points_are_not_ends():
    doc = field(label="interest of", until=".", type="number")
    assert value(para("at an interest of 12.5 per cent. The Vendor agrees."), doc)[0] == 12.5
    doc = field(label="the sum of", until=".", type="integer")
    assert value(para("for the sum of KES 1.200.000. The Vendor agrees."), doc)[0] == 1200000


def test_full_stop_at_the_end_of_a_paragraph():
    doc = field(label="be sent to", until=".")
    lines = para("Notices to the Purchaser shall be sent to P.O. Box", "4455-80100, Mombasa.")
    lines += para("SIGNED by the Vendor.", y=100 + 34 + 60)
    m = one(lines, doc)
    assert m.raw == "P.O. Box 4455-80100, Mombasa" and not m.warnings


def test_missing_full_stop_at_the_end_of_the_text_read():
    doc = field(label="be sent to", until=".")
    m = one(para("Notices to the Purchaser shall be sent to P.O. Box", "4455-80100, Mombasa"), doc)
    assert m.raw == "P.O. Box 4455-80100, Mombasa"
    assert m.warnings == ["end text '.' not found; the value was ended at the end of its paragraph"]


def test_missing_full_stop_before_a_paragraph_gap():
    doc = field(label="be sent to", until=".")
    lines = para("Notices shall be sent to the Purchaser at P.O. Box 4455-80100, Mombasa, Kenya",
                 "or to such other address as the Purchaser names and the Vendor accepts in",
                 "Nairobi") + para("SIGNED by the Vendor and the Purchaser on the day stated.", y=100 + 2 * 34 + 64)
    m = one(lines, doc)
    assert m.raw is not None and m.raw.endswith("Nairobi")
    assert any("end of its paragraph" in w for w in m.warnings)


def test_missing_full_stop_in_mid_paragraph_runs_to_the_next_one():
    # The stop after "County" was lost; the value runs on to the next sentence end. The
    # existing bound and run-on guard are the only protection; here it is bounded by words.
    doc = field(label="situated in", until=".")
    filler = " ".join(["word"] * (matcher.MAX_FLOW_WORDS + 5))
    assert one(para(f"The land is situated in Nakuru County {filler}."), doc).raw is None


def test_full_stop_followed_by_lower_case_is_not_an_end():
    assert one(para("The land is situated in Nakuru approx. two miles out. The Vendor"), LAND).raw == \
        "Nakuru approx. two miles out"


def test_full_stop_end_keeps_the_word_and_line_bounds():
    filler = " ".join(["word"] * (matcher.MAX_FLOW_WORDS + 5))
    assert one(para(f"The land is situated in {filler}."), LAND).raw is None
    lines = para("The land is situated in A", *["B"] * matcher.MAX_FLOW_LINES, "C.")
    assert one(lines, LAND).raw is None


def test_full_stop_on_the_line_after_the_cap_is_not_used():
    # A full stop at the start of the look-ahead line (one past MAX_FLOW_LINES) is not an end
    lines = para("The land is situated in A", *["B"] * (matcher.MAX_FLOW_LINES - 1), ". C")
    assert one(lines, LAND).raw is None


def test_word_end_texts_behave_as_before_with_punctuation_present():
    # A field without punctuation end texts reads through full stops and paragraph ends.
    doc = field(label="sold to", until="who accepts", type="name")
    assert one(para("sold to AMINA. HASSAN OMAR who accepts."), doc).raw == "AMINA. HASSAN OMAR"


def test_invented_comma_cuts_the_value():
    # Documented limit: OCR that invents a mark ends the value there.
    assert one(para("The vehicle is sold to AMINA, HASSAN OMAR, who accepts."), BUYER).raw == "AMINA"


# ---------------------------------------------------------------------------- amounts in words

PRICE = field(label="Kenya Shillings", until="(KES", type="integer")


def test_amount_in_words_over_a_line_break():
    lines = para("for the sum of Kenya Shillings Eight Hundred and Fifty", "Thousand (KES 850,000) payable.")
    m = one(lines, PRICE)
    assert m.raw == "Eight Hundred and Fifty Thousand" and m.confidence > 0.9 and not m.warnings
    assert normalize_value(m.raw, PRICE.fields[0]) == (850000, [])


def test_amount_in_words_with_a_hyphen_broken_across_lines():
    lines = para("for the sum of Kenya Shillings Twenty-", "Five Thousand (KES 25,000) payable.")
    assert value(lines, PRICE) == (25000, [])  # "Twenty-" "Five": kept, then read by the parser


def test_amount_in_words_and_figures_checked():
    doc = field(label="Kenya Shillings", until="payable", type="integer")
    assert value(para("the sum of Kenya Shillings Eight Hundred and Fifty Thousand (KES 850,000) payable"), doc) == \
        (850000, [])
    v, w = value(para("the sum of Kenya Shillings Eight Hundred and Fifty Thousand (KES 830,000) payable"), doc)
    assert v == 850000 and any("disagree" in x for x in w)


def test_amount_ending_at_a_comma_with_words_around_it():
    doc = field(label="a deposit of", until=",", type="integer")
    assert value(para("a deposit of Kenya Shillings Twenty-Five Thousand only, payable on signing"), doc) == \
        (25000, [])


def test_text_without_a_number_is_still_rejected_for_an_integer_field():
    m = one(para("the sum of Kenya Shillings stated below (KES payable"), PRICE)
    assert m.raw is None or normalize_value(m.raw, PRICE.fields[0])[1] == ["no number found"]


def test_form_field_with_a_number_in_words():
    doc = DocType(name="f", fields=[FieldSpec(name="seats", label="Passengers", type="integer")])
    m = run([L("PASSENGERS", 40, 100), L("FIVE", 300, 98)], doc)["seats"]
    assert m.raw == "FIVE" and not m.warnings and normalize_value(m.raw, doc.fields[0])[0] == 5


def test_date_in_words_in_prose():
    doc = field(label="made on the", until="between", type="date")
    lines = para("THIS AGREEMENT is made on the Fifth day of March Two Thousand and", "Twenty-Four between JOHN")
    m = one(lines, doc)
    assert normalize_value(m.raw, doc.fields[0]) == ("2024-03-05", [])


def test_typeset_paragraph_reads_every_field():
    doc = DocType(name="t", fields=[
        FieldSpec(name="date", label="made on the", until="between", type="date"),
        FieldSpec(name="vendor", label="between", until=",", type="name"),
        FieldSpec(name="buyer", label="(hereinafter called the Vendor) and", until=",", type="name"),
        FieldSpec(name="land", label="land situated at", until="."),
        FieldSpec(name="price", label="purchase price is Kenya Shillings", until="(KES", type="integer"),
        FieldSpec(name="figures", label="(KES", until=")", type="integer"),
    ])
    lines = para(
        "THIS AGREEMENT is made on the Fifth day of March Two Thousand and Twen-",
        "ty-Four between JOHN K. KAMAU, of P.O. Box 12 Nakuru (hereinafter cal-",
        "led the Vendor) and AMINA HASSAN OMAR, of Mombasa. The land situ-",
        "ated at Plot No. 4521, Kenyatta Avenue, Nakuru. The purchase price is Ken-",
        "ya Shillings Eight Hundred and Fifty Thousand (KES 850,000) only.",
    )
    res = run(lines, doc)
    specs = {f.name: f for f in doc.fields}
    got = {k: normalize_value(m.raw, specs[k])[0] if m.raw else None for k, m in res.items()}
    assert got == {"date": "2024-03-05", "vendor": "JOHN K. KAMAU", "buyer": "AMINA HASSAN OMAR",
                   "land": "Plot No. 4521, Kenyatta Avenue, Nakuru", "price": 850000, "figures": 850000}, raws(res)


def test_number_words_broken_at_a_line_end():
    # English number words are known words: "THOU-" "SAND" is joined, "TWENTY-" "FIVE" keeps
    # its hyphen, both without a warning.
    doc = field(label="Kenya Shillings", until="(KES", type="integer")
    assert value(para("the sum of Kenya Shillings EIGHT HUNDRED THOU-", "SAND (KES 800,000) payable."), doc) == \
        (800000, [])
    m = one(para("the sum of Kenya Shillings TWENTY-", "FIVE THOUSAND (KES 25,000) payable."), doc)
    assert m.raw == "TWENTY-FIVE THOUSAND" and not m.warnings


def test_matcher_stays_fast_with_punctuation_end_texts():
    import time
    doc = DocType(name="t", fields=[
        FieldSpec(name="a", label="of", until=","), FieldSpec(name="b", label="the", until="."),
        FieldSpec(name="c", label="sold to", until=[",", "for the sum"], type="name"),
        FieldSpec(name="d", label="Kenya Shillings", until=")", type="integer"),
    ])
    sentence = ("The Vendor of P.O. Box 123, Nairobi agrees with the Purchaser of the other part that the "
                "goods are sold to PETER OTIENO, for the sum of Kenya Shillings Five Thousand (KES 5,000). ")
    words = (sentence * 30).split()
    lines, y = [], 100
    while words:
        take, words = words[:13], words[13:]
        lines.append(L(" ".join(take[:-1] + [take[-1][:-1] + "-"] if len(take[-1]) > 6 else take), 40, y))
        y += 34
    started = time.perf_counter()
    for _ in range(3):
        run(lines, doc)
    per_page = (time.perf_counter() - started) / 3
    assert per_page < 1.0, per_page  # generous: a shared machine


# ---------------------------------------------------------------------------- OCR lost the space after a mark

def test_full_stop_inside_a_token_where_ocr_lost_the_space():
    doc = field(label="in the name of", until=".", type="name")
    lines = para("registered in the name of GRACE OTIENO.TheVendor warrants that the land is free.")
    assert one(lines, doc).raw == "GRACE OTIENO"
    # ... also after an initial, which is not an end
    lines = para("registered in the name of PETER N. ACHIENG.TheVendor warrants that the land is free.")
    assert one(lines, doc).raw == "PETER N. ACHIENG"


def test_comma_inside_a_token_where_ocr_lost_the_space():
    doc = field(label="sold to", until=",", type="name")
    assert one(para("sold to AMINA OMAR,who accepts it."), doc).raw == "AMINA OMAR"


def test_marks_between_figures_inside_a_token_are_not_ends():
    doc = field(label="the sum of", until=[",", "."], type="number")
    assert value(para("for the sum of 1,200.50 payable now."), doc)[0] == 1200.5


def test_abbreviation_look_ahead_stops_at_a_mark_inside_a_token():
    doc = field(label="land situated at", until=".")
    m = one(para("the land situated at LR. No.600/6699,Machakos.The said land is registered."), doc)
    assert m.raw == "LR. No.600/6699,Machakos"


def test_comma_misread_as_a_full_stop_gives_no_value():
    # OCR read the comma after the value as a full stop; the sentence does not end there
    # (lower case follows), so the reading runs on to the real end of the sentence and stops.
    doc = field(label="by profession a", until=",")
    m = one(para("The Purchaser, who is by profession a businessman. shall use the land. The Vendor, who"), doc)
    assert m.raw is None
