"""normalize_value: typed values, never raises, idempotent."""

from __future__ import annotations

import pytest

from dococr.models import FieldSpec
from dococr.normalize import normalize_value

DATE = FieldSpec(name="d", type="date")
IDENT = FieldSpec(name="i", type="identifier")
INT = FieldSpec(name="n", type="integer")
NUM = FieldSpec(name="x", type="number")
NAME = FieldSpec(name="nm", type="name")
TEXT = FieldSpec(name="s")


@pytest.mark.parametrize("raw,expected", [
    ("12.03.1990", "1990-03-12"),
    ("12/03/1990", "1990-03-12"),
    ("12-03-1990", "1990-03-12"),
    ("12-MAR-1990", "1990-03-12"),
    ("12 Mar 2031", "2031-03-12"),
    ("12 March 2031", "2031-03-12"),
    ("1990-03-12", "1990-03-12"),
    ("1990/03/12", "1990-03-12"),
    ("1990-3-5", "1990-03-05"),
    ("5.3.1990", "1990-03-05"),
    ("12MAR1990", "1990-03-12"),
    ("12 SEPT 2001", "2001-09-12"),
    ("March 12, 1990", "1990-03-12"),
    ("12 MARS 1990", "1990-03-12"),  # bilingual cards
    ("12-0CT-1974", "1974-10-12"),  # OCR read O as 0
    ("25-N0V-1982", "1982-11-25"),
    ("l2.O3.199O", "1990-03-12"),  # OCR look-alikes in digits
    (": 12.03.1990.", "1990-03-12"),
    ("03/04/1990", "1990-04-03"),  # ambiguous -> day first
])
def test_dates(raw, expected):
    value, _ = normalize_value(raw, DATE)
    assert value == expected


@pytest.mark.parametrize("raw,expected", [
    # dates as running text writes them
    ("5th day of March 2024", "2024-03-05"),
    ("5th day of March, 2024", "2024-03-05"),
    ("21st day of the month of June 2019", "2019-06-21"),
    ("22nd February 2019", "2019-02-22"),
    ("3rd of May 2021", "2021-05-03"),
    ("5 th day of March 2024", "2024-03-05"),  # OCR split the ordinal
    ("5TH DAY OF MARCH 2024", "2024-03-05"),
    ("March 5th, 2024", "2024-03-05"),
    ("1er mars 2024", "2024-03-01"),
    ("12 day of Mar 1990", "1990-03-12"),
])
def test_dates_in_running_text(raw, expected):
    value, warnings = normalize_value(raw, DATE)
    assert value == expected and not warnings


def test_date_in_running_text_with_surrounding_words():
    value, warnings = normalize_value("this 5th day of March 2024 between", DATE)
    assert value == "2024-03-05" and any("extra text" in w for w in warnings)


@pytest.mark.parametrize("raw", ["31st day of February 2020", "5th day of Smarch 2024", "the 5th day"])
def test_impossible_running_text_dates_are_not_guessed(raw):
    value, warnings = normalize_value(raw, DATE)
    assert value is None and any("could not parse" in w for w in warnings)


def test_date_month_first_only_when_day_first_impossible():
    value, warnings = normalize_value("03/25/1990", DATE)
    assert value == "1990-03-25" and warnings
    value, warnings = normalize_value("12.13.1990", DATE)
    assert value == "1990-12-13" and warnings


def test_date_two_digit_year_warns():
    value, warnings = normalize_value("12.03.90", DATE)
    assert value == "1990-03-12" and any("two-digit" in w for w in warnings)


@pytest.mark.parametrize("raw", ["31.02.2000", "garbage", "32.13.1990", "00.00.0000"])
def test_bad_dates_have_no_value_and_warn(raw):
    # A typed field has no value when its text does not parse; the text stays in `raw`.
    value, warnings = normalize_value(raw, DATE)
    assert value is None and warnings


def test_date_with_extra_text():
    value, warnings = normalize_value("DOB 12.03.1990 M", DATE)
    assert value == "1990-03-12" and any("extra text" in w for w in warnings)


@pytest.mark.parametrize("raw,expected", [
    ("kda 123a", "KDA123A"),
    (" DL-12 345 ", "DL-12345"),
    (": A012345678Z", "A012345678Z"),
])
def test_identifiers(raw, expected):
    assert normalize_value(raw, IDENT)[0] == expected


def test_identifier_pattern_repairs_lookalikes():
    spec = FieldSpec(name="pin", type="identifier", pattern=r"[AP]\d{9}[A-Z]")
    value, warnings = normalize_value("A0I2345678Z", spec)
    assert value == "A012345678Z" and any("corrected" in w for w in warnings)
    value, warnings = normalize_value("A01234567", spec)  # too short: warn, keep
    assert value == "A01234567" and any("expected format" in w for w in warnings)


def test_string_pattern_repairs_lookalikes():
    spec = FieldSpec(name="blood", pattern=r"(A|B|AB|O)\s?[+-]?")
    assert normalize_value("0+", spec)[0] == "O+"


def test_pattern_mismatch_is_a_warning_not_a_failure():
    spec = FieldSpec(name="plate", type="identifier", pattern=r"K[A-Z]{2}\d{3}[A-Z]")
    value, warnings = normalize_value("XYZ", spec)
    assert value == "XYZ" and warnings


def test_invalid_pattern_does_not_raise():
    spec = FieldSpec(name="s", pattern="([unclosed")
    value, warnings = normalize_value("abc", spec)
    assert value == "abc" and warnings


@pytest.mark.parametrize("raw,expected", [
    ("1490 CC", 1490), ("1,490", 1490), ("1 490", 1490), ("1O50 KG", 1050), ("5", 5), ("2008", 2008),
])
def test_integers(raw, expected):
    assert normalize_value(raw, INT)[0] == expected


def test_integer_without_digits_has_no_value():
    value, warnings = normalize_value("NONE", INT)
    assert value is None and warnings == ["no number found"]
    assert normalize_value("Not applicable", NUM) == (None, ["no number found"])
    assert normalize_value("FIVE", INT) == (5, [])  # a number in words (see below)


@pytest.mark.parametrize("raw,expected", [
    ("1,200.50", 1200.5), ("1.200,50", 1200.5), ("12,5", 12.5), ("3.75 t", 3.75), ("1,200", 1200),
])
def test_numbers(raw, expected):
    assert normalize_value(raw, NUM)[0] == expected


def test_names():
    assert normalize_value("  JANE   DOE ", NAME)[0] == "JANE DOE"
    assert normalize_value("DOE ,JANE", NAME)[0] == "DOE, JANE"
    assert normalize_value(": JANE DOE.", NAME)[0] == "JANE DOE"
    assert normalize_value("JANE D0E", NAME)[1]  # digits -> warning


def test_strings_collapse_whitespace():
    assert normalize_value("JANE  DOE", TEXT) == ("JANE DOE", [])
    assert normalize_value("1,200.00", TEXT)[0] == "1,200.00"


# ---------------------------------------------------------------------------- numbers in words

@pytest.mark.parametrize("raw,expected", [
    ("Eight Hundred and Fifty Thousand", 850000),
    ("EIGHT HUNDRED AND FIFTY THOUSAND", 850000),
    ("eight hundred fifty thousand", 850000),  # without "and"
    ("Kenya Shillings Eight Hundred and Fifty Thousand only", 850000),  # words around it
    ("Eight Hundred and Fifty Thousand Only", 850000),
    ("One Million Two Hundred and Fifty Thousand", 1250000),
    ("One Million, Two Hundred Thousand", 1200000),
    ("Two Million One Hundred Thousand", 2100000),
    ("Three Billion Five Hundred Million", 3500000000),
    ("Ninety-Five Thousand", 95000),
    ("Ninety Five Thousand", 95000),
    ("Twenty-Five", 25),
    ("Twenty – Five Thousand", 25000),  # spaced dash
    ("a hundred", 100),
    ("a thousand and one", 1001),
    ("One Hundred and Five", 105),
    ("One Thousand and Five", 1005),
    ("Fifteen Hundred", 1500),
    ("Six Hundred and Seventy Five Thousand", 675000),
    ("five hundred thousand and five hundred", 500500),
    ("FIVE", 5),
    ("Five only", 5),
    ("zero", 0),
    ("Fourty Thousand", 40000),  # a common misspelling
    # OCR damage
    ("FiftyThousand", 50000),
    ("EightHundred and FiftyThousand", 850000),
    ("Six Hundred Thousaud", 600000),
])
def test_integers_in_words(raw, expected):
    value, warnings = normalize_value(raw, INT)
    assert value == expected and all("disagree" not in w for w in warnings), (raw, value, warnings)


def test_repaired_number_word_is_reported():
    assert normalize_value("Six Hundred Thousaud", INT) == (600000, ["read 'Thousaud' as 'thousand'"])


@pytest.mark.parametrize("raw,expected", [
    ("Eight Hundred Thousand and Fifty Cents", 800000.5),
    ("Eight Hundred Thousand Shillings and Fifty Cents", 800000.5),  # one word between is skipped
    ("Twenty-Five Cents", 0.25),
    ("Twelve point five", 12.5),
    ("Eight Hundred and Fifty Thousand", 850000),  # a whole number stays an int
])
def test_numbers_in_words(raw, expected):
    value, warnings = normalize_value(raw, NUM)
    assert value == expected and type(value) is type(expected) and not warnings


def test_cents_on_an_integer_field_are_rounded_with_a_warning():
    value, warnings = normalize_value("Eight Hundred Thousand and Seventy-Five Cents", INT)
    assert value == 800001 and any("rounded" in w for w in warnings)


def test_words_and_figures_that_agree():
    assert normalize_value("Eight Hundred and Fifty Thousand (KES 850,000)", INT) == (850000, [])
    assert normalize_value("(KES 850,000) Eight Hundred and Fifty Thousand only", INT) == (850000, [])


def test_words_and_figures_that_disagree_are_reported():
    value, warnings = normalize_value("Eight Hundred and Fifty Thousand (KES 860,000)", INT)
    assert value == 850000
    assert warnings == ["the number in words (850,000) and the number in figures (860,000) disagree; used the words"]


@pytest.mark.parametrize("raw", [
    "One of the vehicles",  # a single number word inside other words is not a number
    "for ten days",
    "the second one",
    "a",
    "and",
    "Kenya Shillings",
    "None",
    "Tenant often attends",  # number words inside other words
    "seventh",  # an ordinal is not a cardinal; nor is it repaired into "seventy"
])
def test_text_that_is_not_a_number_in_words(raw):
    value, warnings = normalize_value(raw, INT)
    assert value is None and warnings == ["no number found"]


@pytest.mark.parametrize("raw", [
    "Five Twenty",  # adjacent number words that do not form one number
    "One Thousand Thousand",
    "Two Thousand Five Million",  # scales out of order
    "Hundred Thousand",  # a scale with nothing to multiply
    "Twenty Twelve",
    "Five Five",
])
def test_number_words_that_do_not_form_a_number(raw):
    value, warnings = normalize_value(raw, INT)
    assert value is None and warnings == ["could not read the number in words"]


def test_unreadable_words_fall_back_to_the_figures_with_a_warning():
    value, warnings = normalize_value("Five Twenty (KES 520)", INT)
    assert value == 520 and warnings == ["could not read the number in words; used the figures"]


def test_several_numbers_in_words():
    value, warnings = normalize_value("Two Hundred Thousand and later Three Hundred Thousand", INT)
    assert value == 200000 and any("several numbers" in w for w in warnings)


def test_figures_only_behave_as_before():
    for raw, expected in [("1490 CC", 1490), ("1,490", 1490), ("1O50 KG", 1050), ("ROOM 5", 5)]:
        assert normalize_value(raw, INT)[0] == expected


@pytest.mark.parametrize("raw,expected", [
    ("Fifth day of March Two Thousand and Twenty-Four", "2024-03-05"),
    ("FIFTH DAY OF MARCH TWO THOUSAND AND TWENTY FOUR", "2024-03-05"),
    ("5th day of March, Two Thousand and Nineteen", "2019-03-05"),
    ("Twenty-First day of June 2019", "2019-06-21"),
    ("Twenty First day of June, in the year Two Thousand and Nineteen", "2019-06-21"),
    ("Thirtieth day of September Two Thousand and Twenty", "2020-09-30"),
    ("first of June Nineteen Hundred and Ninety", "1990-06-01"),
    ("March Fifth, Nineteen Ninety-Nine", "1999-03-05"),
    ("Second day of May Twenty Twenty-Four", "2024-05-02"),
])
def test_dates_in_words(raw, expected):
    assert normalize_value(raw, DATE) == (expected, [])


@pytest.mark.parametrize("raw", [
    "Thirty-First day of February Two Thousand and Twenty",  # impossible
    "Fifth day of Smarch Two Thousand and Twenty",
    "Fifth day of March Two Thousand Thousand",  # not a year
    "Fifth day of March",
])
def test_dates_in_words_that_are_not_dates(raw):
    value, warnings = normalize_value(raw, DATE)
    assert value is None and warnings == ["could not parse date"]


SAMPLES = [
    "12.03.1990", "12-MAR-1990", "garbage", "", "   ", "1990-03-12", "kda 123a", "1,490 CC",
    "1.200,50", "DOE ,JANE", ": x :", "0+", "A0I2345678Z", "12.03.90", "-", "—", "N/A",
    "ñandú 12", "😀", "1e309", "9" * 400, "\x00\x01", "12 Mar 2031 extra", "31.02.2000",
    "5th day of March, 2024", "March 5th, 2024", "the 1st of June 2020", "31st day of February 2020",
    "Eight Hundred and Fifty Thousand", "Eight Hundred Thousand and Fifty Cents", "Five Twenty",
    "Fifty Thousand (KES 60,000)", "Fifth day of March Two Thousand and Twenty-Four", "FiftyThousand",
    "Twelve point five", "One of the vehicles", "Three Trillion",
]
SPECS = [DATE, IDENT, INT, NUM, NAME, TEXT,
         FieldSpec(name="pin", type="identifier", pattern=r"[AP]\d{9}[A-Z]"),
         FieldSpec(name="blood", pattern=r"(A|B|AB|O)\s?[+-]?"),
         FieldSpec(name="bad", pattern="(")]


@pytest.mark.parametrize("spec", SPECS, ids=lambda s: f"{s.name}-{s.type.value}")
def test_never_raises_and_is_idempotent(spec):
    for raw in SAMPLES:
        value, warnings = normalize_value(raw, spec)
        assert isinstance(warnings, list)
        if value is None:
            continue
        again, _ = normalize_value(str(value), spec)
        assert again == value, (raw, value, again)


def test_non_string_input_does_not_raise():
    assert normalize_value(1490, INT)[0] == 1490
    assert normalize_value(None, TEXT)[0] is None
    assert normalize_value(12.5, NUM)[0] == 12.5


def test_number_word_split_by_a_hyphen_is_joined():
    assert normalize_value("TWO MIL-LION SIX HUNDRED AND FORTY THOUSAND", INT) == (2640000, [])
    assert normalize_value("Eight Hund- red Thousand", INT) == (800000, [])


def test_number_words_left_out_of_the_number_are_reported():
    # OCR damage cut "TWO" off the number; the rest is read, and the loss is not silent.
    value, warnings = normalize_value("TWO MXXLION SIX HUNDRED AND FORTY THOUSAND", INT)
    assert value == 640000 and warnings == ["number words outside the number read were ignored: 'TWO'"]


@pytest.mark.parametrize("raw", ["fifty-a-and-Fifth-three-of-a-(-hundred", "a (hundred", "a", "a a a"])
def test_number_word_scan_terminates(raw):
    value, warnings = normalize_value(raw, INT)  # returns, and says why when nothing was read
    assert value is not None or warnings


def test_number_words_glued_by_ocr():
    assert normalize_value("Kenya ShillingsONE HUNDRED THOUSAND only", INT) == (100000, [])
    assert normalize_value("TWO MILLIONEIGHTHUND-RED AND SIXTY-FIVE THOUSAND", INT) == (2865000, [])
    assert normalize_value("Twentieth day of January TwoThousand and Fifteen", DATE) == ("2015-01-20", [])


def test_mixed_separators_between_groups_of_three_are_thousands():
    """OCR read "1,435,000" as "1.435,000". Read as a decimal that is 1435, a wrong value with
    no warning; every group after the first having three digits says it is thousands grouping."""
    from dococr.models import FieldSpec

    spec = FieldSpec(name="amount", type="integer")
    for raw in ("1.435,000", "1,435.000", "KES 1.435,000"):
        value, warnings = normalize_value(raw, spec)
        assert value == 1435000, raw
        assert any("read as thousands" in w for w in warnings), raw
    # ordinary forms are untouched and carry no such warning
    for raw, want in (("1,435,000", 1435000), ("1,435.50", 1436), ("12.5", 12), ("1.435", 1)):
        value, warnings = normalize_value(raw, spec)
        assert value == want, raw
        assert not any("read as thousands" in w for w in warnings), raw
    assert normalize_value("1,435.50", FieldSpec(name="amount", type="number"))[0] == 1435.5


def test_a_hyphen_after_a_letter_is_not_a_minus_sign():
    from dococr.models import FieldSpec

    spec = FieldSpec(name="n", type="integer")
    assert normalize_value("ROOM-5", spec)[0] == 5
    assert normalize_value("LR-4521", spec)[0] == 4521
    assert normalize_value("balance -12", spec)[0] == -12  # a sign after a space still counts
    once = normalize_value("ROOM-5", spec)[0]
    assert normalize_value(str(once), spec)[0] == once


# ---------------------------------------------------------------------------- date_order (YAML)

US = FieldSpec(name="d", type="date", date_order="month_first")


@pytest.mark.parametrize("raw,expected", [
    ("12/05/08", "2008-12-05"),
    ("12/05/2008", "2008-12-05"),
    ("1-2-1999", "1999-01-02"),
    ("2008-05-12", "2008-05-12"),  # year first is not affected
    ("5 December 2008", "2008-12-05"),  # nor is a month name
    ("December 5, 2008", "2008-12-05"),
    ("12052008", "2008-12-05"),  # compact: MMDDYYYY after YYYYMMDD
])
def test_month_first_dates(raw, expected):
    assert normalize_value(raw, US)[0] == expected


def test_month_first_turns_to_day_first_only_when_impossible():
    value, warnings = normalize_value("25/12/2008", US)
    assert value == "2008-12-25" and "read as day-first because month-first is impossible" in warnings
    assert normalize_value("32/13/2008", US) == (None, ["could not parse date"])


def test_default_date_order_is_unchanged():
    assert normalize_value("12/05/2008", DATE)[0] == "2008-05-12"
    assert normalize_value("12/05/2008", FieldSpec(name="d", type="date", date_order="day_first"))[0] \
        == "2008-05-12"


def test_month_first_is_idempotent():
    value, _ = normalize_value("12/05/08", US)
    assert normalize_value(value, US) == (value, [])


def test_document_date_order_is_the_default_for_its_fields():
    from dococr.models import DocType

    doc = DocType(name="t", date_order="month_first", fields=[
        {"name": "a", "type": "date"},
        {"name": "b", "type": "date", "date_order": "day_first"},
    ])
    a, b = doc.fields
    assert normalize_value("12/05/2008", a)[0] == "2008-12-05"
    assert normalize_value("12/05/2008", b)[0] == "2008-05-12"
    with pytest.raises(ValueError):
        DocType(name="t", date_order="MDY")


def test_typed_field_without_a_parse_has_no_value_but_keeps_its_warning():
    for spec, raw, warning in [(DATE, "to be agreed", "could not parse date"),
                               (INT, "Not applicable", "no number found"),
                               (NUM, "n/a", "no number found")]:
        value, warnings = normalize_value(raw, spec)
        assert value is None and warning in warnings
        assert normalize_value(value, spec) == (None, [])  # idempotent
