"""Ruled forms on hand-built OCR lines: labels printed on two or three lines of a cell, fill-in
lines with leaders, and entries struck out with a row of marks. Fast: no OCR model. All values
are invented."""

from __future__ import annotations

from dococr.doctypes import resolve_fields
from dococr.models import DocType, FieldSpec
from dococr.ocr import OcrLine
from dococr.ocr.matcher import group_rows, match_fields, tokenize

H = 34  # text height
CW = 15  # character width


def L(text: str, x: float, y: float, w: float | None = None, h: float = H, conf: float = 0.98) -> OcrLine:
    return OcrLine(text, conf, (x, y, x + (w if w is not None else CW * len(text)), y + h))


def doc(*fields: FieldSpec, other_labels=()) -> DocType:
    return DocType(name="t", fields=list(fields), other_labels=list(other_labels))


def run(lines, doc_type, names=None):
    specs = resolve_fields(doc_type, names or [f.name for f in doc_type.fields])
    matches, text = match_fields(lines, doc_type, specs)
    return {m.name: m for m in matches}, text


def raws(res):
    return {k: v.raw for k, v in res.items()}


DOB = FieldSpec(name="date_of_birth", label="Date of Birth", type="date")
NAME = FieldSpec(name="name", label="Name", type="name")
OFFICER = FieldSpec(name="officer", label="Name of Registering Officer", type="name")
PLACE = FieldSpec(name="where_born", label="Where Born")


# ---------------------------------------------------------------------------- stacked labels

def test_overlapping_label_lines_are_two_rows_not_one():
    """Lines of one label are often closer than a text height. They must not be read as one
    row, which interleaves their words ("Date Birth of")."""
    lines = [L("Date of", 100, 400), L("Birth", 115, 428), L("14th March 2001", 300, 412)]
    words = tokenize(lines, [])
    rows = group_rows(lines, words)
    assert [[w.text for w in r] for r in rows] == [["Date", "of", "14th", "March", "2001"], ["Birth"]]


def test_side_by_side_boxes_on_one_row_are_still_one_row():
    lines = [L("Date of", 100, 400), L("Birth", 230, 404), L("14.03.2001", 400, 398)]
    rows = group_rows(lines, tokenize(lines, []))
    assert len(rows) == 1


def _stack(value_y: float, **kw):
    return [L("Date of", 100, 400), L("Birth", 115, 429), L("14.03.2001", 300, value_y, **kw)]


def test_stacked_label_value_level_with_first_line():
    res, _ = run(_stack(398), doc(DOB))
    assert res["date_of_birth"].raw == "14.03.2001" and res["date_of_birth"].warnings == []


def test_stacked_label_value_level_with_second_line():
    res, _ = run(_stack(430), doc(DOB))
    assert res["date_of_birth"].raw == "14.03.2001"


def test_stacked_label_value_between_its_lines():
    res, _ = run(_stack(414), doc(DOB))
    assert res["date_of_birth"].raw == "14.03.2001"


def test_stacked_label_value_below():
    lines = [L("Date of", 100, 400), L("Birth", 115, 429), L("14.03.2001", 100, 472)]
    res, _ = run(lines, doc(DOB))
    assert res["date_of_birth"].raw == "14.03.2001"


def test_stacked_label_with_its_value_in_the_same_box_as_the_last_line():
    """OCR often reads the second line of the label and the value as one box."""
    lines = [L("Where", 550, 309), L("Born   ST. ANNE HOSPITAL", 557, 338)]
    res, _ = run(lines, doc(PLACE))
    assert res["where_born"].raw == "ST. ANNE HOSPITAL"


def test_three_line_label():
    spec = FieldSpec(name="guardian", label="Name and Surname of Guardian", type="name")
    lines = [L("Name and", 100, 500), L("Surname of", 95, 529), L("Guardian", 105, 558),
             L("PETER OMONDI", 330, 530)]
    res, _ = run(lines, doc(spec))
    assert res["guardian"].raw == "PETER OMONDI"


def test_left_aligned_stacked_label():
    lines = [L("Name of", 80, 680), L("Registering Officer", 80, 708), L("A.B. OCHIENG", 454, 694)]
    res, _ = run(lines, doc(OFFICER))
    assert res["officer"].raw == "A.B. OCHIENG"


def test_short_label_does_not_take_a_word_of_a_stacked_label():
    """The bug: "Name" matched the start of the box "Name of" and returned "of", with full
    confidence, because the longer label printed over two lines was not recognised."""
    lines = [
        L("Name", 960, 324), L("GRACE AKINYI", 1078, 341),
        L("Name of", 147, 680), L("A.B. OCHIENG", 454, 694),
        L("Registering Officer", 80, 708),
    ]
    res, _ = run(lines, doc(NAME, OFFICER))
    assert raws(res) == {"name": "GRACE AKINYI", "officer": "A.B. OCHIENG"}


def test_short_label_inside_a_longer_label_whose_rest_was_not_read_gives_no_value():
    """Only "Name of" is on the page (the second line of the longer label was not read): the
    word after "Name" belongs to that label, so the short label gives nothing."""
    lines = [L("Name of", 147, 680), L("A.B. OCHIENG", 454, 694)]
    res, _ = run(lines, doc(NAME, OFFICER))
    assert res["name"].raw is None
    assert "start of the longer label 'Name of Registering Officer'" in res["name"].warnings[0]


def test_short_label_followed_by_an_unrelated_word_still_reads_its_value():
    lines = [L("Name OTIENO", 100, 100)]
    res, _ = run(lines, doc(NAME, OFFICER))
    assert res["name"].raw == "OTIENO"


def test_two_labels_one_above_the_other_with_values_in_their_boxes_are_not_joined():
    """The upper line has its value in its own box, so it is a row of its own."""
    first = FieldSpec(name="first", label="Name")
    last = FieldSpec(name="last", label="Surname")
    both = FieldSpec(name="both", label="Name Surname")
    lines = [L("Name: JOHN", 100, 100), L("Surname: DOE", 100, 128)]
    res, _ = run(lines, doc(first, last, both))
    assert raws(res) == {"first": "JOHN", "last": "DOE", "both": None}


def test_label_lines_that_are_not_aligned_are_not_joined():
    lines = [L("Date of", 100, 400), L("Birth", 400, 429), L("14.03.2001", 600, 429)]
    res, _ = run(lines, doc(DOB))
    assert res["date_of_birth"].raw is None


def test_label_lines_too_far_apart_are_not_joined():
    lines = [L("Date of", 100, 400), L("Birth", 115, 400 + 2 * H), L("14.03.2001", 300, 400 + 2 * H)]
    res, _ = run(lines, doc(DOB))
    assert res["date_of_birth"].raw is None


def test_label_lines_that_each_have_text_to_the_right_are_rows_of_their_own():
    """Each line has its own value to its right: two rows of the form, not one label, even
    though the two lines joined are a label ("Body Colour"). The lower row's own label wins."""
    body = FieldSpec(name="body", label="Body")
    colour = FieldSpec(name="colour", label="Colour", aliases=["Body Colour"])
    lines = [L("BODY", 40, 300), L("SALOON", 300, 300), L("COLOUR", 40, 340), L("WHITE", 300, 340)]
    res, _ = run(lines, doc(body, colour))
    assert raws(res) == {"body": "SALOON", "colour": "WHITE"}


def test_same_when_the_upper_line_is_not_a_label():
    colour = FieldSpec(name="colour", label="Colour", aliases=["Body Colour"])
    lines = [L("BODY", 40, 300), L("SALOON", 300, 300), L("COLOUR", 40, 336), L("WHITE", 300, 336)]
    res, _ = run(lines, doc(colour))
    assert res["colour"].raw == "WHITE"


def test_stacked_label_whose_lines_are_labels_of_other_fields_is_flagged():
    """"BODY" above "COLOUR" with one value beside them: two labels, or the label "Body Colour"
    on two lines? The page cannot tell, so the value carries a warning."""
    body = FieldSpec(name="body", label="Body")
    colour = FieldSpec(name="colour", label="Colour", aliases=["Body Colour"])
    lines = [L("BODY", 40, 300), L("COLOUR", 40, 330), L("WHITE", 300, 330)]
    res, _ = run(lines, doc(body, colour))
    assert res["colour"].raw == "WHITE"
    assert any("also labels on their own ('Body' / 'Colour')" in w for w in res["colour"].warnings)


def test_text_right_of_every_line_of_a_stacked_label_gives_no_value():
    """Values level with both lines of a two-line label cannot be told from two rows of the
    form with unlisted labels, so the label is not read as stacked and the field says why."""
    spec = FieldSpec(name="informant", label="Name and Description of Informant", multiline=True)
    lines = [L("Name and Description", 106, 584), L("JANE ATIENO", 472, 590),
             L("of Informant", 163, 615), L("MOTHER", 472, 621)]
    res, _ = run(lines, doc(spec))
    assert res["informant"].raw is None
    assert "separate rows" in res["informant"].warnings[0]


def test_stacked_label_anchor_lines_cover_every_line_of_the_label():
    lines = _stack(398)
    res, _ = run(lines, doc(DOB))
    assert res["date_of_birth"].anchor_lines == [0, 1]


# ---------------------------------------------------------------------------- fill-in lines

DISTRICT = FieldSpec(name="district", label="Birth in the", until="District in the")
PROVINCE = FieldSpec(name="province", label="District in the", until="Province")


def _fill(*middle: OcrLine) -> list[OcrLine]:
    return [L("Birth in the", 147, 234), *middle]


def test_fill_in_line_read_as_sentence_values_across_wide_gaps():
    lines = _fill(L("NAKURU", 499, 243), L("District in the", 839, 249), L("RIFT VALLEY", 1150, 250),
                  L("..Province", 1400, 253))
    res, _ = run(lines, doc(DISTRICT, PROVINCE))
    assert raws(res) == {"district": "NAKURU", "province": "RIFT VALLEY"}
    assert res["district"].warnings == [] and res["province"].warnings == []


def test_fill_in_line_with_leaders_read_as_text():
    lines = _fill(L("......NAKURU......", 400, 243), L("District in the", 839, 249),
                  L("....RIFT VALLEY....", 1120, 250), L("Province", 1460, 253))
    res, _ = run(lines, doc(DISTRICT, PROVINCE))
    assert raws(res) == {"district": "NAKURU", "province": "RIFT VALLEY"}


def test_fill_in_line_with_form_labels_and_a_stop_label():
    district = FieldSpec(name="district", label="Birth in the")
    province = FieldSpec(name="province", label="District in the")
    lines = _fill(L("NAKURU", 499, 243), L("District in the", 839, 249), L("RIFT VALLEY", 1150, 250),
                  L("..Province", 1400, 253))
    res, _ = run(lines, doc(district, province, other_labels=["Province"]))
    assert raws(res) == {"district": "NAKURU", "province": "RIFT VALLEY"}


def test_fill_in_reading_needs_the_end_text_to_start_a_box():
    """The end text inside a longer box is not the printed text after a blank: the row is not
    read as a fill-in line, and the field is read in sentence order (and here not found)."""
    lines = _fill(L("NAKURU", 499, 243), L("North District in the", 780, 249))
    res, _ = run(lines, doc(DISTRICT))
    assert res["district"].raw is None


def test_fill_in_entry_too_close_to_its_label_is_read_in_sentence_order():
    """With less than SEG_GAP text heights between the label and the entry, they are one
    visual line and the row is not a fill-in line; the end text in the next box is not reached."""
    lines = [L("District in the", 839, 249), L("RIFT VALLEY", 1090, 250), L("Province", 1400, 253)]
    res, _ = run(lines, doc(PROVINCE))
    assert res["province"].raw is None


def test_fill_in_reading_needs_the_label_to_end_its_box():
    lines = [L("Birth in the town of", 147, 234), L("NAKURU", 499, 243), L("District in the", 839, 249)]
    res, _ = run(lines, doc(DISTRICT))
    assert res["district"].raw is None


def test_leader_dots_before_a_short_label_do_not_stop_it():
    sex = FieldSpec(name="sex", label="Sex")
    lines = [L("..Sex", 100, 100), L("F", 300, 100)]
    res, _ = run(lines, doc(sex))
    assert res["sex"].raw == "F"


def test_leaders_are_not_part_of_a_form_value():
    sex = FieldSpec(name="sex", label="Sex")
    lines = [L("Sex ........ FEMALE ......", 100, 100)]
    res, _ = run(lines, doc(sex))
    assert res["sex"].raw == "FEMALE"


# ---------------------------------------------------------------------------- struck-out entries

def test_a_row_of_x_is_no_value():
    spec = FieldSpec(name="father", label="Name of Father", type="name")
    res, _ = run([L("Name of Father", 100, 100), L("XXXXXXXXXXXXXXXXXXXX", 400, 100)], doc(spec))
    assert res["father"].raw is None
    assert "row of 'X'" in res["father"].warnings[0]


def test_a_row_of_dashes_is_no_value_but_a_short_code_is():
    spec = FieldSpec(name="code", label="Code")
    res, _ = run([L("Code", 100, 100), L("- - - - - -", 300, 100)], doc(spec))
    assert res["code"].raw is None
    res, _ = run([L("Code", 100, 100), L("XXX", 300, 100)], doc(spec))
    assert res["code"].raw == "XXX"


def test_text_far_right_on_the_second_line_is_another_cell_not_a_row_of_its_own():
    """The second label line's row holds another cell's text far to the right (its label not
    recognised): the lines are still one label, because the texts beside them are not one
    column."""
    lines = [L("Date of", 94, 391), L("14th March 2001", 200, 412),
             L("Birth", 109, 420), L("PETER OMONDI", 1064, 424)]
    res, _ = run(lines, doc(DOB))
    assert res["date_of_birth"].raw == "14th March 2001"


def test_last_label_line_glued_to_its_value_by_the_ocr():
    """The value follows the label's last line with one space, and OCR lost it."""
    spec = FieldSpec(name="enrolment_number", label="Enrolment No.")
    reg = FieldSpec(name="registered", label="Date of Registration", type="date")
    lines = [L("Enrolment", 71, 204), L("No.687860/36", 106, 223),
             L("Date of", 900, 204), L("Registration06.03.2009", 880, 223)]
    res, _ = run(lines, doc(spec, reg))
    assert raws(res) == {"enrolment_number": "687860/36", "registered": "06.03.2009"}


def test_a_short_label_word_glued_to_a_figure_without_a_full_stop_is_not_split():
    spec = FieldSpec(name="enrolment_number", label="Enrolment No.")
    lines = [L("Enrolment", 71, 204), L("No687860", 106, 223)]
    res, _ = run(lines, doc(spec))
    assert res["enrolment_number"].raw is None


def test_a_label_word_glued_to_a_figure_inside_a_value_is_left_alone():
    spec = FieldSpec(name="plot", label="Plot")
    other = FieldSpec(name="number", label="Enrolment No.")
    lines = [L("Plot", 100, 100), L("Block 4 No.2237", 300, 100)]
    res, _ = run(lines, doc(spec, other))
    assert res["plot"].raw == "Block 4 No.2237"


def test_shipped_birth_certificate_type_reads_its_layout():
    """The shipped `birth_certificate` type against the layout it was written for: labels on two
    lines in cells, a fill-in line with leaders, and a struck-out entry. Invented values."""
    from dococr import load_doc_type
    from dococr.ocr import OcrLine
    from dococr.ocr.matcher import match_fields

    def line(text, left, top, width=None, height=24):
        return OcrLine(text, 0.98, (left, top, left + (width or 13 * len(text)), top + height))

    lines = [
        line("REPUBLIC OF KENYA", 600, 90), line("No 481516", 1150, 100),
        line("CERTIFICATE OF BIRTH", 520, 170),
        line("Birth in the", 150, 250), line("NAROK", 520, 250), line("District in the", 850, 250),
        line("RIFT VALLEY", 1150, 250), line("..Province", 1360, 250),
        line("Entry", 120, 300), line("No.", 135, 330), line("2345678/01", 260, 345),
        line("Where", 560, 320), line("Born", 565, 350), line("Narok District Hosp.", 670, 350),
        line("Name", 970, 335), line("Mary Atieno", 1080, 350),
        line("Date of", 100, 400), line("Birth", 115, 428), line("3rd March 2001", 210, 425),
        line("Sex", 570, 425), line("Female", 680, 425),
        line("Name and Surname", 800, 412), line("of Father", 860, 440), line("xxxxxxxxxxxxxxxxxxxx", 1070, 428),
        line("Name and Maiden", 90, 500), line("Name of Mother", 95, 528), line("Ruth Akinyi Odera", 350, 515),
        line("Name and Description", 110, 595), line("of Informant", 170, 623), line("Sgd. Nurse", 480, 610),
        line("Name of", 155, 690), line("Registering Officer", 90, 718), line("T.N. KORIR", 460, 705),
        line("Date of", 940, 685), line("Registration", 915, 713), line("14-6-01", 1115, 700),
    ]
    doc = load_doc_type("birth_certificate")
    matches, _ = match_fields(lines, doc, doc.fields, page_size=(1600, 1500))
    got = {m.name: m.raw for m in matches}
    assert got["district"] == "NAROK" and got["province"] == "RIFT VALLEY"
    assert got["entry_number"] == "2345678/01" and got["certificate_number"] == "481516"
    assert got["where_born"].startswith("Narok District Hosp")
    assert got["name"] == "Mary Atieno" and got["sex"] == "Female"
    assert got["date_of_birth"] == "3rd March 2001" and got["date_of_registration"] == "14-6-01"
    assert got["mother_name"] == "Ruth Akinyi Odera" and got["informant"] == "Sgd. Nurse"
    assert got["registering_officer"] == "T.N. KORIR"
    assert got["father_name"] is None  # struck out
