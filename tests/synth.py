"""Synthetic document renderer for tests and accuracy measurement.

Draws label/value layouts that resemble driving licences, national IDs, logbooks and
passport data pages with Pillow, using made-up data only. Each rendered document carries the
expected normalised value of every field (None for fields deliberately left off the page).

Four further made-up types are fixtures for multi-line and prose values (`FIXTURE_TYPES`, YAML
in tests/doctypes/): `synth_form`, a form with a wrapped name and an address block,
`synth_agreement`, paragraphs of running text with the values inside sentences, and
`synth_contract`, typeset paragraphs: words hyphenated at line ends (inside labels, end texts and
values), values that end at a comma or a full stop, and amounts and dates in words, and
`synth_glued`, typeset paragraphs in which the space between a label or end text and its
neighbour is left out on purpose, as OCR often loses it.

    from tests.synth import make_document, VARIANTS
    doc = make_document("driving_license", "clean", seed=0)
    doc.image, doc.truth, doc.fields

Run as a script to write a few samples to a directory:  python tests/synth.py OUT_DIR
"""

from __future__ import annotations

import datetime as dt
import io
import random
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Optional

from PIL import Image, ImageDraw, ImageFilter, ImageFont

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
FONTS = {
    "sans": "DejaVuSans.ttf",
    "sans-bold": "DejaVuSans-Bold.ttf",
    "serif": "DejaVuSerif.ttf",
    "serif-bold": "DejaVuSerif-Bold.ttf",
    "mono": "DejaVuSansMono.ttf",
}

FIRST = ["JOHN", "MARY", "PETER", "GRACE", "JAMES", "ESTHER", "DAVID", "FAITH", "SAMUEL", "LUCY"]
LAST = ["KAMAU", "OTIENO", "WANJIKU", "MUTUA", "ACHIENG", "KIPROTICH", "NJOROGE", "WAFULA"]
PLACES = ["NAIROBI", "MOMBASA", "KISUMU", "NAKURU", "ELDORET", "NYERI", "MACHAKOS"]
MAKES = [("TOYOTA", "COROLLA"), ("NISSAN", "NOTE"), ("MAZDA", "DEMIO"), ("ISUZU", "DMAX"),
         ("SUBARU", "FORESTER"), ("HONDA", "FIT")]
MONTHS = ["JAN", "FEB", "MAR", "APR", "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC"]
STREETS = ["MOI AVENUE", "KENYATTA ROAD", "RIVER ROAD", "TOM MBOYA STREET", "OLOITOKITOK ROAD"]
OCCUPATIONS = ["TEACHER", "FARMER", "NURSE", "ACCOUNTANT", "MECHANIC", "CIVIL ENGINEER"]


def font(name: str, size: int) -> ImageFont.FreeTypeFont:
    try:
        return ImageFont.truetype(str(FONT_DIR / FONTS[name]), size)
    except OSError:  # pragma: no cover - fonts missing
        return ImageFont.load_default()


# ---------------------------------------------------------------------------- data

def fmt_date(d: dt.date, style: str) -> str:
    return {
        "dot": d.strftime("%d.%m.%Y"),
        "slash": d.strftime("%d/%m/%Y"),
        "dash_mon": f"{d.day:02d}-{MONTHS[d.month - 1]}-{d.year}",
        "space_mon": f"{d.day:02d} {MONTHS[d.month - 1].title()} {d.year}",
        "iso": d.isoformat(),
    }[style]


def rand_date(rng: random.Random, lo: int, hi: int) -> dt.date:
    # Day > 12 half the time so day-first vs month-first errors are caught.
    return dt.date(rng.randint(lo, hi), rng.randint(1, 12), rng.choice([rng.randint(1, 12), rng.randint(13, 28)]))


def digits(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("0123456789") for _ in range(n))


def letters(rng: random.Random, n: int) -> str:
    return "".join(rng.choice("ABCDEFGHJKLMNPRSTUVWXYZ") for _ in range(n))


@dataclass
class Item:
    field: str  # field name ('' = decoration label that is not requested)
    label: str  # printed label
    printed: str  # printed value
    expected: object  # expected normalised value
    lines: Optional[list[str]] = None  # value printed on these lines (overrides `printed`)
    wrap: int = 0  # wrap the value to this many pixels (0 = one line)


def _date_item(rng, name, label, lo, hi) -> Item:
    d = rand_date(rng, lo, hi)
    return Item(name, label, fmt_date(d, rng.choice(["dot", "slash", "dash_mon", "space_mon", "iso"])), d.isoformat())


def _pick(rng: random.Random, label: str, aliases: list[str], alias_rate: float) -> str:
    return rng.choice(aliases) if aliases and rng.random() < alias_rate else label


def items_for(doc_type: str, rng: random.Random, alias_rate: float = 0.3) -> tuple[str, list[Item], list[str]]:
    """(title, items, absent field names)."""
    p = lambda label, *aliases: _pick(rng, label, list(aliases), alias_rate)  # noqa: E731
    surname, first, middle = rng.choice(LAST), rng.choice(FIRST), rng.choice(FIRST)
    if doc_type == "driving_license":
        sex = rng.choice(["M", "F"])
        issue = rand_date(rng, 2015, 2023)
        expiry = issue.replace(year=issue.year + 3)
        lic = letters(rng, 3) + digits(rng, 6)
        return "DRIVING LICENCE", [
            Item("license_number", p("Licence No", "Licence Number", "DL No"), lic, lic),
            Item("surname", p("Surname"), surname, surname),
            Item("other_names", p("Other Names", "Given Names"), f"{first} {middle}", f"{first} {middle}"),
            Item("national_id_number", p("National ID No", "ID No"), nid := digits(rng, 8), nid),
            _date_item(rng, "date_of_birth", p("Date of Birth", "DOB"), 1960, 2002),
            Item("sex", p("Sex", "Gender"), sex, sex),
            Item("blood_group", "Blood Group", g := rng.choice(["O+", "A+", "B-", "AB+"]), g),
            Item("date_of_issue", p("Date of Issue", "Issue Date"), fmt_date(issue, rng.choice(["dot", "slash", "dash_mon"])), issue.isoformat()),
            Item("date_of_expiry", p("Date of Expiry", "Expiry Date"), fmt_date(expiry, rng.choice(["dot", "space_mon", "iso"])), expiry.isoformat()),
            Item("vehicle_classes", p("Classes", "Categories"), c := rng.choice(["B", "B C1", "A B C"]), c),
        ], []
    if doc_type == "national_id":
        idn = digits(rng, 8)
        return "NATIONAL IDENTITY CARD", [
            Item("serial_number", p("Serial Number"), s := digits(rng, 9), s),
            Item("id_number", p("ID Number", "ID NO"), idn, idn),
            Item("full_name", p("Full Names"), f"{first} {middle} {surname}", f"{first} {middle} {surname}"),
            _date_item(rng, "date_of_birth", p("Date of Birth"), 1950, 2004),
            Item("sex", p("Sex"), sx := rng.choice(["MALE", "FEMALE"]), sx),
            Item("place_of_birth", p("District of Birth", "Place of Birth"), pl := rng.choice(PLACES), pl),
            Item("place_of_issue", p("Place of Issue"), pi := rng.choice(PLACES), pi),
            _date_item(rng, "date_of_issue", p("Date of Issue"), 2005, 2023),
        ], ["date_of_expiry"]
    if doc_type == "logbook":
        make, model = rng.choice(MAKES)
        reg = f"K{letters(rng, 2)} {digits(rng, 3)}{letters(rng, 1)}"
        chassis = f"{letters(rng, 3)}{digits(rng, 3)}-{digits(rng, 7)}"
        engine = f"{digits(rng, 1)}{letters(rng, 2)}-{digits(rng, 7)}"
        pin = f"A{digits(rng, 9)}{letters(rng, 1)}"
        cc = rng.choice([1290, 1490, 1800, 1990, 2500])
        year = rng.randint(2005, 2020)
        return "CERTIFICATE OF REGISTRATION", [
            Item("registration_number", p("Registration No", "Reg No"), reg, reg.replace(" ", "")),
            Item("chassis_number", p("Chassis No", "Chassis/Frame"), chassis, chassis),
            Item("engine_number", p("Engine No", "Engine Number"), engine, engine),
            Item("make", "Make", make, make),
            Item("model", "Model", model, model),
            Item("body_type", p("Body", "Body Type"), b := rng.choice(["SALOON", "STATION WAGON", "PICK-UP"]), b),
            Item("colour", p("Colour", "Color"), c := rng.choice(["WHITE", "SILVER", "BLUE", "BLACK"]), c),
            Item("fuel", p("Fuel", "Fuel Type"), f := rng.choice(["PETROL", "DIESEL"]), f),
            Item("year_of_manufacture", p("Year of Manufacture", "Year"), str(year), year),
            Item("engine_capacity", p("Rating", "Engine Capacity"), f"{cc} CC", cc),
            Item("passengers", p("Passengers", "Seating Capacity"), str(n := rng.randint(2, 7)), n),
            Item("tare_weight", "Tare Weight", f"{(t := rng.randint(900, 2200))} KG", t),
            Item("owner_name", p("Registered Owner", "Owner"), f"{first} {surname}", f"{first} {surname}"),
            Item("owner_pin", p("PIN", "KRA PIN"), pin, pin),
            _date_item(rng, "date_of_registration", p("Date of Registration", "Registration Date"), 2006, 2023),
        ], ["logbook_number"]
    if doc_type == "passport":
        issue = rand_date(rng, 2016, 2023)
        expiry = issue.replace(year=issue.year + 10)
        num = letters(rng, 2) + digits(rng, 7)
        return "PASSPORT", [
            Item("passport_number", p("Passport No", "Passport Number"), num, num),
            Item("surname", "Surname", surname, surname),
            Item("given_names", "Given Names", f"{first} {middle}", f"{first} {middle}"),
            Item("nationality", "Nationality", "KENYAN", "KENYAN"),
            _date_item(rng, "date_of_birth", "Date of Birth", 1955, 2005),
            Item("sex", "Sex", sx := rng.choice(["M", "F"]), sx),
            Item("place_of_birth", "Place of Birth", pl := rng.choice(PLACES), pl),
            Item("date_of_issue", "Date of Issue", fmt_date(issue, rng.choice(["space_mon", "dash_mon", "dot"])), issue.isoformat()),
            Item("date_of_expiry", "Date of Expiry", fmt_date(expiry, rng.choice(["space_mon", "dash_mon", "dot"])), expiry.isoformat()),
        ], []
    if doc_type == "synth_form":
        # A made-up membership form: a wrapped name and a two- or three-line address block
        # next to single-line fields. The YAML is tests/doctypes/synth_form.yaml.
        names = [rng.choice(FIRST) for _ in range(rng.randint(1, 3))] + [rng.choice(LAST) for _ in range(rng.randint(1, 2))]
        full = " ".join(names)
        box = f"P.O. BOX {rng.randint(10, 99999)}" + (f"-{rng.choice(['00100', '00200', '40100', '80100'])}" if rng.random() < 0.6 else "")
        addr = [box] + ([rng.choice(STREETS)] if rng.random() < 0.5 else []) + [rng.choice(PLACES)]
        wrap = rng.choice([220, 260, 300, 360])
        phone = "07" + digits(rng, 8)
        return "MEMBERSHIP APPLICATION", [
            Item("member_number", p("Member No", "Membership No"), m := f"M{digits(rng, 6)}", m),
            Item("full_name", p("Full Name", "Name of Applicant"), full, full, wrap=wrap),
            Item("id_number", p("ID Number", "ID No"), nid := digits(rng, 8), nid),
            _date_item(rng, "date_of_birth", p("Date of Birth"), 1950, 2004),
            Item("postal_address", p("Postal Address", "Address"), " ".join(addr), " ".join(addr), lines=addr),
            Item("phone_number", p("Telephone", "Phone No"), phone, phone),
            Item("occupation", p("Occupation"), o := rng.choice(OCCUPATIONS), o),
            Item("county", p("County"), c := rng.choice(PLACES), c),
        ], ["next_of_kin"]
    raise ValueError(doc_type)


# ---------------------------------------------------------------------------- layouts

@dataclass
class Style:
    label_font: str = "sans"
    value_font: str = "sans-bold"
    label_size: int = 20
    value_size: int = 24
    upper_labels: bool = True
    bg: tuple[int, int, int] = (238, 242, 230)
    label_color: tuple[int, int, int] = (70, 70, 90)


def _label(text: str, style: Style) -> str:
    return text.upper() if style.upper_labels else text


def _canvas(width: int, height: int, title: str, style: Style) -> tuple[Image.Image, ImageDraw.ImageDraw, int]:
    img = Image.new("RGB", (width, height), style.bg)
    d = ImageDraw.Draw(img)
    d.rectangle([8, 8, width - 9, height - 9], outline=(120, 140, 120), width=3)
    d.text((width // 2, 22), "REPUBLIC OF KENYA", font=font("serif-bold", 26), fill=(20, 60, 20), anchor="mt")
    d.text((width // 2, 58), title, font=font("serif-bold", 22), fill=(20, 60, 20), anchor="mt")
    return img, d, 105


def wrap_text(text: str, fnt: ImageFont.FreeTypeFont, max_w: float) -> list[str]:
    """Greedy word wrap to a pixel width."""
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    lines: list[str] = []
    for word in text.split():
        if lines and probe.textlength(f"{lines[-1]} {word}", font=fnt) <= max_w:
            lines[-1] += f" {word}"
        else:
            lines.append(word)
    return lines or [""]


def _value_lines(it: Item, fnt: ImageFont.FreeTypeFont, max_w: Optional[float] = None) -> list[str]:
    if it.lines:
        return it.lines
    if it.wrap:
        return wrap_text(it.printed, fnt, min(it.wrap, max_w) if max_w else it.wrap)
    return [it.printed]


def _pitch(style: Style) -> int:
    """Distance between the lines of one multi-line value."""
    return int(style.value_size * 1.25)


def layout_rows(title: str, items: list[Item], style: Style, colon: bool = False) -> Image.Image:
    """One label/value pair per row; 'Label  value' (aligned column) or 'Label: value'.
    A multi-line value continues under its first line, aligned with it."""
    row_h = max(style.label_size, style.value_size) + 18
    width = 1000
    lf, vf = font(style.label_font, style.label_size), font(style.value_font, style.value_size)
    values = [_value_lines(it, vf) for it in items]
    pitch = _pitch(style)
    img, d, y = _canvas(width, 105 + sum(row_h + (len(v) - 1) * pitch for v in values) + 30, title, style)
    d.rectangle([width - 230, 110, width - 40, 340], fill=(200, 205, 200))  # photo placeholder
    col = 40 + max(int(d.textlength(_label(i.label, style), font=lf)) for i in items) + 40
    for it, vals in zip(items, values):
        if colon:
            text = f"{_label(it.label, style)}: "
            d.text((40, y), text, font=lf, fill=style.label_color)
            vx = 40 + d.textlength(text, font=lf)
        else:
            d.text((40, y), _label(it.label, style), font=lf, fill=style.label_color)
            vx = col
        for k, val in enumerate(vals):
            d.text((vx, y - 2 + k * pitch), val, font=vf, fill=(0, 0, 0))
        y += row_h + (len(vals) - 1) * pitch
    return img


def layout_two_column(title: str, items: list[Item], style: Style) -> Image.Image:
    half = (len(items) + 1) // 2
    cols = [items[:half], items[half:]]
    row_h = max(style.label_size, style.value_size) + 20
    width = 1400
    lf, vf = font(style.label_font, style.label_size), font(style.value_font, style.value_size)
    pitch = _pitch(style)
    heights = [sum(row_h + (len(_value_lines(it, vf)) - 1) * pitch for it in col) for col in cols]
    img, d, y0 = _canvas(width, 105 + max(heights) + 30, title, style)
    for ci, col_items in enumerate(cols):
        x0 = 40 + ci * (width // 2)
        if not col_items:
            continue
        vx = x0 + max(int(d.textlength(_label(i.label, style), font=lf)) for i in col_items) + 30
        y = y0
        for it in col_items:
            vals = _value_lines(it, vf)
            d.text((x0, y), _label(it.label, style), font=lf, fill=style.label_color)
            for k, val in enumerate(vals):
                d.text((vx, y - 2 + k * pitch), val, font=vf, fill=(0, 0, 0))
            y += row_h + (len(vals) - 1) * pitch
    return img


def layout_value_below(title: str, items: list[Item], style: Style, ncols: int = 3,
                       cell_w: int = 330) -> Image.Image:
    """Small label above its value, laid out in a grid (passport/ID card style). A grid row is
    as tall as its longest multi-line value."""
    cell_h = style.label_size + style.value_size + 34
    nrows = (len(items) + ncols - 1) // ncols
    width = 60 + ncols * cell_w
    lf = font(style.label_font, max(14, style.label_size - 4))
    vf = font(style.value_font, style.value_size)
    pitch = _pitch(style)
    values = [_value_lines(it, vf, cell_w - 30) for it in items]
    extra = [max(len(v) - 1 for v in values[r * ncols:(r + 1) * ncols]) * pitch for r in range(nrows)]
    img, d, y0 = _canvas(width, 105 + nrows * cell_h + sum(extra) + 30, title, style)
    for k, (it, vals) in enumerate(zip(items, values)):
        r, c = divmod(k, ncols)
        x, y = 40 + c * cell_w, y0 + r * cell_h + sum(extra[:r])
        d.text((x, y), _label(it.label, style), font=lf, fill=style.label_color)
        for j, val in enumerate(vals):
            d.text((x, y + style.label_size + 4 + j * pitch), val, font=vf, fill=(0, 0, 0))
    return img


# ---------------------------------------------------------------------------- prose

ORDINAL = {1: "st", 2: "nd", 3: "rd", 21: "st", 22: "nd", 23: "rd", 31: "st"}
MONTH_NAMES = ["January", "February", "March", "April", "May", "June", "July", "August",
               "September", "October", "November", "December"]
AMOUNTS = {450000: "Four Hundred and Fifty Thousand", 800000: "Eight Hundred Thousand",
           1250000: "One Million Two Hundred and Fifty Thousand", 675000: "Six Hundred and Seventy Five Thousand",
           2100000: "Two Million One Hundred Thousand", 95000: "Ninety Five Thousand"}


def prose_date(d: dt.date, rng: random.Random) -> str:
    day = f"{d.day}{ORDINAL.get(d.day, 'th')}"
    month = MONTH_NAMES[d.month - 1]
    return rng.choice([f"{day} day of {month} {d.year}", f"{day} day of {month}, {d.year}",
                       f"{day} {month} {d.year}"])


def _party(rng: random.Random) -> str:
    if rng.random() < 0.2:
        return f"{rng.choice(LAST)} {rng.choice(['HOLDINGS', 'MOTORS', 'TRADERS'])} LIMITED"
    return " ".join([rng.choice(FIRST) for _ in range(rng.randint(1, 2))] + [rng.choice(LAST) for _ in range(rng.randint(1, 2))])


def _po_box(rng: random.Random) -> str:
    code = f"-{rng.choice(['00100', '00200', '40100', '80100'])}" if rng.random() < 0.5 else ""
    return f"P.O. Box {rng.randint(10, 99999)}{code} {rng.choice(PLACES).title()}"


def prose_for(rng: random.Random) -> tuple[str, list[str], dict[str, object]]:
    """(title, paragraphs, truth) of a made-up agreement. Wording varies by seed; the YAML,
    tests/doctypes/synth_agreement.yaml, lists every wording as a label alias or an `until`
    alternative, as an author would for their own documents."""
    date = rand_date(rng, 2015, 2025)
    pay = date + dt.timedelta(days=rng.randint(10, 120))
    vendor, purchaser = _party(rng), _party(rng)
    v_addr, p_addr = _po_box(rng), _po_box(rng)
    reg = f"K{letters(rng, 2)} {digits(rng, 3)}{letters(rng, 1)}"
    chassis = f"{letters(rng, 3)}{digits(rng, 3)}-{digits(rng, 7)}"
    amount = rng.choice(list(AMOUNTS))
    words = AMOUNTS[amount]
    witness = " ".join([rng.choice(FIRST), rng.choice(LAST)])
    style = rng.randrange(3)
    seller, buyer = [("Vendor", "Purchaser"), ("Seller", "Buyer"), ("Vendor", "Purchaser")][style]
    p1 = [
        f"THIS AGREEMENT is made on the {prose_date(date, rng)} BETWEEN {vendor} of {v_addr} "
        f"(hereinafter called the {seller}) of the one part AND {purchaser} of {p_addr} "
        f"(hereinafter called the {buyer}) of the other part.",
        f"This Sale Agreement is dated this {prose_date(date, rng)} and is made between {vendor} of "
        f"{v_addr} (hereinafter referred to as the {seller}) and {purchaser} of {p_addr} "
        f"(hereinafter referred to as the {buyer}).",
        f"THIS AGREEMENT FOR SALE is made on the {prose_date(date, rng)} between {vendor} of {v_addr} "
        f"(the {seller}) and {purchaser} of {p_addr} (the {buyer}).",
    ][style]
    p2 = rng.choice([
        f"WHEREAS the {seller} is the registered owner of the motor vehicle registration number {reg} "
        f"chassis number {chassis} and has agreed to sell the same to the {buyer} at the price of "
        f"Kenya Shillings {words} (KES {amount:,}) payable on or before the {prose_date(pay, rng)} "
        f"by bank transfer.",
        f"The {seller} agrees to sell and the {buyer} agrees to buy the motor vehicle registration "
        f"number {reg}, chassis number {chassis}, for the sum of Kenya Shillings {words} "
        f"(KES {amount:,}) to be paid in full on or before the {prose_date(pay, rng)} by bank transfer.",
    ])
    p3 = (f"The {seller} warrants that the said motor vehicle is free from any encumbrance and shall "
          f"deliver it together with the registration book upon payment of the full price.")
    paragraphs = [p1, p2, p3]
    has_witness = rng.random() < 0.7
    if has_witness:
        paragraphs.append(f"SIGNED by the said {vendor} in the presence of {witness} Advocate.")
    truth = {
        "agreement_date": date.isoformat(), "vendor_name": vendor, "vendor_address": v_addr,
        "purchaser_name": purchaser, "purchaser_address": p_addr,
        "registration_number": reg.replace(" ", ""), "chassis_number": chassis,
        "price_in_words": words, "purchase_price": amount, "completion_date": pay.isoformat(),
        "witness_name": witness if has_witness else None,
    }
    return "AGREEMENT FOR SALE OF MOTOR VEHICLE", paragraphs, truth


def layout_prose(title: str, paragraphs: list[str], rng: random.Random, width: int,
                 spacing: float, wrap: Callable = wrap_text) -> Image.Image:
    size = rng.choice([20, 22, 24, 26])
    fnt = font(rng.choice(["serif", "sans"]), size)
    margin = rng.choice([50, 70, 90])
    indent = rng.choice([0, 0, 40])
    pitch = int(size * spacing)
    laid = []
    for para in paragraphs:
        lines = wrap(para, fnt, width - 2 * margin - indent)
        laid.append(lines)
    height = 130 + sum(len(ls) * pitch + int(size * 0.8) for ls in laid) + 40
    img = Image.new("RGB", (width, height), rng.choice([(255, 255, 255), (250, 248, 240)]))
    d = ImageDraw.Draw(img)
    d.text((width // 2, 40), title, font=font("serif-bold", size + 4), fill=(0, 0, 0), anchor="mt")
    y = 110
    for lines in laid:
        for k, line in enumerate(lines):
            d.text((margin + (indent if k == 0 else 0), y), line, font=fnt, fill=(0, 0, 0))
            y += pitch
        y += int(size * 0.8)
    return img


# ---------------------------------------------------------------------------- typeset prose

_UNITS = ["", "One", "Two", "Three", "Four", "Five", "Six", "Seven", "Eight", "Nine", "Ten", "Eleven",
          "Twelve", "Thirteen", "Fourteen", "Fifteen", "Sixteen", "Seventeen", "Eighteen", "Nineteen"]
_TENS = ["", "", "Twenty", "Thirty", "Forty", "Fifty", "Sixty", "Seventy", "Eighty", "Ninety"]
_ORDINALS = ["", "First", "Second", "Third", "Fourth", "Fifth", "Sixth", "Seventh", "Eighth", "Ninth",
             "Tenth", "Eleventh", "Twelfth", "Thirteenth", "Fourteenth", "Fifteenth", "Sixteenth",
             "Seventeenth", "Eighteenth", "Nineteenth", "Twentieth"]


def num_words(n: int) -> str:
    """British English, as agreements write amounts: 'Eight Hundred and Fifty Thousand'."""
    def below_100(k: int) -> str:
        return _UNITS[k] if k < 20 else _TENS[k // 10] + (f"-{_UNITS[k % 10]}" if k % 10 else "")

    def below_1000(k: int) -> str:
        h, r = divmod(k, 100)
        out = f"{_UNITS[h]} Hundred" if h else ""
        if r:
            out += (" and " if h else "") + below_100(r)
        return out

    parts = []
    for scale, name in ((10**6, "Million"), (10**3, "Thousand")):
        if n >= scale:
            parts.append(f"{below_1000(n // scale)} {name}")
            n %= scale
    if n:
        parts.append(("and " if parts and n < 100 else "") + below_1000(n))
    return " ".join(parts)


def ordinal_words(d: int) -> str:
    if d <= 20:
        return _ORDINALS[d]
    tens, unit = divmod(d, 10)
    return _TENS[tens][:-1] + "ieth" if not unit else f"{_TENS[tens]}-{_ORDINALS[unit]}"


def words_date(d: dt.date, rng: random.Random) -> str:
    month = MONTH_NAMES[d.month - 1]
    day = f"{d.day}{ORDINAL.get(d.day, 'th')}"
    return rng.choice([
        f"{ordinal_words(d.day)} day of {month} {num_words(d.year)}",
        f"{day} day of {month}, {num_words(d.year)}",
        f"{ordinal_words(d.day)} day of {month}, {d.year}",
        f"{day} day of {month} {d.year}",
    ])


_VOWELS = set("aeiouyAEIOUY")


_DIGRAPHS = ("ch", "sh", "th", "ph", "wh", "ck", "ng", "gh", "qu")


def _split_points(word: str) -> list[int]:
    """Where a typesetter might hyphenate a plain word: two letters or more before the break,
    three or more after, between syllable-like pieces: between two consonants that are not a
    digraph ("pur-chase", "Shil-lings"), or after a vowel before a consonant and a vowel
    ("Thou-sand", "Ma-chakos")."""
    w = word.lower()
    out = []
    for i in range(2, len(w) - 2):
        a, b = w[i - 1], w[i]
        if not (a.isalpha() and b.isalpha()):
            continue
        if a not in _VOWELS and b not in _VOWELS and a + b not in _DIGRAPHS:
            out.append(i)
        elif a in _VOWELS and b not in _VOWELS and (
                w[i + 1] in _VOWELS or (w[i:i + 2] in _DIGRAPHS and w[i + 2:i + 3] in _VOWELS)):
            out.append(i)
    return out


def hyphen_wrap(text: str, fnt: ImageFont.FreeTypeFont, max_w: float, rng: random.Random,
                breaks: list[tuple[str, str]], caps_rate: float = 0.5) -> list[str]:
    """Greedy wrap that breaks a word at the line end the way typesetting does: after a hyphen
    the word already has, or with a hyphen added between syllable-like pieces (words of six
    letters or more; words in capitals only `caps_rate` of the time). Records each break as
    (text before, text after)."""
    probe = ImageDraw.Draw(Image.new("RGB", (1, 1)))
    fits = lambda line: probe.textlength(line, font=fnt) <= max_w  # noqa: E731
    lines: list[str] = []
    for word in text.split():
        if not lines or fits(f"{lines[-1]} {word}"):
            if lines:
                lines[-1] += f" {word}"
            else:
                lines.append(word)
            continue
        cuts = [i + 1 for i, ch in enumerate(word[:-1]) if ch == "-" and 0 < i]
        core = re.match(r"[A-Za-z]+", word)
        if core and len(core.group(0)) >= 6 and not cuts and (
                not core.group(0).isupper() or rng.random() < caps_rate):
            cuts = _split_points(core.group(0))
        best = None
        for i in cuts:
            head = word[:i] if word[i - 1] == "-" else word[:i] + "-"
            if fits(f"{lines[-1]} {head}"):
                best = (i, head)
        if best:
            lines[-1] += f" {best[1]}"
            lines.append(word[best[0]:])
            breaks.append((best[1], word[best[0]:]))
        else:
            lines.append(word)
    return lines or [""]


def _contract_party(rng: random.Random) -> str:
    if rng.random() < 0.2:
        return f"{rng.choice(LAST)} {rng.choice(['HOLDINGS', 'MOTORS', 'TRADERS'])} LIMITED"
    middle = rng.choice([f"{rng.choice('ABCDEGHJKMNPRSW')}.", rng.choice(FIRST), ""])
    last = rng.choice(LAST)
    if rng.random() < 0.35:
        last = f"{last}-{rng.choice([x for x in LAST if x != last])}"
    return " ".join(x for x in (rng.choice(FIRST), middle, last) if x)


LAND_USES = ["agricultural", "residential", "agricultural and residential", "commercial",
             "horticultural", "educational"]
PROFESSIONS = ["self-employed civil engineer", "registered nurse", "certified public accountant",
               "commercial farmer", "practising advocate", "secondary school teacher",
               "self-employed businessman"]
STREET_NAMES = ["Moi Avenue", "Kenyatta Avenue", "Oginga Odinga Street", "Kimathi Street", "Ngong Road"]


def contract_for(rng: random.Random, printed: Optional[dict[str, str]] = None
                 ) -> tuple[str, list[str], dict[str, object]]:
    """(title, paragraphs, truth) of a made-up land sale agreement for typeset prose. The YAML,
    tests/doctypes/synth_contract.yaml, lists every wording as a label alias or end text.
    `printed`, when given, is filled with each field's value as printed (for locating it)."""
    date = rand_date(rng, 2015, 2025)
    pay = date + dt.timedelta(days=rng.randint(20, 150))
    vendor, purchaser = _contract_party(rng), _contract_party(rng)
    v_addr, p_addr = _po_box(rng), _po_box(rng)
    plot = rng.choice([f"Plot No. {rng.randint(100, 9999)}, {rng.choice(STREET_NAMES)}, {rng.choice(PLACES).title()}",
                       f"L.R. No. {rng.randint(100, 999)}/{rng.randint(1000, 9999)}, {rng.choice(PLACES).title()}"])
    owner = vendor if rng.random() < 0.6 else f"{rng.choice(LAST)} HOLDINGS LTD"
    price = rng.randrange(40, 600) * 5000
    deposit = rng.choice([25000, 50000, 75000, 100000, 150000, 250000])
    caps = rng.random() < 0.4
    amount = (lambda n: num_words(n).upper()) if caps else num_words  # noqa: E731
    use = rng.choice(LAND_USES)
    job = rng.choice(PROFESSIONS)
    article = "an" if job[0] in "aeiou" else "a"
    code = rng.choice(["00100", "00200", "40100", "80100", "20100"])
    notice = f"P.O. Box {rng.randint(10, 99999)}-{code}, {rng.choice(PLACES).title()}"
    witness = " ".join([rng.choice(FIRST), rng.choice(LAST)])
    style = rng.randrange(2)
    # (the random calls are made in the same order as when these were written inline)
    dates = [words_date(date, rng), words_date(date, rng)]
    p1 = [
        f"THIS AGREEMENT is made on the {dates[0]} between {vendor}, of {v_addr} "
        f"(hereinafter called the Vendor) of the one part and {purchaser}, of {p_addr} "
        f"(hereinafter called the Purchaser) of the other part.",
        f"This Agreement for Sale is dated the {dates[1]} and is made between {vendor}, "
        f"of {v_addr} (the Vendor) and {purchaser}, of {p_addr} (the Purchaser).",
    ][style]
    p2 = (f"The Vendor agrees to sell and the Purchaser agrees to purchase the land situated at {plot}. "
          f"The said land is registered in the name of {owner}. The Vendor warrants that the land is "
          f"free from any charge, caveat or other encumbrance whatsoever.")
    pays = [words_date(pay, rng), words_date(pay, rng)]
    choice3 = [
        f"The purchase price is Kenya Shillings {amount(price)} (KES {price:,}) of which a deposit of "
        f"Kenya Shillings {amount(deposit)} only, shall be paid on signing, and the balance shall be "
        f"paid on or before the {pays[0]} by bank transfer.",
        f"The Purchaser shall pay for the land the sum of Kenya Shillings {amount(price)} "
        f"(KES {price:,}), of which a deposit of Kenya Shillings {amount(deposit)} only, is payable on "
        f"signing, and the balance on or before the {pays[1]} by bank transfer.",
    ]
    p3 = rng.choice(choice3)
    p4 = (f"The Purchaser, who is by profession {article} {job}, shall use the land for {use} purposes "
          f"only and shall not subdivide or transfer it without the written consent of the Vendor.")
    # The notice ends its paragraph; on some pages its full stop is missing, as OCR may lose it.
    p5 = (f"All notices to the Purchaser under this Agreement shall be in writing and shall be sent "
          f"to {notice}" + ("" if rng.random() < 0.25 else "."))
    paragraphs = [p1, p2, p3, p4, p5]
    has_witness = rng.random() < 0.8
    if has_witness:
        paragraphs.append(f"SIGNED by the said {vendor} in the presence of {witness}, Advocate.")
    truth = {
        "agreement_date": date.isoformat(), "vendor_name": vendor, "purchaser_name": purchaser,
        "land_location": plot, "registered_owner": owner, "purchase_price": price,
        "price_in_figures": price, "deposit": deposit, "completion_date": pay.isoformat(),
        "occupation": job, "land_use": use, "notice_address": notice,
        "witness_name": witness if has_witness else None,
    }
    if printed is not None:
        printed.update({
            "agreement_date": dates[style], "vendor_name": vendor, "purchaser_name": purchaser,
            "land_location": plot, "registered_owner": owner, "purchase_price": amount(price),
            "price_in_figures": f"{price:,}", "deposit": f"Kenya Shillings {amount(deposit)} only",
            "completion_date": pays[choice3.index(p3)], "occupation": job, "land_use": use,
            "notice_address": notice, "witness_name": witness if has_witness else "",
        })
    return "AGREEMENT FOR SALE OF LAND", paragraphs, truth


def make_contract_document(variant: str = "clean", seed: int = 0) -> SynthDoc:
    rng = random.Random(f"synth_contract/{variant}/{seed}")
    content_rng = random.Random(f"synth_contract/{seed}")
    title, paragraphs, truth = contract_for(content_rng)
    widths, spacing, degrade = PROSE_VARIANTS[variant]
    width = rng.randint(*widths) if widths else rng.randint(1000, 1300)
    breaks: list[tuple[str, str]] = []
    wrap_rng = random.Random(f"synth_contract/{variant}/{seed}/wrap")
    img = layout_prose(title, paragraphs, rng, width, spacing,
                       wrap=lambda t, f, w: hyphen_wrap(t, f, w, wrap_rng, breaks))
    doc = SynthDoc("synth_contract", variant, seed, degrade(img, rng), truth)
    doc.breaks = breaks
    return doc


# ---------------------------------------------------------------------------- typeset prose with lost spaces

FIRMS = ["Mwangi Otieno and Associates", "Kamau Achieng Advocates", "Wafula and Mutua Advocates",
         "Njoroge Kiprotich Law Chambers"]
PARCEL_BLOCKS = ["MUNICIPALITY BLOCK", "BLOCK", "MUNICIPALITY/"]


def glued_for(rng: random.Random) -> tuple[str, list[str], dict[str, object], list[str]]:
    """(title, paragraphs, truth, glued) of a made-up land sale agreement in which OCR's most
    common fault is printed on purpose: the space between an anchor and its neighbour is left
    out. Each place is glued on some pages only; `glued` names the places glued on this page.
    The YAML, tests/doctypes/synth_glued.yaml, has one wording per field.

    Every place glued here has evidence at the join that the matcher accepts: a change of case,
    a letter against a figure, or an opening bracket; except the label "purchase price ..."
    glued to the capitalised "The" before it."""
    glued: list[str] = []

    def g(place: str, rate: float = 0.6) -> str:
        if rng.random() < rate:
            glued.append(place)
            return ""
        return " "

    date = rand_date(rng, 2015, 2025)
    pay = date + dt.timedelta(days=rng.randint(20, 150))
    first = rng.sample(FIRST, 3)
    vendor = " ".join([first[0], first[1], rng.choice(LAST)])
    purchaser = " ".join([first[2], rng.choice(LAST)])
    place = rng.choice(PLACES)
    block = rng.choice(PARCEL_BLOCKS)
    parcel = f"{place}/{block}{'' if block.endswith('/') else ' '}{rng.randint(1, 99)}/{rng.randint(100, 9999)}"
    county = f"{place.title()} County"
    price = rng.randrange(40, 600) * 5000
    deposit = rng.choice([25000, 50000, 75000, 100000, 150000, 250000])
    caps = rng.random() < 0.5
    amount = (lambda n: num_words(n).upper()) if caps else num_words  # noqa: E731
    box = f"{rng.randint(10, 99999)}-{rng.choice(['00100', '00200', '40100', '80100'])}"
    day = f"{pay.day}{ORDINAL.get(pay.day, 'th')}"
    firm = rng.choice(FIRMS)
    p1 = (f"THIS AGREEMENT FOR SALE is made on the {words_date(date, rng)}{g('date|BETWEEN')}BETWEEN "
          f"{vendor}{g('vendor|of')}of Post Office Box{g('Box|digits')}{box} {rng.choice(PLACES).title()} "
          f"(the Vendor) and{g('and|purchaser')}{purchaser}, of {rng.choice(PLACES).title()} (the Purchaser).")
    p2 = (f"WHEREAS the Vendor is the registered proprietor of the parcel of land known as Title "
          f"Number{g('Number|parcel')}{parcel}{g('parcel|situated')}situated in{g('in|county')}{county}. "
          f"The Vendor has agreed to sell and the Purchaser has agreed to purchase the said land.")
    p3 = (f"The{g('The|purchase', 0.5)}purchase price shall be Kenya Shillings {amount(price)}"
          f"{g('amount|(')}(KES{g('KES|figures', 0.4)}{price:,}) of which a deposit of Kenya Shillings "
          f"{amount(deposit)}{g('deposit|only') if caps else ' '}only shall be paid on signing.")
    p4 = (f"The completion date shall be the{g('the|day', 0.5)}{day} day of {MONTH_NAMES[pay.month - 1]} "
          f"{pay.year}. The Vendor is acting through the firm of{g('of|firm')}{firm}.")
    truth = {
        "agreement_date": date.isoformat(), "vendor_name": vendor, "purchaser_name": purchaser,
        "parcel_number": parcel, "county": county, "purchase_price": price, "price_in_figures": price,
        "deposit": deposit, "completion_date": pay.isoformat(), "advocate": firm,
    }
    return "AGREEMENT FOR SALE OF LAND", [p1, p2, p3, p4], truth, glued


def make_glued_document(variant: str = "clean", seed: int = 0) -> SynthDoc:
    rng = random.Random(f"synth_glued/{variant}/{seed}")
    title, paragraphs, truth, glued = glued_for(random.Random(f"synth_glued/{seed}"))
    widths, spacing, degrade = PROSE_VARIANTS[variant]
    width = rng.randint(*widths) if widths else rng.randint(1000, 1300)
    img = layout_prose(title, paragraphs, rng, width, spacing)
    doc = SynthDoc("synth_glued", variant, seed, degrade(img, rng), truth)
    doc.glued = glued
    return doc


# ---------------------------------------------------------------------------- degradations

def to_jpeg(img: Image.Image, quality: int) -> Image.Image:
    buf = io.BytesIO()
    img.save(buf, "JPEG", quality=quality)
    buf.seek(0)
    return Image.open(buf).convert("RGB")


def rotate(img: Image.Image, degrees: float, fill=(250, 250, 250)) -> Image.Image:
    if degrees % 90 == 0:
        return img.rotate(degrees, expand=True)
    return img.rotate(degrees, expand=True, resample=Image.BICUBIC, fillcolor=fill)


def blur_lowres(img: Image.Image) -> Image.Image:
    w, h = img.size
    small = img.resize((int(w * 0.55), int(h * 0.55)), Image.BILINEAR)
    return to_jpeg(small.filter(ImageFilter.GaussianBlur(0.9)), 55)


def photo(img: Image.Image, rng: random.Random) -> Image.Image:
    """Lower contrast, uneven lighting and sensor noise, then JPEG."""
    import numpy as np

    arr = np.asarray(img).astype(float)
    h, w = arr.shape[:2]
    grad = np.linspace(0.75, 1.0, w)[None, :, None]
    arr = 60 + arr * 0.65 * grad
    arr += np.random.default_rng(rng.randint(0, 10**6)).normal(0, 6, arr.shape)
    return to_jpeg(Image.fromarray(np.clip(arr, 0, 255).astype("uint8")), 70)


# ---------------------------------------------------------------------------- variants

@dataclass
class SynthDoc:
    doc_type: str
    variant: str
    seed: int
    image: Image.Image
    truth: dict[str, object]  # field -> expected normalised value (None = not on the page)
    printed: dict[str, str] = field(default_factory=dict)  # field -> printed value
    labels: dict[str, str] = field(default_factory=dict)  # field -> printed label
    breaks: list[tuple[str, str]] = field(default_factory=list)  # hyphenated line breaks (typeset prose)
    glued: list[str] = field(default_factory=list)  # places printed without their space (synth_glued)

    @property
    def fields(self) -> list[str]:
        return list(self.truth)


def _base(doc_type: str, layout: str, rng: random.Random, style: Style, items_rng: random.Random):
    title, items, absent = items_for(doc_type, items_rng)
    if layout == "rows":
        img = layout_rows(title, items, style)
    elif layout == "colon":
        img = layout_rows(title, items, style, colon=True)
    elif layout == "two_column":
        img = layout_two_column(title, items, style)
    elif layout == "value_below":
        # The fixture form's address lines are too wide for the shipped types' 330 px cells:
        # they would touch the next cell's value, which OCR then reads as one box.
        if doc_type in FIXTURE_TYPES:
            img = layout_value_below(title, items, style, ncols=2, cell_w=480)
        else:
            img = layout_value_below(title, items, style)
    else:
        raise ValueError(layout)
    return img, items, absent


VARIANTS: dict[str, tuple[str, Callable[[Image.Image, random.Random], Image.Image]]] = {
    "clean": ("rows", lambda im, r: im),
    "colon": ("colon", lambda im, r: im),
    "two_column": ("two_column", lambda im, r: im),
    "value_below": ("value_below", lambda im, r: im),
    "rot90": ("rows", lambda im, r: rotate(im, 90)),
    "rot180": ("rows", lambda im, r: rotate(im, 180)),
    "rot270": ("value_below", lambda im, r: rotate(im, 270)),
    "skew": ("rows", lambda im, r: rotate(im, r.choice([-4, -2.5, 2.5, 4]))),
    "blur_lowres": ("rows", lambda im, r: blur_lowres(im)),
    "photo": ("two_column", lambda im, r: photo(rotate(im, r.choice([-1.5, 1.5])), r)),
}
DOC_TYPES = ["driving_license", "national_id", "logbook", "passport"]

# Made-up types that exercise `multiline` and `until`. They are test fixtures, not shipped
# presets: their YAML files are in tests/doctypes/, and `doc_type_ref` gives the path to pass to
# `extract`.
FIXTURE_DIR = Path(__file__).resolve().parent / "doctypes"
FIXTURE_TYPES = ["synth_form", "synth_agreement", "synth_contract", "synth_glued"]
PROSE_TYPES = ["synth_agreement", "synth_contract", "synth_glued"]
# Prose: (page width, line spacing) and the degradation. Form layouts do not apply.
PROSE_VARIANTS: dict[str, tuple[Optional[tuple[int, int]], float, Callable[[Image.Image, random.Random], Image.Image]]] = {
    "clean": (None, 1.4, lambda im, r: im),
    "narrow": ((620, 760), 1.4, lambda im, r: im),
    "spaced": (None, 2.0, lambda im, r: im),
    "rot90": (None, 1.4, lambda im, r: rotate(im, 90)),
    "rot180": (None, 1.4, lambda im, r: rotate(im, 180)),
    "skew": (None, 1.4, lambda im, r: rotate(im, r.choice([-4, -2.5, 2.5, 4]))),
    "blur_lowres": (None, 1.4, lambda im, r: blur_lowres(im)),
    "photo": (None, 1.4, lambda im, r: photo(rotate(im, r.choice([-1.5, 1.5])), r)),
}


def variants_for(doc_type: str) -> list[str]:
    return list(PROSE_VARIANTS) if doc_type in PROSE_TYPES else list(VARIANTS)


def doc_type_ref(doc_type: str):
    """What to pass to `extract` as the document type: a fixture's YAML path, or the name."""
    return FIXTURE_DIR / f"{doc_type}.yaml" if doc_type in FIXTURE_TYPES else doc_type


def make_prose_document(variant: str = "clean", seed: int = 0) -> SynthDoc:
    rng = random.Random(f"synth_agreement/{variant}/{seed}")
    content_rng = random.Random(f"synth_agreement/{seed}")
    title, paragraphs, truth = prose_for(content_rng)
    widths, spacing, degrade = PROSE_VARIANTS[variant]
    width = rng.randint(*widths) if widths else rng.randint(1000, 1300)
    img = degrade(layout_prose(title, paragraphs, rng, width, spacing), rng)
    return SynthDoc("synth_agreement", variant, seed, img, truth)


def make_document(doc_type: str, variant: str = "clean", seed: int = 0, style: Optional[Style] = None) -> SynthDoc:
    if doc_type == "synth_agreement":
        return make_prose_document(variant, seed)
    if doc_type == "synth_contract":
        return make_contract_document(variant, seed)
    if doc_type == "synth_glued":
        return make_glued_document(variant, seed)
    rng = random.Random(f"{doc_type}/{variant}/{seed}")
    items_rng = random.Random(f"{doc_type}/{seed}")
    if style is None:
        style = Style(
            label_font=rng.choice(["sans", "serif"]),
            value_font=rng.choice(["sans-bold", "mono", "serif-bold"]),
            label_size=rng.choice([18, 20, 22]),
            value_size=rng.choice([22, 24, 26]),
            upper_labels=rng.random() < 0.6,
            bg=rng.choice([(238, 242, 230), (245, 235, 238), (232, 238, 248), (255, 255, 255)]),
        )
    layout, degrade = VARIANTS[variant]
    img, items, absent = _base(doc_type, layout, rng, style, items_rng)
    img = degrade(img, rng)
    truth = {it.field: it.expected for it in items if it.field}
    for name in absent:
        truth[name] = None
    return SynthDoc(doc_type, variant, seed, img, truth,
                    printed={it.field: it.printed for it in items},
                    labels={it.field: it.label for it in items})


if __name__ == "__main__":  # pragma: no cover
    out = Path(sys.argv[1] if len(sys.argv) > 1 else "synth_out")
    out.mkdir(parents=True, exist_ok=True)
    for dtp in DOC_TYPES + FIXTURE_TYPES:
        for var in variants_for(dtp):
            doc = make_document(dtp, var, 0)
            doc.image.save(out / f"{dtp}_{var}.png")
    print(f"wrote samples to {out}")
