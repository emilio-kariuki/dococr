"""Turn raw text read off a document into a typed value.

The signature is part of the data contract (see docs/ARCHITECTURE.md). Every extractor's output goes through
`normalize_value`, so it must never raise and must be idempotent: normalising an
already-normalised value returns the same value (warnings about how the raw text was read,
such as "two-digit year interpreted", are not repeated).
"""

from __future__ import annotations

import datetime as _dt
import itertools
import re
from dataclasses import dataclass
from typing import Optional

from dococr.models import FieldSpec, FieldType, Value

# ---------------------------------------------------------------------------- helpers

_EDGE_JUNK = " \t\r\n:;,|_-–—=~*•·'\"`"
_MONTHS = {
    "jan": 1, "january": 1, "janvier": 1, "janv": 1,
    "feb": 2, "february": 2, "febr": 2, "fev": 2, "fevrier": 2, "février": 2,
    "mar": 3, "march": 3, "mars": 3,
    "apr": 4, "april": 4, "avr": 4, "avril": 4,
    "may": 5, "mai": 5,
    "jun": 6, "june": 6, "juin": 6,
    "jul": 7, "july": 7, "juil": 7, "juillet": 7,
    "aug": 8, "august": 8, "aou": 8, "aout": 8, "août": 8,
    "sep": 9, "sept": 9, "september": 9, "septembre": 9,
    "oct": 10, "october": 10, "octobre": 10,
    "nov": 11, "november": 11, "novembre": 11,
    "dec": 12, "december": 12, "decembre": 12, "décembre": 12,
}
# OCR reads these letters where a digit was printed.
_TO_DIGIT = str.maketrans({"O": "0", "o": "0", "Q": "0", "D": "0", "I": "1", "l": "1", "i": "1",
                           "|": "1", "!": "1", "L": "1", "S": "5", "s": "5", "B": "8", "Z": "2",
                           "z": "2", "G": "6", "b": "6", "g": "9", "q": "9", "T": "7"})
# For identifiers checked against a pattern: ambiguous characters and their look-alikes.
_LOOKALIKE = {"O": "0", "0": "O", "D": "0", "Q": "0", "I": "1", "1": "I", "L": "1",
              "S": "5", "5": "S", "B": "8", "8": "B", "Z": "2", "2": "Z", "G": "6", "6": "G"}


def _collapse(text: str) -> str:
    return " ".join(str(text).split())


def _strip_edges(text: str) -> str:
    """Collapse whitespace and trim separator junk OCR leaves around a value (': ABC 123.')."""
    text = _collapse(text).strip(_EDGE_JUNK)
    # A trailing full stop is noise unless it ends an abbreviation like "JR." -- drop it anyway
    # when the text is otherwise alphanumeric.
    if text.endswith(".") and not re.search(r"\b[A-Za-z]\.$", text):
        text = text.rstrip(".").rstrip(_EDGE_JUNK)
    return text


# ---------------------------------------------------------------------------- numbers in words
#
# English number words, as legal documents write amounts: "Eight Hundred and Fifty Thousand",
# "Twenty-Five", "a hundred", "One Million Two Hundred Thousand only", "... and Fifty Cents",
# "Twelve point Five". This is knowledge of written English, like month names; currency names
# and phrases such as "the sum of" are the author's, in the YAML (`label`, `until`).

_UNIT_WORDS = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6, "seven": 7,
    "eight": 8, "nine": 9, "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17, "eighteen": 18, "nineteen": 19,
}
_TEN_WORDS = {"twenty": 20, "thirty": 30, "forty": 40, "fourty": 40, "fifty": 50, "sixty": 60,
              "seventy": 70, "eighty": 80, "ninety": 90}
_SCALE_WORDS = {"thousand": 10**3, "million": 10**6, "billion": 10**9, "trillion": 10**12}
_ORDINAL_WORDS = {
    "first": 1, "second": 2, "third": 3, "fourth": 4, "fifth": 5, "sixth": 6, "seventh": 7,
    "eighth": 8, "ninth": 9, "tenth": 10, "eleventh": 11, "twelfth": 12, "thirteenth": 13,
    "fourteenth": 14, "fifteenth": 15, "sixteenth": 16, "seventeenth": 17, "eighteenth": 18,
    "nineteenth": 19, "twentieth": 20, "thirtieth": 30,
}
# Every word a number may be glued from when OCR drops a space ("FiftyThousand").
_GLUE_VOCAB = sorted({*_UNIT_WORDS, *_TEN_WORDS, *_SCALE_WORDS, "hundred", "and", "only"},
                     key=len, reverse=True)
# Long words that OCR may misread by one letter ("Thousaud"); short number words are not
# repaired, because one edit turns them into other words ("seventh", "fight").
_FUZZY_WORDS = ["hundred", "thousand", "million", "billion", "trillion"]


def number_word_forms() -> tuple[set[str], set[str]]:
    """English number words and month names written whole ("thousand", "fifth", "september"),
    and number words written with a hyphen inside ("twenty-five" -> "twentyfive", "twenty-first"), lower case without the
    hyphen. The matcher uses them to judge a word hyphenated across a line break."""
    whole = {*_UNIT_WORDS, *_TEN_WORDS, *_SCALE_WORDS, "hundred", *_ORDINAL_WORDS,
             *(m for m in _MONTHS if len(m) > 4 and m.isascii())}
    tens = [t for t in _TEN_WORDS if t != "fourty"]
    units = [u for u, v in _UNIT_WORDS.items() if 1 <= v <= 9]
    ords = [o for o, v in _ORDINAL_WORDS.items() if v <= 9]
    hyphenated = {t + u for t in tens for u in units} | {t + o for t in ("twenty", "thirty") for o in ords}
    return whole, hyphenated


def _edit1(a: str, b: str) -> bool:
    """True when a and b differ by at most one insertion, deletion or substitution."""
    if abs(len(a) - len(b)) > 1:
        return False
    if len(a) == len(b):
        return sum(x != y for x, y in zip(a, b)) <= 1
    if len(a) > len(b):
        a, b = b, a
    for i in range(len(b)):
        if b[:i] + b[i + 1:] == a:
            return True
    return False


def _segment(word: str) -> Optional[list[str]]:
    """Split a token OCR glued together into number words ("fiftythousand"), or None."""
    best: dict[int, list[str]] = {0: []}
    for i in range(len(word)):
        if i not in best:
            continue
        for v in _GLUE_VOCAB:
            if word.startswith(v, i) and (i + len(v) not in best
                                          or len(best[i + len(v)]) > len(best[i]) + 1):
                best[i + len(v)] = best[i] + [v]
    parts = best.get(len(word))
    return parts if parts and len(parts) >= 2 else None


def _word_kind(word: str) -> Optional[tuple[str, int]]:
    """(kind, value) of one lower-case number word: unit, ten, hundred, scale, and, a, only,
    point, cent; None for any other word."""
    if word in _UNIT_WORDS:
        return "unit", _UNIT_WORDS[word]
    if word in _TEN_WORDS:
        return "ten", _TEN_WORDS[word]
    if word == "hundred":
        return "hundred", 100
    if word in _SCALE_WORDS:
        return "scale", _SCALE_WORDS[word]
    if word in ("and", "a", "only", "point"):
        return word, 0
    if word in ("cent", "cents"):
        return "cent", 0
    return None


_NUMBER_KINDS = ("unit", "ten", "hundred", "scale")


def _number_tokens(text: str) -> tuple[list[tuple[str, int, int, int, str]], list[str]]:
    """Tokens of `text` as (kind, value, start, end, gap_before) with kind None for other words
    and "fig" for figures; and warnings for words repaired from OCR damage. `gap_before` is
    the text between this token and the previous one."""
    out: list[tuple] = []
    warnings: list[str] = []
    prev_end = 0
    matches = [(m.group(0), m.start(), m.end()) for m in re.finditer(r"[A-Za-z]+|\d+(?:[,.' ]\d+)*", text)]
    # A number word broken by a hyphen ("MIL-LION", "thou- sand"): the hyphen is not part of it.
    k = 0
    while k + 1 < len(matches):
        (a, s1, _), (b, _, e2) = matches[k], matches[k + 1]
        if (a.isalpha() and b.isalpha() and re.fullmatch(r"[\-\u2010\u2011\u2013]\s*", text[matches[k][2]:matches[k + 1][1]])
                and not _word_kind(a.lower()) and (_word_kind((a + b).lower()) or _segment((a + b).lower()))):
            matches[k:k + 2] = [(a + b, s1, e2)]
        k += 1
    for tok, s, e in matches:
        gap = text[prev_end:s]
        prev_end = e
        if tok[0].isdigit():
            out.append(("fig", 0, s, e, gap))
            continue
        low = tok.lower()
        kind = _word_kind(low)
        if kind:
            out.append((kind[0], kind[1], s, e, gap))
            continue
        parts = _segment(low)
        if parts:
            for k, part in enumerate(parts):
                pk = _word_kind(part)
                out.append((pk[0], pk[1], s, e, gap if k == 0 else ""))
            continue
        # OCR glued a word to a number word written in another case ("ShillingsONE")
        pieces = [(x.group(0), s + x.start(), s + x.end())
                  for x in re.finditer(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+", tok)]
        if len(pieces) > 1 and any(_word_kind(x[0].lower()) for x in pieces):
            for k, (x, xs, xe) in enumerate(pieces):
                pk = _word_kind(x.lower())
                out.append((pk[0] if pk else None, pk[1] if pk else 0, xs, xe, gap if k == 0 else ""))
            continue
        near = [w for w in _FUZZY_WORDS if len(low) >= 6 and _edit1(low, w)]
        if len(near) == 1:
            pk = _word_kind(near[0])
            warnings.append(f"read '{tok}' as '{near[0]}'")
            out.append((pk[0], pk[1], s, e, gap))
            continue
        out.append((None, 0, s, e, gap))
    return out, warnings


def _parse_cardinal(seq: list[tuple[str, int]]) -> Optional[int]:
    """Value of a run of number words in English grammar, or None when the words do not form
    one number ("five twenty", "thousand million", "hundred")."""
    total = 0
    cur = 0  # the group below the last scale word
    has_h = has_t = has_u = False
    last_scale = None
    pending_a = False
    for i, (kind, val) in enumerate(seq):
        nxt = seq[i + 1][0] if i + 1 < len(seq) else None
        if kind == "a":
            if cur or has_h or nxt not in ("hundred", "scale"):
                return None
            pending_a = True
            continue
        if kind == "and":
            prev = seq[i - 1][0] if i else None
            if prev not in ("hundred", "scale") or nxt not in ("unit", "ten", "a"):
                return None
            continue
        if kind == "unit":
            if has_u or (has_t and (val >= 10 or val == 0)) or (val == 0 and (cur or total or len(seq) > 1)):
                return None
            cur += val
            has_u = True
        elif kind == "ten":
            if has_t or has_u:
                return None
            cur += val
            has_t = True
        elif kind == "hundred":
            if has_h:
                return None
            mult = cur
            if pending_a and not cur:
                mult = 1
            if not 1 <= mult <= 99:
                return None
            cur = mult * 100
            has_h, has_t, has_u = True, False, False
        elif kind == "scale":
            if last_scale is not None and val >= last_scale:
                return None
            if pending_a and not cur:
                cur = 1
            if not cur:
                return None
            total += cur * val
            last_scale = val
            cur = 0
            has_h = has_t = has_u = False
        else:
            return None
        pending_a = False
    if pending_a:
        return None
    return total + cur


@dataclass
class WordNumber:
    """A number written in words, found by `find_number_words`."""
    value: Optional[float]  # None when number words are present but do not form a number
    start: int
    end: int
    count: int  # how many separate numbers in words the text holds
    warnings: list[str]


def find_number_words(text: str) -> Optional[WordNumber]:
    """The first number written in English words in `text`, or None when there is none.

    A single number word inside other words ("one of the vehicles", "for ten days") is not
    a number: it must be the whole text (apart from "only" and figures), or the run must have
    two or more number words or a scale word. Number words that are adjacent but do not form
    one number ("Five Twenty") give a WordNumber whose value is None."""
    tokens, warnings = _number_tokens(text)
    words_total = sum(1 for t in tokens if t[0] is None)
    runs: list[tuple[int, int]] = []  # token index ranges of number-word runs
    i = 0
    while i < len(tokens):
        kind = tokens[i][0]
        if kind in _NUMBER_KINDS or (kind == "a" and i + 1 < len(tokens)
                                     and tokens[i + 1][0] in ("hundred", "scale")):
            j = i + 1
            while j < len(tokens) and tokens[j][0] in (*_NUMBER_KINDS, "and", "a") and re.fullmatch(
                    r"[\s,\-–—]*", tokens[j][4]):
                j += 1
            while j > i and tokens[j - 1][0] in ("and", "a"):
                j -= 1
            if j == i:  # only "a", with its scale word cut off ("a (hundred")
                i += 1
                continue
            runs.append((i, j))
            i = j
        else:
            i += 1
    accepted = []
    for s, e in runs:
        n_words = sum(1 for t in tokens[s:e] if t[0] in _NUMBER_KINDS)
        strong = (n_words >= 2 or any(t[0] in ("hundred", "scale") for t in tokens[s:e])
                  or (e < len(tokens) and tokens[e][0] in ("point", "cent")))
        if strong or (words_total == 0 and len(runs) == 1):
            accepted.append((s, e))
    if not accepted:
        return None
    s, e = accepted[0]
    value, end = _run_value(tokens, s, e)
    count = 1 + sum(1 for s2, _ in accepted[1:] if tokens[s2][2] >= end)
    # A number word left out of the number read may be part of it, cut off by OCR damage
    # ("TWO MlL LION SIX HUNDRED THOUSAND"): say so rather than return a part silently.
    stray = [text[tokens[a][2]:tokens[b - 1][3]] for a, b in runs
             if (a, b) not in accepted and not (tokens[s][2] <= tokens[a][2] < end)]
    if stray:
        warnings.append("number words outside the number read were ignored: "
                        + ", ".join(f"'{x}'" for x in stray))
    return WordNumber(value, tokens[s][2], end, count, warnings)


def _run_value(tokens: list[tuple], s: int, e: int) -> tuple[Optional[float], int]:
    """Value of the number-word run tokens[s:e] with any fractional part after it, and the
    character offset where the number ends."""
    kinds = [(t[0], t[1]) for t in tokens]
    end = tokens[e - 1][3]
    nxt = tokens[e] if e < len(tokens) else None
    if nxt is not None and nxt[0] == "cent":
        # "... and Fifty Cents", "Fifty Cents": the last group is hundredths
        ands = [k for k in range(s, e) if kinds[k][0] == "and"]
        cut = ands[-1] if ands else s - 1
        whole = _parse_cardinal(kinds[s:cut]) if ands else 0
        cents = _parse_cardinal(kinds[cut + 1:e])
        if whole is None or cents is None or not 1 <= cents <= 99:
            return None, nxt[3]
        return whole + cents / 100, nxt[3]
    value = _parse_cardinal(kinds[s:e])
    if value is None:
        return None, end
    rest = tokens[e:]
    if len(rest) >= 2 and rest[0][0] == "point" and rest[1][0] == "unit" and rest[1][1] <= 9:
        # "... point five": decimal digits as words
        digits = ""
        k = 1
        while k < len(rest) and rest[k][0] == "unit" and rest[k][1] <= 9:
            digits += str(rest[k][1])
            k += 1
        return float(f"{value}.{digits}"), rest[k - 1][3]
    if len(rest) >= 4 and rest[0][0] is None and rest[1][0] == "and":
        # "... Shillings and Fifty Cents": one word between the whole part and the hundredths
        # is skipped, not interpreted
        m = 2
        while m < len(rest) and rest[m][0] in ("unit", "ten"):
            m += 1
        cents = _parse_cardinal([(t[0], t[1]) for t in rest[2:m]]) if m > 2 else None
        if cents is not None and 1 <= cents <= 99 and m < len(rest) and rest[m][0] == "cent":
            return value + cents / 100, rest[m][3]
    return value, end


def _fmt_number(x: float) -> str:
    return f"{int(x):,}" if x == int(x) else f"{x:,.2f}"


# ---------------------------------------------------------------------------- dates

def _year4(year: int, warnings: list[str]) -> int:
    if year >= 100:
        return year
    pivot = (_dt.date.today().year % 100) + 20
    warnings.append("two-digit year interpreted")
    return 2000 + year if year <= pivot else 1900 + year


def _make_date(y: int, m: int, d: int) -> Optional[_dt.date]:
    if not (1900 <= y <= 2100):
        return None
    try:
        return _dt.date(y, m, d)
    except ValueError:
        return None


def _digits(token: str) -> str:
    return token.translate(_TO_DIGIT)


_NUM_DATE = re.compile(r"(?<!\d)(\d{1,4})\s*([./\-\s])\s*(\d{1,2})\s*\2\s*(\d{1,4})(?!\d)")
_NAME_DATE = re.compile(
    r"(?<![A-Za-z0-9])(\d{1,2})\s*[./\-\s]?\s*([A-Za-zéû][A-Za-zéû0]{1,8}|0[A-Za-zéû]{2,8})\.?\s*[./\-\s,]?\s*(\d{2,4})(?!\d)")
_MONTH_FIX = str.maketrans({"0": "o", "1": "i", "5": "s", "8": "b"})
_NAME_DATE_US = re.compile(
    r"(?<![A-Za-z])([A-Za-zéû]{3,9})\.?\s+(\d{1,2})(?:st|nd|rd|th)?,?\s+(\d{4})(?!\d)", re.IGNORECASE)
# Dates as running text writes them: an ordinal day and/or "day of" before the month name
# ("5th March 2024", "5th day of March, 2024", "the 1st of June 2020", "1er mars 2024").
_ORDINAL_DATE = re.compile(
    r"(?<![A-Za-z0-9])(\d{1,2})\s?(?:(?:st|nd|rd|th|er)\.?\s+(?:day\s+of\s+(?:the\s+month\s+of\s+)?|of\s+)?"
    r"|\s+(?:day\s+of\s+(?:the\s+month\s+of\s+)?|of\s+))"
    r"([A-Za-zéû]{3,9})\.?,?\s+(\d{4})(?!\d)", re.IGNORECASE)
_COMPACT = re.compile(r"(?<!\d)(\d{8})(?!\d)")
# Dates with the day and/or the year in words ("the Fifth day of March Two Thousand and
# Twenty-Four", "5th day of March, Two Thousand and Nineteen", "Twenty-First day of June 2019",
# "March Fifth, Nineteen Ninety-Nine"). The words are validated by the number parser.
_DAY_UNITS = "first|second|third|fourth|fifth|sixth|seventh|eighth|ninth"
_DAY_WORDS = (rf"(?:(?:twenty|thirty)[\s\-–]*(?:{_DAY_UNITS})|{_DAY_UNITS}|tenth|eleventh|twelfth|"
              r"thirteenth|fourteenth|fifteenth|sixteenth|seventeenth|eighteenth|nineteenth|"
              r"twentieth|thirtieth)")
_CARDINAL = "|".join(sorted({*_UNIT_WORDS, *_TEN_WORDS, "hundred", "thousand"}, key=len, reverse=True))
_YEAR_WORDS = rf"(?:(?:{_CARDINAL})(?:(?:[\s\-–]*|\s+and\s*)(?:{_CARDINAL}))*)"  # OCR may drop spaces
_DAY = rf"(?P<dw>{_DAY_WORDS})|(?P<dn>\d{{1,2}})(?:\s?(?:st|nd|rd|th))?"
_YEAR = rf"(?P<yw>{_YEAR_WORDS})|(?P<yn>\d{{4}})"
_WORD_DATE = re.compile(
    rf"(?<![A-Za-z0-9])(?:{_DAY})\s+(?:day\s+of\s+(?:the\s+month\s+of\s+)?|of\s+)?"
    rf"(?P<mon>[A-Za-zéû]{{3,9}})\.?,?\s+(?:(?:in\s+)?the\s+year\s+)?(?:{_YEAR})(?![A-Za-z0-9])",
    re.IGNORECASE)
_WORD_DATE_US = re.compile(
    rf"(?<![A-Za-z0-9])(?P<mon>[A-Za-zéû]{{3,9}})\.?\s+(?:the\s+)?(?:{_DAY}),?\s+"
    rf"(?:(?:in\s+)?the\s+year\s+)?(?:{_YEAR})(?![A-Za-z0-9])", re.IGNORECASE)


def _year_from_words(text: str) -> Optional[int]:
    """A year in words: a cardinal ("Two Thousand and Twenty-Four", "Nineteen Hundred and
    Ninety") or the paired form ("Nineteen Ninety-Nine", "Twenty Twenty-Four")."""
    tokens, _ = _number_tokens(text)
    if any(t[0] not in (*_NUMBER_KINDS, "and") for t in tokens):
        return None
    seq = [(t[0], t[1]) for t in tokens]
    year = _parse_cardinal(seq)
    if year is None and len(seq) >= 2 and seq[0][0] in ("unit", "ten") and 11 <= seq[0][1] <= 20:
        rest = _parse_cardinal(seq[1:])
        if rest is not None and 10 <= rest <= 99:
            year = seq[0][1] * 100 + rest
    return year


def _day_from_words(text: str) -> Optional[int]:
    parts = [p for p in re.split(r"[\s\-–]+", text.lower()) if p]
    if len(parts) == 1:
        return _ORDINAL_WORDS.get(parts[0])
    if len(parts) == 2 and parts[0] in ("twenty", "thirty") and parts[1] in _ORDINAL_WORDS:
        return _TEN_WORDS[parts[0]] + _ORDINAL_WORDS[parts[1]]
    return None


def _word_dates(text: str) -> list[tuple[int, int, _dt.date, list[str]]]:
    out = []
    for rx in (_WORD_DATE, _WORD_DATE_US):
        for m in rx.finditer(text):
            if not (m.group("dw") or m.group("yw")):
                continue  # all figures: the other patterns read it
            day = _day_from_words(m.group("dw")) if m.group("dw") else int(m.group("dn"))
            year = _year_from_words(m.group("yw")) if m.group("yw") else int(m.group("yn"))
            month = _MONTHS.get(m.group("mon").lower().translate(_MONTH_FIX))
            date = _make_date(year, month, day) if day and year and month else None
            if date:
                out.append((m.start(), m.end(), date, []))
    return out


def find_date(text: str, month_first: bool = False) -> Optional[tuple[_dt.date, int, int, list[str]]]:
    """Find the first date in `text`. Returns (date, start, end, warnings) or None.

    Numeric dates are read day-first (12/03/1990 = 12 March) unless the first group is a
    four-digit year (1990-03-12) or the day-first reading is impossible (03/25/1990, with a
    warning). With `month_first` (the YAML key `date_order: month_first`) they are read
    month-first (12/03/1990 = 3 December), turning to day-first, with a warning, only when the
    month-first reading is impossible; a compact 8-digit date is then tried as YYYYMMDD and
    MMDDYYYY. Dates with a month name are not affected.
    """
    candidates = []
    # Repair digit look-alikes inside things that are shaped like dates, e.g. "l2.O3.1990".
    fixed = re.sub(r"[0-9OoQDIl|SsBZ]{1,4}(?:\s*[./\-]\s*[0-9OoQDIl|SsBZ]{1,4}){2}",
                   lambda m: _digits(m.group(0)), text)
    for m in _NUM_DATE.finditer(fixed):
        a, _, b, c = m.groups()
        warnings: list[str] = []
        if len(a) == 4:  # year first: 1990-03-12
            date = _make_date(int(a), int(b), int(c)) if len(c) <= 2 else None
        elif len(a) <= 2:
            if len(c) not in (2, 4):
                continue
            year = _year4(int(c), warnings)
            day, month = (int(b), int(a)) if month_first else (int(a), int(b))
            date = _make_date(year, month, day)
            if date is None:
                swapped = _make_date(year, day, month)
                if swapped is not None:
                    warnings.append("read as day-first because month-first is impossible" if month_first
                                    else "read as month-first because day-first is impossible")
                    date = swapped
        else:
            continue
        if date:
            candidates.append((m.start(), m.end(), date, warnings))
    for m in _NAME_DATE.finditer(text):
        day, mon, year = m.groups()
        month = _MONTHS.get(mon.lower().translate(_MONTH_FIX))
        if month is None:
            continue
        warnings = []
        date = _make_date(_year4(int(year), warnings), month, int(day)) if len(year) != 3 else None
        if date:
            candidates.append((m.start(), m.end(), date, warnings))
    for m in _NAME_DATE_US.finditer(text):
        mon, day, year = m.groups()
        month = _MONTHS.get(mon.lower())
        date = _make_date(int(year), month, int(day)) if month else None
        if date:
            candidates.append((m.start(), m.end(), date, []))
    for m in _ORDINAL_DATE.finditer(text):
        day, mon, year = m.groups()
        month = _MONTHS.get(mon.lower().translate(_MONTH_FIX))
        date = _make_date(int(year), month, int(day)) if month else None
        if date:
            candidates.append((m.start(), m.end(), date, []))
    if re.search(_DAY_WORDS + "|" + _CARDINAL, text, re.IGNORECASE):
        candidates += _word_dates(text)
    if not candidates:
        for m in _COMPACT.finditer(fixed):
            s = m.group(1)
            date = _make_date(int(s[:4]), int(s[4:6]), int(s[6:])) or (
                _make_date(int(s[4:]), int(s[:2]), int(s[2:4])) if month_first
                else _make_date(int(s[4:]), int(s[2:4]), int(s[:2])))
            if date:
                candidates.append((m.start(), m.end(), date, []))
    if not candidates:
        return None
    start, end, date, warnings = min(candidates, key=lambda c: c[0])
    return date, start, end, warnings


def _normalize_date(raw: str, month_first: bool = False) -> tuple[Value, list[str]]:
    text = _strip_edges(raw)
    if re.fullmatch(r"\d{4}-\d{2}-\d{2}", text) and _make_date(*map(int, text.split("-"))):
        return text, []  # already ISO: idempotent fast path
    found = find_date(text, month_first)
    if found is None:
        return None, ["could not parse date"]
    date, start, end, warnings = found
    rest = (text[:start] + " " + text[end:]).strip(_EDGE_JUNK + ". ")
    if re.search(r"[A-Za-z0-9]", rest):
        warnings = [*warnings, "extra text around the date was ignored"]
    return date.isoformat(), warnings


# ---------------------------------------------------------------------------- numbers

# A number written with both "," and "." where every group after the first has three digits.
_MIXED_GROUPS = re.compile(r"[-+]?\d{1,3}(?:[.,]\d{3}){2,}")


def _has_mixed_groups(text: str) -> bool:
    for m in re.finditer(r"\d[\d,.]*\d", text):
        tok = m.group(0)
        if "," in tok and "." in tok and _MIXED_GROUPS.fullmatch(tok):
            return True
    return False


def _irregular_separators(text: str) -> bool:
    """True when the first number in `text` has separators that do not form a regular number:
    "9.492157.75", "3,760,12.00", "7,532936", "93.370.00". Reading such a number at all is a
    guess, and a misread separator changes its size by orders of magnitude, so the caller warns.

    Regular means: groups of three digits divided by one kind of separator, then at most one
    decimal part behind a different separator; or a single '.' or ',' decimal. A comma before
    more than three final digits is not a decimal anyone writes."""
    m = re.search(r"\d[\d,.' ]*\d", text)
    if not m:
        return False
    token = re.sub(r"(?<=[,.']) +", "", m.group(0).strip())  # OCR often adds a space after a mark
    parts = re.split(r"([,.' ])", token)
    digits, seps = parts[0::2], parts[1::2]
    if not any(sep in ",.'" for sep in seps):
        return False
    if any(not d for d in digits):
        return True
    if len(seps) == 1:
        if seps[0] == ".":
            return False
        return len(digits[1]) > 3 if seps[0] == "," else len(digits[1]) != 3
    decimal = seps[-1] if seps[-1] in ".," and seps[-1] not in seps[:-1] else None
    group_seps = seps[:-1] if decimal else seps
    groups = digits[1:-1] if decimal else digits[1:]
    if len(set(group_seps)) != 1 or len(digits[0]) > 3:
        return True
    return any(len(g) != 3 for g in groups)


def _parse_number(text: str) -> Optional[tuple[float, int, bool]]:
    """Return (number, count of numbers found, used look-alike repair) or None."""
    repaired = False
    # Repair look-alikes only inside tokens that are already mostly digits ("1O00" -> "1000").
    def fix(m: re.Match) -> str:
        nonlocal repaired
        tok = m.group(0)
        if sum(ch.isdigit() for ch in tok) >= max(1, len(tok) // 2) and not tok.isdigit():
            repaired = True
            return _digits(tok)
        return tok

    text = re.sub(r"[0-9OoIlSB]+", fix, text)
    # A sign counts only at the start of a number, not after a letter or digit: the hyphen in
    # "ROOM-5" is not a minus.
    nums = re.findall(r"(?<![A-Za-z0-9])[-+]?\d[\d,.' ]*\d|(?<![A-Za-z0-9])[-+]?\d|\d[\d,.' ]*\d|\d", text)
    nums = [n.strip() for n in nums if n.strip()]
    if not nums:
        return None
    tok = nums[0].replace(" ", "").replace("'", "")
    if _MIXED_GROUPS.fullmatch(tok):
        # "1.435,000": both marks used and every group after the first has three digits. That is
        # thousands grouping with one mark misread, far more often than a number with exactly
        # three decimals. `_normalize_number` adds the warning.
        tok = tok.replace(",", "").replace(".", "")
    elif "," in tok and "." in tok:
        dec = "." if tok.rfind(".") > tok.rfind(",") else ","
        thou = "," if dec == "." else "."
        tok = tok.replace(thou, "").replace(dec, ".")
    elif "," in tok:
        parts = tok.split(",")
        tok = tok.replace(",", "") if all(len(p) == 3 for p in parts[1:]) else tok.replace(",", ".")
    elif tok.count(".") > 1:
        tok = tok.replace(".", "")
    try:
        return float(tok), len(nums), repaired
    except ValueError:
        return None


def _normalize_number(raw: str, integer: bool) -> tuple[Value, list[str]]:
    text = _strip_edges(raw)
    parsed = _parse_number(text)
    words = find_number_words(text)
    if words is not None:
        return _number_from_words(text, words, parsed, integer)
    if parsed is None:
        return None, ["no number found"]
    number, count, repaired = parsed
    warnings = []
    if count > 1:
        warnings.append("several numbers found; used the first")
    if repaired:
        warnings.append("corrected letters misread as digits")
    if _has_mixed_groups(text):
        warnings.append("both '.' and ',' separate groups of three digits; read as thousands")
    elif _irregular_separators(text):
        warnings.append("the separators in the number are irregular (a misread ',' or '.' changes "
                        "its size); check the value")
    if integer:
        if number != int(number):
            warnings.append("value is not a whole number; rounded")
        return int(round(number)), warnings
    return (int(number) if number == int(number) and "." not in text and "," not in text
            else number), warnings


def _number_from_words(text: str, words: WordNumber, figures: Optional[tuple[float, int, bool]],
                       integer: bool) -> tuple[Value, list[str]]:
    """A number written in words, checked against the figures when the text has both."""
    warnings = list(words.warnings)
    if words.value is None:
        if figures is None:
            return None, [*warnings, "could not read the number in words"]
        number = figures[0]
        warnings.append("could not read the number in words; used the figures")
    else:
        number = words.value
        if words.count > 1:
            warnings.append("several numbers found; used the first")
        if figures is not None and abs(figures[0] - number) > 1e-9:
            warnings.append(f"the number in words ({_fmt_number(number)}) and the number in figures "
                            f"({_fmt_number(figures[0])}) disagree; used the words")
    if integer:
        if number != int(number):
            warnings.append("value is not a whole number; rounded")
        return int(round(number)), warnings
    return (int(number) if number == int(number) else number), warnings


# ---------------------------------------------------------------------------- identifiers

def _fix_with_pattern(value: str, regex: re.Pattern) -> Optional[tuple[str, int]]:
    """Swap look-alike characters (O/0, I/1, S/5, ...) until the value fully matches."""
    positions = [i for i, ch in enumerate(value) if ch in _LOOKALIKE]
    if not positions or len(positions) > 12:
        return None
    best = None
    for k in range(1, min(len(positions), 4) + 1):
        for combo in itertools.combinations(positions, k):
            chars = list(value)
            for i in combo:
                chars[i] = _LOOKALIKE[chars[i]]
            cand = "".join(chars)
            if regex.fullmatch(cand):
                best = (cand, k)
                break
        if best:
            break
    return best


def _normalize_identifier(raw: str, spec: FieldSpec) -> tuple[Value, list[str]]:
    text = _strip_edges(raw).upper()
    value = re.sub(r"\s+", "", text)
    return value, []


# ---------------------------------------------------------------------------- names / strings

def _normalize_name(raw: str) -> tuple[Value, list[str]]:
    text = _strip_edges(raw)
    text = re.sub(r"\s*,\s*", ", ", text)
    warnings = ["name contains digits"] if re.search(r"\d", text) else []
    return text, warnings


# ---------------------------------------------------------------------------- entry point

def _compile(pattern: str) -> Optional[re.Pattern]:
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return None


MAX_TYPED_LENGTH = 2000  # characters; a date or number is never longer, and parsing is not free


def normalize_value(raw: str, spec: FieldSpec) -> tuple[Value, list[str]]:
    """Return (value, warnings). Must never raise, and must be idempotent:
    normalising an already-normalised value returns it unchanged."""
    try:
        if raw is None:
            return None, []
        if not isinstance(raw, str):
            raw = str(raw)
        kind = spec.type
        if len(raw) > MAX_TYPED_LENGTH and kind in (FieldType.DATE, FieldType.INTEGER, FieldType.NUMBER):
            return None, [f"value is {len(raw)} characters long, too long to read as a {kind.value}"]
        if kind == FieldType.DATE:
            value, warnings = _normalize_date(raw, spec.month_first())
        elif kind == FieldType.INTEGER:
            value, warnings = _normalize_number(raw, integer=True)
        elif kind == FieldType.NUMBER:
            value, warnings = _normalize_number(raw, integer=False)
        elif kind == FieldType.IDENTIFIER:
            value, warnings = _normalize_identifier(raw, spec)
        elif kind == FieldType.NAME:
            value, warnings = _normalize_name(raw)
        else:
            value, warnings = _strip_edges(raw), []
        if value == "":
            return None, [*warnings, "empty value"]
        if value is None:
            # A typed field whose text does not parse has no value: callers treat `value` as
            # typed. The text stays in the field's `raw`, and the warning says why.
            return None, warnings

        if spec.pattern:
            regex = _compile(spec.pattern)
            if regex is None:
                warnings.append("field pattern is not a valid regular expression")
            elif not regex.fullmatch(str(value)):
                fixed = None
                if kind in (FieldType.IDENTIFIER, FieldType.STRING) and isinstance(value, str):
                    fixed = _fix_with_pattern(value, regex)
                if fixed:
                    value = fixed[0]
                    warnings.append("corrected look-alike characters to fit the expected format")
                else:
                    warnings.append("value does not match the expected format")
        return value, warnings
    except Exception as exc:  # never raise: degrade to the cleaned text
        try:
            return _collapse(raw) or None, [f"normalisation failed: {type(exc).__name__}"]
        except Exception:
            return None, ["normalisation failed"]
