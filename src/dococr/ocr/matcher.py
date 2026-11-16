"""Rule-based field matcher: OCR lines + field specs -> raw value per field.

Document-agnostic: every label, label variant, stop label and value format comes from the
document type (`DocType`, read from its YAML file) and the requested `FieldSpec`s. The code
knows only how OCR misreads characters, how dates are written in general, and geometry.

How it works
------------
1. Every OCR line is split into words with estimated x-spans (proportional to character
   offsets); ':' becomes its own separator token, except between two figures ("9:00",
   "10:30"), where it stays inside the word, so a time is one word in form and sentence
   values alike. A word that glues a long label to its value
   ("DateofBirth12.03.1990") is split. When the type has `until` fields, so is a word that
   glues the last word of their labels, or the first word of their end texts, to a value
   containing a digit ("numberJSX956", "123Achassis"). Other glued prose anchors are matched
   inside the word without splitting it here (see "Glued anchors" in step 4). So is a word
   that glues the last word of a label of two or more words to a value starting with a figure,
   after a full stop or, for a word of five letters or more, directly ("No.687860/36",
   "Registration12.03.2001") at the start of an OCR box: the last line of a stacked label read
   in one box with its value (`_split_label_tail`). A run of two or
   more leader dots or rules ("......KIAMBU......", "..Province") is a separator token of its
   own, so it does not hide a label or end text and is not part of a value.
2. Lines are grouped into rows by vertical overlap; each row's words are sorted left to right.
   A line that overlaps a member of a row horizontally while its centre is offset vertically
   (STACK_OFFSET) sits under or over it, not beside it, and is not put in that row: the two
   lines of a label in a table cell are often closer than one text height (`_stacked`).
3. Label search. The vocabulary is the `label` and `aliases` of every field of the doc type
   and of the requested (possibly ad-hoc) fields, plus the doc type's `other_labels`, which
   never produce values and serve only as competitors and stop boundaries. No other label
   variants are generated: "No" and "Number" are different labels unless the YAML lists both.
   Each contiguous word span of a row (which may cross OCR boxes that sit close together) is
   compared with each label:
     - word by word, with an edit budget that depends on the word's length
       (<=3 chars: exact; 4-7: one edit; longer: two),
     - single-word labels of <=5 chars must match exactly and must start an OCR box,
     - or as a joined string when OCR split/merged words ("LicenceNo"), with at most one edit,
       which may not be a whole word of one letter added or dropped ("A Price per Share" is not
       "Price per Share", so the "A" of "Series A" before a label or end text stays in the value).
   Stacked labels (`find_stacked_hits`): a label of two or more words may be printed on two
   or three short lines of one cell ("Date of" above "Birth"). Its words are the whole of each
   OCR box but the last and the start of the last box (the value may follow in that box). Each
   line is directly below the one before (centres 0.5 to STACK_PITCH text heights apart, at
   most STACK_GAP of blank between, similar text height) and aligned with it on the left, the
   right or the centre (STACK_ALIGN). A box with other text after the label's words cannot be
   an upper line, so "Name: JOHN" above "Surname: DOE" is not "Name Surname", and lines of
   running text are not joined.
   Overlapping hits are resolved by covered label length x quality, so a longer label beats
   a shorter one inside it ("National ID No" beats "ID No"), also across the lines of a stacked
   label: "Name" cannot take a word of "Name of / Registering Officer". A stacked label each of
   whose lines has text of its own to the right is two rows of the form ("BODY  SALOON" above
   "COLOUR  WHITE" while "Body Colour" is a label): it is dropped and the overlaps resolved
   again (`_rows_of_their_own`), and the field is told why if it ends without a value. A
   stacked label whose lines are each a label on their own, one of another field, is kept
   with a warning ("Entry" above "No.", or "Body" above "Colour": the page cannot tell).
   A single-line hit followed in its own OCR box by the next word of a longer label that
   starts with it ("Name" in a box "Name of", when "Name of Registering Officer" is a label)
   is a piece of that label (`_longer_label`): it still bounds other values but gives none.
4. For each hit, value candidates are the words to the right in the same row (up to the next
   label hit, a ':' or a wide gap) and the nearest line directly below (within the label's
   column, up to the next label). For a stacked label: the words to the right of each of its
   lines (the value may be level with any line, or between two, which puts it on the row of
   one of them), and below its last line in a column as wide as the label (`_value_places`).
   Candidates are scored by
   OCR confidence x label quality x geometry x type plausibility (the field's declared `type`:
   dates must parse, numbers need digits; and its `pattern` must match) and trimmed to the
   date / pattern match when one exists. A field's type is only ever its declared `type`.
   Fields with `multiline: true` extend each candidate with the lines below it that belong to
   the same block (`_block_lines`): a line that starts under the first line's left edge, with
   text of the same height, inside the label's column, with nothing under the label (for a
   value right of its label), and not another label. A line that starts back at the label's
   margin under a value to the right of its label is the value wrapping (`_wrap_back`) when it
   holds no label and no colon and the value plainly goes on: the line above ends with a comma,
   ';', '&', a hyphen, an opening bracket or a word such as "and" / "of" (`_open_end`), or the
   line starts in lower case. Such a line without that evidence is not joined, with a warning;
   so is a value that ends mid-phrase above the next label. The block ends at a blank gap
   taller than BLOCK_GAP lines, or after MAX_BLOCK_LINES lines with a warning. Confidence is
   that of the weakest line. A field without `multiline` whose one-line value is visibly cut
   (it ends mid-phrase, or the next line would join it with that evidence) gets a warning
   (`_cut_short`); its value is unchanged.
   Fields with `until` (prose) do not use steps 3 and 4. Their labels are not row labels;
   they are searched in reading order (`Layout`, `find_flow_hits`): each row is split into
   segments at wide gaps, and the line after a segment is the nearest one below that overlaps
   it horizontally, so a phrase may cross a line break and a value runs on from the end of a
   line to the start of the next. Short words match exactly but need not start an OCR box.
   A word without letters does not start an anchor whose first word has letters ("7 situated
   in" is not "situated in").
   Fill-in lines ("Birth in the ....KIAMBU.... District in the ....CENTRAL.... Province"): when
   the label ends its segment and a word end text starts a later segment on the same row, with
   at least one word between, the value is the segments between them (`_along_row`). This is
   tried first; otherwise the field is read in sentence order as below.
   The value is the text after the label up to the first match of an `until` text, which
   must come within MAX_FLOW_WORDS words and MAX_FLOW_LINES lines (else no value, except that
   a date / `pattern` field accepts a match that starts right after the label, with reduced
   confidence). A reading that contains a distinctive anchor (two or more words, or six or
   more letters) of another field or of `other_labels` ran on past its end and is dropped.
   When several hits of a repeated label reach the same end, the nearest is the label if the
   longer readings from the earlier hits contain an anchor of anything else (they ran on from
   an unrelated occurrence); a longer reading with no such anchor is another possible value
   ("FOR" in "AGREEMENT FOR / ROTOR TESTING FOR LOW NOISE"), and the field gets no value, with a
   warning. Different readings at different ends also make the field ambiguous. A field's
   label found only where another field's longer label also matches (its label is the start
   of theirs) is used only when the field has no reading of its own, with a warning. See
   `flow_candidates`.
   An end text that is one punctuation mark of END_MARKS (, . ; : ) ] ( [) ends the value where
   the mark ends a word or stands alone, and an opening bracket also where it starts the next
   word or stands inside a word ("Thousand(KES"): the value ends just before it. Not at a comma
   between figures, a colon inside a time, a bracket closing one opened in the value, letters
   in brackets inside a word ("Purchaser(s)"), a bracket that opens the value (a value is never
   empty, so with ')' also an end mark, the ')' closing it does not end the value either), or
   a full stop that does not end a sentence
   (`_scan_until`: not before a lower-case word or a figure; after an abbreviation, a
   capitalised word of one or two letters or a word with an inner full stop, only when a
   lower-case word follows before any other mark, with a warning). A full stop end also takes
   the end of the paragraph (the reading stops, or a line spacing PARA_PITCH times the
   paragraph's own after a short line), with a warning when the stop is missing. A field with
   other marks gets no value if its sentence or paragraph ends first. The reading looks one
   line past MAX_FLOW_LINES to judge a sentence end, but never ends a value there.
   Hyphens at line breaks: an anchor is compared both with the two pieces of a word broken by
   a hyphen at a line end and with the word joined. In `until` and `multiline` values the
   pieces are joined (`dehyphenate`); whether the hyphen stays is decided by `_hyphen_reading`
   from digits, the page and YAML as a lexicon, English number words and month names, and
   letter case, with a warning when it is a guess.
   Glued anchors: OCR often loses the space between two words. Besides the split in step 1,
   a prose anchor (label, alias or end text) may start inside its first OCR word or end inside
   its last (`_match_at`, `_cuts`, `_glue_ok`), also when the anchor crosses a line break, is
   broken by a hyphen there, or has words merged inside it. The part of the word inside the
   anchor must be exactly the anchor's whole edge word (plus any words merged into it). On the
   edge that borders the value (a label's right, an end text's left) the value starts or ends
   at the join, so the join must show evidence (`_join_evidence`): punctuation such as ", . ;
   : ) ]" before it or "( [" after it (not "/ - & '", which join parts of one word), a letter
   against a figure ("Box4410"), or a change of case ("NumberNAKURU", "NAKURUThe",
   "MUTHONIof"). A one-word anchor split at a change of case alone needs three characters or
   more and gets a warning when the split sets where the value starts or ends. On the other
   edge (a label's left, an end text's right), evidence is also required, except that a
   capitalised word may be glued before a lower-case label ("Thepurchase price is") when the
   label has two or more words, all matched exactly, and the glued word has MIN_WEAK_GLUE
   characters or more. So "of" is not found in "office", "and" not in "android", "Number" not
   in "Numberplate", and a label in capitals glued to a value in capitals is not split. The
   value's piece of the glued word gets a box estimated from character offsets, and claims
   only its own characters (step 5). The run-on check accepts only splits with evidence.
   A number written in words counts as a number for `integer` and `number` fields.
5. Fields are assigned greedily by score. Two fields cannot take the same words, except when
   both read them through the same label hit (an ad-hoc field and a preset field that share a
   label get the same value). One field's `until` text may be the next field's label: the end
   text is not part of the value, so nothing is claimed twice. A piece of a glued OCR word
   claims only its characters, so two fields may take the two sides of one glued word.
6. A requested field left without a value (label not found, nothing beside the label, or its
   value taken by another field) falls back to a page-wide `pattern` search; this is used only
   when exactly one distinct value matches, and with reduced confidence.
7. Weak matches are rejected (raw=None + warning) rather than returned confidently. So is a
   value that is a row of one mark ("XXXXXXXX", "--------"), which strikes out a blank
   (`_filler`).

Fields that set neither `multiline` nor `until` go through exactly steps 1 to 7 as before,
except that an `integer` or `number` value written in words now fits its type, and that the
ruled-form rules above (leaders, stacked lines kept out of one row, stacked labels, pieces of
longer labels, rows of one mark) apply to every field.

Each FieldMatch also says where its text came from: the OCR line and box of every piece of the
value (`parts`) and the OCR lines of its label and end text (`anchor_lines`). `dococr.pagebreak`
uses them to tell a reading that crosses a page break from one inside a page; they change no
value.
"""

from __future__ import annotations

import dataclasses
import itertools
import re
from dataclasses import dataclass, field
from typing import Iterable, Optional

from rapidfuzz.distance import Levenshtein

from dococr.models import DocType, FieldSpec, FieldType
from dococr.normalize import find_date, find_number_words, normalize_value, number_word_forms
from dococr.ocr import Box, OcrLine

# ---------------------------------------------------------------------------- tuning

REJECT_BELOW = 0.35  # final confidence under this -> raw=None with a warning
WARN_BELOW = 0.6  # final confidence under this -> value kept, with a warning
PATTERN_FALLBACK_FACTOR = 0.5
GEOM_RIGHT = 1.0
GEOM_BELOW = 0.95

_CANON = str.maketrans({"0": "o", "1": "i", "l": "i", "|": "i", "!": "i", "5": "s", "8": "b"})


def canon(text: str) -> str:
    """Comparison form for label matching: lower-case, look-alikes folded, alnum only."""
    text = re.sub(r"[^a-z0-9]", "", text.lower().translate(_CANON))
    return text.replace("rn", "m").replace("vv", "w")  # "Sumame" == "Surname"


def _allowed_edits(n: int) -> int:
    return 0 if n <= 3 else 1 if n <= 7 else 2


# ---------------------------------------------------------------------------- data

@dataclass
class Word:
    text: str
    canon: str
    box: Box
    line: int  # index into the OCR lines
    pos: int  # token index within its line (separators included)
    conf: float
    is_sep: bool = False
    row: int = -1
    idx: int = -1  # index within the row
    hit: int = -1  # label-hit id when this word is part of a label
    seg: int = -1  # segment (visual line of one column) this word belongs to
    sidx: int = -1  # index within the segment
    alpha: bool = False  # has a letter
    gap: bool = True  # preceded by white space (or the start of its OCR line)
    lead: bool = False  # the first word of its OCR box (only separators, such as leader dots, before it)
    parts: tuple = ()  # the OCR words this word was made from (a hyphenated word, a trimmed word)
    span: Optional[tuple[int, int]] = None  # for a piece of one OCR word: its char range in parts[0]
    cuts: Optional[tuple] = field(default=None, init=False, repr=False, compare=False)  # `_cuts` cache

    @property
    def xc(self) -> float:
        return (self.box[0] + self.box[2]) / 2

    @property
    def h(self) -> float:
        return self.box[3] - self.box[1]


@dataclass
class Variant:
    words: tuple[str, ...]  # canonical words
    joined: str
    text: str  # label as written in the spec
    owners: set[str] = field(default_factory=set)  # field names (empty = stop label only)
    strict: bool = field(init=False)  # a single word of <=5 characters: must match exactly
    lead_alpha: bool = field(init=False)  # the first word as written has a letter

    def __post_init__(self) -> None:
        self.strict = len(self.words) == 1 and len(self.joined) <= 5
        lead = re.search(r"[A-Za-z0-9]+", self.text)
        self.lead_alpha = bool(lead and re.search(r"[A-Za-z]", lead.group(0)))


@dataclass
class Hit:
    id: int
    row: int
    start: int  # row index of first word
    end: int  # row index after last word
    variant: Variant
    score: float
    box: Box
    # A label printed on several lines of one cell (`find_stacked_hits`): (row, start, end) of
    # each line's words, top first. row/start/end are then those of the first line, and `box`
    # covers all of them.
    parts: tuple[tuple[int, int, int], ...] = ()
    # The label is only the start of a longer label whose next words follow it in the same OCR
    # box ("Name" in "Name of" when "Name of Mother" is a label): it gives no value.
    fragment_of: Optional[Variant] = None
    # For a stacked label: its lines when each is also a label on its own, one of another field.
    lone: tuple[str, ...] = ()

    @property
    def weight(self) -> float:
        return self.score * len(self.variant.joined)

    def cells(self) -> set[tuple[int, int]]:
        return {(r, k) for r, s, e in (self.parts or ((self.row, self.start, self.end),))
                for k in range(s, e)}


@dataclass(frozen=True)
class Cut:
    """A place inside an OCR word where an anchor may begin or end because OCR lost a space
    (`_cuts`). `at` is a char offset in the word; `anchor` and `rest` are the canonical text of
    the part that belongs to the anchor and of the other part; `why` is the evidence at the join:
    "mark" (punctuation), "digit" (letter against figure), "case" (a change of case), or ""
    (none: a capitalised word followed by a lower-case one, as in "Thepurchase")."""
    at: int
    anchor: str
    rest: str
    why: str


@dataclass
class FlowHit:
    """A label of an `until` field, found in reading order (it may straddle a line break)."""
    id: int
    variant: Variant
    score: float
    words: list[Word]
    rcut: Optional[Cut] = None  # the label ends inside its last word; the value starts there
    notes: list[str] = field(default_factory=list)  # warnings about a split glued word
    # A longer label of another field matches at the same place ("X and" inside "X and Y
    # Ltd, located at"): this hit is used only when the field has no reading from a hit of its
    # own, and then with a warning.
    shadow: Optional[Variant] = None


@dataclass
class Candidate:
    field: str
    words: list[Word]
    raw: str
    score: float  # ranks candidates (mean OCR confidence, as for single-line values)
    hit: Optional[Hit | FlowHit]
    warnings: list[str]
    how: str  # "right" | "below" | "flow" | "pattern"
    conf_scale: float = 1.0  # reported confidence = score x this (weakest line of a multi-line value)
    ends: list[Word] = field(default_factory=list)  # the end text's words (`until` values)


@dataclass
class FieldMatch:
    name: str
    raw: Optional[str] = None
    confidence: Optional[float] = None
    box: Optional[Box] = None  # working-image coordinates
    warnings: list[str] = field(default_factory=list)
    # Where the value's text came from: (OCR line index, box) of each piece of each value word,
    # in reading order, and the OCR lines of the label and end text that bound it. Used to tell
    # whether a reading crosses a page break when pages are read together.
    parts: list[tuple[int, Box]] = field(default_factory=list)
    anchor_lines: list[int] = field(default_factory=list)


def _leaves(words: Iterable[Word]) -> Iterable[Word]:
    """The OCR words (or pieces of one) behind value words: a word joined across a hyphenated
    line break is its two halves."""
    for w in words:
        if w.span is None and w.parts:
            yield from _leaves(w.parts)
        else:
            yield w


# ---------------------------------------------------------------------------- vocabulary

def _label_words(text: str) -> tuple[str, ...]:
    """Canonical words of a label as written (no variants are generated)."""
    return tuple(c for c in (canon(w) for w in re.findall(r"[A-Za-z0-9]+", text)) if c)


def _all_specs(doc_type: DocType, fields: list[FieldSpec]) -> list[FieldSpec]:
    """The requested fields, then the doc type's other fields."""
    return list(fields) + [f for f in doc_type.fields if f.name not in {s.name for s in fields}]


def build_vocabulary(doc_type: DocType, fields: list[FieldSpec]) -> list[Variant]:
    """Labels to look for in rows: fields' labels and aliases, then the doc type's other_labels.

    Labels of `until` fields are not row labels: they are words inside running text, found in
    reading order by `find_flow_hits`, and must not cut the values of form fields."""
    by_words: dict[tuple[str, ...], Variant] = {}
    for spec in _all_specs(doc_type, fields):
        if spec.until_texts():
            continue
        for label in spec.labels():
            words = _label_words(label)
            if words:
                v = by_words.setdefault(words, Variant(words, "".join(words), label))
                v.owners.add(spec.name)
    for label in doc_type.other_labels:
        words = _label_words(label)
        if words and words not in by_words:
            by_words[words] = Variant(words, "".join(words), label.strip())
    return list(by_words.values())


# ---------------------------------------------------------------------------- layout

# A colon is a token of its own ("Name:JOHN" is three tokens), except between two figures, where
# it is part of a time or ratio ("9:00", "10:30:15") and stays inside the word.
_TOKEN = re.compile(r"(?:[^\s:]|(?<=\d):(?=\d))+|:")
# Leaders: the dots or rules of a fill-in blank ("......KIAMBU......", "..Province"). A run of two
# or more is a separator token of its own, so it neither hides a label or end text glued to it
# nor stays on the edge of a value.
_LEADER = re.compile(r"\.{2,}|…+|_{2,}|·{2,}")


def _leader_pieces(text: str, s: int, e: int) -> list[tuple[int, int]]:
    """The token text[s:e] split around leader runs (each run is a piece of its own)."""
    out: list[tuple[int, int]] = []
    pos = s
    for m in _LEADER.finditer(text, s, e):
        if m.start() > pos:
            out.append((pos, m.start()))
        out.append(m.span())
        pos = m.end()
    if pos < e:
        out.append((pos, e))
    return out


def _split_glued(token: str, vocab: list[Variant]) -> Optional[int]:
    """Char offset where a long label glued to a value ends ("DateofBirth12.03.1990" -> 11)."""
    c = canon(token)
    if len(c) < 7:
        return None
    for v in vocab:
        if len(v.joined) >= 6 and len(c) > len(v.joined) and c.startswith(v.joined):
            rest = c[len(v.joined):]
            if not re.search(r"\d", token):
                continue
            # map canonical length back to a raw offset
            count = 0
            for i, ch in enumerate(token):
                if canon(ch):
                    count += 1
                    if count == len(v.joined):
                        if rest and re.search(r"\d", token[i + 1:]):
                            return i + 1
                        break
    return None


def _split_label_tail(token: str, tails: frozenset[str]) -> Optional[int]:
    """Char offset where the last word of a label of two or more words is glued to a value
    that starts with a figure: the last line of a stacked label read in one box with its value
    ("No.687860/36" -> 3, "Registration12.03.2001" -> 12). After a full stop for any such
    word, directly for a word of five letters or more."""
    m = re.match(r"([A-Za-z]+)(\.?)(?=\d)", token)
    if not m or canon(m.group(1)) not in tails:
        return None
    if m.group(2) or len(m.group(1)) >= 5:
        return m.end()
    return None


def _split_anchor_glue(token: str, heads: frozenset[str], tails: frozenset[str]) -> Optional[int]:
    """Char offset where a prose anchor word is glued to a value: the last word of an `until`
    field's label before it ("numberJSX956" -> 6), or the first word of an end text after it
    ("474Hchassis" -> 4). Only when the other part has a digit, as in `_split_glued` (for a
    label word under five characters, a digit straight after it)."""
    c = canon(token)
    pos = [i for i, ch in enumerate(token) if canon(ch)]  # raw index of each canonical char
    if len(pos) != len(c):
        return None  # 'rn'/'vv' folding changed the length; do not guess
    for w in heads:
        if len(c) > len(w) and c.startswith(w):
            rest = token[pos[len(w) - 1] + 1:]
            # a short word only when a digit follows at once ("KES450,000")
            if re.search(r"\d", rest) if len(w) >= 5 else re.match(r"\d", rest):
                return pos[len(w) - 1] + 1
    for w in tails:
        if len(c) > len(w) and c.endswith(w) and re.search(r"\d", token[:pos[len(c) - len(w)]]):
            return pos[len(c) - len(w)]
    return None


def tokenize(lines: list[OcrLine], vocab: list[Variant],
             glue: tuple[frozenset[str], frozenset[str]] = (frozenset(), frozenset())) -> list[Word]:
    words: list[Word] = []
    tails = frozenset(v.words[-1] for v in vocab if len(v.words) >= 2)
    for li, line in enumerate(lines):
        text = line.text
        n = max(len(text), 1)
        l, t, r, b = line.box
        span = r - l
        pieces: list[tuple[int, int]] = []
        for m in _TOKEN.finditer(text):
            for s, e in _leader_pieces(text, *m.span()):
                tok = text[s:e]
                cut = None if tok == ":" or _LEADER.fullmatch(tok) else _split_glued(tok, vocab)
                if cut is None and tok != ":" and (glue[0] or glue[1]):
                    cut = _split_anchor_glue(tok, *glue)
                if cut is None and tails and not pieces and s == m.start():
                    cut = _split_label_tail(tok, tails)  # only the first word of a box
                if cut:
                    pieces += [(s, s + cut), (s + cut, e)]
                else:
                    pieces.append((s, e))
        lead = True  # only separators so far in this OCR box
        prev_leader = False
        for pos, (s, e) in enumerate(pieces):
            tok = text[s:e]
            box = (l + span * s / n, t, l + span * e / n, b)
            sep = tok == ":" or not canon(tok)
            words.append(Word(tok, canon(tok), box, li, pos, line.confidence, is_sep=sep,
                              alpha=bool(re.search(r"[A-Za-z]", tok)),
                              gap=(s == 0 or text[s - 1].isspace() or prev_leader),
                              lead=lead and not sep))
            lead = lead and sep
            prev_leader = bool(_LEADER.fullmatch(tok))
    return words


def _claim_keys(words: Iterable[Word]) -> Iterable[tuple[int, int, int]]:
    """(OCR word id, first char, end char) of the text behind a value's words. A piece of an
    OCR word (`_piece`) covers only its own characters, so the two halves of a word that OCR
    glued together can go to two fields."""
    for w in words:
        if w.span is not None:
            yield id(w.parts[0]), w.span[0], w.span[1]
        elif w.parts:
            yield from _claim_keys(w.parts)
        else:
            yield id(w), 0, len(w.text)


def _piece(w: Word, a: int, b: int) -> Word:
    """The characters [a, b) of a word as a word of their own, with a box estimated from
    character offsets (as for words inside an OCR box). A piece of a piece is a piece of the
    original word."""
    if w.span is not None:
        base, off = w.parts[0], w.span[0]
    else:
        base, off = w, 0
    a, b = off + a, off + b
    text = base.text[a:b]
    n = max(len(base.text), 1)
    l, t, r, bt = base.box
    return dataclasses.replace(
        base, text=text, canon=canon(text), box=(l + (r - l) * a / n, t, l + (r - l) * b / n, bt),
        is_sep=not canon(text), alpha=bool(re.search(r"[A-Za-z]", text)),
        gap=base.gap if a == 0 else False, parts=(base,), span=(a, b))


# ---------------------------------------------------------------------------- hyphens at line breaks

_HYPHENS = "-\u2010\u2011\u2013\u2014"
_SOFT_HYPHENS = "\u00ad\u00ac"  # a discretionary hyphen: always a line-break hyphen


def _line_break_hyphen(a: Word, b: Word) -> bool:
    """`a` ends a line with a hyphen attached to it and `b` starts the next line in reading
    order. Only such a hyphen can be a line-break hyphen."""
    return (a.row != b.row and len(a.text) >= 2 and a.text[-1] in _HYPHENS + _SOFT_HYPHENS
            and a.text[-2].isalnum() and b.text[:1].isalnum())


def _hyphen_joins_letters(a: Word, b: Word) -> bool:
    """A line-break hyphen between letters: the two pieces may be one word ("regis-" "tered")."""
    return _line_break_hyphen(a, b) and a.text[-2].isalpha() and b.text[0].isalpha()


@dataclass
class Lexicon:
    """Evidence from the page and the YAML about how a word broken at a line end is written:
    canonical words seen whole (`words`), and canonical forms of words seen with a hyphen
    inside them (`hyphenated`, "self-employed" -> "selfemployed")."""
    words: set[str] = field(default_factory=set)
    hyphenated: set[str] = field(default_factory=set)

    def add(self, text: str) -> None:
        for tok in text.split():
            core = re.sub(r"^\W+|\W+$", "", tok)
            if not core:
                continue
            pieces = re.split(f"[{_HYPHENS}]", core)
            if len(pieces) == 1:
                if core.isalpha():
                    self.words.add(canon(core))
                continue
            for x, y in zip(pieces, pieces[1:]):
                lx, ry = re.search(r"[A-Za-z]+$", x), re.match(r"[A-Za-z]+", y)
                if lx and ry:
                    self.hyphenated.add(canon(lx.group(0) + ry.group(0)))


def build_lexicon(words: list[Word], doc_type: DocType, fields: list[FieldSpec]) -> Lexicon:
    """The page's words, the YAML's labels and end texts, and English number words and month
    names."""
    whole, hyphenated = number_word_forms()
    lex = Lexicon({canon(w) for w in whole}, {canon(w) for w in hyphenated})
    for w in words:
        if w.is_sep:
            continue
        if w.text.isalpha():
            lex.words.add(w.canon)
        else:
            lex.add(w.text)
    for spec in _all_specs(doc_type, fields):
        for text in [*spec.labels(), *spec.until_texts()]:
            lex.add(text)
    for text in doc_type.other_labels:
        lex.add(text)
    return lex


def _hyphen_reading(left: str, right: str, lex: Lexicon) -> tuple[bool, Optional[str]]:
    """Whether a hyphen printed at a line break (`left` ends with it, `right` starts the next
    line) belongs to the word, and a warning when that is a guess.

    - Next to a digit ("1021-" "00100"), the hyphen is part of the value: kept.
    - A discretionary (soft) hyphen is a line-break hyphen: removed.
    - Evidence from the page and the YAML decides next: the joined word seen whole elsewhere
      means a line-break hyphen; seen with its hyphen mid-line, a real one.
    - Lower case then a capital ("Mary-" "Anne") is a real hyphen: no single word is broken
      into a capitalised second half.
    - Both pieces in capitals ("WANJIKU-" "KAMAU"): kept, with a warning. Names in capitals are
      where hyphens are part of the value, and keeping it leaves exactly what was printed.
    - Otherwise ("regis-" "tered", "self-" "employed"): a line-break hyphen is far more common
      in typeset text, so it is removed, with a warning."""
    if left[-1] in _SOFT_HYPHENS:
        return False, None
    body = left[:-1]
    if body[-1:].isdigit() or right[:1].isdigit():
        return True, None
    lw = re.search(r"[A-Za-z]+$", body)
    rw = re.match(r"[A-Za-z]+", right)
    if not lw or not rw:
        return True, None
    lw, rw = lw.group(0), rw.group(0)
    joined = canon(lw + rw)
    whole, hyphenated = joined in lex.words, joined in lex.hyphenated
    keep_note = (f"kept the hyphen in '{lw}-{rw}' at a line break; it may be one word broken "
                 f"across the line ('{lw}{rw}')")
    if whole and not hyphenated:
        return False, None
    if hyphenated:
        return True, (keep_note if whole else None)
    if re.search(r"[a-z]", lw) and rw[0].isupper():
        return True, None
    if lw.isupper() and rw.isupper():
        return True, keep_note
    return False, (f"joined '{lw}-' and '{rw}' across a line break as '{lw}{rw}'; the hyphen may "
                   f"be part of the word ('{lw}-{rw}')")


def dehyphenate(words: list[Word], lex: Lexicon) -> tuple[list[Word], list[str]]:
    """Join each word broken by a hyphen at a line break to its second half (see
    `_hyphen_reading` for whether the hyphen is kept). Words are in reading order; the joined
    word keeps both OCR words in `parts`."""
    out: list[Word] = []
    notes: list[str] = []
    for w in words:
        if out and not w.is_sep and not out[-1].is_sep and _line_break_hyphen(out[-1], w):
            a = out.pop()
            keep, note = _hyphen_reading(a.text, w.text, lex)
            text = (a.text if keep else a.text[:-1]) + w.text
            out.append(dataclasses.replace(
                w, text=text, canon=canon(text), box=_union([a.box, w.box]), line=a.line,
                conf=min(a.conf, w.conf), alpha=True, gap=a.gap, parts=(a, w)))
            if note:
                notes.append(note)
        else:
            out.append(w)
    return out, notes


STACK_OFFSET = 0.3  # lines whose centres differ by this x text height or more are not side by side
STACK_OVERLAP = 0.3  # ... when they overlap horizontally by this share of the narrower one


def _stacked(a: Box, b: Box) -> bool:
    """Two OCR boxes one above the other: they overlap horizontally, so they cannot be parts of
    one visual line, and their centres are offset vertically. Lines of a label printed on two
    lines in a cell are often closer than one text height and would otherwise be read as one
    row, with their words interleaved ("Date Birth of")."""
    h = min(a[3] - a[1], b[3] - b[1])
    if abs((a[1] + a[3]) - (b[1] + b[3])) / 2 < STACK_OFFSET * h:
        return False
    ov = min(a[2], b[2]) - max(a[0], b[0])
    return ov > STACK_OVERLAP * max(1e-6, min(a[2] - a[0], b[2] - b[0]))


def group_rows(lines: list[OcrLine], words: list[Word]) -> list[list[Word]]:
    order = sorted(range(len(lines)), key=lambda i: (lines[i].box[1] + lines[i].box[3]) / 2)
    rows: list[list[int]] = []
    bands: list[tuple[float, float]] = []
    for i in order:
        t, b = lines[i].box[1], lines[i].box[3]
        best, best_ov = -1, 0.0
        for ri, (rt, rb) in enumerate(bands):
            ov = min(b, rb) - max(t, rt)
            ratio = ov / max(1e-6, min(b - t, rb - rt))
            if ratio >= 0.5 and ratio > best_ov:
                if any(_stacked(lines[i].box, lines[k].box) for k in rows[ri]):
                    continue  # one above the other, not side by side
                best, best_ov = ri, ratio
        if best < 0:
            rows.append([i])
            bands.append((t, b))
        else:
            rows[best].append(i)
            members = rows[best]
            bands[best] = (sum(lines[k].box[1] for k in members) / len(members),
                           sum(lines[k].box[3] for k in members) / len(members))
    by_line: dict[int, list[Word]] = {}
    for w in words:
        by_line.setdefault(w.line, []).append(w)
    out = []
    for ri in sorted(range(len(rows)), key=lambda k: bands[k][0]):
        row_words = sorted((w for li in rows[ri] for w in by_line.get(li, [])),
                           key=lambda w: (w.box[0], w.line, w.pos))
        for idx, w in enumerate(row_words):
            w.row, w.idx = len(out), idx
        out.append(row_words)
    return out


def _union(boxes: Iterable[Box]) -> Box:
    boxes = list(boxes)
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def _height(words: list[Word]) -> float:
    return max(1.0, sum(w.box[3] - w.box[1] for w in words) / len(words))


# ---------------------------------------------------------------------------- label search

def _match_words(cand: list[str], var: Variant) -> Optional[float]:
    """Similarity of candidate canonical words to a label variant, or None if not a match."""
    if len(cand) == len(var.words):
        edits = 0
        for c, l in zip(cand, var.words):
            if c == l:
                continue
            if var.strict:
                return None
            budget = _allowed_edits(len(l))
            if budget == 0:
                return None
            d = Levenshtein.distance(c, l, score_cutoff=budget)
            if d > budget:
                return None
            edits += d
        return 1.0 - edits / max(len(var.joined), 1)
    if var.strict and len(cand) > 1:
        return None
    joined = "".join(cand)
    n = len(var.joined)
    # Joined comparison only repairs split/merged words, so allow at most one edit: dropping
    # a whole word ("Registration" vs "Registration No") must not match.
    budget = 0 if n <= 5 else 1
    d = Levenshtein.distance(joined, var.joined, score_cutoff=budget)
    if d > budget:
        return None
    if d:
        # The edit must repair a split or merged word, not add or drop a whole word of one
        # letter: "A Per Share" is not "Per Share" (that would eat the "A" of "Series A" before
        # an end text or a label), and "Form" is not "Form A".
        for i, c in enumerate(cand):
            if len(c) <= d and "".join(cand[:i] + cand[i + 1:]) == var.joined:
                return None
        for i, lw in enumerate(var.words):
            if len(lw) <= d and "".join(var.words[:i] + var.words[i + 1:]) == joined:
                return None
    return (1.0 - d / max(n, 1)) * 0.97


def find_label_hits(rows: list[list[Word]], vocab: list[Variant],
                    notes: Optional[dict[str, str]] = None) -> list[Hit]:
    """Label hits in rows, single-line and stacked (`find_stacked_hits`), overlaps resolved.
    `notes` receives, per field, why a stacked reading of its label was not used."""
    by_len: dict[int, list[Variant]] = {}
    for v in vocab:
        by_len.setdefault(len(v.joined), []).append(v)
    max_words = max((len(v.words) for v in vocab), default=1) + 1
    raw_hits: list[Hit] = []
    for ri, row in enumerate(rows):
        h = _height(row) if row else 1.0
        for i, first in enumerate(row):
            if first.is_sep or not re.search(r"[A-Za-z]", first.text):
                continue
            span: list[Word] = []
            for j in range(i, min(len(row), i + max_words)):
                w = row[j]
                if w.is_sep:
                    break
                if span and w.line != span[-1].line and w.box[0] - span[-1].box[2] > 1.5 * h:
                    break
                if not re.search(r"[A-Za-z]", w.text) and len(w.canon) > 2:
                    break  # a number cannot be part of a label
                span.append(w)
                cand = [x.canon for x in span]
                total = sum(len(c) for c in cand)
                for n in range(max(1, total - 3), total + 4):
                    for v in by_len.get(n, ()):
                        if v.strict and not first.lead:
                            continue
                        s = _match_words(cand, v)
                        if s is not None and s >= 0.8:
                            raw_hits.append(Hit(-1, ri, i, j + 1, v, s, _union(x.box for x in span)))
    stacked = find_stacked_hits(rows, vocab)
    # Resolve overlaps: longest, best-quality label wins its words, also across the lines of a
    # stacked label, so "Name" alone cannot take a word of "Name of / Registering Officer".
    # A stacked label each of whose lines has text of its own to the right is two rows of the
    # form, not one label ("BODY  SALOON" above "COLOUR  WHITE", with "Body Colour" a label):
    # it is dropped and the overlaps resolved again without it.
    while True:
        hits = _resolve_hits(rows, raw_hits + stacked)
        rival = [hh for hh in hits if hh.parts and _rows_of_their_own(rows, hh)]
        if not rival:
            break
        for hh in rival:
            for owner in hh.variant.owners:
                if notes is not None:
                    notes.setdefault(owner, f"label '{hh.variant.text}' seems to be printed on "
                                     f"{len(hh.parts)} lines, but each line has text to its right, "
                                     "so they were read as separate rows of the form")
        gone = {id(hh) for hh in rival}
        stacked = [hh for hh in stacked if id(hh) not in gone]
    for hh in hits:
        if not hh.parts:
            hh.fragment_of = _longer_label(rows, hh, vocab)
    return hits


def _resolve_hits(rows: list[list[Word]], raw_hits: list[Hit]) -> list[Hit]:
    """Overlapping hits: the longest, best-quality label wins its words. Sets `Word.hit`."""
    for row in rows:
        for w in row:
            w.hit = -1
    raw_hits = sorted(raw_hits, key=lambda hh: (-hh.weight, -hh.score, hh.row, hh.start))
    taken: set[tuple[int, int]] = set()
    hits: list[Hit] = []
    for hh in raw_hits:
        cells = hh.cells()
        if cells & taken:
            continue
        taken |= cells
        hh.id = len(hits)
        hits.append(hh)
        for r, k in cells:
            rows[r][k].hit = hh.id
    return hits


def _rows_of_their_own(rows: list[list[Word]], hit: Hit) -> bool:
    """Does every line of a stacked label have text to its right, before the next label, with
    those texts in one column (all overlapping horizontally), as the values of rows of a form
    are? Text far to the right of one line only (another cell's) does not count."""
    spans = []
    for r, s, e in hit.parts:
        view = dataclasses.replace(hit, row=r, start=s, end=e,
                                   box=_union(w.box for w in rows[r][s:e]), parts=())
        ws = [w for w in _right_words(rows, view, float("inf")) if not w.is_sep]
        if not ws:
            return False
        spans.append((ws[0].box[0], ws[-1].box[2]))
    return max(lft for lft, _ in spans) < min(rgt for _, rgt in spans)


def _box_words(rows: list[list[Word]]) -> dict[int, list[Word]]:
    """The non-separator words of each OCR box, in reading order."""
    out: dict[int, list[Word]] = {}
    for row in rows:
        for w in row:
            if not w.is_sep:
                out.setdefault(w.line, []).append(w)
    for ws in out.values():
        ws.sort(key=lambda w: (w.idx, w.pos))
    return out


def _longer_label(rows: list[list[Word]], hit: Hit, vocab: list[Variant]) -> Optional[Variant]:
    """A longer label that starts with this hit's label and whose next words follow the hit in
    the same OCR box: "Name" in a box "Name of" when "Name of Registering Officer" is a label.
    The hit is then a piece of that label (the rest is elsewhere or was misread), and the words
    after it are not its value."""
    row = rows[hit.row]
    last = row[hit.end - 1]
    after = [w for w in row[hit.end:] if w.line == last.line and not w.is_sep]
    if not after or after[0].hit >= 0:
        return None
    n = len(hit.variant.words)
    for v in vocab:
        if len(v.words) <= n or v.words[:n] != hit.variant.words:
            continue
        nxt = v.words[n]
        if Levenshtein.distance(after[0].canon, nxt) <= _allowed_edits(len(nxt)):
            return v
    return None


STACK_PITCH = 1.6  # lines of one label are at most this x text height apart (centre to centre)
STACK_GAP = 0.6  # ... with at most this x text height of blank between them
STACK_ALIGN = 1.0  # ... and share a left edge, centre or right edge within this x text height
MAX_STACK_LINES = 3


def _aligned(a: Box, b: Box, h: float) -> bool:
    return (abs(a[0] - b[0]) <= STACK_ALIGN * h or abs(a[2] - b[2]) <= STACK_ALIGN * h
            or abs((a[0] + a[2]) - (b[0] + b[2])) / 2 <= STACK_ALIGN * h)


def find_stacked_hits(rows: list[list[Word]], vocab: list[Variant]) -> list[Hit]:
    """Labels printed on two or three short lines of one cell ("Date of" above "Birth").

    The label's words are the whole of each OCR box but the last, and the start of the last
    box (the value may follow in that box: "Born  PCEA Hospital"). Each line is directly below
    the one before: its centre 0.5 to STACK_PITCH text heights lower, at most STACK_GAP heights
    of blank between them, text of a similar height, and aligned with it on the left, the
    right or the centre (`_aligned`). Their words joined must match a label of two or more
    words, as row labels are matched. A line that has other text after the label's words in
    its box cannot be an upper line, which keeps "Name: JOHN" above "Surname: DOE" apart, and
    so does running text, whose lines are full boxes of words."""
    multi = [v for v in vocab if len(v.words) >= 2]
    if not multi:
        return []
    by_len: dict[int, list[Variant]] = {}
    for v in multi:
        by_len.setdefault(len(v.joined), []).append(v)
    maxw = max(len(v.words) for v in multi)
    boxes = _box_words(rows)
    info: dict[int, tuple[Box, float]] = {}
    for li, ws in boxes.items():
        # the box's words must be contiguous in their row (not interleaved with another box)
        idx = [w.idx for w in ws]
        between = rows[ws[0].row][idx[0]:idx[-1] + 1]
        if any(not w.is_sep and w.line != li for w in between):
            continue
        box = _union(w.box for w in ws)
        info[li] = (box, box[3] - box[1])
    order = sorted(info, key=lambda li: info[li][0][1])

    def below(li: int, frag: Box) -> list[int]:
        box, h = info[li]
        out = []
        for lj in order:
            b2, h2 = info[lj]
            hh = max(h, h2)
            if lj == li or not (0.7 <= h2 / h <= 1.4):
                continue
            pitch = (b2[1] + b2[3] - box[1] - box[3]) / 2
            if pitch < 0.5 * hh or pitch > STACK_PITCH * hh or b2[1] - box[3] > STACK_GAP * hh:
                continue
            if min(b2[2], frag[2]) - max(b2[0], frag[0]) <= 0:
                continue
            out.append(lj)
        return out

    def label_word(w: Word) -> bool:
        return w.alpha or len(w.canon) <= 2  # a number cannot be part of a label

    def own_label(words: list[Word]) -> Optional[Variant]:
        """The label these words are on their own, if any."""
        cand = [w.canon for w in words]
        total = sum(len(c) for c in cand)
        best = None
        for v in vocab:
            if abs(len(v.joined) - total) <= 3:
                sc = _match_words(cand, v)
                if sc is not None and sc >= 0.8 and (best is None or sc > best[0]):
                    best = (sc, v)
        return best[1] if best else None

    def separate_labels(lines: list[list[Word]], v: Variant) -> tuple[str, ...]:
        """When every line is a label on its own and one of them belongs to another field
        (or is one of the other_labels), those labels: the lines may be two labels ("Body"
        above "Colour") rather than one ("Body Colour"), or one label whose lines are also
        labels ("Entry" above "No."). Which cannot be told from the page."""
        found = [own_label(ws) for ws in lines]
        if all(found) and any(not f.owners or not f.owners <= v.owners for f in found):
            return tuple(f.text for f in found)
        return ()

    hits: list[Hit] = []

    def extend(chain: list[tuple[int, list[Word]]]) -> None:
        """chain: (box, words) of the upper lines so far, each the whole box."""
        upper_words = [w for _, ws in chain for w in ws]
        last_li = chain[-1][0]
        last_box = _union(w.box for w in chain[-1][1])
        for lj in below(last_li, last_box):
            ws = boxes[lj]
            for k in range(1, min(len(ws), maxw - len(upper_words)) + 1):
                frag = ws[:k]
                if not all(label_word(w) for w in frag):
                    break
                fbox = _union(w.box for w in frag)
                if not _aligned(last_box, fbox, max(info[last_li][1], info[lj][1])):
                    continue
                words = upper_words + frag
                cand = [w.canon for w in words]
                total = sum(len(c) for c in cand)
                for n in range(max(1, total - 3), total + 4):
                    for v in by_len.get(n, ()):
                        s = _match_words(cand, v)
                        if s is not None and s >= 0.8:
                            parts = tuple((ws_[0].row, ws_[0].idx, ws_[-1].idx + 1)
                                          for ws_ in [c for _, c in chain] + [frag])
                            hits.append(Hit(-1, parts[0][0], parts[0][1], parts[0][2], v, s,
                                            _union(w.box for w in words), parts,
                                            lone=separate_labels([c for _, c in chain] + [frag], v)))
            if len(chain) + 1 < MAX_STACK_LINES and len(ws) + len(upper_words) < maxw \
                    and all(label_word(w) for w in ws) and _aligned(last_box, info[lj][0], info[lj][1]):
                extend(chain + [(lj, ws)])

    for li in order:
        ws = boxes[li]
        if len(ws) < maxw and all(label_word(w) for w in ws):
            extend([(li, ws)])
    return hits


# ---------------------------------------------------------------------------- values

def _join(words: list[Word]) -> str:
    return " ".join(w.text for w in words if not w.is_sep or w.text not in (":",)).strip()


def _slice_words(words: list[Word], raw: str, start: int, end: int) -> list[Word]:
    """Words of `words` (joined as `raw`) that overlap the char range [start, end)."""
    out, pos = [], 0
    for w in words:
        s = raw.find(w.text, pos)
        if s < 0:
            continue
        e = s + len(w.text)
        pos = e
        if e > start and s < end:
            out.append(w)
    return out or words


def _compile(pattern: Optional[str]) -> Optional[re.Pattern]:
    if not pattern:
        return None
    try:
        return re.compile(pattern, re.IGNORECASE)
    except re.error:
        return None


def assess(words: list[Word], spec: FieldSpec) -> tuple[list[Word], str, float, list[str]]:
    """Trim a candidate to the plausible part and score how well it fits the field's type."""
    raw = _join(words)
    kind = spec.type
    regex = _compile(spec.pattern)
    if regex is not None:
        m = regex.fullmatch(raw) or regex.fullmatch(raw.replace(" ", ""))
        if m:
            return words, raw, 1.0, []
        m = _search_bounded(regex, raw)
        if m:
            sub = _slice_words(words, raw, m.start(), m.end())
            return sub, m.group(0), 0.95, []
        if _fits_after_repair(raw, spec):
            return words, raw, 0.9, []
    if kind == FieldType.DATE:
        found = find_date(raw, spec.month_first())
        if found:
            _, s, e, _ = found
            sub = _slice_words(words, raw, s, e)
            return sub, raw[s:e].strip(), 1.0, []
        return words, raw, 0.4, ["value does not look like a date"]
    if kind in (FieldType.INTEGER, FieldType.NUMBER):
        if re.search(r"\d", raw):
            return words, raw, 1.0 if regex is None else 0.7, []
        written = find_number_words(raw)
        if written is not None and written.value is not None:
            return words, raw, 1.0 if regex is None else 0.7, []  # a number in words
        return words, raw, 0.4, ["value has no digits"]
    if kind == FieldType.IDENTIFIER:
        if regex is not None:
            return words, raw, 0.45, []  # the normaliser adds the format warning
        return words, raw, (1.0 if re.search(r"\d", raw) else 0.75), []
    if kind == FieldType.NAME:
        letters = sum(ch.isalpha() for ch in raw)
        digits = sum(ch.isdigit() for ch in raw)
        if letters and digits <= 0.2 * (letters + digits):
            return words, raw, 1.0, []
        return words, raw, 0.4, ["value does not look like a name"]
    if regex is not None:
        return words, raw, 0.45, []
    return words, raw, 0.9 if re.search(r"[A-Za-z0-9]", raw) else 0.3, []


def _fits_after_repair(raw: str, spec: FieldSpec) -> bool:
    """Does the normaliser make this fit the pattern (look-alike repair such as 0+ -> O+)?"""
    _, warnings = normalize_value(raw, spec)
    return not any("expected format" in w and "corrected" not in w for w in warnings)


def _search_bounded(regex: re.Pattern, text: str) -> Optional[re.Match]:
    try:
        return re.search(rf"(?<![A-Za-z0-9])(?:{regex.pattern})(?![A-Za-z0-9])", text, re.IGNORECASE)
    except re.error:
        return regex.search(text)


def _right_words(rows: list[list[Word]], hit: Hit, page_w: float) -> list[Word]:
    row = rows[hit.row]
    h = hit.box[3] - hit.box[1]
    out: list[Word] = []
    prev_right = hit.box[2]
    prev_line = row[hit.end - 1].line
    for w in row[hit.end:]:
        if w.hit >= 0:
            break
        if w.is_sep:
            if w.text == ":" and out:
                break  # "Label1: v1 Label2: v2" with an unrecognised Label2
            if out:
                out.append(w)
            prev_right, prev_line = w.box[2], w.line
            continue
        gap = w.box[0] - prev_right
        if w.line != prev_line:
            if out and gap > 2.5 * h:
                break
            if not out and gap > 0.6 * page_w:
                break
        out.append(w)
        prev_right, prev_line = w.box[2], w.line
    while out and out[-1].is_sep:
        out.pop()
    return out


def _value_places(rows: list[list[Word]], hit: Hit, page_w: float
                  ) -> Iterable[tuple[str, float, Hit, list[Word]]]:
    """(how, geometry factor, label view, words) of each place a label's value may be.

    For a label on one line: to its right and below it. For a stacked label, to the right of
    each of its lines, top first, to the right in any row whose middle lies within the label's
    height (a value centred in its cell, level with none of the lines), and below its last line, in a column as wide as
    the whole label. Each place is read through a view of the hit with that line's row, words
    and box; the view keeps the hit's id, so the label's other lines are known as its own."""
    if not hit.parts:
        yield "right", GEOM_RIGHT, hit, _right_words(rows, hit, page_w)
        yield "below", GEOM_BELOW, hit, _below_words(rows, hit)
        return
    view = hit
    for r, s, e in hit.parts:
        box = _union(w.box for w in rows[r][s:e])
        view = dataclasses.replace(hit, row=r, start=s, end=e, box=box, parts=())
        yield "right", GEOM_RIGHT, view, _right_words(rows, view, page_w)
    # A value centred in its cell sits between the label's lines, level with neither, and so
    # in a row of its own: read the rows whose middle falls inside the label's height.
    own_rows = {r for r, _, _ in hit.parts}
    h = view.box[3] - view.box[1]
    for r, row in enumerate(rows):
        if r in own_rows or not row:
            continue
        middle = (min(w.box[1] for w in row) + max(w.box[3] for w in row)) / 2
        if not (hit.box[1] < middle < hit.box[3]):
            continue
        words: list[Word] = []
        prev_right = hit.box[2]
        for w in row:
            if w.box[0] < hit.box[2]:
                continue
            if w.hit >= 0 or (words and w.box[0] - prev_right > 2.5 * h and w.line != words[-1].line):
                break
            if not words and w.box[0] - prev_right > 0.6 * page_w:
                break
            if _keep(w):
                words.append(w)
                prev_right = w.box[2]
        if words:
            yield "right", GEOM_RIGHT, view, words
    last = dataclasses.replace(view, box=(hit.box[0], view.box[1], hit.box[2], view.box[3]))
    yield "below", GEOM_BELOW, last, _below_words(rows, last)


def _hit_words(rows: list[list[Word]], hit: Hit) -> list[Word]:
    return [w for r, s, e in (hit.parts or ((hit.row, hit.start, hit.end),)) for w in rows[r][s:e]]


def _keep(w: Word) -> bool:
    """A word that belongs in a value line: not a separator, except a lone '&', which is part
    of names and addresses ("Smith & Sons")."""
    return not w.is_sep or w.text == "&"


def _below_words(rows: list[list[Word]], hit: Hit) -> list[Word]:
    row = rows[hit.row]
    h = hit.box[3] - hit.box[1]
    # Column window: from the previous label's end to the next label's start in this row.
    left = hit.box[0] - 2 * h
    right = float("inf")
    for w in row[:hit.start]:
        if w.hit >= 0 and w.hit != hit.id:
            left = max(left, (w.box[2] + hit.box[0]) / 2)
    for w in row[hit.end:]:
        if w.hit >= 0 and w.hit != hit.id:
            right = w.box[0] - 0.5 * h
            break
    for ri in range(hit.row + 1, len(rows)):
        cand_row = rows[ri]
        if not cand_row:
            continue
        top = min(w.box[1] for w in cand_row)
        if top - hit.box[3] > 1.8 * h:
            return []
        inside = [w for w in cand_row if w.xc >= left and w.box[0] < right and _keep(w)]
        if not inside:
            continue
        if inside[0].hit >= 0:
            return []  # the next thing below is another label
        out = [inside[0]]
        for w in inside[1:]:
            if w.hit >= 0:
                break
            if w.line != out[-1].line and w.box[0] - out[-1].box[2] > 2.5 * h:
                break
            out.append(w)
        return out
    return []


# ---------------------------------------------------------------------------- multi-line values (`multiline`)

MAX_BLOCK_LINES = 5  # a `multiline` value is cut after this many lines, with a warning
BLOCK_GAP = 1.0  # a blank space taller than this x the value's line height ends the block
BLOCK_ALIGN = 1.5  # continuation lines start within this x line height of the first line's left
BLOCK_HEIGHT = (0.6, 1.5)  # and have a text height within these ratios of the first line's


# English words after which a phrase cannot end ("... August 15 and", "Board of"), like the
# abbreviations of `_ABBREVIATIONS`: knowledge of the language, not of a document.
_OPEN_WORDS = frozenset("and or of the for to in on at by with from nor &".split())
_OPEN_MARKS = ",;&/-(\u2010\u2013\u2014"


def _open_end(words: list[Word]) -> Optional[str]:
    """What shows that a value does not end here: it ends with a comma, ';', '&', a hyphen or
    an opening bracket, or with a word such as "and" or "of", or it has an unclosed bracket.
    The ending (for a message), or None."""
    body = [w for w in words if not w.is_sep]
    if not body:
        return None
    last = body[-1].text
    if last[-1:] in _OPEN_MARKS and (last[-1] in "(&" or re.search(r"[A-Za-z0-9]", last)):
        return f"'{last[-1]}'"  # not a lone dash, which stands for "none"
    word = re.sub(r"^\W+|\W+$", "", last).lower()
    if word in _OPEN_WORDS or last in _OPEN_WORDS:
        return f"'{last}'"
    text = " ".join(w.text for w in body)
    if text.count("(") > text.count(")") or text.count("[") > text.count("]"):
        return "an unclosed bracket"
    return None


def _starts_lower(line: list[Word]) -> bool:
    first = next((w for w in line if not w.is_sep), None)
    return bool(first and re.match(r"[\"'(\[]?[a-z]", first.text))


def _block_lines(rows: list[list[Word]], hit: Hit, first: list[Word], how: str
                 ) -> tuple[list[list[Word]], bool, list[str]]:
    """The first line of a value plus the lines below it that belong to the same block.

    A continuation line starts under the first line's left edge, has text of the same height,
    and is not another label. It must stay inside the label's column (the next label on the
    label's row bounds it). For a value to the right of its label, a line with text in the
    label's column is the next row of the form, unless it is the value wrapping back to the
    label's margin (`_wrap_back`): no label and no colon in it, and the value plainly goes on
    (the line above ends with a comma, "and", "of", an opening bracket ..., or this line starts
    in lower case). The block ends at a gap taller than BLOCK_GAP lines or after
    MAX_BLOCK_LINES lines; the bool says the cap was hit while the block still went on. The
    notes are warnings about a line that looks like more of the value but was not joined."""
    body = [w for w in first if not w.is_sep]
    h = _height(body)
    col_l = min(w.box[0] for w in body)
    label_h = hit.box[3] - hit.box[1]
    right = float("inf")
    for w in rows[hit.row][hit.end:]:
        if w.hit >= 0 and w.hit != hit.id and w.box[0] > col_l:
            right = w.box[0] - 0.5 * label_h
            break
    left = col_l - BLOCK_ALIGN * h
    if how == "right" and body[0].line == rows[hit.row][hit.end - 1].line:
        # Label and value in one OCR box: word positions inside a box are estimated from
        # character counts, so the value's left edge is uncertain. Accept a continuation that
        # starts anywhere past the middle of the label.
        left = min(left, (hit.box[0] + hit.box[2]) / 2)
    gutter = (hit.box[0] - 0.5 * label_h, left) if how == "right" else None
    lines = [first]
    notes: list[str] = []
    last_row = max(w.row for w in body)
    bottom = max(w.box[3] for w in body)
    for ri in range(last_row + 1, len(rows)):
        row = rows[ri]
        if not row:
            continue
        if min(w.box[1] for w in row) - bottom > BLOCK_GAP * h:
            break
        if gutter and any(not w.is_sep and w.hit != hit.id and w.box[2] > gutter[0] and w.box[0] < gutter[1]
                          for w in row):
            # Text in the label column: the next row of the form, or the value wrapping back
            # to the label's margin.
            line, why = _wrap_back(row, hit, h, right, [w for ln in lines for w in ln])
            if line is None:
                if why:
                    notes.append(why)
                break
        else:
            inside = [w for w in row if _keep(w) and w.box[0] >= left and w.xc < right]
            if not inside:
                continue  # text elsewhere on the page at this height
            if inside[0].hit >= 0 or any(w.hit >= 0 and w.line == inside[0].line for w in row):
                break  # another label, or text in one OCR box with a label: the next form row
            line = [inside[0]]
            for w in inside[1:]:
                if w.hit >= 0 or (w.line != line[-1].line and w.box[0] - line[-1].box[2] > 2.5 * h):
                    break
                line.append(w)
            lh = _height(line)
            if not (BLOCK_HEIGHT[0] * h <= lh <= BLOCK_HEIGHT[1] * h):
                break
            if inside[0].box[0] > col_l + BLOCK_ALIGN * h:
                # Indented further than the first line (a centred or tabbed continuation, or a
                # first line whose left edge was estimated inside a shared OCR box): joined
                # only inside the value's width and when the value plainly goes on.
                value = [w for ln in lines for w in ln]
                inner = inside[0].box[0] < max(w.box[2] for w in value if not w.is_sep)
                colon = any(w.is_sep and w.text == ":" for w in row)
                if inner and not colon and (_open_end(value) or _starts_lower(line)):
                    pass
                else:
                    if inner and not colon:
                        notes.append("the line below the value sits inside its column and may continue "
                                     "it; it was not joined because it is indented and nothing shows "
                                     "that the value goes on")
                    break
        if len(lines) == MAX_BLOCK_LINES:
            return lines, True, notes
        lines.append(line)
        bottom = max(w.box[3] for w in line)
    return lines, False, notes


def _wrap_back(row: list[Word], hit: Hit, h: float, right: float, value: list[Word]
               ) -> tuple[Optional[list[Word]], Optional[str]]:
    """A row with text in the label column under a value to the right of its label: is it the
    value wrapped back to the label's margin? (line, None) if so; otherwise (None, a warning
    when the line looks like more of the value but is not joined, or None)."""
    label_h = hit.box[3] - hit.box[1]
    words = [w for w in row if w.xc < right and w.box[2] > hit.box[0] - BLOCK_ALIGN * label_h]
    body = [w for w in words if not w.is_sep]
    if not body:
        return None, None
    line: list[Word] = []
    for w in words:
        if line and not w.is_sep and w.line != line[-1].line and w.box[0] - line[-1].box[2] > 2.5 * h:
            break
        line.append(w)
    line_body = [w for w in line if not w.is_sep]
    is_label = any(w.hit >= 0 for w in line_body)
    has_colon = any(w.is_sep and w.text == ":" for w in line)
    lh = _height(line_body)
    same_size = BLOCK_HEIGHT[0] * h <= lh <= BLOCK_HEIGHT[1] * h
    if line_body[0].box[0] < hit.box[0] - BLOCK_ALIGN * label_h:
        return None, None  # starts left of the label: another column or a heading
    opened = _open_end(value)
    if is_label or has_colon or not same_size:
        if opened and (is_label or has_colon):
            return None, (f"the value ends with {opened} but the next line starts another label; "
                          "it may be incomplete")
        return None, None
    if opened or _starts_lower(line_body):
        return line, None
    return None, ("the line below the value starts at the label's margin and may continue it; "
                  "it was not joined because nothing shows that the value goes on")


def _cut_short(rows: list[list[Word]], hit: Hit, ws: list[Word], how: str) -> list[str]:
    """For a field without `multiline`: a warning when its one-line value is visibly cut, because
    it ends mid-phrase (a comma, "and", "of", an opening bracket ...) or the line below would
    join it as a block with evidence that it goes on."""
    opened = _open_end(ws)
    block, _, _ = _block_lines(rows, hit, ws, how)
    if opened or (len(block) > 1 and _starts_lower(block[1])):
        why = f"ends with {opened}" if opened else "seems to go on on the next line"
        return [f"the value {why}; it may be cut short (if the field can wrap onto more lines, "
                "set `multiline: true` in its YAML)"]
    return []


def _weakest_line_scale(words: list[Word]) -> float:
    """Ratio of the weakest OCR line's confidence to the mean word confidence of a value."""
    body = [w for w in words if not w.is_sep] or words
    mean = sum(w.conf for w in body) / len(body)
    return min(w.conf for w in body) / mean if mean > 0 else 1.0


# ---------------------------------------------------------------------------- values in prose (`until`)

SEG_GAP = 1.5  # boxes on one row closer than this x text height are one visual line
FLOW_GAP = 2.0  # reading order continues to a line at most this x text height below
MAX_FLOW_WORDS = 40  # an `until` value must end within this many words of its label
MAX_FLOW_LINES = 6  # ... and within this many lines
UNTIL_MISSING_FACTOR = 0.7  # end text not found, value cut to its date / pattern


@dataclass
class Segment:
    """A visual line of one column: the words of a row up to a wide horizontal gap."""
    id: int
    row: int
    words: list[Word]
    l: float = 0.0
    t: float = 0.0
    r: float = 0.0
    b: float = 0.0
    h: float = 0.0

    def measure(self) -> None:
        self.l = self.words[0].box[0]
        self.t = min(w.box[1] for w in self.words)
        self.r = max(w.box[2] for w in self.words)
        self.b = max(w.box[3] for w in self.words)
        self.h = _height(self.words)


class Layout:
    """Rows split into segments, and reading order from one line to the next.

    The line after a line is, in the nearest row below (no more than FLOW_GAP text heights
    lower), every segment that overlaps the horizontal extent of the text read so far by at
    least half its own width (or half the extent, if narrower). So a paragraph continues at its
    left margin, a value-below layout continues under its label, and the other column of a
    two-column page is not read as a continuation."""

    def __init__(self, rows: list[list[Word]], page_w: float):
        self.rows = rows
        self.page_w = page_w
        self.segs: list[Segment] = []
        self.row_segs: list[list[Segment]] = []
        for ri, row in enumerate(rows):
            mine: list[Segment] = []
            for w in row:
                if mine:
                    prev = mine[-1].words[-1]
                    if w.line == prev.line or w.box[0] - prev.box[2] <= SEG_GAP * min(w.h, prev.h):
                        w.seg, w.sidx = mine[-1].id, len(mine[-1].words)
                        mine[-1].words.append(w)
                        continue
                seg = Segment(len(self.segs), ri, [w])
                w.seg, w.sidx = seg.id, 0
                self.segs.append(seg)
                mine.append(seg)
            self.row_segs.append(mine)
        for seg in self.segs:
            seg.measure()
        self.row_top = [min((s.t for s in segs), default=0.0) for segs in self.row_segs]
        self._after: dict[int, Optional[list[Segment]]] = {}
        self._next: dict[tuple, Optional[list[Segment]]] = {}
        self._pieces: dict[tuple[int, int], Word] = {}

    def piece(self, w: Word, at: int) -> Word:
        """The rest of a word after a label that ends inside it (one object per word and cut,
        because end-text matches are cached by word identity)."""
        key = (id(w), at)
        if key not in self._pieces:
            self._pieces[key] = _piece(w, at, len(w.text))
        return self._pieces[key]

    def next_line(self, row: int, extent: tuple[float, float], bottom: float, h: float
                  ) -> Optional[list[Segment]]:
        key = (row, extent, bottom, h)
        if key not in self._next:
            self._next[key] = self._next_line(row, extent, bottom, h)
        return self._next[key]

    def _next_line(self, row: int, extent: tuple[float, float], bottom: float, h: float
                   ) -> Optional[list[Segment]]:
        width = max(1.0, extent[1] - extent[0])
        for ri in range(row + 1, len(self.rows)):
            segs = self.row_segs[ri]
            if not segs:
                continue
            if self.row_top[ri] - bottom > FLOW_GAP * h:
                return None
            hits = [s for s in segs
                    if min(s.r, extent[1]) - max(s.l, extent[0]) >= 0.5 * min(width, max(1.0, s.r - s.l))]
            if hits:
                return hits
        return None

    def line_after(self, seg: Segment) -> list[Word]:
        """Words of the line that follows a segment in reading order (cached)."""
        if seg.id not in self._after:
            self._after[seg.id] = self.next_line(seg.row, (seg.l, seg.r), seg.b, seg.h)
        nxt = self._after[seg.id]
        return [w for s in nxt for w in s.words] if nxt else []

    def along_row(self, hit: FlowHit) -> Optional[list[Word]]:
        """A fill-in line ("Birth in the ....KIAMBU.... District in the ....CENTRAL.... Province"):
        when the label ends its segment, the words of the later segments on the same row, read
        across the wide gaps that `flow` does not cross. None when the label does not end its
        segment, or fewer than two segments follow (a value and the end text) within reach
        (each gap at most 0.6 of the page width)."""
        last = hit.words[-1]
        seg = self.segs[last.seg]
        if hit.rcut is not None or any(w.seg != seg.id for w in hit.words):
            return None
        if any(not w.is_sep for w in seg.words[last.sidx + 1:]):
            return None
        out: list[Word] = []
        right, n = seg.r, 0
        for s in self.row_segs[seg.row]:
            if s.l <= seg.r:
                continue
            if s.l - right > 0.6 * self.page_w:
                break
            out += s.words
            right, n = s.r, n + 1
        return out if n >= 2 else None

    def starts_segment(self, w: Word) -> bool:
        """Is this word the first word (separators aside) of its segment?"""
        return all(x.is_sep for x in self.segs[w.seg].words[:w.sidx])

    def flow(self, hit: FlowHit, max_lines: int = MAX_FLOW_LINES) -> Iterable[list[Word]]:
        """Lines of text after a label in reading order: the rest of the label's line first
        (or, for a label alone in its segment, the segment to its right on the same row), then
        the following lines. At most `max_lines` lines."""
        last = hit.words[-1]
        seg = self.segs[last.seg]
        first = seg.words[last.sidx + 1:]
        if hit.rcut is not None:  # the label ends inside its last word: the value starts there
            first = [self.piece(last, hit.rcut.at)] + first
        extent = (seg.l, seg.r)
        row, bottom, h = seg.row, seg.b, seg.h
        if not first:
            alone = all(w.seg == seg.id for w in hit.words) and all(
                w.is_sep for w in seg.words[:hit.words[0].sidx])
            if alone:
                after = [s for s in self.row_segs[seg.row] if s.l > seg.r]
                if after and after[0].l - seg.r <= 0.6 * self.page_w:
                    first = after[0].words
                    extent = (seg.l, after[0].r)
                    bottom = max(bottom, after[0].b)
        yield first
        for _ in range(max_lines - 1):
            nxt = self.next_line(row, extent, bottom, h)
            if not nxt:
                return
            words = [w for s in nxt for w in s.words]
            yield words
            extent = (min(extent[0], min(s.l for s in nxt)), max(extent[1], max(s.r for s in nxt)))
            row = max(s.row for s in nxt)
            bottom = max(s.b for s in nxt)
            h = sum(s.h * len(s.words) for s in nxt) / len(words)


def _variant(text: str, owner: Optional[str] = None) -> Optional[Variant]:
    words = _label_words(text)
    if not words:
        return None
    return Variant(words, "".join(words), text.strip(), {owner} if owner else set())


class Anchors:
    """Label or end-text variants indexed by joined length, for `_match_at`.

    `role` says which edge of an anchor borders the value: the right edge of a label ("label"),
    the left edge of an end text ("end"), or neither ("guard": the run-on check, which accepts
    only splits with evidence at the join, and gives no warnings)."""

    def __init__(self, variants: Iterable[Variant], role: str = "guard"):
        self.variants = list(variants)
        self.role = role
        self.maxw = max((len(v.words) for v in self.variants), default=0) + 1
        self.by_len: dict[int, list[Variant]] = {}
        for v in self.variants:
            self.by_len.setdefault(len(v.joined), []).append(v)

    def __bool__(self) -> bool:
        return bool(self.variants)


@dataclass
class AnchorMatch:
    k: int  # words used
    score: float
    variant: Variant
    lcut: Optional[Cut] = None  # the anchor starts inside its first word
    rcut: Optional[Cut] = None  # the anchor ends inside its last word
    warn: tuple[bool, bool] = (False, False)  # (left, right) split needs a warning


# Punctuation after which, or an opening bracket before which, a lost space is plain to see
# ("Nakuru,Kenya", "Thousand(KES"). Not '/', '-', '&' or an apostrophe, which join the parts
# of one word ("and/or", "Mary-Anne").
_GLUE_AFTER = ",;:.)]!?"
_GLUE_BEFORE = "(["


def _join_evidence(text: str, i: int) -> Optional[str]:
    """What shows that a space was lost between text[:i] and text[i:], or None."""
    a, b = text[i - 1], text[i]
    if a in _GLUE_AFTER and b.isalnum():
        if a in ".,:" and i >= 2 and text[i - 2].isdigit() and b.isdigit():
            return None  # "12.5", "1,200", "10:30"
        return "mark"
    if b in _GLUE_BEFORE and a.isalnum():
        return "mark"
    if not (a.isalnum() and b.isalnum()):
        return None
    if a.isdigit() != b.isdigit():
        return "digit"
    if a.islower() and b.isupper():
        return "case"  # "NumberNAKURU", "theVendor"
    nxt, prev = text[i + 1:i + 2], text[i - 2:i - 1]
    if a.isupper() and b.isupper() and nxt.islower() and prev.isupper():
        return "case"  # "NAKURUThe": capitals, then a capitalised word
    if a.isupper() and b.islower() and prev.isupper() and nxt[:1].isalpha() and nxt.islower():
        return "case"  # "MUTHONIof": capitals, then a lower-case word of two letters or more
    return None


_CAPITALISED = re.compile(r"[\"'“‘(\[]?[A-Z][a-z]+")


def _cuts(w: Word) -> tuple[tuple[Cut, ...], tuple[Cut, ...]]:
    """Places where an anchor may start (first tuple) or end (second) inside an OCR word that
    has text on the other side of the place. Cached on the word."""
    if w.cuts is None:
        lefts: list[Cut] = []
        rights: list[Cut] = []
        text = w.text
        if not w.is_sep and len(w.canon) >= 3:
            for i in range(1, len(text)):
                head, tail = canon(text[:i]), canon(text[i:])
                if not head or not tail:
                    continue
                why = _join_evidence(text, i)
                if why:
                    lefts.append(Cut(i, tail, head, why))
                    rights.append(Cut(i, head, tail, why))
                elif _CAPITALISED.fullmatch(text[:i]) and text[i].islower():
                    lefts.append(Cut(i, tail, head, ""))  # "Thepurchase"
        w.cuts = (tuple(lefts), tuple(rights))
    return w.cuts


MIN_WEAK_GLUE = 4  # a split without evidence at the join needs an anchor word this long


def _glue_ok(v: Variant, side: str, cut: Cut, piece: str, exact: bool, role: str) -> Optional[bool]:
    """Is a split believable for anchor `v`, glued on `side` ("left": the anchor starts inside
    a word; "right": it ends inside one)? None if not; else whether it needs a warning.

    - The part of the word inside the anchor (`piece`, joined to the rest of a word broken by
      a hyphen at a line end) must be exactly the anchor's edge: its whole first or last word,
      possibly with more words OCR merged into it.
    - On the edge that borders the value (a label's right, an end text's left), the value
      starts or ends at the join, so the join must show evidence: punctuation, a letter against
      a figure, or a change of case. A one-word anchor split at a change of case alone must be
      three characters or more ("McDonald" is not "Mc"), and gets a warning there.
    - On a label's left edge, which only decides where the label starts, a capitalised word
      may be glued to a lower-case anchor without other evidence ("Thepurchase price is"), if
      the anchor has two or more words, all matched exactly, and the glued word has at least
      MIN_WEAK_GLUE characters.
    - The run-on guard ("guard") accepts evidence at the join only."""
    edge = v.words[0] if side == "left" else v.words[-1]
    ok_text = v.joined.startswith(piece) if side == "left" else v.joined.endswith(piece)
    if not ok_text or len(piece) < len(edge) or len(edge) < 2:
        return None
    if not cut.why:
        if role == "label" and side == "left" and len(v.words) >= 2 and exact \
                and len(edge) >= MIN_WEAK_GLUE:
            return False
        return None
    if cut.why == "case" and len(v.words) == 1:
        if len(edge) < 3:
            return None
        return (role, side) in (("label", "right"), ("end", "left"))
    return False


def _match_at(content: list[Word], p: int, anchors: Anchors, glue: bool = True,
              per_owner: bool = False):
    """Best variant matching the content words starting at index p.

    Matching is the same as for row labels (edit budget by word length, joined comparison for
    split or merged words, single words of <=5 characters exact), without the rule that a short
    label must start an OCR box: prose anchors sit mid-line. A word broken by a hyphen at a
    line break ("regis-" "tered") is compared both as two words and as one.

    With `glue`, the anchor may also start inside its first word or end inside its last, where
    OCR lost the space between an anchor word and its neighbour ("NumberNAKURU",
    "Box4410-00506", "Thepurchase"); `_glue_ok` decides which splits are believable. A reading
    without a split is preferred to one with a split of equal score.

    With `per_owner`, the result is a list: the best match of each set of owners (fields), best
    first, so that a field's label is found where another field's longer label also matches."""
    best: Optional[AnchorMatch] = None
    best_w = 0.0
    owned: dict[frozenset, tuple[float, AnchorMatch]] = {}
    total = 0
    cand: list[str] = []
    joined: list[str] = []  # the same words with hyphenated line breaks joined
    broken = False
    by_len = anchors.by_len
    role = anchors.role
    first = content[p] if p < len(content) else None
    lefts: list[Optional[Cut]] = [None]
    if glue and first is not None:
        lefts += [c for c in _cuts(first)[0] if role != "guard" or c.why]
    for k in range(1, min(anchors.maxw, len(content) - p) + 1):
        w = content[p + k - 1]
        if k > 1 and not w.alpha and len(w.canon) > 2:
            break  # a number cannot be part of an anchor
        cand.append(w.canon)
        if k > 1 and _hyphen_joins_letters(content[p + k - 2], w):
            joined[-1] += w.canon
            broken = True
        else:
            joined.append(w.canon)
        total += len(w.canon)
        rights: list[Optional[Cut]] = [None]
        if glue:
            rights += [c for c in _cuts(w)[1] if c.why]
        for lc0 in lefts:
            for rc0 in rights:
                lc, rc = lc0, rc0
                forms = [cand, joined] if broken else [cand]
                if lc is not None or rc is not None:
                    if k == 1 and lc is not None and rc is not None:
                        if lc.at >= rc.at:
                            continue
                        mid = canon(w.text[lc.at:rc.at])
                        if not mid:
                            continue
                        lc = dataclasses.replace(lc, anchor=mid)
                        rc = dataclasses.replace(rc, anchor=mid)
                        forms = [[mid]]
                    else:
                        forms = [_recut(f, first, w, lc, rc) for f in forms]
                size = sum(len(x) for x in forms[0])
                for n in range(max(1, size - 3), size + 4):
                    for v in by_len.get(n, ()):
                        if not first.alpha and lc is None and v.lead_alpha:
                            continue  # "7 situated in" is not "situated in" with a typo
                        for c in forms:
                            s = _match_words(c, v)
                            if s is None or s < 0.8:
                                continue
                            warn = [False, False]
                            exact = "".join(c) == v.joined
                            fine = True
                            for side, cut in (("left", lc), ("right", rc)):
                                if cut is None:
                                    continue
                                piece = c[0] if side == "left" else c[-1]
                                r = _glue_ok(v, side, cut, piece, exact, role)
                                if r is None:
                                    fine = False
                                    break
                                warn[side == "right"] = r
                            if not fine:
                                continue
                            weight = s * len(v.joined)
                            if best is None or weight > best_w:
                                best = AnchorMatch(k, s, v, lc, rc, (warn[0], warn[1]))
                                best_w = weight
                            if per_owner:
                                key = frozenset(v.owners)
                                if key not in owned or weight > owned[key][0]:
                                    owned[key] = (weight, AnchorMatch(k, s, v, lc, rc, (warn[0], warn[1])))
    if per_owner:
        return [m for _, m in sorted(owned.values(), key=lambda wm: -wm[0])]
    return best


def _recut(words: list[str], first: Word, last: Word, lc: Optional[Cut], rc: Optional[Cut]) -> list[str]:
    """Canonical words of a candidate with the first word cut at `lc` and the last at `rc`."""
    out = list(words)
    if lc is not None:
        out[0] = lc.anchor + out[0][len(first.canon):]
    if rc is not None:
        out[-1] = out[-1][:len(out[-1]) - len(last.canon)] + rc.anchor
    return out


def find_flow_hits(layout: Layout, variants: list[Variant], first_id: int) -> list[FlowHit]:
    """Labels of `until` fields, searched in reading order so they may cross a line break."""
    if not variants:
        return []
    hits: list[FlowHit] = []
    anchors = Anchors(variants, "label")
    maxw = anchors.maxw
    for seg in layout.segs:
        body = [w for w in seg.words if not w.is_sep]
        if not body:
            continue
        stream = body + [w for w in layout.line_after(seg) if not w.is_sep][:maxw]
        for p, w in enumerate(body):
            if not w.alpha:
                continue
            found = _match_at(stream, p, anchors, per_owner=True)
            for m in found:
                if m is not found[0] and m.variant.owners <= found[0].variant.owners:
                    continue
                # The best match here is the label as before. A field whose own best label here
                # is shorter than another field's (its label is the start of theirs) keeps a
                # hit too, marked as shadowed by the longer label.
                shadow = found[0].variant if m is not found[0] else None
                words = stream[p:p + m.k]
                notes = []
                if m.rcut is not None and m.warn[1]:
                    last = words[-1].text
                    notes.append(f"label '{m.variant.text}' was read inside '{last}', where a space "
                                 f"seems to be missing; the value was taken to start at "
                                 f"'{last[m.rcut.at:]}'")
                hits.append(FlowHit(first_id + len(hits), m.variant, m.score, words, m.rcut, notes,
                                    shadow))
    return hits


@dataclass
class _Scan:
    value: list[Word]  # words between the label and the end text (separators included)
    end: Optional[list[Word]] = None  # the end text's words, when found
    end_score: float = 0.0
    end_variant: Optional[Variant] = None
    region: list[Word] = field(default_factory=list)  # everything read when the end was not found
    notes: list[str] = field(default_factory=list)  # warnings about where the value ends
    sentence: bool = False  # stopped at the end of the sentence before a punctuation end text


# Punctuation marks that may be given alone as end texts (`until: ","`).
END_MARKS = frozenset(",.;:)]([")
_OPENER = {")": "(", "]": "["}
_OPENERS = "(["
# "(s)", "(es)": letters in brackets inside a word are part of it, not the start of a bracket
_WORD_BRACKET = re.compile(r"[(\[][A-Za-z]{1,2}[)\]]")
_CLOSERS = "\"'\u201d\u2019)]"
# English abbreviations that are routinely followed by a capital or a figure ("Mr. Kamau",
# "No. 12", "Ltd. of"). With any capitalised word of one or two letters ("J.", "PO.") and any
# word with a full stop inside ("P.O."), a full stop after them ends a sentence only if what
# follows reads as a new sentence.
_ABBREVIATIONS = frozenset(
    "mr mrs ms mx dr prof rev hon sr jr st messrs esq capt col gen sgt lt maj cpl insp supt "
    "no nos ltd co inc corp plc pty bros dept est approx vs etc viz cf ave rd blvd fig vol ref "
    "tel ext attn".split())
PARA_PITCH = 1.3  # a line spacing this many times the paragraph's own ends a paragraph
ABBREV_WINDOW = 12  # words read past an abbreviation's full stop to see if a sentence began


def _scan_until(lines: Iterable[list[Word]], untils: Anchors,
                memo: Optional[dict] = None, marks: frozenset[str] = frozenset()) -> _Scan:
    """Read words in reading order until one of the end texts; stop after MAX_FLOW_WORDS.
    (Another occurrence of the field's own label does not stop the reading: which occurrence
    is the label is decided from the readings, in `flow_candidates`.)

    `marks` are punctuation end texts (`END_MARKS`). A mark ends the value where it ends a word
    or stands alone; an opening bracket also where it starts the next word or sits inside a
    word, and the value ends just before it. Except: a comma between figures ("1, 200,000"), a
    colon inside a time ("10:30"), a bracket that closes one opened inside the value, letters
    in brackets inside a word ("Purchaser(s)"), and a full stop that does not end a sentence
    (`full_stop`). Marks are looked for after each word, so a bracket that opens the value does
    not end it. An end text that starts inside a word (a glued anchor, `_match_at`) ends the
    value at the join. A full stop end also accepts the end of the paragraph, with a
    warning when the stop itself is missing. A field with other marks cannot run past the end
    of its sentence or paragraph. `lines` may give one line more than MAX_FLOW_LINES: it is
    only looked at, to judge the end of a sentence.

    `memo` caches end-text matches by the words they were tried on, because the scans from
    the hits of a frequent label ("of") can walk over the same text."""
    memo = {} if memo is None else memo
    it = iter(lines)
    buf: list[Word] = []
    content: list[int] = []  # positions in buf of non-separator words
    cwords: list[Word] = []  # those words
    line_of: list[int] = []  # flow line of each content word
    tops: list[Optional[float]] = []  # top of each flow line (None when it has no words)
    rights: list[Optional[float]] = []
    look: list[Word] = []  # words of the line after the last one a value may use
    exhausted = False
    natural_end = False  # the reading ended before the line cap: the end of the paragraph
    need = max(untils.maxw, 2)
    notes: list[str] = []

    def pull(n: int) -> None:
        nonlocal exhausted, natural_end
        while not exhausted and len(content) < n:
            try:
                line = next(it)
            except StopIteration:
                exhausted = natural_end = True
                return
            body = [w for w in line if not w.is_sep]
            tops.append(min(w.box[1] for w in body) if body else None)
            rights.append(max(w.box[2] for w in body) if body else None)
            if len(tops) > MAX_FLOW_LINES:
                look.extend(body)
                exhausted = True
                return
            for w in line:
                if not w.is_sep:
                    content.append(len(buf))
                    cwords.append(w)
                    line_of.append(len(tops) - 1)
                buf.append(w)

    def following(p: int, n: int) -> list[Word]:
        """Up to n content words after position p, the look-ahead line included."""
        pull(p + 1 + n)
        return (cwords[p + 1:] + look)[:n]

    def para_break(p: int, need_short: bool = False) -> bool:
        """Does the paragraph end after word p? Only at the end of a line: the reading ends
        there, or the next line is further below than the paragraph's own line spacing (and,
        with `need_short`, the line stops well short of the others)."""
        if following(p, 1) and p + 1 < len(cwords) and line_of[p + 1] == line_of[p]:
            return False
        i = line_of[p]
        while not exhausted and len(tops) < i + 3:
            pull(len(content) + 1)
        filled = [(k, t) for k, t in enumerate(tops) if t is not None]
        after = [k for k, t in filled if k > i]
        if not after:
            return natural_end
        pos = [k for k, _ in filled].index(i)
        ts = [t for _, t in filled]
        pitch = ts[pos + 1] - ts[pos]
        refs = [ts[pos] - ts[pos - 1]] if pos > 0 else []
        if pos + 2 < len(ts):
            refs.append(ts[pos + 2] - ts[pos + 1])
        refs = [r for r in refs if r > 0]
        if not refs or pitch <= PARA_PITCH * min(refs):
            return False
        if need_short:
            h = _height([cwords[p]])
            return rights[i] < max(r for r in rights if r is not None) - 2 * h
        return True

    def full_stop(p: int, core: str, rest: Optional[str] = None) -> Optional[str]:
        """Does a full stop after word p (written `core` before the stop) end the sentence?
        "end"; "abbrev" when it follows an abbreviation and is taken as the end only because
        a new sentence plainly follows; None when it does not end the sentence. `rest` is the
        text after a stop inside a token, where OCR lost the space ("KAMAU.The")."""
        after = [w.text for w in following(p, ABBREV_WINDOW + 1)]
        if rest is not None:
            after = [rest] + after
        elif not after or para_break(p):
            return "end"
        n = after[0].lstrip("\"'\u201c\u2018([")
        if n[:1].islower():
            return None  # a sentence does not go on in lower case
        word = re.split(r"[,;:/()\[\]]", core)[-1].strip(_CLOSERS + "(['\u201c\u2018")
        letters = sum(ch.isalpha() for ch in word)
        abbrev = letters and ("." in word or (not any(ch.isdigit() for ch in word) and (
            (letters <= 2 and word[:1].isupper()) or word.lower() in _ABBREVIATIONS)))
        if not abbrev:
            return "end"
        if n[:1].isdigit():
            return None  # "No. 12": a sentence does not start with a figure
        # A new sentence: a lower-case word follows before any other punctuation ("LTD. The
        # Vendor warrants"), not "J. K. KAMAU, who" or "P.O. Box 12, Nakuru".
        for k, text in enumerate(after):
            if k and text[:1].islower() and sum(ch.isalpha() for ch in text) >= 2:
                return "abbrev"
            if re.search(r"[.,;:)\]]", text):
                break
        if letters == 2 and "." not in word and word.lower() not in _ABBREVIATIONS and "." in marks:
            notes.append(f"the value runs on past the full stop after '{word}'; check that it does not end there")
        return None

    def done(p: int, value_end: int, end_words: list[Word], mark: str, last: Optional[Word] = None) -> _Scan:
        value = buf[:value_end] + ([last] if last is not None else [])
        while value and value[-1].is_sep:
            value.pop()
        while value and value[0].is_sep:
            value.pop(0)
        return _Scan(value, end_words, 1.0, Variant((), "", mark), notes=list(notes))

    def punctuation(p: int):
        """The end of the value at a mark after word p, "stop" at the end of a sentence for a
        field without a full stop end, or None."""
        w = cwords[p]
        # (text before, mark, cut, token, text after the mark inside the token)
        found: list[tuple[str, str, Optional[int], Word, Optional[str]]] = []
        tail = re.search(r"[^A-Za-z0-9]+$", w.text)
        end = tail.start() if tail else len(w.text)
        # a mark inside the token, where OCR lost the space after it ("KAMAU.The",
        # "Box 12,Nakuru"); never between figures ("12.5", "1,200")
        found += [(w.text[:i], ch, i, w, w.text[i + 1:]) for i, ch in enumerate(w.text[:end])
                  if ch in ".,;)]([" and i and not (w.text[i - 1].isdigit() and w.text[i + 1:i + 2].isdigit())
                  and not (ch in _OPENERS and _WORD_BRACKET.match(w.text, i))]
        found += [(w.text[:i], w.text[i], i, w, None) for i in range(end, len(w.text))]
        stop = content[p + 1] if p + 1 < len(content) else len(buf)
        for t in buf[content[p] + 1:stop]:
            found += [(w.text, ch, None, t, None) for ch in t.text]
        if marks & set(_OPENERS):
            # an opening bracket at the start of the next word ends the value before it
            nxt = following(p, 1)
            lead = re.match(r"[\"'\u201c\u2018]?([(\[])", nxt[0].text) if nxt else None
            if lead and not _WORD_BRACKET.match(nxt[0].text, lead.start(1)):
                found.append((w.text, lead.group(1), None, w, None))
        for before, mark, cut, token, rest in found:
            if mark == ".":
                kind = full_stop(p, before, rest)
                if kind and "." in marks:
                    if kind == "abbrev":
                        notes.append(f"the value ends at the full stop after '{before.strip(_CLOSERS)}', "
                                     "which may be an abbreviation; check where it ends")
                    return _end(p, mark, cut, token)
                if kind:
                    return "stop"
                continue
            if mark not in marks:
                continue
            nxt = following(p, 1)
            n = rest if rest is not None else nxt[0].text if nxt else ""
            if mark == "," and before[-1:].isdigit() and re.match(r"\d{3}(?!\d)", n):
                continue  # thousands separator
            if mark == ":" and before[-1:].isdigit() and n[:1].isdigit() and not token.gap and nxt and not nxt[0].gap:
                continue  # a time
            if mark in _OPENER:
                text = " ".join(x.text for x in cwords[:p]) + " " + before
                if text.count(_OPENER[mark]) > text.count(mark):
                    continue  # closes a bracket opened inside the value
            return _end(p, mark, cut, token)
        if "." in marks and para_break(p, need_short=True):
            notes.append("end text '.' not found; the value was ended at the end of its paragraph")
            return _end(p, ".", None, cwords[p])
        if para_break(p):
            return "stop"
        return None

    def _end(p: int, mark: str, cut: Optional[int], token: Word) -> _Scan:
        w = cwords[p]
        if cut is None:
            return done(p, content[p] + 1, [token], mark)
        return done(p, content[p], [w], mark, _piece(w, 0, cut))

    p = 0
    while True:
        pull(p + need)
        if p >= len(content) or p > MAX_FLOW_WORDS:
            return _Scan([], region=buf[:content[min(p, len(content)) - 1] + 1] if content else [])
        if untils:
            window = cwords[p:p + need]
            key = (id(window[0]), id(window[-1]), len(window))
            if key in memo:
                m = memo[key]
            else:
                m = memo[key] = _match_at(cwords, p, untils)
            if m:
                start = content[p]
                value = buf[:start]
                while value and value[-1].is_sep:
                    value.pop()
                notes_here = list(notes)
                if m.lcut is not None:  # the end text starts inside a word: the value ends there
                    w = cwords[p]
                    value.append(_piece(w, 0, m.lcut.at))
                    if m.warn[0]:
                        notes_here.append(
                            f"end text '{m.variant.text}' was read inside '{w.text}', where a space "
                            f"seems to be missing; the value was taken to end at '{w.text[:m.lcut.at]}'")
                while value and value[-1].is_sep:
                    value.pop()
                while value and value[0].is_sep:
                    value.pop(0)
                return _Scan(value, [buf[i] for i in content[p:p + m.k]], m.score, m.variant,
                             notes=notes_here)
        if marks:
            r = punctuation(p)
            if r == "stop":
                return _Scan([], region=buf[:content[p] + 1], sentence=True)
            if r is not None:
                return r
        p += 1


def _along_row(layout: Layout, hit: FlowHit, ends: Anchors) -> Optional[_Scan]:
    """The reading of a fill-in line (`Layout.along_row`), if it is one: the label ends its
    segment, and a word end text starts a later segment on the same row, with at least one
    word between them. Every gap of the row then bounds a piece of printed text or of the
    entry, so the entry is exactly the segments between the two. None otherwise, and the
    field is read in sentence order as usual."""
    words = layout.along_row(hit)
    if words is None:
        return None
    scan = _scan_until([words], ends, {})
    if scan.end is None or not any(not w.is_sep for w in scan.value):
        return None
    if not layout.starts_segment(scan.end[0]):
        return None
    last = scan.value[-1]
    if last.span is not None and last.parts and last.parts[0] is scan.end[0]:
        return None  # the end text was found inside a word, not at the start of a segment
    return scan


def _contains_anchor(words: list[Word], anchors: Anchors) -> Optional[Variant]:
    content = [w for w in words if not w.is_sep]
    if not anchors or not content:
        return None
    for p in range(len(content)):
        m = _match_at(content, p, anchors)
        if m:
            return m.variant
    return None


def _distinctive(v: Variant) -> bool:
    """Anchors specific enough to show that a value ran past its end: two or more words, or
    one word of six characters or more. A short word ("of") can be part of any value."""
    return len(v.words) >= 2 or len(v.joined) >= 6


def flow_candidates(layout: Layout, specs: list[FieldSpec], wanted: set[str], first_hit_id: int,
                    row_vocab: Iterable[Variant] = (), lexicon: Optional[Lexicon] = None
                    ) -> tuple[dict[str, Candidate], dict[str, str], dict[str, FlowHit]]:
    """For the wanted `until` fields: one candidate (or none) each, the reason for each field
    left without one, and the first label hit of each field (for messages). `specs` holds every
    `until` field of the type, because their anchors bound each other's values.

    For each hit of the field's label, the text after it is read in reading order up to the
    end text. When several hits reach the same end, the nearest is the label, provided every
    longer reading (from an earlier hit, through the nearer one) contains an anchor of
    something else, of any length: the "of" of "day of March ... between JOHN KAMAU of P.O.
    Box 1" is not the label of the address. A longer reading without one is another possible
    value, and the field gets no value (`nearest`). Hits of a label that is the start of
    another field's longer label at the same place (`FlowHit.shadow`) are read only when the
    field's other hits give nothing, and their value carries a warning.
    A reading that contains a distinctive anchor of something else (another field's label or
    end text, or one of the type's other_labels; see `_distinctive`) ran past its own end and
    is discarded. Different values from different occurrences make the field ambiguous, and it
    gets no value.

    End texts that are a single punctuation mark (`END_MARKS`) are found by `_scan_until`.
    Words broken by a hyphen at a line break are joined in the value (`dehyphenate`)."""
    lexicon = lexicon or Lexicon()
    row_vocab = list(row_vocab)
    by_words: dict[tuple[str, ...], Variant] = {}
    untils: dict[str, list[Variant]] = {}
    marks: dict[str, frozenset[str]] = {}
    for spec in specs:
        for label in spec.labels():
            v = _variant(label, spec.name)
            if v:
                by_words.setdefault(v.words, v).owners.add(spec.name)
        untils[spec.name] = [v for v in (_variant(t) for t in spec.until_texts()) if v]
        marks[spec.name] = frozenset(t for t in spec.until_texts() if t in END_MARKS)
    variants = list(by_words.values())
    # Every field's labels are searched, wanted or not, so that a label inside another field's
    # longer label is known to be shadowed by it whichever fields were asked for.
    hits = [hh for hh in find_flow_hits(layout, variants, first_hit_id) if hh.variant.owners & wanted]
    # Distinctive anchors for the run-on guard: each field's labels and end texts, and the row
    # labels (labels of the other fields, and the type's other_labels).
    anchors: dict[str, list[Variant]] = {
        spec.name: [v for v in variants if spec.name in v.owners and _distinctive(v)]
        + [v for v in untils[spec.name] if _distinctive(v)] for spec in specs}
    for v in row_vocab:
        if _distinctive(v):
            for owner in sorted(v.owners) or ["<other_labels>"]:
                anchors.setdefault(owner, []).append(v)

    indexed = {name: Anchors(vs) for name, vs in anchors.items()}
    # Every anchor with letters, short ones included: a longer reading through a repeated label
    # that contains none of them is a second possible value (`nearest` below).
    all_anchors = variants + [v for vs in untils.values() for v in vs] + list(row_vocab)

    def runs_on(name: str, words: list[Word]) -> Optional[tuple[str, Variant]]:
        for other, vs in indexed.items():
            if other != name:
                v = _contains_anchor(words, vs)
                if v is not None:
                    return other, v
        return None

    chosen: dict[str, Candidate] = {}
    notes: dict[str, str] = {}
    first_hit: dict[str, FlowHit] = {}
    for spec in specs:
        name = spec.name
        if name not in wanted:
            continue
        mine = [hh for hh in hits if name in hh.variant.owners]
        if mine:
            first_hit[name] = mine[0]
        if not untils[name] and not marks[name]:
            notes[name] = ("end text ('until') has no letters or digits and is not one of the "
                           f"punctuation marks {' '.join(sorted(END_MARKS))}, so it cannot be matched")
            continue
        if not mine:
            continue
        end_text = "' / '".join([v.text for v in untils[name]]
                                + [t for t in spec.until_texts() if t in marks[name]])
        end_anchors = Anchors(untils[name], "end")
        own_words = {v.words for v in variants if name in v.owners}
        loose = Anchors([v for v in all_anchors if v.words not in own_words and not (v.owners & {name})])
        memo: dict = {}

        def read(group: list[FlowHit]):
            """Readings from these hits: ({end occurrence: [(candidate, value length, value
            words)]}, date/pattern fallbacks, the first run-on, whether a sentence stopped one)."""
            found: dict[int, list[tuple[Candidate, int, list[Word]]]] = {}
            fallback: list[Candidate] = []
            run_on: Optional[tuple[str, Variant]] = None
            sentence_stop = False
            for hh in group:
                scan = _along_row(layout, hh, end_anchors) if end_anchors else None
                if scan is None:
                    scan = _scan_until(layout.flow(hh, MAX_FLOW_LINES + 1), end_anchors, memo, marks[name])
                sentence_stop = sentence_stop or scan.sentence
                if scan.end is not None:
                    if not any(not w.is_sep for w in scan.value):
                        continue
                    crossed = runs_on(name, scan.value)
                    if crossed:
                        run_on = run_on or crossed
                        continue
                    value, joined = dehyphenate(scan.value, lexicon)
                    sub, raw, fit, warns = assess(value, spec)
                    if not raw:
                        continue
                    warns = list(warns) + hh.notes + scan.notes + joined
                    if hh.score < 0.9:
                        warns.append(f"label '{hh.variant.text}' matched loosely")
                    if scan.end_score < 0.9:
                        warns.append(f"end text '{scan.end_variant.text}' matched loosely")
                    if hh.shadow is not None:
                        warns.append(f"label '{hh.variant.text}' was found only as the start of the "
                                     f"longer label '{hh.shadow.text}'; check that the value is not "
                                     "part of that label")
                    ocr = min(w.conf for w in sub)  # the weakest line
                    c = Candidate(name, sub, raw, ocr * hh.score * scan.end_score * fit, hh, warns,
                                  "flow", ends=list(scan.end))
                    found.setdefault(id(scan.end[0]), []).append((c, len(scan.value), scan.value))
                elif (spec.type == FieldType.DATE or _compile(spec.pattern) is not None) and scan.region:
                    # End text not found: accept a date / pattern match only if it starts the text.
                    sub, raw, fit, warns = assess(scan.region, spec)
                    content = [w for w in scan.region if not w.is_sep]
                    if raw and fit >= 0.95 and sub and sub[0] in content[:2]:
                        crossed = runs_on(name, sub)
                        if crossed:
                            run_on = run_on or crossed
                            continue
                        ocr = min(w.conf for w in sub)
                        fallback.append(Candidate(
                            name, sub, raw, ocr * hh.score * fit * UNTIL_MISSING_FACTOR, hh,
                            [f"end text '{end_text}' not found; value cut to its "
                             f"{'date' if spec.type == FieldType.DATE else 'format'}"], "flow"))
            return found, fallback, run_on, sentence_stop

        def nearest(found) -> tuple[list[Candidate], Optional[str]]:
            """One candidate per end occurrence: the reading from the label nearest before it.
            A longer reading from an earlier occurrence of the label that runs through the
            nearer one is another possible value, unless it contains an anchor of anything
            else (then it ran on from an unrelated occurrence, as the "of" of "day of March"
            before "of P.O. Box 1"). If one remains, which occurrence is the label cannot be
            told, and the field gets no value."""
            out = []
            for readings in found.values():
                readings.sort(key=lambda r: (r[1], -r[0].score))
                best = readings[0][0]
                for c, _, words in readings[1:]:
                    if c.raw.split() != best.raw.split() and _contains_anchor(words, loose) is None:
                        return [], (f"label '{best.hit.variant.text}' occurs more than once before the "
                                    f"end text '{end_text}', and the value could start after either "
                                    f"(readings of {len(best.raw.split())} and {len(c.raw.split())} "
                                    "words); make the label longer")
                out.append(best)
            return out, None

        own = [hh for hh in mine if hh.shadow is None]
        found, fallback, run_on, sentence_stop = read(own)
        pool, ambiguous = nearest(found)
        pool = pool or ([] if ambiguous else fallback)
        if not pool and not ambiguous:
            shadowed = [hh for hh in mine if hh.shadow is not None]
            if shadowed:
                found2, fallback2, run_on2, stop2 = read(shadowed)
                pool, ambiguous = nearest(found2)
                pool = pool or ([] if ambiguous else fallback2)
                run_on = run_on or run_on2
                sentence_stop = sentence_stop or stop2
        distinct = {re.sub(r"\s+", " ", c.raw).strip().upper() for c in pool}
        if ambiguous:
            notes[name] = ambiguous
        elif not pool and run_on:
            where = "the type's other_labels" if run_on[0] == "<other_labels>" else f"field '{run_on[0]}'"
            notes[name] = (f"value would run into '{run_on[1].text}' (an anchor of {where}); "
                           f"its own end text '{end_text}' was probably misread")
        elif not pool:
            where = " before the end of the sentence or" if sentence_stop else ""
            notes[name] = (f"label '{mine[0].variant.text}' found but not its end text '{end_text}'"
                           f"{where} within {MAX_FLOW_WORDS} words")
        elif len(distinct) > 1:
            notes[name] = (f"label '{mine[0].variant.text}' and end text '{end_text}' occur more than once "
                           f"with different text between them ({len(distinct)} readings); "
                           "make the label or end text longer")
        else:
            chosen[name] = min(pool, key=lambda c: (min(w.line for w in c.words), -c.score))

    return chosen, notes, first_hit


# ---------------------------------------------------------------------------- main entry

def reading_order_text(rows: list[list[Word]], lines: list[OcrLine]) -> str:
    out = []
    for row in rows:
        seen: list[int] = []
        for w in row:
            if w.line not in seen:
                seen.append(w.line)
        out.append("   ".join(lines[li].text for li in seen))
    return "\n".join(out)


def _anchor_glue(doc_type: DocType, fields: list[FieldSpec]) -> tuple[frozenset[str], frozenset[str]]:
    """Words OCR may glue to a prose value: the last word of each `until` field's label (three
    characters or more) and the first word of each end text (five or more)."""
    heads: set[str] = set()
    tails: set[str] = set()
    for spec in _all_specs(doc_type, fields):
        if spec.until_texts():
            heads |= {ws[-1] for ws in map(_label_words, spec.labels()) if ws and len(ws[-1]) >= 3}
            tails |= {ws[0] for ws in map(_label_words, spec.until_texts()) if ws and len(ws[0]) >= 5}
    return frozenset(heads), frozenset(tails)


def match_fields(
    lines: list[OcrLine],
    doc_type: DocType,
    fields: list[FieldSpec],
    page_size: Optional[tuple[float, float]] = None,
) -> tuple[list[FieldMatch], str]:
    """Return one FieldMatch per requested field (same order) and the page text."""
    vocab = build_vocabulary(doc_type, fields)
    words = tokenize(lines, vocab, _anchor_glue(doc_type, fields))
    rows = group_rows(lines, words)
    stack_notes: dict[str, str] = {}
    hits = find_label_hits(rows, vocab, stack_notes)
    page_w = page_size[0] if page_size else max((l.box[2] for l in lines), default=1.0)
    specs = {f.name: f for f in fields}
    lexicon = (build_lexicon(words, doc_type, fields)
               if any(f.multiline or f.until_texts() for f in fields) else Lexicon())

    candidates: list[Candidate] = []
    labelled: dict[str, Hit | FlowHit] = {}
    fragments: dict[str, Hit] = {}
    for hit in hits:
        for owner in hit.variant.owners & set(specs):
            spec = specs[owner]
            if hit.fragment_of is not None:
                fragments.setdefault(owner, hit)
                continue
            if owner not in labelled or hit.score > labelled[owner].score:
                labelled[owner] = hit
            block_words: set[int] = set()
            for how, geom, view, ws in _value_places(rows, hit, page_w):
                if not ws:
                    continue
                extra: list[str] = []
                scale = 1.0
                if spec.multiline:
                    if how == "below" and id(ws[0]) in block_words:
                        continue  # the line below is the second line of the value to the right
                    block, capped, cut = _block_lines(rows, view, ws, how)
                    ws = [w for line in block for w in line]
                    block_words |= {id(w) for w in ws}
                    if capped:
                        extra.append(f"value may continue past {MAX_BLOCK_LINES} lines; "
                                     f"only the first {MAX_BLOCK_LINES} were read")
                    extra += cut
                    ws, joined = dehyphenate(ws, lexicon)
                    extra += joined
                else:
                    extra += _cut_short(rows, view, ws, how)
                sub, raw, fit, warns = assess(ws, spec)
                if not raw:
                    continue
                ocr = sum(w.conf for w in sub) / len(sub)
                score = ocr * hit.score * geom * fit
                if spec.multiline:
                    scale = _weakest_line_scale(sub)
                warns = list(warns) + extra
                if hit.score < 0.9:
                    warns.append(f"label '{hit.variant.text}' matched loosely")
                if hit.lone:
                    lone = " / ".join(f"'{t}'" for t in hit.lone)
                    warns.append(f"label '{hit.variant.text}' was read over {len(hit.parts)} lines that "
                                 f"are also labels on their own ({lone}); check that the value is this "
                                 "field's")
                candidates.append(Candidate(owner, sub, raw, score, hit, warns, how, scale))

    # Fields with an end text (`until`) are read in reading order; see `flow_candidates`.
    flow_specs = [s for s in _all_specs(doc_type, fields) if s.until_texts()]
    flow_notes: dict[str, str] = {}
    if any(s.until_texts() for s in fields):
        layout = Layout(rows, page_w)
        flow_chosen, flow_notes, flow_first = flow_candidates(layout, flow_specs, set(specs), len(hits),
                                                              vocab, lexicon)
        for name, c in flow_chosen.items():
            if name in specs:
                candidates.append(c)
        for name, fh in flow_first.items():
            if name in specs:
                labelled.setdefault(name, fh)

    # Greedy assignment; a word can serve two fields only if both read it via the same label.
    candidates.sort(key=lambda c: -c.score)
    # id(OCR word) -> [(first char, end char, hit id (or -2 for pattern))]: a piece of a glued
    # word claims only its own characters.
    claimed: dict[int, list[tuple[int, int, int]]] = {}
    chosen: dict[str, Candidate] = {}
    bumped: dict[str, Candidate] = {}
    for c in candidates:
        if c.field in chosen:
            continue
        hid = c.hit.id if c.hit else -2
        keys = list(_claim_keys(c.words))
        if any(a < e and s < b and h != hid
               for k, a, b in keys for s, e, h in claimed.get(k, ())):
            bumped.setdefault(c.field, c)
            continue
        chosen[c.field] = c
        for k, a, b in keys:
            claimed.setdefault(k, []).append((a, b, hid))
    _flag_repeated_labels(candidates, chosen, claimed)

    results: list[FieldMatch] = []
    for spec in fields:
        c = chosen.get(spec.name)
        if c is None:
            c = _pattern_fallback(spec, rows, claimed)
        res = FieldMatch(spec.name)
        if c is None:
            if spec.name in flow_notes:
                res.warnings.append(flow_notes[spec.name])
            elif spec.name in bumped:
                res.warnings.append("value next to the label was already taken by another field")
            elif spec.name in labelled:
                res.warnings.append(
                    f"label '{labelled[spec.name].variant.text}' found but no value next to or below it")
            elif spec.name in stack_notes:
                res.warnings.append(stack_notes[spec.name])
            elif spec.name in fragments:
                fh = fragments[spec.name]
                res.warnings.append(
                    f"label '{fh.variant.text}' found only as the start of the longer label "
                    f"'{fh.fragment_of.text}', whose words follow it; they are not this field's value")
            elif _compile(spec.pattern) is not None:
                res.warnings.append("label not found and no unambiguous value matches the format")
            else:
                res.warnings.append("label not found")
            results.append(res)
            continue
        if c.how == "pattern":
            for k, a, b in _claim_keys(c.words):
                claimed.setdefault(k, []).append((a, b, -2))
            if spec.name in flow_notes:
                c.warnings = [flow_notes[spec.name], "value located by its format"]
            elif spec.name in labelled or spec.name in bumped:
                # The label was on the page; say what actually happened.
                c.warnings = ["no usable value beside the label; value located by its format"]
        conf = round(max(0.0, min(1.0, c.score * c.conf_scale)), 3)
        if _filler(c.raw):
            res.warnings += [*c.warnings, f"the entry is a row of '{c.raw.strip()[0]}', which strikes out or "
                             "fills a blank; no value"]
            results.append(res)
            continue
        if conf < REJECT_BELOW:
            res.confidence = conf
            res.warnings += [*c.warnings, f"weak match rejected (confidence {conf:.2f})"]
            results.append(res)
            continue
        res.raw, res.confidence = c.raw, conf
        res.box = _union(w.box for w in c.words)
        res.parts = [(w.line, w.box) for w in _leaves(c.words) if not w.is_sep]
        label = (c.hit.words if isinstance(c.hit, FlowHit)
                 else _hit_words(rows, c.hit) if isinstance(c.hit, Hit) else [])
        res.anchor_lines = sorted({w.line for w in _leaves([*label, *c.ends])})
        res.warnings = list(c.warnings)
        if conf < WARN_BELOW:
            res.warnings.append(f"low confidence ({conf:.2f})")
        results.append(res)
    return results, reading_order_text(rows, lines)


FILLER_MIN = 4  # a value of one mark repeated this many times or more fills a blank


def _filler(raw: str) -> bool:
    """A row of one mark ("XXXXXXXX", "--------", "********"): how a blank in a form is struck
    out or filled so nothing can be added. It is not a value."""
    s = re.sub(r"\s+", "", raw)
    return len(s) >= FILLER_MIN and len(set(s.lower())) == 1 and s[0] in "xX*-=~_/"


REPEAT_RIVAL = 0.8  # another occurrence's value scoring this share of the chosen one is a rival


def _flag_repeated_labels(candidates: list[Candidate], chosen: dict[str, Candidate],
                          claimed: dict[int, list]) -> None:
    """A form field whose label occurs more than once on the page, with a different value
    beside another occurrence that no other field took and that scores nearly as well, gets a
    warning: which occurrence is the field's cannot be told from the page."""
    for name, c in chosen.items():
        if not isinstance(c.hit, Hit):
            continue
        mine = {re.sub(r"\s+", " ", c.raw).strip().upper()}
        for o in candidates:
            if o.field != name or not isinstance(o.hit, Hit) or o.hit.id == c.hit.id:
                continue
            if o.score < REPEAT_RIVAL * c.score or re.sub(r"\s+", " ", o.raw).strip().upper() in mine:
                continue
            if any(h != o.hit.id for k, _, _ in _claim_keys(o.words) for _, _, h in claimed.get(k, ())):
                continue  # those words are another field's value
            c.warnings = [*c.warnings, f"label '{c.hit.variant.text}' occurs more than once on the page "
                          "with a different value beside another occurrence; check the value"]
            break


def _pattern_fallback(spec: FieldSpec, rows: list[list[Word]], claimed: dict[int, list]) -> Optional[Candidate]:
    regex = _compile(spec.pattern)
    if regex is None:
        return None
    found: dict[str, Candidate] = {}
    for row in rows:
        free = [w for w in row if w.hit < 0 and not w.is_sep and id(w) not in claimed]
        # contiguous runs within one OCR line, so a match cannot straddle unrelated boxes
        for _, grp in itertools.groupby(free, key=lambda w: w.line):
            grp = list(grp)
            raw = _join(grp)
            for start in range(len(grp)):
                for end in range(start + 1, min(len(grp), start + 4) + 1):
                    sub = grp[start:end]
                    text = _join(sub)
                    if regex.fullmatch(text) or regex.fullmatch(text.replace(" ", "")):
                        key = re.sub(r"\s+", "", text).upper()
                        ocr = sum(w.conf for w in sub) / len(sub)
                        prev = found.get(key)
                        if prev is None or len(prev.words) > len(sub):
                            found[key] = Candidate(
                                spec.name, sub, text, ocr * PATTERN_FALLBACK_FACTOR, None,
                                ["label not found; value located by its format"], "pattern")
            del raw
    if len(found) == 1:
        return next(iter(found.values()))
    return None
