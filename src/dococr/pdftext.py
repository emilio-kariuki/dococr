"""The text layer of a PDF page, turned into text lines the matcher can read in place of OCR.

A PDF exported from a word processor holds its text exactly, with positions. Reading that is
faster than OCR and cannot drop a space or confuse `0` with `O`. But not every text layer can be
trusted, so each page is judged from evidence before its text is used (`build_layer`):

- **Invisible text** (render mode 3 or 7) is what a scanner's "searchable PDF" lays over the page
  image: someone else's OCR, of unknown quality. Under `auto` it is ignored and the page image is
  read by our own OCR.
- **Characters without a usable Unicode mapping**: PDFium reports a mapping error, or the
  character is a control character, U+FFFD, or a private-use code point inside a word. A line
  holding one is not used; its area is read by OCR. A private-use character standing alone (a
  bullet from a symbol font) is dropped.
- **Text that does not show**: a line whose area on the rendered page has no ink (white text, text
  under an opaque box) or that is drawn under a picture is dropped.
- **Text drawn at an angle** on the rendered page (including a page whose `/Rotate` turns its text
  sideways) is not laid out as lines; its area is read by OCR.
- **Pictures and annotations** (a pasted scan, a signature, a stamp, a filled form field) may
  hold text the layer lacks; their areas are read by OCR.

When more than `MAX_DOUBTFUL` of the visible characters are untrusted or turned, the whole page
is read by OCR instead. Otherwise the page is read from its text layer, with OCR only for the
areas listed above ("hybrid"), or with no OCR at all when there are none.

Lines are built to look like OCR lines: characters are grouped into visual lines by their
position on the rendered page (not by the order of the content stream), split at horizontal gaps
wider than `SPLIT_GAP` text heights the way the detector splits columns and form cells, and words
are separated by a space where the PDF has a space character or a gap of `SPACE_GAP` text heights.
Boxes are in the pixel coordinates of the page as rendered, so the rest of the pipeline is
unchanged. A hyphen PDFium marks as a line-end hyphen is returned as "-", as printed.

Reading the characters needs the open PDFium page and runs under the caller's lock
(`read_text_layer`); judging them and building lines is plain Python (`build_layer`).
"""

from __future__ import annotations

import ctypes
import math
import statistics
import unicodedata
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

from dococr.ocr import Box, OcrLine

PDF_TEXT_MODES = ("auto", "always", "never")

SPLIT_GAP = 1.0  # a gap wider than this x text height splits a visual line into two lines
SPACE_GAP = 0.15  # a gap wider than this x text height is a word space
MAX_DOUBTFUL = 0.2  # above this share of untrusted or turned characters, OCR the whole page
MIN_PICTURE_AREA = 0.01  # pictures smaller than this share of the page are not read by OCR
WHOLE_PAGE_AREA = 0.6  # OCR areas covering more than this share of the page: OCR it whole
MAX_PAGE_CHARS = 20_000  # more characters than this on one page: not a normal page, OCR it
MIN_INK_STD = 6.0  # grey-level spread under a line below this: nothing is drawn there
UPRIGHT_TOLERANCE = 0.035  # radians (2 degrees)
CONF_TEXT = 1.0  # confidence of a line from a trusted text layer
CONF_INVISIBLE = 0.9  # ... of a line of invisible text (only with pdf_text="always")
CONF_UNMAPPED = 0.5  # ... of a line with characters that have no Unicode mapping ("always")

# Annotation subtypes whose appearance can carry text the text layer does not have.
_TEXT_ANNOTS = ("FPDF_ANNOT_FREETEXT", "FPDF_ANNOT_STAMP", "FPDF_ANNOT_INK", "FPDF_ANNOT_WIDGET")
_LIGATURES = range(0xFB00, 0xFB07)
_ZERO_WIDTH = {"\u200b", "\u200c", "\u200d", "\u2060", "\ufeff"}


@dataclass
class PdfChar:
    ch: str
    box: Box  # loose character box in the rendered page's pixels
    visible: bool = True  # not drawn with an invisible render mode
    mapped: bool = True  # PDFium reports no Unicode mapping error
    upright: bool = True  # reads left to right, upright, on the rendered page
    z: int = 0  # drawing order of the page object it belongs to
    space: bool = False  # a space or a line break character (real or generated)


@dataclass
class PdfText:
    """What was read from one PDF page: characters, pictures and annotations, all in the
    rendered page's pixel coordinates."""
    size: tuple[int, int]
    chars: list[PdfChar] = field(default_factory=list)
    pictures: list[tuple[Box, int]] = field(default_factory=list)  # box, drawing order
    annotations: list[Box] = field(default_factory=list)


@dataclass
class TextLayer:
    """The decision for one page. `kind` is "text" (lines only), "hybrid" (lines, plus OCR of
    `regions`) or "ocr" (do not use the text layer). `notes` are page-level warnings."""
    kind: str
    lines: list[OcrLine] = field(default_factory=list)
    regions: list[Box] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)


# ---------------------------------------------------------------------------- reading (PDFium)

def _addr(handle) -> int:
    return ctypes.cast(handle, ctypes.c_void_p).value or 0


def _page_to_device(page, size: tuple[int, int]):
    """The affine map from page space to the pixels of `page.render(scale)` (no extra rotation,
    no crop), which takes the CropBox and /Rotate into account. Built from PDFium's own
    FPDF_PageToDevice on an enlarged device for sub-pixel precision."""
    import pypdfium2.raw as pr

    k = 64
    w, h = size

    def dev(x: float, y: float) -> tuple[float, float]:
        dx, dy = ctypes.c_int(), ctypes.c_int()
        pr.FPDF_PageToDevice(page, 0, 0, w * k, h * k, 0, x, y, dx, dy)
        return dx.value / k, dy.value / k

    o = dev(0.0, 0.0)
    ex = dev(1000.0, 0.0)
    ey = dev(0.0, 1000.0)
    a, b = (ex[0] - o[0]) / 1000, (ex[1] - o[1]) / 1000
    c, d = (ey[0] - o[0]) / 1000, (ey[1] - o[1]) / 1000
    return a, b, c, d, o[0], o[1]


def _apply(m, x: float, y: float) -> tuple[float, float]:
    a, b, c, d, e, f = m
    return a * x + c * y + e, b * x + d * y + f


def _device_box(m, left: float, bottom: float, right: float, top: float) -> Box:
    pts = [_apply(m, x, y) for x in (left, right) for y in (bottom, top)]
    xs, ys = [p[0] for p in pts], [p[1] for p in pts]
    return (min(xs), min(ys), max(xs), max(ys))


def read_text_layer(page, size: tuple[int, int]) -> PdfText:
    """Read the characters, pictures and text-bearing annotations of an open pypdfium2 page.
    `size` is the size of the bitmap the page was rendered to. Call with PDFium's lock held."""
    import pypdfium2.raw as pr

    m = _page_to_device(page, size)
    out = PdfText(size=size)

    # Drawing order of every page object, and pictures. A form XObject counts as one object;
    # one that contains a picture is treated as a picture.
    order: dict[int, int] = {}

    def walk(obj, z: int, depth: int = 0) -> bool:
        order[_addr(obj)] = z
        kind = pr.FPDFPageObj_GetType(obj)
        if kind == pr.FPDF_PAGEOBJ_IMAGE:
            return True
        if kind == pr.FPDF_PAGEOBJ_FORM and depth < 8:
            has = False
            for i in range(pr.FPDFFormObj_CountObjects(obj)):
                has = walk(pr.FPDFFormObj_GetObject(obj, i), z, depth + 1) or has
            return has
        return False

    for z in range(pr.FPDFPage_CountObjects(page)):
        obj = pr.FPDFPage_GetObject(page, z)
        if walk(obj, z):
            l, b, r, t = ctypes.c_float(), ctypes.c_float(), ctypes.c_float(), ctypes.c_float()
            if pr.FPDFPageObj_GetBounds(obj, l, b, r, t):
                out.pictures.append((_device_box(m, l.value, b.value, r.value, t.value), z))

    wanted = {getattr(pr, n) for n in _TEXT_ANNOTS if hasattr(pr, n)}
    hidden = getattr(pr, "FPDF_ANNOT_FLAG_HIDDEN", 2) | getattr(pr, "FPDF_ANNOT_FLAG_NOVIEW", 32)
    for i in range(pr.FPDFPage_GetAnnotCount(page)):
        annot = pr.FPDFPage_GetAnnot(page, i)
        if not annot:
            continue
        try:
            if pr.FPDFAnnot_GetSubtype(annot) in wanted and not pr.FPDFAnnot_GetFlags(annot) & hidden:
                rect = pr.FS_RECTF()
                if pr.FPDFAnnot_GetRect(annot, rect):
                    out.annotations.append(_device_box(m, rect.left, rect.bottom, rect.right, rect.top))
        finally:
            pr.FPDFPage_CloseAnnot(annot)

    textpage = pr.FPDFText_LoadPage(page)
    if not textpage:
        return out
    try:
        modes: dict[int, int] = {}
        rect = pr.FS_RECTF()
        mat = pr.FS_MATRIX()
        for i in range(pr.FPDFText_CountChars(textpage)):
            code = pr.FPDFText_GetUnicode(textpage, i)
            generated = pr.FPDFText_IsGenerated(textpage, i) == 1
            if code == 2 and pr.FPDFText_IsHyphen(textpage, i) == 1:
                ch = "-"  # PDFium's mark for a hyphen at the end of a line: printed as "-"
            else:
                ch = chr(code) if 0 <= code <= 0x10FFFF else "�"
            if generated or ch.isspace():
                if out.chars:
                    out.chars[-1].space = True  # a word break after the previous character
                continue
            if not pr.FPDFText_GetLooseCharBox(textpage, i, rect):
                continue
            box = _device_box(m, rect.left, rect.bottom, rect.right, rect.top)
            obj = pr.FPDFText_GetTextObject(textpage, i)
            key = _addr(obj)
            if key not in modes:
                modes[key] = pr.FPDFTextObj_GetTextRenderMode(obj) if obj else 0
            visible = modes[key] not in (pr.FPDF_TEXTRENDERMODE_INVISIBLE, pr.FPDF_TEXTRENDERMODE_CLIP)
            upright = True
            if pr.FPDFText_GetMatrix(textpage, i, mat):
                # baseline and up directions of the glyph, on the rendered page (y points down)
                bx, by = m[0] * mat.a + m[2] * mat.b, m[1] * mat.a + m[3] * mat.b
                ux, uy = m[0] * mat.c + m[2] * mat.d, m[1] * mat.c + m[3] * mat.d
                upright = (bx > 0 and abs(math.atan2(by, bx)) <= UPRIGHT_TOLERANCE and uy < 0
                           and abs(math.atan2(-ux, -uy)) <= UPRIGHT_TOLERANCE)
            out.chars.append(PdfChar(
                ch, box, visible=visible, mapped=pr.FPDFText_HasUnicodeMapError(textpage, i) != 1,
                upright=upright, z=order.get(key, -1)))
    finally:
        pr.FPDFText_ClosePage(textpage)
    return out


# ---------------------------------------------------------------------------- judging and lines

def _bad(ch: str) -> bool:
    """A character that is not real text: no mapping (control character, replacement
    character, a non-character) or a private-use code point."""
    cat = unicodedata.category(ch)
    return cat in ("Cc", "Co", "Cs", "Cn") or ch in "�￾￿"


def _clean(ch: str) -> str:
    if ord(ch) in _LIGATURES:
        return unicodedata.normalize("NFKC", ch)
    if ch in _ZERO_WIDTH:
        return ""
    return ch


@dataclass
class _Seg:
    chars: list[PdfChar]

    @property
    def box(self) -> Box:
        return (min(c.box[0] for c in self.chars), min(c.box[1] for c in self.chars),
                max(c.box[2] for c in self.chars), max(c.box[3] for c in self.chars))


def _h(c: PdfChar) -> float:
    return max(1e-3, c.box[3] - c.box[1])


def _rows(chars: list[PdfChar]) -> list[list[PdfChar]]:
    """Characters grouped into rows by vertical overlap, top to bottom; each row left to right.
    Runs of characters that follow each other on one baseline in the content stream are kept
    together, so a row is decided per run rather than per character."""
    runs: list[list[PdfChar]] = []
    for c in chars:
        if runs:
            p = runs[-1][-1]
            h = min(_h(p), _h(c))
            same_line = abs((c.box[1] + c.box[3]) - (p.box[1] + p.box[3])) / 2 <= 0.5 * h
            if same_line and -0.3 * h <= c.box[0] - p.box[2] <= SPLIT_GAP * max(_h(p), _h(c)):
                runs[-1].append(c)
                continue
        runs.append([c])
    runs.sort(key=lambda r: (min(c.box[1] for c in r) + max(c.box[3] for c in r)) / 2)
    rows: list[list[PdfChar]] = []
    bands: list[tuple[float, float]] = []
    counts: list[int] = []
    for run in runs:
        t, b = min(c.box[1] for c in run), max(c.box[3] for c in run)
        best, best_ov = -1, 0.0
        for k, (rt, rb) in enumerate(bands):
            ov = (min(b, rb) - max(t, rt)) / max(1e-3, min(b - t, rb - rt))
            if ov >= 0.5 and ov > best_ov:
                best, best_ov = k, ov
        if best < 0:
            rows.append(list(run))
            bands.append((t, b))
            counts.append(1)
        else:
            rows[best].extend(run)
            rt, rb = bands[best]
            n = counts[best]
            bands[best] = ((rt * n + t) / (n + 1), (rb * n + b) / (n + 1))  # the runs' mean band
            counts[best] = n + 1
    order = sorted(range(len(rows)), key=lambda k: bands[k][0])
    return [sorted(rows[k], key=lambda c: c.box[0]) for k in order]


def _segments(row: list[PdfChar]) -> list[_Seg]:
    """A row split at wide gaps; characters drawn twice on top of each other (fake bold) once."""
    segs: list[_Seg] = []
    for c in row:
        if segs:
            p = segs[-1].chars[-1]
            if c.ch == p.ch and c.box[0] < p.box[2]:
                inter = min(c.box[2], p.box[2]) - max(c.box[0], p.box[0])
                if inter >= 0.7 * min(c.box[2] - c.box[0], p.box[2] - p.box[0]):
                    continue
            if c.box[0] - p.box[2] <= SPLIT_GAP * max(_h(p), _h(c)):
                segs[-1].chars.append(c)
                continue
        segs.append(_Seg([c]))
    return segs


def _text(seg: _Seg) -> str:
    out: list[str] = []
    prev: Optional[PdfChar] = None
    for c in seg.chars:
        if prev is not None:
            gap = c.box[0] - prev.box[2]
            if prev.space or gap > SPACE_GAP * (_h(prev) + _h(c)) / 2:
                out.append(" ")
        out.append(_clean(c.ch))
        prev = c
    return " ".join("".join(out).split())


def _ink(gray: np.ndarray, box: Box) -> float:
    h, w = gray.shape
    l, t, r, b = (int(max(0, box[0])), int(max(0, box[1])), int(min(w, box[2] + 1)), int(min(h, box[3] + 1)))
    if r - l < 1 or b - t < 1:
        return 0.0
    return float(gray[t:b, l:r].std())


def _inside(box: Box, outer: Box) -> bool:
    cx, cy = (box[0] + box[2]) / 2, (box[1] + box[3]) / 2
    return outer[0] <= cx <= outer[2] and outer[1] <= cy <= outer[3]


def _area(box: Box) -> float:
    return max(0.0, box[2] - box[0]) * max(0.0, box[3] - box[1])


def _merge_regions(boxes: list[Box], size: tuple[int, int]) -> list[Box]:
    pad = 0.01 * max(size)
    out = [(max(0.0, b[0] - pad), max(0.0, b[1] - pad), min(size[0], b[2] + pad), min(size[1], b[3] + pad))
           for b in boxes if _area(b) > 0]
    changed = True
    while changed:
        changed = False
        for i in range(len(out)):
            for j in range(i + 1, len(out)):
                a, b = out[i], out[j]
                if a[0] <= b[2] and b[0] <= a[2] and a[1] <= b[3] and b[1] <= a[3]:
                    out[i] = (min(a[0], b[0]), min(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3]))
                    del out[j]
                    changed = True
                    break
            if changed:
                break
    return out


def _clip(box: Box, size: tuple[int, int]) -> Box:
    return (max(0.0, box[0]), max(0.0, box[1]), min(float(size[0]), box[2]), min(float(size[1]), box[3]))


def build_layer(text: PdfText, image: Optional[Image.Image], mode: str = "auto") -> TextLayer:
    """Decide how one page is read, and build its text lines. See the module docstring.

    `image` is the rendered page (used to check that each line shows); `mode` is the `pdf_text`
    setting: "auto" judges the evidence, "always" uses any text layer as it is, "never" is not
    passed here (the page is OCR'd without reading its text layer)."""
    if mode not in PDF_TEXT_MODES:
        raise ValueError(f"pdf_text must be one of {', '.join(PDF_TEXT_MODES)}, not {mode!r}")
    size = text.size
    page_area = float(size[0] * size[1]) or 1.0
    chars = [c for c in text.chars if not c.space or c.ch.strip()]
    chars = [c for c in chars if _inside(c.box, (0, 0, size[0], size[1]))]
    if not chars or mode == "never":
        return TextLayer("ocr")
    if len(chars) > MAX_PAGE_CHARS:
        # Thousands of lines of microscopic text cost minutes in the matcher. A real page has a
        # few thousand characters; read this one as an image, which is bounded.
        return TextLayer("ocr", notes=[f"text layer has {len(chars)} characters, over the limit "
                                       f"of {MAX_PAGE_CHARS}; not used"])

    if mode == "always":
        usable = [c for c in chars if c.upright and not _bad(c.ch)]
        turned = sum(not c.upright for c in chars)
        lines = _lines(usable, size, conf_of=lambda seg: (
            CONF_UNMAPPED if any(not c.mapped for c in seg.chars)
            else CONF_INVISIBLE if any(not c.visible for c in seg.chars) else CONF_TEXT))
        if not lines:
            return TextLayer("ocr", notes=["PDF text layer has no usable characters; page read by OCR"])
        notes = ["read from the PDF's text layer (pdf_text=always)"]
        if any(not c.visible for c in usable):
            notes.append("the text layer is partly or wholly invisible (a scanner's OCR, probably); "
                         "used as it is because pdf_text=always")
        if any(not c.mapped for c in usable) or len(usable) + turned < len(chars):
            notes.append("some characters of the text layer have no usable Unicode mapping")
        if turned:
            notes.append(f"ignored {turned} characters of the text layer drawn at an angle")
        return TextLayer("text", lines=lines, notes=notes)

    # ---- auto
    visible = [c for c in chars if c.visible]
    invisible = len(chars) - len(visible)
    if not visible:
        return TextLayer("ocr", notes=[(
            "the PDF's text layer is invisible (a scanner's OCR layer, probably) and was not used; "
            "page read by OCR (pdf_text=always would use it)")])

    gray = np.asarray(image.convert("L"), dtype=np.float32) if image is not None else None
    notes: list[str] = []
    regions: list[Box] = []
    big_pictures = [(b, z) for b, z in text.pictures if _area(_clip(b, size)) >= MIN_PICTURE_AREA * page_area]
    # Text drawn before a picture that covers it is hidden by the picture.
    under = [c for c in visible if any(z > c.z and _inside(c.box, b) for b, z in text.pictures)]
    if under:
        ids = {id(c) for c in under}
        visible = [c for c in visible if id(c) not in ids]
    turned = [c for c in visible if not c.upright]
    upright = [c for c in visible if c.upright]

    # Lines, and which of them can be trusted.
    trusted: list[OcrLine] = []
    doubtful_chars = len(turned)
    hidden_lines = 0
    for row in _rows(upright):
        for seg in _segments(row):
            seg.chars = [c for c in seg.chars if _clean(c.ch)]
            if not seg.chars:
                continue
            words = _words(seg)
            # a private-use or unmapped character alone (a bullet from a symbol font) is dropped
            keep = [w for w in words if not (len(w) == 1 and _bad(w[0].ch))]
            if not keep:
                continue
            seg.chars = [c for w in keep for c in w]
            box = seg.box
            if gray is not None and _ink(gray, box) < MIN_INK_STD:
                hidden_lines += 1
                continue
            if any(_bad(c.ch) or not c.mapped for c in seg.chars):
                doubtful_chars += len(seg.chars)
                regions.append(box)
                continue
            line = _line(seg, CONF_TEXT)
            if line is not None:
                trusted.append(line)
    regions += [b for b, _ in big_pictures] + list(text.annotations)
    for row in _rows(turned):
        regions.append(_Seg(row).box)

    if doubtful_chars > MAX_DOUBTFUL * len(visible):
        reason = ("most of its text is drawn at an angle" if len(turned) > doubtful_chars / 2
                  else "many of its characters have no usable Unicode mapping")
        return TextLayer("ocr", notes=[f"the PDF's text layer was not used: {reason}; page read by OCR"])
    if not trusted:
        return TextLayer("ocr", notes=[("the PDF's text layer has no text that shows on the page; "
                                        "page read by OCR")])
    if invisible:
        notes.append(f"ignored {invisible} invisible characters of the PDF's text layer")
    hidden_lines += len(_rows(under)) if under else 0
    if hidden_lines:
        notes.append(f"ignored {hidden_lines} line(s) of the PDF's text layer that do not show on the page")
    regions = _merge_regions([_clip(b, size) for b in regions], size)
    if sum(_area(b) for b in regions) > WHOLE_PAGE_AREA * page_area:
        regions = [(0.0, 0.0, float(size[0]), float(size[1]))]
    if regions:
        notes.insert(0, f"read from the PDF's text layer, and by OCR in {len(regions)} area(s) the text "
                        "layer does not cover (pictures, annotations, or text that cannot be trusted)")
        return TextLayer("hybrid", lines=trusted, regions=regions, notes=notes)
    notes.insert(0, "read from the PDF's text layer")
    return TextLayer("text", lines=trusted, notes=notes)


def _words(seg: _Seg) -> list[list[PdfChar]]:
    words: list[list[PdfChar]] = []
    prev: Optional[PdfChar] = None
    for c in seg.chars:
        if prev is None or prev.space or c.box[0] - prev.box[2] > SPACE_GAP * (_h(prev) + _h(c)) / 2:
            words.append([c])
        else:
            words[-1].append(c)
        prev = c
    return words


def _line(seg: _Seg, conf: float) -> Optional[OcrLine]:
    text = _text(seg)
    if not text:
        return None
    return OcrLine(text, conf, seg.box)


def _lines(chars: list[PdfChar], size: tuple[int, int], conf_of) -> list[OcrLine]:
    out: list[OcrLine] = []
    for row in _rows(chars):
        for seg in _segments(row):
            seg.chars = [c for c in seg.chars if _clean(c.ch)]
            if seg.chars:
                line = _line(seg, conf_of(seg))
                if line is not None:
                    out.append(line)
    return out


def text_lines(text: PdfText) -> list[OcrLine]:
    """Every upright character of a text layer as lines, without judging it (for tests)."""
    return _lines([c for c in text.chars if c.upright], text.size, lambda seg: CONF_TEXT)


def median_height(lines: list[OcrLine]) -> float:
    return statistics.median([ln.height for ln in lines]) if lines else 0.0
