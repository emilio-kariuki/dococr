"""Multi-page PDF fixtures: the made-up land sale agreement of `synth.contract_for` (YAML
tests/doctypes/synth_contract.yaml) set as a PDF of two or three pages, with values that run
across the page breaks, running headers, footers, page numbers and initials lines.

The same document in several forms (`build`):

- "digital": a text PDF as a word processor exports it (embedded TrueType font, WinAnsi);
- "image": the digital pages rendered and saved as an image-only PDF, like a clean scan;
- "ocr_layer": the image PDF with an invisible text layer (render mode 3) of the same lines,
  with deliberate errors (lost spaces, O read as 0, l as 1), like a scanner's "searchable PDF";
- "pua": the digital PDF whose font maps every character to the private-use area;
- "nomap": the digital PDF whose font has no Unicode mapping (made-up glyph names);
- "rotated": page 2 has /Rotate 90 and its text drawn turned so that it shows upright;
- "sideways": page 2 has /Rotate 90 and plain text, so it shows turned (text at an angle);
- "cropped": every page has a MediaBox larger than its CropBox, content offset to match;
- "mixed": the last paragraph of the last page is a pasted picture of text, not text.

    doc = make_pdf_document(seed=1)
    data = build(doc, "digital"); doc.truth; doc.crossing

Nothing is written to disk.
"""

from __future__ import annotations

import io
import random
import re
from dataclasses import dataclass, field
from typing import Optional

import pdfs
import synth
from pdfs import PdfPage, Picture, Text
from PIL import Image, ImageDraw

PAGE_W, PAGE_H = 595.0, 842.0
LEFT, RIGHT = 72.0, 72.0
BODY_TOP, BODY_BOTTOM = 735.0, 105.0  # baselines of the body text stay between these
FORMS = ["digital", "image", "ocr_layer", "pua", "nomap", "rotated", "sideways", "cropped", "mixed"]


@dataclass
class Line:
    text: str
    x: float
    y: float  # baseline, points from the bottom of the page as it shows
    size: float
    font: str
    para: int = -1  # paragraph of the body, -1 for furniture and the title
    start: int = 0  # character range of the paragraph printed on this line
    end: int = 0
    furniture: bool = False


@dataclass
class PdfDocument:
    seed: int
    truth: dict[str, object]
    pages: list[list[Line]]
    crossing: set[str] = field(default_factory=set)  # fields whose label and end are on two pages
    font: str = "sans"
    size: float = 11.0
    paragraphs: list[str] = field(default_factory=list)


def wrap_offsets(text: str, width: float, size: float, font: str, rng: random.Random,
                 caps_rate: float = 0.5) -> list[tuple[str, int, int]]:
    """Greedy wrap to `width` points that hyphenates the way `synth.hyphen_wrap` does, and
    records which characters of `text` each line prints: (line, start, end)."""
    fits = lambda line: pdfs.text_width(line, size, font) <= width  # noqa: E731
    lines: list[list] = []  # [text, start, end]
    for m in re.finditer(r"\S+", text):
        word, s, e = m.group(0), m.start(), m.end()
        if lines and fits(f"{lines[-1][0]} {word}"):
            lines[-1][0] += f" {word}"
            lines[-1][2] = e
            continue
        if not lines:
            lines.append([word, s, e])
            continue
        cuts = [i + 1 for i, ch in enumerate(word[:-1]) if ch == "-" and 0 < i]
        core = re.match(r"[A-Za-z]+", word)
        if core and len(core.group(0)) >= 6 and not cuts and (
                not core.group(0).isupper() or rng.random() < caps_rate):
            cuts = synth._split_points(core.group(0))
        best = None
        for i in cuts:
            head = word[:i] if word[i - 1] == "-" else word[:i] + "-"
            if fits(f"{lines[-1][0]} {head}"):
                best = (i, head)
        if best:
            lines[-1][0] += f" {best[1]}"
            lines[-1][2] = s + best[0]
            lines.append([word[best[0]:], s + best[0], e])
        else:
            lines.append([word, s, e])
    return [(t, s, e) for t, s, e in lines]


def _labels(doc_type) -> dict[str, list[str]]:
    return {f.name: f.labels() for f in doc_type.fields}


def _locate(paragraphs: list[str], printed: dict[str, str], labels: dict[str, list[str]]
            ) -> dict[str, tuple[int, int, int]]:
    """For each field: (paragraph, offset of its label, offset of the text after its value)."""
    out = {}
    for name, value in printed.items():
        if not value:
            continue
        for p, text in enumerate(paragraphs):
            low = text.lower()
            for label in labels.get(name, []):
                for m in re.finditer(re.escape(label.lower()), low):
                    rest = text[m.end():]
                    lead = len(rest) - len(rest.lstrip())
                    if rest.lstrip().startswith(value):
                        after = m.end() + lead + len(value)
                        while after < len(text) and text[after] == " ":
                            after += 1
                        out.setdefault(name, (p, m.start(), after))
    return out


def make_pdf_document(seed: int = 0, breaks: int = 2) -> PdfDocument:
    """Lay out the agreement of `synth.contract_for(seed)` on pages, breaking pages inside the
    values of `breaks` fields chosen at random (more pages where the text does not fit)."""
    from dococr import load_doc_type

    rng = random.Random(f"synth_pdf/{seed}")
    printed: dict[str, str] = {}
    title, paragraphs, truth = synth.contract_for(random.Random(f"synth_contract/{seed}"), printed)
    font = rng.choice(["sans", "serif"])
    size = rng.choice([10.0, 11.0, 12.0])
    pitch = size * 1.45
    width = PAGE_W - LEFT - RIGHT
    wrap_rng = random.Random(f"synth_pdf/{seed}/wrap")
    body: list[Line] = []
    for p, text in enumerate(paragraphs):
        for t, s, e in wrap_offsets(text, width, size, font, wrap_rng):
            body.append(Line(t, LEFT, 0.0, size, font, p, s, e))

    def line_of(p: int, offset: int) -> int:
        idx = [k for k, ln in enumerate(body) if ln.para == p]
        for k in idx:
            if body[k].start <= offset < body[k].end:
                return k
        return next((k for k in idx if body[k].start >= offset), idx[-1])

    doc_type = load_doc_type(synth.FIXTURE_DIR / "synth_contract.yaml")
    spans = _locate(paragraphs, printed, _labels(doc_type))
    # page breaks inside values: after a line between a field's label and the text after it
    candidates = []
    for name, (p, ls, after) in spans.items():
        a, b = line_of(p, ls), line_of(p, after)
        if b > a:
            candidates.append((name, a, b))
    rng.shuffle(candidates)
    cut_after: list[int] = []
    for _name, a, b in candidates:
        if len(cut_after) == breaks:
            break
        k = rng.randint(a, b - 1)
        if all(abs(k - c) >= 4 for c in cut_after) and k >= 3:
            cut_after.append(k)
    cut_after.sort()

    # place the lines, starting a new page after each chosen line or when a page is full
    pages: list[list[Line]] = [[]]
    y = BODY_TOP - 40  # room for the title on page 1
    pages[0].append(Line(title, (PAGE_W - pdfs.text_width(title, size + 3, "sans-bold")) / 2,
                         BODY_TOP, size + 3, "sans-bold"))
    prev_para = 0
    for k, ln in enumerate(body):
        if ln.para != prev_para:
            y -= size * 0.7
            prev_para = ln.para
        if y < BODY_BOTTOM:
            pages.append([])
            y = BODY_TOP
        ln.y = y
        pages[-1].append(ln)
        y -= pitch
        if k in cut_after and k + 1 < len(body):
            pages.append([])
            y = BODY_TOP

    # page furniture
    fstyle = rng.randrange(3)
    initials = rng.random() < 0.5
    header = rng.random() < 0.7
    n = len(pages)
    for i, page in enumerate(pages, start=1):
        fsize = 8.5
        if header:
            page.append(Line("Agreement for Sale of Land", LEFT, 800, fsize, "sans", furniture=True))
            ref = f"Ref. AFS/{seed:03d}/2024"
            page.append(Line(ref, PAGE_W - RIGHT - pdfs.text_width(ref, fsize, "sans"), 800, fsize, "sans",
                             furniture=True))
        number = [f"Page {i} of {n}", f"- {i} -", str(i)][fstyle]
        nx = (PAGE_W - pdfs.text_width(number, fsize, "sans")) / 2 if fstyle < 2 else \
            PAGE_W - RIGHT - pdfs.text_width(number, fsize, "sans")
        page.append(Line(number, nx, 45, fsize, "sans", furniture=True))
        if initials:
            page.append(Line("Vendor's initials ________", LEFT, 68, fsize, "sans", furniture=True))
            t = "Purchaser's initials ________"
            page.append(Line(t, PAGE_W - RIGHT - pdfs.text_width(t, fsize, "sans"), 68, fsize, "sans",
                             furniture=True))

    doc = PdfDocument(seed, truth, pages, font=font, size=size, paragraphs=paragraphs)
    page_of = {id(ln): i for i, page in enumerate(pages) for ln in page}
    for name, (p, ls, after) in spans.items():
        if page_of[id(body[line_of(p, ls)])] != page_of[id(body[line_of(p, max(ls, after - 1))])] or \
                page_of[id(body[line_of(p, ls)])] != page_of[id(body[line_of(p, after)])]:
            doc.crossing.add(name)
    return doc


# ---------------------------------------------------------------------------- the forms

_LOOK_ALIKE = {"O": "0", "l": "1", "S": "5", "I": "1", "B": "8"}


def _corrupt(text: str, rng: random.Random) -> str:
    """What a weak OCR layer makes of a line: a space lost now and then, look-alike characters."""
    words = text.split(" ")
    if len(words) > 2 and rng.random() < 0.35:
        k = rng.randrange(len(words) - 1)
        words[k:k + 2] = [words[k] + words[k + 1]]
    out = []
    for w in words:
        if rng.random() < 0.2:
            w = "".join(_LOOK_ALIKE.get(ch, ch) if rng.random() < 0.6 else ch for ch in w)
        out.append(w)
    return " ".join(out)


def render_pages(data: bytes, dpi: float = 150.0) -> list[Image.Image]:
    import pypdfium2 as pdfium

    pdf = pdfium.PdfDocument(data)
    try:
        return [pdf[i].render(scale=dpi / 72).to_pil().convert("RGB") for i in range(len(pdf))]
    finally:
        pdf.close()


def _picture_of(lines: list[Line], scale: float = 3.0) -> tuple[Picture, list[Line]]:
    """A picture of some lines of text (a pasted scan), to replace them on the page."""
    top = max(ln.y + ln.size for ln in lines) + 4
    bottom = min(ln.y for ln in lines) - lines[0].size * 0.5
    x0 = LEFT - 4
    w_pt, h_pt = PAGE_W - LEFT - RIGHT + 8, top - bottom
    img = Image.new("RGB", (int(w_pt * scale), int(h_pt * scale)), (248, 246, 240))
    d = ImageDraw.Draw(img)
    for ln in lines:
        fnt = synth.font("serif" if ln.font == "serif" else "sans", int(ln.size * scale))
        d.text(((ln.x - x0) * scale, (top - ln.y - ln.size * 0.95) * scale), ln.text, font=fnt, fill=(20, 20, 30))
    return Picture(x0, bottom, w_pt, h_pt, img), lines


def build(doc: PdfDocument, form: str = "digital") -> bytes:
    """The document as PDF bytes in one of `FORMS`."""
    if form == "image":
        return pdfs.images_pdf(render_pages(build(doc, "digital")), resolution=150)
    if form == "ocr_layer":
        rng = random.Random(f"synth_pdf/{doc.seed}/ocr_layer")
        images = render_pages(build(doc, "digital"))
        pages = []
        for img, lines in zip(images, doc.pages):
            items: list = [Picture(0, 0, PAGE_W, PAGE_H, img)]
            items += [Text(ln.x, ln.y, _corrupt(ln.text, rng), ln.size, ln.font, mode=3) for ln in lines]
            pages.append(PdfPage(PAGE_W, PAGE_H, items))
        return pdfs.text_pdf(pages)
    pages = []
    for i, lines in enumerate(doc.pages):
        def font_of(ln: Line) -> str:
            return form if form in ("pua", "nomap") and not ln.furniture else ln.font

        items: list = []
        body = lines
        if form == "mixed" and i == len(doc.pages) - 1:
            last = max(ln.para for ln in lines if not ln.furniture)
            pasted = [ln for ln in lines if ln.para == last]
            pic, _ = _picture_of(pasted)
            items.append(pic)
            body = [ln for ln in lines if ln.para != last]
        if form == "rotated" and i == 1:
            # /Rotate 90 shows the MediaBox turned clockwise: draw the text turned the other way
            items += [Text(PAGE_H - ln.y, ln.x, ln.text, ln.size, ln.font, angle=90) for ln in body]
            pages.append(PdfPage(PAGE_H, PAGE_W, items, rotate=90))
            continue
        if form == "sideways" and i == 1:
            items += [Text(ln.x, ln.y, ln.text, ln.size, ln.font) for ln in body]
            pages.append(PdfPage(PAGE_W, PAGE_H, items, rotate=90))
            continue
        if form == "cropped":
            ox, oy = 40.0, 55.0
            items += [Text(ln.x + ox, ln.y + oy, ln.text, ln.size, ln.font) for ln in body]
            pages.append(PdfPage(PAGE_W + 100, PAGE_H + 120, items, crop=(ox, oy, ox + PAGE_W, oy + PAGE_H)))
            continue
        items += [Text(ln.x, ln.y, ln.text, ln.size, font_of(ln)) for ln in body]
        pages.append(PdfPage(PAGE_W, PAGE_H, items))
    return pdfs.text_pdf(pages)


def doc_type_ref():
    return synth.FIXTURE_DIR / "synth_contract.yaml"


def page_image(data: bytes, page: int = 0) -> Optional[Image.Image]:
    imgs = render_pages(data)
    return imgs[page] if page < len(imgs) else None


def save_pngs(data: bytes) -> list[bytes]:  # pragma: no cover - for looking at a fixture
    out = []
    for img in render_pages(data, 100):
        buf = io.BytesIO()
        img.save(buf, "PNG")
        out.append(buf.getvalue())
    return out
