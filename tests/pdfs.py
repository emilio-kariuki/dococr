"""Helpers that build test PDFs and page images in memory. No binary files are committed.

Besides image-only PDFs (Pillow) and blank pages (`raw_pdf`), `text_pdf` writes PDFs with a text
layer, as a word processor or a scanner's "searchable PDF" would: text placed in points with an
embedded TrueType font (DejaVu, the fonts `synth.py` draws with), invisible text (render mode 3),
a font whose characters map to private-use code points, text drawn at an angle, pictures, and
pages with `/Rotate` and a CropBox.
"""

from __future__ import annotations

import functools
import hashlib
import io
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, Union

from PIL import Image, ImageFont

RED, GREEN, BLUE = (255, 0, 0), (0, 255, 0), (0, 0, 255)
COLOURS = [RED, GREEN, BLUE]


def page_index(image: Image.Image) -> int:
    """Which solid-colour page this is (0 red, 1 green, 2 blue), read from the centre pixel."""
    r, g, b = image.convert("RGB").getpixel((image.width // 2, image.height // 2))
    return max(range(3), key=lambda i: (r, g, b)[i])


def pages(n: int, size=(300, 200)) -> list[Image.Image]:
    return [Image.new("RGB", size, COLOURS[i % 3]) for i in range(n)]


def image_pdf(n: int = 2, size=(300, 200)) -> bytes:
    """An image-only PDF, like a scan: one solid-colour page per image."""
    return images_pdf(pages(n, size))


def images_pdf(images: list[Image.Image], resolution: float = 72.0) -> bytes:
    """An image-only PDF of the given page images (Pillow's writer), like a scan."""
    buf = io.BytesIO()
    images[0].save(buf, "PDF", save_all=True, append_images=images[1:], resolution=resolution)
    return buf.getvalue()


def tiff(n: int = 2, size=(300, 200)) -> bytes:
    imgs = pages(n, size)
    buf = io.BytesIO()
    imgs[0].save(buf, "TIFF", save_all=True, append_images=imgs[1:])
    return buf.getvalue()


def png(colour=RED, size=(40, 40)) -> bytes:
    buf = io.BytesIO()
    Image.new("RGB", size, colour).save(buf, "PNG")
    return buf.getvalue()


def _serialise(objs: list[bytes], trailer_extra: str = "") -> bytes:
    """Objects 1..n (object 1 the catalog) as a PDF file with a cross-reference table."""
    out = bytearray(b"%PDF-1.4\n%\xe2\xe3\xcf\xd3\n")
    offsets = []
    for i, body in enumerate(objs, start=1):
        offsets.append(len(out))
        out += f"{i} 0 obj\n".encode() + body + b"\nendobj\n"
    xref = len(out)
    out += f"xref\n0 {len(objs) + 1}\n0000000000 65535 f \n".encode()
    for off in offsets:
        out += f"{off:010d} 00000 n \n".encode()
    out += f"trailer\n<< /Size {len(objs) + 1} /Root 1 0 R{trailer_extra} >>\nstartxref\n{xref}\n%%EOF\n".encode()
    return bytes(out)


def raw_pdf(boxes, encrypt: bool = False) -> bytes:
    """A hand-written PDF with one blank page per (width, height) in points.

    `boxes=[]` gives a PDF with no pages. `encrypt=True` adds the standard security handler
    (RC4 40 bit) with a user password that is not empty, so it cannot be opened without it.
    """
    objs: list[bytes] = []
    kids = " ".join(f"{3 + i} 0 R" for i in range(len(boxes)))
    objs.append(b"<< /Type /Catalog /Pages 2 0 R >>")
    objs.append(f"<< /Type /Pages /Kids [{kids}] /Count {len(boxes)} >>".encode())
    for w, h in boxes:
        objs.append(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {w} {h}] >>".encode())
    trailer_extra = ""
    if encrypt:
        o, u = _encryption_entries(b"secret")
        objs.append(
            b"<< /Filter /Standard /V 1 /R 2 /P -4 /O <" + o.hex().encode() + b"> /U <"
            + u.hex().encode() + b"> >>"
        )
        trailer_extra = f" /Encrypt {len(objs)} 0 R /ID [<{_ID.hex()}> <{_ID.hex()}>]"
    return _serialise(objs, trailer_extra)


_ID = bytes(range(16))
_PAD = bytes.fromhex("28BF4E5E4E758A4164004E56FFFA01082E2E00B6D0683E802F0CA9FE6453697A")


def _rc4(key: bytes, data: bytes) -> bytes:
    s = list(range(256))
    j = 0
    for i in range(256):
        j = (j + s[i] + key[i % len(key)]) % 256
        s[i], s[j] = s[j], s[i]
    out = bytearray()
    i = j = 0
    for byte in data:
        i = (i + 1) % 256
        j = (j + s[i]) % 256
        s[i], s[j] = s[j], s[i]
        out.append(byte ^ s[(s[i] + s[j]) % 256])
    return bytes(out)


def _encryption_entries(user_password: bytes) -> tuple[bytes, bytes]:
    """The /O and /U strings for revision 2 (PDF 1.7 reference, algorithms 3.3 and 3.4)."""
    padded = (user_password + _PAD)[:32]
    owner_key = hashlib.md5(_PAD).digest()[:5]  # owner password = empty
    o = _rc4(owner_key, padded)
    key = hashlib.md5(padded + o + (-4 & 0xFFFFFFFF).to_bytes(4, "little") + _ID).digest()[:5]
    return o, _rc4(key, _PAD)


# ---------------------------------------------------------------------------- PDFs with text

FONT_DIR = Path("/usr/share/fonts/truetype/dejavu")
# Font keys for `Text.font`. Each is the TrueType file embedded, and how its codes map to Unicode:
#   "sans", "serif", "sans-bold": WinAnsi encoding, the way word processors write simple fonts;
#   "pua": the sans font with a ToUnicode map that sends every code to the private-use area
#          (a font without a usable Unicode map: the page looks right, its text is garbage);
#   "nomap": the sans font as a CID font addressed by glyph id (Identity-H) with no ToUnicode
#            map, as PDF writers produce for subset fonts: the page shows the text, but which
#            characters it holds is unknown.
FONT_FILES = {"sans": "DejaVuSans.ttf", "serif": "DejaVuSerif.ttf", "sans-bold": "DejaVuSans-Bold.ttf",
              "pua": "DejaVuSans.ttf", "nomap": "DejaVuSans.ttf"}
_FIRST, _LAST = 32, 255


def _char(code: int) -> str:
    try:
        return bytes([code]).decode("cp1252")
    except UnicodeDecodeError:
        return ""


@functools.lru_cache(maxsize=None)
def _widths(font: str) -> tuple[int, ...]:
    """Advance widths in thousandths of an em of codes _FIRST.._LAST in the WinAnsi encoding."""
    face = ImageFont.truetype(str(FONT_DIR / FONT_FILES[font]), 1000)
    out = []
    for code in range(_FIRST, _LAST + 1):
        ch = _char(code)
        out.append(round(face.getlength(ch)) if ch and ch.isprintable() or ch == "­" else 0)
    return tuple(out)


@functools.lru_cache(maxsize=None)
def _glyph_ids(font: str) -> dict[int, int]:
    """Unicode code point -> glyph id, from the font's (3, 1) format 4 cmap subtable."""
    import struct

    data = (FONT_DIR / FONT_FILES[font]).read_bytes()
    tables = {data[12 + 16 * i:16 + 16 * i]: struct.unpack(">II", data[20 + 16 * i:28 + 16 * i])
              for i in range(struct.unpack(">H", data[4:6])[0])}
    base = tables[b"cmap"][0]
    out: dict[int, int] = {}
    for k in range(struct.unpack(">H", data[base + 2:base + 4])[0]):
        pid, eid, off = struct.unpack(">HHI", data[base + 4 + 8 * k:base + 12 + 8 * k])
        sub = base + off
        if (pid, eid) != (3, 1) or struct.unpack(">H", data[sub:sub + 2])[0] != 4:
            continue
        n = struct.unpack(">H", data[sub + 6:sub + 8])[0] // 2
        def arr(at: int, n: int = n) -> tuple:
            return struct.unpack(f">{n}H", data[at:at + 2 * n])

        ends, starts = arr(sub + 14), arr(sub + 16 + 2 * n)
        deltas, ro_at = arr(sub + 16 + 4 * n), sub + 16 + 6 * n
        offsets = arr(ro_at)
        for i in range(n):
            for cp in range(starts[i], min(ends[i], 0xFFFE) + 1):
                if offsets[i] == 0:
                    gid = (cp + deltas[i]) & 0xFFFF
                else:
                    at = ro_at + 2 * i + offsets[i] + 2 * (cp - starts[i])
                    gid = struct.unpack(">H", data[at:at + 2])[0]
                    gid = (gid + deltas[i]) & 0xFFFF if gid else 0
                if gid:
                    out[cp] = gid
        break
    return out


def text_width(text: str, size: float, font: str = "sans") -> float:
    """Width in points of `text` set in `font` at `size` points (as the PDF will draw it)."""
    widths = _widths(font)
    total = 0
    for byte in _encode(text):
        if _FIRST <= byte <= _LAST:
            total += widths[byte - _FIRST]
    return total * size / 1000


def _encode(text: str) -> bytes:
    return text.encode("cp1252", errors="replace")


def _pdf_string(text: str, font: str = "sans") -> bytes:
    if font == "nomap":  # two-byte glyph ids (Identity-H)
        ids = _glyph_ids("nomap")
        return b"<" + "".join(f"{ids.get(ord(ch), 0):04X}" for ch in text).encode() + b">"
    data = _encode(text)
    return b"(" + data.replace(b"\\", b"\\\\").replace(b"(", b"\\(").replace(b")", b"\\)") + b")"


@dataclass
class Text:
    """A run of text with its baseline starting at (x, y), in points from the MediaBox's
    bottom-left corner."""
    x: float
    y: float
    text: str
    size: float = 11.0
    font: str = "sans"
    mode: int = 0  # text render mode: 0 fill, 3 invisible (a scanner's OCR layer)
    angle: float = 0.0  # degrees counter-clockwise
    gray: float = 0.0  # fill colour, 0 black .. 1 white


@dataclass
class Picture:
    """An image drawn into the rectangle (x, y, w, h) in points."""
    x: float
    y: float
    w: float
    h: float
    image: Image.Image


@dataclass
class Rule:
    """A filled rectangle (a line under a heading, a box)."""
    x: float
    y: float
    w: float
    h: float
    gray: float = 0.0


@dataclass
class Annotation:
    """An annotation over the rectangle (x, y, w, h): a stamp, a filled form field, ..."""
    x: float
    y: float
    w: float
    h: float
    subtype: str = "Stamp"


@dataclass
class PdfPage:
    width: float = 595.0
    height: float = 842.0
    items: list = field(default_factory=list)
    rotate: int = 0  # /Rotate
    crop: Optional[tuple[float, float, float, float]] = None  # /CropBox (x0, y0, x1, y1)


def _stream(data: bytes, extra: str = "", compress: bool = True) -> bytes:
    if compress:
        data = zlib.compress(data)
        extra += " /Filter /FlateDecode"
    return f"<< /Length {len(data)}{extra} >>\nstream\n".encode() + data + b"\nendstream"


def _pua_cmap() -> bytes:
    return (b"/CIDInit /ProcSet findresource begin 12 dict begin begincmap\n"
            b"/CMapName /Garbage def /CMapType 2 def\n"
            b"1 begincodespacerange <00> <FF> endcodespacerange\n"
            b"1 beginbfrange <20> <FF> <E020> endbfrange\n"
            b"endcmap CMapName currentdict /CMap defineresource pop end end\n")


def text_pdf(page_list: list[PdfPage]) -> bytes:
    """A PDF with the given pages. Fonts are embedded once and shared by the pages."""
    objs: list[bytes] = [b"", b""]  # catalog and page tree, filled in at the end

    def add(body: bytes) -> int:
        objs.append(body)
        return len(objs)

    used = sorted({it.font for p in page_list for it in p.items if isinstance(it, Text)})
    font_refs: dict[str, int] = {}
    files: dict[str, int] = {}
    for key in used:
        name = FONT_FILES[key]
        if name not in files:
            data = (FONT_DIR / name).read_bytes()
            files[name] = add(_stream(data, f" /Length1 {len(data)}"))
        base = name.rsplit(".", 1)[0].replace("-", "")
        flags = 4 if key == "nomap" else 32
        desc = add(f"<< /Type /FontDescriptor /FontName /{base} /Flags {flags} /FontBBox [-1021 -463 1793 1232] "
                   f"/ItalicAngle 0 /Ascent 928 /Descent -236 /CapHeight 729 /StemV 80 "
                   f"/FontFile2 {files[name]} 0 R >>".encode())
        widths = " ".join(str(w) for w in _widths(key))
        extra = ""
        if key == "pua":
            extra = f" /ToUnicode {add(_stream(_pua_cmap()))} 0 R"
            encoding = "/WinAnsiEncoding"
        else:
            encoding = "/WinAnsiEncoding"
        if key == "nomap":
            # A CID font addressed by glyph id (Identity-H) with no ToUnicode map, as PDF
            # writers produce for subset fonts: the page shows the text, its Unicode is unknown.
            ids = _glyph_ids("nomap")
            face = ImageFont.truetype(str(FONT_DIR / FONT_FILES[key]), 1000)
            w = " ".join(f"{gid} [{round(face.getlength(chr(cp)))}]" for cp, gid in sorted(
                ((cp, g) for cp, g in ids.items() if 32 <= cp < 0x250), key=lambda t: t[1]))
            cid = add(f"<< /Type /Font /Subtype /CIDFontType2 /BaseFont /{base} /CIDSystemInfo "
                      f"<< /Registry (Adobe) /Ordering (Identity) /Supplement 0 >> /FontDescriptor "
                      f"{desc} 0 R /CIDToGIDMap /Identity /DW 600 /W [{w}] >>".encode())
            font_refs[key] = add(f"<< /Type /Font /Subtype /Type0 /BaseFont /{base} /Encoding /Identity-H "
                                 f"/DescendantFonts [{cid} 0 R] >>".encode())
            continue
        font_refs[key] = add(
            f"<< /Type /Font /Subtype /TrueType /BaseFont /{base} /FirstChar {_FIRST} /LastChar {_LAST} "
            f"/Widths [{widths}] /Encoding {encoding} /FontDescriptor {desc} 0 R{extra} >>".encode())
    fonts = " ".join(f"/F{i} {font_refs[k]} 0 R" for i, k in enumerate(used))
    font_name = {k: f"F{i}" for i, k in enumerate(used)}

    kids = []
    for page in page_list:
        images = {}
        annots: list[int] = []
        ops: list[bytes] = []
        for it in page.items:
            if isinstance(it, Picture):
                buf = io.BytesIO()
                it.image.convert("RGB").save(buf, "JPEG", quality=90)
                ref = add(_stream(buf.getvalue(), f" /Type /XObject /Subtype /Image /Width {it.image.width} "
                                  f"/Height {it.image.height} /ColorSpace /DeviceRGB /BitsPerComponent 8 "
                                  f"/Filter /DCTDecode", compress=False))
                name = f"Im{len(images)}"
                images[name] = ref
                ops.append(f"q {it.w:.3f} 0 0 {it.h:.3f} {it.x:.3f} {it.y:.3f} cm /{name} Do Q".encode())
            elif isinstance(it, Annotation):
                annots.append(add(f"<< /Type /Annot /Subtype /{it.subtype} /Rect [{it.x} {it.y} "
                                  f"{it.x + it.w} {it.y + it.h}] /F 4 >>".encode()))
            elif isinstance(it, Rule):
                ops.append(f"q {it.gray:.3f} g {it.x:.3f} {it.y:.3f} {it.w:.3f} {it.h:.3f} re f Q".encode())
            else:
                import math

                a = math.radians(it.angle)
                c, s = math.cos(a), math.sin(a)
                ops.append(
                    f"BT /{font_name[it.font]} {it.size:.2f} Tf {it.mode} Tr {it.gray:.3f} g "
                    f"{c:.6f} {s:.6f} {-s:.6f} {c:.6f} {it.x:.3f} {it.y:.3f} Tm ".encode()
                    + _pdf_string(it.text, it.font) + b" Tj ET")
        content = add(_stream(b"\n".join(ops)))
        xobj = " ".join(f"/{n} {r} 0 R" for n, r in images.items())
        res = f"<< /Font << {fonts} >> /XObject << {xobj} >> >>"
        extra = f" /Rotate {page.rotate}" if page.rotate else ""
        if annots:
            extra += " /Annots [" + " ".join(f"{a} 0 R" for a in annots) + "]"
        if page.crop:
            extra += " /CropBox [{} {} {} {}]".format(*page.crop)
        kids.append(add(f"<< /Type /Page /Parent 2 0 R /MediaBox [0 0 {page.width} {page.height}]{extra} "
                        f"/Resources {res} /Contents {content} 0 R >>".encode()))
    objs[0] = b"<< /Type /Catalog /Pages 2 0 R >>"
    objs[1] = f"<< /Type /Pages /Kids [{' '.join(f'{k} 0 R' for k in kids)}] /Count {len(kids)} >>".encode()
    return _serialise(objs)


def wrap(text: str, width: float, size: float, font: str = "sans") -> list[str]:
    """Greedy word wrap to a width in points."""
    lines: list[str] = []
    for word in text.split():
        if lines and text_width(f"{lines[-1]} {word}", size, font) <= width:
            lines[-1] += f" {word}"
        else:
            lines.append(word)
    return lines or [""]


Source = Union[bytes, Path]
