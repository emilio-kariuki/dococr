"""Loading the pages of a document from whatever the caller has.

A source is a path, bytes, a PIL image, a PDF (path or bytes, recognised by its content), a
multi-frame image file (TIFF, GIF, WebP), or a list or tuple of sources, which are the pages of
one document in order. Everything is read in memory; nothing is written to disk. PDF pages are
rendered one at a time, only when asked for, with PDFium (`pypdfium2`), which is imported on
first use so that commands that read no document stay light.
"""

from __future__ import annotations

import io
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterator, Optional, Union

from PIL import Image, ImageOps

ImageSource = Union[str, Path, bytes, bytearray, Image.Image, list, tuple]

DEFAULT_MAX_PAGES = 20
# `preprocess.MAX_LONG_SIDE` shrinks its working image to this, so a larger render is wasted.
RENDER_LONG_SIDE = 2000
# Image formats whose frames are pages. JPEG (MPO) is left out on purpose: a phone photo can
# carry a second frame that is not another page.
_PAGED_FORMATS = {"TIFF", "GIF", "WEBP"}
_PDF_MAGIC = b"%PDF-"
_PDF_MAGIC_WINDOW = 1024  # the header may be preceded by a little junk

# PDFium is not thread-safe, and the HTTP service runs requests in a thread pool.
_PDFIUM_LOCK = threading.Lock()


class SourceError(ValueError):
    """A source cannot be read as pages (bad PDF, too many pages, empty list)."""


class PageLimitError(SourceError):
    """The document has more pages than `max_pages`."""


def _finish(image: Image.Image) -> Image.Image:
    return ImageOps.exif_transpose(image).convert("RGB")


def _check_pixels(size: tuple[int, int]) -> None:
    limit = Image.MAX_IMAGE_PIXELS
    if limit is not None and size[0] * size[1] > 2 * limit:
        raise Image.DecompressionBombError(
            f"image of {size[0]}x{size[1]} pixels is over the safety limit"
        )


def load_image(source: ImageSource) -> Image.Image:
    """Return an RGB image with EXIF orientation applied (phone photos are often rotated).

    One image only. For PDFs and several pages use `open_pages`.
    """
    if isinstance(source, Image.Image):
        image = source
    elif isinstance(source, (bytes, bytearray)):
        image = Image.open(io.BytesIO(source))
    else:
        image = Image.open(Path(source))
    return _finish(image)


def _is_pdf_bytes(head: bytes) -> bool:
    return _PDF_MAGIC in head[:_PDF_MAGIC_WINDOW]


@dataclass
class _Leaf:
    count: int
    render: Callable[[int], Image.Image]  # 0-based page of this leaf
    close: Callable[[], None] = lambda: None
    # PDF pages only: render and read the text layer in one go (0-based page, -> image, PdfText)
    render_text: Optional[Callable[[int], tuple[Image.Image, Any]]] = None


@dataclass
class Page:
    """One page: its image and, for a PDF page whose text layer was asked for, what was read
    from that layer (`dococr.pdftext.PdfText`; None for images and when not asked for)."""
    image: Image.Image
    text: Any = None


class Pages:
    """The pages of one document. `len()` is known before any page is rendered."""

    def __init__(self, leaves: list[_Leaf]):
        self._leaves = leaves

    def __len__(self) -> int:
        return sum(leaf.count for leaf in self._leaves)

    def __iter__(self) -> Iterator[Image.Image]:
        """Yield RGB images, EXIF-upright, rendering each page only when reached."""
        for leaf in self._leaves:
            for i in range(leaf.count):
                yield leaf.render(i)

    def with_text(self) -> Iterator[Page]:
        """Like iterating, but each PDF page also carries its text layer (read while the page is
        open for rendering). Other pages have `text=None`."""
        for leaf in self._leaves:
            for i in range(leaf.count):
                if leaf.render_text is not None:
                    image, text = leaf.render_text(i)
                    yield Page(image, text)
                else:
                    yield Page(leaf.render(i))

    def close(self) -> None:
        for leaf in self._leaves:
            leaf.close()


def _mentions_encrypt(data: Union[bytes, Path]) -> bool:
    if isinstance(data, bytes):
        return b"/Encrypt" in data
    tail = b""
    with open(data, "rb") as handle:
        while chunk := handle.read(1 << 20):
            if b"/Encrypt" in tail + chunk:
                return True
            tail = chunk[-8:]
    return False


def _pdf_error(exc: Exception, data: Union[bytes, Path]) -> SourceError:
    """A short message for a PDF PDFium would not open.

    PDFium's error code is a process-wide "last error" that is not reset by a load that fails
    without setting one (measured with pypdfium2 5.15.2: after one encrypted file, a PDF with no
    pages is also reported as "password"). So a password error is only believed when the file
    itself declares an /Encrypt dictionary.
    """
    import pypdfium2.raw as raw

    code = getattr(exc, "err_code", None)
    if code in (raw.FPDF_ERR_PASSWORD, raw.FPDF_ERR_SECURITY) and _mentions_encrypt(data):
        return SourceError("the PDF is encrypted (password protected); remove the password first")
    return SourceError("the PDF is damaged, has no pages, or is not a readable PDF")


def _pdf_leaf(data: Union[bytes, Path]) -> _Leaf:
    import pypdfium2 as pdfium

    try:
        with _PDFIUM_LOCK:
            # Bytes stay in memory (PDFium reads them through a callback); a path is read in place.
            pdf = pdfium.PdfDocument(bytes(data) if not isinstance(data, Path) else str(data))
            count = len(pdf)
    except pdfium.PdfiumError as exc:
        raise _pdf_error(exc, data) from None
    if count == 0:
        with _PDFIUM_LOCK:
            pdf.close()
        raise SourceError("the PDF has no pages")

    def render_text(index: int, text: bool = True):
        try:
            with _PDFIUM_LOCK:
                page = pdf[index]
                try:
                    width, height = page.get_size()
                    longest = max(width, height)
                    if not (longest > 0) or longest == float("inf"):
                        raise SourceError(f"page {index + 1} of the PDF has no usable size")
                    # The longest side comes out at RENDER_LONG_SIDE whatever the page's own
                    # dimensions, so a huge page cannot make a huge bitmap.
                    image = page.render(scale=RENDER_LONG_SIDE / longest).to_pil()
                    layer = None
                    if text:
                        from dococr.pdftext import read_text_layer

                        try:
                            layer = read_text_layer(page, image.size)
                        except Exception:  # an unreadable text layer: the page is OCR'd
                            layer = None
                finally:
                    page.close()
        except pdfium.PdfiumError as exc:
            raise SourceError(f"page {index + 1} of the PDF cannot be rendered") from exc
        return image.convert("RGB"), layer

    def render(index: int) -> Image.Image:
        return render_text(index, text=False)[0]

    def close() -> None:
        with _PDFIUM_LOCK:
            pdf.close()

    return _Leaf(count, render, close, render_text)


def _frames_leaf(image: Image.Image, count: int) -> _Leaf:
    def render(index: int) -> Image.Image:
        image.seek(index)
        _check_pixels(image.size)
        return _finish(image)

    return _Leaf(count, render, image.close)


def _leaf_for(source: object) -> _Leaf:
    if isinstance(source, Image.Image):
        _check_pixels(source.size)
        return _Leaf(1, lambda _i: _finish(source))
    if isinstance(source, (bytes, bytearray)):
        data = bytes(source)
        if _is_pdf_bytes(data):
            return _pdf_leaf(data)
        image = Image.open(io.BytesIO(data))
    elif isinstance(source, (str, Path)):
        path = Path(source)
        with open(path, "rb") as handle:
            head = handle.read(_PDF_MAGIC_WINDOW)
        if _is_pdf_bytes(head):
            return _pdf_leaf(path)
        image = Image.open(path)
    else:
        raise SourceError(f"cannot read a document from a {type(source).__name__}")
    frames = getattr(image, "n_frames", 1) if image.format in _PAGED_FORMATS else 1
    if frames > 1:
        return _frames_leaf(image, frames)
    _check_pixels(image.size)
    return _Leaf(1, lambda _i: _finish(image), image.close)


def _flatten(source: object, out: list, depth: int = 0) -> None:
    if isinstance(source, (list, tuple)):
        if depth > 8:
            raise SourceError("sources are nested too deeply")
        for item in source:
            _flatten(item, out, depth + 1)
    else:
        out.append(source)


@contextmanager
def open_pages(source: ImageSource, max_pages: int = DEFAULT_MAX_PAGES) -> Iterator[Pages]:
    """Open a source as pages. Raises `PageLimitError` before rendering anything when the
    document has more than `max_pages` pages.

        with open_pages(source) as pages:
            for image in pages: ...
    """
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1")
    flat: list = []
    _flatten(source, flat)
    if not flat:
        raise SourceError("no pages were given")
    leaves: list[_Leaf] = []
    pages = Pages(leaves)
    try:
        for item in flat:
            leaves.append(_leaf_for(item))
            # Stop early: do not open the rest of a long list once the limit is passed.
            if len(pages) > max_pages:
                break
        total = len(pages)
        if total > max_pages:
            raise PageLimitError(
                f"the document has {total if len(flat) == len(leaves) else f'more than {max_pages}'}"
                f" pages; the limit is {max_pages} (raise max_pages, or DOCOCR_MAX_PAGES for the service)"
            )
        yield pages
    finally:
        pages.close()
