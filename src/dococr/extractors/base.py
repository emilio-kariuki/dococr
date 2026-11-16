"""Extractor interface. Part of the data contract: see docs/ARCHITECTURE.md."""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Callable, ClassVar

from PIL import Image

from dococr.models import BBox, DocType, ExtractorOutput, FieldSpec


class ExtractorError(RuntimeError):
    """The extractor could not run at all (missing dependency, missing model file, engine failure).

    A field that simply is not on the page is NOT an error: return it with raw=None.
    """


def _identity(box) -> BBox:
    return tuple(int(round(v)) for v in box)  # type: ignore[return-value]


@dataclass
class PageLines:
    """A page's text lines, upright, as the matcher read them: what the pipeline needs to read a
    value across a page break. `lines` are `dococr.ocr.OcrLine`s in a working image of `size`
    (width, height) where the text is upright; `to_original` maps a box in that image to the
    pixel coordinates of the page image the extractor was given."""

    lines: list
    size: tuple[float, float]
    to_original: Callable = field(default=_identity)
    # Boxes (in the working image) of the words each field's reading on this page took, so a
    # value read across a page break does not take words another field already has.
    claimed: dict = field(default_factory=dict)


@dataclass
class PageRead:
    """A page read but not yet matched: its text lines, and what reading it did.

    Returned by `Extractor.read_page` and given back to `Extractor.match_page`, so one reading
    of a page (the OCR, about a second) can be matched against several document types (about
    10 ms each), which is how the document type is detected. `lines`, `size` and `to_original`
    mean what they mean in `PageLines`; `warnings` are the page-level notes of the reading
    (rotation, deskew, text layer) that `extract` would have returned."""

    lines: list
    size: tuple[float, float]
    to_original: Callable = field(default=_identity)
    warnings: list = field(default_factory=list)


class Extractor(ABC):
    name: ClassVar[str]
    # True when `extract` accepts the keyword `text_layer` (a `dococr.pdftext.TextLayer`: text
    # lines read from a PDF's text layer, in the image's pixel coordinates, to use in place of
    # OCR). The pipeline passes it only to extractors that set this; others get images only.
    supports_text_layer: ClassVar[bool] = False

    # Optional, for detecting the document type: an extractor that implements both methods
    # below lets the pipeline read each page once and try every candidate type on the same
    # lines. One that does not still extracts with a named type; asking it to detect the type
    # is refused with a ValueError before any page is read.
    def read_page(self, image: Image.Image, *, text_layer=None) -> PageRead:
        """Read `image` (and `text_layer`, when `supports_text_layer`) into text lines without
        matching any document type."""
        raise NotImplementedError

    def match_page(self, page: PageRead, doc_type: DocType, fields: list[FieldSpec]) -> ExtractorOutput:
        """Match the lines of a page read by `read_page`; the same contract as `extract`."""
        raise NotImplementedError

    @property
    def supports_reading(self) -> bool:
        """True when this extractor implements `read_page` and `match_page`."""
        cls = type(self)
        return cls.read_page is not Extractor.read_page and cls.match_page is not Extractor.match_page

    @abstractmethod
    def extract(
        self, image: Image.Image, doc_type: DocType, fields: list[FieldSpec]
    ) -> ExtractorOutput:
        """Read `fields` off `image`.

        - `image` is RGB, already EXIF-rotated; do not mutate it.
        - Return exactly one FieldResult per entry of `fields`, same order, same `name`.
        - Fill `raw` with the text as printed (None if not found). Leave `value` alone:
          the pipeline normalises `raw` into `value` using the field's type.
        - `bbox` is in the pixel coordinates of `image`.
        - `doc_type.fields` may contain more fields than requested; `fields` may contain
          ad-hoc fields that are not in `doc_type.fields` (label derived from the name).
        - An extractor that sets `supports_text_layer` also accepts `text_layer=` (see above).
        - Setting `output._page_lines` to a `PageLines` lets the pipeline read values that run
          across a page break; without it, pages are read independently.
        """
