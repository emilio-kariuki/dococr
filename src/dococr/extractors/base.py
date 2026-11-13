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


class Extractor(ABC):
    name: ClassVar[str]
    # True when `extract` accepts the keyword `text_layer` (a `dococr.pdftext.TextLayer`: text
    # lines read from a PDF's text layer, in the image's pixel coordinates, to use in place of
    # OCR). The pipeline passes it only to extractors that set this; others get images only.
    supports_text_layer: ClassVar[bool] = False

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
