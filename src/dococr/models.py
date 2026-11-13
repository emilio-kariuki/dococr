"""Shared data contract. Every extractor and every interface (CLI, HTTP, Python) speaks these types.

Part of the data contract: see docs/ARCHITECTURE.md before changing anything here.
"""

from __future__ import annotations

import re
from enum import Enum
from typing import Any, Literal, Optional, Union

from pydantic import BaseModel, Field, PrivateAttr, model_validator

Value = Union[str, int, float, None]
# Pixel box in the coordinates of the image handed to the extractor: (left, top, right, bottom).
BBox = tuple[int, int, int, int]


def slugify(name: str) -> str:
    """Canonical field / doc-type key: 'Order Number' -> 'order_number'."""
    return re.sub(r"[^a-z0-9]+", "_", name.strip().lower()).strip("_")


class FieldType(str, Enum):
    STRING = "string"  # free text, whitespace collapsed
    NAME = "name"  # a person's name
    DATE = "date"  # normalised to ISO 8601 (YYYY-MM-DD)
    INTEGER = "integer"
    NUMBER = "number"
    IDENTIFIER = "identifier"  # serials and reference numbers: upper-cased, no inner spaces


class FieldSpec(BaseModel):
    """One thing the caller wants read off the document."""

    name: str  # canonical key, e.g. "order_date"
    label: str = ""  # text printed on the document next to the value, e.g. "Order Date"
    aliases: list[str] = Field(default_factory=list)  # other printed variants of the label, each listed in full
    type: FieldType = FieldType.STRING
    pattern: Optional[str] = None  # regex a valid value fully matches, e.g. r"[A-Z]{2}\d{6}"
    description: str = ""  # note for people reading the doc type; not used for matching
    # The value may continue on the lines below its first line (a wrapped address or name).
    multiline: bool = False
    # Text printed straight after the value; the value ends just before it and may run on across
    # line breaks in reading order (values inside prose). One text, or a list of alternatives.
    until: Union[str, list[str], None] = None
    # How a numeric date with the year last is written: "day_first" (12/05/08 = 12 May 2008) or
    # "month_first" (12/05/08 = 5 December 2008). None takes the document type's `date_order`,
    # and day first when that is not set either. Only `type: date` fields use it.
    date_order: Optional[Literal["day_first", "month_first"]] = None

    @model_validator(mode="after")
    def _defaults(self) -> "FieldSpec":
        self.name = slugify(self.name)
        if not self.label:
            self.label = self.name.replace("_", " ").title()
        return self

    def labels(self) -> list[str]:
        """Label followed by aliases, de-duplicated case-insensitively."""
        seen: set[str] = set()
        out: list[str] = []
        for text in [self.label, *self.aliases]:
            key = text.strip().lower()
            if key and key not in seen:
                seen.add(key)
                out.append(text.strip())
        return out

    def until_texts(self) -> list[str]:
        """The `until` alternatives as a list (empty when the field has none)."""
        if self.until is None:
            return []
        items = [self.until] if isinstance(self.until, str) else self.until
        return [t.strip() for t in items if t and t.strip()]

    def month_first(self) -> bool:
        """Numeric dates of this field are written month first (`date_order: month_first`)."""
        return self.date_order == "month_first"


class DocType(BaseModel):
    """A kind of document, as described by a YAML file.

    Everything the code knows about a particular document comes from here. The code itself
    holds no document names, labels or value formats.
    """

    name: str  # canonical key; the YAML file is <name>.yaml
    title: str = ""
    description: str = ""
    aliases: list[str] = Field(default_factory=list)  # other names the type can be asked for by
    # Labels printed on the document that are not fields to extract. They are never returned;
    # they tell the matcher where a neighbouring value ends.
    other_labels: list[str] = Field(default_factory=list)
    # Text printed at the top or bottom of pages that is not part of the document's content
    # (a running header the pages do not repeat, "continued overleaf"). Skipped when a value is
    # read across a page break. Page numbers and text repeated on several pages are found
    # without it.
    page_furniture: list[str] = Field(default_factory=list)
    # How the document writes numeric dates: "day_first" or "month_first". The default for every
    # field that does not set its own `date_order`; None means day first.
    date_order: Optional[Literal["day_first", "month_first"]] = None
    fields: list[FieldSpec] = Field(default_factory=list)

    @model_validator(mode="after")
    def _defaults(self) -> "DocType":
        self.name = slugify(self.name)
        if self.date_order is not None:
            for spec in self.fields:
                if spec.date_order is None:
                    spec.date_order = self.date_order
        if not self.title:
            self.title = self.name.replace("_", " ").title()
        return self

    def field(self, name: str) -> Optional[FieldSpec]:
        key = slugify(name)
        return next((f for f in self.fields if f.name == key), None)


class ValuePart(BaseModel):
    """One page's piece of a value that runs across a page break."""

    page: int  # 1-based
    bbox: BBox  # in the pixel coordinates of that page's image


class FieldResult(BaseModel):
    name: str
    value: Value = None  # normalised value; set by the pipeline, not by extractors
    raw: Optional[str] = None  # text exactly as read off the document; None = not found
    confidence: Optional[float] = None  # 0..1, None when the field was not found
    bbox: Optional[BBox] = None  # where the value was read, when known
    warnings: list[str] = Field(default_factory=list)
    # 1-based page the value was read from; None for a missing field. Set by the pipeline.
    # For a value that runs across a page break: the page where it starts (and `bbox` is the box
    # of its piece on that page).
    page: Optional[int] = None
    # Only for a value that runs across a page break: every piece of it in reading order, the
    # first being (`page`, `bbox`). None for a value on one page. Set by the pipeline.
    parts: Optional[list[ValuePart]] = None


class ExtractorOutput(BaseModel):
    """What an extractor returns: exactly one FieldResult per requested field, in request order."""

    fields: list[FieldResult]
    raw_text: Optional[str] = None  # full text of the page in reading order, when available
    warnings: list[str] = Field(default_factory=list)
    # The page's text lines for reading values across a page break
    # (`dococr.extractors.base.PageLines`). Not part of the result; an extractor that does not
    # set it gets no cross-page reading.
    _page_lines: Any = PrivateAttr(default=None)


class ExtractionResult(BaseModel):
    doc_type: str
    backend: str  # name of the extractor that ran; "local" for the built-in one
    fields: dict[str, FieldResult]
    missing: list[str] = Field(default_factory=list)  # requested fields with no value
    warnings: list[str] = Field(default_factory=list)
    raw_text: Optional[str] = None
    elapsed_ms: int = 0
    pages: int = 1  # how many pages were read

    def values(self) -> dict[str, Value]:
        """Just {field: value} — the usual thing callers want."""
        return {name: result.value for name, result in self.fields.items()}
