"""Offline extractor: RapidOCR (ONNX Runtime, CPU) + rule-based label/value matching.

Nothing leaves the machine. The OCR model is loaded once per process, on the first page that
needs OCR, and reused. A PDF page read from its text layer (`text_layer`, see
`dococr.pdftext`) needs no OCR, so a document whose pages all have a trusted text layer never
loads the model.
"""

from __future__ import annotations

import importlib.util
from typing import Optional

from PIL import Image

from dococr.extractors.base import Extractor, ExtractorError, PageLines
from dococr.models import DocType, ExtractorOutput, FieldResult, FieldSpec
from dococr.ocr import Box, OcrEngine, OcrLine, get_default_engine
from dococr.ocr.matcher import match_fields
from dococr.preprocess import prepare

# An OCR line read inside an area of a hybrid page is dropped when this share of it lies on a
# line of the text layer: the text layer already has that text, exactly.
OVERLAP_DROP = 0.3


def _overlap(a: Box, b: Box) -> float:
    """Share of box `a` covered by box `b`."""
    w = min(a[2], b[2]) - max(a[0], b[0])
    h = min(a[3], b[3]) - max(a[1], b[1])
    if w <= 0 or h <= 0:
        return 0.0
    return w * h / max(1e-6, (a[2] - a[0]) * (a[3] - a[1]))


class LocalExtractor(Extractor):
    name = "local"
    supports_text_layer = True

    def __init__(self, engine: Optional[OcrEngine] = None):
        # Fail early (ExtractorError) when the OCR dependency is missing, without importing it:
        # the engine itself is built on the first page that needs OCR.
        if engine is None and importlib.util.find_spec("rapidocr") is None:
            raise ExtractorError("The OCR engine (rapidocr) is not installed. Reinstall dococr.")
        self._given = engine

    @property
    def _engine(self) -> OcrEngine:
        if self._given is None:
            self._given = get_default_engine()
        return self._given

    def warm_up(self) -> None:
        """Load the OCR models now rather than on the first image."""
        self._engine.warm_up()

    def _ocr(self, image: Image.Image):
        try:
            work, transform = prepare(image, self._engine)
            if transform.line_quads is not None:
                # The page's lines were already found by the detector pass in `prepare`.
                lines = self._engine.recognize(work, transform.line_quads)
            else:
                lines = self._engine.read(work)
        except ExtractorError:
            raise
        except Exception as exc:  # pragma: no cover - unexpected engine failure
            raise ExtractorError(f"Local OCR failed: {type(exc).__name__}: {exc}") from exc
        return work, transform, lines

    def extract(self, image: Image.Image, doc_type: DocType, fields: list[FieldSpec],
                *, text_layer=None) -> ExtractorOutput:
        if text_layer is not None and text_layer.kind in ("text", "hybrid"):
            return self._extract_text(image, doc_type, fields, text_layer)
        work, transform, lines = self._ocr(image)
        warnings = list(transform.notes)
        if text_layer is not None:
            warnings = list(text_layer.notes) + warnings
        if not lines:
            warnings.append("no text found on the image")
            return ExtractorOutput(
                fields=[FieldResult(name=f.name, warnings=["no text found"]) for f in fields],
                raw_text="",
                warnings=warnings,
            )

        page_size = (work.shape[1], work.shape[0])
        matches, text = match_fields(lines, doc_type, fields, page_size=page_size)
        results = [
            FieldResult(
                name=m.name,
                raw=m.raw,
                confidence=m.confidence,
                bbox=transform.box_to_original(m.box) if m.box is not None else None,
                warnings=m.warnings,
            )
            for m in matches
        ]
        out = ExtractorOutput(fields=results, raw_text=text, warnings=warnings)
        out._page_lines = PageLines(lines, page_size, transform.box_to_original, _claimed(matches))
        return out

    def _extract_text(self, image: Image.Image, doc_type: DocType, fields: list[FieldSpec],
                      layer) -> ExtractorOutput:
        """A PDF page read from its text layer; areas it does not cover (`layer.regions`) are
        read by OCR and their lines added, except where the text layer already has the text."""
        lines: list[OcrLine] = list(layer.lines)
        warnings = list(layer.notes)
        for region in layer.regions:
            l, t, r, b = (int(max(0, region[0])), int(max(0, region[1])),
                          int(min(image.width, region[2] + 1)), int(min(image.height, region[3] + 1)))
            if r - l < 8 or b - t < 8:
                continue
            crop = image.crop((l, t, r, b))
            _, transform, found = self._ocr(crop)
            for line in found:
                x0, y0, x1, y1 = transform.box_to_original(line.box)
                box = (x0 + l, y0 + t, x1 + l, y1 + t)
                if any(_overlap(box, known.box) > OVERLAP_DROP for known in layer.lines):
                    continue
                lines.append(OcrLine(line.text, line.confidence, box))
        if not lines:
            warnings.append("no text found on the image")
            return ExtractorOutput(
                fields=[FieldResult(name=f.name, warnings=["no text found"]) for f in fields],
                raw_text="", warnings=warnings)
        page_size = (image.width, image.height)
        matches, text = match_fields(lines, doc_type, fields, page_size=page_size)
        results = [
            FieldResult(
                name=m.name, raw=m.raw, confidence=m.confidence,
                bbox=_int_box(m.box, page_size) if m.box is not None else None,
                warnings=m.warnings)
            for m in matches
        ]
        out = ExtractorOutput(fields=results, raw_text=text, warnings=warnings)
        out._page_lines = PageLines(lines, page_size, lambda box: _int_box(box, page_size),
                                    _claimed(matches))
        return out


def _claimed(matches) -> dict:
    return {m.name: [box for _, box in m.parts] for m in matches if m.raw is not None}


def _int_box(box: Box, size: tuple[int, int]) -> tuple[int, int, int, int]:
    import math

    w, h = size
    return (int(max(0, math.floor(box[0]))), int(max(0, math.floor(box[1]))),
            int(min(w, math.ceil(box[2]))), int(min(h, math.ceil(box[3]))))
