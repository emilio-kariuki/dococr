"""The one entry point everything goes through: image + doc type + wanted fields -> result.

Everything runs on this machine. Nothing here, or below it, opens a network connection.
"""

from __future__ import annotations

import threading
import time
from pathlib import Path
from typing import Optional, Union

from dococr.doctypes import load_doc_type, resolve_fields
from dococr.extractors.base import Extractor, ExtractorError
from dococr.imaging import DEFAULT_MAX_PAGES, ImageSource, Page, open_pages
from dococr.models import (
    BBox,
    DocType,
    ExtractionResult,
    ExtractorOutput,
    FieldResult,
    FieldSpec,
    ValuePart,
)
from dococr.normalize import normalize_value

PDF_TEXT_MODES = ("auto", "always", "never")

_default: Optional[Extractor] = None
_default_lock = threading.Lock()


def get_extractor() -> Extractor:
    """The offline OCR extractor, built once per process (loading the model is the slow part)."""
    global _default
    with _default_lock:
        if _default is None:
            try:
                from dococr.extractors.local import LocalExtractor
            except ImportError as exc:
                raise ExtractorError(
                    f"The OCR engine is not installed correctly ({exc}). Reinstall dococr."
                ) from exc
            _default = LocalExtractor()
        return _default


def warm_up() -> Extractor:
    """Build the default extractor and load its models now, so the first `extract` is not slow.

    Safe to call repeatedly and from several threads. An extractor without a `warm_up()` method
    is simply built. Raises ExtractorError if the engine cannot be built or its models cannot be
    loaded.
    """
    engine = get_extractor()
    loader = getattr(engine, "warm_up", None)
    if callable(loader):
        try:
            loader()
        except ExtractorError:
            raise
        except Exception as exc:
            raise ExtractorError(f"The OCR models could not be loaded ({exc}).") from exc
    return engine


def _normalise(found: FieldResult, spec: FieldSpec, page: int) -> FieldResult:
    raw = found.raw.strip() if found.raw else None
    if not raw:
        return FieldResult(name=spec.name, warnings=list(found.warnings))
    value, warnings = normalize_value(raw, spec)
    return found.model_copy(
        update={"raw": raw, "value": value, "warnings": [*found.warnings, *warnings], "page": page}
    )


def _normalise_page(
    output: ExtractorOutput, specs: list[FieldSpec], page: int
) -> dict[str, FieldResult]:
    """One page's output as a FieldResult per spec, normalised, stamped with its page number."""
    by_name = {f.name: f for f in output.fields}
    return {
        spec.name: _normalise(by_name.get(spec.name) or FieldResult(name=spec.name), spec, page)
        for spec in specs
    }


def _rank(result: FieldResult) -> tuple[int, float]:
    # A reading that normalised to a value beats one that did not; then higher confidence.
    # A reading with no confidence ranks below any that has one.
    conf = -1.0 if result.confidence is None else result.confidence
    return (0 if result.value is None else 1, conf)


def _merge_field(name: str, pages: list[dict[str, FieldResult]], many: bool,
                 extra: Optional[list[tuple[int, FieldResult]]] = None) -> FieldResult:
    """Pick one page's reading of a field. Highest rank wins, the earlier page on a tie.

    `extra` holds further readings (page, result) found across a page break where the page's
    own reading is of a different occurrence."""
    readings = [(n, p[name]) for n, p in enumerate(pages, start=1) if p[name].raw]
    if extra:
        readings = sorted(readings + [(n, r) for n, r in extra if r.raw], key=lambda nr: nr[0])
    if not readings:
        warnings = [
            f"page {n}: {w}" if many else w
            for n, p in enumerate(pages, start=1)
            for w in p[name].warnings
        ]
        return FieldResult(name=name, warnings=warnings)
    best_page, best = readings[0]
    for n, reading in readings[1:]:
        if _rank(reading) > _rank(best):  # strict: a tie keeps the earlier page
            best_page, best = n, reading
    warnings = list(best.warnings)
    others = sorted({
        n
        for n, reading in readings
        if reading is not best and reading.value is not None and reading.value != best.value
    })
    if best_page in others:  # a reading across a page break of another occurrence
        warnings.append(f"page {best_page} has two different readings; check the document")
        others = [n for n in others if n != best_page]
    if others:
        names = ", ".join(str(n) for n in others)
        plural = "pages" if len(others) > 1 else "page"
        warnings.append(
            f"{plural} {names} read a different value than page {best_page}; check the document"
        )
    return best.model_copy(update={"warnings": warnings, "page": best_page})


def _intersects(a: BBox, b: BBox) -> bool:
    return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])


def _join_breaks(layouts: list, doc: DocType, specs: list[FieldSpec],
                 page_results: list[dict[str, FieldResult]]) -> dict[str, list[tuple[int, FieldResult]]]:
    """Read `until` and `multiline` values across each page break (`dococr.pagebreak`) and fold
    the readings into `page_results`. A reading replaces the page's own reading of the field when
    that is missing or is the same value cut short (their boxes overlap); a reading of another
    occurrence is returned as an extra reading for the merge."""
    from dococr.pagebreak import find_furniture, join_pair

    extra: dict[str, list[tuple[int, FieldResult]]] = {}
    furniture = find_furniture(layouts, doc.page_furniture)
    by_name = {s.name: s for s in specs}
    for k in range(len(layouts) - 1):
        a, b = layouts[k], layouts[k + 1]
        if a is None or b is None:
            continue
        for name, j in join_pair(a, b, furniture[k], furniture[k + 1], doc, specs).items():
            pieces = [(k + off, (a if off == 0 else b).to_original(box)) for off, box in j.parts]
            start = k + j.start  # 0-based page where the value starts
            own = page_results[start][name]
            if j.clash is not None:  # two fields cannot take the same words
                own.warnings.append(f"the value may continue on page {k + 2}; not joined because "
                                    f"that text was read as field '{j.clash}'")
                continue
            found = FieldResult(
                name=name, raw=j.raw, confidence=j.confidence, bbox=pieces[0][1],
                warnings=list(j.warnings),
                parts=[ValuePart(page=p + 1, bbox=box) for p, box in pieces] if len(pieces) > 1 else None,
            )
            reading = _normalise(found, by_name[name], start + 1)
            if not own.raw or (own.bbox is not None and _intersects(own.bbox, pieces[0][1])):
                page_results[start][name] = reading
            else:
                extra.setdefault(name, []).append((start + 1, reading))
    return extra


def extract(
    source: ImageSource,
    doc_type: Union[str, Path, DocType],
    fields: Optional[list[Union[str, FieldSpec]]] = None,
    *,
    extractor: Optional[Extractor] = None,
    include_raw_text: bool = False,
    max_pages: int = DEFAULT_MAX_PAGES,
    pdf_text: str = "auto",
) -> ExtractionResult:
    """Extract `fields` (default: all preset fields) of a `doc_type` document.

    `source` is an image (path, bytes or PIL image), a PDF (path or bytes), a multi-frame image
    file such as a multi-page TIFF, or a list or tuple of sources, which are the pages of one
    document in order. Each page is read on its own and the readings are merged into one result:
    per field the reading with the highest confidence wins, the earlier page on a tie, and a
    different value on another page adds a warning. More than `max_pages` pages raises a
    ValueError (`dococr.imaging.PageLimitError`).

    A value inside a sentence (`until`) or a block of lines (`multiline`) that runs from the
    bottom of one page onto the next is read across the break, skipping running headers,
    footers and page numbers; its `page` and `bbox` are where it starts and `parts` lists every
    piece. Form-style values are read within one page.

    `pdf_text` decides how a PDF page's own text layer is used (see `dococr.pdftext`): "auto"
    (default) reads a trustworthy text layer in place of OCR and OCRs the page, or the parts of
    it the layer does not cover, otherwise; "always" uses any text layer as it is, invisible or
    not; "never" OCRs every page. Each page says what was done in the result's `warnings`.
    Images are always OCR'd.

    Runs fully offline. Pass `extractor` only to supply your own Extractor implementation.
    """
    started = time.perf_counter()
    if pdf_text not in PDF_TEXT_MODES:
        raise ValueError(f"pdf_text must be one of {', '.join(PDF_TEXT_MODES)}, not {pdf_text!r}")
    doc = load_doc_type(doc_type)
    specs = resolve_fields(doc, fields)
    engine = extractor or get_extractor()
    use_text = pdf_text != "never" and getattr(engine, "supports_text_layer", False)

    page_results: list[dict[str, FieldResult]] = []
    page_warnings: list[list[str]] = []
    page_texts: list[Optional[str]] = []
    layouts: list = []
    with open_pages(source, max_pages) as pages:
        for page in (pages.with_text() if use_text else (Page(image) for image in pages)):
            layer = None
            if page.text is not None:
                from dococr.pdftext import TextLayer, build_layer

                try:
                    layer = build_layer(page.text, page.image, pdf_text)
                except Exception:  # never let a strange text layer stop the page being read
                    layer = TextLayer("ocr", notes=[("the PDF's text layer could not be read; "
                                                     "page read by OCR")])
            if layer is not None and layer.kind != "ocr":
                output = engine.extract(page.image, doc, specs, text_layer=layer)
            else:
                output = engine.extract(page.image, doc, specs)
                if layer is not None and layer.notes:
                    output.warnings = [*layer.notes, *output.warnings]
            page_results.append(_normalise_page(output, specs, len(page_results) + 1))
            page_warnings.append(list(output.warnings))
            page_texts.append(output.raw_text)
            layouts.append(getattr(output, "_page_lines", None))
            del page  # let go of the page before the next one is rendered
    count = len(page_results)
    many = count > 1

    extra: dict[str, list[tuple[int, FieldResult]]] = {}
    cross = [s for s in specs if s.until_texts() or s.multiline]
    if many and cross and any(layouts):
        extra = _join_breaks(layouts, doc, specs, page_results)

    results = {
        spec.name: _merge_field(spec.name, page_results, many, extra.get(spec.name))
        for spec in specs
    }
    # A field read but not parsed into its type (raw text, value None) has no value either.
    missing = [name for name, r in results.items() if r.value is None]

    warnings = [
        f"page {n}: {w}" if many else w
        for n, ws in enumerate(page_warnings, start=1)
        for w in ws
    ]
    raw_text = None
    if include_raw_text:
        if many:
            raw_text = "\n".join(
                f"--- page {n} ---\n{text or ''}" for n, text in enumerate(page_texts, start=1)
            )
        else:
            raw_text = page_texts[0]

    return ExtractionResult(
        doc_type=doc.name,
        backend=engine.name,
        fields=results,
        missing=missing,
        warnings=warnings,
        raw_text=raw_text,
        elapsed_ms=int((time.perf_counter() - started) * 1000),
        pages=count,
    )
