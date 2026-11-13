"""Command line interface: a thin wrapper over `dococr.pipeline.extract`."""

from __future__ import annotations

import json
import os
from enum import Enum
from pathlib import Path
from typing import Optional

import typer

from dococr import pipeline
from dococr.doctypes import find_doc_types, load_doc_type, load_doc_type_file
from dococr.extractors.base import ExtractorError
from dococr.imaging import DEFAULT_MAX_PAGES, SourceError
from dococr.models import ExtractionResult

app = typer.Typer(
    help="Extract the fields you ask for from photos of documents.",
    add_completion=False,
    no_args_is_help=True,
)


_TYPES_DIR = typer.Option(
    None,
    "--types-dir",
    help="Folder of document-type YAML files, searched before DOCOCR_DOCTYPES_DIR and the shipped types. Repeatable.",
)


class Format(str, Enum):
    json = "json"
    table = "table"


class PdfText(str, Enum):
    auto = "auto"
    always = "always"
    never = "never"


def _fail(message: str) -> "typer.Exit":
    typer.echo(f"error: {' '.join(str(message).split())}", err=True)
    return typer.Exit(code=1)


def _table(result: ExtractionResult, label: str) -> str:
    lines = [f"{label}  [{result.doc_type}]"]
    width = max((len(n) for n in result.fields), default=0)
    for name, f in result.fields.items():
        value = "(missing)" if f.value is None else str(f.value)
        page = f"  p{f.page}" if result.pages > 1 and f.page else ""
        if f.parts:
            page = "  p" + "+".join(str(p.page) for p in f.parts)
        conf = page + ("" if f.confidence is None else f"  conf={f.confidence:.2f}")
        warn = f"  ! {'; '.join(f.warnings)}" if f.warnings else ""
        lines.append(f"  {name.ljust(width)}  {value}{conf}{warn}")
    for w in result.warnings:
        lines.append(f"  warning: {w}")
    if result.raw_text:
        lines.append("  --- text on the page ---")
        lines.extend(f"  {line}" for line in result.raw_text.splitlines())
    return "\n".join(lines)


@app.command()
def extract(
    images: list[Path] = typer.Argument(
        ..., help="Image or PDF file(s) to read. A PDF or multi-page TIFF is one document."
    ),
    doc_type: Optional[str] = typer.Option(None, "--type", "-t", help="Document type name (see `dococr types`) or a YAML file."),
    fields: Optional[str] = typer.Option(
        None, "--fields", "-f", help="Comma-separated field names. Default: all fields of the type."
    ),
    doc_type_file: Optional[Path] = typer.Option(
        None, "--doc-type-file", help="YAML file describing a document type."
    ),
    types_dir: Optional[list[Path]] = _TYPES_DIR,
    output_format: Format = typer.Option(Format.json, "--format", help="json or table."),
    raw_text: bool = typer.Option(False, "--raw-text", help="Include the full text read off the page."),
    one_document: bool = typer.Option(
        False,
        "--one-document",
        help="Treat the given files as the pages of ONE document, in order, and print one result. "
        "Without it, each file is a separate document.",
    ),
    max_pages: int = typer.Option(
        DEFAULT_MAX_PAGES, "--max-pages", min=1, help="Refuse a document with more pages than this."
    ),
    pdf_text: PdfText = typer.Option(
        PdfText.auto,
        "--pdf-text",
        help="How to use a PDF's own text layer: auto (when it can be trusted; OCR otherwise), "
        "always (as it is, even an invisible scanner's layer), never (OCR every page).",
    ),
) -> None:
    """Extract fields from one or more document images or PDFs. Runs fully offline."""
    if doc_type_file is not None:
        try:
            doc = load_doc_type_file(doc_type_file)
        except Exception as exc:
            raise _fail(str(exc))
    elif doc_type:
        try:
            doc = load_doc_type(doc_type, types_dir)
        except Exception as exc:  # --type also accepts a YAML path
            raise _fail(str(exc))
    else:
        raise _fail("give --type or --doc-type-file")
    wanted = [f.strip() for f in fields.split(",") if f.strip()] if fields else None

    for path in images:
        if not path.is_file():
            raise _fail(f"no such file: {path}")
    groups = [list(images)] if one_document else [[p] for p in images]
    labels = [" + ".join(str(p) for p in g) for g in groups]

    results = []
    for group, label in zip(groups, labels):
        source = group[0] if len(group) == 1 else group
        try:
            results.append(
                pipeline.extract(source, doc, wanted, include_raw_text=raw_text, max_pages=max_pages,
                                 pdf_text=pdf_text.value)
            )
        except ExtractorError as exc:
            raise _fail(str(exc))
        except SourceError as exc:
            raise _fail(f"{label}: {exc}")
        except ValueError as exc:
            raise _fail(str(exc))
        except Exception as exc:  # unreadable / corrupt image
            raise _fail(f"cannot process {label}: {exc}")

    if output_format is Format.table:
        typer.echo("\n\n".join(_table(r, label) for r, label in zip(results, labels)))
        return
    payload = [r.model_dump(mode="json") for r in results]
    typer.echo(json.dumps(payload[0] if len(payload) == 1 else payload, indent=2, ensure_ascii=False))


@app.command("types")
def types_(types_dir: Optional[list[Path]] = _TYPES_DIR) -> None:
    """List the document types found in the searched folders."""
    try:
        docs, skipped = find_doc_types(types_dir)
    except Exception as exc:
        raise _fail(str(exc))
    for message in skipped:
        typer.echo(f"skipped {' '.join(message.split())}", err=True)
    for doc in docs:
        typer.echo(f"{doc.name}\t{doc.title}")


@app.command()
def fields(
    doc_type: str = typer.Argument(..., help="Document type name or alias (see `dococr types`)."),
    types_dir: Optional[list[Path]] = _TYPES_DIR,
) -> None:
    """List the fields of a document type."""
    try:
        doc = load_doc_type(doc_type, types_dir)
    except Exception as exc:
        raise _fail(str(exc))
    if not doc.fields:
        raise _fail(f"unknown document type '{doc_type}'. Run: dococr types")
    for f in doc.fields:
        typer.echo(f"{f.name}\t{f.type.value}\t{' | '.join(f.labels())}")


@app.command()
def serve(
    host: str = typer.Option("127.0.0.1", help="Interface to bind. Default is local only."),
    port: int = typer.Option(8000, help="Port."),
    warm_up: bool = typer.Option(
        True,
        "--warm-up/--no-warm-up",
        help="Load the OCR models before accepting requests (a few seconds, some memory). "
        "Without it the first request pays for loading. DOCOCR_WARMUP=0 also turns it off.",
    ),
) -> None:
    """Run the HTTP service."""
    try:
        import uvicorn

        from dococr.api import WARMUP_ENV, warmup_enabled
        from dococr.api import app as api_app
    except ImportError:
        raise _fail("the HTTP service needs extra packages. Run: pip install 'dococr[api]'")
    if not warm_up:
        os.environ[WARMUP_ENV] = "0"  # the app's start-up hook reads this
    elif warmup_enabled():
        typer.echo("loading OCR models...", err=True)
        try:
            pipeline.warm_up()
        except ExtractorError as exc:  # fail here, with a plain message, not on the first request
            raise _fail(f"cannot start the service: {exc}")
    uvicorn.run(api_app, host=host, port=port)
