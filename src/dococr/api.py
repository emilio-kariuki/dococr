"""HTTP service: a thin wrapper over `dococr.pipeline.extract`.

Extraction runs on this machine; the service makes no outbound connections. Uploads are held
in memory only: never written to disk, and neither image bytes nor extracted values are logged.
Run it with `dococr serve`.
"""

from __future__ import annotations

import json
import logging
import os
import threading
from contextlib import asynccontextmanager
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, UploadFile
from PIL import Image
from starlette.datastructures import Headers
from starlette.formparsers import MultiPartParser
from starlette.responses import JSONResponse

from dococr import pipeline
from dococr.detection import UnknownDocumentError, is_auto, max_candidates
from dococr.doctypes import DocTypeError, find_doc_types, load_doc_type
from dococr.extractors.base import ExtractorError
from dococr.imaging import DEFAULT_MAX_PAGES

log = logging.getLogger(__name__)

MAX_UPLOAD_MB_ENV = "DOCOCR_MAX_UPLOAD_MB"
DEFAULT_MAX_UPLOAD_MB = 10
_CHUNK = 1024 * 1024

MAX_PAGES_ENV = "DOCOCR_MAX_PAGES"
PDF_TEXT_ENV = "DOCOCR_PDF_TEXT"

_FORM_OVERHEAD = 64 * 1024  # multipart boundaries and the non-file form fields

WARMUP_ENV = "DOCOCR_WARMUP"


def warmup_enabled() -> bool:
    """Start-up warming is on unless DOCOCR_WARMUP is 0, false, no or off."""
    return os.environ.get(WARMUP_ENV, "1").strip().lower() not in ("0", "false", "no", "off")


@asynccontextmanager
async def _lifespan(_app: FastAPI):
    # Runs when the server starts, before it accepts requests. Not run by `TestClient(app)` unless
    # it is used as a context manager. A failure here stops the server instead of the first request.
    if warmup_enabled():
        try:
            pipeline.warm_up()
        except ExtractorError as exc:
            log.error("cannot start: %s", exc)
            raise
    yield


DOCS_ENV = "DOCOCR_API_DOCS"


def _docs_enabled() -> bool:
    return os.environ.get(DOCS_ENV, "").strip().lower() in ("1", "true", "yes", "on")


# The interactive API pages describe every endpoint to whoever can reach the service, so they
# are off unless asked for.
app = FastAPI(
    title="dococr", version="0.1.0", lifespan=_lifespan,
    docs_url="/docs" if _docs_enabled() else None,
    redoc_url="/redoc" if _docs_enabled() else None,
    openapi_url="/openapi.json" if _docs_enabled() else None,
)

MAX_CONCURRENT_ENV = "DOCOCR_MAX_CONCURRENT"
DEFAULT_MAX_CONCURRENT = 2
BUSY_WAIT_SECONDS = 30.0
MAX_FIELDS = 200
MAX_FIELDS_TEXT = 20_000  # characters of the `fields` form value
MAX_NAME_LENGTH = 200  # characters of a document type or field name


def _max_concurrent() -> int:
    try:
        n = int(os.environ.get(MAX_CONCURRENT_ENV, DEFAULT_MAX_CONCURRENT))
    except ValueError:
        return DEFAULT_MAX_CONCURRENT
    return n if n >= 1 else DEFAULT_MAX_CONCURRENT


# Each extraction decodes and preprocesses whole pages in memory. The OCR engine reads one page
# at a time anyway, so letting dozens of requests decode at once only multiplies memory.
_extract_slots = threading.BoundedSemaphore(_max_concurrent())


def max_upload_bytes() -> int:
    try:
        mb = float(os.environ.get(MAX_UPLOAD_MB_ENV, DEFAULT_MAX_UPLOAD_MB))
    except ValueError:
        mb = DEFAULT_MAX_UPLOAD_MB
    return int(mb * 1024 * 1024)


def max_pages() -> int:
    """Most pages one request may hold (PDF pages plus image parts); DOCOCR_MAX_PAGES."""
    try:
        pages = int(os.environ.get(MAX_PAGES_ENV, DEFAULT_MAX_PAGES))
    except ValueError:
        return DEFAULT_MAX_PAGES
    return pages if pages >= 1 else DEFAULT_MAX_PAGES


def pdf_text_mode() -> str:
    """How PDF text layers are used (auto, always, never); DOCOCR_PDF_TEXT, default auto. Any
    other value is an error on each request, so a typo is not silent."""
    mode = os.environ.get(PDF_TEXT_ENV, "auto").strip().lower() or "auto"
    if mode not in pipeline.PDF_TEXT_MODES:
        raise HTTPException(500, f"{PDF_TEXT_ENV} must be one of {', '.join(pipeline.PDF_TEXT_MODES)}")
    return mode


class _UploadGuard:
    """Refuse oversized request bodies before they are received, and keep uploads in memory.

    Starlette parses the multipart body before the endpoint runs and moves any file part over
    1 MB into a temporary file on disk. So the size check has to happen here, on the declared
    Content-Length, and the in-memory threshold has to be raised to the upload limit.
    """

    def __init__(self, inner):
        self.inner = inner

    async def __call__(self, scope, receive, send):
        if scope["type"] != "http" or scope["method"] not in ("POST", "PUT", "PATCH"):
            return await self.inner(scope, receive, send)
        limit = max_upload_bytes()
        cap = limit + _FORM_OVERHEAD
        declared = Headers(scope=scope).get("content-length", "")
        if not declared.isdigit():
            # Without a declared length the body could not be bounded before it is spooled.
            reply = JSONResponse({"detail": "Content-Length is required"}, status_code=411)
            return await reply(scope, receive, send)
        if int(declared) > cap:
            reply = JSONResponse(
                {"detail": f"upload exceeds the {limit / (1024 * 1024):g} MB limit"}, status_code=413
            )
            return await reply(scope, receive, send)
        MultiPartParser.spool_max_size = cap  # process-wide: file parts up to the cap stay in RAM
        await self.inner(scope, receive, send)


app.add_middleware(_UploadGuard)


def _parse_fields(text: Optional[str]) -> Optional[list[str]]:
    if not text or not text.strip():
        return None
    if len(text) > MAX_FIELDS_TEXT:
        raise HTTPException(422, f"fields: longer than {MAX_FIELDS_TEXT} characters")
    text = text.strip()
    if text.startswith("["):
        try:
            items = json.loads(text)
        except (ValueError, RecursionError):
            raise HTTPException(422, "fields: invalid JSON list")
        if not isinstance(items, list) or not all(isinstance(i, str) for i in items):
            raise HTTPException(422, "fields: expected a list of strings")
    else:
        items = text.split(",")
    names = [i.strip() for i in items if i.strip()]
    if len(names) > MAX_FIELDS:
        raise HTTPException(422, f"fields: more than {MAX_FIELDS} names")
    if any(len(n) > MAX_NAME_LENGTH for n in names):
        raise HTTPException(422, f"fields: a name is longer than {MAX_NAME_LENGTH} characters")
    return names or None


@app.get("/health")
def health() -> dict:
    return {"status": "ok"}


def _is_path(text: str) -> bool:
    return text.strip().lower().endswith((".yaml", ".yml"))


@app.get("/doc-types")
def doc_types() -> list[dict]:
    try:
        docs, skipped = find_doc_types()
    except DocTypeError as exc:
        raise HTTPException(500, str(exc))
    for message in skipped:
        log.warning("skipped document type file %s", message)
    return [{"name": d.name, "title": d.title} for d in docs]


@app.get("/doc-types/{name}")
def doc_type(name: str) -> dict:
    try:
        doc = None if _is_path(name) else load_doc_type(name)
    except DocTypeError as exc:
        raise HTTPException(422, str(exc))
    if doc is None or not doc.fields:
        raise HTTPException(404, f"unknown document type '{name}'")
    return doc.model_dump(mode="json")


def _parse_candidates(text: Optional[str]) -> Optional[list[str]]:
    """Candidate document types for detection: names only (never a file path), at most
    `max_candidates()`."""
    if not text or not text.strip():
        return None
    limit = max_candidates()
    if len(text) > MAX_FIELDS_TEXT:
        raise HTTPException(422, f"candidates: longer than {MAX_FIELDS_TEXT} characters")
    text = text.strip()
    if text.startswith("["):
        try:
            items = json.loads(text)
        except (ValueError, RecursionError):
            raise HTTPException(422, "candidates: invalid JSON list")
        if not isinstance(items, list) or not all(isinstance(i, str) for i in items):
            raise HTTPException(422, "candidates: expected a list of strings")
    else:
        items = text.split(",")
    names = [i.strip() for i in items if i.strip()]
    if len(names) > limit:
        raise HTTPException(422, f"candidates: more than {limit} names")
    if any(len(n) > MAX_NAME_LENGTH for n in names):
        raise HTTPException(422, f"candidates: a name is longer than {MAX_NAME_LENGTH} characters")
    if any(_is_path(n) for n in names):
        raise HTTPException(422, "candidates must be document type names, not file paths")
    return names or None


def _read_upload(file: list[UploadFile], pages_allowed: int):
    """The uploaded parts as bytes (one source, or a list of pages), within the size limits."""
    if len(file) > pages_allowed:
        raise HTTPException(
            422, f"the request has {len(file)} files; the limit is {pages_allowed} pages (DOCOCR_MAX_PAGES)"
        )
    limit = max_upload_bytes()
    parts: list[bytes] = []
    total = 0
    for part in file:
        data = bytearray()
        while True:
            chunk = part.file.read(_CHUNK)
            if not chunk:
                break
            data += chunk
            total += len(chunk)
            if total > limit:  # the limit is for the whole request, not per file
                raise HTTPException(413, f"upload exceeds the {limit / (1024 * 1024):g} MB limit")
        if not data:
            raise HTTPException(422, "the uploaded file is empty")
        parts.append(bytes(data))
    return parts[0] if len(parts) == 1 else parts


def _run(call):
    """Run one extraction or detection in a slot, mapping errors to HTTP statuses."""
    if not _extract_slots.acquire(timeout=BUSY_WAIT_SECONDS):
        raise HTTPException(503, "the service is busy; try again shortly", headers={"Retry-After": "10"})
    try:
        return call()
    except ExtractorError as exc:
        # The engine's own message can carry library paths and internals; it goes to the
        # server log (it holds no document content), and the client gets a plain statement.
        log.error("extraction failed: %s", " ".join(str(exc).split())[:500])
        raise HTTPException(503, "the OCR engine could not process this document")
    except UnknownDocumentError as exc:
        raise HTTPException(422, {"message": str(exc), "detection": exc.detection.model_dump(mode="json")})
    except ValueError as exc:
        raise HTTPException(422, str(exc))
    except (OSError, MemoryError, Image.DecompressionBombError):
        raise HTTPException(422, "the uploaded file is not a readable image or PDF")
    finally:
        _extract_slots.release()


@app.post("/extract")
def extract_endpoint(
    file: list[UploadFile] = File(
        ..., description="An image or a PDF. Several parts are the pages of one document, in order."
    ),
    doc_type: Optional[str] = Form(None, description="Document type name; omitted or 'auto': detect it."),
    fields: Optional[str] = Form(None, description="Comma-separated field names (or a JSON list)."),
    include_raw_text: bool = Form(False),
    candidates: Optional[str] = Form(
        None, description="With detection: comma-separated type names to choose among (default: all)."),
) -> dict:
    auto = doc_type is None or not doc_type.strip() or is_auto(doc_type)
    if not auto and _is_path(doc_type):
        # load_doc_type would read a file path from the server's disk
        raise HTTPException(422, "doc_type must be a document type name, not a file path")
    if not auto and len(doc_type) > MAX_NAME_LENGTH:
        raise HTTPException(422, f"doc_type: longer than {MAX_NAME_LENGTH} characters")
    wanted = _parse_fields(fields)
    among = _parse_candidates(candidates)
    pdf_text = pdf_text_mode()
    pages_allowed = max_pages()
    source = _read_upload(file, pages_allowed)
    result = _run(lambda: pipeline.extract(
        source, "auto" if auto else doc_type, wanted, include_raw_text=include_raw_text,
        max_pages=pages_allowed, pdf_text=pdf_text, candidates=among,
    ))
    return result.model_dump(mode="json")


@app.post("/detect")
def detect_endpoint(
    file: list[UploadFile] = File(
        ..., description="An image or a PDF. Several parts are the pages of one document, in order."
    ),
    candidates: Optional[str] = Form(
        None, description="Comma-separated type names to choose among (default: all)."),
) -> dict:
    among = _parse_candidates(candidates)
    pdf_text = pdf_text_mode()
    pages_allowed = max_pages()
    source = _read_upload(file, pages_allowed)
    found = _run(lambda: pipeline.detect(source, among, max_pages=pages_allowed, pdf_text=pdf_text))
    return found.model_dump(mode="json")
