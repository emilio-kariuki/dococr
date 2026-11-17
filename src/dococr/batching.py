"""Many documents in one run: the models are loaded once per worker, not once per document.

    for item in dococr.batch(["scans/"], "auto", recursive=True, workers=2):
        if item.ok:
            print(item.source, item.result.values())
        else:
            print(item.source, item.status, item.error)

`batch` yields one `BatchItem` per document. A document that cannot be read, is none of the
candidate types, is over a limit, or kills the process that was reading it is an item with a
`status` and a short `error`; the batch goes on. Only a problem with the whole run (an unknown
type name, an input that does not exist, an OCR engine that cannot start) raises.

Workers are separate processes started with the "spawn" method, each with its own copy of the
models (about 0.5 GB resident each, more while reading a very large page). The OCR engine is
native code; a worker that dies is recorded against the document it was reading and replaced.
Nothing here logs or stores images or extracted values.
"""

from __future__ import annotations

import multiprocessing
import os
import signal
import time
from dataclasses import dataclass
from multiprocessing.connection import wait
from pathlib import Path
from typing import Any, Callable, Iterable, Iterator, Optional, Sequence, Union

from PIL import Image

from dococr import pipeline
from dococr.detection import AUTO, UnknownDocumentError, candidate_types, is_auto
from dococr.doctypes import load_doc_type, resolve_fields
from dococr.extractors.base import Extractor, ExtractorError
from dococr.imaging import DEFAULT_MAX_PAGES, IMAGE_FORMATS, SourceError
from dococr.models import Detection, DocType, ExtractionResult
from dococr.ocr.rapid import DEFAULT_THREADS

# Statuses of a document. Everything but "ok" counts as failed.
OK = "ok"
UNKNOWN_TYPE = "unknown_type"  # none of the candidate types, or the page cannot tell which
BAD_INPUT = "bad_input"  # unreadable, damaged, encrypted, or over a size or page limit
ENGINE_ERROR = "engine_error"  # the OCR engine reported that it cannot run
CRASHED = "crashed"  # the worker process died while reading this document
TIMEOUT = "timeout"  # the document took longer than `timeout` seconds; its worker was killed
ERROR = "error"  # anything else
STATUSES = (OK, UNKNOWN_TYPE, BAD_INPUT, ENGINE_ERROR, CRASHED, TIMEOUT, ERROR)

DEFAULT_TIMEOUT = 600.0  # seconds per document, with workers
_MAX_ERROR = 300  # characters of an error message kept
_STARTUP_ATTEMPTS = 3  # consecutive workers that die while loading the models before giving up

WORKERS_ENV = "DOCOCR_BATCH_WORKERS"
# Resident memory of one worker with the models loaded, measured (docs/ARCHITECTURE.md has the
# page-size limit that sets the worst case, which is much higher). See `default_workers`.
WORKER_RESIDENT_MB = 700
THREADS_ENV = "DOCOCR_OCR_THREADS"
THREADS_PER_WORKER = 2  # with several workers, unless the caller or DOCOCR_OCR_THREADS says


def document_extensions() -> frozenset[str]:
    """File extensions (lower case, with the dot) a folder scan picks up: the image formats
    `dococr.imaging` decodes, and .pdf."""
    exts = {ext for ext, fmt in Image.registered_extensions().items() if fmt in IMAGE_FORMATS}
    return frozenset(exts | {".pdf"})


def find_documents(
    inputs: Iterable[Union[str, Path]], *, recursive: bool = False
) -> list[Path]:
    """The files to read, as absolute paths in a deterministic order.

    A file named in `inputs` is taken as it is, whatever its extension. A folder contributes the
    files directly inside it whose extension `document_extensions` lists, sorted by name; with
    `recursive` it also contributes its subfolders, depth first, entries sorted by name. Names
    starting with a dot (files and folders) are skipped, and so are folders reached through a
    symbolic link and anything that is not a regular file. (A symbolic link to a folder that is
    itself named in `inputs` is followed: it was asked for.) A file reached twice is listed
    once, at its first place. Raises FileNotFoundError for an input that does not exist.
    """
    exts = document_extensions()
    found: list[Path] = []
    seen: set[str] = set()

    def add(path: str) -> None:
        if path not in seen:
            seen.add(path)
            found.append(Path(path))

    def walk(folder: str) -> None:
        try:
            with os.scandir(folder) as it:
                entries = sorted(it, key=lambda e: e.name)
        except OSError:
            return  # an unreadable folder contributes nothing
        for entry in entries:
            if entry.name.startswith("."):
                continue
            try:
                if entry.is_dir(follow_symlinks=False):
                    if recursive:
                        walk(entry.path)
                elif entry.is_file() and os.path.splitext(entry.name)[1].lower() in exts:
                    add(entry.path)
            except OSError:
                continue

    for item in inputs:
        path = os.path.abspath(os.fspath(item))
        if os.path.isdir(path):
            walk(path)
        elif os.path.isfile(path):
            add(path)
        else:
            raise FileNotFoundError(f"no such file or folder: {item}")
    return found


@dataclass
class BatchItem:
    """The outcome for one document."""

    index: int  # position among the documents of this run, from 0
    source: Path  # absolute path
    status: str  # one of STATUSES
    error: str = ""  # short, one line; empty when status is "ok"
    result: Optional[ExtractionResult] = None  # set when status is "ok"
    # Set when status is "unknown_type": the ranking that found no type.
    detection: Optional[Detection] = None
    elapsed_ms: int = 0  # reading this document, in the worker
    size: int = 0  # bytes, when the batch started reading it
    mtime_ns: int = 0

    @property
    def ok(self) -> bool:
        return self.status == OK


@dataclass
class _Options:
    """What `pipeline.extract` is called with for every document. Picklable."""

    doc_type: Any = AUTO
    fields: Optional[list] = None
    include_raw_text: bool = False
    max_pages: int = DEFAULT_MAX_PAGES
    pdf_text: str = "auto"
    candidates: Optional[list] = None
    dirs: Optional[list] = None


_Outcome = tuple  # (status, error, result, detection, elapsed_ms)


def _short(text: object) -> str:
    one = " ".join(str(text).split())
    return one if len(one) <= _MAX_ERROR else one[: _MAX_ERROR - 3] + "..."


def _read_one(path: Path, opts: _Options, extractor: Optional[Extractor]) -> _Outcome:
    """Read one document; never raises (except for KeyboardInterrupt and the like)."""
    started = time.perf_counter()

    def done(status: str, error: str = "", result=None, detection=None) -> _Outcome:
        return (status, _short(error), result, detection, int((time.perf_counter() - started) * 1000))

    try:
        result = pipeline.extract(
            path, opts.doc_type, opts.fields, extractor=extractor,
            include_raw_text=opts.include_raw_text, max_pages=opts.max_pages,
            pdf_text=opts.pdf_text, candidates=opts.candidates, dirs=opts.dirs,
        )
        return done(OK, result=result)
    except UnknownDocumentError as exc:
        return done(UNKNOWN_TYPE, str(exc), detection=exc.detection)
    except ExtractorError as exc:
        return done(ENGINE_ERROR, str(exc))
    except SourceError as exc:
        return done(BAD_INPUT, str(exc))
    except FileNotFoundError:
        return done(BAD_INPUT, "the file no longer exists")
    except Image.UnidentifiedImageError:
        return done(BAD_INPUT, "not a readable image or PDF")
    except Image.DecompressionBombError:
        return done(BAD_INPUT, "the image is too large to decode")
    except PermissionError:
        return done(BAD_INPUT, "permission denied reading the file")
    except ValueError as exc:  # a limit, or a request the pipeline refuses
        return done(BAD_INPUT, str(exc))
    except OSError as exc:
        return done(BAD_INPUT, f"cannot read the file: {exc.strerror or type(exc).__name__}")
    except MemoryError:
        return done(ERROR, "MemoryError: the document needed more memory than is available")
    except Exception as exc:
        return done(ERROR, f"{type(exc).__name__}: {exc}")


# ------------------------------------------------------------------------------ parameters

def _cpus() -> int:
    try:
        return max(1, len(os.sched_getaffinity(0)))
    except AttributeError:  # not Linux
        return max(1, os.cpu_count() or 1)


def _available_mb() -> Optional[int]:
    """Memory that can be used without swapping, from /proc/meminfo (Linux); None elsewhere."""
    try:
        with open("/proc/meminfo", encoding="ascii") as handle:
            for line in handle:
                if line.startswith("MemAvailable:"):
                    return int(line.split()[1]) // 1024
    except (OSError, ValueError, IndexError):
        pass
    return None


def default_workers() -> int:
    """How many worker processes to use when the caller does not say.

    DOCOCR_BATCH_WORKERS overrides it. Otherwise 2, fewer on a machine with fewer than 4 CPUs
    or with too little free memory for a second worker: each holds its own copy of the models
    (`WORKER_RESIDENT_MB`) and a page at the size limit can take 1.6 GB more.
    """
    wanted = os.environ.get(WORKERS_ENV, "")
    if wanted.isdigit() and int(wanted) >= 0:
        return int(wanted)
    n = 2 if _cpus() >= 4 else 1
    free = _available_mb()
    if free is not None and free < 2 * (WORKER_RESIDENT_MB + 1600):
        n = 1
    return n


def result_fields(
    doc_type: Union[str, Path, DocType, None] = AUTO,
    fields: Optional[Sequence[str]] = None,
    *,
    candidates: Optional[Sequence[Union[str, Path, DocType]]] = None,
    dirs: Optional[Iterable[Union[str, Path]]] = None,
) -> list[str]:
    """The field names the results of a batch can have, known before any document is read:
    those of the named type (or the requested `fields`); with automatic detection, the union
    over the candidate types, in order of first appearance."""
    if is_auto(doc_type):
        docs, _notes = candidate_types(candidates, dirs)
    else:
        docs = [load_doc_type(doc_type, dirs)]
    names: list[str] = []
    for doc in docs:
        for spec in resolve_fields(doc, list(fields) if fields else None):
            if spec.name not in names:
                names.append(spec.name)
    return names


# ------------------------------------------------------------------------------ workers

def _worker_main(conn, factory, ocr_threads: Optional[int], opts: _Options) -> None:
    """The body of a worker process: load the models once, then read what it is sent."""
    signal.signal(signal.SIGINT, signal.SIG_IGN)  # Ctrl-C belongs to the parent
    if ocr_threads:
        os.environ["DOCOCR_OCR_THREADS"] = str(ocr_threads)
    try:
        if factory is not None:
            extractor = factory()
            warm = getattr(extractor, "warm_up", None)
            if callable(warm):
                warm()
        else:
            extractor = pipeline.warm_up()
    except BaseException as exc:  # the engine cannot start; the parent stops the batch
        conn.send(("fatal", _short(f"the OCR engine cannot start: {exc}")))
        return
    conn.send(("ready",))
    while True:
        try:
            job = conn.recv()
        except (EOFError, OSError):
            return
        if job is None:
            return
        index, path = job
        outcome = _read_one(path, opts, extractor)
        try:
            conn.send(("done", index, outcome))
        except Exception as exc:  # a result that cannot be sent
            conn.send(("done", index, (ERROR, _short(f"result could not be returned: {exc}"),
                                       None, None, outcome[4])))


@dataclass(eq=False)
class _Slot:
    proc: Any
    conn: Any
    ready: bool = False
    job: Optional[int] = None
    started: float = 0.0


def _describe_exit(code: Optional[int]) -> str:
    if code is None:
        return "the worker process died"
    if code < 0:
        try:
            return f"the worker process was killed by {signal.Signals(-code).name}"
        except ValueError:
            return f"the worker process was killed by signal {-code}"
    return f"the worker process exited with code {code}"


def _run_workers(
    jobs: list, count: int, opts: _Options, factory, ocr_threads: Optional[int],
    timeout: Optional[float], ordered: bool,
) -> Iterator[BatchItem]:
    ctx = multiprocessing.get_context("spawn")
    window = max(8, 4 * count)
    slots: list[_Slot] = []
    finished: dict[int, BatchItem] = {}
    state = {"next_job": 0, "emitted": 0, "startup_failures": 0}

    def spawn() -> _Slot:
        parent, child = ctx.Pipe()
        proc = ctx.Process(target=_worker_main, args=(child, factory, ocr_threads, opts), daemon=True)
        proc.start()
        child.close()
        return _Slot(proc, parent)

    def complete(index: int, outcome: _Outcome) -> None:
        status, error, result, detection, elapsed = outcome
        item = jobs[index]
        item.status, item.error, item.result, item.detection, item.elapsed_ms = (
            status, error, result, detection, elapsed)
        finished[index] = item

    def handle(slot: _Slot, message: tuple) -> None:
        kind = message[0]
        if kind == "ready":
            slot.ready = True
            state["startup_failures"] = 0
        elif kind == "fatal":
            raise ExtractorError(message[1])
        elif kind == "done":
            slot.job = None
            complete(message[1], message[2])

    def drain(slot: _Slot) -> bool:
        """Read what the worker has sent. False when its pipe is closed."""
        try:
            while slot.conn.poll():
                handle(slot, slot.conn.recv())
        except (EOFError, OSError):
            return False
        return True

    def retire(slot: _Slot, status: str, error: str) -> None:
        """A worker is gone (or is being killed): record its document, and replace it."""
        if slot.job is not None:
            complete(slot.job, (status, error, None, None, int((time.monotonic() - slot.started) * 1000)))
        elif not slot.ready:
            state["startup_failures"] += 1
            if state["startup_failures"] >= _STARTUP_ATTEMPTS:
                raise ExtractorError(
                    f"worker processes keep dying while loading the models ({error})")
        slot.job = None
        try:
            slot.conn.close()
        except OSError:
            pass
        idx = slots.index(slot)
        if state["next_job"] < len(jobs) or any(s.job is not None for s in slots):
            slots[idx] = spawn()
        else:
            slots.pop(idx)

    try:
        slots.extend(spawn() for _ in range(count))
        while state["emitted"] < len(jobs):
            for slot in slots:
                limit = state["emitted"] + window if ordered else len(jobs)
                if (slot.ready and slot.job is None and state["next_job"] < len(jobs)
                        and state["next_job"] < limit):
                    index = state["next_job"]
                    state["next_job"] += 1
                    slot.job, slot.started = index, time.monotonic()
                    try:
                        slot.conn.send((index, jobs[index].source))
                    except (OSError, ValueError):  # the worker is gone: the job was not started
                        slot.job = None
                        state["next_job"] = index
            wait([s.conn for s in slots] + [s.proc.sentinel for s in slots], timeout=0.5)
            now = time.monotonic()
            for slot in list(slots):
                alive_pipe = drain(slot)
                if not slot.proc.is_alive() or not alive_pipe:
                    drain(slot)
                    slot.proc.join(timeout=1)
                    if slot.proc.is_alive():
                        slot.proc.kill()
                        slot.proc.join(timeout=1)
                    retire(slot, CRASHED, _describe_exit(slot.proc.exitcode))
                elif timeout and slot.job is not None and now - slot.started > timeout:
                    slot.proc.kill()
                    slot.proc.join(timeout=5)
                    retire(slot, TIMEOUT, f"no result after {timeout:g} seconds; the worker was stopped")
            if ordered:
                while state["emitted"] in finished:
                    index = state["emitted"]
                    yield finished.pop(index)
                    jobs[index] = None  # let go of the result once it has been handed over
                    state["emitted"] += 1
            else:
                for index in sorted(finished):
                    yield finished.pop(index)
                    jobs[index] = None
                    state["emitted"] += 1
    finally:
        for slot in slots:
            try:
                slot.conn.send(None)
            except (OSError, ValueError):
                pass
        deadline = time.monotonic() + 5
        for slot in slots:
            slot.proc.join(timeout=max(0.0, deadline - time.monotonic()))
            if slot.proc.is_alive():
                slot.proc.kill()
                slot.proc.join(timeout=2)
            try:
                slot.conn.close()
            except OSError:
                pass


def _run_here(jobs: list, opts: _Options, factory) -> Iterator[BatchItem]:
    extractor = factory() if factory is not None else pipeline.warm_up()
    for index, item in enumerate(jobs):
        outcome = _read_one(item.source, opts, extractor if factory is not None else None)
        item.status, item.error, item.result, item.detection, item.elapsed_ms = outcome
        jobs[index] = None  # the consumer holds the item; the list must not
        yield item


def batch(
    sources: Iterable[Union[str, Path]],
    doc_type: Union[str, Path, DocType, None] = AUTO,
    fields: Optional[list[str]] = None,
    *,
    recursive: bool = False,
    workers: Optional[int] = None,
    ocr_threads: Optional[int] = None,
    timeout: Optional[float] = DEFAULT_TIMEOUT,
    ordered: bool = True,
    include_raw_text: bool = False,
    max_pages: int = DEFAULT_MAX_PAGES,
    pdf_text: str = "auto",
    candidates: Optional[Sequence[Union[str, Path, DocType]]] = None,
    dirs: Optional[Iterable[Union[str, Path]]] = None,
    extractor_factory: Optional[Callable[[], Extractor]] = None,
) -> Iterator[BatchItem]:
    """Read many documents with the models loaded once per worker; yield one `BatchItem` each.

    `sources` are files and folders (see `find_documents`; `recursive` goes into subfolders).
    Each file is one document, as `extract` reads it (a PDF or multi-page TIFF is one document).
    `doc_type`, `fields`, `candidates`, `dirs`, `pdf_text`, `max_pages` and `include_raw_text`
    are those of `dococr.extract`.

    `workers` is the number of worker processes (default `default_workers()`); each loads the
    models once and reads one document at a time. 0 reads in this process instead: no extra
    memory and nothing to start, but a crash of the OCR engine then ends the program and
    `timeout` does not apply. `ocr_threads` sets DOCOCR_OCR_THREADS in each worker; by default
    each worker uses 2 when there are several (a single process keeps its default of 4). `timeout` is
    the most seconds a worker may spend on one document (None: no limit); a document that takes
    longer is recorded as "timeout" and its worker replaced.

    Items come in input order (`ordered`, default), so a slow document holds back the ones after
    it and at most `max(8, 4 * workers)` documents are read ahead of the one being waited for.
    With `ordered=False` they come as they finish, and each item's `index` says where it was.

    `extractor_factory` builds the extractor in each worker (default: the OCR engine). With
    workers it must be picklable, such as a module-level function or class.

    Raises, when called, a ValueError (including DocTypeError) for a type, field or option that
    cannot be used, and FileNotFoundError for a source that does not exist. While iterating it
    raises ExtractorError if the OCR engine cannot start. It never raises for a bad document.
    """
    if workers is None:
        workers = default_workers()
    if workers < 0:
        raise ValueError("workers must be 0 or more")
    if max_pages < 1:
        raise ValueError("max_pages must be at least 1")
    if ocr_threads is not None and ocr_threads < 1:
        raise ValueError("ocr_threads must be at least 1")
    pipeline._check_pdf_text(pdf_text)
    result_fields(doc_type, fields, candidates=candidates, dirs=dirs)  # fail early on a bad type
    paths = find_documents(sources, recursive=recursive)
    jobs: list[BatchItem] = []
    for path in paths:
        try:
            st = path.stat()
        except OSError as exc:
            raise FileNotFoundError(f"cannot read {path}: {exc.strerror}") from exc
        jobs.append(BatchItem(len(jobs), path, "", size=st.st_size, mtime_ns=st.st_mtime_ns))
    opts = _Options(
        doc_type=doc_type if doc_type is not None else AUTO, fields=list(fields) if fields else None,
        include_raw_text=include_raw_text, max_pages=max_pages, pdf_text=pdf_text,
        candidates=list(candidates) if candidates else None,
        dirs=[Path(d) for d in dirs] if dirs else None,
    )
    if not jobs:
        return iter(())
    if ocr_threads is None and workers > 1 and not os.environ.get(THREADS_ENV, "").isdigit():
        # More threads per process was slower in every measurement with several processes.
        ocr_threads = min(DEFAULT_THREADS, THREADS_PER_WORKER)
    if workers == 0:
        return _run_here(jobs, opts, extractor_factory)
    return _run_workers(jobs, min(workers, len(jobs)), opts, extractor_factory, ocr_threads,
                        timeout or None, ordered)
