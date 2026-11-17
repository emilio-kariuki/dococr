"""Writing the results of a batch: JSON Lines, CSV and JSON files, resuming, and progress.

Every record is the same dictionary (`record`); the formats differ in how it is laid out.

    JSON Lines  one record per line, appended and flushed as each document finishes
    CSV         one row per document: fixed columns, then three columns per field
    JSON        one array of records, written at the end through a temporary file

Extracted values are written as they are, except in CSV, where a text cell that a spreadsheet
would run as a formula is neutralised (see `neutralise`). Nothing here logs.
"""

from __future__ import annotations

import csv
import io
import json
import os
import sys
import tempfile
import time
from pathlib import Path
from typing import Any, IO, Iterable, Optional

from dococr.batching import STATUSES, BatchItem

FORMATS = ("csv", "jsonl", "json")
_EXTENSIONS = {".csv": "csv", ".jsonl": "jsonl", ".ndjson": "jsonl", ".json": "json"}

# Fixed CSV columns, before the per-field ones. `size_bytes` and `mtime_ns` identify the file
# version for `--resume`.
FIXED_COLUMNS = ["file", "status", "error", "doc_type", "detection_score", "pages",
                 "missing_fields", "elapsed_ms", "size_bytes", "mtime_ns"]
# A text cell starting with one of these is run as a formula by spreadsheet programs.
FORMULA_STARTS = ("=", "+", "-", "@", "\t", "\r")


class OutputError(ValueError):
    """The output file cannot be used as asked (exists, wrong format, cannot be resumed)."""


def infer_format(path: Path, explicit: Optional[str] = None) -> str:
    if explicit:
        if explicit not in FORMATS:
            raise OutputError(f"unknown format '{explicit}'; use one of {', '.join(FORMATS)}")
        return explicit
    fmt = _EXTENSIONS.get(path.suffix.lower())
    if fmt is None:
        raise OutputError(f"cannot tell the format from '{path.name}'; name it .csv, .jsonl or "
                          ".json, or give --format")
    return fmt


# ------------------------------------------------------------------------------ records

def record(item: BatchItem) -> dict:
    """One document as a dictionary: the line of a JSON Lines file."""
    out: dict[str, Any] = {
        "file": str(item.source),
        "status": item.status,
        "error": item.error,
        "elapsed_ms": item.elapsed_ms,
        "size_bytes": item.size,
        "mtime_ns": item.mtime_ns,
        "result": item.result.model_dump(mode="json") if item.result is not None else None,
    }
    if item.detection is not None:
        out["detection"] = item.detection.model_dump(mode="json")
    return out


def neutralise(text: str) -> str:
    """Make a text cell safe to open in a spreadsheet: a leading apostrophe when it starts with
    `=`, `+`, `-`, `@`, a tab or a carriage return. The spreadsheet shows the text without the
    apostrophe, and a program reading the CSV sees it and can strip it."""
    return "'" + text if text.startswith(FORMULA_STARTS) else text


def _cell(value: Any) -> Any:
    """A CSV cell. Numbers stay numbers; any text is neutralised."""
    if value is None:
        return ""
    if isinstance(value, bool):
        return str(value)
    if isinstance(value, (int, float)):
        return value
    return neutralise(str(value))


def csv_header(field_names: Iterable[str]) -> list[str]:
    """Fixed columns, then `<field>`, `<field>_confidence`, `<field>_warnings` for each field.
    A column that would repeat another gets `_field` appended until it is unique."""
    used = set(FIXED_COLUMNS)
    header = list(FIXED_COLUMNS)

    def take(name: str) -> str:
        while name in used:
            name += "_field"
        used.add(name)
        return name

    for name in field_names:
        header.append(take(name))
        header.append(take(f"{name}_confidence"))
        header.append(take(f"{name}_warnings"))
    return header


def csv_row(rec: dict, field_names: list[str]) -> list:
    """The CSV cells of a record. A value that runs across pages is one cell (its text); the
    pages it came from are only in the JSON forms. Warnings of a field are joined with ' | ',
    names of missing fields with ';'."""
    result = rec.get("result") or {}
    detection = result.get("detection") or rec.get("detection") or {}
    row: list[Any] = [
        _cell(rec["file"]), rec["status"], _cell(rec["error"]),
        _cell(result.get("doc_type", "")),
        detection.get("score", ""),
        result.get("pages", ""),
        _cell(";".join(result.get("missing", []))),
        rec["elapsed_ms"], rec["size_bytes"], rec["mtime_ns"],
    ]
    fields = result.get("fields", {})
    for name in field_names:
        f = fields.get(name)
        if f is None:
            row += ["", "", ""]
            continue
        conf = f.get("confidence")
        row += [_cell(f.get("value")), "" if conf is None else round(conf, 4),
                _cell(" | ".join(f.get("warnings", [])))]
    return row


# ------------------------------------------------------------------------------ resume

def _complete_prefix(data: bytes, fmt: str) -> int:
    """Length of the part of `data` made of whole records. A run that was interrupted can leave
    half a line (JSON Lines) or half a row (CSV, where a quoted value may contain line breaks)
    at the end."""
    if fmt == "jsonl":
        return data.rfind(b"\n") + 1
    inside, end = False, 0
    for i, byte in enumerate(data):
        if byte == 0x22:  # a doubled quote inside a value toggles twice
            inside = not inside
        elif byte == 0x0A and not inside:
            end = i + 1
    return end


def read_done(path: Path, fmt: str, header: Optional[list[str]]) -> tuple[set, bool]:
    """(the (file, size, mtime) of every document already recorded in `path`, whether an
    incomplete last record was cut off). Raises OutputError if the file is not a file of this
    format, or its columns are not `header`'s (CSV)."""
    if fmt == "json":
        raise OutputError("a .json output is one array written at the end and cannot be resumed; "
                          "use .jsonl or .csv")
    data = path.read_bytes()
    if not data:
        return set(), False
    keep = _complete_prefix(data, fmt)
    done: set = set()
    text = data[:keep].decode("utf-8", errors="replace")
    if fmt == "jsonl":
        for number, line in enumerate(text.splitlines(), start=1):
            if not line.strip():
                continue
            try:
                rec = json.loads(line)
                done.add((rec["file"], int(rec["size_bytes"]), int(rec["mtime_ns"])))
            except (ValueError, KeyError, TypeError):
                raise OutputError(f"{path}: line {number} is not a record of this tool; "
                                  "cannot resume into it") from None
    else:
        rows = list(csv.reader(io.StringIO(text, newline="")))
        if not rows or rows[0] != header:
            raise OutputError(f"{path}: its columns differ from this run's (another type, "
                              "fields or candidates?); cannot resume into it")
        at = {name: i for i, name in enumerate(rows[0])}
        for n, row in enumerate(rows[1:], start=2):
            try:
                key = (row[at["file"]], int(row[at["size_bytes"]]), int(row[at["mtime_ns"]]))
            except (ValueError, IndexError):
                raise OutputError(f"{path}: row {n} is not a record of this tool; "
                                  "cannot resume into it") from None
            done.add(key)
    return done, keep < len(data)


def key_of(item: BatchItem) -> tuple:
    return (str(item.source), item.size, item.mtime_ns)


# ------------------------------------------------------------------------------ writers

class Writer:
    """Receives records as documents finish. `close(complete)` is told whether the batch ran to
    the end; only the JSON writer cares."""

    def __init__(self, path: Path, *, overwrite: bool = False, resume: bool = False) -> None:
        self.path = path
        self.overwrite = overwrite
        self.resume = resume
        self.skipped_partial = False

    def write(self, rec: dict) -> None:
        raise NotImplementedError

    def close(self, complete: bool = True) -> None:
        raise NotImplementedError


def _open_stream(path: Path, overwrite: bool, resume: bool, cut: Optional[int] = None) -> IO[bytes]:
    if resume and path.exists():
        handle = open(path, "r+b")
        if cut is not None:
            handle.truncate(cut)
        handle.seek(0, os.SEEK_END)
        return handle
    # "xb" refuses to overwrite a file that appeared since the check; "wb" is --overwrite.
    return open(path, "wb" if overwrite else "xb")


class JsonlWriter(Writer):
    def __init__(self, path, *, overwrite=False, resume=False, cut=None):
        super().__init__(path, overwrite=overwrite, resume=resume)
        self._out = _open_stream(path, overwrite, resume, cut)

    def write(self, rec: dict) -> None:
        self._out.write((json.dumps(rec, ensure_ascii=False) + "\n").encode("utf-8"))
        self._out.flush()

    def close(self, complete: bool = True) -> None:
        self._out.close()


class CsvWriter(Writer):
    def __init__(self, path, field_names: list[str], *, overwrite=False, resume=False, cut=None):
        super().__init__(path, overwrite=overwrite, resume=resume)
        self.field_names = list(field_names)
        appending = resume and path.exists() and path.stat().st_size > 0
        self._out = _open_stream(path, overwrite, resume, cut)
        if not appending:
            self._emit(csv_header(self.field_names))

    def _emit(self, row: list) -> None:
        buffer = io.StringIO(newline="")
        csv.writer(buffer).writerow(row)  # one write() per row, so an interruption cuts at most one
        self._out.write(buffer.getvalue().encode("utf-8"))
        self._out.flush()

    def write(self, rec: dict) -> None:
        self._emit(csv_row(rec, self.field_names))

    def close(self, complete: bool = True) -> None:
        self._out.close()


class JsonWriter(Writer):
    """Collects the records and writes one array at the end: to a temporary file in the output's
    folder, flushed to disk, then renamed over the target (`--overwrite`) or linked to it (which
    fails if the file appeared meanwhile). An interrupted run writes nothing."""

    def __init__(self, path, *, overwrite=False, resume=False):
        super().__init__(path, overwrite=overwrite, resume=resume)
        self._records: list[dict] = []

    def write(self, rec: dict) -> None:
        self._records.append(rec)

    def close(self, complete: bool = True) -> None:
        if not complete:
            return
        folder = self.path.parent
        fd, tmp_name = tempfile.mkstemp(prefix=f".{self.path.name}.", suffix=".tmp", dir=folder)
        tmp = Path(tmp_name)
        try:
            with os.fdopen(fd, "w", encoding="utf-8") as handle:
                json.dump(self._records, handle, ensure_ascii=False, indent=2)
                handle.write("\n")
                handle.flush()
                os.fsync(handle.fileno())
            if self.overwrite:
                os.replace(tmp, self.path)
            else:
                try:
                    os.link(tmp, self.path)  # atomic, and refuses an existing file
                except FileExistsError:
                    raise OutputError(f"{self.path} appeared while the batch ran; not "
                                      "overwritten (use --overwrite)") from None
                except OSError:  # a file system without hard links
                    if self.path.exists():
                        raise OutputError(f"{self.path} appeared while the batch ran; not "
                                          "overwritten (use --overwrite)") from None
                    os.replace(tmp, self.path)
        finally:
            try:
                tmp.unlink()
            except FileNotFoundError:
                pass


def open_writer(
    path: Path, fmt: str, field_names: list[str], *, overwrite: bool = False, resume: bool = False,
) -> tuple[Writer, set]:
    """Check the output path and open a writer. Returns it and the keys of the documents already
    recorded (empty unless `resume` found a file). Raises OutputError."""
    if overwrite and resume:
        raise OutputError("--overwrite and --resume cannot be used together")
    if not path.parent.is_dir():
        raise OutputError(f"the output folder does not exist: {path.parent}")
    if path.is_dir():
        raise OutputError(f"{path} is a folder")
    done: set = set()
    cut: Optional[int] = None
    if path.exists():
        if resume:
            header = csv_header(field_names) if fmt == "csv" else None
            done, partial = read_done(path, fmt, header)
            if partial:
                cut = _complete_prefix(path.read_bytes(), fmt)
        elif not overwrite:
            raise OutputError(f"{path} already exists; use --overwrite to replace it, or "
                              "--resume to add to it")
    elif resume and fmt == "json":
        raise OutputError("a .json output cannot be resumed; use .jsonl or .csv")
    if fmt == "jsonl":
        writer: Writer = JsonlWriter(path, overwrite=overwrite, resume=resume, cut=cut)
    elif fmt == "csv":
        writer = CsvWriter(path, field_names, overwrite=overwrite, resume=resume, cut=cut)
    else:
        if resume:
            raise OutputError("a .json output cannot be resumed; use .jsonl or .csv")
        writer = JsonWriter(path, overwrite=overwrite)
    writer.skipped_partial = cut is not None
    return writer, done


# ------------------------------------------------------------------------------ progress

def _duration(seconds: float) -> str:
    seconds = int(round(seconds))
    if seconds < 60:
        return f"{seconds}s"
    minutes, sec = divmod(seconds, 60)
    if minutes < 60:
        return f"{minutes}m{sec:02d}s"
    hours, minutes = divmod(minutes, 60)
    return f"{hours}h{minutes:02d}m"


class Progress:
    """The progress line on a stream (stderr). Counts and times only: no file contents, no
    extracted values. On a terminal it rewrites one line; otherwise it prints a line at most
    every `every` seconds. A failed document always gets its own line, also when `enabled` is
    False (that switches the rolling line off)."""

    def __init__(self, total: int, stream=None, *, every: float = 5.0, enabled: bool = True):
        self.total = total
        self.stream = stream if stream is not None else sys.stderr
        self.every = every
        self.enabled = enabled
        self.done = 0
        self.failed = 0
        self.by_status: dict[str, int] = {}
        self.started = time.monotonic()
        self._last = 0.0
        self._tty = bool(getattr(self.stream, "isatty", lambda: False)())

    def line(self) -> str:
        elapsed = max(time.monotonic() - self.started, 1e-9)
        rate = self.done / elapsed
        eta = "-" if not self.done or not rate else _duration((self.total - self.done) / rate)
        return (f"[{self.done}/{self.total}] failed {self.failed}  "
                f"{rate:.2f} docs/s  ETA {eta}")

    def _write(self, text: str, newline: bool, force: bool = False) -> None:
        if not (self.enabled or force):
            return
        if self._tty:
            self.stream.write("\r\x1b[2K" + text + ("\n" if newline else ""))
        else:
            self.stream.write(text + "\n")
        self.stream.flush()

    def update(self, item: BatchItem) -> None:
        self.done += 1
        self.by_status[item.status] = self.by_status.get(item.status, 0) + 1
        if not item.ok:
            self.failed += 1
            self._write(f"failed: {item.source}: {item.status}: {item.error}", newline=True, force=True)
        now = time.monotonic()
        if self._tty or now - self._last >= self.every or self.done == self.total:
            self._last = now
            self._write(self.line(), newline=False)

    def finish(self) -> None:
        if self.enabled and self._tty and self.done:
            self.stream.write("\n")
            self.stream.flush()

    def summary(self, *, skipped: int, output: Path, fmt: str, interrupted: bool = False) -> str:
        elapsed = time.monotonic() - self.started
        rate = self.done / elapsed if elapsed > 0 else 0.0
        counts = ", ".join(f"{self.by_status[s]} {s}" for s in STATUSES if s in self.by_status)
        head = (f"interrupted: {self.done} of {self.total} documents" if interrupted
                else f"finished: {self.done} documents")
        lines = [f"{head} in {_duration(elapsed)} ({rate:.2f} docs/s)",
                 f"  {counts or 'nothing read'}"]
        if skipped:
            lines.append(f"  {skipped} already recorded in the output, not read again")
        lines.append(f"  output: {output} ({fmt})")
        return "\n".join(lines)
