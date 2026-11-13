"""Document-type lookup. Every document type is a YAML file; this module holds no document names.

Where types are searched, in this order:

1. folders passed as `dirs`,
2. folders listed in the DOCOCR_DOCTYPES_DIR environment variable (separated by `os.pathsep`,
   read at call time),
3. the folder shipped with the package.

A user's file therefore overrides a shipped one of the same name.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass
from importlib import resources
from pathlib import Path
from typing import Any, Iterable, Optional, Union

import yaml
from pydantic import ValidationError

from dococr.models import DocType, FieldSpec, slugify

DOCTYPES_DIR_ENV = "DOCOCR_DOCTYPES_DIR"
_SUFFIXES = (".yaml", ".yml")


class DocTypeError(ValueError):
    """A document-type file or folder cannot be used. The message names it and says why."""


@dataclass
class _Entry:
    path: Any  # pathlib.Path or importlib.resources Traversable
    names: set  # slugified `name` and file name without extension
    aliases: set  # slugified `aliases`
    doc: Optional[DocType] = None
    error: str = ""


def _problem(exc: Exception) -> str:
    if isinstance(exc, ValidationError):
        first = exc.errors()[0]
        where = ".".join(str(p) for p in first["loc"])
        return f"{where}: {first['msg']}" if where else first["msg"]
    if isinstance(exc, yaml.YAMLError):
        mark = getattr(exc, "problem_mark", None)
        line = f" (line {mark.line + 1})" if mark is not None else ""
        return f"not valid YAML{line}: {getattr(exc, 'problem', None) or 'syntax error'}"
    return str(exc) or type(exc).__name__


# The C loader is several times faster. Both are safe loaders (plain data only, no Python objects).
_SafeLoader = getattr(yaml, "CSafeLoader", yaml.SafeLoader)


def _read(path: Any) -> Any:
    return yaml.load(path.read_text(encoding="utf-8"), Loader=_SafeLoader)


def _entry(path: Any) -> _Entry:
    stem = path.name.rsplit(".", 1)[0]
    entry = _Entry(path=path, names={slugify(stem)}, aliases=set())
    try:
        data = _read(path)
        # Keys come from the raw data, so a file that fails validation can still be recognised.
        if isinstance(data, dict):
            if isinstance(data.get("name"), str):
                entry.names.add(slugify(data["name"]))
            if isinstance(data.get("aliases"), list):
                entry.aliases = {slugify(a) for a in data["aliases"] if isinstance(a, str)}
        if not isinstance(data, dict):
            raise ValueError("expected a mapping with at least a 'name'")
        entry.doc = DocType.model_validate(data)
        entry.names.add(entry.doc.name)
        entry.aliases = {slugify(a) for a in entry.doc.aliases}
    except Exception as exc:  # unreadable, not YAML, or does not validate
        entry.error = _problem(exc)
    entry.names.discard("")
    entry.aliases.discard("")
    return entry


def _folders(dirs: Optional[Iterable[Union[str, Path]]]) -> list:
    """Search order: explicit dirs, then the environment variable's, then the shipped folder."""
    given = [Path(d) for d in (dirs or [])]
    given += [Path(p) for p in os.environ.get(DOCTYPES_DIR_ENV, "").split(os.pathsep) if p.strip()]
    for folder in given:
        if not folder.is_dir():
            raise DocTypeError(f"{folder}: not a folder (document types are searched there)")
    return [*given, resources.files("dococr") / "presets"]


# Parsed entries per folder, kept while the folder's listing is unchanged. Each call re-checks the
# folder (file names, modification times, sizes, inodes), so an edit, new file or deletion is seen
# at once; nothing is trusted for a period of time. Entries are private: callers get copies.
_cache: dict = {}
_cache_lock = threading.Lock()


def _signature(files: list) -> tuple:
    sig = []
    for p in files:
        try:
            st = p.stat()
            sig.append((p.name, st.st_mtime_ns, st.st_size, st.st_ino))
        except (OSError, AttributeError):  # vanished meanwhile, or a package-resource path without stat
            sig.append((p.name,))
    return tuple(sig)


def _entries(folder: Any) -> list[_Entry]:
    files = sorted((p for p in folder.iterdir() if p.name.lower().endswith(_SUFFIXES)), key=lambda p: p.name)
    sig = _signature(files)
    key = (str(folder), os.getcwd())  # a relative folder means something else after a chdir
    with _cache_lock:
        hit = _cache.get(key)
        if hit is not None and hit[0] == sig:
            return hit[1]
        entries = [_entry(p) for p in files]
        _cache[key] = (sig, entries)
        return entries


def _copy(doc: DocType) -> DocType:
    return doc.model_copy(deep=True)


def find_doc_types(dirs: Optional[Iterable[Union[str, Path]]] = None) -> tuple[list[DocType], list[str]]:
    """Every type found in the search folders, and a message for each file that was skipped.

    A type that appears in several places is listed once, from the first place searched
    (see the module docstring). Within one folder files are taken in order of file name.
    Skipped files are those that cannot be read or do not validate; they never stop the scan.
    """
    found: dict[str, DocType] = {}
    skipped: list[str] = []
    for folder in _folders(dirs):
        for entry in _entries(folder):
            if entry.doc is None:
                skipped.append(f"{entry.path}: {entry.error}")
            else:
                found.setdefault(entry.doc.name, _copy(entry.doc))
    return sorted(found.values(), key=lambda d: d.name), skipped


def list_doc_types(dirs: Optional[Iterable[Union[str, Path]]] = None) -> list[str]:
    """Names of every type found in the search folders, sorted, without duplicates."""
    return [d.name for d in find_doc_types(dirs)[0]]


def load_doc_type_file(path: Union[str, Path]) -> DocType:
    try:
        data = _read(Path(path))
        return DocType.model_validate(data)
    except Exception as exc:
        raise DocTypeError(f"{path}: {_problem(exc)}") from exc


def _find(key: str, dirs: Optional[Iterable[Union[str, Path]]]) -> Optional[DocType]:
    for folder in _folders(dirs):
        entries = _entries(folder)
        for attr in ("names", "aliases"):  # a name beats an alias within one folder
            for entry in entries:
                if key in getattr(entry, attr):
                    if entry.doc is None:
                        raise DocTypeError(f"{entry.path}: {entry.error}")
                    return _copy(entry.doc)
    return None


def load_doc_type(
    name_or_path: Union[str, Path, DocType], dirs: Optional[Iterable[Union[str, Path]]] = None
) -> DocType:
    """Resolve a doc type.

    Accepts a DocType, a path to a YAML file, or a name.

    A name is compared, after `slugify`, with each YAML file's `name`, its file name without
    the extension, and its `aliases`. Folders are searched in this order: `dirs`, then the
    folders in the DOCOCR_DOCTYPES_DIR environment variable, then the shipped folder; the first
    folder with a match wins. Inside a folder a match on `name` or file name beats a match on an
    alias, and ties go to the file that sorts first by file name. If the file that matches
    cannot be read or does not validate, DocTypeError (a ValueError) names it. Other broken
    files are ignored here; `find_doc_types` reports them. A folder in `dirs` or the variable
    that does not exist raises DocTypeError.

    A name that matches no file is not an error: it yields an empty DocType, and the caller
    must then list the fields.
    """
    if isinstance(name_or_path, DocType):
        return name_or_path
    text = str(name_or_path)
    if text.lower().endswith(_SUFFIXES):
        return load_doc_type_file(text)
    key = slugify(text)
    return _find(key, dirs) or DocType(name=key)


def resolve_fields(doc_type: DocType, requested: Optional[list[Union[str, FieldSpec]]]) -> list[FieldSpec]:
    """Turn what the caller asked for into FieldSpecs.

    - None / empty -> every field of the doc type.
    - A name known to the doc type -> its preset spec (with aliases, type, pattern).
    - Any other name -> an ad-hoc string field whose label is derived from the name.
    """
    if not requested:
        if not doc_type.fields:
            raise ValueError(
                f"Document type '{doc_type.name}' has no YAML file, so you must list the fields to extract."
            )
        return list(doc_type.fields)
    out: list[FieldSpec] = []
    seen: set[str] = set()
    for item in requested:
        spec = item if isinstance(item, FieldSpec) else (doc_type.field(item) or FieldSpec(name=item))
        if spec.name and spec.name not in seen:
            seen.add(spec.name)
            out.append(spec)
    if not out:
        raise ValueError("No valid field names were given.")
    return out
