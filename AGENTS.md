# Instructions for coding agents

`dococr` extracts caller-chosen fields from photos of documents and returns them as JSON. A
document type is a YAML file listing the fields and the labels printed next to them; the code is a
generic engine that knows nothing about any particular document. It runs fully offline: RapidOCR
on the CPU reads the text, and rule-based matching pairs printed labels with values.

Read [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md) before changing code. The [README](README.md)
documents the user-facing behaviour.

## Rules that must not be broken

1. **Offline only.** No network calls, no cloud or hosted models, no API keys, no SDK for a remote
   service, no model downloads at run time. Do not add a cloud backend, even as an option. The
   owner removed one deliberately.
2. **Never commit real documents.** Scans of real IDs, licences or logbooks are personal data. They
   go in `samples/`, which git ignores. Do not copy their contents into tests, docs, commit
   messages or logs. Test data comes from `tests/synth.py`.
3. **Do not download photos of real people's documents** for testing.
4. **Do not log or persist images or extracted values**, in the HTTP service or anywhere else.
5. **A missing value is better than a wrong one.** Do not make the matcher or normalisers guess.
6. **No document knowledge in the code.** No document names, labels, value formats, example
   documents in user-facing strings, or behaviour keyed on what a field is called. If something
   would be different for another document or another country, it goes in the YAML file
   (`label`, `aliases`, `type`, `pattern`, `other_labels`, type-level `aliases`). The owner asked
   for this explicitly; tables of common labels, word variants and name-based type guessing were
   removed and must not come back. See "No document knowledge in the code" in
   `docs/ARCHITECTURE.md` for what may stay.
7. **Field names in the shipped YAML files are public API.** Add fields and aliases. Do not rename
   or remove.
8. **Do not commit or push unless asked.**

## Setup and commands

The virtualenv runs Python 3.12, installed for this user with `uv python install 3.12` (the
system Python is 3.10, which cannot install the fixed Pillow 13). Use the virtualenv for everything.

```bash
"$(uv python find 3.12)" -m venv .venv      # only if .venv does not exist
.venv/bin/pip install -e ".[all,dev]"

.venv/bin/pytest -m "not slow"              # fast suite, seconds, no OCR model
.venv/bin/pytest                            # everything; the slow tests load the real model, minutes
.venv/bin/python tests/local/eval_synth.py --seeds 3            # accuracy on synthetic pages
.venv/bin/python tests/local/eval_synth.py --types logbook --variants clean,rot90
.venv/bin/python tests/local/eval_synth.py --types synth_form,synth_agreement --seeds 5   # multi-line and prose fixtures
.venv/bin/python tests/local/eval_synth.py --types synth_contract --seeds 5               # typeset prose: hyphens, punctuation ends, number words
.venv/bin/python tests/local/eval_synth.py --types synth_glued --seeds 5                  # words run together next to anchors
.venv/bin/python tests/local/eval_pdf.py --seeds 5               # multi-page PDFs: text layer, OCR layer, image-only, page breaks
.venv/bin/python tests/local/bench.py --reps 5                  # per-stage timings, wall and CPU (--pdf for PDFs)

.venv/bin/dococr types
.venv/bin/dococr fields logbook
.venv/bin/dococr extract IMAGE --type national_id --format table --raw-text
```

Generate a test image without a real document (run from the repository root):

```bash
.venv/bin/python -c "
import sys; sys.path.insert(0, 'tests')
from synth import make_document
doc = make_document('logbook', 'photo', seed=1)   # types and variants are listed in tests/synth.py
doc.image.save('/tmp/logbook.jpg'); print(doc.truth)"
```

## Where things are

| To change | Edit | Then run |
|---|---|---|
| Fields, labels, aliases, patterns or stop labels of a shipped type | `src/dococr/presets/<type>.yaml` | fast suite, then the eval for that type |
| How a type's YAML file is found | `src/dococr/doctypes.py` | `tests/test_doctypes.py` |
| How a label is matched to a value | `src/dococr/ocr/matcher.py` | `tests/local/test_matcher.py`, then the full eval |
| Dates, numbers, identifiers clean-up | `src/dococr/normalize.py` | `tests/test_normalize.py` |
| Rotation, skew, scaling | `src/dococr/preprocess.py` | `tests/local/test_local_extractor.py`, then the slow suite and the eval |
| The OCR engine wrapper | `src/dococr/ocr/rapid.py` | slow suite |
| The result shape or field description | `src/dococr/models.py` | everything; update the README's Output section |
| Reading a digital PDF's text, and when to trust it | `src/dococr/pdftext.py` | `tests/test_pdf_text.py`, then `tests/local/eval_pdf.py` |
| Values across page breaks, headers and footers | `src/dococr/pagebreak.py` | `tests/test_pagebreak.py`, then `tests/local/eval_pdf.py` |
| CLI | `src/dococr/cli.py` | `tests/test_cli.py` |
| HTTP service | `src/dococr/api.py` | `tests/test_api.py` |

## Contracts to keep

- `Extractor.extract` returns exactly one `FieldResult` per requested field, in request order. It
  fills `raw`, `confidence`, `bbox`, `warnings`. It does not set `value`; the pipeline does.
- A field that is not on the page is `raw=None`, never an exception. `ExtractorError` means the
  engine cannot run.
- `normalize_value` never raises and is idempotent.
- `bbox` is in the pixel coordinates of the image given to the extractor (the caller's image
  after its EXIF orientation is applied), not those of the preprocessed working image.
- `cli.py` and `api.py` contain no extraction logic. They call `pipeline.extract`.

## Common tasks

**Support a new document.** Write a YAML file. No code changes. The README section "Writing a
document type" lists the keys. For a user's own document, the file goes in their folder
(`--types-dir` or `DOCOCR_DOCTYPES_DIR`), not in this repository.

**Ship a new document type with the package.**

1. Create `src/dococr/presets/<name>.yaml` following an existing file. Put other names for the
   type under `aliases`, and labels that are printed but not extracted under `other_labels`.
2. List every printed wording of each label in the field's `aliases`. Nothing is expanded.
3. Add the type to `tests/synth.py`: a new branch in `items_for` returning
   `(title, [Item, ...], absent_fields)`, and the name in `DOC_TYPES`. This puts it in the eval
   and in the slow test `test_synthetic_accuracy_per_variant`, which requires at least 90% per
   variant, so a weak new type fails the slow suite.
4. Update the docs that list the shipped types: the table under "The included files" in the
   README and the "4 types × 10 variants" line in `docs/ARCHITECTURE.md`.

**A value picks up text that belongs to something else.** Add the label of that other text to the
type's `other_labels`. Do not add it to the code.

**A real document prints a label that is not recognised.** Add the printed text to that field's
`aliases`. If the label is a short single word, check `tests/local/test_matcher.py` for the
short-label rules before adding it, because short aliases match unrelated text easily.

**A value runs over several lines, or sits inside a sentence.** That is YAML too: `multiline: true`
on the field, or `until: "<words after the value>"`. The README sections "Values over several
lines" and "Values inside sentences" give the rules.

**Change the matcher.** Write the failing case first as a unit test with hand-built `OcrLine`s in
`tests/local/test_matcher.py`. After the change, run the full eval and compare each variant with
the figures before. A fix for one layout often breaks another.

**Upgrade `rapidocr`.** The offline guarantee depends on the wheel shipping the model files named
in `_BUNDLED_MODELS` in `src/dococr/ocr/rapid.py`. Check the new wheel's `models/` folder, and
check that the recogniser model embeds its character dictionary (ONNX metadata key `character`);
without it RapidOCR downloads a dictionary file. `RapidOcrEngine` also uses RapidOCR internals
that are not public API; `docs/ARCHITECTURE.md` lists them under "RapidOCR internals", and each
must be re-checked. Update the names and the version pin in `pyproject.toml` together, and run the
slow suite, which includes the no-network test and the single-pass equivalence test.

**Make it faster.** Measure first with `tests/local/bench.py`, before and after, close together,
and compare best-of-N and CPU time, because this machine's load swings widely. Read "Performance"
in `docs/ARCHITECTURE.md` for what has been done, tried and rejected. Run the evaluation after any
change to `ocr/` or `preprocess.py`; a speed-up that loses fields is not accepted.

## Traps

- **RapidOCR can download models.** Given no explicit path it resolves a model by name and
  downloads it when the file is missing or fails its checksum. Always construct it through
  `RapidOcrEngine`, which passes the bundled paths. Never call `rapidocr.RapidOCR()` bare.
- **RapidOCR keeps per-call flags on the instance.** All calls go through
  `RapidOcrEngine._call`, which holds `_call_lock`. Do not call the underlying object directly.
- **`pypdfium2` must not be imported at module level.** It is imported inside `imaging.py` on the
  first PDF, and every PDFium call holds `_PDFIUM_LOCK`, because PDFium is not thread-safe. A test
  checks that `dococr types` does not import it.
- **Pages are read one at a time and merged in `pipeline.py`.** Extractors never see more than one
  page and do not set `FieldResult.page`.
- **Test PDFs are built inside the tests** by `tests/pdfs.py` (image PDFs, TIFFs, and raw text
  PDFs with render modes, broken font maps, rotation and crop boxes) and `tests/synth_pdf.py`.
  Do not commit PDF or image files.
- **A scanner's invisible text layer is not trusted by default.** Do not change that without
  measuring: on the fixture it cost 13 of 65 fields.
- **A `date`, `integer` or `number` field that does not parse has `value` null**, not the text.
- **Real public documents used for measurement live in a scratch folder, never in the
  repository**, and their contents and people's names must not be copied into tests or docs.
- **Changes to sentence reading need all four fixture evaluations**, not only the shipped types:
  no shipped YAML uses `until`, so the default evaluation cannot see them.
- **Every limit on untrusted input exists because an attack was measured.** They are listed under
  "Limits on untrusted input" in `docs/ARCHITECTURE.md`, with a regression test each in
  `tests/test_security.py`. Do not raise or remove one without re-running the attack. Images are
  opened only through `imaging.py`, which checks size and shape before decoding and allows six
  formats; never call `Image.open` elsewhere on untrusted bytes.
- **Starlette spools uploads over 1 MB to disk.** `_UploadGuard` in `api.py` prevents that. A
  directory listing cannot detect the spooled file because it is unlinked at once; the test watches
  `SpooledTemporaryFile.rollover` instead.
- **FastAPI parses the request body before dependencies run**, so size checks belong in the ASGI
  middleware, not in the endpoint.
- **The slow suite takes minutes** and is CPU-heavy. Run it in the background for long sessions.
- **`ruff` is configured but the code is not lint-clean** (mostly `Optional[X]` style). Do not
  mass-fix lint inside an unrelated change.
- **`tests/synth.py` is imported as `synth`.** `tests/local/conftest.py` adds `tests/` to the
  import path.
- **Nothing about a label or a type is inferred.** `No` and `Number` are different labels unless
  both are listed, and a field is a date only if its YAML says `type: date`. An ad-hoc field
  (requested by name, not in the YAML) gets the one label derived from its name and type `string`.
  When a test or a real document fails for this reason, fix the YAML.
- **`tests/synth.py` has its own label lists** for rendering. A label it prints must be in the
  field's YAML `label` or `aliases`, or the eval drops. The same holds for the fixture types in
  `tests/doctypes/` (`synth_form`, `synth_agreement`, `synth_contract`, `synth_glued`), which are test data and
  are not shipped; do not add an agreement type to `src/dococr/presets/`, because agreements
  differ in wording and each author writes the YAML for theirs.
- **The orientation classifier must only see a short slice of a line** (`_classifier_window` in
  `preprocess.py`). Given whole lines of running text it turned prose pages upside down.
- **Handwriting is read unreliably and wrong values come back confident.** The measurements and
  the option that was not taken are under "Handwriting" in `docs/ARCHITECTURE.md`. Do not claim
  handwriting support, and do not add a second OCR model without the owner's decision.
- **Document type YAML is cached but re-checked on every call** against each file's name,
  modification time, size and inode, so edits take effect at once. Do not replace that with a
  time-based cache. A folder named in `DOCOCR_DOCTYPES_DIR` that does not exist makes every lookup
  raise `DocTypeError`.
- **More OCR threads is slower.** The default is 4 (`DEFAULT_THREADS`). One thread per CPU, which
  is ONNX Runtime's own default, was three to four times slower here.
- **Do not benchmark while the test suite or the evaluation is running.** They use the same CPUs.
- **The first detection is reused for recognition.** A change to how the page is warped must keep
  `PageTransform.line_quads` correct, or lines are cropped from the wrong place. The tests in
  `tests/local/test_single_pass.py` cover turned and skewed pages.
- **`_UploadGuard` changes a Starlette class attribute**, so it affects every multipart parse in
  the process.

## Calling dococr from another project

If you are an agent wiring dococr into a different codebase:

```python
from dococr import ExtractorError, extract

result = extract(image_path_or_bytes, "national_id", ["id_number", "full_name"])
values = result.values()          # {"id_number": "...", "full_name": "..."}; None where not found
doubtful = [n for n, f in result.fields.items() if f.value is None or f.warnings]
```

- The first argument can be an image (path, bytes or PIL image), a PDF (path or bytes), or a list
  of these as the pages of one document. Each field in the result has `page`. A damaged or
  password-protected PDF, or more than `max_pages` pages (default 20), raises `SourceError`, a
  `ValueError`.
- Install it as a package (`pip install /path/to/python-ocr`, or `"/path/to/python-ocr[api]"` when
  the HTTP service is needed). Do not vendor the source folder.
- Call it from a long-lived process, and call `dococr.warm_up()` at start-up. Loading the models
  takes about three seconds, once per process.
- It is CPU-bound and takes about a second per image. Keep it off an async event loop.
- Each process uses up to `DOCOCR_OCR_THREADS` (default 4) CPU threads for OCR. Size the number of
  worker processes with that in mind.
- OCR work is serialised within a process. Use more processes for throughput.
- Document types are YAML files. Keep the project's own in a folder and set
  `DOCOCR_DOCTYPES_DIR` to it; then ask for a type by its name.
- Catch `ExtractorError` (engine cannot run) and `ValueError` (bad request; its subclass
  `DocTypeError` means a document type YAML file or folder is missing or invalid). A missing image
  path raises `FileNotFoundError`, an unreadable image raises Pillow's `UnidentifiedImageError`,
  and an absurdly large image raises Pillow's `DecompressionBombError`.
- For a non-Python project, run `dococr serve` and post to `POST /extract`. The service has no
  authentication and must not be exposed publicly as it is.
- Treat the output as unverified. Route fields with warnings or low confidence to a person.

## Before you say a change is done

- The fast suite passes.
- The slow suite passes if you touched `ocr/`, `preprocess.py`, `extractors/` or `pyproject.toml`.
- The eval figures are no worse if you touched the matcher, the normalisers or a shipped YAML file.
- No document name, label or value format was added to Python code.
- The README and `docs/ARCHITECTURE.md` still describe what the code does.
- Report what you ran and what you did not.
