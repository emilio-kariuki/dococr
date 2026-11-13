# dococr architecture

How the code is put together and why. For how to use the tool, see the [README](../README.md).
For working rules and commands, see [AGENTS.md](../AGENTS.md).

## What it does

One function: an image, a document type and a list of wanted fields go in; a value per field comes
out, with the evidence needed to decide whether to trust it.

```
extract(source, doc_type, fields)
   │
   ├─ doctypes.load_doc_type      name / YAML path / DocType  ─▶ DocType
   ├─ doctypes.resolve_fields     wanted names                ─▶ [FieldSpec]
   ├─ imaging.open_pages          image / PDF / TIFF / list   ─▶ pages: RGB images, upright, one at a time
   ├─ for each page:
   │   ├─ Extractor.extract       image + specs               ─▶ raw text, confidence, box per field
   │   │     └─ LocalExtractor
   │   │           ├─ preprocess.prepare      straighten and scale the page, detect lines
   │   │           ├─ OcrEngine.recognize     text lines with boxes and confidences
   │   │           └─ matcher.match_fields    label ─▶ value
   │   └─ normalize.normalize_value   raw text + field type   ─▶ typed value
   └─ merge the pages             best reading per field      ─▶ ExtractionResult
```

The command line (`cli.py`) and the HTTP service (`api.py`) are thin wrappers over
`pipeline.extract`. No extraction logic lives in them.

## Decisions

| Decision | Choice | Why |
|---|---|---|
| Where it runs | Entirely on the local machine | The inputs are identity documents. Nothing may leave the process. |
| OCR engine | RapidOCR 3.10 on ONNX Runtime, CPU | Installs with pip alone, needs no GPU or system packages, and ships its models inside the wheel. |
| Field extraction | Rule-based matching of printed labels to values | Deterministic, explainable, and needs no second model. The cost is that it only handles labelled layouts. |
| Document types | YAML files only; no document knowledge in the code | The owner's requirement. Supporting a new document, or correcting an existing one, must never need a code change. |
| Failure policy | A missing value is better than a wrong one | Weak matches are dropped or flagged; nothing is guessed. |
| Interfaces | Python function, CLI, HTTP service | One code path behind all three. |

A cloud vision backend existed in an early version and was removed on purpose. Do not add one back.

## No document knowledge in the code

The Python code is a generic engine. It holds no document names, no labels, no value formats, and
no behaviour keyed on what a field is called. All of that is in the YAML file of the document
type:

| Knowledge | Where it lives |
|---|---|
| The names a type can be asked for by | `name` and `aliases` at the top of its YAML |
| Which labels identify a field | the field's `label` and `aliases` |
| Which other printed text ends a value | `other_labels` at the top of its YAML |
| Whether a value runs over several lines | the field's `multiline` |
| Whether numeric dates are written month first | `date_order`, on the document or a field |
| Page text to skip between pages that the detector cannot recognise | `page_furniture` at the top of its YAML |
| Which words follow a value inside a sentence | the field's `until` |
| What kind of value a field holds | the field's `type` |
| What a valid value looks like | the field's `pattern` |

Earlier versions had three tables in the code: a list of about sixty common labels used as stop
boundaries for every document, word variants (No/Number, Licence/License, Colour/Color) expanded
for every label, and a rule that treated a field as a date if its name contained a word like
`date`. All three were removed, along with a table of document-type name synonyms. Do not
reintroduce anything of the kind. The test for whether something may live in the code: if it would
be different for another document or another country, it belongs in YAML.

What does remain in the code is knowledge that is independent of any document:

- how OCR confuses characters (`0`/`O`, `1`/`l`/`I`, `rn`/`m`),
- how dates are written (English and French month names; numeric dates read day first),
- how a value is judged against its declared `type` (a `name` should not be mostly digits),
- how English writes numbers in words (`Eight Hundred and Fifty Thousand`), ends sentences, and
  breaks words at the end of a line, including a short list of abbreviations whose full stop does
  not end a sentence (`Mr`, `Ltd`, `Co`, `No`),
- geometric thresholds.

The day-first reading of numeric dates is now only a default: `date_order` in the YAML overrides
it. The month-name languages remain the one item close to the line. The English word lists in the
matcher (`_ABBREVIATIONS`, and `_OPEN_WORDS`, the words after which a phrase cannot end, such as
`and`, `of`, `the`) are language knowledge of the same kind.

`tests/test_doctypes.py` fails if a shipped document name or alias appears in `doctypes.py`,
`cli.py` or `api.py`, and `tests/local/test_matcher.py` checks that the matcher's vocabulary is
exactly the YAML's labels, aliases and `other_labels`, and that a type with no `other_labels` gets
no stop labels from the code.

## Modules

| File | Responsibility |
|---|---|
| `src/dococr/pipeline.py` | `extract()`, the single entry point, and `get_extractor()`, which builds the extractor once per process. |
| `src/dococr/models.py` | The data contract: `FieldSpec`, `DocType`, `FieldResult`, `ExtractorOutput`, `ExtractionResult`, `FieldType`, `slugify`. |
| `src/dococr/doctypes.py` | Finds a document type's YAML file by name, alias or path across the search folders, and turns requested names into `FieldSpec`s. Holds no document names. |
| `src/dococr/presets/*.yaml` | The four YAML files shipped with the package. Plain data, replaceable by the user's own. |
| `src/dococr/pdftext.py` | Reads a digital PDF page's text layer into lines with boxes, and decides per page whether to trust it, OCR the page, or both. |
| `src/dococr/pagebreak.py` | Finds page furniture (headers, footers, page numbers) and reads `until` and `multiline` values across a page break. |
| `src/dococr/imaging.py` | `open_pages`: any source (image, PDF, multi-frame image, list of sources) to a sequence of upright RGB page images, rendered one at a time. |
| `src/dococr/extractors/base.py` | The `Extractor` interface and `ExtractorError`. |
| `src/dococr/extractors/local.py` | `LocalExtractor`: preprocess, OCR, match. |
| `src/dococr/preprocess.py` | Rotation, deskew and scaling, and the mapping of boxes back to the original image. |
| `src/dococr/ocr/__init__.py` | `OcrEngine` interface, `OcrLine`, and the shared default engine. |
| `src/dococr/ocr/rapid.py` | `RapidOcrEngine`: the RapidOCR wrapper. Owns the offline guarantee and call serialisation. |
| `src/dococr/ocr/matcher.py` | `match_fields`: the label-to-value matcher. The largest and most delicate file. |
| `src/dococr/normalize.py` | `normalize_value`: raw text to a typed value. |
| `src/dococr/cli.py` | The `dococr` command (Typer). |
| `src/dococr/api.py` | The HTTP service (FastAPI). |

## The data contract

`models.py` and `extractors/base.py` define what every part agrees on. Change them with care,
because the CLI output, the HTTP response and callers' code all depend on their shape.

- **`FieldSpec`** describes one wanted field: `name`, printed `label`, `aliases`, `type`, optional
  `pattern`, `multiline` (the value may continue on the lines below), `until` (the text that
  follows the value, one or several alternatives; used for values inside sentences), and a
  `description` for human readers. `name` is slugified (`"Order Number"` becomes
  `order_number`).
- **`DocType`** is one YAML file: `name`, `title`, `description`, `aliases` (other names for the
  type), `other_labels` (printed labels that are not fields and only mark where a value ends), and
  the list of `FieldSpec`s.
- **`Extractor.extract(image, doc_type, fields)`** returns an `ExtractorOutput` holding exactly one
  `FieldResult` per requested field, in order.
  - It fills `raw` (the text as printed, or `None`), `confidence`, `bbox` and `warnings`.
  - `bbox` is in the pixel coordinates of the image passed to the extractor, which is the caller's
    image after `load_image` has applied its EXIF orientation.
  - It leaves `value` alone. The pipeline computes `value` from `raw`.
  - A field that is not on the page is `raw=None`. `ExtractorError` is only for "the engine cannot
    run".
- **`normalize_value(raw, spec)`** returns `(value, warnings)`. It never raises, and it is
  idempotent in the value: a value that is already normalised comes back unchanged. Warnings about
  how the raw text was read (for example a two-digit year) are not repeated on a second pass.
- **`ExtractionResult`** is what callers get. Its JSON form is documented in the README.
  `FieldResult.page` (1-based) and `ExtractionResult.pages` are set by the pipeline when it merges
  pages. Extractors see one page at a time and do not set `page`.

Rules that follow from the contract:

- Field names in the shipped YAML files are public API. Add fields and aliases; do not rename.
- `doc_type.fields` may hold more fields than were requested. The matcher uses all of them to know
  where one value ends and the next label begins.
- Requested fields that are not in the document type are ad-hoc fields: a string `FieldSpec` whose
  only label is derived from the name. They get no other help.

## Finding a document type (`doctypes.py`)

`load_doc_type(name_or_path, dirs=None)` accepts a `DocType`, a path ending in `.yaml` or `.yml`,
or a name.

- **Search order:** the folders in `dirs`, then the folders in the `DOCOCR_DOCTYPES_DIR`
  environment variable (separated by `os.pathsep`, read at call time), then the shipped folder.
  The first folder with a match wins, so a user's file overrides a shipped one.
- **Matching:** after `slugify`, the name is compared with each file's `name`, its file name
  without the extension, and its `aliases`. Within a folder a match on name or file name beats a
  match on an alias, and ties go to the file that sorts first.
- **No match:** an empty `DocType`. The caller must then list the fields.
- **Broken files:** if the file that matches cannot be read or does not validate, `DocTypeError`
  (a `ValueError`) names the file and the problem. Other broken files in a folder are ignored by
  lookups and reported by `find_doc_types`, which `dococr types` prints on stderr.
- **Missing folder:** a folder in `dirs` or the variable that does not exist raises `DocTypeError`,
  so a typo is not silent. A wrong variable therefore makes every lookup fail until it is fixed.
- **Cache checked on every call.** Parsed files are cached per folder. Each call still lists the
  folder and stats its YAML files, and reuses the cache only if every file's name, modification
  time, size and inode are unchanged; otherwise the folder is parsed again. So an edited, added or
  deleted file takes effect on the next call, with no restart. The cache is guarded by a lock, and
  callers get deep copies, so mutating a returned `DocType` cannot corrupt it. YAML is parsed with
  PyYAML's C safe loader when it is available.

## Preprocessing and the single detection pass (`preprocess.py`, `extractors/local.py`)

`prepare(image, engine)` runs the text detector once, on the page reduced to at most 2000 px on
the long side, and uses the detected line quads to decide, in this order:

1. **Quarter turn.** If most text lines are vertical, the page is rotated 90 degrees.
2. **Skew.** The median angle of the text lines, corrected when it is between 0.5 and 20 degrees
   and at least two elongated lines were found.
3. **Scale.** The page is enlarged, up to 3 times, when the median text height is under 22 px (the
   target is 30 px), or when no text was detected and the long side is under 1000 px. Large pages
   are capped at 2000 px on the long side.
4. **Upside down.** Up to seven line crops from the levelled, scaled page are given to the
   engine's orientation classifier in one batched call. If they read at 180 degrees, the page is
   turned over.

The result is one affine warp. The matrix is kept in a `PageTransform`, so every box read from the
working image is mapped back to the coordinates of the image the extractor was given. A contrast
step (CLAHE) is applied to low-contrast pages. What was done is reported in the result's page-level
`warnings`, for example `rotated 270 degrees`, `deskewed -1.0 degrees` or
`low contrast: applied CLAHE`.

The orientation classifier squeezes every crop into a shape about four times wider than tall. A
full-width line of running text is 25 or more times wider than tall, and squeezed that far its
verdict was close to a coin flip: on prose pages it left some upside-down pages as they were and
turned some upright pages over. `_classifier_window` therefore shows it only the first
`CLASSIFIER_ASPECT` (6) line heights of each line. Over 345 synthetic pages of six document types
that took wrong orientation decisions from 5 to 0, with every verdict at least 0.3 from the
threshold.

**The detector runs once per image.** The quads from that first pass are mapped through the page
matrix into the working image (`PageTransform.line_quads`), and `LocalExtractor` calls
`engine.recognize(work, quads)`, which crops each line and runs the recogniser only. Earlier the
warped page was detected a second time before recognition; detection is a third or more of the OCR
time, so that second pass was the largest avoidable cost.

Reuse falls back to a full `engine.read(work)` (detect and recognise) in three cases:

- **No quads were found.** The enlarged page gets a second look.
- **The second detection would see the text much finer than the first did.** Each engine reports
  how its detector resizes an input (`detector_resolution`; RapidOCR enlarges inputs to a 736 px
  short side). If the working image would be detected at more than `MAX_DETECTOR_GAIN` (1.1) times
  the first pass's resolution, reuse is not trusted. This is small text on a large canvas.
- **RapidOCR would resize or pad the working image itself** (long side over 2000, short side under
  30, or wider than 8:1), because its crops would then come from altered pixels.

When the page needs no turn, deskew, rescale or contrast step, the second detection would have
seen the same pixels, and the single pass gives an identical result; a slow test checks that with
the real model. On transformed pages it is not identical but was measured to be no worse (see
Performance).

## Matching (`ocr/matcher.py`)

The module docstring describes the same steps next to the code. In outline:

1. **Tokenise.** Each OCR line is split into words with estimated positions. A colon becomes its
   own token. A word that glues a label to its value (`DateofBirth12.03.1990`) is split.
2. **Rows.** Lines are grouped into rows by vertical overlap and sorted left to right.
3. **Find labels.** The vocabulary is exactly what the YAML gives: every label and alias of every
   field in the document type, the requested ad-hoc fields, and the type's `other_labels`, which
   never produce values but mark where a value stops. No variants are generated: `No` and `Number`
   are different labels unless both are listed.
   - Text is compared in a canonical form: lower case, punctuation removed, and look-alike
     characters folded together (`0`/`o`, `1`/`l`/`i`, `5`/`s`, `8`/`b`). "Exact" below means exact
     in that form, so `P1N` matches `PIN`.
   - Words are compared one by one with an edit budget by length: up to 3 characters must match
     exactly, 4 to 7 may differ by one edit, longer words by two.
   - A single-word label of 5 characters or fewer (`Sex`, `PIN`, `Make`, `Year`, `Type`) must match
     exactly and must start an OCR box. Without this rule short labels match unrelated text.
   - A label is also compared as one joined string, which is how `LicenceNo` matches `Licence No`
     when OCR drops the space.
   - A hit needs a similarity of at least 0.8. Below 0.9 the field gets a "matched loosely" warning.
   - Overlapping hits are resolved in favour of the longer, better match (similarity × label
     length), so `National ID No` beats the `ID No` inside it.
4. **Find values.** For each label hit there are two candidates.
   - **To the right, on the same row:** words up to the next label hit. A colon ends the value once
     at least one word has been taken. A wide gap between two OCR boxes also ends it (2.5 × the
     label height after the first value word). Inside a single OCR box there is no gap limit.
   - **Directly below:** the nearest row within 1.8 × the label height, starting slightly left of
     the label. Its right edge is the next label on the label's row; when no label follows, it is
     unbounded.

   Each candidate is scored by OCR confidence × label match quality × position (right 1.0, below
   0.95) × how well the text suits the field's type and `pattern` (graded from 1.0 down to 0.3). A
   candidate is trimmed to the date or the pattern match inside it when there is one.
5. **Assign.** Fields take their best candidate in order of score. Two fields cannot take the same
   words, with one exception: fields that share a label hit (an ad-hoc field and a YAML field
   with the same label) get the same value.
6. **Pattern fallback.** A requested field that ended up with no value is searched for by its
   `pattern` across the page. That covers a label that was not found, a label with nothing usable
   beside it, and a value that another field took. The result is used only when exactly one
   distinct value matches, and its confidence is multiplied by `PATTERN_FALLBACK_FACTOR` (0.5).
   The fallback does not run for a candidate that was found and then rejected as too weak.
7. **Reject weak matches.** Below `REJECT_BELOW` (0.35) the field is returned as not found with a
   warning. Below `WARN_BELOW` (0.6) the value is kept with a `low confidence` warning.

Two optional field keys extend this. A field that sets neither goes through the steps above
unchanged.

**`multiline: true` (a block of lines).** The first line is found as above. Lines below are added
while each one starts within 1.5 line heights of the first line's left edge, has text 0.6 to 1.5
times its height, stays left of the next label on the label's row, is not itself a label or one
of the `other_labels`, and, when the value is to the right of its label, has nothing under the
label on that row (which is what stops it at the next form row even when that row's label is not
in the YAML). It ends at a blank gap taller than one line height or after 5 lines; a sixth
qualifying line produces a warning. `raw` is the lines joined by single spaces, `bbox` is their
union, and the confidence is that of the weakest line.

**`until` (a value inside running text).** The page is read in sentence order: each row is split
into visual lines at gaps wider than 1.5 text heights, and the next line is the nearest row below
(at most 2 text heights lower) that overlaps the text read so far, so a paragraph wraps to its own
margin and another column is not read. The label and the end text are matched like row labels and
may each straddle a line break; a single word of 5 characters or fewer must match exactly but need
not start an OCR box. The value is the text after the label up to the first match of any `until`
text.

- **Bound:** the end text must come within 40 words and 6 lines, or the field is missing. A
  `type: date` or `pattern` field may instead take a date or pattern match that starts within the
  first two words, at 0.7 of the confidence and with a warning.
- **Repeated anchors:** a value never contains its own label, so the label nearest before the
  value is the one used. A reading that runs through a distinctive anchor of something else
  (another field's label or end text, or one of the `other_labels`; two or more words, or one word
  of six or more characters) is discarded. If two different readings remain the field is missing,
  with a warning to lengthen the label or end text.
- **Shared anchors:** one field's end text may be the next field's label; the end text is never
  part of the value.
- **Not row labels:** the label of an `until` field does not act as a stop for form-style values.

**Punctuation as an end text.** `until` also accepts a single mark from `END_MARKS`:
`, . ; : ) ] ( [`, alone or in a list with word end texts.

- A mark counts where it ends a word, stands alone, or sits inside a token where the OCR lost the
  following space (`OTIENO.The`). An opening bracket ends the value just before it.
- **Not an end:** a mark between two digits (`1,200`, `12.5`, `10:30`); a closing bracket that
  matches one opened inside the value; letters in brackets inside a word (`Purchaser(s)`).
- **A full stop must end a sentence.** It is not an end before a lower-case word or a figure. After
  an abbreviation (a capitalised word of one or two letters, a word with a full stop inside it, or
  a word on the short list `_ABBREVIATIONS`) it ends the value only if a lower-case word follows
  before any other mark, and the value then carries a warning. After an unlisted two-letter
  capitalised word the value runs on, with a warning.
- **End of paragraph.** A full-stop end also accepts the end of the paragraph: the reading stops,
  or the line spacing is more than `PARA_PITCH` (1.3) times the paragraph's own after a short
  line. If the full stop itself was not read, the value carries a warning.
- **Sentence guard.** A field ending at `, ; : ) ] ( [` gets no value if its sentence or paragraph
  ends first. This is what protects against a dropped comma.

**Hyphens at a line break.** Applied only where a reading crosses a line break: the first piece
ends in a hyphen attached to a letter or digit and the next line starts with one.

- **Anchors** (labels and end texts) are compared against the two pieces and against the joined
  word. Nothing is guessed, so there is no warning.
- **Values** are decided in this order: a digit on either side keeps the hyphen; evidence from the
  page's own words and the YAML's texts (the joined word seen whole elsewhere removes it, the
  hyphenated form seen mid-line keeps it); lower case then a capital keeps it; both pieces in
  capitals keep it, with a warning, because hyphens in capitalised text are mostly names; anything
  else is joined without the hyphen, with a warning.

**Anchors glued by the OCR.** The recogniser sometimes drops a space (`NumberNAKURU`,
`Box4410-00506`, `Thepurchase`). An anchor of an `until` field may start inside its first OCR word
or end inside its last.

- The part of the word inside the anchor must be exactly the anchor's edge word, at least two
  characters, with no edit budget.
- **The join must show evidence:** punctuation, a letter against a figure, or a change of case
  (lower then upper, capitals then a capitalised word, capitals then a lower-case word). Same-case
  joins (`BETWEENJOSEPHINE`, `Kamauof`) are refused, as are `of` in `office` and `and` in `android`.
- A one-word anchor split by a change of case alone needs three or more characters and carries a
  warning. One exception to the evidence rule: a capitalised word of four or more characters glued
  before a lower-case label of two or more words that all match exactly (`Thepurchase price`).
- The value starts or ends exactly at the join, its box is estimated from character offsets, and
  two fields may take the two halves of one glued word.
- This applies to sentence anchors only, not to form-style row labels.
- A space dropped inside the value is not repaired: the value is returned as read.

On a 134-word page with 11 sentence fields the matcher takes about 10 ms of CPU, and 34 ms on a
960-word stress page, against about a second for the OCR. That was measured before the three
additions above; they were not re-timed, and the fast performance test (under 0.5 s per page)
still passes.

The matcher is tuned against the synthetic evaluation set. After any change to it, run
`tests/local/eval_synth.py` and compare the per-variant figures.

## Normalising (`normalize.py`)

| Type | Behaviour |
|---|---|
| `date` | Recognises numeric dates (day first unless the first group is a four-digit year or day-first is impossible), dates with English or French month names in either order (`12 Mar 1990`, `March 12, 1990`), dates written out in running text (`5th day of March, 2024`, `3rd of May 2021`, `March 5th, 2024`), and, only when nothing else matched, compact 8-digit dates (`YYYYMMDD` tried before `DDMMYYYY`). Repairs digit look-alikes inside date-shaped text (`l2.O3.1990`). A two-digit year is read as 20xx up to the current year plus 20, otherwise 19xx, with a warning; this can be wrong for a date of birth. Output is ISO 8601. An unparseable date is returned as cleaned text with a warning. |
| `integer`, `number` | Takes the first number in the text, repairs look-alikes inside mostly-digit tokens, handles thousands separators. `integer` rounds a decimal value, with a warning. When both `.` and `,` separate groups of three digits (`1.435,000`, an OCR misreading), it is read as thousands, with a warning. A hyphen after a letter is not a minus sign (`LR-4521`). A number written in English words is parsed (units to trillions, `and`, `Twenty-Five`, `a hundred`, `only`, `and Fifty Cents`, OCR-glued `FiftyThousand`); a lone number word inside other text is not. Words and figures together are checked against each other, and on disagreement the words are used, with a warning. |
| `identifier` | Upper-cases and removes whitespace. |
| `name` | Trims separator junk; warns when the name contains digits. |
| `string` | Collapses whitespace and trims separator junk. |

The `pattern` is checked against the normalised value (an identifier with its spaces already
removed, for example). If the value does not match, a limited number of look-alike characters are
swapped to see whether a matching reading exists (identifiers and strings only); a repaired value
carries the warning "corrected look-alike characters to fit the expected format". Otherwise the
value is kept and a warning is added. A pattern mismatch never discards a value.

## The offline guarantee (`ocr/rapid.py`)

RapidOCR resolves models by name and downloads them when the file is absent. To make that
impossible, `_bundled_model_paths()` passes explicit `Det.model_path`, `Rec.model_path` and
`Cls.model_path` values pointing at the three files inside the installed `rapidocr` package. With
explicit paths RapidOCR does not take its download branch for the model files. If a file is
missing, dococr raises `ExtractorError` before RapidOCR is constructed.

One more download path exists inside RapidOCR: the recogniser fetches a character dictionary when
the model file does not carry one. The bundled `PP-OCRv6_rec_small.onnx` embeds its dictionary in
the ONNX metadata (key `character`), so that path is not taken. A different recogniser model must
be checked for this before it is adopted.

Three things keep the guarantee true:

- `pyproject.toml` pins `rapidocr>=3.10,<3.11`, the series known to ship these file names and
  whose internals `RapidOcrEngine` uses (listed under "RapidOCR internals" below).
- `tests/local/test_end_to_end.py::test_runs_with_the_network_cut_off` runs an extraction on a
  fresh engine with socket connections and name resolution patched to fail. It covers the
  dictionary path as well as the model files.
- `tests/local/test_rapid_engine.py::test_missing_model_file_is_an_error_not_a_download` checks the
  missing-file path.

No dependency of the core package is a network client that dococr calls. Installing the package is
the only step that uses a network.

## Performance

Measured with `tests/local/bench.py` on a 12-CPU machine shared with other work (load 15 to 50),
so wall-clock figures are noisy. Compare best-of-N and CPU time, and run before and after close
together.

| Change | Effect | Why it is safe |
|---|---|---|
| OCR threads default to 4, not one per CPU (`DEFAULT_THREADS` in `ocr/rapid.py`) | About 3 to 4 times faster per image here, and about a sixth of the CPU time. | The models are small and do not scale past a few threads; ONNX Runtime's idle threads spin while they wait. Output is unchanged. |
| One detection pass, not two | OCR stages about 35% faster, 30% less CPU, on top of the above. | Exactly equivalent on untransformed pages; guarded by `MAX_DETECTOR_GAIN` elsewhere. Evaluation 1313 of 1320, against 1311 before, with no variant worse. |
| Upside-down check batched into one classifier call | Small. | Result equals one call per crop; checked by a slow test. |
| Document type YAML cached, validated by stat on every call | Lookup from about 20 ms (4 files) or 350 ms (40 files) to under 0.5 ms. | Edits still take effect on the next call. |
| `warm_up()` and service start-up warming | Moves the three-second model load off the first image. | No effect on results. |

After all of it, an image takes about one second with the models loaded (0.9 to 1.4 s over five
synthetic pages and one real ID photo at load 36), of which detection is 0.2 to 0.8 s and
recognition 0.4 to 0.7 s. Everything outside the OCR engine is under 10 ms.

Tried and rejected, with the measurement:

- **Recognition batch of 16** in place of 6: slower (padding to the widest line in the batch).
- **Reusing the first detection when its copy was capped at 1280 px:** `two_column` fell from 100%
  to 99.2%, because shrinking a wide page and letting the detector enlarge it again merged
  neighbouring words. The copy is now capped at 2000 px, the same as the working image.
- **Reusing the first detection with no resolution check:** small text on a large canvas lost
  fields (130 of 132 against 132). Hence `MAX_DETECTOR_GAIN`.

Not tried, and the likely next saving: the detector enlarges every input to a 736 px short side
(`Det.limit_side_len`, `limit_type`), which makes wide pages large detector inputs. Changing it
alters detection on small pages and needs its own evaluation.

Rules for further work: do not trade accuracy for speed; do not replace the models, because the
offline guarantee depends on the three files shipped in the `rapidocr` wheel; run the evaluation
after any change to `ocr/` or `preprocess.py`.

## RapidOCR internals

RapidOCR's public call cannot recognise a list of line crops, so `RapidOcrEngine.recognize`,
`classify_upside_down` and `warm_up` use parts of RapidOCR 3.10 that are not public API. Every use
holds `_call_lock`. An upgrade must re-check all of these:

- **On the `RapidOCR` instance:** `_load_det_model()`, `_load_cls_model()`, `_load_rec_model()`;
  `text_score`, `min_side_len`, `max_side_len`, `min_height`, `width_height_ratio`;
  `cfg.Global.use_preprocess_img`, `cfg.Global.use_vertical_padding`, `cfg.Det.limit_side_len`,
  `cfg.Det.limit_type`; and the components `text_det`, `text_cls`, `text_rec`.
- **Components and helpers:** `ch_ppocr_det.TextDetector.sorted_boxes`,
  `ch_ppocr_rec.TextRecInput` and the recogniser output's `.txts` and `.scores`, the classifier
  called on a list and its `.cls_res`, `utils.process_img.get_rotate_crop_image` and
  `resize_image_within_bounds`.
- **Behaviour copied from RapidOCR that must stay in step with it:** corner ordering, clipping and
  the small-box filter; dropping empty texts and lines under `text_score`; the conditions under
  which RapidOCR resizes or pads an image; the detector's input resizing.

`test_single_pass_equals_two_pass_on_an_untransformed_page` (slow, real model) fails if any of
these drifts.

## Several pages (`imaging.py`, `pipeline.py`)

- **Sources.** `open_pages(source, max_pages)` accepts a path, bytes, a PIL image, a PDF (path or
  bytes), a multi-frame image, or a list or tuple of these, flattened in order. A PDF is
  recognised by `%PDF-` in its first kilobyte, never by its name. Only TIFF, GIF and WebP frames
  count as pages; a JPEG's extra frames do not, because a phone photo can carry a second frame
  that is not a page. Only TIFF was tested.
- **Lazy.** The page count is known before anything is rendered, and pages are rendered one at a
  time as the loop reaches them, so memory stays at about one page.
- **PDF rendering** uses `pypdfium2`, which bundles PDFium (permissive licence; PyMuPDF was ruled
  out because it is AGPL). It is imported on the first PDF, not at module level, so commands that
  read no document stay light; a test checks this. PDFium is not thread-safe, so every call holds
  `_PDFIUM_LOCK`.
- **Render size.** Each page is rendered so that its long side is `RENDER_LONG_SIDE` (2000 px,
  the same cap `preprocess` applies). The scale is derived from the page's own size, so a page
  with enormous dimensions cannot produce an enormous bitmap. That is about 170 dpi on A4.
- **Nothing touches the disk.** PDF bytes are read through a callback and a path is read in place.
  This was checked with `strace` on the HTTP route and is asserted by a test.
- **Limits.** `max_pages` defaults to `DEFAULT_MAX_PAGES` (20), checked before any page is
  rendered; over the limit is `PageLimitError`, never a truncation. The service reads
  `DOCOCR_MAX_PAGES`.
- **Bad PDFs** raise `SourceError`. PDFium's error code is a process-wide "last error" that a
  failing load does not always reset, so a password error is believed only when the file contains
  `/Encrypt`; everything else is reported as damaged or unreadable.
- **Merging.** The extractor runs on each page with the same document type and fields. Per field:
  a reading that normalised to a value beats one that did not; then the higher confidence wins
  (an unknown confidence ranks last); then the earlier page. If another page has a different
  non-empty value, the chosen field gets a warning naming the pages, never the values. Page-level
  warnings are prefixed `page N: ` when there is more than one page, and `raw_text` separates
  pages with `--- page N ---`. A single image gives the same values as before, with `page` 1 and
  `pages` 1.
- **Not done:** running pages in parallel (the OCR engine is serialised on purpose); a value
  over three or more pages.

### The text layer (`pdftext.py`)

While a page is open for rendering, `imaging.py` also reads its characters (`Pages.with_text()`
yields each image with a `PdfText`): Unicode value, box, whether drawn invisibly (render mode 3
or 7), whether PDFium reports a mapping error, whether it reads upright, and drawing order; plus
pictures and FreeText, Stamp, Ink and Widget annotations. Boxes go through PDFium's own
page-to-device map, so `/Rotate` and the CropBox are handled, and land in the pixels of the
rendered page, the same image the extractor receives.

**Lines.** Characters are grouped along a baseline into runs, runs into rows by vertical overlap;
a row is split at a horizontal gap over 1.0 text heights (matching how the detector splits
columns and cells); a space is inserted where the PDF has one or where the gap exceeds 0.15 text
heights. Overprinted characters are read once, ligatures expanded, zero-width characters dropped,
and a line-end hyphen arrives as `-`. Trusted lines have confidence 1.0.

**Trust, per page, under `auto`:**

| Page content | Result |
|---|---|
| No characters | OCR, no note (exactly as before) |
| Only invisible text (a scanner's searchable PDF) | OCR, note "invisible text layer … not used" |
| A line with a control character, U+FFFD, a non-character, a mapping error, or a private-use character inside a word | that line is not used; its area is OCR'd |
| Text that does not show (grey-level spread under 6 in its area, or drawn before a picture that covers it) | dropped, with a note |
| Text drawn at an angle | its area is OCR'd |
| Pictures of at least 1% of the page, and the annotations above | their areas are OCR'd |
| Over 20% of visible characters untrusted or angled | whole page OCR'd |
| Otherwise | text layer ("text"), or "hybrid" when some areas need OCR: each area is cropped and OCR'd, the whole page if areas cover over 60%; an OCR line more than 30% on top of a text-layer line is dropped |

Invisible OCR layers are not trusted under `auto` because they cannot be judged without running
our own OCR, which then gives our own result anyway. On a fixture with deliberate errors,
`always` scored 46 of 65 and `auto` 59 of 65. `always` uses any text layer and OCRs only pages with
none, without OCRing pictures; `never` is the old path. The thresholds (1%, 20%, 60%, 30%) are
choices, not tuned.

`LocalExtractor` builds the OCR engine lazily, so a document whose pages all use the text layer
never imports `rapidocr` or `onnxruntime`. The extractor contract grew compatibly:
`Extractor.supports_text_layer` (default False) decides whether `text_layer=` is passed.

**Not detectable:** a text layer whose characters map to the wrong letters with no error flag.

### Across page breaks (`pagebreak.py`)

**Furniture.** A line is furniture when it is among the three outermost rows of text in the top or
bottom 15% of the page and is a page number ("Page n", "p. n", "n of m", "n/m"; a bare number or
"- n -" if repeated at that edge of another page or equal to the page's position), or repeats at
the same edge of another page (digits ignored, similarity 0.8, within 5% of the page height), or
has no letters or digits, or is listed in the YAML's `page_furniture`. Rows are peeled from the
edge inwards and the first non-furniture row stops the peeling.

**Joining.** `join_pair` takes the last 8 rows of one page and the first 8 of the next, without
furniture, places the second page's rows under the first at the first page's line spacing,
aligned on the left margins, and runs the matcher on that strip for `until` and `multiline` fields
only. A reading is kept only if its label, value or end text lie on both pages. It replaces the
start page's own reading when that is missing or overlaps it, otherwise it joins the normal
merge; it is refused, with a warning, if it would take words another field already read. Pages
whose line heights differ by more than 1.6 times are not joined (OCR boxes are about 1.4 times as
tall as text-layer boxes, so a text page still joins an OCR page). A joined `multiline` value
carries a warning. Form values are never read across a break.

**Result.** `FieldResult.parts` (`ValuePart{page, bbox}`) lists every piece of a value whose own
text is on two pages; the first piece equals `page` and `bbox`.

**Measured** (`tests/local/eval_pdf.py --seeds 5`, a three-page agreement with headers, footers,
page numbers and two values per document crossing a break): digital PDF from its text 65 of 65
(crossing 10 of 10) at 0.15 s a page; the same pages by OCR 65 of 65 at about 4 s a page;
image-only PDF 56 of 65 (crossing 9 of 10, misses are OCR errors). Rotated, cropped, mixed,
private-use and unmapped-font fixtures 65 of 65; a page that shows sideways 63 of 65.

## Concurrency

RapidOCR stores each call's `use_det` / `use_cls` / `use_rec` flags on the shared instance, and
`RapidOcrEngine` makes detect-only, classify-only and full-read calls. Two threads inside it at
once would overwrite each other's flags. `_call_lock` therefore serialises every call into
RapidOCR, and `tests/local/test_rapid_engine.py` checks it.

The lock is per call, not per image. An image makes several calls (one detect, one batched
classify, one recognise or one full read), and the calls of two images being processed by
different threads can interleave. That is safe, because each call sets its own flags while holding the lock.

`pipeline.get_extractor()` and `ocr.get_default_engine()` build their shared object under a lock,
so two first requests arriving together still load the models once.

Consequences: within a process OCR work does not run in parallel, and ONNX Runtime uses several
CPU threads for the call in hand. Throughput comes from running more processes. Each process loads
its own copy of the models.

## HTTP service (`api.py`)

- **Endpoints** are documented in the README.
- **`_UploadGuard`** is ASGI middleware that exists because Starlette parses the multipart body
  before the endpoint runs and moves any file part over 1 MB to a temporary file on disk. The guard
  rejects `POST`, `PUT` and `PATCH` requests with no `Content-Length` (411) or with one over the
  limit plus 64 KB for the other form fields (413), then raises Starlette's in-memory threshold to
  that cap. The threshold is a class attribute, so the change is process-wide. Uploads therefore
  never reach the disk. The endpoint checks the file's own size again.
- **Document types by name only.** A `doc_type` ending in `.yaml` or `.yml` is rejected so that a
  request cannot make the server read a file.
- **No logging of content.** The module has no logging calls. Keep it that way.
- **Errors.** `ExtractorError` is 503; `ValueError` from the pipeline and unreadable images are 422.

## Tests

| Location | What it covers | Needs the OCR model |
|---|---|---|
| `tests/test_contract.py` | Pipeline behaviour with a fake extractor | No |
| `tests/test_doctypes.py` | Document type lookup across folders; no document names in the lookup, CLI or API code | No |
| `tests/test_normalize.py` | Normalisers | No |
| `tests/test_cli.py`, `tests/test_api.py` | Interfaces, with a fake extractor | No |
| `tests/local/test_matcher.py` | The matcher on hand-built OCR lines | No |
| `tests/local/test_multiline_prose.py` | `multiline` and `until`: each behaviour and failure mode, on hand-built OCR lines | No |
| `tests/local/test_prose_typeset.py` | Punctuation end texts, hyphens at line breaks, numbers in words in the matcher | No |
| `tests/local/test_prose_glue.py` | Anchors glued by the OCR, and the false anchors that must be refused | No |
| `tests/test_pages.py`, `tests/pdfs.py` | Sources, page order, the merge rule, limits and bad PDFs with a fake extractor; the helper builds PDFs and TIFFs inside the tests | No |
| `tests/local/test_pdf_end_to_end.py` | A two-page PDF through the real model | Yes (`slow`) |
| `tests/test_pdf_text.py` | Text-layer lines, trust decisions, rotation, crop, annotations, the `pdf_text` setting, no OCR import for digital PDFs | No |
| `tests/test_pagebreak.py` | Furniture, joining across a break, `parts`, the clash and merge rules | No |
| `tests/local/test_pdf_text_end_to_end.py` | Image-only PDF across breaks, invisible layer, fallbacks, hybrid page | Yes (`slow`) |
| `tests/local/test_real_layouts.py` | Layouts found on real documents (wrapped values, times, repeated and prefix labels), with invented values | No |
| `tests/synth_pdf.py`, `tests/local/eval_pdf.py` | Builds the multi-page agreement as digital, image-only, OCR-layer, rotated, cropped, mixed PDFs; scores them | |
| `tests/local/test_synth_fixtures.py` | The fixture YAML and the renderer stay in step | No |
| `tests/doctypes/*.yaml` | Fixture document types for the tests, not shipped: `synth_form` (multi-line values), `synth_agreement` (sentence values), `synth_contract` (typeset: hyphenated breaks, punctuation ends, amounts and dates in words), `synth_glued` (words run together next to anchors). | |
| `tests/local/test_local_extractor.py` | `LocalExtractor` and preprocessing (upscale, cap, quarter turn, upside down, skew) with a fake engine | No |
| `tests/local/test_rapid_engine.py` | Call serialisation; a missing model file is an error | No |
| `tests/local/test_end_to_end.py` | Real model on synthetic pages; the offline tests | Yes (`slow`) |
| `tests/local/test_single_pass.py` | Reuse of the first detection: box mapping, crop orientation on turned pages, fallback conditions, `warm_up` | No |
| `tests/local/eval_synth.py` | Accuracy report, not a test | Yes |
| `tests/local/bench.py` | Per-stage timings, not a test | Yes |
| `tests/synth.py` | Renders synthetic documents with Pillow: 4 types × 10 variants | |

The synthetic variants are `clean`, `colon`, `two_column`, `value_below`, `rot90`, `rot180`,
`rot270`, `skew`, `blur_lowres` and `photo`.

## Known limits

- Accuracy is measured on synthetic pages and one real national ID. Real logbooks, licences and
  passports are untested.
- Not handled: labels wrapped onto two lines, bilingual `A / B` labels, the machine-readable zone
  of a passport, non-Latin scripts.
- Sentence values depend on the wording in the YAML. Not handled there: prose set in several
  columns, a sentence that continues on the next page, number words in languages other than
  English, a glued anchor with a misread letter or a same-case join, and a comma or other small
  mark that the OCR invents or drops (the first cuts a value short, the second makes it missing).
  The prose test pages are easier than real agreements: their YAML was written knowing every
  wording, the text is clean type, and the end texts are distinctive.
- A space the OCR drops inside a value is returned as read (`JOSEPHINEWANJIRU`), with high
  confidence and no warning.
- A wrapped line that starts with a figure, or that wraps back under the label with nothing
  showing the value goes on, is not joined; the value carries a warning instead.
- A sentence value in a two-column form usually comes back missing, because reading in sentence
  order crosses into the other column.
- OCR character errors inside a correctly found value are not flagged (`Baneshares`).
- Measured on the fixture types (`eval_synth.py --types <type> --seeds 5`): `synth_form` 436 of
  450, `synth_agreement` 435 of 440, `synth_contract` 500 of 520, `synth_glued` 399 of 400.
  - Form and agreement: all 19 misses are OCR character errors inside correctly bounded values
    (`O` read as `0` in `P.O. BOX` accounts for 13), with confidence around 0.88 and no warning.
  - Contract, 20 misses: 7 where the OCR lost the space after a comma, 7 where blur changed `P.O.`
    or `L.R.`, 5 hyphenated words where the page gave no evidence and the rule guessed wrong (all
    flagged), and 1 comma read as a full stop.
  - Glued: the one miss is `April2016` inside a value, which the date parser rejects.
  - `synth_glued` places its glue where the join has evidence, so its figure shows the mechanism
    works through real OCR, not how often real documents glue in ways it accepts.
- Handwriting is read unreliably: see "Handwriting" below.
- A value over three or more pages is not joined, and a text layer with wrongly mapped letters
  is trusted.
- The pattern fallback can return a wrong value when exactly one unrelated string on the page fits
  the pattern. It is flagged with a warning and a confidence of at most 0.5.
- A value can run on into neighbouring text whose label the YAML does not list. The remedy is to
  list that label under `other_labels`. In a value-below layout the same happens when no label
  follows on the label's row.
- The synthetic pages print only field labels, so `other_labels` is covered by unit tests but not
  by the accuracy evaluation. The `other_labels` in the shipped files were written from general
  knowledge of those documents, not from real scans.
- Inside a single OCR box there is no gap limit, so two values that the OCR engine joined into one
  box are returned together.
- An ambiguous two-digit year can be placed in the wrong century.
- The shipped YAML files describe Kenyan documents. Documents from elsewhere need their own files.

## Handwriting

Measured, not assumed. The data was 604 lines of real joined-up English handwriting (the IAM
line-level test set as mirrored on Hugging Face, `Teklia/IAM-line`; check IAM's own terms before
any commercial use of that data), the same text in ten handwriting-style fonts, and the same text
in ordinary type. Scoring ignored case and punctuation.

| Recogniser | Real handwriting: characters wrong / words right | Printed: words right | Size | Time per line |
|---|---|---|---|---|
| PP-OCRv6 small (shipped) | 12.9% / 62.5% | 99.8% | 21 MB | 0.04 s |
| Eight other RapidOCR recognisers | all worse, or equal at much greater cost | | 4 to 90 MB | |
| TrOCR small handwritten | 6.5% / 87.3% | 98.3% | 246 MB | about 1 s |
| TrOCR base handwritten | 4.6% / 89.2% | 98.1% | 1.3 GB | about 4 s |

- **Detection is not the problem.** On synthetic forms mixing printed labels with handwritten
  entries, the shipped detector located 99% of the entries.
- **Confidence does not flag the errors.** Of 488 lines with an error, 298 had a confidence of 0.9
  or more. This cuts against "a missing value is better than a wrong one": handwritten fields can
  come back wrong and confident.
- **No better drop-in exists.** No model RapidOCR 3.10 lists does better than the shipped one.
- **TrOCR is the only thing measured that helps**, and it is not a RapidOCR model: an
  encoder-decoder with its own tokenizer and decoding loop, needing PyTorch or a separate ONNX
  path, files far larger than the current 32 MB, and one to four seconds per line on a CPU. It was
  trained on IAM, so those figures flatter it; quantised and ONNX versions were not measured.
  Adopting it is a decision for the owner, because of size, speed and the second runtime.
- **Not measured:** block capitals, and photographs of real handwritten forms.

## Real documents

The first measurement on real pages (scratch harness, not in the repository; the documents are
public US Treasury and NASA agreements and stay outside the repository): 24 documents, 188 pages,
280 fields. With YAML written from the first document of each family and corrected once: before
the fixes it prompted, 250 correct, 5 missing, 3 wrong with a warning, 22 wrong without; after,
269, 7, 3 and 1. The fixes, each covered by `tests/local/test_real_layouts.py` with invented
values:

- **Wrapped form values** (11 silent errors): a continuation back at the label's margin or at a
  different indent is joined when the line above ends open (`, ; & / -`, an open bracket, or a
  word in `_OPEN_WORDS`) or the new line starts in lower case; otherwise it is left out with a
  warning. A field without `multiline` whose value visibly stops mid-phrase gets a warning.
- **Times** (9): a colon between two figures stays inside the word, so `9:00 a.m.` is not cut.
- **A short label occurring several times**: a longer reading is no longer cut at a second
  occurrence of the label; if two readings remain plausible the field is missing with a warning.
- **A label that is the start of another field's label**: found as a "shadowed" hit, used only
  when the field has no other reading, with a warning. All fields' labels are searched whatever is
  requested.
- **Typed fields that do not parse**: `value` is null, `raw` and the warning are kept; `missing`
  lists every field whose value is null.
- **US dates**: `date_order: month_first`.
- **A dropped one-letter word**: the one-edit tolerance of the joined label comparison may no
  longer add or drop a whole one-letter word (it was eating the `A` of `Series A`).

These thresholds and word lists are generic rules seen working on 24 documents, not validated on
more. Warning rates on real documents were not measured.

## Possible next steps

- Measure on real scans of each document type and tune aliases and thresholds from the failures.
- PDF and multi-page input with one merged result and a page number per field.
- Multi-line values and wrapped labels.
- Passport machine-readable zone parsing as a cross-check.
