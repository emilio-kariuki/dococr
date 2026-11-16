<img src="assets/icon.svg" width="64" height="64" alt="">

# dococr

Give it a photo, a scan or a PDF of a document, say what kind of document it is and which fields
you want, and get those fields back as JSON. A document of several pages is one call and one
result. It runs entirely on your own machine.

```bash
dococr extract id.jpg --type national_id --fields id_number,full_name,date_of_birth
```

```python
from dococr import extract

result = extract("id.jpg", "national_id", ["id_number", "full_name", "date_of_birth"])
result.values()   # {"id_number": "12345678", "full_name": "JANE WANJIRU DOE", "date_of_birth": "1990-03-12"}
```

A document type is a YAML file that lists the fields and the labels printed next to them. The code
knows nothing about any particular document: it reads the YAML you give it and returns those
fields. Four YAML files are included (national ID, driving licence, vehicle logbook, passport), and
you add your own the same way. It reads forms and cards that print a label next to each value, values
that run over several lines, and values inside the sentences of a typed agreement.

**Contents:** [Install](#install) · [Fully offline](#fully-offline) · [Command line](#command-line) ·
[Python](#python) · [Output](#output) · [HTTP service](#http-service) ·
[Using it in an existing project](#using-it-in-an-existing-project) ·
[Document types and fields](#document-types-and-fields) · [How it reads a page](#how-it-reads-a-page) ·
[Accuracy and limits](#accuracy-and-limits) · [Security](#security) · [Performance](#performance) · [Development](#development)

Working on the code? See [docs/ARCHITECTURE.md](docs/ARCHITECTURE.md). Coding agents: see
[AGENTS.md](AGENTS.md).

## Install

Python 3.11 or newer is recommended (3.10 works, with an older image library; see
[Security](#security)). No GPU and no system packages are needed.

```bash
pip install .                # library and the `dococr` command
pip install ".[api]"         # plus the HTTP service
pip install -e ".[all,dev]"  # everything, editable, with the test tools
```

## Fully offline

There is no cloud model, no API key and no network call at run time. The image never leaves the
process.

- **Engine.** [RapidOCR](https://github.com/RapidAI/RapidOCR) on ONNX Runtime reads the text on
  the CPU. Rule-based matching then pairs each printed label with its value.
- **Models.** The three OCR model files (about 32 MB) are inside the `rapidocr` package, so they
  arrive with `pip install`. dococr points at those files explicitly and never downloads a model.
  If a file is missing you get an error, not a download. `rapidocr` is pinned to the 3.10 series
  for this reason.
- **Installing is the only step that needs a network.** For a machine with no internet, collect
  the packages on a connected machine of the same OS and Python version, then install from the
  folder:

  ```bash
  pip wheel . -w wheels/                              # on the connected machine
  pip install --no-index --find-links wheels/ dococr  # on the offline machine
  ```

  For the HTTP service as well, use `pip wheel ".[api]" -w wheels/` and install `"dococr[api]"`.

A test runs a full extraction with every socket connection blocked, and the package was checked by
installing it into a clean environment and running it with no network interfaces.

## Command line

```bash
dococr types                              # every document type it can find
dococr fields logbook                     # a type's fields, their kind, and the labels they match
dococr extract photo.jpg --type logbook --fields registration_number,owner_name
dococr extract photo.jpg --type logbook   # every field of the type
dococr extract a.jpg b.jpg --type national_id          # several documents give a JSON list
dococr extract agreement.pdf --type my_type            # a PDF: all its pages, one result
dococr extract p1.jpg p2.jpg --type my_type --one-document   # photos of the pages of one document
dococr extract photo.jpg --type logbook --format table # quick look in the terminal
dococr extract photo.jpg --type logbook --raw-text     # also return all text read off the page
dococr extract cert.jpg --doc-type-file my_type.yaml   # one YAML file of your own
dococr extract cert.jpg --type my_type --types-dir ./doctypes   # your own folder of YAML files
dococr serve --port 8000                  # run the HTTP service
```

| Option | Meaning |
|---|---|
| `--type`, `-t` | Document type, by the `name` in its YAML file, its file name, or one of the `aliases` the file lists (the included `driving_license` file lists `driving licence`, for example). A name with no YAML file works together with `--fields`. A value ending in `.yaml` or `.yml` is read as a file path. |
| `--fields`, `-f` | Comma-separated field names. Leave it out to get every field of the type. |
| `--doc-type-file` | A YAML file describing a document type. Use it in place of `--type`; if both are given, the file wins. |
| `--types-dir` | A folder of your own YAML files, searched before the included ones. Repeatable. Also on `types` and `fields`. The `DOCOCR_DOCTYPES_DIR` environment variable does the same for every command. |
| `--format` | `json` (default) or `table`. |
| `--raw-text` | Include the full text of the page in the result. |
| `--one-document` | Treat the files given as the pages of one document, in order, and return one result. Without it, each file is a separate document. A PDF or multi-page TIFF is always one document. |
| `--max-pages` | The most pages one document may have. Default 20. A longer document is an error, not a partial result. |

A problem with the input (a missing or unreadable image, an unknown type with no fields, a bad
YAML file, a types folder that does not exist) prints a short `error: ...` message to stderr and
exits with a non-zero status. It does not print a traceback. A YAML error names the file and the
problem.

## Python

```python
from dococr import extract

result = extract(source, "driving_license", ["license_number", "date_of_expiry"])
```

`source` is any of these:

| Source | Read as |
|---|---|
| A file path, image bytes (for example an upload), or a `PIL.Image` | One page. |
| A PDF, as a path or as bytes | All its pages. It is recognised by its content, not its file name. |
| A multi-page TIFF | All its pages. |
| A list of any of the above | The pages of one document, in order. |

```python
extract("agreement.pdf", "my_type")                    # every page of the PDF
extract(["page1.jpg", "page2.jpg"], "my_type")         # photos of two pages, one document
extract(pdf_bytes, "my_type", max_pages=50)            # raise the 20-page limit
```

Each page is read on its own and the results are merged; see [Several pages](#several-pages).

```python
result.values()                           # {"license_number": "DL1234567", "date_of_expiry": "2031-03-12"}
result.missing                            # names of requested fields that were not found
result.fields["license_number"].value     # the cleaned value
result.fields["license_number"].raw       # the text exactly as printed
result.fields["license_number"].confidence
result.fields["license_number"].bbox      # (left, top, right, bottom) in pixels
result.fields["license_number"].warnings
result.model_dump(mode="json")            # the whole result as a plain dict
```

Describe a field in more detail by passing a `FieldSpec` where you would pass a name:

```python
from dococr import FieldSpec, extract

county = FieldSpec(name="county", label="County", aliases=["County of Residence"])
extract("id.jpg", "national_id", ["id_number", county])
```

A field that is simply not on the page is not an error: it comes back in `missing`. `extract`
raises in these cases:

| Exception | When |
|---|---|
| `ValueError` | The request makes no sense: a document type with no YAML file and no fields listed. |
| `dococr.DocTypeError` | A document type YAML file is missing, is not valid YAML or has a wrong key, or a types folder does not exist. The message names the file. It is a `ValueError`. |
| `FileNotFoundError` | The image path does not exist. |
| `PIL.UnidentifiedImageError` | The file or bytes are not an image Pillow can read. |
| `dococr.SourceError` | A PDF is damaged, password protected or has no pages, or the document has more pages than `max_pages` (`dococr.PageLimitError`). It is a `ValueError`. |
| `dococr.ExtractorError` | The OCR engine cannot run. |

## Output

This is the result for one field found and one not found. The CLI, the Python
`result.model_dump(mode="json")` and the HTTP service all return this shape.

```json
{
  "doc_type": "national_id",
  "backend": "local",
  "fields": {
    "date_of_birth": {
      "name": "date_of_birth",
      "value": "1990-03-12",
      "raw": "12.03.1990",
      "confidence": 0.947,
      "bbox": [451, 318, 644, 359],
      "warnings": []
    },
    "date_of_expiry": {
      "name": "date_of_expiry",
      "value": null,
      "raw": null,
      "confidence": null,
      "bbox": null,
      "warnings": ["label not found"],
      "page": null
    }
  },
  "missing": ["date_of_expiry"],
  "warnings": [],
  "raw_text": null,
  "elapsed_ms": 4995,
  "pages": 1
}
```

A found field also has `"page"`, the page its value came from (`1` for a single image).

| Key | Meaning |
|---|---|
| `fields.<name>.value` | The cleaned value to store. Dates are `YYYY-MM-DD`, identifiers are upper-case with no spaces, integers and numbers are JSON numbers. `null` when not found. |
| `fields.<name>.raw` | The text exactly as it was read off the page. |
| `fields.<name>.confidence` | 0 to 1. Combines how sure the OCR was about the characters and how well the label matched. Use it to decide what a person should check. |
| `fields.<name>.bbox` | Where the value is on the image you supplied: left, top, right, bottom, in pixels. For a phone photo that carries an orientation tag, the coordinates are those of the picture as a viewer displays it, upright. |
| `fields.<name>.warnings` | Anything doubtful about this field, for example `low confidence (0.52)` or `could not parse date`. |
| `missing` | Requested fields with no value. That includes a `date`, `integer` or `number` field whose text was found but could not be read as one (`Not applicable`): its `value` is `null`, `raw` keeps the text, and a warning says why. |
| `fields.<name>.parts` | Only for a value that continues onto the next page: one `{"page", "bbox"}` per piece. Otherwise `null`. |
| `warnings` | Notes about the whole page, for example `rotated 90 degrees`. |
| `raw_text` | All text on the page in reading order. Only filled when you ask for it. |
| `backend` | Always `local`. |
| `fields.<name>.page` | The page the value came from, counting from 1. `null` when not found. |
| `pages` | How many pages were read. |
| `elapsed_ms` | Time taken for the whole document. |

These are extraction results, not verified data. Check by eye anything that matters.

### Several pages

A PDF, a multi-page TIFF or a list of images is read page by page, and the pages are merged into
one result:

- **Each field comes from the page that read it most confidently**, and from the earlier page on a
  tie. `page` says which, and `bbox` is a position on that page.
- **If another page read a different value for the same field**, the chosen value carries a
  warning such as `page 2 read a different value than page 1; check the document`. The same value
  on several pages is not a warning.
- **A field found on no page** is missing, as for a single image.
- **Page notes say which page**, for example `page 2: rotated 90 degrees`, and `raw_text` marks
  where each page starts.
- **A sentence value or a `multiline` block can continue onto the next page.** Running headers,
  footers and page numbers between the two are skipped. Such a value has `page` and `bbox` where it
  starts and a `parts` list with a piece per page; the table output shows it as `p1+2`. Form
  values are never read across a page break. If the detector misses a header or footer that
  appears only once, list its text under `page_furniture` in the YAML.
- **Time grows with the pages**: about a second each once the models are loaded, or about 0.15 s
  for a page read from a digital PDF's text.

### Digital PDFs and scans

A PDF exported from a word processor contains its text exactly. dococr reads that text instead of
running OCR, which is more accurate (no dropped spaces or misread letters) and about 25 times
faster. A scanned PDF is read by OCR. Each page's note says which happened, for example
`page 2: read from the PDF's text layer`.

The choice is made page by page, from what the page contains:

- **Text layer used** when the text is visible and every character maps to a real letter.
- **OCR instead** for a scan, including a scanner's "searchable PDF", whose hidden text came from
  the scanner's own OCR and may be wrong; for text whose characters do not map to letters; for
  text drawn at an angle; and for text hidden under something else.
- **Both, on a mixed page**: a signed or stamped page, or one with a pasted picture, takes the text
  layer for the typed part and runs OCR on the pictures and stamps only.

Override it with `pdf_text="always"` / `"never"` in Python, `--pdf-text always|never` on the CLI, or
`DOCOCR_PDF_TEXT` for the service. `always` trusts any text layer, hidden or not; `never` reads every
page as an image. When no page needs OCR, the OCR engine is not even loaded.

A text layer whose letters are mapped to the wrong characters without any error mark cannot be
detected without OCR; it would be trusted.

## HTTP service

```bash
pip install ".[api]"
dococr serve                 # listens on 127.0.0.1:8000

curl -F file=@logbook.jpg -F doc_type=logbook -F fields=registration_number,owner_name \
     http://127.0.0.1:8000/extract
```

| Endpoint | |
|---|---|
| `POST /extract` | Multipart form: `file` (an image or a PDF; send several `file` parts for the pages of one document, in order), `doc_type` (a name), optional `fields` (comma-separated, or a JSON list), optional `include_raw_text` (`true`/`false`). Returns the JSON shown under [Output](#output). |
| `GET /doc-types` | Every document type found, including those in your own folder. |
| `GET /doc-types/{name}` | One type with its fields. |
| `GET /health` | `{"status": "ok"}` |

| Status | When |
|---|---|
| 200 | Extraction ran. Check `missing` for fields that were not found. |
| 404 | Unknown document type on `GET /doc-types/{name}`. |
| 411 | The request has no `Content-Length`. |
| 413 | The request body is over the size limit. |
| 422 | A form field is missing or malformed, the file is empty or not a readable image, `doc_type` has no YAML file and no `fields` were given, `doc_type` is a file path, or the type's YAML file is broken. |
| 503 | The OCR engine could not process the document (details go to the server log, not the response), or the service is busy: too many extractions or connections at once. Retry shortly. |

Things to know before you deploy it:

- **No authentication.** It binds to `127.0.0.1` by default. Keep it internal and have your own
  backend call it, or put it behind your own access control.
- **Uploads stay in memory.** They are never written to disk, and the service does not log image
  contents or extracted values.
- **Page limit.** A document may have at most 20 pages; change it with `DOCOCR_MAX_PAGES`. A
  longer one is refused with 422, as is a damaged or password-protected PDF.
- **Upload limit.** 10 MB by default, for the whole request, however many files it carries. Change it with the `DOCOCR_MAX_UPLOAD_MB` environment
  variable, which also accepts fractions. The limit is checked against the declared size of the
  request before the body is read.
- **Image size limits.** A page over 64 megapixels (`DOCOCR_MAX_MEGAPIXELS`), smaller than 16
  pixels on a side, or more than 30 times longer than it is wide is refused with 422 before it is
  decoded. These exist because a file of a few kilobytes can otherwise demand gigabytes of memory.
  Only PNG, JPEG, TIFF, WebP, GIF and BMP images, and PDFs, are read.
- **Load limits.** At most 2 extractions run at once (`DOCOCR_MAX_CONCURRENT`); others wait up to
  30 seconds and then get 503. Expect up to about 1.6 GB of memory per running extraction for the
  largest accepted image. At most 100 connections are open at once (`dococr serve
  --max-connections`). Request input is bounded: 200 field names, 200 characters per name.
- **Put a reverse proxy in front if it faces a network you do not control.** The service has no
  request-header timeout, so clients that open connections and send nothing can hold all 100
  connection slots and lock others out. A proxy such as nginx or Caddy adds timeouts, rate limits,
  TLS and authentication.
- **API description pages are off.** Set `DOCOCR_API_DOCS=1` to serve `/docs`, `/redoc` and
  `/openapi.json`.
- **OCR work is not parallel inside a process.** Requests wait for each other at the OCR engine.
  For more throughput, run several processes: `uvicorn dococr.api:app --workers 4`.
- **Document types are names only.** A request cannot make the server read a file path. To serve
  your own types, put their YAML files in a folder and start the service with
  `DOCOCR_DOCTYPES_DIR=/path/to/folder`. The folder is read on every request, so a new or edited
  file takes effect without a restart.

## Using it in an existing project

**If your project is Python, install dococr as a package and call `extract`.** Do not copy the
source folder into your codebase.

```bash
pip install /path/to/python-ocr                                             # from a local checkout
pip install "dococr @ git+https://github.com/<you>/python-ocr.git"          # from your git remote
```

```python
from dococr import ExtractorError, extract

def read_id(upload_bytes: bytes) -> dict:
    result = extract(upload_bytes, "national_id", ["id_number", "full_name", "date_of_birth"])
    needs_review = [
        name for name, field in result.fields.items()
        if field.value is None or field.warnings or (field.confidence or 0) < 0.8
    ]
    return {"values": result.values(), "needs_review": needs_review}
```

Three things matter in a web application:

1. **Keep the process alive.** Loading the OCR models takes about three seconds and happens once
   per process. Call `extract` from a long-running worker, not from a new process per image, and
   call `dococr.warm_up()` when the worker starts so the first image is not the slow one.
2. **Do not block your event loop.** An image takes about a second of CPU work. In an async framework run it
   in a thread pool (`await run_in_threadpool(extract, ...)`) or a background job queue.
3. **OCR work is not parallel inside a process.** Concurrent calls wait for each other at the OCR
   engine. Add worker processes for throughput. Each one loads its own copy of the models.

**If your project is not Python** (Node, PHP, a mobile backend and so on), run the
[HTTP service](#http-service) next to your application and post images to it.

## Document types and fields

Everything dococr knows about a document comes from that document's YAML file: its name, the
fields, the labels printed next to them, and the other text on the page. Nothing about any
particular document is written into the code, so supporting a new document never needs a code
change.

### Where the YAML files are

| Place | How it is used |
|---|---|
| Your own folder | Any folder of `.yaml` files. Point at it with `--types-dir DIR` or the `DOCOCR_DOCTYPES_DIR` environment variable (several folders separated by `:`). |
| The included files | `src/dococr/presets/`: `national_id.yaml`, `driving_license.yaml`, `logbook.yaml`, `passport.yaml`. |
| A single file | `--doc-type-file FILE.yaml`, or the path as the document type in Python. |

Your folder is searched first, so a file of yours with the same name as an included one replaces
it. `dococr types` lists everything found, and `dococr fields <type>` shows what a type contains.

```bash
export DOCOCR_DOCTYPES_DIR=./doctypes
dococr types
dococr extract card.jpg --type gym_card --fields member_number,joined_on
```

```python
from dococr import extract

extract("card.jpg", "gym_card", ["member_number", "joined_on"])   # uses DOCOCR_DOCTYPES_DIR
extract("card.jpg", "doctypes/gym_card.yaml")                     # or name the file
```

### Writing a document type

```yaml
name: gym_card                        # the name you ask for it by
title: Gym membership card
aliases: ["membership card"]          # other names you can ask for it by
other_labels:                         # printed on the card, but not something to extract
  - Member Signature
  - Emergency Contact
fields:
  - name: member_number               # the key in the result
    label: Member No                  # the label exactly as printed
    aliases: ["Membership Number"]    # every other way the label is printed
    type: identifier
    pattern: 'GM-\d{5}'               # the shape of a valid value
  - name: member_name
    label: Member
    type: name
  - name: joined_on
    label: Joined
    type: date
  - name: monthly_fee
    label: Monthly Fee
    type: integer
```

Keys for the document:

| Key | Meaning |
|---|---|
| `name` | Required. The name you ask for the type by. Name the file `<name>.yaml`. |
| `title`, `description` | For people. `dococr types` shows the title. |
| `aliases` | Other names the type can be asked for by. |
| `date_order` | `day_first` (default) or `month_first`, for numeric dates such as `12/05/08`. Set it for documents that write dates the American way. Can also be set on a single field. |
| `page_furniture` | Text printed on a page that is not part of the document's content and that dococr should skip when a value continues onto the next page, such as `Continued overleaf`. Repeated headers, footers and page numbers are found without it. |
| `other_labels` | Labels printed on the document that you do not want extracted, such as a signature line. They are never returned. They tell dococr where a neighbouring value ends, so list them when a value picks up text that belongs to something else. |
| `fields` | The list of fields. |

Keys for each field:

| Key | Meaning |
|---|---|
| `name` | Required. The key used in the result. |
| `label` | The label as printed. Defaults to the name with underscores turned into spaces. |
| `aliases` | Every other printed form of the label. Nothing is expanded for you: if a document prints `Member No` on one version and `Member Number` on another, list both. |
| `type` | How the value is cleaned up. See the table below. Defaults to `string`. A date is only returned as `YYYY-MM-DD` when the field says `type: date`. |
| `pattern` | A regular expression that a valid value matches in full. A value that does not match is still returned, with a warning. If no value is found beside the label and exactly one piece of text on the page matches the pattern, that text is used, with its confidence halved (so at most 0.5) and a warning. A pattern that repeats a group which itself repeats without limit, such as `(a+)+`, is refused when the file is loaded, because it can take practically forever on some text. |
| `handwritten` | `true` when the value is written by hand. The reading is then always returned with a warning to check it by eye, however confident the engine is. See [Handwriting](#handwriting). |
| `multiline` | `true` when the value can run onto the following lines, such as an address. See [Values over several lines](#values-over-several-lines). |
| `until` | The words that come straight after the value. Use it for a value inside a sentence. One text or a list of alternatives. See [Values inside sentences](#values-inside-sentences). |
| `description` | A note for people reading the file. It does not affect matching. |

| `type` | Result |
|---|---|
| `string` | Text with extra whitespace removed. |
| `name` | A person's name, tidied. |
| `date` | `YYYY-MM-DD`. Reads `12.03.1990`, `12/03/1990`, `12-MAR-1990`, `12 Mar 2031`, `March 12, 1990`, `1990-03-12`, and dates written out in a sentence, with the day or year in figures or in words: `5th day of March 2024`, `the Fifth day of March Two Thousand and Twenty-Four`. A numeric date is read day first unless the YAML sets `date_order: month_first`; the other order is used only when the first is impossible, with a warning. A two-digit year gets a warning. |
| `integer` | A whole number. A value with a decimal part is rounded, with a warning. A number written in English words is read too: `Eight Hundred and Fifty Thousand` gives `850000`. |
| `number` | A number that may have a decimal part, in figures or in words (`Twelve point five`, `... and Fifty Cents`). |
| `identifier` | Serials and reference numbers: upper-case, inner spaces removed. When the field has a `pattern` and the value does not fit it, look-alike characters (`O`/`0`, `I`/`1`, `S`/`5` and similar) are swapped if that makes it fit, with a warning. |

Small differences are tolerated when a label is read: letter case, punctuation, and one or two
wrong characters from the OCR. Different wording is not, which is why each wording goes in
`aliases`.

### Values over several lines

Add `multiline: true` to a field whose value can continue below its first line.

```yaml
  - name: postal_address
    label: Postal Address
    multiline: true
```

The first line is found as usual. Lines below it are added while they line up with it, are about
the same size, and are not a label the YAML lists. The value ends at a blank gap, at the next
label, or after 5 lines. The result joins the lines with spaces, and its `bbox` covers all of them.
If there was a sixth line the value is cut at five and carries a warning.

A wrapped line that starts further left, back under the label, or that is indented differently, is
added only when something shows the value goes on: the line above ends with a comma, `and`, `of`,
an open bracket and the like, or the new line starts in lower case. When a line looks like a
continuation but there is no such evidence, it is left out and the value carries a warning
saying so.

A field without `multiline` whose value visibly stops mid-phrase (it ends with a comma or `and`,
or the next line plainly continues it) is returned as read, with a warning suggesting
`multiline: true`.

### Values inside sentences

For a typed agreement or letter, a value sits in the middle of a sentence. Describe it by the words
just before it (`label`) and the words just after it (`until`).

> THIS AGREEMENT is entered into on this **14th day of August 2025** by and between **PETER NJOROGE
> WAWERU** of ID Number **23456789** residing at **Plot 45, Kimathi Street, Nairobi** (the Seller) ...

```yaml
name: car_sale
title: Vehicle sale agreement
other_labels: ["NOW IT IS HEREBY AGREED", "IN WITNESS WHEREOF"]
fields:
  - name: agreement_date
    label: entered into on this
    until: by and between
    type: date
  - name: seller_name
    label: by and between
    until: of ID Number
    type: name
  - name: seller_id
    label: of ID Number
    until: residing at
    type: identifier
  - name: seller_address
    label: residing at
    until: "(the Seller)"
```

This returns `2025-08-14`, `PETER NJOROGE WAWERU`, `23456789` and the address. The value may wrap
over line breaks, and so may the label and the end text. One field's `until` can be the next
field's `label`, as above.

How to write these so they work:

- **Copy the wording from the document.** Small OCR mistakes are tolerated; different words are not.
- **Make the label and end text long enough to be unique.** Punctuation is ignored, so `(the Seller)`
  also matches a later "the Seller". If the same label and end text fit two places with different
  text between them, the field comes back missing with a warning telling you to lengthen them.
- **List the alternatives.** If agreements word it two ways, give both: `aliases` for the label,
  and a list for `until`. The value ends at whichever end text comes first.
- **A short end text ends the value early.** `until: of` stops at the first "of", which is wrong
  for a value that can contain it.
- **The end text must come soon.** It has to appear within 40 words and 6 lines of the label, or
  the field is missing. This stops a value from swallowing a paragraph.
- **A value can end at a punctuation mark.** `until` also takes one of `,` `.` `;` `:` `)` `]` `(`
  `[`, alone or in a list with words. See below.
- **Add headings and fixed phrases to `other_labels`** (`IN WITNESS WHEREOF`, for example) so that
  no value runs through them.
- **Use `until` for running text and `multiline` for forms.** `multiline` does nothing on a field
  that has `until`.

**Ending at punctuation.** When nothing more distinctive follows the value, end it at a mark:

```yaml
  - name: purchaser_name
    label: "(the Vendor) and"
    until: ","                 # "... and DANIEL KIPCHUMBA ROTICH, a businessman of ..."
    type: name
  - name: county
    label: situated in
    until: "."                 # "... situated in Nakuru County. The Vendor has ..."
  - name: purchase_price
    label: purchase price shall be Kenya Shillings
    until: "("                 # "... Two Million Four Hundred and Fifty Thousand (KES 2,450,000)"
    type: integer              # comes back as 2450000
```

- **A full stop must end a sentence.** Full stops in initials, in `P.O. Box`, after short forms
  such as `Ltd.`, `Co.` and `No.`, and inside figures are read past, so `Mwangi, Otieno & Co.
  Advocates.` comes back whole. When a value ends at a full stop that follows such a short form,
  it carries a warning. At the end of a paragraph the value ends there even if the full stop was
  not read.
- **A comma ends the value at the first comma**, except between figures (`1,200,000`). Do not use
  it for a value that contains commas, such as an address; use the words that follow.
- **If the sentence ends before a comma, semicolon, colon or bracket is reached, the field is
  missing.** This is deliberate: the OCR sometimes drops a small mark, and a value that ran on to
  the next one would be wrong.
- **Give one mark per entry.** `until: [",", "of"]` is fine; `until: ")."` is refused.

**Amounts and dates in words.** With `type: integer` or `type: number`, an amount written in
English words is returned as a number. If the text has both words and figures, as in `Eight Hundred
and Fifty Thousand (KES 850,000)`, they are compared, and a disagreement is reported in a warning
(the words are used). With `type: date`, days and years in words are read as well. Put the currency
and phrases such as "the sum of" in the `label`; they are part of your document's wording.

**Words broken at the end of a line.** A typeset word split by a hyphen, `encum-` at the end of one
line and `brances` at the start of the next, is put back together, both when it is part of your
label or end text and when it is in the value. A real hyphen is kept when a figure is next to it
(`1021-00100`) or when the page shows the word written with it elsewhere. When the page gives no
evidence, a word in lower case is joined, a word in capitals keeps its hyphen, since that is
usually a double-barrelled name, and either way the value carries a warning saying what was done.

**Words run together by the OCR.** The OCR sometimes drops a space, giving `NumberNAKURU` or
`Box4410`. Your label and end text are still found when their first or last word is stuck to its
neighbour, provided something marks the join: punctuation, a letter next to a figure, or a change
between lower case and capitals. A missing space inside the value itself is returned as it was
read (`JOSEPHINEWANJIRU`), so check names.

This is matching on the wording you supply, not understanding of the text. It suits agreements that
follow a standard form. A document worded differently from your YAML returns missing fields, not
wrong ones.

### The included files

These are ordinary YAML files. Edit them, copy them, or replace them with your own.

| Type | Fields |
|---|---|
| `national_id` | `id_number`, `serial_number`, `full_name`, `date_of_birth`, `sex`, `nationality`, `place_of_birth`, `place_of_issue`, `date_of_issue`, `date_of_expiry` |
| `driving_license` | `license_number`, `surname`, `other_names`, `national_id_number`, `date_of_birth`, `sex`, `blood_group`, `date_of_issue`, `date_of_expiry`, `vehicle_classes`, `country` |
| `logbook` | `registration_number`, `chassis_number`, `engine_number`, `make`, `model`, `body_type`, `colour`, `fuel`, `year_of_manufacture`, `engine_capacity`, `passengers`, `tare_weight`, `gross_weight`, `owner_name`, `owner_pin`, `date_of_registration`, `logbook_number`, `previous_registration` |
| `passport` | `passport_number`, `surname`, `given_names`, `nationality`, `date_of_birth`, `sex`, `place_of_birth`, `date_of_issue`, `date_of_expiry`, `country_code`, `issuing_authority` |

### Asking for a field that is not in the YAML

A name that the YAML does not list still works as an ad-hoc field. Its label is the name with
underscores turned into spaces, so `county_of_residence` looks for the printed label "County Of
Residence" and nothing else, and the value is returned as plain text. For other printed wordings,
or for a date or number type, add the field to the YAML.

```bash
dococr extract id.jpg --type national_id --fields id_number,county_of_residence
dococr extract permit.jpg --type work_permit --fields permit_number,employer   # no YAML: list the fields
```

## How it reads a page

1. **Straighten.** The page is turned upright if it is rotated by 90, 180 or 270 degrees, deskewed
   if it is tilted, and enlarged if the text is small.
2. **Read.** The OCR engine returns every line of text with its position and a confidence.
3. **Match.** For each field you asked for, the label (or an alias) from the YAML is searched for
   among the text, allowing for small OCR mistakes. The value is taken from the right of the label
   or directly below it, stopping at the next label the YAML lists, whether a field or one of the
   `other_labels`. A `multiline` field continues onto the lines below, and a field with `until`
   reads on in sentence order until its end text.
4. **Clean.** The text is converted according to the field's type.

A match with confidence below 0.35 is dropped and the field is reported missing. Between 0.35 and
0.6 the value is kept with a `low confidence` warning. The intent is that a missing value is
better than a wrong one.

## Accuracy and limits

What has been measured:

- **Synthetic documents.** 1313 of 1320 fields correct (99.5%) on 120 rendered pages of the four
  included types, across clean, rotated, skewed, blurred and photo-like variants. Blurred
  low-resolution pages were the weakest at 97.7%. The seven misses were OCR errors, all dropped
  spaces (`A B C` read as `ABC`); none was a value taken from the wrong label. These pages print
  only field labels, so they do not test `other_labels`.
- **Multi-line and sentence values.** On 170 further rendered pages of four made-up document types:

  | Test document | Fields correct |
  |---|---|
  | Form with wrapped names and two- or three-line addresses | 436 of 450 |
  | Typed agreement, values inside sentences | 435 of 440 |
  | Typeset contract: hyphenated line breaks, values ending at punctuation, amounts and dates in words | 500 of 520 |
  | Agreement with words deliberately run together next to labels and end texts | 399 of 400 |

  Almost every miss was a character misread by the OCR inside a correctly found value: `P.O. BOX`
  read as `P.0. B0X`, a dropped full stop, or a dropped space after a comma. These come back with
  a confidence near 0.88 and usually no warning. Five were hyphenated words where the page gave no
  evidence and the guess was wrong; those carry a warning. None started or ended at the wrong
  label. These pages are easier than real agreements: clean type, and YAML written knowing every
  wording used.
- **Real documents.** 24 public documents, 188 pages: 14 scanned US Treasury agreements with
  printed labels and typed values (with stamps, redactions and handwriting on them), and 10
  digital NASA agreements in two templates. YAML was written as a user would, from the first
  document of each kind, then corrected once after seeing the failures. Of 280 fields:

  | | Correct | Missing | Wrong, with a warning | Wrong, no warning |
  |---|---|---|---|---|
  | YAML from the first document only | 224 | 29 | 3 | 24 |
  | YAML corrected, before the fixes this evaluation prompted | 250 | 5 | 3 | 22 |
  | YAML corrected, current code | 269 | 7 | 3 | 1 |

  Most early failures were wording: a sibling document printing a label differently, which an
  alias fixes. The engine failures it found (wrapped values cut short, times cut at their colon, a
  short label matching inside a title, unparseable numbers returned as text) are fixed. The one
  remaining silent error is an OCR misreading (`Bancshares` read as `Baneshares`) inside a
  correctly found value.
- **One Kenyan national ID photo:** all eight printed fields read correctly, and the two fields
  not on the card reported missing. No real logbooks, driving licences or passports, and no real
  Kenyan agreements, have been tested.

Expect real documents with security backgrounds, glare, creases or steep camera angles to do worse
than the synthetic figure. Measure on your own scans before relying on it.

What it does not do:

- **Handwriting, reliably.** See [Handwriting](#handwriting) below.
- **Barcodes, the machine-readable lines on a passport, and non-Latin scripts.**
- **Agreements worded differently from the YAML.** Sentence values are found by the wording you
  give. Also not handled in sentences: text set in several columns (a sentence value in a
  two-column form usually comes back missing), a value running over three or more pages, and
  number words in languages other than English.
- **OCR mistakes inside a value** of a scanned or photographed page: a dropped space
  (`JOSEPHINEWANJIRU`) or a misread letter (`Baneshares`), with high confidence and no warning.
  Digital PDFs read from their text do not have this problem.
- **A label wrapped over two lines.**
- **Text the YAML does not mention.** A value can pick up neighbouring text that belongs to a
  label the YAML does not list. List that label under `other_labels` to stop it.
- **Bilingual labels** such as `Surname / Nom`.

### Handwriting

The OCR engine was built for print. On handwriting it works partly, and it is not dependable.

| Measured on | Characters wrong | Words exactly right |
|---|---|---|
| Printed text | 0.0% | 99.8% |
| Real handwriting, 400 lines of joined-up English from many writers | 12.9% | 62.5% |
| Real handwriting, 204 short lines or lines with digits | 9.6% | 68.1% |

- **About one handwritten word in three is misread**, so a short handwritten field often comes back
  wrong.
- **The confidence score does not catch it.** Most misread lines still had a confidence of 0.9 or
  more, so "check the low-confidence fields" will miss most handwriting errors. Treat every
  handwritten field as needing a person's check.
- **Handwritten entries are found.** On test forms mixing printed labels and handwritten entries,
  99% of the entries were located. Reading them is the weak part.
- **Block capitals are easier, and still not dependable.** On 1,000 form-style fields composed
  from real hand-printed capitals and digits (the NIST handprinted character set, 728 writers),
  83% were read exactly: digit strings 92%, amounts 89%, names and dates 80%, mixed letter-and-digit
  codes 72%. Through the whole extractor on test forms, 79% of fields were right, 5% missing and
  16% wrong, and most of the wrong ones scored 0.9 or more. The matcher always took the value from
  the right place; the misreading is in the characters. These fields were composed from separate
  characters, so real form entries are probably worse.
- **Mark handwritten fields in the YAML** with `handwritten: true`. Any value found for them then
  carries a warning to check it by eye, however confident the engine is.
- **Not measured:** photographs of real handwritten forms.

Eight other recognition models that the engine can use were measured on the same lines and none
did better. A dedicated handwriting model (TrOCR) read 87 to 89% of the words, but it is 250 MB to
1.3 GB, takes one to four seconds per line on a CPU against 0.04, and needs a second runtime. It is
not included.

Other things to know:

- **The included YAML files describe Kenyan documents.** Their labels, patterns and
  `other_labels` were written from general knowledge of those documents, and only the national ID
  has been checked against a real one. Correct them against your own scans, or write your own
  files for documents from elsewhere.
- **Speed.** See [Performance](#performance).

## Security

The service and the library were attacked deliberately (hostile images and PDFs, oversized and
malformed requests, path tricks in every parameter, slow connections, hostile YAML) and the
findings fixed; `tests/test_security.py` keeps each attack as a regression test.

- **Untrusted input is bounded before it is decoded:** upload size, page count, megapixels, page
  shape, characters per PDF page, length of typed values, number and length of field names.
- **Nothing a request names is read from disk.** A document type is looked up by name only, upload
  file names are ignored, and uploads are never written to disk.
- **Corrupted PDFs** (200 mutated files) were handled without a crash or hang.
- **Use Python 3.11 or newer.** Pillow 13.0.0 fixes a long list of published vulnerabilities and
  needs Python 3.11+; on Python 3.10 the newest available is 12.3.0, which has 42 known
  advisories. dococr installs and runs on 3.10, and limits the exposure there by decoding only
  six image formats, but it is developed and tested on Python 3.12, where a scan of all installed
  packages finds no known vulnerabilities. `requirements.lock` records those exact versions.
- **YAML files are trusted input.** Do not load document type files from people you do not trust.

## Performance

About one second per image once the models are loaded, measured on a 12-CPU machine that was busy
with other work at the time. Loading the models takes about three seconds, once per process.

| What | Detail |
|---|---|
| Per image, models loaded | 0.9 to 1.4 s across five synthetic pages and one real ID photo of about 1000 px. Expect larger photos to take longer. |
| First image in a process | About 3 s more, for loading the models. `dococr.warm_up()` or the HTTP service's start-up does this ahead of time. |
| `dococr extract` from the shell | Pays the model load on every run, so 2.5 to 4 s for one image. For many images, pass them all to one command or use the HTTP service. |
| A page of a digital PDF read from its text | About 0.15 s, and no model load at all when no page needs OCR. A two-page agreement took 0.7 s from the shell. |
| Looking up a document type | Well under a millisecond. Parsed YAML files are cached and re-checked against the file system on every call, so an edited file still takes effect immediately. |

Settings:

| Setting | Default | Meaning |
|---|---|---|
| `DOCOCR_OCR_THREADS` | `4` | CPU threads the OCR uses per image. More is not faster for these models: on the test machine 12 threads took about three times longer than 4. With several worker processes, each one uses this many threads. |
| `dococr.warm_up()` | | Loads the models now, not on the first image. Safe to call more than once. |
| `dococr serve --no-warm-up`, or `DOCOCR_WARMUP=0` | warm-up on | The service loads the models before it accepts requests, and refuses to start if it cannot. Turn that off to start instantly and load on the first request instead. |

`tests/local/bench.py` prints where the time goes, stage by stage.

## Development

```bash
python3 -m venv .venv
.venv/bin/pip install -e ".[all,dev]"

.venv/bin/pytest -m "not slow"                       # seconds; no OCR model needed
.venv/bin/pytest                                     # everything, including the real OCR model
.venv/bin/python tests/local/eval_synth.py --seeds 3 # accuracy report on synthetic documents
.venv/bin/python tests/local/bench.py                # per-stage timings
```

Never commit real document scans. Put them in `samples/`, which git ignores.
