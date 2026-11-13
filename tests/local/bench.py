"""Where the time goes: per-stage timings of the extraction pipeline (real OCR model).

    .venv/bin/python tests/local/bench.py [--reps 5] [--warm-up] [IMAGE:DOCTYPE ...]

Runs a fixed set of synthetic pages, plus any images you name, and prints for each the best
wall-clock time of `--reps` warm runs, split by stage, and the CPU time of that run. On a
machine that is busy with other work wall-clock time is noisy; CPU time and the best-of-N
figure are the ones to compare between two versions of the code.

Stages: `detect` is the detector pass in preprocessing, `classify` the upside-down check, `read`
a full detect-and-recognise pass on the working image (pages where the first detection is not
reused), `recognize` recognition of the reused quads (the single pass).

With --warm-up the models are loaded by `extractor.warm_up()` first, and that time is reported
separately from the first call.

With --pdf a three-page digital PDF (`tests/synth_pdf.py`, seed 3) is timed twice: read from its
text layer (`pdf_text="auto"`) and by OCR (`"never"`), and as an image-only PDF of the same pages.
`textlayer` is the time spent reading and judging the text layer; `render` is rendering PDF
pages; `per page` divides the wall time by the page count.
"""

from __future__ import annotations

import argparse
import collections
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

SYNTHETIC = [("logbook", "clean"), ("driving_license", "photo"), ("passport", "rot90"),
             ("national_id", "blur_lowres"), ("national_id", "skew")]
# Test-fixture types with multi-line and prose values (`--fixtures`); YAML in tests/doctypes/.
FIXTURES = [("synth_form", "clean"), ("synth_agreement", "clean"), ("synth_agreement", "narrow")]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--reps", type=int, default=5)
    ap.add_argument("--warm-up", action="store_true", help="call extractor.warm_up() first")
    ap.add_argument("--fixtures", action="store_true",
                    help="also time the multi-line form and prose fixture pages")
    ap.add_argument("--pdf", action="store_true",
                    help="also time a digital PDF read from its text layer and by OCR")
    ap.add_argument("images", nargs="*", help="IMAGE:DOCTYPE pairs to time as well")
    args = ap.parse_args()

    started = time.perf_counter()
    import dococr.pipeline as pipeline
    from dococr import extract
    from dococr.extractors import local
    from dococr.ocr import rapid
    from synth import doc_type_ref, make_document

    t_import = time.perf_counter() - started

    stage: dict[str, float] = collections.defaultdict(float)

    def timed(owner, attr, key):
        fn = getattr(owner, attr)

        def wrapper(*a, **k):
            begun = time.perf_counter()
            try:
                return fn(*a, **k)
            finally:
                stage[key] += time.perf_counter() - begun

        setattr(owner, attr, wrapper)

    # Stage names -> where they are measured. Update this list when the pipeline is restructured.
    # Loading and rendering pages is lazy (one page at a time inside the extraction loop), so it
    # is timed where the bytes are actually decoded.
    from dococr import imaging

    from dococr import pdftext

    timed(imaging, "_finish", "image")
    timed(pdftext, "read_text_layer", "textlayer")
    timed(pdftext, "build_layer", "textlayer")
    import pypdfium2

    timed(pypdfium2.PdfPage, "render", "render")
    timed(pipeline, "load_doc_type", "yaml")
    for method in ("detect", "classify_upside_down", "read", "recognize"):
        if hasattr(rapid.RapidOcrEngine, method):
            timed(rapid.RapidOcrEngine, method, {"classify_upside_down": "classify"}.get(method, method))
    timed(local, "match_fields", "match")
    keys = ["image", "render", "textlayer", "yaml", "detect", "classify", "read", "recognize", "match"]

    pages = [(f"{d}/{v}", make_document(d, v, 3).image, d, {}) for d, v in SYNTHETIC]
    if args.fixtures:
        pages += [(f"{d}/{v}", make_document(d, v, 3).image, doc_type_ref(d), {}) for d, v in FIXTURES]
    if args.pdf:
        import synth_pdf

        doc = synth_pdf.make_pdf_document(3)
        digital = synth_pdf.build(doc, "digital")
        pages += [("pdf/text layer", digital, synth_pdf.doc_type_ref(), {"pdf_text": "auto"}),
                  ("pdf/same, by OCR", digital, synth_pdf.doc_type_ref(), {"pdf_text": "never"}),
                  ("pdf/image-only", synth_pdf.build(doc, "image"), synth_pdf.doc_type_ref(), {})]
    for pair in args.images:
        path, _, doc_type = pair.rpartition(":")
        pages.append((Path(path).name[:22], path, doc_type, {}))

    t_warm = None
    if args.warm_up:
        begun = time.perf_counter()
        pipeline.get_extractor().warm_up()
        t_warm = time.perf_counter() - begun

    begun = time.perf_counter()
    extract(pages[0][1], pages[0][2])
    t_first = time.perf_counter() - begun

    warm = f"warm_up={t_warm:.2f}s  first call={t_first:.2f}s" if t_warm is not None else \
        f"first call (loads models)={t_first:.2f}s"
    print(f"cpus={os.cpu_count()} threads={rapid._thread_count()} load={os.getloadavg()[0]:.0f}  "
          f"import={t_import:.2f}s  {warm}")
    print(f"{'page':26}{'wall':>7}{'cpu':>7}{'pages':>6}{'per page':>9}" + "".join(f"{k:>10}" for k in keys))
    total_wall = total_cpu = 0.0
    for name, source, doc_type, kwargs in pages:
        best = None
        for _ in range(args.reps):
            stage.clear()
            wall, cpu = time.perf_counter(), time.process_time()
            res = extract(source, doc_type, **kwargs)
            wall, cpu = time.perf_counter() - wall, time.process_time() - cpu
            if best is None or wall < best[0]:
                best = (wall, cpu, dict(stage), res.pages)
        total_wall += best[0]
        total_cpu += best[1]
        print(f"{name:26}{best[0]:7.2f}{best[1]:7.2f}{best[3]:6d}{best[0] / best[3]:9.2f}"
              + "".join(f"{best[2].get(k, 0):10.2f}" for k in keys))
    print(f"{'total':26}{total_wall:7.2f}{total_cpu:7.2f}")


if __name__ == "__main__":
    main()
