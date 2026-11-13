"""Accuracy and speed on the multi-page PDF fixtures (`tests/synth_pdf.py`), text layer against
OCR on the same pages (real OCR model where a page is OCR'd).

    .venv/bin/python tests/local/eval_pdf.py [--seeds 5] [--runs digital:auto,digital:never]

Each run is FORM:PDF_TEXT. A field is right when its value equals the truth; "crossing" counts
only the fields whose label and end text lie on two pages (`PdfDocument.crossing`); "flagged"
counts right values that carry a warning; time is wall-clock per page.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

import synth_pdf  # noqa: E402

from dococr import extract, get_extractor  # noqa: E402

RUNS = ["digital:auto", "digital:never", "image:auto", "ocr_layer:auto", "ocr_layer:always",
        "pua:auto", "nomap:auto", "rotated:auto", "sideways:auto", "cropped:auto", "mixed:auto"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=5)
    ap.add_argument("--runs", default=",".join(RUNS))
    args = ap.parse_args()
    runs = args.runs.split(",")
    extractor = get_extractor()
    stats = defaultdict(lambda: defaultdict(float))
    failures = []
    for seed in range(args.seeds):
        doc = synth_pdf.make_pdf_document(seed)
        cache: dict[str, bytes] = {}
        for run in runs:
            form, mode = run.split(":")
            data = cache.setdefault(form, synth_pdf.build(doc, form))
            started = time.perf_counter()
            res = extract(data, synth_pdf.doc_type_ref(), extractor=extractor, pdf_text=mode)
            took = time.perf_counter() - started
            s = stats[run]
            s["pages"] += res.pages
            s["time"] += took
            for name, expected in doc.truth.items():
                f = res.fields[name]
                ok = f.value == expected
                s["ok"] += ok
                s["n"] += 1
                s["flagged"] += ok and expected is not None and bool(f.warnings)
                if name in doc.crossing:
                    s["cross_ok"] += ok
                    s["cross_n"] += 1
                if not ok:
                    failures.append((run, seed, name, expected, f.value, f.confidence, f.warnings))
    print(f"{'run':20}{'right':>12}{'crossing':>12}{'flagged':>9}{'s/page':>9}")
    for run in runs:
        s = stats[run]
        print(f"{run:20}{int(s['ok']):>7}/{int(s['n']):<4}{int(s['cross_ok']):>7}/{int(s['cross_n']):<4}"
              f"{int(s['flagged']):>9}{s['time'] / max(s['pages'], 1):>9.2f}")
    print("\nfailures:")
    for f in failures:
        print("  ", f)


if __name__ == "__main__":
    main()
