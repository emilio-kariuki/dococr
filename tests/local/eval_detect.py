"""Document type detection on the synthetic set (real OCR model): confusion matrix and refusals.

    .venv/bin/python tests/local/eval_detect.py [--seeds 3] [--types logbook,synth_form] [--no-ident]

Every page is read once (`Extractor.read_page`) and then detected twice:

1. among every type at once: the shipped types and the fixture types (`synth.FIXTURE_TYPES`,
   YAML in tests/doctypes/). Each page should come back as its own type.
2. fixture pages among the shipped types only. None of them is a shipped type, so each should
   come back unknown.

`--no-ident` drops `identifying_text` and `title` from every type, to measure the labels alone.
The synthetic pages print some of the shipped types' identifying text (their headings), so the
default figure flatters those types; the labels-only figure does not.
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from synth import DOC_TYPES, FIXTURE_TYPES, doc_type_ref, make_document, variants_for

from dococr import get_extractor, list_doc_types, load_doc_type
from dococr.detection import detect_lines

UNKNOWN = "(unknown)"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--types", default=",".join(DOC_TYPES + FIXTURE_TYPES))
    ap.add_argument("--no-ident", action="store_true", help="labels only: no identifying text or title")
    args = ap.parse_args()

    def load(ref):
        d = load_doc_type(ref)
        if args.no_ident:
            d.identifying_text = []
            d.title = d.name
        return d

    shipped = [load(n) for n in list_doc_types() if n not in FIXTURE_TYPES]
    everything = shipped + [load(doc_type_ref(n)) for n in FIXTURE_TYPES]
    extractor = get_extractor()
    conf: dict[str, Counter] = defaultdict(Counter)
    refused: Counter = Counter()
    wrongly_known: list = []
    margins = []
    read_s = match_s = 0.0
    n_pages = 0
    for dtp in args.types.split(","):
        for var in variants_for(dtp):
            for seed in range(args.seeds):
                doc = make_document(dtp, var, seed)
                t = time.perf_counter()
                page = extractor.read_page(doc.image)
                read_s += time.perf_counter() - t
                t = time.perf_counter()
                d = detect_lines([page], everything)
                match_s += time.perf_counter() - t
                n_pages += 1
                conf[dtp][d.doc_type or UNKNOWN] += 1
                if d.doc_type != dtp:
                    print(f"  {dtp}/{var}/{seed}: {d.doc_type or UNKNOWN} ({d.reason})")
                true = next((c.score for c in d.candidates if c.doc_type == dtp), 0.0)
                other = next((c.score for c in d.candidates if c.doc_type != dtp), 0.0)
                margins.append((true, other, dtp, var, seed))
                if dtp in FIXTURE_TYPES:
                    s = detect_lines([page], shipped)
                    refused[s.doc_type or UNKNOWN] += 1
                    if s.doc_type:
                        wrongly_known.append((dtp, var, seed, s.doc_type, s.reason))
    types = list(conf)
    cols = types + [UNKNOWN]
    print(f"\nconfusion, {len(everything)} candidates (rows: printed type, columns: detected)")
    print(f"{'':18}" + "".join(f"{c[:10]:>11}" for c in cols))
    ok = unknown = 0
    for t in types:
        print(f"{t:18}" + "".join(f"{conf[t][c] or '.':>11}" for c in cols))
        ok += conf[t][t]
        unknown += conf[t][UNKNOWN]
    total = sum(sum(c.values()) for c in conf.values())
    print(f"correct {ok}/{total}, unknown {unknown}, wrong {total - ok - unknown}")
    print(f"lowest score of the right type: {min(m[0] for m in margins):.3f}; "
          f"highest score of a wrong type: {max(m[1] for m in margins):.3f}; "
          f"highest wrong/right ratio: {max(m[1] / m[0] if m[0] else 9 for m in margins):.2f}")
    if refused:
        n = sum(refused.values())
        print(f"\nfixture pages among the {len(shipped)} shipped types: {refused[UNKNOWN]}/{n} unknown")
        for w in wrongly_known:
            print("   wrongly detected:", w)
    print(f"\nper page: reading {1000 * read_s / n_pages:.0f} ms, detection among {len(everything)} types "
          f"{1000 * match_s / n_pages:.0f} ms ({n_pages} pages)")


if __name__ == "__main__":
    main()
