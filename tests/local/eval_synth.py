"""Per-field accuracy of the offline extractor on the synthetic set (real OCR model).

    .venv/bin/python tests/local/eval_synth.py [--seeds 2] [--variants clean,rot90] [--types logbook]
    .venv/bin/python tests/local/eval_synth.py --types synth_form,synth_agreement --seeds 5

The default types are the four shipped ones. The fixture types (`synth.FIXTURE_TYPES`, YAML in
tests/doctypes/) test multi-line and prose values; each type runs its own variants, and a
variant a type does not have is shown as "-".
"""

from __future__ import annotations

import argparse
import sys
import time
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))

from synth import DOC_TYPES, FIXTURE_TYPES, doc_type_ref, make_document, variants_for  # noqa: E402

from dococr import extract, get_extractor  # noqa: E402


def evaluate(doc_types, variants, seeds):
    extractor = get_extractor()
    per_variant = defaultdict(lambda: [0, 0])
    per_field = defaultdict(lambda: [0, 0])
    failures = []
    flagged = []  # right values that carry a warning (fixture types only)
    times = []
    for var in variants:
        for dtp in doc_types:
            if var not in variants_for(dtp):
                continue
            for seed in range(seeds):
                doc = make_document(dtp, var, seed)
                t = time.perf_counter()
                res = extract(doc.image, doc_type_ref(dtp), doc.fields, extractor=extractor)
                times.append(time.perf_counter() - t)
                for name, expected in doc.truth.items():
                    got = res.fields[name].value
                    ok = got == expected
                    per_variant[(var, dtp)][0] += ok
                    per_variant[(var, dtp)][1] += 1
                    per_field[(dtp, name)][0] += ok
                    per_field[(dtp, name)][1] += 1
                    if not ok:
                        failures.append((var, dtp, seed, name, expected, got, res.fields[name].raw,
                                         res.fields[name].confidence, res.fields[name].warnings))
                    elif got is not None and res.fields[name].warnings and dtp in FIXTURE_TYPES:
                        flagged.append((var, dtp, seed, name, got, res.fields[name].warnings))
    return per_variant, per_field, failures, times, flagged


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", type=int, default=1)
    ap.add_argument("--variants", default="", help="default: every variant of the chosen types")
    ap.add_argument("--types", default=",".join(DOC_TYPES), help=f"also: {','.join(FIXTURE_TYPES)}")
    args = ap.parse_args()
    types = args.types.split(",")
    variants = args.variants.split(",") if args.variants else list(
        dict.fromkeys(v for t in types for v in variants_for(t)))
    pv, pf, failures, times, flagged = evaluate(types, variants, args.seeds)
    print(f"{'variant':<12}" + "".join(f"{t:>17}" for t in types) + f"{'all':>10}")
    tot_ok = tot_n = 0
    for var in variants:
        line = f"{var:<12}"
        vo = vn = 0
        for t in types:
            if var not in variants_for(t):
                line += f"{'-':>15}  "
                continue
            ok, n = pv[(var, t)]
            vo += ok
            vn += n
            line += f"{ok:>11}/{n:<3}  "
        tot_ok += vo
        tot_n += vn
        print(line + f"{100 * vo / max(vn, 1):>8.1f}%")
    print(f"overall {tot_ok}/{tot_n} = {100 * tot_ok / max(tot_n, 1):.1f}%")
    print("\nper field (all variants):")
    for (t, name), (ok, n) in sorted(pf.items()):
        if ok < n:
            print(f"  {t:<16} {name:<22} {ok}/{n}")
    print("\nfailures:")
    for f in failures:
        print("  ", f)
    if flagged:
        print(f"\nright values with a warning ({len(flagged)}):")
        for f in flagged:
            print("  ", f)
    if times:
        print(f"\nlatency per image: first {times[0]:.2f}s, median {sorted(times)[len(times) // 2]:.2f}s, "
              f"min {min(times):.2f}s, max {max(times):.2f}s over {len(times)} images")


if __name__ == "__main__":
    main()
