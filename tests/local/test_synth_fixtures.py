"""The multi-line and prose fixture types: their YAML matches what tests/synth.py prints, and
they are test fixtures, not shipped presets. Fast: renders pages, no OCR model."""

from __future__ import annotations

import pytest

from dococr import load_doc_type
from dococr.doctypes import find_doc_types
from synth import FIXTURE_TYPES, doc_type_ref, make_document, variants_for


def test_fixture_types_are_not_shipped():
    shipped = {d.name for d in find_doc_types()[0]}
    for name in FIXTURE_TYPES:
        assert name not in shipped
        assert load_doc_type(name).fields == []  # not found by name, only by path


@pytest.mark.parametrize("seed", range(5))
def test_form_labels_printed_are_in_the_yaml(seed):
    doc = make_document("synth_form", "clean", seed)
    yaml = load_doc_type(doc_type_ref("synth_form"))
    assert set(doc.truth) <= {f.name for f in yaml.fields}
    for name, label in doc.labels.items():
        assert label.lower() in [x.lower() for x in yaml.field(name).labels()], (name, label)
    assert yaml.field("full_name").multiline and yaml.field("postal_address").multiline


def test_form_has_wrapped_and_multi_line_values():
    lines = {len(make_document("synth_form", "clean", s).printed["postal_address"].split()) for s in range(5)}
    assert max(lines) > 3  # several words, printed on two or three lines


@pytest.mark.parametrize("seed", range(5))
def test_agreement_truth_and_yaml_agree(seed):
    doc = make_document("synth_agreement", "clean", seed)
    yaml = load_doc_type(doc_type_ref("synth_agreement"))
    assert set(doc.truth) == {f.name for f in yaml.fields}
    assert all(f.until_texts() for f in yaml.fields)


def test_every_variant_renders():
    for name in FIXTURE_TYPES:
        for variant in variants_for(name):
            doc = make_document(name, variant, 0)
            assert doc.image.width > 300 and doc.truth


@pytest.mark.parametrize("seed", range(5))
def test_contract_truth_and_yaml_agree(seed):
    doc = make_document("synth_contract", "clean", seed)
    yaml = load_doc_type(doc_type_ref("synth_contract"))
    assert set(doc.truth) == {f.name for f in yaml.fields}
    assert all(f.until_texts() for f in yaml.fields)
    assert {t for f in yaml.fields for t in f.until_texts()} >= {",", ".", ")"}


def test_contract_pages_are_typeset_with_hyphenated_line_breaks():
    breaks = [b for s in range(5) for v in ("clean", "narrow") for b in make_document("synth_contract", v, s).breaks]
    assert len(breaks) > 20
    assert all(head.endswith("-") for head, _ in breaks)
    assert any(head[:-1].isupper() for head, _ in breaks)  # capitals are hyphenated too
    assert any("-" in head[:-1] or head.count("-") == 1 and head[-2].isupper() for head, _ in breaks)


def test_number_words_round_trip_through_the_normaliser():
    from dococr.models import FieldSpec
    from dococr.normalize import normalize_value
    from synth import num_words
    spec = FieldSpec(name="n", type="integer")
    for n in [5, 19, 20, 25, 100, 105, 999, 1000, 1005, 25000, 95000, 200000, 850000, 1250000, 2575000,
              2999995, 2024]:
        assert normalize_value(num_words(n), spec) == (n, []), n
        assert normalize_value(num_words(n).upper(), spec) == (n, []), n


@pytest.mark.parametrize("seed", range(5))
def test_glued_truth_and_yaml_agree(seed):
    doc = make_document("synth_glued", "clean", seed)
    yaml = load_doc_type(doc_type_ref("synth_glued"))
    assert set(doc.truth) == {f.name for f in yaml.fields}
    assert all(f.until_texts() for f in yaml.fields)


def test_glued_pages_leave_out_spaces_at_every_place():
    places = [p for s in range(10) for p in make_document("synth_glued", "clean", s).glued]
    assert set(places) == {"date|BETWEEN", "vendor|of", "Box|digits", "and|purchaser", "Number|parcel",
                           "parcel|situated", "in|county", "The|purchase", "amount|(", "KES|figures",
                           "deposit|only", "the|day", "of|firm"}


@pytest.mark.parametrize("seed", range(5))
def test_ruled_labels_printed_are_in_the_yaml(seed):
    doc = make_document("synth_ruled", "clean", seed)
    yaml = load_doc_type(doc_type_ref("synth_ruled"))
    assert set(doc.truth) == {f.name for f in yaml.fields}
    for name, label in doc.labels.items():
        assert label.lower() in [x.lower() for x in yaml.field(name).labels()], (name, label)


def test_ruled_form_has_stacked_labels_every_value_place_and_struck_entries():
    from synth import ruled_for
    import random
    cells = [c for s in range(20) for row in ruled_for(random.Random(f"synth_ruled/{s}"))[0] for c in row]
    assert {c.place for c in cells if len(c.label) > 1} == {"top", "middle", "bottom", "joined", "below"}
    assert any(len(c.label) == 3 for c in cells)
    assert any(c.expected is None and set(c.value) == {"X"} for c in cells)
