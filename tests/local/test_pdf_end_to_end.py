"""A PDF through the real OCR model: two pages of different content, one merged result. Slow."""

from __future__ import annotations

import io

import pytest

from dococr import DocType, extract, load_doc_type
from synth import make_document

pytestmark = pytest.mark.slow


def _combined_type() -> DocType:
    """One type holding the fields of two shipped types, as a mixed bundle of pages would need."""
    fields, seen = [], set()
    for name in ("driving_license", "logbook"):
        for spec in load_doc_type(name).fields:
            if spec.name not in seen:
                seen.add(spec.name)
                fields.append(spec)
    return DocType(name="bundle", fields=fields)


def _pdf(images) -> bytes:
    buf = io.BytesIO()
    images[0].save(buf, "PDF", save_all=True, append_images=images[1:], resolution=150)
    return buf.getvalue()


def test_two_page_pdf_gives_fields_from_both_pages():
    first = make_document("driving_license", "clean", 0)
    second = make_document("logbook", "clean", 0)
    wanted = ["license_number", "date_of_expiry", "registration_number", "make"]
    res = extract(_pdf([first.image, second.image]), _combined_type(), wanted)

    assert res.pages == 2
    for name in ("license_number", "date_of_expiry"):
        assert res.fields[name].value == first.truth[name], name
        assert res.fields[name].page == 1
    for name in ("registration_number", "make"):
        assert res.fields[name].value == second.truth[name], name
        assert res.fields[name].page == 2
    for name in wanted:
        l, t, r, b = res.fields[name].bbox
        # the page is rendered at a different size than the source image: the box is in the
        # coordinates of the rendered page, so only check it is a sensible box
        assert 0 <= l < r and 0 <= t < b, name
        assert r <= 2001 and b <= 2001, name
    assert res.missing == []


def test_pdf_path_and_list_of_images_agree(tmp_path):
    first = make_document("driving_license", "clean", 1)
    second = make_document("logbook", "clean", 1)
    wanted = ["license_number", "registration_number"]
    p = tmp_path / "scan.pdf"
    p.write_bytes(_pdf([first.image, second.image]))
    from_pdf = extract(p, _combined_type(), wanted)
    from_list = extract([first.image, second.image], _combined_type(), wanted)
    assert from_pdf.values() == from_list.values() == {
        "license_number": first.truth["license_number"],
        "registration_number": second.truth["registration_number"],
    }
    assert from_pdf.pages == from_list.pages == 2
