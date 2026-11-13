"""Multi-page PDF fixtures through the real OCR model: pages the text layer cannot serve, values
across page breaks read from OCR lines, and the hybrid page. Slow."""

from __future__ import annotations

import pytest
import synth_pdf

from dococr import extract

pytestmark = pytest.mark.slow

SEED = 1


@pytest.fixture(scope="module")
def doc():
    return synth_pdf.make_pdf_document(SEED)


def run(doc, form, **kw):
    res = extract(synth_pdf.build(doc, form), synth_pdf.doc_type_ref(), **kw)
    wrong = {n: (res.fields[n].value, v) for n, v in doc.truth.items() if res.fields[n].value != v}
    return res, wrong


def test_image_only_pdf_reads_values_across_page_breaks(doc):
    res, wrong = run(doc, "image")
    assert len(wrong) <= 1, wrong
    assert not (set(wrong) & doc.crossing), wrong
    spanning = [n for n in doc.crossing if res.fields[n].parts]
    assert spanning and all(len(res.fields[n].parts) == 2 for n in spanning)
    assert not any("text layer" in w for w in res.warnings)  # image-only: no note


def test_invisible_ocr_layer_is_ignored_and_the_page_read_by_ocr(doc):
    res, wrong = run(doc, "ocr_layer")
    assert len(wrong) <= 1, wrong
    assert all("invisible" in w for w in res.warnings if "text layer" in w)
    assert sum("invisible" in w for w in res.warnings) == res.pages


@pytest.mark.parametrize("form", ["pua", "nomap", "sideways"])
def test_untrusted_text_layers_fall_back_to_ocr(doc, form):
    res, wrong = run(doc, form)
    assert len(wrong) <= 1, wrong
    assert any("not used" in w for w in res.warnings)


def test_mixed_page_reads_the_pasted_paragraph_by_ocr(doc):
    res, wrong = run(doc, "mixed")
    assert wrong == {}, wrong
    last = res.pages
    assert any(w.startswith(f"page {last}: read from the PDF's text layer, and by OCR") for w in res.warnings)
    assert res.fields["witness_name"].page == last


def test_text_layer_and_ocr_agree_on_the_digital_pdf(doc):
    by_text, wrong_text = run(doc, "digital")
    by_ocr, wrong_ocr = run(doc, "digital", pdf_text="never")
    assert wrong_text == {}
    assert len(wrong_ocr) <= 1, wrong_ocr
    assert all(f.confidence == pytest.approx(1.0, abs=0.11) for f in by_text.fields.values() if f.value)


def test_multiline_block_split_between_two_photographed_pages():
    """A form photographed as two pages, cut between the first and second line of the address
    block, each page with a page number: the block is read across the break."""
    import random

    import synth
    from PIL import Image, ImageDraw

    seed = 0
    form = synth.make_document("synth_form", "clean", seed)
    whole = extract(form.image, synth.doc_type_ref("synth_form"), ["postal_address"])
    l, t, r, b = whole.fields["postal_address"].bbox
    _, items, _ = synth.items_for("synth_form", random.Random(f"synth_form/{seed}"))
    n = len(next(it for it in items if it.field == "postal_address").lines)
    cut = int(t + (b - t) / n)
    W, H = form.image.size
    bg = form.image.getpixel((W - 20, H - 20))

    def page(img, number):
        out = Image.new("RGB", (W, img.height + 260), bg)
        out.paste(img, (0, 0 if number == 1 else 120))
        ImageDraw.Draw(out).text((W // 2 - 60, out.height - 60), f"Page {number} of 2",
                                 font=synth.font("sans", 22), fill=(0, 0, 0))
        return out

    pages = [page(form.image.crop((0, 0, W, cut)), 1), page(form.image.crop((0, cut, W, H)), 2)]
    res = extract(pages, synth.doc_type_ref("synth_form"), list(form.truth))
    f = res.fields["postal_address"]
    # the OCR may read "P.O." as "P.0." (a known look-alike error, not this feature's)
    assert f.value.replace("0.", "O.") == form.truth["postal_address"]
    assert [p.page for p in f.parts] == [1, 2]
    assert any("next page" in w for w in f.warnings)
    others = {k: v for k, v in form.truth.items() if k != "postal_address"}
    assert {k: res.fields[k].value for k in others} == others
