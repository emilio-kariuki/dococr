"""A fake reader for the batch tests: what it finds on a page depends on the page's size, so a
test can make a document that crashes the process, hangs, or carries a formula, with Pillow only.
It is a module of its own because worker processes start fresh and import it by name."""

from __future__ import annotations

import os
import time

from dococr.extractors.base import ExtractorError, PageRead
from dococr.extractors.local import LocalExtractor
from dococr.ocr import OcrLine

PERMIT = """name: permit
title: Pastoral grazing permit
identifying_text: ["Pastoral Grazing Permit"]
fields:
  - {name: permit_code, label: Permit Code, type: identifier}
  - {name: holder_name, label: Holder Name, type: name}
  - {name: grazing_zone, label: Grazing Zone}
  - {name: herd_size, label: Herd Size, type: integer}
"""
TICKET = """name: ticket
title: Stadium ticket
fields:
  - {name: ticket_serial, label: Ticket Serial}
  - {name: holder_name, label: Holder Name, type: name}
  - {name: seat_row, label: Seat Row}
  - {name: gate_letter, label: Gate Letter}
"""

# Page widths with a meaning (see FakeReader.read_page). Any other width is a permit page.
OTHER = 301  # a page that is none of the types
CRASH = 77  # kills the process, like a fault in the native OCR engine
HANG = 78  # never returns
ENGINE = 79  # the engine reports an error
FORMULA = 80  # a permit whose zone reads as a spreadsheet formula (the matcher strips a leading = or -, but not @ or +)

SLEEP_HEIGHT = 1000  # a page this tall waits a second first (to make completion order differ)


def line(y, x, text):
    return OcrLine(text, 0.95, (x, y, x + 13 * len(text), y + 24))


def permit_page(code, zone="NORTH 4", herd="37"):
    return [line(20, 300, "PASTORAL GRAZING PERMIT"),
            line(80, 40, "PERMIT CODE"), line(80, 420, code),
            line(130, 40, "HOLDER NAME"), line(130, 420, "ZANELE OKAFOR"),
            line(180, 40, "GRAZING ZONE"), line(180, 420, zone),
            line(230, 40, "HERD SIZE"), line(230, 420, herd)]


class FakeReader(LocalExtractor):
    name = "fake-reader"

    def __init__(self):
        super().__init__(engine=object())

    def warm_up(self):
        pass

    def read_page(self, image, *, text_layer=None):
        w, h = image.size
        if w == CRASH:
            os._exit(7)
        if w == HANG:
            time.sleep(120)
        if w == ENGINE:
            raise ExtractorError("the fake engine failed")
        if h >= SLEEP_HEIGHT:
            time.sleep(1.0)
        if w == OTHER:
            lines = [line(80, 40, "INVOICE NO"), line(80, 420, "88"),
                     line(130, 40, "AMOUNT DUE"), line(130, 420, "1,200.00")]
        elif w == FORMULA:
            lines = permit_page(f"PX-{h}", zone="@SUM(1)")
        else:
            lines = permit_page(f"PX-{h}")
        return PageRead(lines, (1000, 800))


def make():
    """The `extractor_factory` of the tests."""
    return FakeReader()


def fail_to_start():
    raise RuntimeError("no engine here")


def die_at_start():
    os._exit(9)
