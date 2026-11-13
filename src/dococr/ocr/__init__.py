"""OCR engine interface used by the offline extractor.

An engine turns an RGB numpy image into text lines with a confidence and an axis-aligned box
in that image's pixel coordinates. The default engine is RapidOCR on ONNX Runtime
(`dococr.ocr.rapid`), loaded lazily and reused.
"""

from __future__ import annotations

import threading
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Optional, Sequence

import numpy as np

Box = tuple[float, float, float, float]  # left, top, right, bottom
Quad = Sequence[Sequence[float]]  # four (x, y) corners as returned by the detector


@dataclass
class OcrLine:
    text: str
    confidence: float
    box: Box

    @property
    def height(self) -> float:
        return self.box[3] - self.box[1]


class OcrEngine(ABC):
    @abstractmethod
    def read(self, image: np.ndarray) -> list[OcrLine]:
        """Detect and recognise all text lines in an RGB uint8 image."""

    def detect(self, image: np.ndarray) -> list[Quad]:
        """Text-line quads only (no recognition). Used to estimate page orientation/skew."""
        return []

    def detector_resolution(self, width: int, height: int) -> float:
        """The factor by which the detector resizes an image of this size before it runs
        (1.0: at its own resolution). Lets preprocessing tell whether detecting again on a
        rescaled page would see the text any finer."""
        return 1.0

    def recognize(self, image: np.ndarray, quads: list[Quad]) -> list[OcrLine]:
        """Recognise the text lines at `quads` (corners in `image`'s pixel coordinates, text
        upright in `image`) without detecting again. Same output contract as `read`.

        The default ignores the quads and runs a full `read`, for engines that cannot recognise
        given regions."""
        return self.read(image)

    def warm_up(self) -> None:
        """Load models and run a first inference ahead of the first image. Default: nothing."""

    def classify_upside_down(self, crops: list[np.ndarray]) -> Optional[float]:
        """Fraction (0..1, confidence-weighted) of crops that look rotated by 180 degrees.
        None when the engine has no orientation classifier."""
        return None


_default: Optional[OcrEngine] = None
_default_lock = threading.Lock()


def get_default_engine() -> OcrEngine:
    """The shared RapidOCR engine (model loaded once per process)."""
    global _default
    with _default_lock:
        if _default is None:
            from dococr.ocr.rapid import RapidOcrEngine

            _default = RapidOcrEngine()
        return _default


__all__ = ["Box", "OcrEngine", "OcrLine", "Quad", "get_default_engine"]
