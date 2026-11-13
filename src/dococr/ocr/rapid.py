"""RapidOCR 3.x (ONNX Runtime, CPU) behind the OcrEngine interface."""

from __future__ import annotations

import os
import threading
from pathlib import Path
from typing import Any, Optional

import numpy as np

from dococr.extractors.base import ExtractorError
from dococr.ocr import OcrEngine, OcrLine, Quad


# The model files that ship inside the rapidocr wheel. Pointing at them explicitly matters:
# given no path, RapidOCR resolves a model by name and DOWNLOADS it when the file is absent.
# With explicit paths a missing file is an error, never a network request.
_BUNDLED_MODELS = {
    "Det.model_path": "PP-OCRv6_det_small.onnx",
    "Rec.model_path": "PP-OCRv6_rec_small.onnx",
    "Cls.model_path": "ch_ppocr_mobile_v2.0_cls_mobile.onnx",
}


def _bundled_model_paths() -> dict[str, str]:
    import rapidocr

    folder = Path(rapidocr.__file__).resolve().parent / "models"
    paths = {key: folder / name for key, name in _BUNDLED_MODELS.items()}
    missing = [p.name for p in paths.values() if not p.is_file()]
    if missing:
        raise ExtractorError(
            f"OCR model file(s) missing from {folder}: {', '.join(missing)}. dococr never "
            "downloads models; reinstall the pinned rapidocr version, which ships them."
        )
    return {key: str(path) for key, path in paths.items()}


# ONNX Runtime's own default is one thread per CPU. These models are small and do not use
# that many: measured on a 12-CPU machine shared with other work, 12 threads took about four
# times longer per image than 2-4 threads, and burned sixteen times the CPU, because the idle
# threads spin while they wait. A small fixed number is faster and leaves the other CPUs to
# other worker processes.
DEFAULT_THREADS = 4


def _thread_count() -> int:
    cpus = os.cpu_count() or 1
    wanted = os.environ.get("DOCOCR_OCR_THREADS", "")
    threads = int(wanted) if wanted.isdigit() and int(wanted) >= 1 else DEFAULT_THREADS
    return max(1, min(threads, cpus))


class RapidOcrEngine(OcrEngine):
    """Wraps `rapidocr.RapidOCR`. The ONNX models are loaded on first use and then reused.

    Runs offline: the models are read from the installed rapidocr package and nothing is fetched.

    Environment:
      DOCOCR_OCR_THREADS  CPU threads ONNX Runtime uses per call (default: DEFAULT_THREADS,
                          capped at the number of CPUs).
    """

    def __init__(self, params: Optional[dict[str, Any]] = None):
        try:
            from rapidocr import RapidOCR  # noqa: F401
        except ImportError as exc:  # pragma: no cover - depends on the environment
            raise ExtractorError(
                "The OCR engine (rapidocr) is not installed. Reinstall dococr."
            ) from exc
        self._params = {
            **_bundled_model_paths(),
            "Global.log_level": "error",
            # Page orientation is fixed in preprocessing; per-line 180-degree classification
            # is not needed for the main pass and costs time.
            "Global.use_cls": False,
            **(params or {}),
        }
        self._params.setdefault("EngineConfig.onnxruntime.intra_op_num_threads", _thread_count())
        self._ocr = None
        self._lock = threading.Lock()  # guards the one-time model load
        # RapidOCR stores each call's use_det/use_cls/use_rec flags on the shared instance, and
        # this class makes detect-only, classify-only and full-read calls. Two threads inside it
        # at once would overwrite each other's flags, so calls are serialised.
        self._call_lock = threading.Lock()

    def _engine(self):
        if self._ocr is None:
            with self._lock:
                if self._ocr is None:
                    try:
                        from rapidocr import RapidOCR

                        self._ocr = RapidOCR(params=self._params)
                    except Exception as exc:
                        raise ExtractorError(f"Could not start RapidOCR: {exc}") from exc
        return self._ocr

    def _call(self, image: np.ndarray, **kwargs):
        try:
            engine = self._engine()
            with self._call_lock:
                return engine(np.ascontiguousarray(image), **kwargs)
        except ExtractorError:
            raise
        except Exception as exc:
            # ONNX Runtime errors surface here.
            raise ExtractorError(f"RapidOCR failed: {type(exc).__name__}: {exc}") from exc

    def _guarded(self, fn):
        """Run `fn(rapidocr_instance)` holding `_call_lock`, with the same error mapping as
        `_call`. Every use of a RapidOCR object or one of its components goes through here or
        `_call`."""
        try:
            engine = self._engine()
            with self._call_lock:
                return fn(engine)
        except ExtractorError:
            raise
        except Exception as exc:
            raise ExtractorError(f"RapidOCR failed: {type(exc).__name__}: {exc}") from exc

    def read(self, image: np.ndarray) -> list[OcrLine]:
        out = self._call(image, use_det=True, use_cls=False, use_rec=True)
        boxes = getattr(out, "boxes", None)
        txts = getattr(out, "txts", None)
        scores = getattr(out, "scores", None)
        if boxes is None or txts is None or len(txts) == 0:
            return []
        return _to_lines(boxes, txts, scores)

    def detect(self, image: np.ndarray) -> list[Quad]:
        out = self._call(image, use_det=True, use_cls=False, use_rec=False)
        boxes = getattr(out, "boxes", None)
        if boxes is None:
            return []
        return [np.asarray(b, dtype=float).tolist() for b in boxes]

    def detector_resolution(self, width: int, height: int) -> float:
        """RapidOCR first caps the long side at `Global.max_side_len`, then `DetPreProcess`
        resizes so the short side is at least `Det.limit_side_len` (`limit_type: min`, the
        default) or the long side at most that (`max`). Rounding to multiples of 32 is ignored."""

        def run(ocr):
            g, d = ocr.cfg.Global, ocr.cfg.Det
            factor = 1.0
            if g.use_preprocess_img and max(width, height) > ocr.max_side_len:
                factor = ocr.max_side_len / max(width, height)
            w, h = width * factor, height * factor
            limit = float(d.limit_side_len)
            if str(d.limit_type) == "max":
                return factor * (limit / max(w, h) if max(w, h) > limit else 1.0)
            return factor * (limit / min(w, h) if min(w, h) < limit else 1.0)

        return self._guarded(run)

    def recognize(self, image: np.ndarray, quads: list[Quad]) -> list[OcrLine]:
        """Recognise the lines at `quads` without running the detector again.

        Reproduces what `RapidOCR.__call__` does between detection and recognition (RapidOCR
        3.10, `main.py`): corner order and clipping as in `DBPostProcess.filter_det_res`, the
        reading order of `TextDetector.sorted_boxes`, perspective crops with tall crops turned
        by `get_rotate_crop_image`, one batched `TextRecognizer` call, then the removal of empty
        texts and of lines under `Global.text_score`. Given the quads `detect` returned for the
        same image, the result equals `read(image)`.

        When RapidOCR would first resize or pad this image (long side over `max_side_len`, short
        side under `min_side_len`, or a strip that triggers its vertical padding) its crops come
        from the altered image, so this falls back to a full `read`.
        """
        image = np.ascontiguousarray(image)
        h, w = image.shape[:2]
        if not len(quads):
            return []
        if self._guarded(lambda ocr: _rapidocr_alters(ocr, h, w)):
            return self.read(image)

        boxes = _order_and_clip(quads, w, h)
        if not boxes:
            return []
        from rapidocr.ch_ppocr_det import TextDetector
        from rapidocr.ch_ppocr_rec import TextRecInput
        from rapidocr.utils.process_img import get_rotate_crop_image

        boxes = TextDetector.sorted_boxes(np.array(boxes, dtype=np.float32))
        crops = [get_rotate_crop_image(image, box.copy()) for box in boxes]

        def run(ocr):
            rec = ocr._load_rec_model()(TextRecInput(img=crops, return_word_box=False))
            return rec.txts, rec.scores, float(ocr.text_score)

        txts, scores, text_score = self._guarded(run)
        kept = [(b, t, s) for b, t, s in zip(boxes, txts, scores) if t.strip() and s >= text_score]
        if not kept:
            return []
        return _to_lines(*zip(*kept))

    def classify_upside_down(self, crops: list[np.ndarray]) -> Optional[float]:
        crops = [np.ascontiguousarray(c) for c in crops if c.size]
        if not crops:
            return None

        def run(ocr):
            from rapidocr.utils.process_img import resize_image_within_bounds

            # The same per-crop resize RapidOCR's own call applies before classifying, so the
            # batched result equals one call per crop.
            sized = [resize_image_within_bounds(c, ocr.min_side_len, ocr.max_side_len)[0]
                     if ocr.cfg.Global.use_preprocess_img else c for c in crops]
            return ocr._load_cls_model()(sized).cls_res

        flipped = total = 0.0
        for label, score in self._guarded(run) or []:
            total += float(score)
            if str(label) == "180":
                flipped += float(score)
        return None if total == 0 else flipped / total

    def warm_up(self) -> None:
        """Load the detector, classifier and recogniser and run one tiny inference through
        each, so the first image does not pay for it."""
        from rapidocr.ch_ppocr_rec import TextRecInput

        page = np.full((64, 256, 3), 255, dtype=np.uint8)
        page[24:40, 16:240] = 0
        line = np.full((48, 192, 3), 255, dtype=np.uint8)

        def run(ocr):
            ocr._load_det_model()(page)
            ocr._load_cls_model()([line])
            ocr._load_rec_model()(TextRecInput(img=[line], return_word_box=False))

        self._guarded(run)


def _to_lines(boxes, txts, scores) -> list[OcrLine]:
    lines = []
    for quad, text, score in zip(boxes, txts, scores):
        quad = np.asarray(quad, dtype=float)
        text = str(text).strip()
        if not text:
            continue
        box = (float(quad[:, 0].min()), float(quad[:, 1].min()),
               float(quad[:, 0].max()), float(quad[:, 1].max()))
        lines.append(OcrLine(text=text, confidence=float(score), box=box))
    return lines


def _rapidocr_alters(ocr, h: int, w: int) -> bool:
    """Would `RapidOCR.__call__` resize (`preprocess_img`) or pad (`apply_vertical_padding`)
    an image of this size before detecting and cropping?"""
    g = ocr.cfg.Global
    if g.use_preprocess_img and (max(h, w) > ocr.max_side_len or min(h, w) < ocr.min_side_len):
        return True
    if g.use_vertical_padding:
        ratio = ocr.width_height_ratio
        if h <= ocr.min_height or (ratio != -1 and w / h > ratio):
            return True
    return False


def _order_points(pts: np.ndarray) -> np.ndarray:
    """Top-left, top-right, bottom-right, bottom-left, as `DBPostProcess.order_points_clockwise`."""
    x_sorted = pts[np.argsort(pts[:, 0]), :]
    left, right = x_sorted[:2, :], x_sorted[2:, :]
    tl, bl = left[np.argsort(left[:, 1]), :]
    tr, br = right[np.argsort(right[:, 1]), :]
    return np.array([tl, tr, br, bl], dtype=np.float32)


def _order_and_clip(quads: list[Quad], w: int, h: int) -> list[np.ndarray]:
    """Corner order, clipping to the image and the size filter of `DBPostProcess.filter_det_res`.
    Idempotent on quads that already went through it."""
    out = []
    for q in quads:
        box = _order_points(np.asarray(q, dtype=np.float32).reshape(4, 2))
        box[:, 0] = np.clip(box[:, 0], 0, w - 1).astype(int)
        box[:, 1] = np.clip(box[:, 1], 0, h - 1).astype(int)
        if int(np.linalg.norm(box[0] - box[1])) <= 3 or int(np.linalg.norm(box[0] - box[3])) <= 3:
            continue
        out.append(box)
    return out
