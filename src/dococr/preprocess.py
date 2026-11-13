"""Prepare a page image for OCR and map OCR boxes back to the caller's image.

`prepare` runs the text detector once on a reduced copy of the page and uses the detected
line boxes to decide:
  - quarter-turn rotation (most text lines vertical -> rotate 90 degrees),
  - small skew (median angle of the text lines),
  - upside-down pages (180 degrees; asks the engine's orientation classifier),
  - scale (upscale when the text is small, cap huge images).
It then applies one affine warp and returns the matrix, so every box read off the working
image can be mapped back to the original with `PageTransform.box_to_original`.

The detector's line quads, mapped through the same matrix, are returned as
`PageTransform.line_quads` when they can be recognised directly (`OcrEngine.recognize`), so the
working image is not detected a second time. That is exact when the working image is the
detector copy itself (no turn, deskew or rescale, no contrast step), and on other pages is used
only while a second detection would not see the text any finer (`MAX_DETECTOR_GAIN`).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from PIL import Image

from dococr.ocr import Box, OcrEngine, Quad

MAX_LONG_SIDE = 2000  # the recogniser downsizes beyond this anyway
# The detector pass runs on a copy reduced to the working image's size cap, so that its line
# quads can be recognised directly (see `_reuse_detection`). A smaller copy (1280 px was used
# before) saves nothing on pages whose short side is under the detector's own 736 px floor, which
# it enlarges to that anyway, and the reduce-then-enlarge round trip blurs word gaps shut: on the
# synthetic wide pages it merged neighbouring words into one line.
DET_LONG_SIDE = MAX_LONG_SIDE
# Recognise the first pass's quads only when detecting again on the working image would not see
# the text at a materially higher resolution. Measured on the synthetic set, where every reused
# page has a gain of 0.88-1.06 (upscaled pages included, because the detector enlarges small pages
# to a 736 px short side either way); larger gains are not measured, so they keep the second pass.
MAX_DETECTOR_GAIN = 1.1
TARGET_TEXT_HEIGHT = 30.0  # px; upscale pages whose median text line is much smaller
MIN_TEXT_HEIGHT = 22.0
MAX_UPSCALE = 3.0
MAX_SKEW_DEG = 20.0
CLASSIFIER_ASPECT = 6.0  # widest crop, in line heights, given to the orientation classifier
MIN_SKEW_DEG = 0.5


@dataclass
class PageTransform:
    """Affine map original -> working image (3x3), plus what was decided."""

    matrix: np.ndarray
    original_size: tuple[int, int]  # (width, height)
    rotation: int = 0  # quarter turns applied, in degrees counter-clockwise (0/90/180/270)
    skew: float = 0.0  # degrees counter-clockwise applied to level the text
    scale: float = 1.0
    notes: list[str] = field(default_factory=list)
    # The first detector pass's line quads mapped into the working image, when they may be
    # recognised directly instead of detecting again on the working image (see `_reuse_detection`).
    # None means: run a full read on the working image.
    line_quads: Optional[list[np.ndarray]] = None

    def box_to_original(self, box: Box) -> tuple[int, int, int, int]:
        inv = np.linalg.inv(self.matrix)
        l, t, r, b = box
        pts = np.array([[l, t, 1], [r, t, 1], [r, b, 1], [l, b, 1]], dtype=float).T
        x, y, _ = inv @ pts
        w, h = self.original_size
        return (int(max(0, math.floor(x.min()))), int(max(0, math.floor(y.min()))),
                int(min(w, math.ceil(x.max()))), int(min(h, math.ceil(y.max()))))


def _quad_geometry(quad: Quad) -> tuple[float, float, float]:
    """(long side, short side, angle of the long side in degrees, in (-90, 90])."""
    p = np.asarray(quad, dtype=float)
    e1, e2 = p[1] - p[0], p[2] - p[1]
    l1, l2 = float(np.hypot(*e1)), float(np.hypot(*e2))
    edge = e1 if l1 >= l2 else e2
    angle = math.degrees(math.atan2(edge[1], edge[0]))
    while angle <= -90:
        angle += 180
    while angle > 90:
        angle -= 180
    return max(l1, l2), min(l1, l2), angle


def _rotation_matrix(angle_deg: float) -> np.ndarray:
    """3x3 counter-clockwise rotation about the origin (image coords, y down), as cv2 does."""
    a = math.radians(angle_deg)
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])


def _fit(matrix: np.ndarray, size: tuple[int, int]) -> tuple[np.ndarray, tuple[int, int]]:
    """Add the translation that puts the transformed page at (0, 0); return output size."""
    w, h = size
    corners = np.array([[0, 0, 1], [w, 0, 1], [w, h, 1], [0, h, 1]], dtype=float).T
    x, y, _ = matrix @ corners
    shift = np.array([[1, 0, -x.min()], [0, 1, -y.min()], [0, 0, 1]], dtype=float)
    out_w = int(math.ceil(x.max() - x.min()))
    out_h = int(math.ceil(y.max() - y.min()))
    return shift @ matrix, (max(1, out_w), max(1, out_h))


def _warp(image: np.ndarray, matrix: np.ndarray, size: tuple[int, int], scale: float) -> np.ndarray:
    import cv2

    interp = cv2.INTER_AREA if scale < 1 else cv2.INTER_CUBIC
    if np.allclose(matrix[:2, :2], np.diag([scale, scale])):
        if size == (image.shape[1], image.shape[0]):
            return image.copy()
        return cv2.resize(image, size, interpolation=interp)
    if scale < 0.75:  # shrink first with area averaging, then rotate
        small = cv2.resize(image, None, fx=scale, fy=scale, interpolation=cv2.INTER_AREA)
        pre = np.array([[1 / scale, 0, 0], [0, 1 / scale, 0], [0, 0, 1]])
        return cv2.warpAffine(small, (matrix @ pre)[:2], size, flags=cv2.INTER_LINEAR,
                              borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))
    return cv2.warpAffine(image, matrix[:2], size, flags=interp,
                          borderMode=cv2.BORDER_CONSTANT, borderValue=(255, 255, 255))


def _enhance(image: np.ndarray) -> tuple[np.ndarray, Optional[str]]:
    """Modest contrast stretch for flat, low-contrast photos. Leaves good scans alone."""
    import cv2

    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    lo, hi = np.percentile(gray, [2, 98])
    if hi - lo >= 110:
        return image, None
    lab = cv2.cvtColor(image, cv2.COLOR_RGB2LAB)
    clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    lab[:, :, 0] = clahe.apply(lab[:, :, 0])
    return cv2.cvtColor(lab, cv2.COLOR_LAB2RGB), "low contrast: applied CLAHE"


def _apply(matrix: np.ndarray, quad: np.ndarray) -> np.ndarray:
    """Map the corners of a quad through a 3x3 affine matrix."""
    q = np.asarray(quad, dtype=float)
    return (matrix @ np.c_[q, np.ones(len(q))].T).T[:, :2]


def detector_gain(engine: OcrEngine, original_size: tuple[int, int], det_size: tuple[int, int],
                  work_size: tuple[int, int], scale: float) -> float:
    """How many times finer, relative to the original page, a detector pass on the working
    image would see the text than the pass on the detector copy did. Sizes are (width, height);
    the engine says how its detector resizes each input."""
    w = original_size[0]
    first = det_size[0] / w * engine.detector_resolution(*det_size)
    second = scale * engine.detector_resolution(*work_size)
    return second / first


def _reuse_detection(quads: list, gain: float) -> bool:
    """May the first detector pass's quads, mapped through the page transform, stand in for a
    second detection on the working image?

    Yes when there are quads and the second pass would not see the page at a materially higher
    resolution (`MAX_DETECTOR_GAIN`). Rotation, deskew, upscaling and the contrast step do not
    by themselves rule it out: the quads are mapped through the same matrix as the pixels, and on
    the synthetic set reuse on such pages was as accurate as the second pass. With no quads the
    second pass runs, on the upscaled page, in case it finds what the first missed."""
    return bool(quads) and gain <= MAX_DETECTOR_GAIN


def _classifier_window(box: Box) -> Box:
    """The part of a text line to show the orientation classifier.

    The classifier squeezes every crop into a fixed shape about four times wider than tall. A
    full-width line of running text is 25 or more times wider than tall, and squeezed that far
    it is unreadable, so the classifier's answer becomes close to a coin flip. A window of at
    most CLASSIFIER_ASPECT heights, taken from the start of the line, keeps the letters legible.
    """
    left, top, right, bottom = box
    return (left, top, min(right, left + CLASSIFIER_ASPECT * (bottom - top)), bottom)


def _crop(image: np.ndarray, box: Box) -> np.ndarray:
    l, t, r, b = (int(round(v)) for v in box)
    h, w = image.shape[:2]
    return image[max(0, t):min(h, b), max(0, l):min(w, r)]


def prepare(image: Image.Image, engine: Optional[OcrEngine] = None) -> tuple[np.ndarray, PageTransform]:
    """Return (working RGB image, transform original->working). Never mutates `image`."""
    import cv2

    arr = np.asarray(image.convert("RGB"))
    h, w = arr.shape[:2]
    long_side = max(w, h)
    notes: list[str] = []

    # 1. Detector pass on a reduced copy.
    det_scale = min(1.0, DET_LONG_SIDE / long_side)
    det_img = arr if det_scale == 1.0 else cv2.resize(arr, None, fx=det_scale, fy=det_scale,
                                                      interpolation=cv2.INTER_AREA)
    quads = engine.detect(det_img) if engine is not None else []
    quads = [np.asarray(q, dtype=float) / det_scale for q in quads]  # original coordinates
    geo = [_quad_geometry(q) for q in quads]

    # 2. Quarter turn: are most (clearly elongated) text boxes vertical?
    rotation = 0
    horiz = sum(lng * sh for lng, sh, a in geo if lng >= 1.5 * sh and abs(a) < 45)
    vert = sum(lng * sh for lng, sh, a in geo if lng >= 1.5 * sh and abs(a) >= 45)
    if vert > horiz:
        rotation = 90
    base = _rotation_matrix(rotation)

    # 3. Skew: median angle of elongated boxes after the quarter turn.
    skew = 0.0
    angles, heights = [], []
    for q in quads:
        pts = (base @ np.c_[q, np.ones(len(q))].T).T[:, :2]
        lng2, sh2, a2 = _quad_geometry(pts)
        if lng2 >= 2.0 * sh2 and abs(a2) <= MAX_SKEW_DEG:
            angles.append(a2)
            heights.append(sh2)
        elif lng2 >= 1.2 * sh2:
            heights.append(sh2)
    if len(angles) >= 2:
        med = float(np.median(angles))
        if MIN_SKEW_DEG <= abs(med) <= MAX_SKEW_DEG:
            skew = med  # rotating by +angle levels a line at +angle (cv2 convention)

    # 4. Scale from the median text height.
    scale = 1.0
    if heights:
        med_h = float(np.median(heights))
        if med_h < MIN_TEXT_HEIGHT:
            scale = min(TARGET_TEXT_HEIGHT / max(med_h, 1.0), MAX_UPSCALE)
    elif long_side < 1000:
        scale = min(1000 / long_side, MAX_UPSCALE)
    if long_side * scale > MAX_LONG_SIDE:
        scale = MAX_LONG_SIDE / long_side

    def build(total_rotation: float) -> tuple[np.ndarray, tuple[int, int]]:
        m = _rotation_matrix(total_rotation) @ np.diag([scale, scale, 1.0])
        return _fit(m, (w, h))

    warped: dict[float, tuple[np.ndarray, np.ndarray]] = {}

    def page_at(total_rotation: float) -> tuple[np.ndarray, np.ndarray]:
        """(matrix, page) for a total rotation, warped once: the classifier's levelled page is
        the working page unless the page is then turned over."""
        if total_rotation not in warped:
            if not total_rotation and det_scale < 1.0 and scale == det_scale:
                # A large page that only needs reducing: the detector copy is that page already.
                m = np.diag([det_img.shape[1] / w, det_img.shape[0] / h, 1.0])
                warped[total_rotation] = (m, det_img)
            else:
                m, size = build(total_rotation)
                warped[total_rotation] = (m, _warp(arr, m, size, scale))
        return warped[total_rotation]

    # 5. Upside down? Ask the classifier about the widest boxes of the levelled page.
    flip = False
    if engine is not None and quads:
        m, size = build(rotation + skew)
        cand = []
        for q in quads:
            pts = (m @ np.c_[q, np.ones(len(q))].T).T[:, :2]
            box = (pts[:, 0].min(), pts[:, 1].min(), pts[:, 0].max(), pts[:, 1].max())
            bw, bh = box[2] - box[0], box[3] - box[1]
            if bh > 4 and bw >= 2.0 * bh:
                cand.append((bw, box))
        if cand:
            cand.sort(key=lambda c: -c[0])
            levelled = page_at(rotation + skew)[1]
            frac = engine.classify_upside_down(
                [_crop(levelled, _classifier_window(b)) for _, b in cand[:7]])
            if frac is not None and frac > 0.6:
                flip = True

    total = (rotation + (180 if flip else 0)) % 360
    matrix, work = page_at(total + skew)
    work, note = _enhance(work)
    line_quads = None
    if engine is not None and quads:
        gain = detector_gain(engine, (w, h), (det_img.shape[1], det_img.shape[0]),
                             (work.shape[1], work.shape[0]), scale)
        if _reuse_detection(quads, gain):
            line_quads = [_apply(matrix, q) for q in quads]
    if total:
        notes.append(f"rotated {total} degrees")
    if skew:
        notes.append(f"deskewed {skew:.1f} degrees")
    if note:
        notes.append(note)
    return work, PageTransform(matrix=matrix, original_size=(w, h), rotation=total, skew=skew,
                               scale=scale, notes=notes, line_quads=line_quads)
