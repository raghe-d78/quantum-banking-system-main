"""
Signature analysis (spec §7) — experimental heuristic.

1. Locate the signature region: ink in the bottom-right 45 % x 35 % of the
   document whose connected components look hand-written (large, irregular,
   low fill ratio) rather than printed text.
2. Normalise the ink of a crop: Otsu mask, ruled baseline / box borders removed,
   specks dropped, tight crop, aspect-ratio-preserving fit on a fixed canvas.
3. Compare with the customer's enrolled reference (signature_templates):
     raw  = 0.8 x stroke-overlap F-score (distance-transform based, best of small shifts)
          + 0.2 x centred HOG cosine of the normalised ink
     score = (raw - SIG_FLOOR) / (SIG_CEIL - SIG_FLOOR), clipped to [0, 1]
   SIG_FLOOR / SIG_CEIL calibrate the raw score: FLOOR = what two unrelated
   signatures reach, CEIL = what two genuine samples reach. Tune them with
   compare_signatures.py on real pairs.
   No reference enrolled -> similarity 0.5 (neutral), signature_present reported.
"""
from __future__ import annotations
import os
from dataclasses import dataclass

import cv2
import numpy as np

CANVAS_W, CANVAS_H = 160, 80
SIG_TOL = float(os.environ.get("SIGNATURE_TOL_PX", "3.0"))      # stroke tolerance in canvas pixels
SIG_FLOOR = float(os.environ.get("SIGNATURE_RAW_FLOOR", "0.50"))
SIG_CEIL = float(os.environ.get("SIGNATURE_RAW_CEIL", "0.90"))


@dataclass
class SignatureResult:
    present: bool
    bbox: tuple | None
    descriptor: np.ndarray | None
    similarity: float
    compared: bool


def _descriptor(crop: np.ndarray) -> np.ndarray:
    crop = cv2.resize(crop, (128, 64), interpolation=cv2.INTER_AREA)
    gx = cv2.Sobel(crop, cv2.CV_32F, 1, 0); gy = cv2.Sobel(crop, cv2.CV_32F, 0, 1)
    mag, ang = cv2.cartToPolar(gx, gy, angleInDegrees=True)
    hist = []
    for y in range(0, 64, 16):
        for x in range(0, 128, 16):
            m = mag[y:y + 16, x:x + 16].ravel(); a = ang[y:y + 16, x:x + 16].ravel()
            h, _ = np.histogram(a, bins=9, range=(0, 180), weights=m)
            hist.extend(h)
    v = np.array(hist, np.float32)
    return v / (np.linalg.norm(v) + 1e-6)


def locate(gray: np.ndarray) -> tuple[np.ndarray | None, tuple | None]:
    h, w = gray.shape
    roi = gray[int(h * 0.65):, int(w * 0.55):]
    thr = cv2.threshold(roi, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    thr = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, np.ones((5, 15), np.uint8))
    n, _, stats, _ = cv2.connectedComponentsWithStats(thr, 8)
    best = None
    for i in range(1, n):
        x, y, cw, ch, area = stats[i]
        if cw < roi.shape[1] * 0.15 or ch < 10:
            continue
        fill = area / float(cw * ch)
        if 0.05 <= fill <= 0.55 and cw / max(ch, 1) >= 1.5:   # stroke-like, wide, sparse
            if best is None or area > best[4]:
                best = (x, y, cw, ch, area)
    if best is None:
        return None, None
    x, y, cw, ch, _ = best
    crop = roi[y:y + ch, x:x + cw]
    return crop, (int(w * 0.55) + int(x), int(h * 0.65) + int(y), int(cw), int(ch))


# ── ink normalisation ─────────────────────────────────────────────
def _ink(crop: np.ndarray) -> np.ndarray | None:
    """Normalised stroke mask (bool, CANVAS_H x CANVAS_W) or None when there is no usable ink."""
    if crop.ndim == 3:
        crop = cv2.cvtColor(crop, cv2.COLOR_BGR2GRAY)
    g = cv2.GaussianBlur(crop, (3, 3), 0)
    b = cv2.threshold(g, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1]
    h, w = b.shape
    # Remove the ruled signature line / box borders: long, perfectly horizontal or vertical runs.
    if w >= 40 and h >= 20:
        hl = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (max(15, int(w * 0.7)), 1)))
        vl = cv2.morphologyEx(b, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (1, max(15, int(h * 0.9)))))
        rules = cv2.dilate(cv2.bitwise_or(hl, vl), np.ones((3, 3), np.uint8))
        # Slightly tilted ruled line: long, almost horizontal Hough segments are painted out too.
        segs = cv2.HoughLinesP(b, 1, np.pi / 360, threshold=int(w * 0.35), minLineLength=int(w * 0.6), maxLineGap=6)
        if segs is not None:
            for x1, y1, x2, y2 in np.asarray(segs).reshape(-1, 4):   # shape differs between OpenCV versions
                if abs(np.degrees(np.arctan2(y2 - y1, x2 - x1))) <= 8:
                    cv2.line(rules, (int(x1), int(y1)), (int(x2), int(y2)), 255, 5)
        b = cv2.subtract(b, rules)
    # Drop specks (paper grain, guilloche dots).
    n, lab, stats, _ = cv2.connectedComponentsWithStats(b, 8)
    min_area = max(6, int(0.0006 * h * w))
    keep = np.zeros_like(b)
    for i in range(1, n):
        if stats[i, cv2.CC_STAT_AREA] >= min_area:
            keep[lab == i] = 255
    ys, xs = np.where(keep > 0)
    if len(xs) < 20:
        return None
    keep = keep[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    # Aspect-ratio-preserving fit, centred on the canvas (no stretching).
    ch, cw = keep.shape
    s = min((CANVAS_W - 8) / cw, (CANVAS_H - 8) / ch)
    r = cv2.resize(keep, (max(1, int(round(cw * s))), max(1, int(round(ch * s)))), interpolation=cv2.INTER_AREA)
    canvas = np.zeros((CANVAS_H, CANVAS_W), np.uint8)
    oy, ox = (CANVAS_H - r.shape[0]) // 2, (CANVAS_W - r.shape[1]) // 2
    canvas[oy:oy + r.shape[0], ox:ox + r.shape[1]] = r
    return canvas > 48


def _skeleton(m: np.ndarray) -> np.ndarray:
    """1-px-wide centre line of a stroke mask (morphological skeleton, OpenCV only), so pen thickness
    does not matter. Returns a bool array of the same shape."""
    img = m.astype(np.uint8) * 255
    skel = np.zeros_like(img)
    el = cv2.getStructuringElement(cv2.MORPH_CROSS, (3, 3))
    for _ in range(60):
        er = cv2.erode(img, el)
        opened = cv2.dilate(er, el)
        skel = cv2.bitwise_or(skel, cv2.subtract(img, opened))
        img = er
        if not img.any():
            break
    return skel > 0


def _rotate(m: np.ndarray, deg: float) -> np.ndarray:
    if deg == 0:
        return m
    h, w = m.shape
    M = cv2.getRotationMatrix2D((w / 2, h / 2), deg, 1.0)
    r = cv2.warpAffine(m.astype(np.uint8) * 255, M, (w, h), flags=cv2.INTER_LINEAR)
    return cv2.dilate(r, np.ones((2, 2), np.uint8)) > 60


def _stroke_f(a: np.ndarray, b: np.ndarray, tol: float = SIG_TOL) -> float:
    """F-score of centre-line overlap: share of A's centre line within `tol` px of B's centre line and vice
    versa. The best of a few small shifts and rotations (+-6 deg) is kept, so a slightly offset or tilted
    signature, or one written with a thicker pen, is not punished."""
    sb = _skeleton(b)
    dt_b = cv2.distanceTransform((~sb).astype(np.uint8), cv2.DIST_L2, 3)
    sa0 = _skeleton(a)
    best = 0.0
    for ang in (-6, -3, 0, 3, 6):
        sa = _skeleton(_rotate(a, ang)) if ang else sa0
        dt_a = cv2.distanceTransform((~sa).astype(np.uint8), cv2.DIST_L2, 3)
        for dy in (-4, -2, 0, 2, 4):
            for dx in (-8, -4, 0, 4, 8):
                a_s = np.roll(sa, (dy, dx), axis=(0, 1))
                dt_as = np.roll(dt_a, (dy, dx), axis=(0, 1))
                p = float((dt_b[a_s] <= tol).mean()) if a_s.any() else 0.0     # A on B's lines
                r = float((dt_as[sb] <= tol).mean()) if sb.any() else 0.0      # B on A's lines
                f = 2 * p * r / (p + r + 1e-6)
                if f > best:
                    best = f
    return best


def raw_similarity(crop_a: np.ndarray, crop_b: np.ndarray) -> float | None:
    ia, ib = _ink(crop_a), _ink(crop_b)
    if ia is None or ib is None:
        return None
    stroke = _stroke_f(ia, ib)
    da = _descriptor(cv2.GaussianBlur(ia.astype(np.uint8) * 255, (5, 5), 0))
    db = _descriptor(cv2.GaussianBlur(ib.astype(np.uint8) * 255, (5, 5), 0))
    ca, cb = da - da.mean(), db - db.mean()
    cos = float(np.dot(ca, cb) / (np.linalg.norm(ca) * np.linalg.norm(cb) + 1e-6))
    return 0.8 * stroke + 0.2 * max(0.0, cos)


def calibrate(raw: float) -> float:
    return float(np.clip((raw - SIG_FLOOR) / max(SIG_CEIL - SIG_FLOOR, 1e-6), 0.0, 1.0))


def compare(desc_a, crop_a: np.ndarray, desc_b, crop_b: np.ndarray) -> float:
    """Similarity in [0, 1]. The stored HOG descriptors are not used any more (both sides are recomputed
    from the normalised ink), the arguments are kept so existing callers and enrolments keep working."""
    raw = raw_similarity(crop_a, crop_b)
    return 0.0 if raw is None else calibrate(raw)


def analyse(gray: np.ndarray, reference: tuple[np.ndarray, np.ndarray] | None) -> SignatureResult:
    crop, bbox = locate(gray)
    if crop is None:
        # reference enrolled but no signature found on the document -> low similarity (flagged downstream)
        return SignatureResult(False, None, None, 0.5 if reference is None else 0.2, reference is not None)
    desc = _descriptor(crop)
    if reference is None:
        return SignatureResult(True, bbox, desc, 0.5, False)
    ref_desc, ref_crop = reference
    return SignatureResult(True, bbox, desc, round(compare(desc, crop, ref_desc, ref_crop), 4), True)