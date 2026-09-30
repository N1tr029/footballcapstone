"""Finding the paint, so a human only has to name it.

Reading pixel coordinates off a frame by eye is slow and imprecise, and it is the
wrong division of labour: a computer is far better than a person at locating a white
line on green grass to the pixel, and far worse at knowing that the line in question
is the 35. So this module does the locating and leaves the naming.

The output is a labelled overlay — every candidate yard line drawn with an index —
and the operator's whole job becomes "line 4 is the 30, and the offense attacks
left". That is a handful of numbers typed once, against a picture, with no
opportunity to be a few pixels out.

It is also the first half of fully automatic registration. Once something can name
the lines on its own — by counting them off a detected end zone, or by reading the
painted numbers — this same detection feeds it and the human drops out.

**Status: does not work well enough to use.** On four real clips it finds three lines
on a frame with fifteen, and the fragments it does find cluster on players' limbs
rather than on paint. Two causes are understood and neither is solved here: white
trousers and jerseys on the sideline survive the paint mask, and a night game is
washed out enough that the paint-to-turf contrast is comparable to the noise. It is
committed because the masks and the fragment-merging are sound and worth building on,
not because it is ready. Use hand-identified points until this reports clean,
evenly-spaced lines with plausible perspective gaps.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def _cv2():
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise ImportError("line detection needs opencv: pip install opencv-python") from e
    return cv2


@dataclass
class FieldLine:
    """A detected line, in the image."""

    index: int
    x_at_top: float
    x_at_bottom: float
    y_top: float
    y_bottom: float
    strength: float

    def x_at(self, y: float) -> float:
        """Where this line sits at a given image row."""
        if abs(self.y_bottom - self.y_top) < 1e-6:
            return self.x_at_top
        f = (y - self.y_top) / (self.y_bottom - self.y_top)
        return self.x_at_top + f * (self.x_at_bottom - self.x_at_top)

    def point_at(self, y: float) -> tuple[float, float]:
        return (self.x_at(y), y)


def turf_mask(img, sat_lo: int = 40, hue_lo: int = 25, hue_hi: int = 95):
    """Where the grass is.

    Everything useful is on the turf and almost everything unhelpful — crowd, sky,
    scoreboard, the band — is not, so isolating green first removes the great
    majority of false lines before any line finder runs.
    """
    cv2 = _cv2()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    m = cv2.inRange(hsv, (hue_lo, sat_lo, 40), (hue_hi, 255, 255))
    m = cv2.morphologyEx(m, cv2.MORPH_CLOSE, np.ones((25, 25), np.uint8))
    # Keep only the largest green region: the field, not a green advertising board.
    n, lab, stats, _ = cv2.connectedComponentsWithStats(m, 8)
    if n > 1:
        big = 1 + int(np.argmax(stats[1:, cv2.CC_STAT_AREA]))
        m = (lab == big).astype(np.uint8) * 255
    return m


def paint_mask(img, mask=None):
    """Where the white paint is, within the turf."""
    cv2 = _cv2()
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
    # Paint is bright and unsaturated. A threshold on value alone also catches white
    # jerseys, which is why the turf mask is eroded inward first.
    white = cv2.inRange(hsv, (0, 0, 150), (180, 70, 255))
    if mask is not None:
        inner = cv2.erode(mask, np.ones((9, 9), np.uint8), iterations=2)
        white = cv2.bitwise_and(white, inner)
    return white


def find_yard_lines(
    img,
    min_length_px: float = 26.0,
    max_lines: int = 40,
    merge_px: float = 22.0,
    min_support: float = 80.0,
) -> list[FieldLine]:
    """Yard lines, left to right, as they cross the visible field.

    Yard lines run across the field and so appear steep in any sideline view; the
    sidelines themselves run along it and appear shallow. Filtering on angle keeps
    the two apart without knowing anything about the camera.

    The line detector returns fragments, not lines — a yard line broken by players,
    worn paint and mown stripes comes back as a dozen pieces with a median length of
    about a dozen pixels. Demanding long segments finds nothing (the first version of
    this asked for 130 px and detected zero lines on a frame with fifteen of them).
    So fragments are kept short and merged, and they are merged by where they cross
    the *middle* of the turf rather than the top: extrapolating a 12 px fragment to
    the top of the frame amplifies its angle error into hundreds of pixels, while at
    mid-field the extrapolation is short and the grouping is stable.
    """
    cv2 = _cv2()
    h, w = img.shape[:2]
    turf = turf_mask(img)
    paint = paint_mask(img, turf)

    ys, xs = np.nonzero(turf)
    if len(ys) < 1000:
        return []
    y_lo, y_hi = float(np.percentile(ys, 2)), float(np.percentile(ys, 98))
    y_mid = 0.5 * (y_lo + y_hi)

    lsd = cv2.createLineSegmentDetector(cv2.LSD_REFINE_STD)
    segs = lsd.detect(paint)[0]
    if segs is None:
        return []
    segs = segs.reshape(-1, 4)

    cands = []
    for x1, y1, x2, y2 in segs:
        dx, dy = x2 - x1, y2 - y1
        length = float(np.hypot(dx, dy))
        if length < min_length_px:
            continue
        if abs(dy) < abs(dx) * 1.2:            # shallow: a sideline, not a yard line
            continue
        if y1 > y2:
            x1, y1, x2, y2 = x2, y2, x1, y1
        slope = (x2 - x1) / (y2 - y1)           # dx per row
        cands.append((x1 + (y_mid - y1) * slope, slope, length))

    if not cands:
        return []

    cands.sort(key=lambda c: c[0])
    groups: list[list[float]] = []
    for x_mid, slope, length in cands:
        if groups and abs(x_mid - groups[-1][0]) < merge_px:
            g = groups[-1]
            tot = g[2] + length
            g[0] = (g[0] * g[2] + x_mid * length) / tot
            g[1] = (g[1] * g[2] + slope * length) / tot
            g[2] = tot
        else:
            groups.append([x_mid, slope, length])

    # Support is total fragment length. A real yard line accumulates hundreds of
    # pixels across its fragments; a chance alignment of two scuff marks does not.
    groups = [g for g in groups if g[2] >= min_support]
    groups.sort(key=lambda g: -g[2])
    groups = groups[:max_lines]
    groups.sort(key=lambda g: g[0])

    out = []
    for i, (x_mid, slope, support) in enumerate(groups):
        out.append(FieldLine(
            index=i,
            x_at_top=x_mid + (y_lo - y_mid) * slope,
            x_at_bottom=x_mid + (y_hi - y_mid) * slope,
            y_top=y_lo, y_bottom=y_hi, strength=support,
        ))
    return out


def find_sidelines(img) -> tuple[float, float] | None:
    """The top and bottom rows where the turf ends, at the image centre.

    Crude but useful: it gives the vertical extent of the playing surface, which is
    the sanity check on whether a proposed registration has the field the right size.
    """
    turf = turf_mask(img)
    h, w = turf.shape[:2]
    col = turf[:, w // 2]
    on = np.nonzero(col)[0]
    if len(on) < 10:
        return None
    return float(on.min()), float(on.max())


def overlay(img, lines: list[FieldLine], out_path: str):
    """Draw the detected lines with their indices, for a human to name."""
    cv2 = _cv2()
    canvas = img.copy()
    for ln in lines:
        p0 = (int(ln.x_at_top), int(ln.y_top))
        p1 = (int(ln.x_at_bottom), int(ln.y_bottom))
        cv2.line(canvas, p0, p1, (0, 0, 255), 2, cv2.LINE_AA)
        for pt in (p0, p1):
            cv2.circle(canvas, pt, 5, (0, 255, 255), -1)
        label = str(ln.index)
        cv2.putText(canvas, label, (p0[0] - 12, p0[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 0), 5)
        cv2.putText(canvas, label, (p0[0] - 12, p0[1] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 255), 2)
    sl = find_sidelines(img)
    if sl:
        for y in sl:
            cv2.line(canvas, (0, int(y)), (canvas.shape[1], int(y)), (255, 0, 255), 1, cv2.LINE_AA)
    cv2.imwrite(out_path, canvas)
    return out_path
