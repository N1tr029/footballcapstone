"""The one piece of ground truth a broadcast hands you for free.

Everything else in this package checks the pipeline against itself. Reprojection error
checks a fit against the points that made it; the agreement figure checks recovered
positions against the detections they came from; even the paint check, which is the
strongest of them, only says the drawn lines sit on painted lines — it says nothing about
whether a yard is a yard.

A televised game says exactly that, in writing, on every play. The virtual line of
scrimmage and the virtual first-down line are drawn by the broadcaster through the same
field the camera is looking at, and the down-and-distance graphic in the corner says how
far apart they are. "2nd & 6" is an assertion that those two lines are six yards apart,
made by somebody with a chain and a surveyed field, and it can be checked against a
registration in the unit that matters.

Two things it is worth being clear about.

**The truth is quantised.** The distance is the officials' spot rounded to a whole yard,
so "& 6" means six yards give or take a half. A measurement inside that band is as good
as this check can report, and claiming better precision from it would be inventing
resolution the source does not have.

**It measures scale, not position.** Two lines six yards apart confirm that the map's
yard is the right size where those lines are. A registration shifted bodily down the
field would pass. Use it beside the paint check, which catches exactly that.

The second thing this returns is worth as much as the first: a drawn line is a single
yard line, so its field x must be the *same* down its whole length. Scatter in that is
perspective error the distance check cannot see.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

# The broadcaster's palette, as measured rather than assumed. The first-down line is the
# awkward one: its hue sits inside the turf's, and what separates it is that it is far
# brighter — 168 against 105 on the reference clip — so a hue-only mask finds the grass.
LOS_HSV = ((105, 100, 60), (128, 255, 255))
FIRST_DOWN_HSV = ((28, 120, 140), (42, 255, 255))


@dataclass
class MarkerCheck:
    """What the broadcast's own lines say about a registration."""

    stated_yards: float
    samples: list[tuple[int, float, float]] = dc_field(default_factory=list)
    scatter: list[float] = dc_field(default_factory=list)

    @property
    def measured(self) -> float:
        return float(np.median([abs(b - a) for _, a, b in self.samples])) if self.samples else float("nan")

    @property
    def error_yards(self) -> float:
        return self.measured - self.stated_yards

    @property
    def straightness(self) -> float:
        """Median scatter of a drawn line's field x down its own length, in yards."""
        return float(np.median(self.scatter)) if self.scatter else float("nan")

    @property
    def ok(self) -> bool:
        # Half a yard is the resolution of the source, not a tolerance chosen to pass.
        return abs(self.error_yards) <= 0.5 and self.straightness <= 0.5

    def table(self) -> str:
        n = len(self.samples)
        if not n:
            return "no frames had both broadcast lines visible; nothing to check against"
        d = [abs(b - a) for _, a, b in self.samples]
        return "\n".join([
            "the broadcast's own markers, measured through the registration",
            "",
            f"  stated        {self.stated_yards:.0f} yards apart  (from the down-and-distance graphic)",
            f"  measured      {self.measured:.2f} yards  (median of {n} frames, sd {np.std(d):.2f})",
            f"  error         {self.error_yards:+.2f} yards"
            f"   — the stated figure is rounded to the yard, so |error| under 0.5 is as",
            f"                good as this check can resolve",
            f"  straightness  {self.straightness:.2f} yards of scatter in a line that should be one x",
            "",
            f"  verdict       {'scale confirmed in yards' if self.ok else 'OUTSIDE what the source can excuse'}",
        ])


def _stripe(mask, rows, min_w: int = 3, max_w: int = 26):
    """The centre of the one narrow coloured stripe on each sampled row."""
    pts = []
    for y in rows:
        xs = np.flatnonzero(mask[y])
        if len(xs) < min_w:
            continue
        runs = np.split(xs, np.flatnonzero(np.diff(xs) > 4) + 1)
        run = max(runs, key=len)
        if min_w <= len(run) <= max_w:
            pts.append((float(run.mean()), float(y)))
    return np.array(pts) if pts else np.zeros((0, 2))


def check_markers(
    frames: dict,
    registration,
    stated_yards: float,
    rows: range | list | None = None,
    every: int = 10,
    min_points: int = 8,
) -> MarkerCheck:
    """Measure the gap between the two virtual lines, in yards, on every nth frame.

    ``stated_yards`` is the distance the down-and-distance graphic asserts — read it off
    the picture. ``rows`` are the image rows to sample across; the default spans the
    middle of the frame, avoiding the crowd at the top and the score bug at the bottom.
    """
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise ImportError("this check needs opencv") from e

    out = MarkerCheck(stated_yards=stated_yards)
    order = sorted(frames)
    if not order:
        return out
    h_img = frames[order[0]].shape[0]
    rows = rows if rows is not None else range(int(h_img * 0.30), int(h_img * 0.80), 10)

    for fi in order[::max(1, every)]:
        h = registration.at(fi)
        if h is None:
            continue
        hsv = cv2.cvtColor(frames[fi], cv2.COLOR_BGR2HSV)
        los = _stripe(cv2.inRange(hsv, *LOS_HSV), rows)
        fd = _stripe(cv2.inRange(hsv, *FIRST_DOWN_HSV), rows)
        if len(los) < min_points or len(fd) < min_points:
            continue
        fl = np.atleast_2d(h.to_field(los))
        ff = np.atleast_2d(h.to_field(fd))
        fl = fl[np.isfinite(fl).all(axis=1)]
        ff = ff[np.isfinite(ff).all(axis=1)]
        if len(fl) < min_points or len(ff) < min_points:
            continue
        out.samples.append((fi, float(np.median(fl[:, 0])), float(np.median(ff[:, 0]))))
        out.scatter.extend([float(np.std(fl[:, 0])), float(np.std(ff[:, 0]))])
    return out
