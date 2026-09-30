"""Finding players by what colour they are wearing.

Two jobs in one place, because they are the same measurement.

**Detection.** On footage where the two sides are strongly coloured, segmenting
jersey colour finds players directly from decoded pixels — no weights, no GPU, no
download. It is not a replacement for a learned detector on real broadcast footage,
where lighting varies and half the sideline wears the same colours. It is the right
tool for rendered or controlled footage, and it is what lets the whole decode →
detect → register → track chain be exercised on an actual video file.

**Teams.** The harder and more useful job. A detector says "person"; it does not say
which side. The answer is in the torso: pool the dominant colour of every track's
torso crop across the whole play, cluster into two, and assign each track by majority
vote. Pooling across the play rather than deciding per frame is what makes it robust
— one frame of a player in shadow, or mid-tackle with his back turned, cannot flip
his team when fifty other frames disagree.

Clustering happens in **Lab**, not RGB or HSV. Euclidean distance in Lab approximates
how different two colours look to a person, which is the property that actually
matters when the question is "are these two men on the same team". The same distance
in RGB is dominated by brightness, so a navy jersey in sunlight and the same navy in
shadow land further apart than navy and maroon do.
"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass

import numpy as np

from . import Detection


def _cv2():
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise ImportError("colour detection needs opencv: pip install opencv-python") from e
    return cv2


@dataclass
class ColorSpec:
    """One team's kit, as an HSV range."""

    name: str
    lo: tuple[int, int, int]
    hi: tuple[int, int, int]


class ColorBlobDetector:
    """Detect players as coloured blobs.

    ``min_area`` and ``max_area`` are in pixels and do most of the work of rejecting
    junk: a player at this distance occupies a predictable area, a scoreboard pixel
    does not.
    """

    def __init__(
        self,
        specs: list[ColorSpec],
        min_area: int = 60,
        max_area: int = 20000,
        min_aspect: float = 0.6,
        erode: int = 0,
    ) -> None:
        self.specs = specs
        self.min_area = min_area
        self.max_area = max_area
        self.min_aspect = min_aspect
        # Players standing shoulder to shoulder merge into one blob, and a merged
        # blob is one detection where there should be two. Eroding separates them —
        # but it also splits a single player's helmet from his torso, so this trades
        # missed players against invented ones and there is no setting that gets
        # both. It is the reason colour blobs are a scaffold and not a detector.
        self.erode = erode

    def detect(self, frame_index: int, image=None) -> list[Detection]:
        if image is None:
            raise ValueError(f"ColorBlobDetector needs pixels for frame {frame_index}")
        cv2 = _cv2()
        hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
        out: list[Detection] = []

        for spec in self.specs:
            mask = cv2.inRange(hsv, spec.lo, spec.hi)
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, np.ones((7, 7), np.uint8))
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, np.ones((3, 3), np.uint8))
            if self.erode:
                mask = cv2.erode(mask, np.ones((3, 3), np.uint8), iterations=self.erode)
            n, lab, stats, _ = cv2.connectedComponentsWithStats(mask, 8)
            for i in range(1, n):
                x, y, w, h, area = stats[i]
                if not (self.min_area <= area <= self.max_area):
                    continue
                # A standing player is taller than he is wide. A blob that is not is
                # a shadow, a line, or two players merged.
                if h < w * self.min_aspect:
                    continue
                out.append(Detection(
                    frame=frame_index, x1=float(x), y1=float(y),
                    x2=float(x + w), y2=float(y + h),
                    score=float(min(1.0, area / 400.0)), kind="body",
                    team=spec.name,
                ))
        return out


# ------------------------------------------------------------------ team assignment


def torso_color(image, det: Detection, shrink: float = 0.32) -> np.ndarray | None:
    """The dominant Lab colour of a detection's torso.

    The crop is the middle of the upper half of the box: below the helmet, above the
    legs, and inset from the edges so the grass around the player does not vote.
    """
    cv2 = _cv2()
    h, w = image.shape[:2]
    bw, bh = det.width, det.height
    if bw < 4 or bh < 8:
        return None
    cx = (det.x1 + det.x2) / 2.0
    x0 = int(max(0, cx - bw * shrink))
    x1 = int(min(w, cx + bw * shrink))
    y0 = int(max(0, det.y1 + bh * 0.22))
    y1 = int(min(h, det.y1 + bh * 0.55))
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    crop = image[y0:y1, x0:x1]
    lab = cv2.cvtColor(crop, cv2.COLOR_BGR2LAB).reshape(-1, 3).astype(np.float32)
    # Median, not mean: a white number painted across the chest is a big outlier and
    # the mean drags the whole jersey toward it.
    return np.median(lab, axis=0)


def cluster_teams(
    samples: dict[str, list[np.ndarray]],
    k: int = 2,
    seed: int = 0,
) -> tuple[dict[str, int], np.ndarray]:
    """Assign each track to a team from its pooled torso colours.

    Returns the per-track assignment and the cluster centres. ``k=3`` is worth using
    on real footage, where the officials are a third kit and would otherwise be split
    arbitrarily between the two sides.
    """
    cv2 = _cv2()
    track_ids = [t for t, v in samples.items() if v]
    if not track_ids:
        return {}, np.zeros((0, 3), np.float32)

    # One representative colour per track first, so a track seen 200 times does not
    # outvote one seen 20 times when the centres are fitted.
    reps = np.array([np.median(np.stack(samples[t]), axis=0) for t in track_ids], np.float32)
    k = min(k, len(reps))
    if k < 2:
        return {t: 0 for t in track_ids}, reps[:1]

    crit = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 50, 0.5)
    _, labels, centres = cv2.kmeans(reps, k, None, crit, 8, cv2.KMEANS_PP_CENTERS)
    return {t: int(l) for t, l in zip(track_ids, labels.ravel())}, centres


def separation(centres: np.ndarray) -> float:
    """How far apart the team colours are, in Lab units.

    Under about 20 the two kits are not reliably distinguishable and the assignment
    should be reported as unreliable rather than presented as fact — two teams in
    white-on-white, or a night game where everything desaturates, genuinely cannot be
    separated by colour and the honest answer is to say so.
    """
    if len(centres) < 2:
        return 0.0
    d = [np.linalg.norm(centres[i] - centres[j])
         for i in range(len(centres)) for j in range(i + 1, len(centres))]
    return float(min(d))


def majority_team(votes: list[str | None]) -> str | None:
    real = [v for v in votes if v]
    return Counter(real).most_common(1)[0][0] if real else None
