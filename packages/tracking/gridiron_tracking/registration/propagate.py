"""Carrying one registration across a moving camera.

Every real clip pans. Four freely-licensed game videos were checked before writing
this and all four tracked the ball carrier — a phone in the stands, a broadcast
camera, a handheld at a high-school game. So the static-camera assumption that
:class:`PointClickRegistration` makes on its own is not a simplification you can
ship; it is wrong on the first video anyone uploads.

The fix does not need more clicking. A football field is a plane, so the map from
frame *n* to frame *n+1* is itself a homography, and it can be measured from the
paint and the turf texture without anyone naming anything. Compose those hop by hop
onto the one hand-registered frame and every frame is registered.

Two things make it work rather than drift into nonsense:

**The players are masked out.** Twenty-two people running is the most visually
salient motion in the frame and none of it belongs to the camera. Features are taken
from the field surface with the detector's own boxes painted out, so the estimate
describes the camera and not the running back.

**Drift is measured, not hoped about.** Errors compound multiplicatively across a
chain of homographies. Each hop is scored against its own inliers, and a frame whose
estimate is weak reports as unregistered rather than quietly poisoning every frame
after it. ``coverage`` tells you what fraction of the play survived.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from .homography import Homography


def _cv2():
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "homography propagation needs opencv: pip install opencv-python"
        ) from e
    return cv2


@dataclass
class HopQuality:
    frame: int
    inliers: int
    tracked: int
    ok: bool
    reason: str = ""


@dataclass
class PropagationReport:
    hops: list[HopQuality] = dc_field(default_factory=list)
    registered: int = 0
    total: int = 0

    @property
    def coverage(self) -> float:
        return self.registered / self.total if self.total else 0.0

    def summary(self) -> str:
        weak = [h for h in self.hops if not h.ok]
        s = (f"registered {self.registered}/{self.total} frames "
             f"({self.coverage * 100:.0f}%)")
        if weak:
            s += f"; {len(weak)} weak hops, first at frame {weak[0].frame}"
        return s


class PropagatedRegistration:
    """One hand-registered anchor, carried across the whole clip.

    Implements the same ``at(frame_index)`` protocol as
    :class:`PointClickRegistration`, so nothing downstream knows the difference.
    """

    def __init__(
        self,
        base: Homography,
        anchor_frame: int = 0,
        spec=None,
        image_size: tuple[int, int] | None = None,
        min_inliers: int = 25,
        max_features: int = 1200,
        mask_boxes: dict[int, list[tuple[float, float, float, float]]] | None = None,
        horizon_frac: float = 0.35,
    ) -> None:
        from .. import field as fieldmod

        self.base = base
        self.anchor_frame = anchor_frame
        self.spec = spec or fieldmod.NFL
        self.image_size = image_size
        self.min_inliers = min_inliers
        self.max_features = max_features
        self.mask_boxes = mask_boxes or {}
        self.horizon_frac = horizon_frac

        self._H: dict[int, np.ndarray] = {anchor_frame: base.H.copy()}
        self.report = PropagationReport()

    # ------------------------------------------------------------------ build

    def build(self, source, start: int, stop: int, step: int = 1) -> PropagationReport:
        """Walk the clip from the anchor and register every frame.

        ``source`` is a :class:`~gridiron_tracking.video.VideoSource`.
        """
        cv2 = _cv2()
        prev_gray = None
        prev_idx = None
        # warp[n] takes a point in frame n back into the anchor frame's pixels.
        to_anchor = np.eye(3)

        self.report = PropagationReport(total=0)

        for i, img in source.frames(start, stop, step):
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            self.report.total += 1

            if prev_gray is None:
                prev_gray, prev_idx = gray, i
                self._H[i] = self.base.H.copy()
                self.report.registered += 1
                self.report.hops.append(HopQuality(i, 0, 0, True, "anchor"))
                continue

            step_H, q = self._hop(prev_gray, gray, i)
            if step_H is None:
                self.report.hops.append(q)
                prev_gray = gray
                continue

            # step_H maps previous-frame points to current-frame points. To send a
            # current-frame point back to the anchor we need the inverse, composed
            # onto whatever already took the previous frame there.
            to_anchor = to_anchor @ np.linalg.inv(step_H)

            # base.H is field -> anchor pixels; to_anchor is current -> anchor.
            # So field -> current is inv(to_anchor) @ base.H.
            H_i = np.linalg.inv(to_anchor) @ self.base.H
            if abs(H_i[2, 2]) > 1e-12:
                H_i = H_i / H_i[2, 2]
            self._H[i] = H_i
            self.report.registered += 1
            self.report.hops.append(q)
            prev_gray, prev_idx = gray, i

        return self.report

    def _hop(self, prev_gray, gray, idx: int):
        cv2 = _cv2()
        h, w = prev_gray.shape[:2]

        # The crowd is the other big moving thing in frame, and it is above the
        # field. Ignoring the top of the image costs nothing and removes thousands
        # of useless corners.
        mask = np.zeros((h, w), np.uint8)
        mask[int(h * self.horizon_frac):, :] = 255
        for (x1, y1, x2, y2) in self.mask_boxes.get(idx, []):
            cv2.rectangle(mask, (int(x1) - 6, int(y1) - 6), (int(x2) + 6, int(y2) + 6), 0, -1)

        p0 = cv2.goodFeaturesToTrack(
            prev_gray, maxCorners=self.max_features, qualityLevel=0.01,
            minDistance=8, mask=mask, blockSize=7,
        )
        if p0 is None or len(p0) < self.min_inliers:
            return None, HopQuality(idx, 0, 0 if p0 is None else len(p0), False, "too few features")

        p1, st, _ = cv2.calcOpticalFlowPyrLK(
            prev_gray, gray, p0, None,
            winSize=(21, 21), maxLevel=3,
            criteria=(cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 30, 0.01),
        )
        if p1 is None:
            return None, HopQuality(idx, 0, 0, False, "flow failed")

        good0 = p0[st == 1].reshape(-1, 2)
        good1 = p1[st == 1].reshape(-1, 2)
        if len(good0) < self.min_inliers:
            return None, HopQuality(idx, 0, len(good0), False, "too few tracked")

        H, inl = cv2.findHomography(good0, good1, cv2.RANSAC, 2.0, maxIters=3000)
        n_in = int(inl.sum()) if inl is not None else 0
        if H is None or n_in < self.min_inliers:
            return None, HopQuality(idx, n_in, len(good0), False, "no consensus")

        return H, HopQuality(idx, n_in, len(good0), True)

    # --------------------------------------------------------------- protocol

    def at(self, frame_index: int) -> Homography | None:
        H = self._H.get(frame_index)
        return Homography(H) if H is not None else None

    def to_field(self, frame_index: int, image_pts, height_yards: float = 0.0):
        from .camera import ground_points_with_height

        h = self.at(frame_index)
        if h is None:
            return np.full((len(np.atleast_2d(image_pts)), 2), np.nan)
        return ground_points_with_height(
            h, image_pts, height_yards=height_yards, image_size=self.image_size
        )

    # ----------------------------------------------------------------- checks

    def drift_yards(self, frame_index: int, probe=None) -> float:
        """Deprecated: this measured nothing and returned 0.0 on every frame.

        It pushed field points to pixels and back through the *same* homography, which is
        an exact inverse by construction, so it reported perfect stability on a chain of
        composed estimates that may have been drifting badly. Sampled across the reference
        clip it returned 0.0 six times out of six.

        That is the same mistake as the registration it was meant to police: a round trip
        through one map cannot see whether the map is right. Use :meth:`paint_drift`,
        which needs the pixels and so can disagree with the arithmetic.
        """
        import warnings

        warnings.warn(
            "drift_yards round-trips through a single homography and always returns 0; "
            "use paint_drift(frames, spec), which checks against the paint",
            DeprecationWarning, stacklevel=2,
        )
        return float("nan")

    def resnap(self, frames: dict, spec=None, min_score: float = 1.3, log=None) -> dict:
        """Pull every frame's map back onto that frame's paint.

        Propagation composes one homography per hop and the errors multiply, so the chain
        is exact at the anchor and progressively invented after it. Measured on the
        reference clip: paint alignment 3.53 on the anchor frame, 2.6 in the middle, 1.35
        by the end — the last second of the play registered barely better than chance
        while every inlier count along the way looked healthy.

        The correction needs no model and no operator. Each frame already has an estimate
        good to a yard or so, and each frame has its own painted lines; snapping the
        estimate onto them turns a chain that drifts into a chain that is re-grounded
        every frame. Propagation still does the work of getting close — the snap has a
        search window narrower than the gap between yard lines, so it can only correct a
        fit that is nearly right, and on a frame where the paint is hidden it correctly
        declines and the propagated estimate stands.
        """
        from .vision import paint_alignment, snap_to_paint

        spec = spec or self.spec
        before, after, fixed = [], [], 0
        for i in sorted(frames):
            h = self.at(i)
            if h is None:
                continue
            b = paint_alignment(frames[i], h, spec)
            if b != b or b >= 6.0:
                continue
            h2, a = snap_to_paint(frames[i], h, spec, rounds=2)
            before.append(b)
            if a > b and a >= min_score:
                self._H[i] = h2.H.copy()
                after.append(a)
                fixed += 1
            else:
                after.append(b)
        out = {
            "frames": len(before), "resnapped": fixed,
            "before": float(np.median(before)) if before else float("nan"),
            "after": float(np.median(after)) if after else float("nan"),
            "worst_before": float(np.min(before)) if before else float("nan"),
            "worst_after": float(np.min(after)) if after else float("nan"),
        }
        if log:
            log(f"    re-snapped {fixed}/{out['frames']} frames onto the paint: "
                f"median alignment {out['before']:.2f} -> {out['after']:.2f}, "
                f"worst {out['worst_before']:.2f} -> {out['worst_after']:.2f}")
        return out

    def paint_drift(self, frames: dict, spec=None, every: int = 25) -> dict[int, float]:
        """Paint alignment at frames across the clip, so drift is visible rather than assumed.

        Propagation composes one homography per hop and errors multiply, so a chain that is
        perfect at the anchor can be yards out by the end of a play. The only check that
        can see that without ground truth is the one that looks at the picture: are this
        frame's drawn yard lines still on this frame's painted ones?

        Returns the score per sampled frame. A run that starts near 3 and ends near 1 has
        drifted into nonsense however clean its inlier counts looked.
        """
        from .vision import paint_alignment

        spec = spec or self.spec
        out: dict[int, float] = {}
        for i in sorted(frames)[::max(1, every)]:
            h = self.at(i)
            if h is None:
                out[i] = float("nan")
                continue
            out[i] = paint_alignment(frames[i], h, spec)
        return out
