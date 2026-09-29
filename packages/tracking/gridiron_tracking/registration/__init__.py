"""Field registration: the thing that makes the output *data* instead of boxes.

The protocol is small on purpose. Everything downstream asks a registration one
question — "for frame N, what is the map?" — and every way of answering it, from a
human clicking four corners to a learned keypoint model, is a different
implementation of that one method. Stream 2's roadmap is three implementations
behind this interface, not three pipelines.

:class:`PointClickRegistration` is the one that ships first. It is not a
placeholder for a cleverer method: for a coach's tripod on a high-school sideline,
twenty seconds of clicking beats any automatic line detector that has to cope with
a worn 40-yard line in November, and it works on the first field it ever sees.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Protocol, Sequence, runtime_checkable

import numpy as np

from .. import field as fieldmod
from .camera import (
    Camera,
    DegenerateGeometry,
    HELMET_HEIGHT_YARDS,
    TORSO_HEIGHT_YARDS,
    ground_points_with_height,
)
from .homography import Homography, dlt_homography, ransac_homography, transform

__all__ = [
    "Camera", "DegenerateGeometry", "HELMET_HEIGHT_YARDS", "TORSO_HEIGHT_YARDS",
    "Homography", "dlt_homography", "ransac_homography", "transform",
    "ground_points_with_height", "Correspondence", "Anchor", "Registration",
    "PointClickRegistration", "RegistrationQuality",
]


@dataclass(frozen=True)
class Correspondence:
    """One click: a named point on the field, and the pixel it appeared at."""

    landmark_id: str
    image_xy: tuple[float, float]

    def field_xy(self, spec: fieldmod.FieldSpec) -> tuple[float, float]:
        return fieldmod.landmark(self.landmark_id, spec).xy


@dataclass
class Anchor:
    """A frame the operator registered by hand."""

    frame_index: int
    correspondences: list[Correspondence]
    spec: fieldmod.FieldSpec = fieldmod.NFL

    def fit(self, threshold_px: float = 3.0) -> Homography:
        if len(self.correspondences) < 4:
            raise ValueError(
                f"frame {self.frame_index} has {len(self.correspondences)} clicks; "
                "a homography needs at least 4, and 6 well-spread ones is the number "
                "that actually behaves"
            )
        fld = np.array([c.field_xy(self.spec) for c in self.correspondences], dtype=float)
        img = np.array([c.image_xy for c in self.correspondences], dtype=float)
        return Homography.fit(fld, img, threshold_px=threshold_px)


@dataclass
class RegistrationQuality:
    """What to show an operator before letting them run a whole play through."""

    rms_px: float
    rms_yards: float
    p95_yards: float
    used: int
    rejected: int
    camera: dict[str, float | bool] | None
    warnings: list[str] = dc_field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.warnings and self.rms_yards < 0.5


@runtime_checkable
class Registration(Protocol):
    """Frame index in, ground-plane map out."""

    def at(self, frame_index: int) -> Homography | None: ...


class PointClickRegistration:
    """Registration from hand-clicked landmarks.

    One anchor covers a locked-off camera for the whole play. Several anchors cover
    a camera that was re-aimed between plays: each frame takes the nearest anchor's
    map.

    What this deliberately does *not* do is interpolate between anchors. A camera
    panning through a play does not move its homography linearly, and pretending
    otherwise produces an error that is worst in the middle of the play — which is
    where the ball is. Panning is a job for frame-to-frame propagation against the
    static background, which needs real pixels and lands with the video milestone;
    until then, ``at()`` returning the nearest anchor is at least honestly wrong in
    a way the quality report can measure.
    """

    def __init__(
        self,
        anchors: Sequence[Anchor],
        spec: fieldmod.FieldSpec = fieldmod.NFL,
        threshold_px: float = 3.0,
        image_size: tuple[int, int] | None = None,
    ) -> None:
        if not anchors:
            raise ValueError("PointClickRegistration needs at least one anchor")
        self.spec = spec
        self.image_size = image_size
        self.anchors = sorted(anchors, key=lambda a: a.frame_index)
        self._fits: dict[int, Homography] = {
            a.frame_index: a.fit(threshold_px=threshold_px) for a in self.anchors
        }

    # ------------------------------------------------------------------ protocol

    def at(self, frame_index: int) -> Homography | None:
        if not self._fits:
            return None
        nearest = min(self._fits, key=lambda i: abs(i - frame_index))
        return self._fits[nearest]

    # ------------------------------------------------------------------ helpers

    def camera_at(self, frame_index: int) -> Camera | None:
        h = self.at(frame_index)
        if h is None or self.image_size is None:
            return None
        try:
            return Camera.from_homography(h, image_size=self.image_size)
        except (DegenerateGeometry, np.linalg.LinAlgError, ValueError):
            return None

    def to_field(self, frame_index: int, image_pts, height_yards: float = 0.0):
        """Image points to field yards, correcting for how high off the ground the
        detector's reference point sits. Pass ``HELMET_HEIGHT_YARDS`` for helmet
        boxes, ``0`` for a body box's bottom edge."""
        h = self.at(frame_index)
        if h is None:
            return np.full((len(np.atleast_2d(image_pts)), 2), np.nan)
        return ground_points_with_height(
            h, image_pts, height_yards=height_yards, image_size=self.image_size
        )

    # ------------------------------------------------------------------- quality

    def assess(self, frame_index: int | None = None) -> RegistrationQuality:
        """Score one anchor, with the warnings an operator needs to see.

        The failure this catches is not imprecision, it is *geometry*: four points
        clicked along a single yard line, or all four bunched in one corner of a
        wide shot, fit beautifully and then place a receiver in the parking lot,
        because the fit has no evidence about the direction nobody clicked in. The
        residual on the clicks themselves cannot see that, so it is checked here
        directly.
        """
        anchor = (
            self.anchors[0] if frame_index is None
            else min(self.anchors, key=lambda a: abs(a.frame_index - frame_index))
        )
        h = self._fits[anchor.frame_index]

        fld = np.array([c.field_xy(self.spec) for c in anchor.correspondences], dtype=float)
        img = np.array([c.image_xy for c in anchor.correspondences], dtype=float)
        mask = h.inliers if h.inliers is not None else np.ones(len(fld), dtype=bool)

        err_px = h.reprojection_error_px(fld[mask], img[mask])
        err_yd = h.reprojection_error_yards(fld[mask], img[mask])

        warnings: list[str] = []
        if mask.sum() < len(fld):
            warnings.append(
                f"{len(fld) - int(mask.sum())} of {len(fld)} clicks disagree with the "
                "rest — most likely a landmark named one yard line off"
            )

        spread_x = float(np.ptp(fld[mask][:, 0])) if mask.sum() else 0.0
        spread_y = float(np.ptp(fld[mask][:, 1])) if mask.sum() else 0.0
        if spread_x < 10.0:
            warnings.append(
                f"clicks span only {spread_x:.1f} yards downfield; the fit has almost "
                "no evidence about the x direction"
            )
        if spread_y < 10.0:
            warnings.append(
                f"clicks span only {spread_y:.1f} yards across; click both sidelines "
                "or both hash rows"
            )
        if len(anchor.correspondences) < 6:
            warnings.append(
                f"{len(anchor.correspondences)} clicks — four is the minimum and leaves "
                "nothing over to detect a mistake with; six makes the residual meaningful"
            )

        cam_desc = None
        try:
            if self.image_size is not None:
                cam = Camera.from_homography(h, image_size=self.image_size)
                cam_desc = cam.describe()
                if not (1.0 <= cam.height_yards <= 60.0):
                    warnings.append(
                        f"solved camera height is {cam.height_yards:.1f} yards, which is "
                        "not where a camera goes — check the landmark names"
                    )
        except (DegenerateGeometry, np.linalg.LinAlgError, ValueError):
            cam_desc = None

        rms_yd = float(np.sqrt(np.mean(err_yd ** 2))) if len(err_yd) else float("inf")
        if rms_yd > 0.5:
            warnings.append(f"reprojection error is {rms_yd:.2f} yards on the clicks themselves")

        return RegistrationQuality(
            rms_px=float(np.sqrt(np.mean(err_px ** 2))) if len(err_px) else float("inf"),
            rms_yards=rms_yd,
            p95_yards=float(np.percentile(err_yd, 95)) if len(err_yd) else float("inf"),
            used=int(mask.sum()),
            rejected=int(len(fld) - mask.sum()),
            camera=cam_desc,
            warnings=warnings,
        )
