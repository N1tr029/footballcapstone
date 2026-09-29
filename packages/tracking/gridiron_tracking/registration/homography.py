"""Pixels to yards, on the plane of the grass.

A football field is flat and a camera is a projection, so the map between the two
is a homography — eight numbers, recoverable from four points that are not three-
in-a-line. That is the whole reason the operator-clicks-landmarks approach works
on any field, any camera, in any light: the geometry does not care how worn the
paint is, only that a human could tell you where a corner was.

Everything here is plain numpy. No OpenCV, deliberately — this is the piece the
accuracy of the entire stream rests on, it is 150 lines of linear algebra, and it
should be testable on a machine with no CV stack installed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

Array = np.ndarray


def _as_points(pts) -> Array:
    a = np.asarray(pts, dtype=float)
    if a.ndim != 2 or a.shape[1] != 2:
        raise ValueError(f"expected an (N, 2) array of points, got shape {a.shape}")
    return a


def _normalize(pts: Array) -> tuple[Array, Array]:
    """Hartley normalisation: centre on the origin, scale so the mean distance from
    it is sqrt(2). Skipping this is the classic way to get a homography that looks
    fine on the points you fitted and falls apart twenty yards downfield, because
    the least-squares problem is badly conditioned in raw pixel units."""
    c = pts.mean(axis=0)
    d = np.sqrt(((pts - c) ** 2).sum(axis=1)).mean()
    s = np.sqrt(2.0) / d if d > 1e-12 else 1.0
    T = np.array([[s, 0.0, -s * c[0]], [0.0, s, -s * c[1]], [0.0, 0.0, 1.0]])
    homo = np.hstack([pts, np.ones((len(pts), 1))])
    return (homo @ T.T)[:, :2], T


def dlt_homography(src: Array, dst: Array) -> Array:
    """Least-squares homography taking ``src`` to ``dst``. Needs four or more
    correspondences; more than four is solved in the total-least-squares sense."""
    src, dst = _as_points(src), _as_points(dst)
    if len(src) != len(dst):
        raise ValueError("src and dst must be the same length")
    if len(src) < 4:
        raise ValueError(f"a homography needs at least 4 correspondences, got {len(src)}")

    ns, Ts = _normalize(src)
    nd, Td = _normalize(dst)

    rows = []
    for (x, y), (u, v) in zip(ns, nd):
        rows.append([-x, -y, -1, 0, 0, 0, u * x, u * y, u])
        rows.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
    _, _, vt = np.linalg.svd(np.asarray(rows, dtype=float))
    H = vt[-1].reshape(3, 3)

    H = np.linalg.inv(Td) @ H @ Ts
    if abs(H[2, 2]) > 1e-12:
        H = H / H[2, 2]
    return H


def transform(H: Array, pts) -> Array:
    """Push points through a homography. Points that map behind the camera or onto
    the horizon come back as NaN rather than as a plausible-looking finite number —
    a silent infinity here becomes a player standing in the car park."""
    pts = _as_points(pts)
    homo = np.hstack([pts, np.ones((len(pts), 1))])
    out = homo @ H.T
    w = out[:, 2]
    bad = np.abs(w) < 1e-9
    w = np.where(bad, np.nan, w)
    xy = out[:, :2] / w[:, None]
    xy[bad] = np.nan
    return xy


def ransac_homography(
    src,
    dst,
    threshold: float = 3.0,
    iterations: int = 2000,
    seed: int | None = 0,
) -> tuple[Array, Array]:
    """Homography plus an inlier mask, robust to a couple of misplaced clicks.

    ``threshold`` is in the units of ``dst``. An operator clicking landmarks on a
    1280-wide frame is reliably within a few pixels of the line and occasionally
    fifty pixels out because they clicked the 35 thinking it was the 30 — which is
    exactly the failure RANSAC exists to survive, and exactly the one that a plain
    least-squares fit quietly spreads across every other point.
    """
    src, dst = _as_points(src), _as_points(dst)
    n = len(src)
    if n < 4:
        raise ValueError(f"a homography needs at least 4 correspondences, got {n}")
    if n == 4:
        H = dlt_homography(src, dst)
        return H, np.ones(4, dtype=bool)

    rng = np.random.default_rng(seed)
    best_H, best_inliers = None, np.zeros(n, dtype=bool)

    for _ in range(iterations):
        idx = rng.choice(n, size=4, replace=False)
        try:
            H = dlt_homography(src[idx], dst[idx])
        except np.linalg.LinAlgError:
            continue
        proj = transform(H, src)
        err = np.sqrt(((proj - dst) ** 2).sum(axis=1))
        inliers = np.nan_to_num(err, nan=np.inf) < threshold
        if inliers.sum() > best_inliers.sum():
            best_H, best_inliers = H, inliers
            if best_inliers.all():
                break

    if best_H is None or best_inliers.sum() < 4:
        # Nothing consensual in the data. Fit everything and let the caller see the
        # error rather than pretending a model was found.
        return dlt_homography(src, dst), np.ones(n, dtype=bool)

    # Refit on the consensus set — the four-point model that won is not the best
    # model over the points it agrees with.
    return dlt_homography(src[best_inliers], dst[best_inliers]), best_inliers


@dataclass
class Homography:
    """The ground plane's map, in the direction that makes the camera maths work.

    ``H`` takes field yards to image pixels. The interesting direction for a
    pipeline is the other one, so use :meth:`to_field`, but keep the stored matrix
    field-to-image: that is the one :class:`~gridiron_tracking.registration.camera.Camera`
    can be decomposed out of, and having two conventions in flight is how a sign
    error survives a code review.
    """

    H: Array
    inliers: Array | None = None
    residual_px: float = 0.0

    def __post_init__(self) -> None:
        self.H = np.asarray(self.H, dtype=float).reshape(3, 3)

    @property
    def H_inv(self) -> Array:
        return np.linalg.inv(self.H)

    def to_image(self, field_pts) -> Array:
        return transform(self.H, field_pts)

    def to_field(self, image_pts) -> Array:
        return transform(self.H_inv, image_pts)

    # ------------------------------------------------------------------ quality

    def reprojection_error_px(self, field_pts, image_pts) -> Array:
        """Per-point error in pixels — how well the fit explains the clicks."""
        proj = self.to_image(field_pts)
        return np.sqrt(((proj - _as_points(image_pts)) ** 2).sum(axis=1))

    def reprojection_error_yards(self, field_pts, image_pts) -> Array:
        """Per-point error in yards, which is the number anyone should be quoting.

        Pixels are not a unit of accuracy on a perspective image: three pixels at
        the near sideline is a few inches and three pixels at the far hash is most
        of a yard. Taking the error back through the inverse map is the only way
        the figure means the same thing across the frame.
        """
        back = self.to_field(image_pts)
        return np.sqrt(((back - _as_points(field_pts)) ** 2).sum(axis=1))

    @classmethod
    def fit(
        cls,
        field_pts,
        image_pts,
        threshold_px: float = 3.0,
        iterations: int = 2000,
        seed: int | None = 0,
    ) -> "Homography":
        """Fit field-to-image from correspondences, robustly."""
        field_pts, image_pts = _as_points(field_pts), _as_points(image_pts)
        H, inliers = ransac_homography(
            field_pts, image_pts, threshold=threshold_px, iterations=iterations, seed=seed
        )
        obj = cls(H, inliers=inliers)
        err = obj.reprojection_error_px(field_pts[inliers], image_pts[inliers])
        obj.residual_px = float(np.sqrt(np.mean(err ** 2)))
        return obj
