"""A full camera, recovered from the ground-plane homography.

A homography answers "where on the grass is this pixel?" and that is the right
question for a pixel that is *on* the grass. It is the wrong question for a helmet.

A helmet sits about two yards above the turf, and pushing it through the ground
map puts the player further from the camera along the viewing ray, systematically,
always in the same direction. Measured against a synthetic 12-yard-high sideline
camera (tests/test_camera.py), the displacement runs from 5 yards near the camera
to 12 at the far hash. That is not a correction on the error budget, it is an order
of magnitude more than the whole budget, and it is not noise that averages out: it
biases every player on every frame, toward the far sideline. The Kaggle oracle boxes
are helmet boxes, so this is not a corner case, it is the first thing that happens.

The fix is to stop treating the problem as planar. A homography of the plane Z = 0
is the first, second and fourth columns of a full 3x4 camera; given the intrinsics
you can recover the missing third column from the fact that a rotation matrix is
orthonormal, which is Zhang's calibration argument run on a single view. With the
whole camera in hand, "where is the player whose helmet is at this pixel?" becomes
a ray-plane intersection at Z = helmet height, which is exact.

What it costs: an assumption that the principal point is the image centre and the
pixels are square. Both are close enough to true for broadcast and consumer
cameras that the residual is well under the errors we are chasing, and both are
recorded on the object so a caller can see what was assumed.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .homography import Homography

Array = np.ndarray

HELMET_HEIGHT_YARDS = 1.90
"""Ground to the middle of the helmet, in yards. A six-foot-two player in pads and
a helmet has his head centred around five feet eight — 1.9 yards. Players vary by
maybe a tenth of a yard either side of that, which is an order of magnitude less
than the error the correction removes."""

TORSO_HEIGHT_YARDS = 1.15
"""Ground to the middle of the numbers, for a detector that boxes whole bodies."""


class DegenerateGeometry(RuntimeError):
    """The homography carries no usable perspective.

    Raised when the focal length solves imaginary, which happens for a genuinely
    fronto-parallel view — a straight-down drone shot, or a synthetic test camera
    with no tilt. In that case the projection really is affine, height really does
    not displace a point, and the ground homography alone is the correct answer.
    """


@dataclass
class Camera:
    """A calibrated view of one field.

    World coordinates are the field's: x downfield, y across, z up, all in yards.
    """

    K: Array
    R: Array
    C: Array                       # camera centre in world yards
    homography: Homography
    principal_point_assumed: bool = True

    # --------------------------------------------------------------- factory

    @classmethod
    def from_homography(
        cls, h: Homography, image_size: tuple[int, int] | None = None,
        principal_point: tuple[float, float] | None = None,
    ) -> "Camera":
        """Decompose a field-to-image homography into K, R and a camera centre.

        ``image_size`` is (width, height) and is used only to place the principal
        point at the centre; pass ``principal_point`` directly if you know better.
        """
        if principal_point is not None:
            cx, cy = principal_point
            assumed = False
        elif image_size is not None:
            cx, cy = image_size[0] / 2.0, image_size[1] / 2.0
            assumed = True
        else:
            raise ValueError("need image_size or principal_point to place the optical axis")

        H = h.H / np.linalg.norm(h.H[:, 0])
        h1, h2, h3 = H[:, 0], H[:, 1], H[:, 2]

        # With K = [[f, 0, cx], [0, f, cy], [0, 0, 1]], K^-1 h = ((a - cx c)/f,
        # (b - cy c)/f, c). Orthogonality of the two rotation columns and their
        # equal norms each give an independent estimate of f squared.
        def uvc(col: Array) -> tuple[float, float, float]:
            return col[0] - cx * col[2], col[1] - cy * col[2], col[2]

        u1, v1, c1 = uvc(h1)
        u2, v2, c2 = uvc(h2)

        est: list[float] = []
        denom_o = c1 * c2
        if abs(denom_o) > 1e-12:
            est.append(-(u1 * u2 + v1 * v2) / denom_o)
        denom_n = c2 * c2 - c1 * c1
        if abs(denom_n) > 1e-12:
            est.append(((u1 * u1 + v1 * v1) - (u2 * u2 + v2 * v2)) / denom_n)

        positive = [e for e in est if e > 1e-9]
        if not positive:
            raise DegenerateGeometry(
                "no positive focal length solves this homography — the view has no "
                "usable perspective, so height cannot displace a point and the "
                "ground homography is already the right answer"
            )
        f = float(np.sqrt(np.mean(positive)))

        K = np.array([[f, 0.0, cx], [0.0, f, cy], [0.0, 0.0, 1.0]])
        Ki = np.linalg.inv(K)

        lam = 1.0 / np.linalg.norm(Ki @ h1)
        r1 = lam * (Ki @ h1)
        r2 = lam * (Ki @ h2)
        t = lam * (Ki @ h3)
        r3 = np.cross(r1, r2)

        # r1 and r2 came from noisy data and are not quite orthonormal. The nearest
        # true rotation is the one with the singular values flattened to one.
        R = np.column_stack([r1, r2, r3])
        u, _, vt = np.linalg.svd(R)
        R = u @ vt
        if np.linalg.det(R) < 0:
            u[:, -1] *= -1
            R = u @ vt

        # The field is in front of the camera, not behind it: a negative depth for
        # the world origin means the scale came out with the wrong sign.
        if t[2] < 0:
            R, t = -R, -t
            u2_, _, vt2 = np.linalg.svd(R)
            R = u2_ @ vt2

        C = -R.T @ t
        return cls(K=K, R=R, C=C, homography=h, principal_point_assumed=assumed)

    # ------------------------------------------------------------- projection

    @property
    def P(self) -> Array:
        """The 3x4 projection matrix."""
        return self.K @ np.hstack([self.R, (-self.R @ self.C).reshape(3, 1)])

    def project(self, world_pts) -> Array:
        """World (x, y, z) in yards to image pixels."""
        pts = np.asarray(world_pts, dtype=float)
        if pts.ndim == 1:
            pts = pts[None, :]
        if pts.shape[1] == 2:
            pts = np.hstack([pts, np.zeros((len(pts), 1))])
        homo = np.hstack([pts, np.ones((len(pts), 1))])
        out = homo @ self.P.T
        w = out[:, 2]
        bad = np.abs(w) < 1e-9
        w = np.where(bad, np.nan, w)
        xy = out[:, :2] / w[:, None]
        xy[bad] = np.nan
        return xy

    def ground_point(self, image_pts, height_yards: float = 0.0) -> Array:
        """Where on the field is the player whose *this-high* part is at this pixel?

        Back-project the pixel to a ray through the camera centre, intersect it
        with the horizontal plane at ``height_yards``, and return the (x, y) —
        which is the player's position on the turf, because he is standing up.

        With ``height_yards = 0`` this is exactly the ground homography, so the
        same call site works whether the detector boxes feet or helmets.
        """
        pts = np.asarray(image_pts, dtype=float)
        if pts.ndim == 1:
            pts = pts[None, :]
        homo = np.hstack([pts, np.ones((len(pts), 1))])

        # Ray direction in world coordinates, one per point.
        d = (np.linalg.inv(self.K) @ homo.T)
        d = (self.R.T @ d).T

        dz = d[:, 2]
        # A ray parallel to the plane never meets it; so does one pointing away.
        with np.errstate(divide="ignore", invalid="ignore"):
            s = (height_yards - self.C[2]) / dz
        bad = (~np.isfinite(s)) | (s <= 0) | (np.abs(dz) < 1e-12)

        xy = self.C[None, :2] + s[:, None] * d[:, :2]
        xy[bad] = np.nan
        return xy

    # ------------------------------------------------------------------ extras

    @property
    def focal_px(self) -> float:
        return float(self.K[0, 0])

    @property
    def height_yards(self) -> float:
        """How high the camera is above the field. A sideline tripod is 5-15 yards
        up, a press box 25-40, a phone in the front row about 2. A number outside
        those ranges means the landmarks were clicked wrong, and it is the cheapest
        sanity check on a registration there is."""
        return float(self.C[2])

    def describe(self) -> dict[str, float | bool]:
        return {
            "focal_px": self.focal_px,
            "camera_x": float(self.C[0]),
            "camera_y": float(self.C[1]),
            "camera_height_yards": self.height_yards,
            "principal_point_assumed": self.principal_point_assumed,
        }


def ground_points_with_height(
    h: Homography,
    image_pts,
    height_yards: float,
    image_size: tuple[int, int] | None = None,
) -> Array:
    """Height-corrected field positions, falling back to the plain plane map.

    The convenience wrapper most call sites want: try for the full camera, and if
    the view is degenerate take the homography's answer, which in that case is not
    an approximation but the correct one.
    """
    if height_yards == 0.0:
        return h.to_field(image_pts)
    try:
        cam = Camera.from_homography(h, image_size=image_size)
    except (DegenerateGeometry, np.linalg.LinAlgError, ValueError):
        return h.to_field(image_pts)
    return cam.ground_point(image_pts, height_yards=height_yards)
