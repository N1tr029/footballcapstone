"""The height correction, which is the largest single error this package removes."""

import numpy as np
import pytest

from gridiron_tracking.detect.synthetic import sideline_camera
from gridiron_tracking.registration.camera import (
    HELMET_HEIGHT_YARDS, Camera, DegenerateGeometry,
)
from gridiron_tracking.registration.homography import Homography

IMAGE = (1280, 720)


def _clicked(cam, ids=None):
    from gridiron_tracking.field import catalog
    cat = catalog()
    ids = ids or ["y20A/sideline_y0", "y50/sideline_y0", "y20B/sideline_y0",
                  "y20A/sideline_ymax", "y50/sideline_ymax", "y20B/sideline_ymax"]
    fld = np.array([cat[i].xy for i in ids])
    img = cam.project(np.hstack([fld, np.zeros((len(fld), 1))]))
    return fld, img


def test_decomposition_recovers_a_known_camera():
    truth = sideline_camera(image_size=IMAGE, x=60.0, y=-25.0, height=12.0, focal_px=1400.0)
    fld, img = _clicked(truth)
    cam = Camera.from_homography(Homography.fit(fld, img), image_size=IMAGE)

    assert cam.focal_px == pytest.approx(1400.0, rel=1e-6)
    assert np.allclose(cam.C, truth.C, atol=1e-6)
    assert cam.height_yards == pytest.approx(12.0, abs=1e-6)


def test_height_correction_is_exact_and_the_naive_answer_is_not():
    truth = sideline_camera(image_size=IMAGE, x=60.0, y=-25.0, height=12.0)
    fld, img = _clicked(truth)
    h = Homography.fit(fld, img)
    cam = Camera.from_homography(h, image_size=IMAGE)

    standing = np.array([[45.0, 20.0], [75.0, 40.0], [60.0, 5.0], [100.0, 26.65]])
    helmets = truth.project(np.hstack([standing, np.full((len(standing), 1), HELMET_HEIGHT_YARDS)]))

    corrected = cam.ground_point(helmets, height_yards=HELMET_HEIGHT_YARDS)
    assert np.allclose(corrected, standing, atol=1e-6)

    naive = h.to_field(helmets)
    err = np.linalg.norm(naive - standing, axis=1)
    # Not a rounding difference: pushing a helmet through the ground plane is
    # yards wrong, and always in the same direction (away from the camera).
    assert err.min() > 4.0
    assert (naive[:, 1] > standing[:, 1]).all()


def test_ground_level_correction_is_the_plain_homography():
    cam = sideline_camera(image_size=IMAGE)
    fld, img = _clicked(cam)
    h = Homography.fit(fld, img)
    c = Camera.from_homography(h, image_size=IMAGE)
    pts = np.array([[500.0, 600.0], [800.0, 500.0]])
    assert np.allclose(c.ground_point(pts, 0.0), h.to_field(pts), atol=1e-6)


def test_a_view_with_no_perspective_says_so_rather_than_guessing():
    # A straight-down view: the projection really is affine, so no focal length
    # solves it and height genuinely cannot displace anything.
    H = Homography(np.array([[10.0, 0, 100.0], [0, 10.0, 50.0], [0, 0, 1.0]]))
    with pytest.raises(DegenerateGeometry):
        Camera.from_homography(H, image_size=IMAGE)


def test_solved_camera_height_is_a_usable_sanity_check():
    for height in (6.0, 12.0, 30.0):
        cam = sideline_camera(image_size=IMAGE, height=height)
        fld, img = _clicked(cam)
        got = Camera.from_homography(Homography.fit(fld, img), image_size=IMAGE)
        assert got.height_yards == pytest.approx(height, abs=1e-5)
