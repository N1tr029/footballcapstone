import numpy as np
import pytest

from gridiron_tracking.registration.homography import Homography, dlt_homography, transform


def _camera_H(seed=0):
    """A homography at a realistic image scale.

    The first version of this helper used a random 3x3, which mapped the whole field
    into about forty pixels — and at that scale a three-pixel RANSAC threshold is
    wider than the field, so every outlier is an inlier. The threshold is in pixels,
    so the test has to be at the scale pixels actually occur at.
    """
    from gridiron_tracking.detect.synthetic import sideline_camera
    return sideline_camera(image_size=(1280, 720), height=12.0 + seed).homography.H


def test_round_trip_is_exact_on_clean_correspondences():
    H = _camera_H()
    fld = np.array([[10.0, 0.0], [110.0, 0.0], [10.0, 53.3], [110.0, 53.3], [60.0, 26.65]])
    img = transform(H, fld)
    fit = Homography.fit(fld, img)
    back = fit.to_field(img)
    assert np.allclose(back, fld, atol=1e-8)
    assert fit.residual_px < 1e-8


def test_error_is_reported_in_yards_not_pixels():
    H = _camera_H()
    fld = np.array([[10.0, 0.0], [110.0, 0.0], [10.0, 53.3], [110.0, 53.3], [60.0, 26.65]])
    img = transform(H, fld)
    fit = Homography.fit(fld, img)
    nudged = img + np.array([2.0, 0.0])
    yards = fit.reprojection_error_yards(fld, nudged)
    # The same pixel offset is a different distance in yards across the frame —
    # which is the entire reason the figure is quoted in yards.
    assert yards.std() > 0


def test_ransac_survives_a_landmark_named_one_line_off():
    H = _camera_H()
    fld = np.array([
        [10.0, 0.0], [60.0, 0.0], [110.0, 0.0],
        [10.0, 53.3], [60.0, 53.3], [110.0, 53.3],
    ])
    img = transform(H, fld)
    img[2] += np.array([40.0, 25.0])          # one click a whole yard line off
    fit = Homography.fit(fld, img, threshold_px=3.0)
    assert fit.inliers is not None
    assert not fit.inliers[2], "the bad click should have been rejected"
    good = [i for i in range(len(fld)) if i != 2]
    assert np.allclose(fit.to_field(img[good]), fld[good], atol=1e-6)


def test_four_points_are_the_minimum():
    with pytest.raises(ValueError, match="at least 4"):
        dlt_homography(np.zeros((3, 2)), np.zeros((3, 2)))


def test_points_on_the_horizon_come_back_as_nan_not_as_a_number():
    H = np.array([[1.0, 0, 0], [0, 1.0, 0], [1.0, 0, 0]])   # w = x, so x = 0 is at infinity
    out = transform(H, [[0.0, 5.0]])
    assert np.isnan(out).all()
