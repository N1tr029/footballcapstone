import numpy as np

from gridiron_tracking.ball_motion import Sighting, candidates, link
from gridiron_tracking.registration.homography import Homography


def _point(frame, x, y=20):
    return Sighting(frame, (x * 20, y * 20), (x, y), (0, 0, 20, 12), .8)


def test_isolated_flight_survives_but_static_stains_do_not():
    pool = {i: [_point(i, 30 + i * .7), _point(i, 70)] for i in range(9)}
    flights = link(pool, 30)
    assert len(flights) == 1
    assert len(flights[0]) == 9
    assert flights[0][0].field[0] == 30


def test_short_noise_and_impossible_jumps_cannot_start_a_flight():
    pool = {i: [_point(i, 30 + i * 8)] for i in range(9)}
    assert link(pool, 30) == []
    assert link({i: [_point(i, 30 + i * .7)] for i in range(4)}, 30) == []


def test_jitter_is_not_sustained_motion():
    pool = {i: [_point(i, 30 + (i % 2) * .3)] for i in range(12)}
    assert link(pool, 30) == []


def test_brown_candidates_inside_people_are_excluded():
    import cv2

    image = np.full((1080, 1920, 3), (40, 100, 45), np.uint8)
    cv2.ellipse(image, (900, 450), (11, 7), 0, 0, 360, (30, 55, 85), -1)
    h = Homography(np.array([[30., 0, 0], [0, 30., 0], [0, 0, 1.]]))
    got = candidates(image, [], h, 5)
    assert len(got) == 1
    assert np.linalg.norm(np.array(got[0].pixel) - [900, 450]) < 1
    assert candidates(image, [[870, 410, 930, 510]], h, 5) == []


def test_camera_pan_of_a_fixed_brown_object_is_not_a_flight():
    pool = {i: [Sighting(i, (900 + i*20, 450), (50, 20),
                         (0, 0, 20, 12), .8)] for i in range(12)}
    assert link(pool, 30) == []


def test_source_video_marker_requires_this_frames_pixel_evidence():
    from gridiron_tracking.verify import draw_ball_sighting

    image = np.zeros((200, 320, 3), np.uint8)
    observations = {5: {"pixel": (160, 100), "field": (40, 20)}}
    assert not draw_ball_sighting(image, 6, observations)
    assert not image.any(), "a missing sighting must not borrow a past ring"
    assert draw_ball_sighting(image, 5, observations)
    assert image.any()
