"""The broadcaster's two lines, and the three facts they establish.

``check_markers`` was written to check a registration's scale and threw away
everything except the gap between the lines. Those two numbers are also the only
automatic source of the line of scrimmage, the distance, and — because the line to
gain is always downfield — which way the offense is attacking. On a clip cut near
the snap with no ball track there is no other answer to that last one, and
everything else about the play depends on it.

So the derived properties are pinned here, including on an image built for the
purpose: a blue stripe and a yellow stripe through a known homography, which
exercises the colour masks and the projection rather than only the arithmetic.
"""

import numpy as np
import pytest

from gridiron_tracking.broadcast import FIRST_DOWN_HSV, LOS_HSV, MarkerCheck, check_markers


def marks(pairs, stated=10.0, scatter=0.1):
    """``(los_x, first_down_x)`` per frame, as check_markers records them.

    Scatter is filled in because check_markers always appends it alongside a
    sample — and ``ok`` reads NaN straightness as not-ok, correctly, so a fixture
    that omits it is testing a state the real code cannot produce.
    """
    m = MarkerCheck(stated_yards=stated)
    m.samples = [(i * 10, a, b) for i, (a, b) in enumerate(pairs)]
    m.scatter = [scatter] * (2 * len(m.samples))
    return m


# ----------------------------------------------------------------- the numbers


def test_the_lines_are_medians_so_one_bad_frame_cannot_move_them():
    """A frame where a mask caught a sideline should not drag the answer."""
    m = marks([(47.0, 57.0), (47.1, 57.2), (46.9, 56.8), (12.0, 99.0)])
    assert m.line_of_scrimmage == pytest.approx(47.0, abs=0.1)
    assert m.first_down_line == pytest.approx(57.1, abs=0.3)
    assert m.attacks == "right"


def test_no_samples_is_not_an_answer():
    m = marks([])
    assert np.isnan(m.line_of_scrimmage)
    assert np.isnan(m.first_down_line)
    assert m.attacks is None
    s = m.situation()
    assert s["lineOfScrimmage"] is None and s["attacks"] is None and s["frames"] == 0


@pytest.mark.parametrize("los,fd,expect", [
    (47.0, 57.0, "right"),      # line to gain downfield of the line, toward +x
    (57.0, 47.0, "left"),
    (47.0, 47.2, None),         # indistinguishable: not an answer
    (20.0, 30.0, "right"),
])
def test_direction_comes_from_which_side_the_line_to_gain_is_on(los, fd, expect):
    assert marks([(los, fd)] * 3).attacks == expect


def test_the_situation_block_carries_what_prepare_reads():
    m = marks([(47.0, 53.0)] * 4, stated=6.0)
    s = m.situation()
    assert s["lineOfScrimmage"] == pytest.approx(47.0)
    assert s["firstDownLine"] == pytest.approx(53.0)
    assert s["yardsToGo"] == 6.0
    assert s["attacks"] == "right"
    assert s["frames"] == 4
    # The stated distance is the operator's reading of the graphic; the measured
    # gap is what the registration thinks. They are reported separately on purpose.
    assert s["measured_gap_yards"] == pytest.approx(6.0)
    assert "broadcaster's own virtual lines" in s["source"]


def test_stated_and_measured_are_not_conflated():
    """The graphic says 6; the registration measures 5.7. Both are reported, and
    the error between them is the registration's scale check."""
    m = marks([(47.0, 52.7)] * 3, stated=6.0)
    assert m.situation()["yardsToGo"] == 6.0
    assert m.measured == pytest.approx(5.7, abs=0.01)
    assert m.error_yards == pytest.approx(-0.3, abs=0.01)
    assert m.ok is True          # inside the half-yard the source can resolve


def test_a_scale_error_past_what_the_source_can_excuse_is_not_ok():
    m = marks([(47.0, 55.0)] * 3, stated=6.0)   # measured 8 against a stated 6
    assert not m.ok
    assert "OUTSIDE" in m.table()


def test_a_crooked_line_fails_even_when_the_distance_is_right():
    """A drawn line is one yard line, so its field x must be the same down its
    whole length. Scatter there is perspective error the distance cannot see."""
    straight = marks([(47.0, 53.0)] * 3, stated=6.0, scatter=0.1)
    crooked = marks([(47.0, 53.0)] * 3, stated=6.0, scatter=1.4)
    assert straight.ok
    assert not crooked.ok
    assert crooked.error_yards == pytest.approx(0.0, abs=1e-9)


def test_straightness_with_nothing_recorded_is_not_a_pass():
    """NaN must not read as 'within tolerance'."""
    m = MarkerCheck(stated_yards=6.0)
    m.samples = [(0, 47.0, 53.0)]
    assert np.isnan(m.straightness)
    assert not m.ok


# ------------------------------------------------------- the masks and the fit


class _Flat:
    """A homography that maps image x straight to field x, for the mask test."""

    def __init__(self, scale=0.05):
        self.scale = scale

    def at(self, _frame):
        return self

    def to_field(self, pts):
        pts = np.atleast_2d(np.asarray(pts, dtype=float))
        return np.column_stack([pts[:, 0] * self.scale, pts[:, 1] * self.scale])


def _painted(width=1280, height=720, los_px=600, fd_px=800):
    """A green field with a blue line and a yellow line down it."""
    cv2 = pytest.importorskip("cv2")
    img = np.zeros((height, width, 3), np.uint8)
    img[:, :] = (60, 120, 60)                       # BGR turf

    def stripe(x, hsv_lo):
        h, s, v = hsv_lo[0] + 4, 220, 220
        patch = np.full((height, 7, 3), (h, s, v), np.uint8)
        img[:, x - 3:x + 4] = cv2.cvtColor(patch, cv2.COLOR_HSV2BGR)

    stripe(los_px, LOS_HSV[0])
    stripe(fd_px, FIRST_DOWN_HSV[0])
    return img


def test_the_masks_find_the_two_lines_and_place_them_in_yards():
    """End to end over the colour masks, not just the arithmetic.

    Blue is the line of scrimmage and yellow is the line to gain — which is the one
    assumption in this module that a reader cannot check from the numbers.
    """
    pytest.importorskip("cv2")
    frames = {i: _painted(los_px=600, fd_px=800) for i in range(0, 30, 10)}
    reg = _Flat(scale=0.05)                        # 20 px to the yard

    got = check_markers(frames, reg, stated_yards=10.0, every=1, min_points=5)

    assert len(got.samples) == len(frames)
    assert got.line_of_scrimmage == pytest.approx(600 * 0.05, abs=0.3)
    assert got.first_down_line == pytest.approx(800 * 0.05, abs=0.3)
    assert got.measured == pytest.approx(10.0, abs=0.5)
    assert got.attacks == "right"
    assert got.ok


def test_the_line_to_gain_behind_the_line_reads_as_attacking_left():
    pytest.importorskip("cv2")
    frames = {i: _painted(los_px=800, fd_px=600) for i in range(0, 30, 10)}
    got = check_markers(frames, _Flat(0.05), stated_yards=10.0, every=1, min_points=5)
    assert got.attacks == "left"


def test_a_frame_with_only_one_line_is_skipped_not_guessed():
    """The scale check needs both; so does the direction."""
    pytest.importorskip("cv2")
    cv2 = pytest.importorskip("cv2")
    both = _painted(los_px=600, fd_px=800)
    one = _painted(los_px=600, fd_px=800)
    one[:, 790:810] = (60, 120, 60)               # paint the yellow line out

    got = check_markers({0: both, 10: one}, _Flat(0.05), stated_yards=10.0,
                        every=1, min_points=5)
    assert len(got.samples) == 1, "a frame missing a line must not contribute"
