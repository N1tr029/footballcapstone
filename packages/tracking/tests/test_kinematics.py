"""Derivatives, checked against motion whose answer is known in closed form."""

import numpy as np
import pytest

from gridiron_tracking import kinematics
from gridiron_tracking.contract import shortest_deg


def assert_angles(got, want, atol=1e-8):
    """Compare angles the short way round.

    359.9999999999997 and 0.0 are the same heading and 360 apart as numbers, so a
    plain tolerance on the difference fails on a player running dead straight —
    which is the most ordinary thing there is on a football field.
    """
    err = np.array([abs(shortest_deg(g, want)) for g in np.atleast_1d(got)])
    assert err.max() <= atol, f"worst angle error {err.max()} deg"


def test_constant_velocity_gives_constant_speed_and_zero_acceleration():
    t = np.arange(0, 3, 0.1)
    xy = np.column_stack([20.0 + 5.0 * t, 26.0 + 0.0 * t])   # 5 yd/s downfield
    k = kinematics.derive(t, xy)
    assert np.allclose(k["s"], 5.0, atol=1e-8)
    assert np.allclose(k["a"], 0.0, atol=1e-8)
    assert_angles(k["dir"], 0.0)                              # 0 degrees is +x
    assert np.allclose(k["dis"][1:], 0.5, atol=1e-8)


def test_constant_acceleration_recovers_the_rate():
    t = np.arange(0, 3, 0.1)
    a = 3.0
    xy = np.column_stack([0.5 * a * t ** 2, np.full_like(t, 26.0)])
    k = kinematics.derive(t, xy)
    assert np.allclose(k["s"][2:-2], a * t[2:-2], atol=1e-6)
    assert np.allclose(k["a"][2:-2], a, atol=1e-6)


def test_direction_is_measured_toward_positive_y():
    t = np.arange(0, 2, 0.1)
    xy = np.column_stack([np.full_like(t, 50.0), 10.0 + 4.0 * t])
    k = kinematics.derive(t, xy)
    assert_angles(k["dir"], 90.0, atol=1e-6)


def test_a_stationary_player_does_not_spin():
    """A lineman in his stance has a velocity of nearly nothing, whose direction is
    numerically meaningless. Held, not recomputed — or the renderer draws him
    pirouetting."""
    rng = np.random.default_rng(0)
    t = np.arange(0, 2, 0.1)
    xy = np.column_stack([50 + rng.normal(0, 0.01, len(t)), 26 + rng.normal(0, 0.01, len(t))])
    k = kinematics.derive(t, xy)
    held = k["dir"]
    assert max(abs(shortest_deg(held[0], d)) for d in held) < 1e-6


def test_resample_never_invents_frames_outside_the_track():
    t = np.array([1.0, 1.1, 1.2])
    xy = np.column_stack([[10.0, 11.0, 12.0], [26.0, 26.0, 26.0]])
    grid = np.arange(0.0, 3.0, 0.1)
    t_out, pos = kinematics.resample(t, xy, grid)
    assert t_out.min() >= 1.0 and t_out.max() <= 1.2
    assert len(pos) == len(t_out)


def test_smoothing_keeps_a_straight_line_straight():
    t = np.arange(0, 3, 0.1)
    xy = np.column_stack([2.0 * t, np.full_like(t, 26.0)])
    assert np.allclose(kinematics.smooth(xy), xy, atol=1e-8)
