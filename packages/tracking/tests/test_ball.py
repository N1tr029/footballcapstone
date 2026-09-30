"""The ball tracker — physics gating, and honest sourcing."""

import numpy as np
import pytest

from gridiron_tracking import ball as B


def _thrown(n=30, dt=0.1):
    """A ball leaving a quarterback at 18 yd/s across the field."""
    return [(40.0 + 18.0 * dt * k, 20.0 + 6.0 * dt * k) for k in range(n)]


def test_a_clean_flight_is_tracked_as_detected():
    path = _thrown()
    times = [round(k * 0.1, 3) for k in range(len(path))]
    cands = {t: [(x, y, 0.5)] for t, (x, y) in zip(times, path)}
    tr = B.track_ball(times, cands, dt=0.1)
    assert tr.counts().get("detected", 0) >= len(path) - 2


def test_a_gap_is_coasted_not_dropped():
    path = _thrown()
    times = [round(k * 0.1, 3) for k in range(len(path))]
    cands = {t: [(x, y, 0.5)] for t, (x, y) in zip(times, path)}
    for t in times[10:14]:
        cands[t] = []                      # four frames behind a lineman
    tr = B.track_ball(times, cands, dt=0.1)
    assert tr.counts().get("coasted", 0) >= 4
    for s in tr.samples[10:14]:
        assert s.source == "coasted" and np.isfinite(s.x)


def test_an_impossible_jump_is_refused():
    """The failure this gate exists for: a wider gate accepted more detections and
    produced a ball moving at 61 yd/s, which is not a thing that happens."""
    path = _thrown()
    times = [round(k * 0.1, 3) for k in range(len(path))]
    cands = {t: [(x, y, 0.5)] for t, (x, y) in zip(times, path)}
    cands[times[15]] = [(95.0, 50.0, 0.9)]          # a confident detection 50 yd away
    tr = B.track_ball(times, cands, dt=0.1)
    assert tr.samples[15].source != "detected", "a 50-yard jump must not be believed"
    speeds = [np.hypot(b.x - a.x, b.y - a.y) / 0.1
              for a, b in zip(tr.samples, tr.samples[1:])
              if np.isfinite([a.x, a.y, b.x, b.y]).all()]
    assert max(speeds) <= B.MAX_SPEED_YPS


def test_coasting_expires_rather_than_inventing_a_ball():
    path = _thrown(n=8)
    times = [round(k * 0.1, 3) for k in range(30)]
    cands = {t: [] for t in times}
    for t, (x, y) in zip(times[:8], path):
        cands[t] = [(x, y, 0.5)]
    tr = B.track_ball(times, cands, dt=0.1)
    assert tr.samples[-1].source == "none"
    assert not np.isfinite(tr.samples[-1].x)


def test_noise_far_from_every_player_is_ignored():
    times = [round(k * 0.1, 3) for k in range(10)]
    players = {t: np.array([[50.0, 26.0], [52.0, 28.0]]) for t in times}
    cands = {t: [(5.0, 2.0, 0.4)] for t in times}       # a corner of the field, nobody there
    tr = B.track_ball(times, cands, players, dt=0.1)
    assert tr.counts().get("detected", 0) == 0


def test_every_sample_says_how_it_was_arrived_at():
    """A renderer that draws a sighting and a coast identically is lying about one."""
    times = [round(k * 0.1, 3) for k in range(6)]
    cands = {t: [(40.0 + k, 26.0, 0.4)] for k, t in enumerate(times)}
    tr = B.track_ball(times, cands, dt=0.1)
    assert all(s.source in ("detected", "coasted", "carrier", "none") for s in tr.samples)
    assert tr.samples[0].measured is (tr.samples[0].source == "detected")
