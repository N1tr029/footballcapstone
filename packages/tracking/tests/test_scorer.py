"""A scorer that cannot score a perfect answer as perfect is not measuring anything."""

import numpy as np
import pytest

from gridiron_tracking.bench.score import score
from gridiron_tracking.contract import PlayTrack, PlayerFrame, RosterEntry, TrackingFrame


def _play(n=30, players=6, shift=(0.0, 0.0), seed=0):
    rng = np.random.default_rng(seed)
    roster = {f"p{i}": RosterEntry(id=f"p{i}", team="offense") for i in range(players)}
    starts = rng.uniform([20, 5], [80, 48], size=(players, 2))
    vel = rng.uniform(-4, 4, size=(players, 2))
    frames = []
    for k in range(n):
        t = round(-1.0 + k * 0.1, 3)
        pf = {}
        for i in range(players):
            x, y = starts[i] + vel[i] * (t + 1.0)
            pf[f"p{i}"] = PlayerFrame(x=x + shift[0], y=y + shift[1], s=float(np.hypot(*vel[i])))
        frames.append(TrackingFrame(t=t, players=pf, events=["ball_snap"] if k == 10 else []))
    return PlayTrack(meta={"play_id": "synthetic"}, roster=roster, frames=frames)


def test_truth_against_itself_is_exactly_zero():
    truth = _play()
    s = score(truth, truth)
    assert s.median_yards == pytest.approx(0.0, abs=1e-12)
    assert s.p95_yards == pytest.approx(0.0, abs=1e-12)
    assert s.coverage == 1.0
    assert s.misses == 0 and s.false_positives == 0 and s.id_switches == 0
    assert s.mota == pytest.approx(1.0)
    assert s.meets()


def test_a_uniform_offset_is_reported_as_that_offset():
    truth = _play()
    s = score(_play(shift=(0.0, 0.75)), truth)
    assert s.median_yards == pytest.approx(0.75, abs=1e-9)


def test_dropping_players_costs_coverage_rather_than_flattering_the_error():
    truth = _play()
    thin = _play()
    for f in thin.frames:
        for pid in ("p0", "p1", "p2"):
            f.players.pop(pid, None)
    s = score(thin, truth)
    assert s.median_yards == pytest.approx(0.0, abs=1e-12)   # what is left is perfect
    assert s.coverage == pytest.approx(0.5, abs=1e-6)        # and half of it is missing
    assert s.misses == 3 * len(truth.frames)
    assert not s.meets() or s.coverage < 1.0


def test_swapping_two_players_is_counted_as_an_identity_failure():
    truth = _play()
    swapped = _play()
    half = len(swapped.frames) // 2
    for f in swapped.frames[half:]:
        f.players["p0"], f.players["p1"] = f.players["p1"], f.players["p0"]
    s = score(swapped, truth)
    assert s.id_switches >= 2, "a swap must show up as a switch, not as a small distance"


def test_output_somewhere_else_entirely_says_so_instead_of_reporting_a_number():
    truth = _play()
    s = score(_play(shift=(40.0, 0.0)), truth)
    assert s.coverage == 0.0
    assert any("not merely" in n for n in s.notes)


def test_frames_are_paired_on_time_not_on_index():
    truth = _play()
    late = _play()
    late.frames = late.frames[5:]            # same clock, different indexing
    s = score(late, truth)
    assert s.median_yards == pytest.approx(0.0, abs=1e-12)
