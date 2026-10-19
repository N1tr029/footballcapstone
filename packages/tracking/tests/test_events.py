"""Finding the throw and the arrival in a ball track, and refusing to.

The refusals are the important half. ``gridiron_counterfactual.review`` forecasts
from the throw to the arrival, and with the window in the wrong place the model
returns numbers without complaint — 61 of 67 real throws read as mistakes. So a
derived window has to be right or absent, and these tests are mostly about absent.
"""

import math

import pytest

from gridiron_tracking import events
from gridiron_tracking.contract import (
    Ball, PlayTrack, PlayerFrame, RosterEntry, TrackingFrame,
)


def play(ball_path, player_paths=None, measured=0.6, n=20):
    """``ball_path`` is a list of (x, y) or None, one per frame."""
    players = player_paths or {
        "qb": [(40.0, 26.0)] * n,
        "wr": [(40.0 + i * 1.2, 34.0) for i in range(n)],
        "cb": [(42.0 + i * 1.2, 36.0) for i in range(n)],
    }
    roster = {pid: RosterEntry(id=pid, team="teamA" if pid != "cb" else "teamB")
              for pid in players}
    frames = []
    for i in range(n):
        pf = {pid: PlayerFrame(x=pts[i][0], y=pts[i][1], s=3.0, o_source="from_velocity")
              for pid, pts in players.items() if i < len(pts) and pts[i]}
        b = ball_path[i] if i < len(ball_path) else None
        frames.append(TrackingFrame(
            t=round(i / 10, 3), players=pf,
            ball=Ball(b[0], b[1]) if b else None,
            events=["ball_snap"] if i == 0 else [],
        ))
    return PlayTrack(
        meta={"source": "vision", "rate": 10,
              "ball": {"measured_fraction": measured}},
        roster=roster, frames=frames)


def a_pass(n=20, throw_at=5, land_at=13):
    """Held by the quarterback, then across the field to the receiver."""
    path = []
    for i in range(n):
        if i <= throw_at:
            path.append((40.0, 26.0))                       # in his hands
        elif i < land_at:
            f = (i - throw_at) / (land_at - throw_at)
            path.append((40.0 + f * 16.0, 26.0 + f * 8.0))  # in the air
        else:
            path.append((40.0 + i * 1.2 - 2.4, 34.0))       # with the receiver
    return path


def test_a_thrown_ball_gives_a_throw_and_an_arrival():
    track = play(a_pass())
    found = events.apply(track)

    assert found.found
    assert found.thrown_at == pytest.approx(0.5, abs=0.11)
    assert found.arrived_at == pytest.approx(1.3, abs=0.21)
    assert found.flight_yards > events.MIN_FLIGHT_DISTANCE
    assert found.confidence > 0.5

    labelled = {e for f in track.frames for e in f.events}
    assert "pass_forward" in labelled
    assert "pass_arrived" in labelled
    assert track.event_time("pass_forward") < track.event_time("pass_arrived")


def test_a_ball_nobody_ever_lets_go_of_is_a_run():
    """Inventing a throw here hands the model a receiver nobody threw to."""
    track = play([(40.0 + i * 1.0, 26.0) for i in range(20)],
                 player_paths={"rb": [(40.0 + i * 1.0, 26.3) for i in range(20)],
                               "lb": [(50.0 - i * 0.5, 27.0) for i in range(20)]})
    found = events.apply(track)

    assert not found.found
    assert any("looks like a run" in n for n in found.notes)
    assert "pass_forward" not in {e for f in track.frames for e in f.events}


def test_a_handoff_is_not_a_throw():
    """It also leaves one player and reaches another. Duration and distance are
    what separate them, and a shovel pass sits on that boundary and is missed —
    which is the right error: a handoff called a throw puts the forecast window in
    completely the wrong place."""
    path = [(40.0, 26.0)] * 6 + [(41.0, 26.0), (42.0, 26.0)] + \
           [(42.0 + i, 26.0) for i in range(12)]
    track = play(path, player_paths={
        "qb": [(40.0, 26.0)] * 20,
        "rb": [(43.0 + i * 0.9, 26.2) for i in range(20)],
        "lb": [(60.0 - i * 0.7, 27.0) for i in range(20)],
    })
    found = events.apply(track)
    assert not found.found
    assert any("handoff or a pitch" in n for n in found.notes)


def test_a_mostly_coasted_ball_is_refused():
    """The speeds being measured would be the interpolation's, not the ball's."""
    track = play(a_pass(), measured=0.05)
    found = events.apply(track)

    assert not found.found
    assert any("actually seen on 5%" in n for n in found.notes)


def test_a_static_detection_is_not_a_throw():
    """The reference clip's failure: the generic sports-ball class finds the
    down-and-distance graphic, which sits near the line and never moves. It was
    detected on 42% of frames and travelled 2.4 yards in 2.6 seconds."""
    track = play([(71.0 + 0.02 * i, 30.5) for i in range(20)],
                 player_paths={"a": [(60.0, 20.0)] * 20,
                               "b": [(80.0, 40.0)] * 20})
    found = events.apply(track)

    assert not found.found
    assert found.flight_yards < events.MIN_FLIGHT_DISTANCE


def test_a_play_with_no_ball_at_all_says_so():
    track = play([None] * 20)
    found = events.apply(track)
    assert not found.found
    assert any("too few frames" in n for n in found.notes)


def test_the_result_is_recorded_on_the_track_either_way():
    for path in (a_pass(), [None] * 20):
        track = play(path)
        events.apply(track)
        assert "pass_events" in track.meta
        assert track.meta["pass_events"]["source"].startswith("derived")


def test_applying_twice_does_not_double_label():
    track = play(a_pass())
    events.apply(track)
    events.apply(track)
    counts = [f.events.count("pass_forward") for f in track.frames]
    assert max(counts) == 1
