"""The loader, and the two conventions it exists to get right."""

import numpy as np
import pandas as pd
import pytest

from gridiron_tracking.contract import validate
from gridiron_tracking.sources import ngs


def _frame(play_direction=None, n=20):
    """A player running diagonally, with the NGS `dir` a release would publish.

    Diagonally on purpose. Motion straight along one axis cannot distinguish the four
    candidate angle conventions — two of them agree on it — so a fixture that runs
    due +y would let the wrong reading pass the convention check.
    """
    rows = []
    for k in range(n):
        rows.append({
            "gameId": 1, "playId": 7, "nflId": 99, "displayName": "Test",
            "jerseyNumber": 12, "team": "home", "frame.id": k + 1,
            # 3 yd/s in +x and 4 in +y: a compass bearing of atan2(3, 4) ~ 36.87.
            "x": 60.0 + 0.3 * k, "y": 10.0 + 0.4 * k, "s": 5.0, "dis": 0.5,
            "dir": 36.87,
            "event": "ball_snap" if k == 5 else None,
        })
        rows.append({
            "gameId": 1, "playId": 7, "nflId": None, "displayName": "football",
            "jerseyNumber": None, "team": "ball", "frame.id": k + 1,
            "x": 60.0 + 0.6 * k, "y": 26.0, "s": 0.0, "dis": 0.0, "dir": 0.0,
            "event": None,
        })
    df = pd.DataFrame(rows)
    if play_direction:
        df["playDirection"] = play_direction
    return ngs.load_tracking(df)


def test_columns_are_resolved_across_spellings():
    df = _frame()
    assert {"player_id", "frame", "x", "y", "dir"} <= set(df.columns)


def test_a_missing_required_column_names_what_it_found():
    with pytest.raises(ValueError, match="missing"):
        ngs.load_tracking(pd.DataFrame({"x": [1.0], "y": [2.0]}))


def test_compass_bearings_become_contract_degrees():
    # NGS 0 deg means +y; the contract calls that 90.
    assert ngs.ngs_angle_to_contract(0.0) == pytest.approx(90.0)
    assert ngs.ngs_angle_to_contract(90.0) == pytest.approx(0.0)
    assert ngs.ngs_angle_to_contract(180.0) == pytest.approx(270.0)


def test_the_convention_check_picks_the_right_hypothesis():
    """The empirical guard: difference the positions and see which reading agrees.
    This is how the convention was established in the first place, and it is kept
    as a test so a future release that changes it cannot pass silently."""
    res = ngs.check_angle_convention(_frame(), min_speed=1.0)
    best = min(res, key=res.get)
    assert best == "compass_cw_from_y"
    assert res[best] < 1.0
    # The claim that matters is separation: the right reading must win by a margin
    # no amount of differencing noise could close. How far the losers land depends on
    # the heading, so it is the gap that is asserted, not their absolute values.
    runner_up = min(v for k, v in res.items() if k != best)
    assert runner_up - res[best] > 15.0


def test_direction_survives_the_trip_into_the_contract():
    track = ngs.to_play_track(_frame(play_direction="right"), play_id=7)
    p = track.frames[10].players["99"]
    # Compass 36.87 is contract 53.13, and the motion is indeed atan2(4, 3).
    assert p.dir == pytest.approx(53.13, abs=0.01)


def test_a_left_running_play_is_mirrored_including_its_angles():
    right = ngs.to_play_track(_frame(play_direction="right"), play_id=7)
    left = ngs.to_play_track(_frame(play_direction="left"), play_id=7)
    r, l = right.frames[10].players["99"], left.frames[10].players["99"]
    assert l.x == pytest.approx(120.0 - r.x)
    assert l.y == pytest.approx(53.3 - r.y)
    # Mirroring through the centre is a 180 degree rotation, so headings turn too.
    assert l.dir == pytest.approx((r.dir + 180.0) % 360.0)


def test_a_release_without_orientation_says_so_rather_than_inventing_it():
    track = ngs.to_play_track(_frame(play_direction="right"), play_id=7)
    assert track.meta["orientation_measured"] is False
    assert all(p.o_source == "from_dir" for f in track.frames for p in f.players.values())


def test_the_snap_is_t_zero_and_the_ball_is_carried_through():
    track = ngs.to_play_track(_frame(play_direction="right"), play_id=7)
    assert track.event_time("ball_snap") == pytest.approx(0.0)
    assert track.frames[0].ball is not None
    assert validate(track).ok
