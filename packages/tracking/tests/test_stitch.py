"""Stitching joins fragments of one player and refuses to join two."""

import numpy as np

from gridiron_tracking.contract import PlayTrack, PlayerFrame, RosterEntry, TrackingFrame
from gridiron_tracking.stitch import stitch


def _track(paths: dict[str, list[tuple[float, float, float]]]) -> PlayTrack:
    ts = sorted({round(t, 3) for p in paths.values() for t, _, _ in p})
    frames = {t: TrackingFrame(t=t) for t in ts}
    roster = {}
    for pid, pts in paths.items():
        roster[pid] = RosterEntry(id=pid, team="teamA", label=pid)
        for t, x, y in pts:
            frames[round(t, 3)].players[pid] = PlayerFrame(x=x, y=y, s=0, a=0, dir=0, o=0)
    return PlayTrack(meta={}, roster=roster, frames=[frames[t] for t in ts])


def _line(t0, n, x0, y0, vx, vy, dt=0.1):
    return [(round(t0 + i * dt, 3), x0 + vx * i * dt, y0 + vy * i * dt) for i in range(n)]


def test_joins_a_player_who_kept_running_through_a_gap():
    # One man at 6 yd/s, lost for 0.5s in the middle of his run.
    t = _track({
        "a": _line(0.0, 10, 20.0, 25.0, 6.0, 0.0),
        "b": _line(1.4, 10, 26.4, 25.0, 6.0, 0.0),
    })
    rep = stitch(t)
    assert rep.merged == 1, "a fragment continuing the same motion must be rejoined"
    assert len(t.roster) == 1
    # Every sample survives the merge; a join that loses frames is not a join.
    assert sum(1 for f in t.frames if "a" in f.players) == 20


def test_refuses_two_players_on_the_field_at_once():
    t = _track({
        "a": _line(0.0, 20, 20.0, 25.0, 0.0, 0.0),
        "b": _line(0.0, 20, 21.0, 25.0, 0.0, 0.0),
    })
    rep = stitch(t)
    assert rep.merged == 0, "overlapping tracks are two people however close they are"
    assert len(t.roster) == 2


def test_refuses_a_join_that_would_need_superhuman_speed():
    t = _track({
        "a": _line(0.0, 10, 20.0, 25.0, 0.0, 0.0),
        "b": _line(1.0, 10, 45.0, 25.0, 0.0, 0.0),
    })
    rep = stitch(t)
    assert rep.merged == 0, "25 yards in a tenth of a second is not a player"


def test_refuses_a_bystander_standing_near_where_somebody_vanished():
    # 'a' is running hard when it ends; 'b' appears where 'a' *was*, not where it was
    # going. Proximity alone would join them; the prediction must not.
    t = _track({
        "a": _line(0.0, 10, 20.0, 25.0, 9.0, 0.0),
        "b": _line(1.2, 10, 20.5, 25.0, 0.0, 0.0),
    })
    rep = stitch(t)
    assert rep.merged == 0, "the gap must be bridged by the motion, not by nearness"


def test_chains_three_fragments_into_one():
    t = _track({
        "a": _line(0.0, 6, 10.0, 20.0, 5.0, 0.0),
        "b": _line(1.0, 6, 15.0, 20.0, 5.0, 0.0),
        "c": _line(2.0, 6, 20.0, 20.0, 5.0, 0.0),
    })
    rep = stitch(t)
    assert rep.merged == 2 and len(t.roster) == 1, "a chain must collapse to one player"
