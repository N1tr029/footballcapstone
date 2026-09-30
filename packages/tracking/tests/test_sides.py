"""Which team has the ball — the fact almost everything downstream needs."""

import pytest

from gridiron_tracking.contract import Ball, PlayTrack, PlayerFrame, RosterEntry, TrackingFrame
from gridiron_tracking.sides import infer_sides


def _snap(centre_team="home", ball=(60.0, 26.0), presnap_speeds=(0.05, 0.4)):
    """A snap with a centre bent over the ball and a nose tackle just beyond him."""
    roster = {
        "C": RosterEntry(id="C", team=centre_team, name="centre"),
        "G": RosterEntry(id="G", team=centre_team, name="guard"),
        "NT": RosterEntry(id="NT", team="away" if centre_team == "home" else "home", name="tackle"),
        "LB": RosterEntry(id="LB", team="away" if centre_team == "home" else "home", name="linebacker"),
    }
    off_s, def_s = presnap_speeds
    frames = []
    for k in range(12):
        t = round(-0.6 + 0.1 * k, 3)
        frames.append(TrackingFrame(
            t=t,
            players={
                "C": PlayerFrame(x=ball[0] - 0.4, y=ball[1], s=off_s),
                "G": PlayerFrame(x=ball[0] - 0.5, y=ball[1] + 2.0, s=off_s),
                "NT": PlayerFrame(x=ball[0] + 1.4, y=ball[1] + 0.3, s=def_s),
                "LB": PlayerFrame(x=ball[0] + 5.0, y=ball[1], s=def_s),
            },
            ball=Ball(*ball),
            events=["ball_snap"] if k == 6 else [],
        ))
    return PlayTrack(roster=roster, frames=frames)


def test_the_centre_gives_away_the_offense():
    s = infer_sides(_snap("home"))
    assert s.offense == "home" and s.defense == "away"
    assert s.method == "ball_at_snap"
    assert s.confidence > 0.95


def test_it_works_the_other_way_round_too():
    assert infer_sides(_snap("away")).offense == "away"


def test_a_caller_who_knows_is_believed_outright():
    s = infer_sides(_snap("home"), offense="away")
    assert s.offense == "away" and s.method == "caller" and s.confidence == 1.0


def test_with_no_ball_it_falls_back_to_stillness_and_says_so():
    """A vision pipeline has no ball track. An 81% guess that announces itself is
    workable; a 98% guess that hides a missing basis is not."""
    track = _snap("home")
    for f in track.frames:
        f.ball = None
    s = infer_sides(track)
    assert s.offense == "home"
    assert s.method == "presnap_stillness"
    assert s.confidence == pytest.approx(0.812)


def test_side_lookup_maps_players_to_offense_and_defense():
    track = _snap("home")
    s = infer_sides(track)
    assert s.team_of(track, "C") == "offense"
    assert s.team_of(track, "NT") == "defense"
    assert s.team_of(track, "nobody") is None
