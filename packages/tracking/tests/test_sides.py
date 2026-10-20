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


# ------------------------------------------- possession from the broadcast's lines


def _lined_up(los: float, attacks: str, n: int = 11):
    """Offense behind the line, defense in front of it, as at any snap."""
    from gridiron_tracking.contract import (
        Ball, PlayTrack, PlayerFrame, RosterEntry, TrackingFrame,
    )

    sign = 1.0 if attacks == "right" else -1.0
    roster = {}
    players = {}
    for i in range(n):
        off, deff = f"o{i}", f"d{i}"
        roster[off] = RosterEntry(id=off, team="white")
        roster[deff] = RosterEntry(id=deff, team="purple")
        players[off] = PlayerFrame(x=los - sign * (1.0 + i * 0.2), y=10.0 + i * 3,
                                   o_source="from_velocity")
        players[deff] = PlayerFrame(x=los + sign * (2.0 + i * 0.2), y=10.0 + i * 3,
                                    o_source="from_velocity")
    frames = [TrackingFrame(t=0.0, players=players, ball=None, events=["ball_snap"])]
    return PlayTrack(meta={"source": "vision", "rate": 10}, roster=roster, frames=frames)


def test_possession_from_the_line_when_there_is_no_ball_and_no_presnap():
    """The case every broadcast clip is in: cut near the snap, no ball track.

    Before this path existed infer_sides returned the first team alphabetically at
    zero confidence, and a caller who ignored the confidence got the offense wrong.
    """
    from gridiron_tracking import sides

    for attacks in ("left", "right"):
        track = _lined_up(los=46.7, attacks=attacks)
        got = sides.infer_sides(track, los_x=46.7, attacks=attacks)
        assert got.offense == "white", f"{attacks}: got {got.offense}"
        assert got.defense == "purple"
        assert got.method == "behind_the_los"
        assert got.confidence > 0.9


def test_an_unseparated_snap_is_reported_as_unseparated():
    """Both sides on one side of the line means the line, the snap frame or the
    direction is wrong. A coin flip dressed as a measurement is worse than nothing."""
    from gridiron_tracking import sides
    from gridiron_tracking.contract import PlayerFrame, TrackingFrame

    track = _lined_up(los=46.7, attacks="right")
    # Shove everybody downfield of the line, as a clip cut after the snap does.
    for f in track.frames:
        for p in f.players.values():
            p.x += 12.0

    got = sides.infer_sides(track, los_x=46.7, attacks="right")
    assert got.method == "behind_the_los_unseparated"
    assert got.confidence <= 0.5


def test_a_hint_is_ignored_when_the_caller_states_the_offense():
    from gridiron_tracking import sides

    track = _lined_up(los=46.7, attacks="right")
    got = sides.infer_sides(track, offense="purple", los_x=46.7, attacks="right")
    assert got.offense == "purple"
    assert got.method == "caller"


# ------------------------------------------------------------------ mirroring


def test_mirroring_turns_a_play_round_and_is_its_own_inverse():
    from gridiron_tracking import field

    track = _lined_up(los=46.7, attacks="left")
    before = {pid: (p.x, p.y, p.dir, p.o)
              for pid, p in track.frames[0].players.items()}

    field.mirror_track(track)
    assert track.meta["mirrored"] is True
    # The source ran left; the coordinates now run right. play_direction names the
    # former, because that is what tells features how to undo it.
    assert track.meta["play_direction"] == "left"
    moved = track.frames[0].players[next(iter(before))]
    assert moved.x == pytest.approx(field.LENGTH - before[next(iter(before))][0])

    field.mirror_track(track)
    assert track.meta["mirrored"] is False
    assert track.meta["play_direction"] == "right"
    for pid, p in track.frames[0].players.items():
        x, y, d, o = before[pid]
        assert p.x == pytest.approx(x)
        assert p.y == pytest.approx(y)
        assert p.dir == pytest.approx(d)
        assert p.o == pytest.approx(o)


def test_mirroring_turns_angles_by_a_hundred_and_eighty():
    """A 180 degree rotation of the field is a 180 degree turn for every heading.
    Mirroring positions without the angles leaves everyone running backwards."""
    from gridiron_tracking import field
    from gridiron_tracking.contract import PlayerFrame

    track = _lined_up(los=46.7, attacks="right")
    p = track.frames[0].players["o0"]
    p.dir, p.o = 30.0, 350.0

    field.mirror_track(track)
    assert p.dir == pytest.approx(210.0)
    assert p.o == pytest.approx(170.0)
