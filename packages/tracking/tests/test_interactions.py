"""Layer two: what happened between the players, derived from where they were."""

import json

import pytest

from gridiron_tracking import export
from gridiron_tracking import interactions as I
from gridiron_tracking.contract import Ball, PlayTrack, PlayerFrame, RosterEntry, TrackingFrame
from gridiron_tracking.sides import Sides


def _play(n=60, carry_from=0.0, tackle_at=3.0, with_ball=True):
    """A carrier running downfield, a would-be tackler converging, and a blocking
    pair who stay locked together the whole way."""
    roster = {
        "RB": RosterEntry(id="RB", team="home", name="carrier"),
        "C": RosterEntry(id="C", team="home", name="centre"),
        "OT": RosterEntry(id="OT", team="home", name="tackle"),
        "LB": RosterEntry(id="LB", team="away", name="linebacker"),
        "DE": RosterEntry(id="DE", team="away", name="end"),
    }
    frames = []
    for k in range(n):
        t = round(-1.0 + 0.1 * k, 3)
        run = max(t, 0.0)
        # The carrier runs downfield and is stopped at tackle_at.
        rb_x = 60.0 + 6.0 * min(run, tackle_at)
        rb_s = 6.0 if 0 < t < tackle_at else 0.2
        # The linebacker closes on him and arrives exactly at tackle_at.
        lb_x = 60.0 + 6.0 * min(run, tackle_at) + max(0.0, 8.0 * (tackle_at - run))
        lb_s = 6.5 if 0 < t < tackle_at else 0.2
        players = {
            "RB": PlayerFrame(x=rb_x, y=26.0, s=rb_s, dir=0.0, o=0.0, o_source="measured"),
            "C": PlayerFrame(x=61.0, y=20.0, s=0.5, dir=0.0, o=0.0, o_source="measured"),
            "OT": PlayerFrame(x=61.0, y=34.0, s=0.5, dir=0.0, o=0.0, o_source="measured"),
            "LB": PlayerFrame(x=lb_x, y=26.0, s=lb_s, dir=180.0, o=180.0, o_source="measured"),
            # The end is locked with the offensive tackle for the whole play.
            "DE": PlayerFrame(x=61.9, y=34.0, s=0.5, dir=180.0, o=180.0, o_source="measured"),
        }
        frames.append(TrackingFrame(
            t=t, players=players,
            ball=Ball(rb_x, 26.0) if with_ball else None,
            events=["ball_snap"] if k == 10 else [],
        ))
    return PlayTrack(meta={"play_id": "unit", "source": "ngs", "rate": 10}, roster=roster, frames=frames)


SIDES = Sides("home", "away", "caller", 1.0)


def test_the_carrier_is_found_from_the_ball_track():
    track = _play()
    ok, why = I.ball_track_usable(track, sides=SIDES)
    assert ok, why
    poss = I.possessions(track, sides=SIDES)
    held = [p for p in poss if p.player == "RB" and p.t_end - p.t_start > 1.0]
    assert held, [p.to_json() for p in poss]


def test_a_ball_column_that_names_defenders_is_rejected():
    """The failure that made a Patriots centre and a Chiefs defensive tackle share
    the carries on a real play."""
    track = _play()
    for f in track.frames:
        if f.t >= 0 and f.ball:
            f.ball = Ball(f.players["LB"].x, f.players["LB"].y)   # ball glued to a defender
    ok, why = I.ball_track_usable(track, sides=SIDES)
    assert not ok
    assert "defense" in why or "spot marker" in why


def test_possession_is_inferred_when_there_is_no_ball_at_all():
    """The vision pipeline's situation: no ball track, so convergence has to do it."""
    track = _play(with_ball=False)
    poss = I.infer_carrier(track, sides=SIDES)
    named = [p for p in poss if p.player is not None]
    assert named, "convergence found nobody"
    assert all(SIDES.team_of(track, p.player) == "offense" for p in named)
    longest = max(named, key=lambda p: p.t_end - p.t_start)
    assert longest.player == "RB"


def test_teammates_standing_together_are_not_blocking_each_other():
    """The centre and the offensive tackle are close all play. Without the
    opposite-sides rule the output is mostly the offensive line noticing itself."""
    engs = I.engagements(_play(), sides=SIDES)
    for e in engs:
        ta, tb = SIDES.team_of(_play(), e.a), SIDES.team_of(_play(), e.b)
        assert not (ta is not None and ta == tb), f"{e.a} vs {e.b} are teammates"


def test_a_sustained_cross_team_engagement_is_one_block_not_many():
    """A single jittery frame used to end a block and start a new one, turning
    sixteen engagements into thirty-five."""
    engs = I.engagements(_play(), sides=SIDES)
    ot_de = [e for e in engs if {e.a, e.b} == {"OT", "DE"}]
    assert len(ot_de) == 1, [e.to_json() for e in ot_de]
    assert ot_de[0].kind == "block"
    assert ot_de[0].duration > 3.0


def test_engagements_with_the_carrier_are_tackles_not_blocks():
    """A defender engaging the ball carrier is never blocking him."""
    engs = I.engagements(_play(), sides=SIDES)
    rb_lb = [e for e in engs if {e.a, e.b} == {"RB", "LB"}]
    assert rb_lb, "the tackler never registered"
    assert rb_lb[0].kind in ("tackle", "tackle_attempt")
    assert rb_lb[0].involved_carrier


def test_first_contact_and_going_down_are_recorded_separately():
    """They are a median of 0.9 s apart on real plays; conflating them costs most of
    a second."""
    engs = I.engagements(_play(tackle_at=3.0), sides=SIDES)
    tackles = [e for e in engs if e.kind == "tackle"]
    assert tackles, [e.to_json() for e in engs]
    e = tackles[0]
    assert e.t_down is not None
    assert e.t_down >= e.t_start


def test_ball_in_view_is_labelled_with_what_it_was_inferred_from():
    views = I.ball_in_view(_play())
    assert views
    assert all(v.confidence in ("measured", "from_dir", "from_velocity", "assumed") for v in views)
    doc = I.Interactions(views=views).to_json()
    assert all(v["inferred"] is True for v in doc["ball_in_view"])
    assert "not measured" in doc["caveats"]["ball_in_view"]


def test_derive_reports_how_the_sides_were_decided():
    ix = I.derive(_play(), offense="home")
    assert ix.sides.offense == "home"
    assert ix.sides.method == "caller"
    assert any("ball track" in n for n in ix.notes)
