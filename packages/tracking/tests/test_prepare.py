"""The order the steps go in, and the blockers it is supposed to name.

Each step has its own tests; what this file checks is that they compose in the one
order that works, and that a track which cannot be reviewed says why in terms
someone can act on rather than failing three modules later.
"""

import pytest

from gridiron_tracking import field, prepare
from gridiron_tracking.contract import (
    Ball, PlayTrack, PlayerFrame, RosterEntry, TrackingFrame,
)


def clip(attacks="left", los=47.0, with_ball=False, n_frames=12, extra_kit=False,
         togo=6, ball_measured=0.9):
    """A vision-shaped track: anonymous ids, kit labels, offense behind the line."""
    sign = 1.0 if attacks == "right" else -1.0
    roster, base = {}, {}
    for i in range(11):
        roster[f"o{i}"] = RosterEntry(id=f"o{i}", team="teamA")
        roster[f"d{i}"] = RosterEntry(id=f"d{i}", team="teamB")
        base[f"o{i}"] = (los - sign * (1.0 + i * 0.3), 8.0 + i * 3.5)
        base[f"d{i}"] = (los + sign * (2.5 + i * 0.3), 8.0 + i * 3.5)
    if extra_kit:
        roster["x0"] = RosterEntry(id="x0", team="kitA")
        base["x0"] = (los, 2.0)

    frames = []
    for i in range(n_frames):
        players = {
            pid: PlayerFrame(x=x + sign * i * 0.25, y=y, s=2.0, dir=0.0, o=0.0,
                             o_source="from_velocity")
            for pid, (x, y) in base.items()
        }
        frames.append(TrackingFrame(
            t=round((i - 3) / 10, 3), players=players,
            ball=Ball(los, 26.0) if with_ball else None,
            events=["ball_snap"] if i == 3 else [],
        ))
    meta = {
        "source": "vision", "rate": 10,
        **({"ball": {"measured_fraction": ball_measured,
                      "travelled_yards": 30.0}} if with_ball else {}),
        "broadcast": {
            "lineOfScrimmage": los,
            "firstDownLine": los - sign * togo,
            "yardsToGo": togo,
            "attacks": attacks,
        },
    }
    return PlayTrack(meta=meta, roster=roster, frames=frames)


def test_a_left_running_clip_comes_out_attacking_plus_x():
    """The silent one: nothing complains about a play running the wrong way."""
    track = clip(attacks="left", los=47.0)
    out, report = prepare.prepare(track)

    assert report.mirrored is True
    assert out.meta["play_direction"] == "right"
    # The line of scrimmage moves with everybody, or possession was decided against
    # a line that is no longer where the players are.
    assert report.line_of_scrimmage == pytest.approx(field.LENGTH - 47.0)
    assert out.meta["lineOfScrimmage"] == pytest.approx(field.LENGTH - 47.0, abs=0.01)

    offense = [pid for pid, e in out.roster.items() if e.team == out.meta["possessionTeam"]]
    snap = out.frame_at(0.0)
    assert all(snap.players[p].x < out.meta["lineOfScrimmage"] for p in offense)


def test_a_right_running_clip_is_left_alone():
    track = clip(attacks="right", los=47.0)
    out, report = prepare.prepare(track)
    assert report.mirrored is False
    assert report.line_of_scrimmage == pytest.approx(47.0)


def test_possession_is_decided_and_recorded():
    out, report = prepare.prepare(clip())
    assert report.sides.offense == "teamA"
    assert report.sides.method == "behind_the_los"
    assert out.meta["possessionTeam"] == "teamA"
    assert out.meta["defensiveTeam"] == "teamB"


def test_the_third_kit_label_is_resolved_before_sides_is_asked():
    """Order matters: sides reports zero confidence on three labels."""
    out, report = prepare.prepare(clip(extra_kit=True))
    assert report.roster.dropped_unlabelled == 1
    assert report.sides.confidence > 0.9


def test_a_missing_ball_is_named_as_the_blocker_it_is():
    out, report = prepare.prepare(clip(with_ball=False))
    assert not report.reviewable
    assert report.has_ball is False
    assert any("no ball track" in b for b in report.blockers)
    assert any("track.ball = true" in b for b in report.blockers)


def test_a_sparse_ball_is_a_blocker_not_a_pass():
    """Presence is not the test. A ball on a handful of frames is worse than none:
    every derivation downstream runs, returns a number, and is wrong."""
    out, report = prepare.prepare(clip(with_ball=True, ball_measured=0.05))

    assert report.has_ball is True
    assert not report.reviewable
    assert any("actually seen on 5%" in b for b in report.blockers)


def test_a_clip_with_everything_is_reviewable():
    out, report = prepare.prepare(clip(with_ball=True))
    assert report.reviewable, report.blockers
    assert report.has_ball
    assert not report.blockers
    assert out.meta["firstDownLine"] == pytest.approx(
        out.meta["lineOfScrimmage"] + 6, abs=0.01)


def test_a_coach_upload_gets_its_line_from_the_formation():
    """No virtual lines and no graphic — three of the project's clips are this.

    The line then comes off the two front ranks at the snap, and the report says
    which of the two sources it used, because one is a surveyed line drawn by a
    broadcaster and the other is an estimate from where people were standing.
    """
    track = clip(with_ball=True, attacks="right", los=47.0)
    del track.meta["broadcast"]
    out, report = prepare.prepare(track)

    assert report.line_source == "formation"
    assert report.line_of_scrimmage == pytest.approx(47.0, abs=2.0)
    assert out.meta["formation_line"]["attacks"] == "right"
    assert report.reviewable, report.blockers
    # The distance is genuinely unavailable without a graphic, and that is a note
    # rather than a blocker: the review only uses the down to choose between
    # recommending a run and recommending a punt.
    assert any("no distance to go" in n for n in report.notes)
    assert not any("distance" in b for b in report.blockers)


def test_the_broadcast_line_is_preferred_over_the_formation():
    track = clip(with_ball=True, attacks="right", los=47.0)
    out, report = prepare.prepare(track)
    assert report.line_source == "broadcast"
    assert "formation_line" not in out.meta


def test_a_frame_that_is_not_a_snap_is_called_out():
    """A clip cut after the ball is gone has both teams mixed together."""
    track = clip(with_ball=True, attacks="right", los=47.0)
    del track.meta["broadcast"]
    for f in track.frames:                       # shove the defense miles downfield
        for pid, p in f.players.items():
            if pid.startswith("d"):
                p.x += 25.0

    out, report = prepare.prepare(track)
    assert report.line_source is None
    assert any("not a snap" in b for b in report.blockers)


def test_the_report_survives_a_round_trip_through_meta():
    out, report = prepare.prepare(clip(with_ball=True))
    assert out.meta["prepare_report"]["reviewable"] is True
    assert out.meta["prepare_report"]["roster"]["after"] == 22


def test_a_caller_who_knows_the_offense_is_believed():
    out, report = prepare.prepare(clip(), offense="teamB")
    assert report.sides.offense == "teamB"
    assert report.sides.method == "caller"
    assert report.sides.confidence == 1.0


def test_a_ball_that_never_moves_is_not_a_ball():
    """The reference clip's actual failure, and the reason a detection rate is not
    a quality measure. Tiling took the rate from 5% to 42% by finding the
    down-and-distance graphic: it sits on the field near the line of scrimmage, it
    is roughly ball-shaped, and it never moves."""
    track = clip(with_ball=True, ball_measured=0.42)
    track.meta["ball"]["travelled_yards"] = 2.4

    out, report = prepare.prepare(track)

    assert report.ball_measured == 0.42      # a confident detection rate
    assert not report.reviewable             # and a worthless track
    assert any("does not move" in b for b in report.blockers)
    assert any("graphic" in b for b in report.blockers)


def test_a_ball_that_travels_passes():
    track = clip(with_ball=True, ball_measured=0.42)
    track.meta["ball"]["travelled_yards"] = 34.0
    out, report = prepare.prepare(track)
    assert report.reviewable, report.blockers
