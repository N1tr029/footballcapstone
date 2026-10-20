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
        # A ball that is held, then thrown across the field, then caught — so the
        # pass-window derivation has something to find. A stationary ball is a run,
        # and prepare is right to say so.
        ball = None
        if with_ball:
            if i <= 5:
                bx, by = los - sign * 5.0, 26.0
            elif i < 9:
                f = (i - 5) / 4.0
                bx = los - sign * 5.0 + sign * f * 22.0
                by = 26.0 + f * 12.0
            else:
                bx, by = los + sign * 17.0, 38.0
            ball = Ball(bx, by)
        frames.append(TrackingFrame(
            t=round((i - 3) / 10, 3), players=players,
            ball=ball,
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
    # play_direction records the SOURCE's direction and mirrored says what was done
    # about it; the two together are the convention features and forecast read.
    assert out.meta["play_direction"] == "left"
    assert out.meta["mirrored"] is True
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
    every derivation downstream runs, returns a number, and is wrong.

    The assertion names prepare's own blocker rather than just "not reviewable".
    Mutation-found: deleting prepare's threshold entirely left this test green,
    because events.derive refuses the same track for its own reasons and the play
    was unreviewable either way. Two guards, two tests.
    """
    out, report = prepare.prepare(clip(with_ball=True, ball_measured=0.05))

    assert report.has_ball is True
    assert not report.reviewable
    mine = [b for b in report.blockers if "Below 25%" in b]
    assert mine, (
        f"prepare did not raise its own sparse-ball blocker; got {report.blockers}"
    )
    assert "actually seen on 5%" in mine[0]


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


def test_a_folded_registration_stops_everything_downstream():
    """devin_td's actual failure. It registered, tracked 41 players and wrote a
    well-formed track.json in which no number was a coordinate — the checks had
    called the fit folded and collinear and said so only on the console."""
    track = clip(with_ball=True)
    track.meta["registration"] = {"stage": "computed", "trustworthy": False}
    track.meta["render_problems"] = [
        "no usable perspective — the points are probably collinear",
        "the field projects to a self-crossing shape — the fit is folded",
    ]

    out, report = prepare.prepare(track)

    assert not report.reviewable
    assert any("nothing here is in yards" in b for b in report.blockers)
    assert any("collinear" in b for b in report.blockers)


def test_a_trustworthy_registration_passes_through():
    track = clip(with_ball=True)
    track.meta["registration"] = {"stage": "computed", "trustworthy": True}
    track.meta["render_problems"] = []
    out, report = prepare.prepare(track)
    assert report.reviewable, report.blockers


def test_the_fatal_fit_patterns_match_what_the_checks_actually_say():
    """render refuses to call a fit trustworthy when these appear in its problems.

    Matching on the checks' own wording rather than a structured flag, so the
    wording and the match have to be kept together — this test is that pairing.
    """
    from gridiron_tracking.render import _FATAL_FIT

    real_complaints = [
        "no usable perspective — the points are probably collinear",
        "the field projects to a self-crossing shape — the fit is folded",
        "points lie on only 2 yard line(s) — not enough to fix scale down the field",
        "no registration — nothing can be placed in yards",
    ]
    for complaint in real_complaints:
        assert _FATAL_FIT.search(complaint), f"not matched: {complaint}"

    survivable = [
        "best attempt (frame 165) aligned to the paint at only 1.37x; 1.60 is the bar",
        "implies a camera 80 yards up, which is not where a camera goes",
        "ball tracking failed: ImportError: no ultralytics",
    ]
    for complaint in survivable:
        assert not _FATAL_FIT.search(complaint), f"wrongly fatal: {complaint}"


def test_the_registration_verdict_refuses_a_degenerate_fit():
    """The one line deciding whether a play's numbers mean anything.

    It lives in render(), which needs a video, a detector and a vision model, so
    it had no test — and a mutation making every fit trustworthy survived the
    whole suite.
    """
    from gridiron_tracking.render import registration_verdict

    folded = registration_verdict(
        ["no usable perspective — the points are probably collinear",
         "the field projects to a self-crossing shape — the fit is folded"],
        "computed")
    assert folded["trustworthy"] is False
    assert folded["stage"] == "computed"

    fine = registration_verdict(
        ["best attempt (frame 165) aligned to the paint at only 1.37x"], "computed")
    assert fine["trustworthy"] is True

    assert registration_verdict([], "cached+propagated")["trustworthy"] is True


# ------------------------------------------------------- the report a person reads


def test_the_table_renders_for_a_clip_that_worked():
    out, report = prepare.prepare(clip(with_ball=True))
    text = report.table()

    assert "roster consolidation" in text
    assert "possession" in text and "teamA has the ball" in text
    assert "line of scrimmage" in text and "from the broadcast" in text
    assert "mirrored so the offense attacks +x" in text
    assert "reviewable        yes" in text
    assert "pass window" in text and "in the air" in text


def test_the_table_renders_for_every_way_a_clip_can_fail():
    """It is read exactly when something is wrong, so it must not be the thing
    that breaks. A None in an f-string here costs the diagnosis."""
    cases = {}

    no_ball = clip(with_ball=False)
    cases["no ball"] = prepare.prepare(no_ball)[1]

    sparse = clip(with_ball=True, ball_measured=0.05)
    cases["sparse ball"] = prepare.prepare(sparse)[1]

    static = clip(with_ball=True)
    static.meta["ball"]["travelled_yards"] = 1.0
    cases["static ball"] = prepare.prepare(static)[1]

    folded = clip(with_ball=True)
    folded.meta["registration"] = {"stage": "computed", "trustworthy": False}
    folded.meta["render_problems"] = ["the fit is folded"]
    cases["folded"] = prepare.prepare(folded)[1]

    nolines = clip(with_ball=True)
    del nolines.meta["broadcast"]
    cases["no broadcast"] = prepare.prepare(nolines)[1]

    unknown = clip(with_ball=True)
    del unknown.meta["broadcast"]
    for f in unknown.frames:                       # no snap => no formation either
        f.events = []
    cases["no snap"] = prepare.prepare(unknown)[1]

    for name, report in cases.items():
        text = report.table()
        assert isinstance(text, str) and text, name
        assert "reviewable" in text, name
        if not report.reviewable:
            assert "NO" in text, name
            assert any(b[:20] in text for b in report.blockers), name


def test_the_table_says_when_a_side_is_short():
    track = clip(with_ball=True)
    for pid in [p for p in list(track.roster) if p.startswith("o")][:4]:
        track.roster.pop(pid)
        for f in track.frames:
            f.players.pop(pid, None)

    out, report = prepare.prepare(track)
    text = report.table()
    assert "SHORT OF A FULL SIDE" in text
    assert "more open than he was" in text


def test_the_report_json_matches_the_table(caplog):
    out, report = prepare.prepare(clip(with_ball=True))
    d = report.to_json()
    assert d["reviewable"] is report.reviewable
    assert d["line_source"] == report.line_source
    assert d["roster"]["after"] == report.roster.after
    assert d["events"]["thrown_at"] == report.events.thrown_at
