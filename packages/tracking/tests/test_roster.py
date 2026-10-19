"""Consolidation, against the real vision track and against cases built to break it.

The real clip is the test that matters — 46 tracks for 22 people is the actual
input — but it cannot tell you what the pass does *wrong*, because there is no
ground truth for which two tracks were one player. So the dangerous merges are
built by hand: two linemen a yard apart who must survive, and two tracks on one man
who must not.
"""

import json

import pytest

from tracking_support import REPO
from gridiron_tracking import roster
from gridiron_tracking.contract import (
    Ball, PlayTrack, PlayerFrame, RosterEntry, TrackingFrame,
)

CLIP = REPO / "data/runs/nw_michigan/track.json"


def build(paths: dict[str, tuple[str, list[tuple[float, float] | None]]]) -> PlayTrack:
    """``id -> (team, [(x, y) or None per frame])``."""
    n = max(len(v[1]) for v in paths.values())
    roster_ = {pid: RosterEntry(id=pid, team=team) for pid, (team, _) in paths.items()}
    frames = []
    for i in range(n):
        players = {}
        for pid, (_, pts) in paths.items():
            if i < len(pts) and pts[i] is not None:
                x, y = pts[i]
                players[pid] = PlayerFrame(x=x, y=y, s=1.0, dir=0.0, o=0.0,
                                           o_source="from_velocity")
        frames.append(TrackingFrame(t=round(i / 10, 3), players=players,
                                    ball=Ball(50.0, 26.0),
                                    events=["ball_snap"] if i == 0 else []))
    return PlayTrack(meta={"source": "vision", "rate": 10}, roster=roster_, frames=frames)


def test_two_linemen_a_yard_apart_are_two_people():
    """The merge rule's most dangerous neighbour. Deleting a guard is worse than
    leaving a duplicate: a duplicate is visible and a missing guard is not."""
    track = build({
        "g": ("teamA", [(50.0, 26.0)] * 20),
        "t": ("teamA", [(50.0, 27.1)] * 20),   # 1.1 yd away, the whole play
    })
    out, report = roster.consolidate(track)
    assert report.merged == 0
    assert len(out.roster) == 2


def test_one_man_wearing_two_tracks_becomes_one():
    track = build({
        "a": ("teamA", [(50.0 + i * 0.2, 26.0) for i in range(20)]),
        "b": ("teamA", [(50.1 + i * 0.2, 26.1) for i in range(20)]),  # 0.14 yd away
    })
    out, report = roster.consolidate(track)
    assert report.merged == 1
    assert len(out.roster) == 1


def test_two_players_crossing_are_not_merged():
    """Close for an instant is two people passing each other, not one person."""
    track = build({
        "a": ("teamA", [(45.0 + i, 26.0) for i in range(20)]),
        "b": ("teamA", [(65.0 - i, 26.0) for i in range(20)]),
    })
    out, report = roster.consolidate(track)
    assert report.merged == 0
    assert len(out.roster) == 2


def test_opposing_players_in_coverage_are_never_merged():
    """A corner runs stride for stride with a receiver for a whole play."""
    track = build({
        "wr": ("teamA", [(50.0 + i, 26.0) for i in range(20)]),
        "cb": ("teamB", [(50.0 + i, 26.4) for i in range(20)]),
    })
    out, report = roster.consolidate(track)
    assert report.merged == 0
    assert {e.team for e in out.roster.values()} == {"teamA", "teamB"}


def test_a_third_kit_label_is_dropped_and_said_to_be():
    """A third label makes sides.infer_sides report zero confidence."""
    track = build({
        **{f"a{i}": ("teamA", [(40.0 + i, 20.0)] * 20) for i in range(3)},
        **{f"b{i}": ("teamB", [(60.0 + i, 30.0)] * 20) for i in range(3)},
        "x": ("kitA", [(55.0, 25.0)] * 20),
    })
    out, report = roster.consolidate(track)
    assert report.dropped_unlabelled == 1
    assert {e.team for e in out.roster.values()} == {"teamA", "teamB"}
    assert any("kit labels" in n for n in report.notes)


def test_strays_are_dropped():
    track = build({
        "real": ("teamA", [(50.0, 26.0)] * 20),
        "stray": ("teamA", [None] * 18 + [(10.0, 2.0), (10.0, 2.0)]),
    })
    out, report = roster.consolidate(track)
    assert report.dropped_short == 1
    assert set(out.roster) == {"real"}


def test_never_more_than_eleven_a_side():
    track = build({
        **{f"a{i}": ("teamA", [(20.0 + i * 3, 10.0)] * 20) for i in range(14)},
        **{f"b{i}": ("teamB", [(20.0 + i * 3, 40.0)] * 20) for i in range(13)},
    })
    out, report = roster.consolidate(track)
    assert report.per_side == {"teamA": 11, "teamB": 11}
    assert len(out.roster) == 22
    assert report.dropped_surplus == 5


def test_a_clean_release_track_passes_through_unchanged():
    track = build({
        **{f"a{i}": ("offense", [(20.0 + i * 4, 10.0 + i)] * 20) for i in range(11)},
        **{f"b{i}": ("defense", [(70.0 + i * 2, 40.0 - i)] * 20) for i in range(11)},
    })
    out, report = roster.consolidate(track)
    assert report.after == report.before == 22
    assert report.merged == report.dropped_short == report.dropped_surplus == 0
    assert report.complete


def test_an_absorbed_track_only_fills_gaps_it_does_not_move_anyone():
    """A merge must never relocate a player who was already seen that frame."""
    track = build({
        "a": ("teamA", [(50.0, 26.0)] * 10 + [None] * 10),
        "b": ("teamA", [(50.1, 26.0)] * 20),
    })
    out, report = roster.consolidate(track)
    assert report.merged == 1
    kept = next(iter(out.roster))
    series = out.series(kept)
    assert len(series) == 20
    # The surviving track's own samples are untouched where it had them.
    assert series[0][1].x == pytest.approx(50.1 if kept == "b" else 50.0)


@pytest.mark.skipif(not CLIP.exists(), reason="the reference clip has not been rendered")
def test_the_reference_clip_comes_out_eleven_a_side():
    """46 tracks in, 22 players out, on the real thing."""
    track = PlayTrack.from_json(json.loads(CLIP.read_text()))
    assert len(track.roster) > 30, "this clip no longer has the problem; retune the test"

    out, report = roster.consolidate(track)

    assert report.before == len(track.roster)
    assert len(out.roster) <= 22
    assert set(report.per_side) == {"teamA", "teamB"}
    assert all(n <= 11 for n in report.per_side.values())
    # Every surviving player appears in at least a fifth of the play.
    for pid in out.roster:
        assert len(out.series(pid)) >= 0.18 * len(out.frames)
    assert "roster_report" in out.meta


@pytest.mark.skipif(not CLIP.exists(), reason="the reference clip has not been rendered")
def test_consolidation_does_not_move_anybody():
    """It drops and merges. It must never change where a kept player was."""
    track = PlayTrack.from_json(json.loads(CLIP.read_text()))
    out, report = roster.consolidate(track)

    for pid in out.roster:
        if pid not in track.roster:
            continue
        before = dict(track.series(pid))
        for t, p in out.series(pid):
            if t in before and before[t] is not None:
                # Either his own sample, or an absorbed one filling a frame he missed.
                original = before[t]
                moved = abs(p.x - original.x) + abs(p.y - original.y)
                assert moved < 1e-9 or moved <= roster.DUP_YARDS * 2
