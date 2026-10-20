"""Role inference, measured against 115 real plays rather than asserted.

The targeted receiver is the player the whole recommendation is a comparison
against, so getting him wrong does not break anything visibly — it just changes
which player the What-If panel argues about, silently. The play-by-play description
names the real target, so the inference can be scored, and that score is the only
claim worth making about this module.
"""

import pytest

from cf_support import HINDSIGHT_PLAYS
from gridiron_counterfactual import roles
from gridiron_tracking import hindsight

pytestmark = pytest.mark.skipif(
    not HINDSIGHT_PLAYS.exists(),
    reason="tests/fixtures/hindsight-plays is missing",
)


def real_plays():
    """The app's measured plays. Exported video clips are excluded by game id."""
    return [
        hindsight.read_play(p)
        for p in hindsight.plays_in(HINDSIGHT_PLAYS)
        if not p.name.startswith("999000001_")
    ]


def test_the_targeted_receiver_inference_is_measured_not_assumed():
    tracks = real_plays()
    assert len(tracks) >= 100, "the play set has shrunk; this measurement is thin"

    report = roles.check_against_descriptions(tracks)

    assert report["checked"] >= 50, f"only {report['checked']} plays were checkable"
    # 60/62 when this was written. The floor is set below that rather than at it:
    # a regression worth knowing about is a real drop, not a one-play wobble.
    assert report["hit_rate"] >= 0.90, (
        f"targeted-receiver inference fell to {report['hit_rate']}; "
        f"misses: {report['misses_sample']}"
    )


def test_a_target_is_found_on_every_pass_and_on_no_run():
    """The skips must be the plays where no pass arrived, not plays it fumbled.

    A target invented on a run play would hand the model a receiver who was never
    thrown to; a target missed on a pass would drop the play from the review
    entirely. Both are silent, so the partition is asserted.
    """
    for track in real_plays():
        result = track.meta.get("passResult") or ""
        assigned = roles.infer_roles(track)
        has_target = any(r == roles.TARGET for r in assigned.values())

        if result in ("C", "I", "IN"):
            assert has_target, f"play {track.meta.get('play_id')} ({result}) has no target"
        else:
            # '' run, 'S' sack, 'R' scramble — no pass arrived.
            assert not has_target, (
                f"play {track.meta.get('play_id')} ({result or 'run'}) was given a "
                f"target, but no pass arrived on it"
            )


def test_the_offense_is_established_on_every_play():
    """Swapped sides score the defense's separation and look reasonable doing it."""
    for track in real_plays():
        assigned = roles.infer_roles(track)
        coverage = sum(1 for r in assigned.values() if r == roles.COVERAGE)
        assert coverage == 11, f"play {track.meta.get('play_id')}: {coverage} defenders"
        assert len(track.roster) == 22
        assert set(assigned) <= set(track.roster)
        # Linemen are deliberately left unassigned, so the map is shorter than the
        # roster. It must not be shorter than the 11 defenders plus the receivers.
        assert len(assigned) >= 11


def test_a_passer_is_named_on_passes_and_on_nothing_else():
    """Only a throw identifies a thrower; a run play has no passer to name."""
    for track in real_plays():
        assigned = roles.infer_roles(track)
        passers = sum(1 for r in assigned.values() if r == roles.PASSER)
        assert sum(1 for r in assigned.values() if r == roles.TARGET) <= 1
        assert passers <= 1

        threw = any(e in {ev for f in track.frames for ev in f.events}
                    for e in roles.THROW_EVENTS)
        assert passers == int(threw), (
            f"play {track.meta.get('play_id')}: threw={threw} but {passers} passers"
        )


def test_the_passer_is_the_quarterback():
    """A free check: the release carries positions, and the inference ignores them.

    Picking the offensive player nearest the ball at the throw should land on the
    quarterback. Where it does not, the throw event itself is suspect.
    """
    agree = total = 0
    for track in real_plays():
        assigned = roles.infer_roles(track)
        passer = next((pid for pid, r in assigned.items() if r == roles.PASSER), None)
        if passer is None:
            continue  # no throw on this play
        position = track.roster[passer].role
        if position:
            total += 1
            agree += position == "QB"
    assert total > 50
    # 67/68 when written, after linemen stopped being candidates. Measured at the
    # snap instead of the throw it was 1/115, because the centre has his hands on
    # the ball; that is the regression this number guards against.
    assert agree / total >= 0.95, f"passer was the QB on only {agree}/{total} plays"


def test_apply_records_the_roles_and_says_they_were_inferred():
    track = next(t for t in real_plays() if (t.meta.get("passResult") or "") == "C")
    roles.apply(track)

    assert track.meta["roles"]
    assert "inferred" in track.meta["roles_source"]
    starred = [pid for pid, e in track.roster.items() if e.star]
    assert len(starred) == 1
    assert track.meta["roles"][starred[0]] == roles.TARGET


def test_a_track_with_no_ball_cannot_have_roles_invented():
    """The vision pipeline's output today. No ball, so no passer and no target."""
    track = next(iter(real_plays()))
    for f in track.frames:
        f.ball = None

    assigned = roles.infer_roles(track)
    assert not any(r in (roles.PASSER, roles.TARGET) for r in assigned.values())
    # Sides still partition, because that comes from the roster, not the ball.
    assert sum(1 for r in assigned.values() if r == roles.COVERAGE) == 11


def test_the_roles_feed_straight_into_the_feature_builder():
    """The point of the module: a play the model could not previously score."""
    from gridiron_counterfactual import features

    track = next(t for t in real_plays() if (t.meta.get("passResult") or "") == "C")
    roles.apply(track)

    arrival = next(t for name in roles.ARRIVAL_EVENTS
                   if (t := track.event_time(name)) is not None)
    ball = track.frame_at(arrival).ball
    features.with_play_state(track, ball_land_x=ball.x, ball_land_y=ball.y,
                             yardline=track.meta["lineOfScrimmage"], frames_output=12)

    rows = features.feature_rows(track)
    assert len(rows) == 22 * 12
    assert set(rows.role_code.unique()) <= set(roles_codes())


def roles_codes():
    from gridiron_counterfactual import features

    return set(features.ROLE_CODES.values()) | {features.MISSING_ROLE}


def test_linemen_are_not_offered_as_men_who_should_have_been_thrown_to():
    """Mutation-found gap: making every offensive player a route runner survived
    the whole suite. The hit-rate floor is 0.90 and including linemen scored 0.968,
    so nothing failed — but the inference did once offer Rob Havenstein, a right
    tackle, as the open alternative."""
    for track in real_plays():
        assigned = roles.infer_roles(track)
        for pid, role in assigned.items():
            position = track.roster[pid].role
            if position in roles.LINE and role != roles.COVERAGE:
                pytest.fail(
                    f"play {track.meta.get('play_id')}: {position} {pid} was given "
                    f"the role {role!r}; linemen must be left unassigned so the "
                    f"model's fillna(-1) keeps them out of the alternatives"
                )
        # And the men who ARE candidates are all eligible positions.
        for pid, role in assigned.items():
            if role in (roles.ROUTE, roles.TARGET):
                assert track.roster[pid].role in roles.ELIGIBLE, (
                    f"{track.roster[pid].role} was offered as a receiver"
                )


def test_a_zero_confidence_side_call_is_refused_not_used():
    """Mutation-found gap: deleting the confidence check survived.

    infer_sides returns a team name even when it had nothing to go on — most often
    because the roster carries three kit labels. Taking that name makes one stray
    track the whole offense and everybody else a defender: 44 'defenders' and 2
    route runners on the reference clip, and every receiver then reads as covered.
    """
    from gridiron_tracking.contract import (
        Ball, PlayTrack, PlayerFrame, RosterEntry, TrackingFrame,
    )

    # Three kit labels is what makes infer_sides give up.
    roster = {}
    for i in range(8):
        roster[f"a{i}"] = RosterEntry(id=f"a{i}", team="teamA")
        roster[f"b{i}"] = RosterEntry(id=f"b{i}", team="teamB")
    roster["x"] = RosterEntry(id="x", team="kitA")

    frames = [TrackingFrame(
        t=0.0,
        players={pid: PlayerFrame(x=50.0 + i, y=26.0, o_source="assumed")
                 for i, pid in enumerate(roster)},
        ball=None, events=["ball_snap"])]
    track = PlayTrack(meta={"source": "vision", "rate": 10}, roster=roster, frames=frames)

    from gridiron_tracking import sides as sidesmod
    assert sidesmod.infer_sides(track).confidence == 0.0, "fixture no longer ambiguous"

    with pytest.raises(ValueError, match="cannot tell which side"):
        roles.infer_roles(track)
