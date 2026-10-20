"""The whole chain on one play, and a sanity check on what it says about a game.

The test that matters here is the last one. Every individual piece can be correct
while the system as a whole says something absurd, and the only way to notice is to
run it over a real game and look at the distribution of its answers. It caught a
genuine bug: forecasting from the end of the play rather than from the throw made
the model claim 61 of 67 throws were wrong by a median of 8.65 yards.
"""

import pytest

from cf_support import HINDSIGHT_PLAYS, MODEL
from gridiron_counterfactual import review, roles
from gridiron_tracking import hindsight

pytestmark = pytest.mark.skipif(
    not HINDSIGHT_PLAYS.exists() or not MODEL.exists(),
    reason="needs the hindsight subtree and the fitted model",
)


def plays():
    return [p for p in hindsight.plays_in(HINDSIGHT_PLAYS)
            if not p.name.startswith("999000001_")]


def a_completed_pass():
    for path in plays():
        track = hindsight.read_play(path)
        if (track.meta.get("passResult") or "") == "C":
            return track
    pytest.skip("no completed pass in the play set")


def test_one_play_goes_all_the_way_through():
    pytest.importorskip("sklearn")
    result = review.review_play(a_completed_pass())

    assert result.recommendation is not None
    assert result.recommendation.call.type in ("pass", "run", "punt", "unchanged")
    assert result.forecast.meta["source"] == "simulated"
    assert result.forecast.frames
    assert result.track.meta["roles"]
    assert result.notes


def test_the_forecast_starts_at_the_throw_and_not_at_the_end_of_the_play():
    """The bug this module's hardest-won line of code exists to prevent.

    The model was fitted on input that stops at the throw. A whole play runs on to
    the tackle, so forecasting from its last frame asks the model to continue a
    play that is already over, from a state it never saw in training.
    """
    pytest.importorskip("sklearn")
    track = a_completed_pass()
    throw = next(t for name in roles.THROW_EVENTS
                 if (t := track.event_time(name)) is not None)
    end = track.times[-1]
    assert throw < end - 0.5, "this play ends at the throw; pick another"

    result = review.review_play(track)

    first = result.forecast.times[0]
    assert first == pytest.approx(throw + 0.1, abs=1e-6), (
        f"forecast starts at {first}, not just after the throw at {throw}"
    )
    assert result.forecast.times[-1] < end + 0.5
    assert any("from the throw" in n for n in result.notes)


def test_the_forecast_runs_to_where_the_ball_arrived():
    """Separation is judged at the last forecast frame, so that frame has to be
    the catch point rather than an arbitrary horizon."""
    pytest.importorskip("sklearn")
    track = a_completed_pass()
    throw = next(t for name in roles.THROW_EVENTS
                 if (t := track.event_time(name)) is not None)
    arrival = next(t for name in roles.ARRIVAL_EVENTS
                   if (t := track.event_time(name)) is not None)

    result = review.review_play(track)
    assert result.forecast.times[-1] == pytest.approx(arrival, abs=0.15)


def test_a_run_play_gets_a_forecast_but_no_judgement():
    """There is no throw to review, and saying so beats inventing a receiver."""
    pytest.importorskip("sklearn")
    for path in plays():
        track = hindsight.read_play(path)
        if (track.meta.get("passResult") or "") == "":
            break
    else:
        pytest.skip("no run play in the set")

    with pytest.raises(ValueError, match="no pass arrival"):
        review.review_play(track)


def test_the_decision_comes_from_decision_engine_itself():
    """Not reimplemented here: two definitions of "open" would drift apart."""
    engine = review.decision_engine()
    assert hasattr(engine, "decisions")
    assert engine.OPEN_SEPARATION_YARDS == 3.0

    pytest.importorskip("sklearn")
    result = review.review_play(a_completed_pass())
    assert result.report is not None
    assert result.recommendation.evidence["open_threshold_yards"] == engine.OPEN_SEPARATION_YARDS
    assert any("decision_engine's own" in n for n in result.notes)


def test_what_it_says_about_a_whole_game_is_not_absurd():
    """The system-level check. Each piece can be right while the whole is nonsense.

    Over one real game, a review that flags nearly every throw as a mistake by a
    wide margin is wrong about football regardless of what the unit tests say. The
    thresholds here are deliberately loose — this is a smoke test for an
    implausible system, not a claim that the model is good.
    """
    pytest.importorskip("sklearn")
    import statistics

    calls, advantages = [], []
    for path in plays():
        track = hindsight.read_play(path)
        if (track.meta.get("passResult") or "") not in ("C", "I", "IN"):
            continue
        result = review.review_play(track)
        if result.recommendation is None:
            continue
        calls.append(result.recommendation.call.type)
        advantage = result.recommendation.evidence.get(
            "alternative_separation_advantage_yards")
        if advantage is not None:
            advantages.append(advantage)

    assert len(calls) >= 50, f"only {len(calls)} passes reviewed"

    # 35 of 67 when written. A model telling you 90% of NFL throws were wrong is
    # not reviewing the game, it is misreading its own inputs.
    flagged = sum(1 for c in calls if c != "unchanged") / len(calls)
    assert flagged <= 0.80, (
        f"{flagged:.0%} of throws flagged as wrong; forecasting from the wrong "
        f"instant produced 94% and a median advantage of 8.65 yd"
    )

    # 1.52 yd when written, against 8.65 with the forecast instant wrong.
    median = statistics.median(advantages)
    assert median < 5.0, (
        f"median claimed advantage over the real target is {median:.2f} yd, which "
        f"is implausibly large for professional quarterbacks"
    )


def test_the_plays_it_disputes_are_the_plays_that_failed():
    """External validation, and the only kind available: the model never sees outcomes.

    Nothing in the chain is given ``passResult``. The forecast is positions, the
    judgement is separation geometry. So if the plays it disputes most are also the
    plays that actually failed, that agreement came from somewhere real.

    It is correlational and it is one game. "The receiver was covered" predicts both
    "the model objects" and "the throw failed", so this is **not** evidence that the
    alternative would have worked — only that the model is reading the coverage
    rather than producing noise. That distinction belongs in the write-up.
    """
    pytest.importorskip("sklearn")

    rows = []
    for path in plays():
        track = hindsight.read_play(path)
        result_code = track.meta.get("passResult") or ""
        if result_code not in ("C", "I", "IN"):
            continue
        reviewed = review.review_play(track)
        if reviewed.recommendation is None:
            continue
        advantage = reviewed.recommendation.evidence.get(
            "alternative_separation_advantage_yards")
        if advantage is not None:
            rows.append((float(advantage), result_code))

    assert len(rows) >= 50
    rows.sort(reverse=True)
    quartile = len(rows) // 4

    def incompletion_rate(subset):
        return sum(1 for _, r in subset if r != "C") / len(subset)

    disputed = incompletion_rate(rows[:quartile])
    agreed = incompletion_rate(rows[-quartile:])

    # 50% against 6% when written. The gap is the signal; the floor is set low
    # because one game is a small sample and the point is a direction, not a value.
    assert disputed > agreed, (
        f"plays the model disputes failed {disputed:.0%} of the time, plays it "
        f"agrees with {agreed:.0%} — the ranking carries no outcome signal at all"
    )
    assert disputed - agreed > 0.15, (
        f"separation only {disputed:.0%} vs {agreed:.0%}; this was 50% vs 6%"
    )


def test_a_play_with_a_forecast_but_no_target_says_so_and_keeps_the_forecast():
    """The model reviews a throw. A play with the state supplied but nobody
    targeted still gets its forecast — it just gets no judgement, and the note
    names the reason rather than leaving a silent None."""
    pytest.importorskip("sklearn")
    from gridiron_counterfactual import features

    track = a_completed_pass()
    # Supply the play state so review does not refuse for lack of it, then strip
    # the arrival so no receiver can be identified.
    arrival = next(t for name in roles.ARRIVAL_EVENTS
                   if (t := track.event_time(name)) is not None)
    ball = track.frame_at(arrival).ball
    features.with_play_state(track, ball_land_x=ball.x, ball_land_y=ball.y,
                             yardline=track.meta["lineOfScrimmage"], frames_output=8)
    for f in track.frames:
        f.events = [e for e in f.events if e not in roles.ARRIVAL_EVENTS]

    result = review.review_play(track)

    assert result.recommendation is None
    assert result.report is None
    assert result.forecast.frames, "the forecast should still stand"
    assert result.forecast.meta["source"] == "simulated"
    assert any("no targeted receiver" in n for n in result.notes)


def test_a_play_with_no_defenders_is_refused_rather_than_softly_reviewed():
    """A roster that does not partition into two sides is broken input, not a
    degraded case: separation to the nearest defender has no meaning without any,
    and a soft note here would let the number through."""
    pytest.importorskip("sklearn")
    track = a_completed_pass()
    defenders = [pid for pid, e in track.roster.items()
                 if str(e.team).startswith("def")]
    assert defenders
    for pid in defenders:
        track.roster.pop(pid)
        for f in track.frames:
            f.players.pop(pid, None)

    with pytest.raises(ValueError, match="does not partition this roster"):
        review.review_play(track)
