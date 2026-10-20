"""A forecast must animate, and must never claim to be an observation."""

import numpy as np
import pytest

from cf_support import MODEL, bdb_play, predictions_from
from gridiron_counterfactual import bdb, features, forecast
from gridiron_tracking import export
from gridiron_tracking.contract import RATE, validate


def a_track(direction="right", frames_output=15):
    return bdb.to_play_track(
        bdb.load(bdb_play(play_direction=direction, frames_output=frames_output)),
        game_id=2023091002, play_id=110,
    )


def test_a_forecast_becomes_a_drawable_play():
    src = a_track()
    sim = forecast.to_play_track(predictions_from(src), src)

    assert len(sim.frames) == 15
    assert set(sim.roster) == set(src.roster)
    # The clock continues the real play rather than restarting, which is what lets
    # the two be drawn side by side.
    assert sim.frames[0].t == pytest.approx(src.times[-1] + 1.0 / RATE)
    assert sim.times == sorted(sim.times)


def test_provenance_leads_with_the_fact_that_it_did_not_happen():
    src = a_track()
    sim = forecast.to_play_track(predictions_from(src), src)
    prov = export.provenance(sim)

    assert prov["coordinates"]["simulated"] is True
    assert prov["coordinates"]["measured"] is False
    assert prov["coordinates"]["note"].startswith("SIMULATED")
    # The wrong answer here is the video one: a forecast is not an estimate from
    # footage, and telling a viewer to go read the registration block for a play
    # that was never filmed is worse than saying nothing.
    assert "ESTIMATED from video" not in prov["coordinates"]["note"]


def test_orientation_is_never_claimed_as_measured():
    """Nothing in the model predicts which way a man will be facing."""
    src = a_track()
    sim = forecast.to_play_track(predictions_from(src), src)

    sources = {p.o_source for f in sim.frames for p in f.players.values()}
    assert sources <= {"from_velocity", "assumed"}
    assert "measured" not in sources
    assert sim.meta["orientation_measured"] is False
    assert export.provenance(sim)["orientation"]["measured_fraction"] == 0.0


def test_derived_kinematics_are_consistent_with_the_predicted_path():
    """Speed is differenced from the positions, so it has to agree with them."""
    src = a_track()
    sim = forecast.to_play_track(predictions_from(src), src)

    pid = sorted(sim.roster)[0]
    series = sim.series(pid)
    for (t0, p0), (t1, p1) in zip(series, series[1:]):
        step = np.hypot(p1.x - p0.x, p1.y - p0.y)
        assert step / (t1 - t0) == pytest.approx(p1.s, rel=0.25, abs=0.5)


@pytest.mark.parametrize("direction", ["right", "left"])
def test_the_forecast_lands_in_contract_coordinates(direction):
    """Predictions arrive in the release's frame and must be mirrored back.

    Unmirrored, a left-direction forecast runs the offense the wrong way down the
    field — and looks entirely normal doing it.
    """
    src = a_track(direction)
    sim = forecast.to_play_track(predictions_from(src), src)

    # The offense attacks +x in contract coordinates, whichever way the play ran.
    offense = [pid for pid, e in sim.roster.items() if e.team.lower().startswith("off")]
    for pid in offense:
        series = sim.series(pid)
        assert series[-1][1].x > series[0][1].x, f"{pid} is going backwards"

    for f in sim.frames:
        for p in f.players.values():
            assert -5.0 <= p.x <= 125.0
            assert -5.0 <= p.y <= 58.0


def test_the_joined_play_records_where_invention_starts():
    src = a_track()
    sim = forecast.to_play_track(predictions_from(src), src)
    joined = forecast.continue_play(src, sim)

    cut = joined.meta["simulated_from_t"]
    assert cut == sim.frames[0].t
    assert joined.meta["source"] == "simulated"
    assert joined.meta["simulated_from"] == "bdb2026"
    assert len(joined.frames) == len([f for f in src.frames if f.t < cut]) + len(sim.frames)
    assert joined.times == sorted(joined.times)
    # Real up to the cut, invented after it — and a viewer scrubbing across the
    # join can only know that because the number is in the file.
    assert export.provenance(joined)["coordinates"]["simulated"] is True


def test_a_gap_in_the_forecast_steps_is_refused():
    """A missing step is a player teleporting, not a player absent."""
    src = a_track()
    preds = predictions_from(src)
    with pytest.raises(ValueError, match="1..N with no gaps"):
        forecast.to_play_track(preds[preds.horizon != 4], src)


def test_the_contract_validator_s_remaining_complaints_are_the_honest_ones():
    """A forecast has no ball and no snap, and says so rather than inventing them."""
    src = a_track()
    sim = forecast.to_play_track(predictions_from(src), src)
    v = validate(sim)

    assert not v.ok
    assert set(v.problems) == {"no ball track", "no ball_snap event"}


@pytest.mark.skipif(not MODEL.exists(), reason="the fitted model is not in the tree")
def test_the_fitted_model_agrees_with_the_feature_builder():
    """The model file is the authority on its own feature list."""
    pytest.importorskip("joblib")
    verdict = features.check_against_model(str(MODEL))
    assert verdict["ok"], verdict["note"]


@pytest.mark.skipif(not MODEL.exists(), reason="the fitted model is not in the tree")
def test_end_to_end_through_the_real_model():
    """Release play -> features -> the committed regressors -> a drawable play."""
    pytest.importorskip("joblib")
    pytest.importorskip("sklearn")
    src = a_track()
    preds = forecast.predict(src, str(MODEL))
    sim = forecast.to_play_track(preds, src)

    assert len(sim.frames) == 15
    assert export.provenance(sim)["coordinates"]["simulated"] is True
    for f in sim.frames:
        for p in f.players.values():
            assert -5.0 <= p.x <= 125.0
            assert -5.0 <= p.y <= 58.0


def test_a_predicted_player_the_source_never_had_is_kept_and_flagged():
    """The model is given the roster, so this should not happen — and if it does,
    dropping the player silently would be worse than drawing an unknown one."""
    src = a_track()
    preds = predictions_from(src)
    extra = preds[preds.player_id == preds.player_id.iloc[0]].copy()
    extra["player_id"] = "ghost"
    preds = __import__("pandas").concat([preds, extra], ignore_index=True)

    sim = forecast.to_play_track(preds, src)

    assert "ghost" in sim.roster
    assert sim.roster["ghost"].team == "?"
    assert sim.meta["players_not_in_source_roster"] == ["ghost"]
