"""Run the real model on the same video JSON consumed by the frontend."""
import json
from pathlib import Path

import pytest

from cf_support import MODEL
from gridiron_tracking import hindsight
from gridiron_tracking.contract import PlayTrack
from gridiron_tracking.video_play import prepared, play_json
from gridiron_counterfactual import forecast, review, features

EXAMPLE = Path(__file__).resolve().parents[3]/"docs/examples/nfl_sf_sea.track.json"


@pytest.mark.skipif(not MODEL.exists(),reason="fitted coordinate model missing")
def test_checked_video_export_runs_model_and_forecast_preserves_identity_and_clock():
    pytest.importorskip("sklearn")
    raw = PlayTrack.from_json(json.loads(EXAMPLE.read_text()))
    doc = play_json(raw)
    track = hindsight.read_play(doc)
    result = review.review_play(track)
    assert result.recommendation is not None
    assert result.recommendation.confidence == "low"
    assert result.recommendation.evidence["tracking_source"] == "estimated_from_video"
    ids = {p["nflId"] for p in doc["players"]}
    assert result.recommendation.evidence["targeted_receiver_id"] in ids
    assert result.recommendation.evidence["targeted_receiver_id"] != 0
    assert 1 < result.recommendation.evidence["tracked_player_count"] < 22
    assert result.forecast.times[0] == pytest.approx(3.3)
    assert result.forecast.times[-1] == pytest.approx(3.6)
    continuation = hindsight.to_play_json(forecast.continue_play(track,result.forecast))
    assert continuation["snapFrameId"] == doc["snapFrameId"]
    assert {p["nflId"] for p in continuation["players"]} == ids


def test_anonymous_video_forecast_subset_keeps_original_export_ids():
    raw = PlayTrack.from_json(json.loads(EXAMPLE.read_text()))
    track,_ = prepared(raw)
    # Simulate a strict subset of the roster, without loading model weights.
    import pandas as pd
    predictions = pd.DataFrame([{"player_id":"p07","horizon":1,"pred_x":60.,"pred_y":25.}])
    future = forecast.to_play_track(predictions,track,t0=3.2)
    original = hindsight.nfl_ids(track)
    assert hindsight.nfl_ids(future) == {"p07":original["p07"]}
    document = hindsight.to_play_json(future)
    restored = hindsight.read_play(document)
    assert hindsight.nfl_ids(restored) == {str(original["p07"]):original["p07"]}


def test_off_camera_last_sighting_is_not_a_current_model_state():
    raw = PlayTrack.from_json(json.loads(EXAMPLE.read_text()))
    track,_ = prepared(raw)
    at = track.event_time("pass_forward")
    state = features.last_state(track,at=at)
    assert all(at-t <= .25 for t in state["t"])
    assert len(state) < 22


def test_sparse_unprepared_video_cannot_fall_back_to_last_frame_forecast():
    raw = PlayTrack.from_json(json.loads(EXAMPLE.read_text()))
    raw.meta.pop("reviewed_pass",None)
    with pytest.raises(ValueError,match="not ready for analysis"):
        review.review_play(raw)
