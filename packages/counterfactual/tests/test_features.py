"""The round trip, which is the only test in this suite that can fail silently elsewhere.

A mirrored play fed to the model as though it were not mirrored produces plausible
numbers and a reflected forecast. Nothing raises, nothing looks wrong, and the
scatter of dots is the right shape. So the conversions are asserted to be exact
inverses, on a left-direction play as well as a right one, because the left case is
the one that carries the mirroring.
"""

import numpy as np
import pandas as pd
import pytest

from cf_support import bdb_play
from gridiron_counterfactual import bdb, features


@pytest.mark.parametrize("direction", ["right", "left"])
def test_release_rows_survive_the_round_trip(direction):
    play = bdb_play(play_direction=direction)
    err = features.round_trip_error(bdb.load(play), game_id=2023091002, play_id=110)

    assert err["players"] == 6
    # Positions to a thousandth of a yard, angles to a thousandth of a degree:
    # the transforms are exact reflections, so the only error is float rounding.
    assert err["max_x"] < 1e-3
    assert err["max_y"] < 1e-3
    assert err["max_dir"] < 1e-3
    assert err["max_o"] < 1e-3
    assert err["max_s"] < 1e-6


@pytest.mark.parametrize("direction", ["right", "left"])
def test_feature_rows_are_in_the_model_s_frame_not_the_contract_s(direction):
    """The features must match the file, not the track built from it."""
    play = bdb_play(play_direction=direction)
    track = bdb.to_play_track(bdb.load(play), game_id=2023091002, play_id=110)
    rows = features.feature_rows(track)

    want = play.sort_values("frame_id").groupby("nfl_id").tail(1).set_index("nfl_id")
    got = rows[rows.horizon == 1].set_index(rows[rows.horizon == 1].player_id.astype(int))

    for pid in want.index:
        assert got.loc[pid, "x"] == pytest.approx(want.loc[pid, "x"], abs=1e-6)
        assert got.loc[pid, "y"] == pytest.approx(want.loc[pid, "y"], abs=1e-6)
        assert got.loc[pid, "dir"] == pytest.approx(want.loc[pid, "dir"], abs=1e-6)

    if direction == "left":
        # And the track itself really was mirrored — otherwise the assertions above
        # would pass for the trivial reason that nothing happened in either direction.
        contract_x = [p.x for f in track.frames for p in f.players.values()]
        assert max(contract_x) > 60.0


def test_play_direction_code_is_signed_the_way_the_model_was_fitted():
    for direction, code in (("right", 1.0), ("left", -1.0)):
        track = bdb.to_play_track(bdb.load(bdb_play(play_direction=direction)))
        rows = features.feature_rows(track)
        assert set(rows.play_direction_code.unique()) == {code}


def test_role_codes_match_the_decision_engine_s_mapping():
    """``ROLE_CODES`` here and ``ROLES`` in decision_engine.py must not drift."""
    import re

    src = (__import__("pathlib").Path(__file__).resolve().parents[1]
           / "decision_engine.py").read_text()
    literal = re.search(r"^ROLES = (\{[^}]*\})", src, re.M)
    assert literal, "decision_engine.py no longer defines ROLES as a literal"
    assert eval(literal.group(1)) == features.ROLE_CODES


def test_velocity_components_use_the_release_s_bearing_convention():
    """``vx`` from sin and ``vy`` from cos, because ``dir`` is a bearing.

    Swapping them is a reflection about the 45 degree line and is invisible in
    aggregate error, so it is pinned against a hand-computed case.
    """
    track = bdb.to_play_track(bdb.load(bdb_play()))
    rows = features.feature_rows(track)
    theta = np.deg2rad(rows["dir"].to_numpy())
    assert np.allclose(rows["vx"], rows["s"] * np.sin(theta))
    assert np.allclose(rows["vy"], rows["s"] * np.cos(theta))
    # A man running straight downfield has a bearing of 90, and all of his speed in x.
    straight = np.deg2rad(90.0)
    assert np.sin(straight) == pytest.approx(1.0)
    assert np.cos(straight) == pytest.approx(0.0, abs=1e-12)


def test_horizon_defaults_to_the_release_s_own_output_length():
    track = bdb.to_play_track(bdb.load(bdb_play(frames_output=11)))
    assert track.meta["frames_output"] == 11
    rows = features.feature_rows(track)
    assert sorted(rows.horizon.unique()) == list(range(1, 12))
    assert len(rows) == 6 * 11


def test_a_vision_track_is_refused_rather_than_scored_with_zeros():
    """The failure this module exists to prevent, in its most likely form.

    A track from ``packages/tracking`` has no ball-landing spot and no absolute
    yardline, because video does not contain either. ``decision_engine`` would
    fill them with zeros; a zero ball_land_x is the back of the opposite end zone.
    """
    track = bdb.to_play_track(bdb.load(bdb_play()))
    for key in ("ball_land_x", "ball_land_y", "yardline"):
        del track.meta[key]

    with pytest.raises(ValueError, match="does not carry them in meta"):
        features.feature_rows(track, horizons=5)

    features.with_play_state(track, ball_land_x=58.0, ball_land_y=31.0, yardline=38.0)
    rows = features.feature_rows(track, horizons=5)
    assert len(rows) == 30
    assert "supplied by caller" in track.meta["play_state_source"]


def test_a_horizonless_track_is_refused():
    track = bdb.to_play_track(bdb.load(bdb_play()))
    del track.meta["frames_output"]
    with pytest.raises(ValueError, match="no forecast horizon"):
        features.feature_rows(track)


def test_ragged_tracks_take_each_player_s_own_last_sample():
    """Video loses people. The state row must not be the last frame's membership."""
    track = bdb.to_play_track(bdb.load(bdb_play()))
    victim = sorted(track.roster)[0]
    for f in track.frames[-6:]:
        f.players.pop(victim, None)

    state = features.last_state(track).set_index("player_id")
    assert victim in state.index
    assert state.loc[victim, "t"] < max(track.times)


def test_airborne_truth_is_the_label_the_forecast_is_scored_against(tmp_path):
    """The release's output file: where players actually went while the ball was
    up. Returned in the release's own coordinates, because mirroring belongs in
    exactly one place and this is not it."""
    import pandas as pd

    rows = pd.DataFrame([
        {"game_id": 1, "play_id": 7, "nfl_id": 100, "frame_id": 1, "x": 40.0, "y": 20.0},
        {"game_id": 1, "play_id": 7, "nfl_id": 100, "frame_id": 2, "x": 42.0, "y": 21.0},
        {"game_id": 1, "play_id": 9, "nfl_id": 100, "frame_id": 1, "x": 10.0, "y": 10.0},
        {"game_id": 2, "play_id": 7, "nfl_id": 100, "frame_id": 1, "x": 99.0, "y": 50.0},
    ])
    got = bdb.airborne_truth(rows, game_id=1, play_id=7)

    assert len(got) == 2
    assert set(got.columns) >= {"game_id", "play_id", "player_id", "frame", "x", "y"}
    assert list(got["frame"]) == [1, 2]
    assert list(got["x"]) == [40.0, 42.0]

    csv = tmp_path / "out.csv"
    rows.to_csv(csv, index=False)
    from_disk = bdb.airborne_truth(csv, game_id=1, play_id=7)
    assert len(from_disk) == 2


def test_airborne_truth_without_a_filter_returns_everything():
    import pandas as pd

    rows = pd.DataFrame([
        {"game_id": 1, "play_id": 7, "nfl_id": 1, "frame_id": 1, "x": 1.0, "y": 2.0},
        {"game_id": 2, "play_id": 8, "nfl_id": 2, "frame_id": 1, "x": 3.0, "y": 4.0},
    ])
    assert len(bdb.airborne_truth(rows)) == 2
