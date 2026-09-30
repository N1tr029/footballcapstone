"""The data file the team consumes — and the provenance that keeps it honest."""

import json

import pandas as pd
import pytest

from gridiron_tracking import export
from gridiron_tracking import interactions as I
from test_interactions import SIDES, _play


def test_csv_is_one_row_per_player_per_frame_in_release_shape():
    track = _play()
    df = export.to_dataframe(track, I.derive(track, offense="home"))
    assert list(df.columns) == export.CSV_COLUMNS
    players = df[df.player_id != "ball"]
    assert len(players) == len(track.frames) * len(track.roster)
    assert set(players.side.unique()) == {"offense", "defense"}
    assert players.has_ball.any(), "nobody is ever marked as carrying it"


def test_the_ball_gets_its_own_rows():
    track = _play()
    df = export.to_dataframe(track, I.derive(track, offense="home"))
    assert (df.player_id == "ball").sum() == len(track.frames)


def test_provenance_says_measured_coordinates_are_measured():
    track = _play()
    p = export.provenance(track, I.derive(track, offense="home"))
    assert p["coordinates"]["measured"] is True
    assert "ground truth" in p["coordinates"]["note"]


def test_provenance_says_vision_coordinates_are_estimates():
    track = _play()
    track.meta["source"] = "vision"
    p = export.provenance(track)
    assert p["coordinates"]["measured"] is False
    assert "ESTIMATED" in p["coordinates"]["note"]
    assert "not exact" in p["coordinates"]["note"]


def test_the_file_refuses_to_pretend_it_knows_where_anyone_was_looking():
    """The one field that cannot be sourced, and must not be quietly emitted."""
    track = _play()
    p = export.provenance(track, I.derive(track, offense="home"))
    assert p["gaze"]["available"] is False
    assert "not measured" in p["gaze"]["note"]
    df = export.to_dataframe(track)
    assert not any("gaze" in c or "look" in c for c in df.columns)


def test_orientation_provenance_is_broken_down_by_source():
    track = _play()
    p = export.provenance(track)
    assert p["orientation"]["measured_fraction"] == pytest.approx(1.0)
    track.frames[0].players["RB"].o_source = "assumed"
    assert export.provenance(track)["orientation"]["measured_fraction"] < 1.0


def test_write_play_produces_both_files_and_they_parse(tmp_path):
    track = _play()
    ix = I.derive(track, offense="home")
    paths = export.write_play(tmp_path, track, ix, stem="p1")
    assert paths["csv"].exists() and paths["json"].exists()

    df = pd.read_csv(paths["csv"])
    assert len(df) > 0

    doc = json.loads(paths["json"].read_text())
    assert set(doc) == {"meta", "provenance", "roster", "frames", "interactions"}
    assert doc["provenance"]["contract_valid"] is True
    assert doc["interactions"]["sides"]["offense"] == "home"
    assert len(doc["frames"]) == len(track.frames)
