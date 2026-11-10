"""Compatibility with the actual Big Data Bowl 2025 tracking interface."""
import csv
import io
import json
from pathlib import Path

import pytest

from gridiron_tracking import bdb
from gridiron_tracking.contract import Ball, PlayTrack, PlayerFrame, RosterEntry, TrackingFrame
from gridiron_tracking.sources import ngs

# Header inspected from the release's tracking_week_1.csv, not the internal CSV.
RELEASE_HEADER = "gameId,playId,nflId,displayName,frameId,frameType,time,jerseyNumber,club,playDirection,x,y,s,a,dis,o,dir,event"


def _track():
    return PlayTrack(meta={"source": "vision", "rate": 10, "play_id": "clip"},
        roster={"p": RosterEntry(id="p", team="teamA")}, frames=[
            TrackingFrame(t=-.1, players={"p": PlayerFrame(x=10, y=20, s=5, o=0, dir=90)}),
            TrackingFrame(t=0, players={"p": PlayerFrame(x=13, y=20, s=5, o=0, dir=90)},
                          ball=Ball(x=15, y=21, source="detected"), events=["ball_snap"]),
            TrackingFrame(t=.1, players={"p": PlayerFrame(x=14, y=20, s=5, o=0, dir=90)},
                          ball=Ball(x=16, y=21, source="coasted")),
        ])


def test_exact_release_header_and_clock_conventions():
    rows, meta = bdb.convert(_track(), {"gameId": 2022091200, "playId": 64,
        "playDirection": "right", "teamClubs": {"teamA": "DEN"},
        "startTime": "2022-09-13T00:16:03.500Z"})
    assert list(rows[0]) == RELEASE_HEADER.split(",")
    players = [r for r in rows if r["nflId"] is not None]
    assert [r["frameId"] for r in players] == [1, 2, 3]
    assert [r["frameType"] for r in players] == ["BEFORE_SNAP", "SNAP", "AFTER_SNAP"]
    assert [r["time"] for r in players] == ["2022-09-13 00:16:03.500", "2022-09-13 00:16:03.600", "2022-09-13 00:16:03.700"]
    assert players[0]["club"] == "DEN" and players[0]["gameId"] == 2022091200
    assert players[0]["o"] == 90 and players[0]["dir"] == 0
    assert meta["snapFrameId"] == 2 and not meta["syntheticGameId"]


def test_dis_is_observed_displacement_not_speed_divided_by_ten():
    rows, _ = bdb.convert(_track())
    players = [r for r in rows if r["nflId"] is not None]
    assert [r["dis"] for r in players] == [None, 3, 1]
    assert players[1]["s"] == 5


def test_unknown_identifiers_and_metadata_are_explicit():
    rows, meta = bdb.convert(_track())
    assert rows[0]["time"] is None and rows[0]["jerseyNumber"] is None
    assert rows[0]["nflId"] >= 900000
    assert meta["playerIdMap"][str(rows[0]["nflId"])]["synthetic"]
    assert meta["syntheticGameId"] and meta["syntheticPlayId"]
    assert not meta["wallClockTimeKnown"] and not meta["coordinatesMeasured"]
    assert bdb.convert(_track())[0][0]["playId"] == rows[0]["playId"]


def test_football_uses_null_nfl_id_and_only_present_ball_samples():
    rows, meta = bdb.convert(_track())
    balls = [r for r in rows if r["club"] == "football"]
    assert len(balls) == 2
    assert all(r["nflId"] is None and r["displayName"] == "football" for r in balls)
    assert meta["ballSourcesByFrame"] == {"2": "detected", "3": "coasted"}
    assert balls[0]["dis"] is None and balls[1]["dis"] == 1
    assert balls[1]["s"] == 10


def test_an_occlusion_does_not_create_displacement_across_missing_samples():
    track = _track()
    track.frames[1].players.clear()
    rows, _ = bdb.convert(track)
    player_rows = [r for r in rows if r["nflId"] is not None]
    assert player_rows[-1]["dis"] is None
    assert len(player_rows) == 2


def test_mirrored_ngs_import_is_restored_to_original_field_and_compass_angles():
    track = _track()
    track.meta.update(source="ngs", mirrored=True, play_direction="left")
    row = bdb.convert(track)[0][0]
    assert row["x"] == 110 and row["y"] == pytest.approx(33.3)
    assert row["o"] == 270 and row["dir"] == 180 and row["playDirection"] == "left"


def test_files_parse_as_bdb_tracking_and_roundtrip_angles(tmp_path):
    track = _track()
    paths = bdb.write_play(tmp_path, track, {"playDirection": "right"})
    assert next(csv.reader(io.StringIO(paths["csv"].read_text()))) == RELEASE_HEADER.split(",")
    raw = ngs.load_tracking(paths["csv"])
    loaded = ngs.to_play_track(raw, play_direction="right")
    point = next(iter(loaded.frames[0].players.values()))
    assert point.x == 10 and point.o == 0 and point.dir == 90
    assert len(json.loads(paths["json"].read_text())) == 5
    assert json.loads(paths["metadata"].read_text())["schema"] == bdb.SCHEMA


def test_header_matches_real_release_when_available():
    path = Path(__file__).resolve().parents[3] / "data/bdb2025/tracking_week_1.csv"
    if not path.exists():
        pytest.skip("private local dataset not checked into Git")
    with path.open() as source:
        assert next(csv.reader(source)) == bdb.COLUMNS
