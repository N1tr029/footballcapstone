"""The Unity app's schema, checked against the 115 real plays committed with it.

No fixtures here beyond one hand-built case. ``packages/replay-engine/hindsight``
ships the real Big Data Bowl 2025 release through its own exporter, so the reader
and writer are tested against the thing they will actually be handed — which is
also the only way the angle-convention mismatch could be caught. Its ``o`` is a
compass bearing and the contract's is not, and a play converted without the
reflection animates perfectly smoothly with everybody facing the wrong way.
"""

import json

import pytest

from tracking_support import HINDSIGHT_PLAYS
from gridiron_tracking import hindsight
from gridiron_tracking.contract import PlayTrack, validate

pytestmark = pytest.mark.skipif(
    not HINDSIGHT_PLAYS.exists(),
    reason="packages/replay-engine/hindsight is not merged into this tree",
)


def some_plays(n=8):
    return hindsight.plays_in(HINDSIGHT_PLAYS)[:n]


def test_the_real_plays_read_as_valid_tracks():
    files = some_plays()
    assert files, "no play files found"

    for path in files:
        track = hindsight.read_play(path)
        v = validate(track)
        assert v.ok, f"{path.name}: {v.problems}"
        assert len(track.roster) == 22
        assert v.events.get("ball_snap") == 1


def test_the_snap_is_t_zero():
    """The app's clock is a frame index; the contract's is seconds from the snap."""
    for path in some_plays(4):
        track = hindsight.read_play(path)
        assert track.event_time("ball_snap") == pytest.approx(0.0, abs=1e-9)
        # And there is genuinely a pre-snap period, so t really does go negative.
        assert min(track.times) < -1.0


def test_orientation_is_read_as_measured_because_it_was():
    """This is the real release. ``o`` is a measurement, and saying otherwise
    would make the renderer badge good data as a guess."""
    track = hindsight.read_play(some_plays(1)[0])
    sources = {p.o_source for f in track.frames for p in f.players.values()}
    assert sources == {"measured"}
    assert track.meta["orientation_measured"] is True
    # Acceleration is not in the schema, so it is differenced and said to be.
    assert track.meta["acceleration_measured"] is False


@pytest.mark.parametrize("path", some_plays(5), ids=lambda p: p.name)
def test_round_trip_through_the_contract_is_lossless(path):
    """Read the app's own file, convert, convert back, compare to the file.

    The assertion that catches a reflected angle. Both conversions use the same
    reflection, so dropping or double-applying it shows up here as a large error
    rather than as a play that looks slightly odd in Unity.
    """
    original = json.loads(path.read_text())
    track = hindsight.read_play(original)
    again = hindsight.to_play_json(track)

    assert again["schemaVersion"] == original["schemaVersion"]
    assert again["gameId"] == original["gameId"]
    assert again["playId"] == original["playId"]
    assert again["snapFrameId"] == original["snapFrameId"]
    assert again["frameRate"] == original["frameRate"]
    assert len(again["frames"]) == len(original["frames"])
    assert len(again["players"]) == len(original["players"])

    # The situation fields ride through meta untouched.
    for key in ("down", "yardsToGo", "possessionTeam", "lineOfScrimmage",
                "firstDownLine", "passResult", "description", "quarter"):
        assert again[key] == original[key], key

    for a, b in zip(original["frames"], again["frames"]):
        assert a["frameId"] == b["frameId"]
        assert a["frameType"] == b["frameType"]
        assert a["event"] == b["event"]
        if a.get("ball"):
            assert b["ball"]["x"] == pytest.approx(a["ball"]["x"], abs=0.011)
            assert b["ball"]["y"] == pytest.approx(a["ball"]["y"], abs=0.011)

        want = {p["nflId"]: p for p in a["players"]}
        got = {p["nflId"]: p for p in b["players"]}
        assert set(want) == set(got)
        for pid, p in want.items():
            q = got[pid]
            assert q["x"] == pytest.approx(p["x"], abs=0.011), f"{pid} x"
            assert q["y"] == pytest.approx(p["y"], abs=0.011), f"{pid} y"
            assert q["s"] == pytest.approx(p["s"], abs=0.011), f"{pid} s"
            # Angles the short way round: 359.9 against 0.1 is a fifth of a degree.
            for key in ("o", "dir"):
                d = abs((q[key] - p[key] + 180.0) % 360.0 - 180.0)
                assert d < 0.011, f"{pid} {key}: {p[key]} -> {q[key]}"


def test_the_angle_conversion_is_not_a_no_op():
    """Guards the round trip against passing for the wrong reason.

    If both directions silently stopped converting, every assertion above would
    still hold. So the contract value must actually differ from the file's.
    """
    path = some_plays(1)[0]
    original = json.loads(path.read_text())
    track = hindsight.read_play(original)

    f = original["frames"][0]
    pid = str(f["players"][0]["nflId"])
    bearing = f["players"][0]["o"]
    contract_o = track.frames[0].players[pid].o

    assert bearing == pytest.approx((90.0 - contract_o) % 360.0, abs=0.011)
    differs = [
        abs((p["o"] - track.frames[0].players[str(p["nflId"])].o + 180) % 360 - 180) > 1.0
        for p in f["players"]
    ]
    assert any(differs), "no player's angle changed; the reflection is not happening"


def test_a_future_schema_version_is_refused():
    doc = json.loads(some_plays(1)[0].read_text())
    doc["schemaVersion"] = SCHEMA_NEXT = hindsight.SCHEMA_VERSION + 1
    with pytest.raises(ValueError, match=f"schemaVersion {SCHEMA_NEXT}"):
        hindsight.read_play(doc)


def test_a_simulated_play_cannot_be_exported_looking_measured():
    """The one thing this writer must never do."""
    track = hindsight.read_play(some_plays(1)[0])
    track.meta["source"] = "simulated"
    for f in track.frames:
        for p in f.players.values():
            p.o_source = "from_velocity"

    doc = hindsight.to_play_json(track)
    assert doc["simulated"] is True
    assert doc["provenance"]["coordinates"]["simulated"] is True
    assert doc["provenance"]["coordinates"]["note"].startswith("SIMULATED")
    # Per-frame too, because the client draws frames, not documents.
    assert {p["oSource"] for f in doc["frames"] for p in f["players"]} == {"from_velocity"}


def test_the_line_of_scrimmage_comes_from_the_ball_rather_than_a_zero():
    """A defaulted 0 would draw the marker at the back of the end zone."""
    track = hindsight.read_play(some_plays(1)[0])
    ball_x = track.frames[
        min(range(len(track.frames)), key=lambda i: abs(track.frames[i].t))
    ].ball.x

    for key in ("lineOfScrimmage", "firstDownLine"):
        track.meta.pop(key, None)
    doc = hindsight.to_play_json(track, play_state={"yardsToGo": 10})

    assert doc["lineOfScrimmage"] == pytest.approx(ball_x, abs=0.011)
    assert doc["firstDownLine"] == pytest.approx(min(ball_x + 10.0, 110.0), abs=0.011)


def test_writing_a_play_and_an_index_lands_where_unity_looks(tmp_path):
    tracks = [hindsight.read_play(p) for p in some_plays(3)]
    for t in tracks:
        path = hindsight.write_play(tmp_path, t)
        assert path.name == f"{t.meta['game_id']}_{t.meta['play_id']}.json"
        assert hindsight.read_play(path).meta["play_id"] == t.meta["play_id"]

    index = hindsight.write_index(tmp_path, tracks, home_team="LA", visitor_team="BUF")
    doc = json.loads(index.read_text())
    assert index.name == "index.json"
    assert len(doc["plays"]) == 3
    # The client requires plays grouped contiguously by quarter and drive, which
    # holds when they are sorted by playId.
    ids = [p["playId"] for p in doc["plays"]]
    assert ids == sorted(ids)
    assert doc["plays"][0]["id"] == f"{doc['plays'][0]['gameId']}_{ids[0]}"


def test_an_empty_play_is_refused():
    with pytest.raises(ValueError, match="no frames"):
        hindsight.read_play({"schemaVersion": 1, "frames": []})


def test_a_play_with_no_snap_says_its_clock_is_arbitrary():
    doc = json.loads(some_plays(1)[0].read_text())
    del doc["snapFrameId"]
    track = hindsight.read_play(doc)
    assert "clock_warning" in track.meta
    assert track.times[0] == 0.0


# --------------------------------------------------- vision tracks, which have no ids


def _vision_track():
    """A track shaped like the vision pipeline's output: anonymous, team-only."""
    from gridiron_tracking.contract import (
        Ball, PlayTrack, PlayerFrame, RosterEntry, TrackingFrame,
    )

    roster = {f"t{i}": RosterEntry(id=f"t{i}", team="teamA" if i % 2 else "teamB")
              for i in range(22)}
    frames = [
        TrackingFrame(
            t=round(i / 10, 3),
            players={pid: PlayerFrame(x=40.0 + i * 0.3 + j, y=10.0 + j,
                                      s=4.0, dir=0.0, o=0.0, o_source="from_velocity")
                     for j, pid in enumerate(roster)},
            ball=Ball(45.0, 26.0),
            events=["ball_snap"] if i == 0 else [],
        )
        for i in range(12)
    ]
    return PlayTrack(meta={"source": "vision", "rate": 10, "game_id": 0, "play_id": 1},
                     roster=roster, frames=frames)


def test_anonymous_tracks_get_distinct_ids_rather_than_all_becoming_zero():
    """The Unity client matches players by nflId, so a shared id merges them.

    The vision pipeline names its tracks t0, t1, t17. Mapping each to 0 — which is
    what an int() cast does — put all 46 tracks of a real clip onto one player.
    """
    track = _vision_track()
    doc = hindsight.to_play_json(track)

    ids = [p["nflId"] for p in doc["players"]]
    assert len(set(ids)) == len(track.roster) == 22
    assert all(i > hindsight.SYNTHETIC_ID_BASE for i in ids), (
        "synthetic ids must sit above the real nflId range so a vision track "
        "cannot be mistaken for identified players"
    )
    # The original track name survives, or there is no way back to the track.
    assert {p["trackId"] for p in doc["players"]} == set(track.roster)
    # Frames use the same ids as the roster, or the client matches nothing.
    assert {q["nflId"] for f in doc["frames"] for q in f["players"]} <= set(ids)


def test_synthetic_ids_are_stable_across_exports():
    """A re-export must not shuffle the roster."""
    track = _vision_track()
    first = hindsight.nfl_ids(track)
    assert first == hindsight.nfl_ids(_vision_track())
    assert first == hindsight.nfl_ids(track)


def test_real_ids_are_left_alone():
    track = hindsight.read_play(some_plays(1)[0])
    ids = hindsight.nfl_ids(track)
    assert ids == {pid: int(pid) for pid in track.roster}
    assert all(i < hindsight.SYNTHETIC_ID_BASE for i in ids.values())


def test_a_forecast_reads_back_as_a_forecast():
    """Mutation-found gap: dropping the flag on the way in survived the suite.

    A play this package writes as a forecast must not become an ordinary measured
    play the moment it round-trips. It did, and the counterfactual CLI then
    reviewed its own output and wrote forecasts of forecasts.
    """
    track = hindsight.read_play(some_plays(1)[0])
    track.meta["source"] = "simulated"
    track.meta["simulated_from_play"] = 56
    for f in track.frames:
        for p in f.players.values():
            p.o_source = "from_velocity"

    doc = hindsight.to_play_json(track)
    assert doc["simulated"] is True
    assert doc["simulatedFrom"] == 56

    back = hindsight.read_play(doc)
    assert back.meta["simulated"] is True
    assert back.meta["source"] == "simulated"
    assert back.meta["simulated_from_play"] == 56
    # And provenance on the re-read play says what it is, rather than calling it
    # measurement from the league's tracking system.
    from gridiron_tracking import export
    assert export.provenance(back)["coordinates"]["simulated"] is True


def test_a_measured_play_does_not_become_simulated_by_round_tripping():
    track = hindsight.read_play(some_plays(1)[0])
    assert track.meta["simulated"] is False
    back = hindsight.read_play(hindsight.to_play_json(track))
    assert back.meta["simulated"] is False
    assert back.meta["source"] == "ngs"
