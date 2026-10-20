"""Every producer of a PlayTrack must produce the same shape.

Four things write one now — the public-release loader, the Big Data Bowl adapter,
the Unity app's exporter read back, and the play model's forecast — plus the
vision pipeline through ``prepare``. The contract fixes what they have to agree on,
and the agreements that are not checked anywhere else are checked here.

The direction of attack is the one that matters most and complains least. The
contract says the offense attacks +x; a release carries a playDirection column and
is mirrored on it, the app's exporter has already normalised, and a broadcast clip
carries nothing at all. A play running the wrong way animates perfectly smoothly,
and the play model takes direction as a feature.
"""

import json

import numpy as np
import pandas as pd
import pytest

from tracking_support import HINDSIGHT_PLAYS, REPO
from gridiron_tracking import field, hindsight, prepare
from gridiron_tracking.contract import RATE, PlayTrack, validate
from gridiron_tracking.sources import ngs


def _ngs_rows(direction="left", n=12):
    """A public-release frame, in the release's own dialect."""
    rows = []
    for frame in range(1, n + 1):
        for pid, team in ((1, "home"), (2, "home"), (3, "away"), (4, "away")):
            sign = -1.0 if direction == "left" else 1.0
            rows.append({
                "gameId": 1, "playId": 7, "nflId": pid, "frame.id": frame,
                "team": team, "displayName": f"P{pid}",
                "x": 60.0 + sign * frame * (1.0 if team == "home" else -0.4),
                "y": 20.0 + pid * 4.0,
                "s": 3.0, "dir": 90.0 if direction == "right" else 270.0,
                "playDirection": direction,
                "event": "ball_snap" if frame == 3 else None,
            })
        rows.append({"gameId": 1, "playId": 7, "nflId": None, "frame.id": frame,
                     "team": "ball", "displayName": "football",
                     "x": 60.0, "y": 26.0, "s": 0.0, "dir": 0.0,
                     "playDirection": direction, "event": None})
    return pd.DataFrame(rows)


def producers():
    """``name -> PlayTrack``, one from each thing that makes them."""
    out = {}

    for direction in ("left", "right"):
        df = ngs.load_tracking(_ngs_rows(direction))
        out[f"ngs ({direction})"] = ngs.to_play_track(df, play_direction=direction)

    path = next(p for p in hindsight.plays_in(HINDSIGHT_PLAYS)
                if not p.name.startswith("999000001_")
                and not json.loads(p.read_text()).get("simulated"))
    out["hindsight exporter"] = hindsight.read_play(path)

    clip = REPO / "data/runs/nw_michigan_snap/track.json"
    if clip.exists():
        prepared, _ = prepare.prepare(PlayTrack.from_json(json.loads(clip.read_text())))
        out["vision via prepare"] = prepared
    return out


@pytest.mark.parametrize("name", list(producers()))
def test_play_direction_and_mirrored_say_the_same_thing(name):
    """``play_direction`` is the direction in the SOURCE; ``mirrored`` is what was
    done about it. The two have to agree, or the same key means different things
    depending on which producer made the track — which is what it did.

    features and forecast key on ``play_direction == "left"`` to put a play back
    into the frame of reference the model was fitted in, so a producer that writes
    "right" because the offense now attacks +x opts its tracks out of that
    conversion without saying so.
    """
    track = producers()[name]
    direction = track.meta.get("play_direction")
    assert direction in ("left", "right"), f"{name}: {direction!r}"

    if "mirrored" in track.meta:
        assert track.meta["mirrored"] == (direction == "left"), (
            f"{name}: play_direction={direction!r} but mirrored="
            f"{track.meta['mirrored']}"
        )


@pytest.mark.parametrize("name", list(producers()))
def test_every_producer_agrees_on_the_clock(name):
    track = producers()[name]
    assert track.meta.get("rate") == RATE
    times = track.times
    assert times == sorted(times), f"{name}: time runs backwards"
    assert len(set(times)) == len(times), f"{name}: duplicate timestamps"
    steps = np.diff(times)
    if len(steps):
        assert np.allclose(steps, 1.0 / RATE, atol=1e-6), (
            f"{name}: samples are not on the {RATE} Hz grid"
        )


@pytest.mark.parametrize("name", list(producers()))
def test_every_producer_stays_on_the_field(name):
    track = producers()[name]
    for f in track.frames:
        for pid, p in f.players.items():
            assert -5.0 <= p.x <= 125.0, f"{name}: {pid} at x={p.x}"
            assert -5.0 <= p.y <= 58.0, f"{name}: {pid} at y={p.y}"
            assert 0.0 <= p.o < 360.0, f"{name}: {pid} o={p.o}"
            assert 0.0 <= p.dir < 360.0, f"{name}: {pid} dir={p.dir}"


@pytest.mark.parametrize("name", list(producers()))
def test_every_producer_says_where_its_orientations_came_from(name):
    track = producers()[name]
    sources = {p.o_source for f in track.frames for p in f.players.values()}
    assert sources, f"{name}: no samples"
    assert sources <= {"measured", "from_dir", "from_velocity", "assumed", "authored"}
    assert "authored" not in sources, (
        f"{name}: a pipeline claimed 'authored', which is reserved for a human "
        f"drawing a play by hand"
    )


@pytest.mark.parametrize("name", list(producers()))
def test_every_producer_names_its_source(name):
    track = producers()[name]
    assert track.meta.get("source") in ("ngs", "vision", "bdb2026", "simulated"), (
        f"{name}: source is {track.meta.get('source')!r}, which provenance cannot "
        f"turn into a sentence about how far to trust the coordinates"
    )


def test_mirroring_is_the_only_thing_that_changes_direction():
    """Both halves of the one transform, on one track, in one place."""
    df = ngs.load_tracking(_ngs_rows("left"))
    track = ngs.to_play_track(df, play_direction="left")
    before = [(p.x, p.y, p.dir, p.o)
              for f in track.frames for p in f.players.values()]

    field.mirror_track(track)
    field.mirror_track(track)
    after = [(p.x, p.y, p.dir, p.o)
             for f in track.frames for p in f.players.values()]

    for a, b in zip(before, after):
        assert a == pytest.approx(b, abs=1e-9)
