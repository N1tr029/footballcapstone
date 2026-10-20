"""The commands people actually type.

Both ``__main__`` modules were at 0% coverage: ``unity``, ``label`` and
``train-ball`` are the whole interface to this package and none of them had a test.
The parts worth pinning are the ones that write files somewhere real, the ones that
refuse, and the idempotence — ``unity`` reads the folder it writes into, and an
earlier version treated its own previous output as input and accumulated a stale
copy of every clip on each run.
"""

import json

import pytest

from tracking_support import REPO
from gridiron_tracking import __main__ as cli
from gridiron_tracking import hindsight
from gridiron_tracking.contract import (
    Ball, PlayTrack, PlayerFrame, RosterEntry, TrackingFrame,
)


def a_track(n=14, los=47.0):
    """A vision-shaped track good enough for prepare to accept."""
    roster, base = {}, {}
    for i in range(11):
        roster[f"o{i}"] = RosterEntry(id=f"o{i}", team="teamA")
        roster[f"d{i}"] = RosterEntry(id=f"d{i}", team="teamB")
        base[f"o{i}"] = (los - 1.0 - i * 0.3, 8.0 + i * 3.5)
        base[f"d{i}"] = (los + 2.5 + i * 0.3, 8.0 + i * 3.5)

    frames = []
    for i in range(n):
        players = {pid: PlayerFrame(x=x + i * 0.25, y=y, s=2.0, dir=0.0, o=0.0,
                                    o_source="from_velocity")
                   for pid, (x, y) in base.items()}
        frames.append(TrackingFrame(
            t=round((i - 3) / 10, 3), players=players,
            ball=Ball(los + i * 1.5, 26.0),
            events=["ball_snap"] if i == 3 else []))
    return PlayTrack(
        meta={"source": "vision", "rate": 10,
              "ball": {"measured_fraction": 0.8, "travelled_yards": 30.0},
              "registration": {"stage": "computed", "trustworthy": True},
              "broadcast": {"lineOfScrimmage": los, "firstDownLine": los + 6,
                            "yardsToGo": 6, "attacks": "right"}},
        roster=roster, frames=frames)


@pytest.fixture
def runs(tmp_path):
    """A ``data/runs``-shaped directory holding one rendered clip."""
    d = tmp_path / "runs" / "testclip"
    d.mkdir(parents=True)
    (d / "track.json").write_text(json.dumps(a_track().to_json()))
    return tmp_path / "runs"


@pytest.fixture
def plays(tmp_path):
    return tmp_path / "plays"


# ------------------------------------------------------------------- unity


def test_unity_writes_a_play_and_an_index(runs, plays):
    assert cli.main(["unity", "testclip", "--runs", str(runs),
                     "--unity-dir", str(plays)]) == 0

    written = [p for p in plays.glob("*.json") if p.name != "index.json"]
    assert len(written) == 1
    doc = json.loads(written[0].read_text())
    assert doc["gameId"] == cli.CLIP_GAME_ID
    assert len(doc["players"]) == 22, "prepare should have consolidated the roster"

    index = json.loads((plays / "index.json").read_text())
    assert len(index["plays"]) == 1


def test_unity_is_idempotent(runs, plays):
    """It reads the folder it writes into. An earlier version treated its own
    output as a release play and accumulated a stale copy on every run."""
    for _ in range(3):
        cli.main(["unity", "testclip", "--runs", str(runs), "--unity-dir", str(plays)])

    clips = list(plays.glob(f"{cli.CLIP_GAME_ID}_*.json"))
    assert len(clips) == 1
    assert len(json.loads((plays / "index.json").read_text())["plays"]) == 1


def test_unity_keeps_plays_that_were_already_there(runs, plays):
    """The app ships 115 measured plays; an export must not evict them."""
    plays.mkdir(parents=True)
    existing = a_track()
    existing.meta.update(game_id=2022090800, play_id=56, description="a real play")
    hindsight.write_play(plays, existing)

    cli.main(["unity", "testclip", "--runs", str(runs), "--unity-dir", str(plays)])

    index = json.loads((plays / "index.json").read_text())
    ids = {p["id"] for p in index["plays"]}
    assert "2022090800_56" in ids
    assert len(ids) == 2
    # The index names the majority game, not whichever sorted first.
    assert index["gameId"] in (2022090800, cli.CLIP_GAME_ID)


def test_unity_replace_drops_the_release_plays_from_the_index(runs, plays):
    plays.mkdir(parents=True)
    existing = a_track()
    existing.meta.update(game_id=2022090800, play_id=56)
    hindsight.write_play(plays, existing)

    cli.main(["unity", "testclip", "--runs", str(runs), "--unity-dir", str(plays),
              "--replace"])

    index = json.loads((plays / "index.json").read_text())
    assert len(index["plays"]) == 1
    # The file stays on disk; only the index stops pointing at it.
    assert (plays / "2022090800_56.json").exists()


def test_unity_raw_skips_consolidation(runs, plays):
    cli.main(["unity", "testclip", "--runs", str(runs), "--unity-dir", str(plays),
              "--raw"])
    doc = json.loads(next(p for p in plays.glob("*.json") if p.name != "index.json").read_text())
    assert len(doc["players"]) == 22   # this fixture is already 22; raw must not add
    assert all("trackId" in p for p in doc["players"])


def test_unity_on_a_clip_that_was_never_rendered_says_so(tmp_path, plays, capsys):
    empty = tmp_path / "runs"
    empty.mkdir()
    assert cli.main(["unity", "nope", "--runs", str(empty), "--unity-dir", str(plays)]) == 1
    assert "render" in capsys.readouterr().err.lower()


def test_unity_needs_a_clip_or_all():
    with pytest.raises(SystemExit):
        cli.main(["unity"])


# ------------------------------------------------------------------- labelling


def test_label_without_frames_refuses_rather_than_serving_nothing(tmp_path, capsys):
    from gridiron_tracking.label_server import serve

    with pytest.raises(SystemExit) as e:
        serve(frame_dir=tmp_path / "frames", label_file=tmp_path / "labels.json")
    assert "--extract" in str(e.value)


def test_train_ball_refuses_an_empty_store(tmp_path, capsys):
    rc = cli.main(["train-ball", "--frames", str(tmp_path / "frames"),
                   "--labels", str(tmp_path / "labels.json")])
    assert rc == 1
    assert "nothing labelled" in capsys.readouterr().err


def test_train_ball_refuses_too_few_boxes(tmp_path, capsys):
    """A fine-tune on a handful of boxes learns the clip, not a football."""
    from gridiron_tracking import labelling

    frames = tmp_path / "frames"
    frames.mkdir()
    shots = []
    for i in range(3):
        name = f"c_{i:05d}.jpg"
        (frames / name).write_bytes(b"x")
        shots.append(labelling.Shot(key=f"c/{i}", clip="c", frame=i,
                                    path=str(frames / name), width=1280, height=720))
    (frames / "manifest.json").write_text(json.dumps([s.to_json() for s in shots]))

    store = labelling.LabelSet()
    store.labels["c/0"] = labelling.Label(key="c/0", x=1.0, y=2.0)
    store.save(tmp_path / "labels.json")

    rc = cli.main(["train-ball", "--frames", str(frames),
                   "--labels", str(tmp_path / "labels.json")])
    assert rc == 1
    err = capsys.readouterr().err
    assert "only 1 frames have a ball" in err
    assert "--min-boxes" in err


def test_train_ball_export_only_writes_a_dataset_and_stops(tmp_path, capsys):
    from gridiron_tracking import labelling

    frames = tmp_path / "frames"
    frames.mkdir()
    shots = []
    for clip in ("a", "b"):
        for i in range(4):
            name = f"{clip}_{i:05d}.jpg"
            (frames / name).write_bytes(b"x")
            shots.append(labelling.Shot(key=f"{clip}/{i}", clip=clip, frame=i,
                                        path=str(frames / name), width=1280, height=720))
    (frames / "manifest.json").write_text(json.dumps([s.to_json() for s in shots]))

    store = labelling.LabelSet()
    for s in shots:
        store.labels[s.key] = labelling.Label(key=s.key, x=640.0, y=360.0)
    store.save(tmp_path / "labels.json")

    rc = cli.main(["train-ball", "--frames", str(frames),
                   "--labels", str(tmp_path / "labels.json"),
                   "--dataset", str(tmp_path / "ds"),
                   "--min-boxes", "4", "--export-only"])
    assert rc == 0
    assert (tmp_path / "ds/data.yaml").exists()
    assert "dataset at" in capsys.readouterr().out


# ------------------------------------------------------------------- clips


def test_clips_lists_what_is_registered(capsys):
    rc = cli.main(["--spec-dir", str(REPO / "data/clips/specs"), "clips"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "nw_michigan" in out
