"""The video-model harness. No API calls — the parse, clock and scoring path."""

import numpy as np
import pytest

from gridiron_tracking import bench
from gridiron_tracking.bench import vlm_video as VV
from gridiron_tracking.contract import PlayTrack, PlayerFrame, RosterEntry, TrackingFrame

pytestmark = pytest.mark.skipif(not VV.HAVE_PYDANTIC, reason="needs pydantic")


def _truth(n=40, players=6):
    rng = np.random.default_rng(0)
    roster = {f"p{i}": RosterEntry(id=f"p{i}", team="offense") for i in range(players)}
    start = rng.uniform([30, 10], [70, 45], size=(players, 2))
    vel = rng.uniform(-3, 3, size=(players, 2))
    frames = []
    for k in range(n):
        t = round(-1.0 + 0.1 * k, 3)
        frames.append(TrackingFrame(
            t=t,
            players={f"p{i}": PlayerFrame(x=float(start[i][0] + vel[i][0] * max(t, 0)),
                                          y=float(start[i][1] + vel[i][1] * max(t, 0)))
                     for i in range(players)},
            events=["ball_snap"] if k == 10 else [],
        ))
    return PlayTrack(meta={"play_id": "unit"}, roster=roster, frames=frames)


def _answer_from(truth, every=3, dx=0.0, relabel=False):
    frames = []
    for f in truth.frames[::every]:
        frames.append(VV.FrameAt(
            t=round(f.t - truth.frames[0].t, 3),
            players=[VV.PlayerAt(player_id=("q" + pid if relabel else pid),
                                 team="offense", x=p.x + dx, y=p.y)
                     for pid, p in f.players.items()],
        ))
    return VV.VideoTracking(frames=frames, frames_actually_seen=len(frames), notes="")


def test_a_perfect_answer_scores_zero():
    """If the harness cannot score a correct answer as correct it measures nothing."""
    t = _truth()
    got = VV.to_play_track(_answer_from(t), snap_offset=-t.frames[0].t)
    s = bench.score(got, t)
    assert s.median_yards == pytest.approx(0.0, abs=1e-9)
    assert s.coverage == pytest.approx(1.0)
    assert s.id_switches == 0


def test_a_known_shift_is_measured_as_that_shift():
    t = _truth()
    got = VV.to_play_track(_answer_from(t, dx=3.0), snap_offset=-t.frames[0].t)
    assert bench.score(got, t).median_yards == pytest.approx(3.0, abs=1e-9)


def test_the_model_may_use_its_own_labels():
    """It has no way to know the league's player ids, so scoring must not depend on them."""
    t = _truth()
    got = VV.to_play_track(_answer_from(t, relabel=True), snap_offset=-t.frames[0].t)
    assert bench.score(got, t).median_yards == pytest.approx(0.0, abs=1e-9)


def test_the_clip_clock_is_shifted_onto_the_snap_clock():
    """The video starts at 0; the contract starts at the snap. Getting this wrong
    scores a correct answer as badly wrong."""
    t = _truth()
    got = VV.to_play_track(_answer_from(t), snap_offset=-t.frames[0].t)
    assert got.frames[0].t == pytest.approx(t.frames[0].t)


def test_sparse_answers_are_reported_as_sparse():
    t = _truth()
    ans = _answer_from(t, every=10)          # 0.1 s grid sampled every 10 -> 1 Hz
    got = VV.to_play_track(ans, snap_offset=-t.frames[0].t)
    span = max(f.t for f in ans.frames) - min(f.t for f in ans.frames)
    hz = (len(ans.frames) - 1) / span
    assert hz == pytest.approx(1.0, abs=0.01)
    assert len(got.frames) < len(t.frames)


def test_a_missing_key_fails_with_a_useful_message(tmp_path, monkeypatch):
    """Hermetic on purpose. The first version of this test popped the key from the
    environment but left the .env search alone, so the moment a real .env existed in
    the repo the test started finding it and failing — a test that depends on the
    developer's credentials is not a test."""
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.chdir(tmp_path)                      # no .env at or above here
    r = VV.run(tmp_path / "nonexistent.mp4", _truth())
    assert r.error and "key" in r.error.lower(), r.error
    assert "aistudio" in r.error


def test_the_prompt_forbids_interpolating_a_denser_series():
    """The failure mode that would make a sparse model look dense and accurate."""
    assert "do not interpolate" in VV.SYSTEM.lower()
    assert "frames_actually_seen" in VV.SYSTEM
    assert "identical across timestamps" in VV.SYSTEM
