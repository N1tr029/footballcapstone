"""Finding the ball by asking who has it, rather than by looking for it.

The detector route is measured in the package README and does not work: 12% of
frames on the reference clip, and a track so static the pipeline refuses it. This
is the other route, and these tests pin the three things that make it honest --
that it never reads a ball, that it says when nobody has one, and that what comes
out is labelled an attribution rather than a sighting.
"""

import numpy as np
import pytest

from gridiron_tracking import carrier
from gridiron_tracking.contract import PlayerFrame, PlayTrack, RosterEntry, TrackingFrame


def a_play(n_frames=12, ball_at=None):
    """Eleven a side in two ranks. ``ball_at`` puts a true ball on every frame."""
    roster, players = {}, []
    for k in range(22):
        pid = f"p{k}"
        roster[pid] = RosterEntry(id=pid, team="offense" if k < 11 else "defense")
        players.append(pid)
    frames = []
    for i in range(n_frames):
        pf = {}
        for k, pid in enumerate(players):
            side = 0 if k < 11 else 1
            pf[pid] = PlayerFrame(x=20.0 + (k % 11) * 2.0 + i * 0.1,
                                  y=20.0 + side * 3.0, s=1.0 + 0.1 * k, dir=90.0)
        f = TrackingFrame(t=round(i * 0.1, 3), players=pf)
        if i == 0:
            f.events = ["ball_snap"]
        if ball_at is not None:
            from gridiron_tracking.contract import Ball
            f.ball = Ball(*ball_at)
        frames.append(f)
    return PlayTrack(frames=frames, roster=roster)


class Always:
    """A stand-in picker: fixed confidence for everybody, so thresholds are testable."""

    def __init__(self, p):
        self.p = p

    def predict_proba(self, X):
        n = len(X)
        # the first man of each frame is the favourite, everyone else is unlikely
        out = np.full((n, 2), 0.01)
        out[::22, 1] = self.p
        out[:, 0] = 1 - out[:, 1]
        return out


def test_the_features_never_look_at_the_ball():
    """The whole point: it runs on a clip where the ball was never found.

    If a ball leaked into the features the model would score well in testing and
    produce nothing on video, which is the failure mode this route exists to avoid.
    """
    without = a_play()
    withball = a_play(ball_at=(60.0, 25.0))
    X1, _ = carrier.frame_features(without)
    X2, _ = carrier.frame_features(withball)
    assert np.array_equal(X1, X2)


def test_direction_goes_in_as_sine_and_cosine():
    """Not as a raw angle, so the convention cannot silently matter.

    A bearing and a mathematical angle differ by a reflection. A model fitted on
    one and fed the other keeps every number in range and points every player the
    wrong way -- the same trap the contract's angle conversion exists for.
    """
    assert "sin_dir" in carrier.FEATURES and "cos_dir" in carrier.FEATURES
    assert "dir" not in carrier.FEATURES
    assert "o" not in carrier.FEATURES, "o is measured on 0% of frames here"

    track = a_play()
    i = carrier.FEATURES.index("sin_dir"), carrier.FEATURES.index("cos_dir")
    X, _ = carrier.frame_features(track)
    assert np.allclose(np.hypot(X[:, i[0]], X[:, i[1]]), 1.0)


def test_nobody_is_named_when_nobody_is_confident():
    """A ball in flight is in no hands, and must not be put in the nearest pair."""
    track = a_play()
    xy, conf = carrier.carrier_by_t(track, Always(0.05))
    assert xy == {} and conf == {}


def test_a_confident_frame_names_somebody_on_the_field():
    track = a_play()
    xy, conf = carrier.carrier_by_t(track, Always(0.9))
    assert len(xy) == len(track.frames)
    for t, (x, y) in xy.items():
        f = next(fr for fr in track.frames if round(fr.t, 3) == t)
        assert any(abs(p.x - x) < 1e-9 and abs(p.y - y) < 1e-9 for p in f.players.values())
        assert conf[t] == pytest.approx(0.9)


def test_the_threshold_is_what_decides():
    assert carrier.carrier_by_t(a_play(), Always(carrier.HELD - 0.01))[0] == {}
    assert carrier.carrier_by_t(a_play(), Always(carrier.HELD + 0.01))[0] != {}


def test_an_attributed_ball_reaches_the_client_saying_so():
    """End to end: attribution must not arrive looking like a sighting.

    This is the join with ball.py's three sources and hindsight's dataOrigin -- a
    position derived from whoever was holding it is not a measurement of where the
    ball was, and a renderer that cannot tell them apart is drawing a guess as a
    fact.
    """
    from gridiron_tracking import ball as ballmod, hindsight

    track = a_play()
    xy, _ = carrier.carrier_by_t(track, Always(0.9))
    bt = ballmod.track_ball([round(f.t, 3) for f in track.frames], {}, carrier_by_t=xy)
    assert bt.counts().get("carrier", 0) == len(track.frames)

    for f, s in zip(track.frames, bt.samples):
        from gridiron_tracking.contract import Ball
        f.ball = Ball(x=s.x, y=s.y, source=s.source)

    doc = hindsight.to_play_json(track)
    origins = {fr["ball"]["dataOrigin"] for fr in doc["frames"] if "ball" in fr}
    assert origins == {"carrier"}
    assert not any(s.measured for s in bt.samples)
