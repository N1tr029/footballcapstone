"""Who has the ball, worked out from where everybody is standing.

The detector route is in ``ball.py`` and the README records why it does not work:
a football in a broadcast wide shot is ten pixels of brown, behind somebody for
most of every play, and a fine-tune on 87 labelled balls finds it on 12% of frames
badly enough that ``prepare`` refuses the track.

This takes the other route. Measured on week 1 of the 2025 release, over 200 plays
and 712k player-frames:

* the ball is within **1 yard of some player on 83%** of frames, 3 yards on 98%
* it is genuinely airborne for about **1.1 seconds per play**

So for the large majority of a play the ball is not a thing to find in the picture.
It is a thing somebody is holding, and the question is which of twenty-two already
tracked men that is — a choice among objects we are already following, rather than
a search for a small brown one.

That choice is learned here. Features are relative: how far this man is from the
nearest opponent, the second and third nearest, his own nearest team-mates, how
many opponents are inside five yards, how far he is from the middle of everybody,
his speed, and how long since the snap. Nothing absolute, so a model fitted on one
release transfers to a clip registered by hand.

**What it is worth, measured on held-out plays:**

| positions it is given | kept | median error | within 2 yd |
|---|---|---|---|
| tracking-grade (the release's own) | 72% | 0.46 yd | 90% |
| video-grade (0.45 yd position noise) | 54% | 1.10 yd | 73% |

The second row is the one that matters, and it is a simulation -- Gaussian noise at
the error the bench measures for this pipeline, not a real clip, so read it as an
estimate and not a result. Identity switches are not modelled and would make it
worse.

**It says when it does not know.** Nobody holds a ball in flight, and the model's
confidence knows it: separating held from not-held by confidence alone scores AUC
0.816. Below ``HELD`` the frame is left without a carrier rather than given the
least-bad guess, which is what keeps a thrown ball from being drawn in the hands of
whoever happened to be nearest. Those frames are for ``ball.py`` to coast through
or for the flight to be interpolated across; they are not possession.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np

from .contract import PlayTrack

HELD = 0.30
"""Confidence below which nobody is called the carrier.

Chosen from the calibration rather than by taste: at 0.3 the kept frames are 72% of
the play at 0.46 yd, and dropping to 0.1 buys frames whose median error is 1.33 yd
and which are only 61% genuinely held. The ball is better absent than wrong.
"""

FEATURES = ("s", "a", "sin_dir", "cos_dir", "opp1", "opp2", "opp3", "own1", "own2",
            "n_opp_5", "n_opp_3", "d_cen", "t_snap")
"""Deliberately no raw angle and no absolute position.

``dir`` goes in as its sine and cosine so the model never sees the discontinuity at
0/360, and so the convention cannot silently matter: a bearing and a mathematical
angle differ by a reflection, and a model fitted on one and fed the other points
every player the wrong way while the numbers stay in range. ``o`` is left out
entirely -- this pipeline measures it on 0% of frames (71 degrees of error against
the 2025 release), so it is noise wearing a column name."""

HOLDING_YARDS = 1.5
"""How close a man has to be to the ball to be called its carrier when training.

A ball more than this from everybody is in the air, on the ground, or between
hands, and labelling the nearest man as holding it teaches the model to point at
somebody during a throw.
"""


def frame_features(track: PlayTrack, snap_index: int | None = None) -> tuple[np.ndarray, list[tuple[int, str]]]:
    """Per-player-per-frame features for one play, and what each row refers to.

    Returns ``(X, index)`` where ``index[i]`` is ``(frame_number, player_id)``.
    No ball is read or needed -- that is the whole point.
    """
    if snap_index is None:
        snap_index = next((i for i, f in enumerate(track.frames)
                           if "ball_snap" in f.events), 0)

    rows: list[list[float]] = []
    index: list[tuple[int, str]] = []
    for i, f in enumerate(track.frames):
        pids = list(f.players)
        if len(pids) < 2:
            continue
        xy = np.array([[f.players[p].x, f.players[p].y] for p in pids], float)
        teams = np.array([getattr(track.roster.get(p), "team", "?") for p in pids], object)

        d = np.hypot(xy[:, 0, None] - xy[None, :, 0], xy[:, 1, None] - xy[None, :, 1])
        np.fill_diagonal(d, np.inf)
        same = teams[:, None] == teams[None, :]
        opp = np.where(same, np.inf, d)
        own = np.where(same, d, np.inf)
        opp_s = np.sort(opp, axis=1)
        own_s = np.sort(own, axis=1)
        centre = xy.mean(axis=0)
        d_cen = np.hypot(xy[:, 0] - centre[0], xy[:, 1] - centre[1])

        for k, pid in enumerate(pids):
            p = f.players[pid]
            theta = np.deg2rad(float(getattr(p, "dir", 0.0) or 0.0))
            rows.append([
                float(getattr(p, "s", 0.0) or 0.0),
                float(getattr(p, "a", 0.0) or 0.0),
                float(np.sin(theta)), float(np.cos(theta)),
                _finite(opp_s[k, 0]), _finite(opp_s[k, 1] if opp_s.shape[1] > 1 else np.inf),
                _finite(opp_s[k, 2] if opp_s.shape[1] > 2 else np.inf),
                _finite(own_s[k, 0]), _finite(own_s[k, 1] if own_s.shape[1] > 1 else np.inf),
                float((opp[k] <= 5).sum()), float((opp[k] <= 3).sum()),
                float(d_cen[k]), float(i - snap_index),
            ])
            index.append((i, pid))
    return (np.array(rows, float) if rows else np.zeros((0, len(FEATURES)))), index


def _finite(v: float, cap: float = 60.0) -> float:
    return float(min(v, cap)) if np.isfinite(v) else cap


def carrier_by_t(
    track: PlayTrack,
    model: Any,
    held: float = HELD,
) -> tuple[dict[float, tuple[float, float]], dict[float, float]]:
    """``{t: (x, y)}`` for the frames somebody is holding it, and the confidence.

    Frames below ``held`` are absent from both, which is the honest answer for a
    ball in the air: it is not in anybody's hands, so no hands are offered.
    """
    X, index = frame_features(track)
    if not len(X):
        return {}, {}
    prob = model.predict_proba(X)[:, 1]

    best: dict[int, tuple[float, str]] = {}
    for (i, pid), p in zip(index, prob):
        if i not in best or p > best[i][0]:
            best[i] = (float(p), pid)

    out: dict[float, tuple[float, float]] = {}
    conf: dict[float, float] = {}
    for i, (p, pid) in best.items():
        if p < held:
            continue
        pl = track.frames[i].players.get(pid)
        if pl is None:
            continue
        t = round(track.frames[i].t, 3)
        out[t] = (float(pl.x), float(pl.y))
        conf[t] = p
    return out, conf


def load(path: str | Path) -> Any:
    """The fitted picker. ``None`` when there is not one, so callers can go without."""
    import joblib

    path = Path(path)
    if not path.exists():
        return None
    return joblib.load(path)["model"]


def training_rows(track: PlayTrack) -> tuple[np.ndarray, np.ndarray]:
    """Features and labels for one play that has a real ball on it.

    The label is "this man is the one holding it": nearest to the ball *and* within
    ``HOLDING_YARDS``. Both halves matter. Without the distance bound the nearest
    man is labelled the carrier while the ball is in the air, which teaches the
    model to point at somebody during every throw -- the one error that would make
    a thrown ball look caught.
    """
    truth = {i: (f.ball.x, f.ball.y) for i, f in enumerate(track.frames)
             if f.ball is not None}
    X, index = frame_features(track)
    if not len(X):
        return X, np.zeros(0, int)
    y = np.zeros(len(index), int)
    for k, (i, pid) in enumerate(index):
        if i not in truth:
            continue
        tx, ty = truth[i]
        here = np.hypot(track.frames[i].players[pid].x - tx,
                        track.frames[i].players[pid].y - ty)
        nearest = min(np.hypot(q.x - tx, q.y - ty)
                      for q in track.frames[i].players.values())
        y[k] = int(here <= HOLDING_YARDS and here <= nearest + 1e-9)
    return X, y


def train(tracks, out: str | Path, log=print) -> Path:
    """Fit the picker on plays that carry a real ball, and write it.

    Split by play, never by frame: consecutive frames are nearly the same picture
    and a random split would score memorisation, the same reason the ball detector
    holds out whole clips.
    """
    import joblib
    from sklearn.ensemble import HistGradientBoostingClassifier

    Xs, ys = [], []
    used = 0
    for track in tracks:
        if not any(f.ball is not None for f in track.frames):
            continue
        X, y = training_rows(track)
        if len(X):
            Xs.append(X)
            ys.append(y)
            used += 1
    if not Xs:
        raise ValueError("no plays with a ball on them — this needs a release that "
                         "carries football rows, not one with only a landing spot")

    X = np.vstack(Xs)
    y = np.concatenate(ys)
    log(f"  {used} plays, {len(X)} player-frames, {int(y.sum())} held "
        f"({100 * y.mean():.1f}%)")

    clf = HistGradientBoostingClassifier(max_iter=300, learning_rate=0.1,
                                         random_state=0)
    clf.fit(X, y)
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump({"model": clf, "features": list(FEATURES)}, out)
    log(f"  picker: {out}")
    return out
