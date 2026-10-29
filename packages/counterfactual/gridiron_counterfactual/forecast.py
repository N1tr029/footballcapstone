"""Predicted coordinates as a ``PlayTrack`` the renderer can draw, labelled SIMULATED.

``decision_engine`` predicts x and y, per player, per 10 Hz step into the future,
and then measures distances between them. It never produces a play. Stream 1 draws
plays. This is the last link of the chain roles.md describes — stream 3 "produces
... a simulated tracking-frame array for stream 1 to draw" — and the array has to
be a real ``PlayTrack``, not a lookalike, because the renderer's validator is the
only thing standing between a broken forecast and a confident-looking animation.

Two positions per player per frame is less than the contract wants. Speed,
acceleration and direction of travel are differenced from the predicted path, which
is legitimate — they are definitionally derivable from it. Body orientation is not:
nothing in the model predicts which way a man will be facing. So ``o`` is inferred
from the forecast velocity and every sample is stamped ``from_velocity``, or
``assumed`` where the predicted path barely moves. A renderer reading ``o_source``
can badge the drawing as a drawing. That is the entire reason the field exists.

``meta["source"]`` is ``"simulated"``, which ``export.provenance`` recognises and
answers with a note saying the play did not happen before it says anything else.
Do not change that string without reading the other side.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from gridiron_tracking import kinematics, orientation
from gridiron_tracking.contract import (
    RATE, PlayTrack, PlayerFrame, RosterEntry, TrackingFrame, wrap_deg,
)

from gridiron_tracking.sources.ngs import player_key as _key

from .features import to_ngs_space


def to_play_track(
    predictions: pd.DataFrame,
    source: PlayTrack,
    label: str = "counterfactual",
    x_col: str = "pred_x",
    y_col: str = "pred_y",
    step_col: str = "horizon",
    t0: float | None = None,
) -> PlayTrack:
    """Forecast rows plus the play they continue, as one simulated play.

    ``predictions`` is what comes off ``model.predict`` with the ids kept beside it:
    one row per player per step, positions in the release's own frame of reference,
    which is what the model emits and what ``features`` fed it.

    ``source`` is the real play the forecast continues. Its roster is reused, so
    the forecast contains the same players the source actually tracks. Some sources
    (including this competition release) contain only a subset of the 22 players;
    the forecast cannot add players without observed states. Its ``play_direction``
    is also reused to mirror positions back into contract coordinates. Passing the
    wrong source play mirrors the forecast and nothing complains, so the two must
    come from the same ``bdb.to_play_track``.
    """
    for col in (x_col, y_col, step_col, "player_id"):
        if col not in predictions.columns:
            raise ValueError(
                f"predictions needs a {col!r} column; it has {list(predictions.columns)}"
            )
    if predictions.empty:
        raise ValueError("no predicted rows")

    flip = str(source.meta.get("play_direction", "right")).lower() == "left"
    dt = 1.0 / RATE
    # The forecast continues from the instant it was made, which is the throw on a
    # pass rather than the end of the play. Defaults to the end of the source track,
    # which is right only when the track itself ends where the forecast starts.
    if t0 is None:
        t0 = source.times[-1] if source.frames else 0.0

    steps = sorted(int(s) for s in predictions[step_col].unique())
    if steps != list(range(1, len(steps) + 1)):
        raise ValueError(
            f"the forecast steps are {steps[:5]}...; they must be 1..N with no gaps, "
            f"because the contract's frames are a clock and a missing step is a "
            f"player teleporting rather than a player absent"
        )

    # --- per player, in contract coordinates, with derived kinematics ---------
    by_player: dict[str, dict[int, PlayerFrame]] = {}
    for pid, grp in predictions.groupby("player_id"):
        key = _key(pid)
        grp = grp.sort_values(step_col)
        xy = np.array([
            to_ngs_space(float(r[x_col]), float(r[y_col]), flip)
            for _, r in grp.iterrows()
        ], dtype=float)
        # to_ngs_space is its own inverse, which is why it serves both directions:
        # the release-space prediction comes back to contract space through it.
        t = t0 + np.asarray(grp[step_col], dtype=float) * dt

        k = kinematics.derive(t, xy)
        team = source.roster[key].team if key in source.roster else None
        o = orientation.from_motion(
            t, k["s"], k["dir"], default=orientation.resting_default(team)
        )

        by_player[key] = {
            int(step): PlayerFrame(
                x=float(xy[i, 0]), y=float(xy[i, 1]),
                s=float(k["s"][i]), a=float(k["a"][i]),
                dir=float(wrap_deg(k["dir"][i])),
                o=float(wrap_deg(o.o[i])),
                o_source=o.source[i],
            )
            for i, step in enumerate(grp[step_col].astype(int))
        }

    frames = [
        TrackingFrame(
            t=round(t0 + step * dt, 3),
            players={pid: f[step] for pid, f in by_player.items() if step in f},
            ball=None,
            events=[],
        )
        for step in steps
    ]

    roster = {
        pid: RosterEntry(**{**vars(entry)})
        for pid, entry in source.roster.items()
        if pid in by_player
    }
    unknown = set(by_player) - set(source.roster)
    if unknown:
        for pid in sorted(unknown):
            roster[pid] = RosterEntry(id=pid, team="?", label=pid)

    meta = {
        "game_id": source.meta.get("game_id"),
        "play_id": source.meta.get("play_id"),
        "source": "simulated",
        "simulated_from": source.meta.get("source", "unknown"),
        "label": label,
        "rate": RATE,
        "play_direction": source.meta.get("play_direction"),
        "t0": t0,
        "roles": dict(source.meta.get("roles") or {}),
        "roster_complete": source.meta.get("roster_complete"),
        "roster_coverage": source.meta.get("roster_coverage", "unspecified"),
        "roster_coverage_note": source.meta.get("roster_coverage_note"),
        "orientation_measured": False,
        "acceleration_measured": False,
        "predicted_by": "packages/counterfactual decision_engine coordinate regressors",
        "note": (
            "This play did not happen. Positions are a model's forecast; speed, "
            "acceleration and direction are differenced from them; body "
            "orientation is not predicted by anything and is inferred from the "
            "forecast velocity, as o_source on every sample says."
        ),
    }
    if unknown:
        meta["players_not_in_source_roster"] = sorted(unknown)

    return PlayTrack(meta=meta, roster=roster, frames=frames)


def continue_play(source: PlayTrack, forecast: PlayTrack) -> PlayTrack:
    """The real play and its forecast on one clock, for the side-by-side view.

    Stream 1 draws "two plays on one clock: the real one and the simulated one".
    The forecast's timestamps already continue the source's, so this is a
    concatenation — but the result is marked ``simulated`` as a whole, because a
    track that is real up to the throw and invented afterwards is not a real track,
    and a viewer scrubbing through it has no way to see the join unless it is
    recorded. ``simulated_from_t`` is where the invention starts.
    """
    if not forecast.frames:
        raise ValueError("the forecast has no frames")
    cut = forecast.frames[0].t

    meta = {
        **source.meta,
        "source": "simulated",
        "simulated_from": source.meta.get("source", "unknown"),
        "simulated_from_t": cut,
        "note": (
            f"Real up to t={cut}, simulated after it. Every frame at or after that "
            f"time is a model's forecast and must be labelled as one."
        ),
    }
    roster = {**source.roster, **{k: v for k, v in forecast.roster.items()
                                 if k not in source.roster}}
    frames = [f for f in source.frames if f.t < cut] + list(forecast.frames)
    return PlayTrack(meta=meta, roster=roster, frames=frames)


def predict(
    track: PlayTrack,
    model_path: str,
    horizons: int | None = None,
    at: float | None = None,
) -> pd.DataFrame:
    """Score one play with the fitted model, returning rows :func:`to_play_track` takes.

    The join, end to end, in one call: a ``PlayTrack`` from anywhere — the release,
    or the vision pipeline with ``features.with_play_state`` applied — becomes a
    forecast. Needs ``joblib`` and ``scikit-learn``, which is why it is a function
    here rather than an import at the top of the module.
    """
    import joblib

    from . import features

    bundle = joblib.load(model_path)
    cols = list(bundle.get("feature_columns") or features.FEATURE_COLUMNS)
    models = bundle["models"]

    rows = features.feature_rows(track, horizons=horizons, at=at)
    missing = [c for c in cols if c not in rows.columns]
    if missing:
        raise ValueError(
            f"the fitted model wants {missing}, which this feature builder does not "
            f"produce. Run features.check_against_model to see both lists."
        )
    out = rows[["player_id", "horizon"]].copy()
    out["pred_x"] = models["x"].predict(rows[cols])
    out["pred_y"] = models["y"].predict(rows[cols])
    return out



