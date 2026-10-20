"""A ``PlayTrack`` as feature rows for the trained model. The load-bearing join.

This is the direction that matters. ``bdb`` exists so the model's training data can
be *drawn*; this exists so the model can be *run* on something other than Kaggle —
specifically on whatever ``packages/tracking`` produces from video, which is the
whole reason roles.md promised stream 3 that "stream 2's output is designed to land
in the same schema later". That promise is kept here or nowhere.

**The trap, stated plainly.** ``coordinate_model.joblib`` was fitted on the Big Data
Bowl release exactly as it ships: x and y as the release gives them, with
left-direction plays *not* mirrored — ``play_direction_code`` is a feature precisely
because the model is expected to learn the two cases separately — and ``dir``/``o``
as compass bearings, zero along +y, increasing clockwise. A ``PlayTrack`` is none of
those things: it is mirrored so the offense always attacks +x, and its angles are
the mathematical convention.

Hand a model trained in one frame of reference a row from the other and nothing
raises. The features are all in plausible range, the regressors return plausible
yards, and the predicted play is a reflection of the real one. That is the same
class of bug as ``docs/animation.md``, one layer further in, and it is undetectable
by eye on a scatter of dots.

So every transform ``bdb`` applied is undone here, and :func:`round_trip_error`
exists to prove it: it takes the release's own rows, runs them through
``bdb.to_play_track`` and back through this module, and measures the difference
against the numbers the file started with. It should be zero to floating point. A
test asserts that, and if it ever fails, the model is being fed a mirror image.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from gridiron_tracking.contract import PlayTrack, wrap_deg
from gridiron_tracking.sources import ngs
from gridiron_tracking.sources.ngs import player_key as _key

FEATURE_COLUMNS = [
    "x", "y", "s", "a", "dir", "o", "ball_land_x", "ball_land_y",
    "absolute_yardline_number", "role_code", "play_direction_code",
    "vx", "vy", "horizon",
]
"""The column list and order ``decision_engine.train`` fitted on.

Not imported from ``decision_engine`` deliberately — that module does its work at
import time behind an argparse ``main``, and this package must be usable without a
Kaggle download on disk. :func:`check_against_model` compares this list to the one
stored inside the fitted model file, which is the real authority, and is the test
that catches the two drifting apart."""

ROLE_CODES = {"Passer": 0, "Targeted Receiver": 1, "Other Route Runner": 2,
              "Defensive Coverage": 3}
"""Must stay identical to ``decision_engine.ROLES``. An unknown or absent role is
-1 there, so it is -1 here."""

MISSING_ROLE = -1


def to_ngs_space(x: float, y: float, flip: bool) -> tuple[float, float]:
    """Contract position back to the release's own, mirroring if it was mirrored."""
    return (120.0 - x, 53.3 - y) if flip else (x, y)


def to_ngs_angle(a: float, flip: bool) -> float:
    """Contract degrees back to a compass bearing, un-rotating a mirrored play.

    The exact inverse of ``ngs._angle``: mirroring the field through its centre
    turns every direction by 180, and the bearing conversion is a reflection, which
    is its own inverse.
    """
    a = wrap_deg(a - 180.0) if flip else wrap_deg(a)
    return float(ngs.contract_angle_to_ngs(a))


def last_state(track: PlayTrack, at: float | None = None) -> pd.DataFrame:
    """Every player's last sample at or before ``at``, in the release's frame.

    ``at`` is the instant the forecast starts from, and getting it wrong is not a
    small error. The model was fitted on the 2026 competition file, whose input
    ends at the throw: its state row is the last **pre-throw** frame and its labels
    are where everyone went while the ball was in the air. Hand it the last frame
    of a whole play instead — which is what omitting ``at`` does on a track that
    runs to the tackle — and it is being asked to continue a play that is already
    over, from a state it never saw in training. The numbers come back without
    complaint and mean nothing.

    So callers reviewing a pass pass the throw time; see
    :func:`gridiron_counterfactual.review.review_play`. ``at=None`` keeps the old
    behaviour, which is correct for a track that genuinely ends at the throw.

    Tracks from video are ragged — a player swallowed by the pile is genuinely
    absent from the tail rather than stationary in it — so this takes each player's
    own last sample rather than the last frame's membership, which is what
    ``decision_engine.last_states`` does with its ``groupby(...).tail(1)``.
    """
    flip = str(track.meta.get("play_direction", "right")).lower() == "left"
    roles = track.meta.get("roles") or {}

    rows = []
    for pid in track.roster:
        series = track.series(pid)
        if at is not None:
            series = [(t, p) for t, p in series if t <= at]
        if not series:
            continue
        t, p = series[-1]
        x, y = to_ngs_space(p.x, p.y, flip)
        role = roles.get(pid)
        rows.append({
            "player_id": pid,
            "t": t,
            "x": x,
            "y": y,
            "s": p.s,
            "a": p.a,
            "dir": to_ngs_angle(p.dir, flip),
            "o": to_ngs_angle(p.o, flip),
            "o_source": p.o_source,
            "role": role,
            "role_code": ROLE_CODES.get(role, MISSING_ROLE),
        })
    return pd.DataFrame(rows)


def feature_rows(track: PlayTrack, horizons: int | None = None,
                 at: float | None = None) -> pd.DataFrame:
    """One row per player per forecast step, ready for ``model.predict``.

    ``horizons`` is how many 10 Hz steps ahead to ask for; it defaults to the
    release's own ``num_frames_output`` for this play when ``bdb`` carried it into
    meta, because a forecast scored against the release's labels has to run to
    exactly the horizon the labels cover.

    Raises when the play-level inputs the model was fitted on are absent, rather
    than filling them with zeros. ``decision_engine`` does fill them with zeros —
    ``state[NUMERIC].fillna(0)`` — and on a feature like ``ball_land_x`` a zero is
    not neutral, it is the back of the opposite end zone. Tolerable when every row
    in training got the same treatment; not tolerable when it silently turns one
    play's forecast into nonsense.
    """
    state = last_state(track, at=at)
    if state.empty:
        raise ValueError(
            "this track has no player samples at all"
            if at is None else
            f"no player samples at or before t={at}; the forecast has nothing to "
            f"start from"
        )

    missing = [
        name for name, key in (
            ("ball_land_x", "ball_land_x"),
            ("ball_land_y", "ball_land_y"),
            ("absolute_yardline_number", "yardline"),
        ) if track.meta.get(key) is None
    ]
    if missing:
        raise ValueError(
            f"the fitted model keys on {missing}, and this track does not carry "
            f"them in meta. A play from bdb.to_play_track has them; one from the "
            f"vision pipeline does not, because video does not say where a pass "
            f"was going to land or what the absolute yardline was. Supply them "
            f"from play-by-play with with_play_state() before scoring."
        )

    n = horizons if horizons is not None else track.meta.get("frames_output")
    if n is None:
        raise ValueError(
            "no forecast horizon: pass horizons=, or load the play through "
            "bdb.to_play_track so the release's num_frames_output comes with it"
        )
    n = int(n)
    if n < 1:
        raise ValueError(f"horizons must be at least 1, got {n}")

    direction = str(track.meta.get("play_direction", "right")).lower()
    if direction not in ("left", "right"):
        raise ValueError(
            f"play_direction is {direction!r}; the model takes it as a feature "
            f"(+1 right, -1 left) and cannot be scored without it"
        )

    rows = state.loc[state.index.repeat(n)].copy()
    rows["horizon"] = np.tile(np.arange(1, n + 1), len(state))
    rows["ball_land_x"] = float(track.meta["ball_land_x"])
    rows["ball_land_y"] = float(track.meta["ball_land_y"])
    rows["absolute_yardline_number"] = float(track.meta["yardline"])
    rows["play_direction_code"] = 1.0 if direction == "right" else -1.0

    # Velocity components in the release's convention: dir is a bearing, so the
    # downfield component comes from sin and the across-field one from cos. This is
    # decision_engine's expression, kept character for character on purpose.
    theta = np.deg2rad(rows["dir"].to_numpy(dtype=float))
    rows["vx"] = rows["s"].to_numpy(dtype=float) * np.sin(theta)
    rows["vy"] = rows["s"].to_numpy(dtype=float) * np.cos(theta)

    return rows.reset_index(drop=True)


def with_play_state(
    track: PlayTrack,
    ball_land_x: float,
    ball_land_y: float,
    yardline: float,
    frames_output: int | None = None,
) -> PlayTrack:
    """Attach the play-level inputs video cannot supply, and say they were supplied.

    A track from ``packages/tracking`` knows where twenty-two people were and
    nothing about the situation; the model needs the situation. Rather than let a
    caller poke meta directly, this records that the numbers came from outside the
    track, so provenance can tell a scored vision play from a scored release play.
    """
    track.meta["ball_land_x"] = float(ball_land_x)
    track.meta["ball_land_y"] = float(ball_land_y)
    track.meta["yardline"] = float(yardline)
    if frames_output is not None:
        track.meta["frames_output"] = int(frames_output)
    track.meta["play_state_source"] = "supplied by caller, not measured from this track"
    return track


def check_against_model(model_path: str) -> dict[str, Any]:
    """Compare :data:`FEATURE_COLUMNS` with the list inside the fitted model.

    ``decision_engine`` dumps ``feature_columns`` alongside the regressors, so the
    file itself settles the question. Returns the verdict rather than raising, so a
    caller can report a stale model instead of crashing on one.
    """
    import joblib

    bundle = joblib.load(model_path)
    stored = list(bundle.get("feature_columns") or [])
    return {
        "ok": stored == FEATURE_COLUMNS,
        "stored": stored,
        "expected": list(FEATURE_COLUMNS),
        "note": (
            "match" if stored == FEATURE_COLUMNS else
            "the fitted model was trained on a different feature list than this "
            "module builds. Re-fit the model or update FEATURE_COLUMNS; scoring "
            "across this mismatch returns numbers that mean nothing."
        ),
    }


def round_trip_error(play: pd.DataFrame, game_id: Any = None, play_id: Any = None) -> dict[str, float]:
    """Release rows -> ``PlayTrack`` -> features, measured against the rows.

    The proof that the two conversions are inverses. Every number here should be
    zero to floating point; a non-zero ``dir`` or ``y`` means a mirrored play is
    being fed to the model as though it were not, and the forecast is a reflection
    of the truth.
    """
    from . import bdb

    track = bdb.to_play_track(play, game_id=game_id, play_id=play_id, check_angles=False)
    got = last_state(track).set_index("player_id")

    canon = play if "player_id" in play.columns else ngs.load_tracking(play)
    if play_id is not None and "play_id" in canon.columns:
        canon = canon[canon["play_id"] == play_id]
    if game_id is not None and "game_id" in canon.columns:
        canon = canon[canon["game_id"] == game_id]
    want = (
        canon[canon["team"].astype(str).str.lower() != "ball"]
        .sort_values("frame").groupby("player_id").tail(1)
        .assign(player_id=lambda d: d["player_id"].map(_key))
        .set_index("player_id")
    )

    shared = got.index.intersection(want.index)
    if not len(shared):
        raise ValueError("no players in common between the track and the rows it came from")

    out: dict[str, float] = {"players": float(len(shared))}
    for col in ("x", "y", "s", "dir", "o"):
        if col not in want.columns:
            continue
        a = got.loc[shared, col].to_numpy(dtype=float)
        b = pd.to_numeric(want.loc[shared, col], errors="coerce").to_numpy(dtype=float)
        keep = ~np.isnan(b)
        if not keep.any():
            continue
        if col in ("dir", "o"):
            # Compared the short way round: 359 against 1 is two degrees out.
            d = np.abs(((a[keep] - b[keep]) + 180.0) % 360.0 - 180.0)
        else:
            d = np.abs(a[keep] - b[keep])
        out[f"max_{col}"] = float(d.max())
    return out



