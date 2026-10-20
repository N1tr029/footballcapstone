"""The Big Data Bowl 2026 release, in the shared contract.

``decision_engine.py`` reads this release natively and never leaves its dialect.
That is fine for training and useless for showing anybody: the renderer draws a
``PlayTrack``, so the model's own training plays cannot be put on screen beside the
vision pipeline's output until they are in the same units as everything else.

This module is that conversion, and it is a thin wrapper on purpose. The transforms
— the column aliases, the compass-to-maths angle reflection, the mirroring of
left-direction plays — all already exist in ``gridiron_tracking.sources.ngs``,
which was written for the 2017 release. Reimplementing them here would give the
repository two answers to "which way is +y", which is the specific failure
``docs/animation.md`` is a post-mortem of.

Two things about this release in particular.

**It is half a play.** ``input_2023_wXX.csv`` stops at the throw and
``output_2023_wXX.csv`` is the airborne portion, and only for the players the
competition asks you to predict. So a track built from the input alone ends at the
release and carries no ball: the contract validator will say "no ball track" and
"no ball_snap event", and both complaints are correct. They are recorded in meta
rather than silenced, because the alternative is inventing a ball.

**Its angle convention is checked, not assumed.** Conventions have moved between
releases, and a reflected ``dir`` does not look broken — it looks like everyone is
running slightly wrong, and it survives review. ``ngs.check_angle_convention``
settles it empirically by differencing positions, so :func:`to_play_track` runs it
and refuses a play whose angles disagree with its own movement.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from gridiron_tracking.contract import PlayTrack
from gridiron_tracking.sources import ngs
from gridiron_tracking.sources.ngs import player_key as _key

ROLES = ("Passer", "Targeted Receiver", "Other Route Runner", "Defensive Coverage")
"""``player_role`` in the 2026 release. The model keys its features on these, and
:mod:`gridiron_counterfactual.features` must reproduce the same integer coding."""

MAX_ANGLE_DISAGREEMENT_DEG = 15.0
"""How far ``dir`` may sit from the direction the player demonstrably moved before
the conversion is treated as wrong. The 2017 release agrees to 1.5 degrees under
the compass reading and is 60-110 degrees out under every other, so anything in
between means the release has changed and nobody has looked yet."""


def load(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """An ``input_2023_wXX.csv`` with its columns renamed to the canonical set.

    Thin by design — the alias table in ``ngs`` carries the 2026 spellings, so this
    exists to be the name you reach for rather than a second loader.
    """
    return ngs.load_tracking(source)


def plays(df: pd.DataFrame) -> list[tuple[Any, Any]]:
    """Every ``(game_id, play_id)`` in the frame, in file order."""
    return ngs.plays(df)


def to_play_track(
    df: pd.DataFrame,
    game_id: Any = None,
    play_id: Any = None,
    check_angles: bool = True,
) -> PlayTrack:
    """One play of the release, in contract coordinates.

    ``play_direction`` is read from the release's own column rather than inferred,
    and recorded in meta — :mod:`gridiron_counterfactual.features` needs it to put
    the play back into the model's frame of reference, and a track that has lost it
    cannot be scored.
    """
    play = df
    if game_id is not None and "game_id" in play.columns:
        play = play[play["game_id"] == game_id]
    if play_id is not None and "play_id" in play.columns:
        play = play[play["play_id"] == play_id]
    if play.empty:
        raise ValueError(f"no rows for game {game_id!r} play {play_id!r}")

    direction = None
    if "play_direction" in play.columns and play["play_direction"].notna().any():
        direction = str(play["play_direction"].dropna().iloc[0]).lower()

    if check_angles and "dir" in play.columns and play["dir"].notna().any():
        agreement = ngs.check_angle_convention(play)
        best = min(agreement, key=lambda k: agreement[k])
        if agreement[best] > MAX_ANGLE_DISAGREEMENT_DEG:
            raise ValueError(
                f"this release's dir column does not agree with its own movement "
                f"under any known convention (best: {best} at "
                f"{agreement[best]:.1f} deg). Angles have moved between releases "
                f"before; work out which convention this one uses and extend "
                f"ngs.ngs_angle_to_contract before trusting a frame of it. "
                f"Pass check_angles=False only to inspect the damage."
            )

    track = ngs.to_play_track(play, game_id=game_id, play_id=play_id,
                              play_direction=direction)

    # --- the play-level inputs the model keys on -----------------------------
    # Carried in meta rather than dropped: they are not tracking measurements, so
    # they have no home on a frame, but features.py cannot rebuild a feature row
    # without them and re-reading the CSV to find them is how two code paths end
    # up disagreeing about which play they are looking at.
    for name in ("ball_land_x", "ball_land_y", "yardline", "frames_output"):
        if name in play.columns and play[name].notna().any():
            track.meta[name] = float(pd.to_numeric(play[name], errors="coerce").dropna().iloc[0])

    track.meta["source"] = "bdb2026"
    track.meta["play_direction"] = direction or track.meta.get("play_direction")
    track.meta["coverage"] = (
        "Pre-throw only: this release's input file ends at the release, so the "
        "track has no ball and no snap. Both are real gaps, not bugs."
    )

    # --- roles ---------------------------------------------------------------
    # The release says both what a man plays and what he was doing on this snap.
    # to_play_track keeps the former; the latter is what the model reasons about,
    # and the targeted receiver is the one the What-If view is arguing about, so he
    # is starred for the renderer.
    # The per-snap role goes in meta rather than on RosterEntry: that dataclass is
    # a line-for-line mirror of the renderer's contract.js, and growing it to carry
    # one release's vocabulary would mean changing the renderer's contract to suit
    # a Kaggle column. RosterEntry.role stays the position a man plays.
    if "role" in play.columns:
        roles = {
            _key(pid): str(role)
            for pid, role in play.dropna(subset=["role"])
                                 .groupby("player_id")["role"].first().items()
        }
        track.meta["roles"] = {k: v for k, v in roles.items() if k in track.roster}
        for pid, entry in track.roster.items():
            if roles.get(pid) == "Targeted Receiver":
                entry.star = True

    return track





def airborne_truth(
    output_csv: str | Path | pd.DataFrame,
    game_id: Any = None,
    play_id: Any = None,
) -> pd.DataFrame:
    """The ``output_2023_wXX.csv`` rows for one play: where players actually went.

    This is the label the model is scored against, and it is what makes a forecast
    checkable rather than merely plausible. Returned in the release's own
    coordinates — :func:`gridiron_counterfactual.forecast.to_play_track` is where
    the mirroring happens, and it must happen in exactly one place.
    """
    df = output_csv if isinstance(output_csv, pd.DataFrame) else pd.read_csv(output_csv)
    cols = {c.lower(): c for c in df.columns}
    out = df.rename(columns={
        cols.get("game_id", "game_id"): "game_id",
        cols.get("play_id", "play_id"): "play_id",
        cols.get("nfl_id", "nfl_id"): "player_id",
        cols.get("frame_id", "frame_id"): "frame",
    })
    if game_id is not None and "game_id" in out.columns:
        out = out[out["game_id"] == game_id]
    if play_id is not None and "play_id" in out.columns:
        out = out[out["play_id"] == play_id]
    return out
