"""Public Next Gen Stats tracking, loaded into the shared contract.

Three jobs, and the middle one is the one that bites.

**Columns.** Every NGS release spells its columns differently — ``frame.id`` in the
2017 Big Data Bowl, ``frameId`` in 2021, ``step`` in the Kaggle contact-detection
set, and ``nflId`` / ``nfl_player_id`` / ``player`` for the same identifier. Rather
than a loader per release, there is one alias table and a resolver that says what it
found when it cannot resolve something.

**Angles.** ``dir`` and ``o`` in NGS data are *compass bearings*: zero degrees points
along +y and the angle increases clockwise. The contract is the mathematical
convention: zero along +x, increasing toward +y. They are not the same and the error
is not a constant offset, it is a reflection — which means a play loaded naively does
not look obviously broken, it looks like everyone is running slightly wrong, and it
survives review. The conversion is ``(90 - a) % 360``, and it is not taken on faith:
:func:`check_angle_convention` differences positions and compares, and on the 2017
release it agrees to 1.5 degrees while every other reading of the column is 60 to 110
degrees out.

**What is missing.** The 2017 release has no ``o`` and no ``a`` at all. That is not a
reason to invent them — it is exactly what ``o_source`` is for, so orientation falls
back to ``from_dir`` and says so, and acceleration is differenced from speed and
marked in the play's meta. A renderer that knows the body angle was inferred can
badge it; one handed a confident-looking number cannot.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd

from ..contract import (
    Ball, EVENTS, PlayTrack, PlayerFrame, RATE, RosterEntry, TrackingFrame, wrap_deg,
)

# Canonical name -> every spelling seen in a public release, best first.
ALIASES: dict[str, tuple[str, ...]] = {
    "game_id": ("gameId", "game_id", "gameKey", "game_key"),
    "play_id": ("playId", "play_id", "playID"),
    "game_play": ("game_play",),
    "player_id": ("nflId", "nfl_player_id", "nflID", "player_id", "player"),
    "display_name": ("displayName", "display_name", "player_name"),
    "jersey": ("jerseyNumber", "jersey_number", "jersey"),
    "position": ("position", "officialPosition", "player_position"),
    "team": ("team", "club", "teamAbbr", "team_abbr"),
    "frame": ("frame.id", "frameId", "frame_id", "step", "frame"),
    "time": ("time", "datetime", "timestamp"),
    "x": ("x",),
    "y": ("y",),
    "s": ("s", "speed"),
    "a": ("a", "acceleration"),
    "dis": ("dis", "distance"),
    "dir": ("dir", "direction"),
    "o": ("o", "orientation"),
    "event": ("event",),
    "play_direction": ("playDirection", "play_direction"),
}

REQUIRED = ("player_id", "frame", "x", "y")


def ngs_angle_to_contract(a: float | np.ndarray) -> float | np.ndarray:
    """Compass bearing (0 = +y, clockwise) to contract degrees (0 = +x, toward +y)."""
    return np.mod(90.0 - np.asarray(a, dtype=float), 360.0)


def contract_angle_to_ngs(a: float | np.ndarray) -> float | np.ndarray:
    """The inverse — which happens to be the same reflection."""
    return np.mod(90.0 - np.asarray(a, dtype=float), 360.0)


# --------------------------------------------------------------------- loading


def resolve_columns(df: pd.DataFrame) -> dict[str, str]:
    """Map canonical names onto whatever this release actually calls them."""
    found: dict[str, str] = {}
    lower = {c.lower(): c for c in df.columns}
    for canon, spellings in ALIASES.items():
        for s in spellings:
            if s in df.columns:
                found[canon] = s
                break
            if s.lower() in lower:
                found[canon] = lower[s.lower()]
                break
    missing = [c for c in REQUIRED if c not in found]
    if missing:
        raise ValueError(
            f"tracking file is missing {missing}; its columns are {list(df.columns)}. "
            f"If this is a release we have not seen, add the spelling to ALIASES."
        )
    return found


def load_tracking(source: str | Path | pd.DataFrame) -> pd.DataFrame:
    """Read a tracking file and rename its columns to the canonical set.

    Everything not in :data:`ALIASES` is dropped; nothing is converted yet, because
    the unit and angle conversions belong to one play at a time.
    """
    df = source if isinstance(source, pd.DataFrame) else pd.read_csv(source)
    cols = resolve_columns(df)
    out = df[[v for v in cols.values()]].rename(columns={v: k for k, v in cols.items()})

    # The Kaggle sets key on a single "58168_003392" string rather than two columns.
    if "game_play" in out.columns and "play_id" not in out.columns:
        parts = out["game_play"].astype(str).str.split("_", n=1, expand=True)
        out["game_id"] = parts[0]
        out["play_id"] = parts[1]
    return out


def plays(df: pd.DataFrame) -> list[tuple[Any, Any]]:
    """Every (game_id, play_id) in the frame, in file order."""
    gid = df["game_id"] if "game_id" in df.columns else pd.Series(["?"] * len(df))
    pid = df["play_id"] if "play_id" in df.columns else pd.Series(["?"] * len(df))
    return list(dict.fromkeys(zip(gid, pid)))


# --------------------------------------------------------------- play direction


@dataclass
class PlayDirection:
    direction: str          # 'right' if the offense already attacks +x
    confidence: float       # 0-1; how far the ball actually travelled
    source: str             # 'column' | 'ball_displacement' | 'assumed'


def infer_play_direction(play: pd.DataFrame) -> PlayDirection:
    """Which way the offense was going, when nothing says so.

    The contract requires the offense to attack +x, and a raw NGS file is in
    stadium coordinates where half the plays run the other way. Releases that carry
    ``playDirection`` say so outright. The 2017 release does not, and neither does
    a file that has been stripped of its play-by-play, so the fallback is the ball:
    from the snap to the end of the play it mostly moves the way the offense was
    going.

    It is a heuristic and it is wrong on a sack, a big loss, or a punt — hence the
    confidence, which is just how many yards of evidence there were. Below about
    three yards, do not trust it: get the direction from play-by-play instead. This
    matters far less for the vision pipeline, which learns the direction from which
    way the defence lines up, but a benchmark that silently mirrors half its ground
    truth is worse than no benchmark.
    """
    if "play_direction" in play.columns and play["play_direction"].notna().any():
        val = str(play["play_direction"].dropna().iloc[0]).lower()
        return PlayDirection("right" if val.startswith("r") else "left", 1.0, "column")

    ball = play[play.get("team", pd.Series(dtype=str)).astype(str).str.lower() == "ball"]
    track = ball if len(ball) > 2 else play
    track = track.sort_values("frame")
    if len(track) < 2:
        return PlayDirection("right", 0.0, "assumed")

    start = track.groupby("frame")["x"].mean().iloc[0]
    end = track.groupby("frame")["x"].mean().iloc[-1]
    dx = float(end - start)
    return PlayDirection(
        "right" if dx >= 0 else "left",
        min(abs(dx) / 10.0, 1.0),
        "ball_displacement",
    )


# ------------------------------------------------------------------ conversion


def to_play_track(
    df: pd.DataFrame,
    game_id: Any = None,
    play_id: Any = None,
    play_direction: str | None = None,
    rate: int = RATE,
) -> PlayTrack:
    """One play, in contract coordinates and contract angles.

    ``play_direction`` overrides the inference; pass it whenever play-by-play is at
    hand, which is always for a benchmark.
    """
    play = df
    if play_id is not None and "play_id" in df.columns:
        play = play[play["play_id"] == play_id]
    if game_id is not None and "game_id" in df.columns:
        play = play[play["game_id"] == game_id]
    if play.empty:
        raise ValueError(f"no rows for game {game_id!r} play {play_id!r}")

    play = play.sort_values("frame").copy()

    pd_ = (
        PlayDirection(play_direction, 1.0, "caller")
        if play_direction
        else infer_play_direction(play)
    )
    flip = pd_.direction == "left"

    teams = play["team"].astype(str).str.lower() if "team" in play.columns else pd.Series("?", index=play.index)
    is_ball = teams == "ball"
    players = play[~is_ball]
    ball_rows = play[is_ball]

    has_o = "o" in play.columns and players["o"].notna().any()
    has_a = "a" in play.columns and players["a"].notna().any()

    # --- roster -----------------------------------------------------------
    roster: dict[str, RosterEntry] = {}
    for pid, grp in players.groupby("player_id"):
        first = grp.iloc[0]
        key = _player_key(first)
        roster[key] = RosterEntry(
            id=key,
            team=str(first.get("team", "?")),
            label=str(first.get("display_name") or key),
            name=str(first.get("display_name") or None) if "display_name" in grp else None,
            num=_maybe_int(first.get("jersey")),
            role=str(first["position"]) if "position" in grp and pd.notna(first.get("position")) else None,
        )

    # --- frames -----------------------------------------------------------
    frame_ids = sorted(play["frame"].unique())
    t0 = _snap_frame(play, frame_ids)
    dt = 1.0 / rate

    ball_by_frame = {int(r["frame"]): r for _, r in ball_rows.iterrows()}
    frames: list[TrackingFrame] = []

    for fid in frame_ids:
        rows = players[players["frame"] == fid]
        pf: dict[str, PlayerFrame] = {}
        for _, r in rows.iterrows():
            x, y = float(r["x"]), float(r["y"])
            if flip:
                x, y = 120.0 - x, 53.3 - y

            d = _angle(r.get("dir"), flip)
            o = _angle(r.get("o"), flip) if has_o else None
            if o is None:
                o, o_src = (d if d is not None else 0.0), "from_dir"
            else:
                o_src = "measured"

            pf[_player_key(r)] = PlayerFrame(
                x=x, y=y,
                s=float(r["s"]) if "s" in rows and pd.notna(r.get("s")) else 0.0,
                a=float(r["a"]) if has_a and pd.notna(r.get("a")) else 0.0,
                dir=d if d is not None else 0.0,
                o=wrap_deg(o),
                o_source=o_src,
            )

        ball = None
        if int(fid) in ball_by_frame:
            b = ball_by_frame[int(fid)]
            bx, by = float(b["x"]), float(b["y"])
            if flip:
                bx, by = 120.0 - bx, 53.3 - by
            ball = Ball(bx, by)

        ev = _events_at(rows, ball_by_frame.get(int(fid)))
        frames.append(TrackingFrame(t=round((fid - t0) * dt, 3), players=pf, ball=ball, events=ev))

    if not has_a:
        _difference_acceleration(frames, dt)

    return PlayTrack(
        meta={
            "game_id": game_id if game_id is not None else _first(play, "game_id"),
            "play_id": play_id if play_id is not None else _first(play, "play_id"),
            "source": "ngs",
            "rate": rate,
            "play_direction": pd_.direction,
            "play_direction_source": pd_.source,
            "play_direction_confidence": round(pd_.confidence, 3),
            "orientation_measured": has_o,
            "acceleration_measured": has_a,
        },
        roster=roster,
        frames=frames,
    )


# --------------------------------------------------------------------- checks


def check_angle_convention(play: pd.DataFrame, min_speed: float = 2.0, rate: int = RATE) -> dict[str, float]:
    """Difference the positions and see which reading of ``dir`` agrees.

    Run this on any release before trusting it. The winning hypothesis should come
    out around a degree or two — the residual is just differencing noise — and every
    other one tens of degrees. If nothing wins, the file's ``dir`` does not mean what
    any of these think it does, and inventing a fourth guess is not the answer.
    """
    dt = 1.0 / rate
    errs: dict[str, list[float]] = {k: [] for k in
                                    ("compass_cw_from_y", "math_ccw_from_x", "compass_cw_from_x", "ccw_from_y")}
    if "dir" not in play.columns:
        return {}

    for _, g in play.groupby("player_id"):
        g = g.sort_values("frame")
        dx, dy = g["x"].diff(), g["y"].diff()
        speed = np.hypot(dx, dy) / dt
        moving = np.asarray(speed > min_speed)
        if not moving.any():
            continue
        mov = np.degrees(np.arctan2(dy, dx)) % 360
        d = g["dir"].to_numpy(dtype=float)
        for name, pred in (
            ("compass_cw_from_y", (90.0 - d) % 360),
            ("math_ccw_from_x", d % 360),
            ("compass_cw_from_x", (-d) % 360),
            ("ccw_from_y", (d + 90.0) % 360),
        ):
            e = (pred - mov.to_numpy() + 180) % 360 - 180
            errs[name].extend(np.abs(e[moving & np.isfinite(e)]).tolist())

    return {k: float(np.median(v)) for k, v in errs.items() if v}


# -------------------------------------------------------------------- helpers


def _player_key(row) -> str:
    pid = row.get("player_id")
    if pd.isna(pid):
        return "ball"
    try:
        return str(int(pid))
    except (TypeError, ValueError):
        return str(pid)


def _maybe_int(v) -> int | None:
    try:
        return int(v) if pd.notna(v) else None
    except (TypeError, ValueError):
        return None


def _first(df: pd.DataFrame, col: str):
    return df[col].iloc[0] if col in df.columns and len(df) else None


def _angle(v, flip: bool) -> float | None:
    """NGS degrees to contract degrees, mirrored if the play has been turned round.

    Mirroring the field through its centre — (x, y) to (120 - x, 53.3 - y) — is a
    180 degree rotation, so every direction turns by 180 too.
    """
    if v is None or (isinstance(v, float) and math.isnan(v)) or pd.isna(v):
        return None
    a = float(ngs_angle_to_contract(float(v)))
    return wrap_deg(a + 180.0) if flip else wrap_deg(a)


def _snap_frame(play: pd.DataFrame, frame_ids: list) -> float:
    """t = 0 is the snap, per the contract. Failing a snap — a kickoff, a punt
    return, a file with the event column stripped — t = 0 is the first frame, which
    is at least a definition rather than a silent offset."""
    if "event" in play.columns:
        snap = play[play["event"].astype(str).isin(("ball_snap", "autoevent_ballsnap", "snap_direct"))]
        if len(snap):
            return float(snap["frame"].iloc[0])
    return float(frame_ids[0])


def _events_at(rows: pd.DataFrame, ball_row) -> list[str]:
    seen: list[str] = []
    for src in (rows.get("event") if "event" in rows else None,):
        if src is None:
            continue
        for v in src.dropna().astype(str).unique():
            if v in EVENTS and v not in seen:
                seen.append(v)
    if ball_row is not None and "event" in ball_row.index:
        v = str(ball_row["event"])
        if v in EVENTS and v not in seen:
            seen.append(v)
    return seen


def _difference_acceleration(frames: list[TrackingFrame], dt: float) -> None:
    """Fill ``a`` from successive speeds when the release does not carry it.

    Central differences, so the value is not half a frame late. This is measurement,
    not modelling — but it is differenced measurement, which is noisier than the
    ``a`` column of a release that has one, and the play's meta records which it was.
    """
    ids = {pid for f in frames for pid in f.players}
    for pid in ids:
        idx = [i for i, f in enumerate(frames) if pid in f.players]
        for k, i in enumerate(idx):
            prev = frames[idx[k - 1]] if k > 0 else None
            nxt = frames[idx[k + 1]] if k + 1 < len(idx) else None
            if prev is None or nxt is None:
                frames[i].players[pid].a = 0.0
                continue
            span = nxt.t - prev.t
            if span <= 0:
                frames[i].players[pid].a = 0.0
                continue
            frames[i].players[pid].a = abs(
                (nxt.players[pid].s - prev.players[pid].s) / span
            )
