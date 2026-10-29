"""Every receiver at the moment of the throw: the table the what-if model learns from.

The question the model answers is "was there a better man to throw to?". Answering it
needs the same description of *every* receiver running a route at the instant the ball
left the quarterback's hand — the one he chose and the ones he did not. That is one row
here: a (play, receiver) pair, in contract coordinates, at ``pass_forward``.

Only the targeted receiver has an outcome. The others are what the model is later asked
to score, which is where the honest limit lives: a quarterback does not throw to a man he
can see is covered, so the throws that were made are not a fair sample of the throws that
could have been. The table keeps every receiver so that the gap is visible rather than
hidden; closing it is the model's problem, not this module's.

Everything here is geometry over one frame (plus one earlier frame for the trend), so it
runs unchanged on tracking that came from video instead of from the league.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import bdb2025

THROW_EVENTS = ("pass_forward", "pass_shovel")
TREND_FRAMES = 5
"""Half a second. Long enough that tracking jitter does not decide whether a receiver is
"getting open", short enough that it is still the same break."""

LANE_HALF_WIDTH = 2.0
"""Yards either side of the straight line from passer to receiver. A defender inside it
is one the ball has to get past; arm length plus a step."""

FEATURES = [
    "depth", "lateral", "sideline", "throw_dist", "separation", "separation_2",
    "separation_trend", "defender_speed", "lane_defenders", "speed", "moving_downfield",
    "facing_passer", "pressure", "time_to_throw", "n_routes",
]

SITUATION = ["down", "yards_to_go", "yards_to_goal"]
"""What a completion is worth depends on these; whether it is completed mostly does not."""


def _dist(ax, ay, bx, by):
    return np.hypot(np.subtract.outer(ax, bx), np.subtract.outer(ay, by))


def _lane_count(px, py, rx, ry, dx, dy) -> np.ndarray:
    """Defenders within LANE_HALF_WIDTH of each passer->receiver segment."""
    vx, vy = rx - px, ry - py                               # (R,)
    length2 = np.maximum(vx * vx + vy * vy, 1e-9)
    wx, wy = dx[None, :] - px, dy[None, :] - py             # (1, D)
    t = np.clip((wx * vx[:, None] + wy * vy[:, None]) / length2[:, None], 0.0, 1.0)
    cx, cy = px + t * vx[:, None], py + t * vy[:, None]
    return (np.hypot(dx[None, :] - cx, dy[None, :] - cy) <= LANE_HALF_WIDTH).sum(axis=1)


PASSER_REACH = 8.0
"""Yards. At the frame marked ``pass_forward`` the ball has sometimes already left the
hand and is nearer a lineman than the man who threw it."""


def receiver_features(now: pd.DataFrame, before: pd.DataFrame | None, offense: str,
                      route_ids: list[int], los: float,
                      quarterbacks: list[int] | None = None) -> pd.DataFrame:
    """One row per receiver in ``route_ids``, from one frame of contract-coordinate rows.

    ``now`` is every row of the play at the throw (22 players and the ball); ``before`` is
    the same play ``TREND_FRAMES`` earlier, or None if the play is too young to have one.
    The passer is the quarterback if one is within ``PASSER_REACH`` of the ball, and
    otherwise whoever on offense is nearest it — which keeps the trick play, where the man
    throwing is not a quarterback, from being credited to one standing forty yards away.
    """
    ball = now[now["nflId"].isna()]
    men = now[now["nflId"].notna()]
    off, dfn = men[men["club"] == offense], men[men["club"] != offense]
    rec = off[off["nflId"].isin(route_ids)]
    if rec.empty or dfn.empty or ball.empty:
        return pd.DataFrame(columns=["nflId", "passer", *FEATURES])

    bx, by = float(ball["x"].iloc[0]), float(ball["y"].iloc[0])
    to_ball = np.hypot(off["x"] - bx, off["y"] - by).to_numpy()
    qb = off["nflId"].isin(quarterbacks or []).to_numpy() & (to_ball <= PASSER_REACH)
    passer = off.iloc[np.where(qb, to_ball, np.inf).argmin() if qb.any() else to_ball.argmin()]
    rec = rec[rec["nflId"] != passer["nflId"]]
    if rec.empty:
        return pd.DataFrame(columns=["nflId", "passer", *FEATURES])
    px, py = float(passer["x"]), float(passer["y"])

    rx, ry = rec["x"].to_numpy(float), rec["y"].to_numpy(float)
    dx, dy = dfn["x"].to_numpy(float), dfn["y"].to_numpy(float)
    d = _dist(rx, ry, dx, dy)                               # receivers x defenders
    order = np.sort(d, axis=1)
    nearest = d.argmin(axis=1)

    trend = np.full(len(rec), np.nan)
    if before is not None and len(before):
        b = before.set_index("nflId")
        ids, def_ids = rec["nflId"].to_numpy(), dfn["nflId"].to_numpy()
        if set(ids) <= set(b.index) and set(def_ids) <= set(b.index):
            was = _dist(b.loc[ids, "x"].to_numpy(float), b.loc[ids, "y"].to_numpy(float),
                        b.loc[def_ids, "x"].to_numpy(float), b.loc[def_ids, "y"].to_numpy(float))
            trend = order[:, 0] - was.min(axis=1)

    to_px, to_py = px - rx, py - ry
    throw = np.hypot(to_px, to_py)
    o = np.deg2rad(rec["o"].to_numpy(float))
    facing = (np.cos(o) * to_px + np.sin(o) * to_py) / np.maximum(throw, 1e-9)

    return pd.DataFrame({
        "nflId": rec["nflId"].to_numpy().astype(int),
        "passer": int(passer["nflId"]),
        "depth": rx - los,                                   # yards past the line of scrimmage
        "lateral": ry - py,                                  # + is the passer's right
        "sideline": np.minimum(ry, bdb2025.FIELD_WIDTH - ry),
        "throw_dist": throw,
        "separation": order[:, 0],
        "separation_2": order[:, 1] if d.shape[1] > 1 else np.nan,
        "separation_trend": trend,                           # + means getting more open
        "defender_speed": dfn["s"].to_numpy(float)[nearest],
        "lane_defenders": _lane_count(px, py, rx, ry, dx, dy),
        "speed": rec["s"].to_numpy(float),
        "moving_downfield": np.cos(np.deg2rad(rec["dir"].to_numpy(float))),
        "facing_passer": facing,                             # 1 looking at him, -1 back turned
        "pressure": float(np.hypot(dx - px, dy - py).min()),  # nearest defender to the passer
        "n_routes": len(rec),
    })


def throw_table(game_ids=None, log=None, throw_offset: int = 0, position_noise: float = 0.0,
                seed: int = 0) -> pd.DataFrame:
    """Every (pass play, receiver) at the throw, with what happened to the one targeted.

    Kept: dropbacks that ended complete, incomplete or intercepted, have a throw event in
    the tracking, and name a targeted receiver. Dropped: sacks and scrambles (no throw),
    spikes, and throwaways (no target), because none of them is a choice between receivers.

    ``throw_offset`` and ``position_noise`` make the tracking deliberately worse, to measure
    what the model needs from tracking recovered from video: the throw detected that many
    frames late (negative is early), and every position moved by a random amount with that
    standard deviation in yards along each axis.
    """
    rng = np.random.default_rng(seed)
    plays = bdb2025.plays()
    plays = plays[plays["passResult"].isin(["C", "I", "IN"]) & ~plays["qbSpike"].fillna(False).astype(bool)]
    pp = bdb2025.player_play()
    routes = pp[pp["wasRunningRoute"] == 1]
    targets = pp[pp["wasTargettedReceiver"] == 1][bdb2025.KEY + ["nflId"]].rename(columns={"nflId": "target"})
    plays = plays.merge(targets.drop_duplicates(bdb2025.KEY), on=bdb2025.KEY)
    weeks = dict(zip(bdb2025.games()["gameId"], bdb2025.games()["week"]))
    position = bdb2025.players().set_index("nflId")["position"]
    quarterbacks = position[position == "QB"].index.tolist()

    out = []
    for gid in sorted(plays["gameId"].unique()) if game_ids is None else game_ids:
        raw = bdb2025.raw_tracking(gid)
        game_plays = plays[plays["gameId"] == gid].set_index("playId")
        raw = raw[raw["playId"].isin(game_plays.index)]
        game_routes = routes[routes["gameId"] == gid].groupby("playId")
        route_ids = {pid: g["nflId"].tolist() for pid, g in game_routes}
        route_name = {(pid, int(r.nflId)): r.routeRan for pid, g in game_routes for r in g.itertuples()}

        for pid, rows in raw.groupby("playId"):
            ev = rows.dropna(subset=["event"])
            thrown = ev.loc[ev["event"].isin(THROW_EVENTS), "frameId"]
            snap = ev.loc[ev["event"] == "ball_snap", "frameId"]
            if thrown.empty or snap.empty or pid not in route_ids:
                continue
            f = int(thrown.iloc[0]) + throw_offset
            play = game_plays.loc[pid]
            rows = bdb2025.contract_frame(rows[rows["frameId"].isin((f, f - TREND_FRAMES))])
            if position_noise:
                rows = rows.assign(x=rows["x"] + rng.normal(0, position_noise, len(rows)),
                                   y=rows["y"] + rng.normal(0, position_noise, len(rows)))
            los = float(bdb2025.line_of_scrimmage(play, rows["playDirection"].iloc[0]))
            feats = receiver_features(rows[rows["frameId"] == f], rows[rows["frameId"] == f - TREND_FRAMES],
                                      play["possessionTeam"], route_ids[pid], los, quarterbacks)
            if feats.empty:
                continue
            feats["time_to_throw"] = (f - int(snap.iloc[0])) / 10.0
            feats["down"] = int(play["down"])
            feats["yards_to_go"] = int(play["yardsToGo"])
            feats["yards_to_goal"] = 110.0 - los
            feats.insert(0, "gameId", gid)
            feats.insert(1, "week", weeks.get(gid))
            feats.insert(2, "playId", pid)
            feats["route"] = [route_name.get((pid, i)) for i in feats["nflId"]]
            feats["targeted"] = feats["nflId"] == int(play["target"])
            feats["pass_result"] = play["passResult"]
            feats["complete"] = play["passResult"] == "C"
            feats["yards_gained"] = play["yardsGained"]
            feats["epa"] = play["expectedPointsAdded"]
            out.append(feats)
        if log:
            log(f"game {gid}: {len(game_plays)} pass plays")

    table = pd.concat(out, ignore_index=True) if out else pd.DataFrame()
    if len(table):
        # The state of the game, so a gap in points can be reported in win probability.
        from .winprob import situations

        state = situations()[bdb2025.KEY + ["wp_before", "seconds_left", "score_diff"]]
        table = table.merge(state, on=bdb2025.KEY, how="left")
    # A play whose targeted receiver is not among the rows cannot teach or be scored.
    if len(table):
        has_target = table.groupby(bdb2025.KEY)["targeted"].transform("any")
        table = table[has_target].reset_index(drop=True)
    return table
