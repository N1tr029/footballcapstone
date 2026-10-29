"""decision review for the play model, on the nfl big data bowl 2025 release.

the 2025 release tracks all 22 players for the whole play, so this does it the
honest way: freeze everyone at the moment the ball is thrown, forecast where each
player will be when it arrives, and ask whether a route runner other than the
targeted receiver was projected to be clearly more open.

    tracking at the throw ──> forecast at the catch ──> separation per receiver
                                                     └─> verdict + reason
                                                     └─> suggested-play replay

the forecast never sees anything after the throw, so it judges the throw from
the qb's side of it. the model is evaluated on whole weeks it was not trained on,
against a "keep running at current speed" baseline.

the 2026 path (``decisions`` at the bottom) is untouched: the whatif command in
gridiron_counterfactual still calls it with the 2026 coordinate model.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.ensemble import HistGradientBoostingRegressor

KEYS = ["game_id", "play_id", "nfl_id"]
OPEN_SEPARATION_YARDS = 3.0
ROLES = {"Passer": 0, "Targeted Receiver": 1, "Other Route Runner": 2, "Defensive Coverage": 3}

# ---------------------------------------------------------------------------
# 2025 release
# ---------------------------------------------------------------------------

FIELD_LENGTH = 120.0
FIELD_WIDTH = 160.0 / 3.0
DEMO_GAME_ID = 2022090800  # the game the hindsight app ships with
THROW_EVENTS = {"pass_forward", "pass_shovel"}
ARRIVAL_EVENT = "pass_arrived"
HISTORY_FRAMES = (5, 10)  # 0.5 s and 1.0 s before the throw, at 10 frames a second
PRE_SNAP_FRAMES = 10  # suggested replays start 1 s before the snap

# a route runner has to be projected this much more open than the target to get
# flagged. with the past-the-line rule in review_passes, about 1 throw in 9 is
# flagged across weeks 1-9; the original 1.5 yards flagged about half of all
# throws, which is a model second-guessing everything.
ALTERNATIVE_ADVANTAGE_YARDS = 5.0

# fixed heuristic values, not fitted probabilities (kept from the original engine)
PASS_PROBABILITIES = {
    "consider_alternative": {"target_was_reasonable": 0.25, "consider_alternative": 0.75},
    "target_was_reasonable": {"target_was_reasonable": 0.72, "consider_alternative": 0.28},
    "suggest_run_play": {"pass": 0.2, "run": 0.8},
    "suggest_punt": {"pass": 0.1, "run": 0.2, "punt": 0.7},
}

POSITION_GROUPS = {
    "QB": 0, "RB": 1, "FB": 1, "WR": 2, "TE": 3, "T": 4, "G": 4, "C": 4,
    "DE": 5, "DT": 5, "NT": 5, "OLB": 6, "ILB": 6, "MLB": 6, "LB": 6,
    "CB": 7, "DB": 7, "SS": 8, "FS": 8,
}

FORECAST_FEATURES = [
    "x_rel", "y_mid", "s", "a", "vx", "vy", "ox", "oy",
    "dx5", "dy5", "dx10", "dy10", "cv_dx", "cv_dy", "h",
    "offense", "route", "pos", "opp_dx", "opp_dy", "opp_dist",
]

PASS_CAVEAT = (
    "Retrospective review of the throw. Each player's position when the ball arrived "
    "is forecast from the moment of the throw, then the most open route runner is "
    "compared with the targeted receiver (alternatives must be past the line of "
    "scrimmage). Separation is not catch probability or value: it ignores pressure, "
    "throw difficulty, the qb's read and what a catch is worth. Flagged throws do "
    "complete less often, but their expected points are only slightly lower. "
    "Probabilities are fixed heuristic values, not fitted."
)


def read_2025(data_dir: Path, weeks: list[int] | None = None) -> dict[str, pd.DataFrame]:
    """load the 2025 tables. takes the kaggle download (tracking_week_N.csv) or the
    smaller tracking_subset.csv that make_bdb_subset.py writes."""
    def find(name: str) -> Path:
        path = next(iter(sorted(data_dir.glob(f"**/{name}"))), None)
        if path is None:
            raise FileNotFoundError(f"could not find {name} under {data_dir}")
        return path

    games = pd.read_csv(find("games.csv"))
    week_files = sorted(data_dir.glob("**/tracking_week_*.csv"))
    if weeks:
        week_files = [p for p in week_files if int(p.stem.rsplit("_", 1)[1]) in weeks]
    if week_files:
        tracking = pd.concat([pd.read_csv(p, low_memory=False) for p in week_files], ignore_index=True)
    else:
        tracking = pd.read_csv(find("tracking_subset.csv"), low_memory=False)
    if weeks:
        tracking = tracking[tracking.gameId.isin(games.loc[games.week.isin(weeks), "gameId"])]
    return {
        "tracking": tracking.reset_index(drop=True),
        "plays": pd.read_csv(find("plays.csv"), low_memory=False),
        "player_play": pd.read_csv(find("player_play.csv"), low_memory=False),
        "players": pd.read_csv(find("players.csv"), low_memory=False),
        "games": games,
    }


def normalize(tracking: pd.DataFrame) -> pd.DataFrame:
    """flip plays going left so the offense always attacks +x, same as hindsight's
    exporter. angles are compass bearings (0 = +y, clockwise), so a flip is +180."""
    t = tracking.copy()
    left = t.playDirection.astype(str).str.lower().eq("left")
    t.loc[left, "x"] = FIELD_LENGTH - t.loc[left, "x"]
    t.loc[left, "y"] = FIELD_WIDTH - t.loc[left, "y"]
    for col in ("o", "dir"):
        t.loc[left, col] = (t.loc[left, col] + 180.0) % 360.0
    return t


def denormalize_xy(x, y, left: bool):
    # back to the release's own coordinates for plays going left
    return (FIELD_LENGTH - x, FIELD_WIDTH - y) if left else (x, y)


def pass_windows(tracking: pd.DataFrame) -> pd.DataFrame:
    """one row per pass: the frame it was thrown and the frame it arrived."""
    event = tracking.event.fillna("").astype(str).str.lower()
    frames = tracking.assign(_event=event)
    snap = frames[frames._event.eq("ball_snap")].groupby(["gameId", "playId"]).frameId.min().rename("snap")
    throw = frames[frames._event.isin(THROW_EVENTS)].groupby(["gameId", "playId"]).frameId.min().rename("throw")
    arrive = frames[frames._event.eq(ARRIVAL_EVENT)].groupby(["gameId", "playId"]).frameId.min().rename("arrival")
    w = pd.concat([snap, throw, arrive], axis=1).dropna(subset=["throw", "arrival"]).reset_index()
    w["h"] = w.arrival - w.throw
    w = w[w.h > 0].copy()
    for col in ("throw", "arrival", "h"):
        w[col] = w[col].astype(int)
    return w


def forecast_samples(data: dict[str, pd.DataFrame], windows: pd.DataFrame) -> pd.DataFrame:
    """one row per player per pass: their state at the throw (plus 0.5 s and 1 s of
    history) as features, and where they were when the ball arrived as the target."""
    t = normalize(data["tracking"])
    cols = ["gameId", "playId", "nflId", "frameId", "club", "jerseyNumber", "displayName", "x", "y", "s", "a", "o", "dir"]
    people = t[t.nflId.notna()][cols]

    def at(frame_col: str, back: int = 0) -> pd.DataFrame:
        k = windows[["gameId", "playId", frame_col]].copy()
        k["frameId"] = k[frame_col] - back
        return people.merge(k[["gameId", "playId", "frameId"]], on=["gameId", "playId", "frameId"])

    keys = ["gameId", "playId", "nflId"]
    df = at("throw")
    for back in HISTORY_FRAMES:
        hist = at("throw", back)[keys + ["x", "y"]].rename(columns={"x": f"x_{back}", "y": f"y_{back}"})
        df = df.merge(hist, on=keys, how="left")
    arrived = at("arrival")[keys + ["x", "y"]].rename(columns={"x": "x_arr", "y": "y_arr"})
    df = df.merge(arrived, on=keys, how="left")
    df = df.merge(windows[["gameId", "playId", "h", "throw", "arrival"]], on=["gameId", "playId"])

    plays = data["plays"][["gameId", "playId", "possessionTeam", "absoluteYardlineNumber"]]
    roles = data["player_play"][["gameId", "playId", "nflId", "wasRunningRoute", "wasTargettedReceiver"]]
    direction = t.drop_duplicates(["gameId", "playId"])[["gameId", "playId", "playDirection"]]
    df = (df.merge(plays, on=["gameId", "playId"])
            .merge(roles, on=keys, how="left")
            .merge(data["players"][["nflId", "position"]], on="nflId", how="left")
            .merge(data["games"][["gameId", "week"]], on="gameId", how="left")
            .merge(direction, on=["gameId", "playId"], how="left"))

    df["offense"] = (df.club == df.possessionTeam).astype(int)
    df["route"] = df.wasRunningRoute.fillna(0).astype(int)
    df["target"] = df.wasTargettedReceiver.fillna(0).astype(int)
    df["pos"] = df.position.map(POSITION_GROUPS).fillna(-1)
    bearing, facing = np.deg2rad(df.dir), np.deg2rad(df.o)
    # dir is a bearing, so x comes from sin and y from cos
    df["vx"], df["vy"] = df.s * np.sin(bearing), df.s * np.cos(bearing)
    df["ox"], df["oy"] = np.sin(facing), np.cos(facing)
    for back in HISTORY_FRAMES:
        # no history (a very quick throw)? fall back to what the current speed implies
        df[f"dx{back}"] = (df.x - df[f"x_{back}"]).fillna(df.vx * back / 10)
        df[f"dy{back}"] = (df.y - df[f"y_{back}"]).fillna(df.vy * back / 10)
    df["cv_dx"], df["cv_dy"] = df.vx * df.h / 10, df.vy * df.h / 10
    left = df.playDirection.astype(str).str.lower().eq("left")
    df["los"] = np.where(left, FIELD_LENGTH - df.absoluteYardlineNumber, df.absoluteYardlineNumber)
    df["x_rel"], df["y_mid"] = df.x - df.los, df.y - FIELD_WIDTH / 2
    df = df.join(_nearest_opponent(df))
    df["tx"], df["ty"] = df.x_arr - df.x, df.y_arr - df.y
    return df.reset_index(drop=True)


def _nearest_opponent(df: pd.DataFrame) -> pd.DataFrame:
    # where's the closest guy on the other team at the throw (defenders shadow receivers)
    parts = []
    for _, g in df.groupby(["gameId", "playId"], sort=False):
        xy, off = g[["x", "y"]].to_numpy(dtype=float), g.offense.to_numpy()
        d = np.hypot(xy[:, None, 0] - xy[None, :, 0], xy[:, None, 1] - xy[None, :, 1])
        d[off[:, None] == off[None, :]] = np.inf
        j = d.argmin(axis=1)
        nearest = d.min(axis=1)
        parts.append(pd.DataFrame({
            "opp_dx": np.where(np.isfinite(nearest), xy[j, 0] - xy[:, 0], 0.0),
            "opp_dy": np.where(np.isfinite(nearest), xy[j, 1] - xy[:, 1], 0.0),
            "opp_dist": np.where(np.isfinite(nearest), nearest, 50.0),
        }, index=g.index))
    return pd.concat(parts) if parts else pd.DataFrame(columns=["opp_dx", "opp_dy", "opp_dist"])


def _fit_axis(x: pd.DataFrame, y: pd.Series) -> HistGradientBoostingRegressor:
    model = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.08, max_leaf_nodes=31,
                                          l2_regularization=1.0, random_state=17)
    return model.fit(x, y)


def train_forecast(samples: pd.DataFrame) -> dict:
    """fit the throw -> catch displacement for x and y on every sample given."""
    rows = samples.dropna(subset=["tx", "ty"])
    return {
        "models": {"tx": _fit_axis(rows[FORECAST_FEATURES], rows.tx),
                   "ty": _fit_axis(rows[FORECAST_FEATURES], rows.ty)},
        "feature_columns": list(FORECAST_FEATURES),
        "target": "displacement in yards from the throw to the ball's arrival (offense attacks +x)",
        "training_games": sorted(int(g) for g in rows.gameId.unique()),
        "training_rows": int(len(rows)),
        "sklearn_version": sklearn.__version__,
    }


def predict_forecast(bundle: dict, samples: pd.DataFrame) -> tuple[np.ndarray, np.ndarray]:
    """predicted (x, y) at the ball's arrival, in normalized field coordinates."""
    x = samples[bundle["feature_columns"]]
    return (samples.x.to_numpy() + bundle["models"]["tx"].predict(x),
            samples.y.to_numpy() + bundle["models"]["ty"].predict(x))


def cross_fit_forecast(samples: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """predict every sample with a model that never saw its week (or its game, if
    there's only one week), and score that against simple baselines."""
    groups = samples.week if samples.week.nunique() > 1 else samples.gameId
    px, py = np.full(len(samples), np.nan), np.full(len(samples), np.nan)
    for g in sorted(groups.unique()):
        held = (groups == g).to_numpy()
        if held.all():
            raise ValueError("need at least two weeks (or games) to evaluate the forecast honestly")
        bundle = train_forecast(samples[~held])
        px[held], py[held] = predict_forecast(bundle, samples[held])
    out = samples.assign(px=px, py=py)

    scored = out.dropna(subset=["tx", "ty"])
    def err(dx, dy):
        e = np.hypot(scored.tx - dx, scored.ty - dy)
        return {"mean": round(float(e.mean()), 3), "median": round(float(e.median()), 3),
                "p90": round(float(e.quantile(0.9)), 3)}
    model = err(scored.px - scored.x, scored.py - scored.y)
    speed = err(scored.cv_dx, scored.cv_dy)
    evaluation = {
        "held_out_by": "week" if samples.week.nunique() > 1 else "game",
        "folds": int(groups.nunique()),
        "samples": int(len(scored)),
        "passes": int(scored.groupby(["gameId", "playId"]).ngroups),
        "error_yards_at_arrival": {"model": model, "keep_current_speed": speed,
                                   "stand_still": err(0 * scored.tx, 0 * scored.ty)},
        # a perfect baseline (only happens on made-up straight-line data) has nothing to improve on
        "improvement_over_keep_current_speed": round(1 - model["mean"] / speed["mean"], 3) if speed["mean"] else None,
    }
    return out, evaluation


def _separation(px: np.ndarray, py: np.ndarray, dx: np.ndarray, dy: np.ndarray) -> np.ndarray:
    # distance from each player to the nearest defender
    if len(dx) == 0:
        return np.full(len(px), np.inf)
    return np.hypot(px[:, None] - dx[None, :], py[:, None] - dy[None, :]).min(axis=1)


def _name(row) -> str:
    return f"{row.displayName} ({row.position})" if isinstance(row.position, str) else str(row.displayName)


def _context(plays: pd.DataFrame, game_id: int, play_id: int) -> dict:
    row = plays[(plays.gameId == game_id) & (plays.playId == play_id)]
    if row.empty:
        return {}
    r = row.iloc[0]
    pairs = {"play_description": "playDescription", "pass_result": "passResult", "yards_gained": "yardsGained",
             "down": "down", "yards_to_go": "yardsToGo", "expected_points_added": "expectedPointsAdded",
             "quarter": "quarter", "game_clock": "gameClock"}
    out = {}
    for key, col in pairs.items():
        if col in r and pd.notna(r[col]):
            out[key] = r[col].item() if hasattr(r[col], "item") else r[col]
    return out


def review_passes(predicted: pd.DataFrame, plays: pd.DataFrame,
                  advantage_yards: float = ALTERNATIVE_ADVANTAGE_YARDS) -> list[dict]:
    """one decision report per pass, from the forecast positions at the catch."""
    reports = []
    for (game_id, play_id), d in predicted.groupby(["gameId", "playId"], sort=False):
        offense, defense = d[d.offense.eq(1)], d[d.offense.eq(0)]
        targets = offense[offense.target.eq(1)]
        if targets.empty or defense.empty:
            continue
        # alternatives are guys who actually ran a route and were past the line of
        # scrimmage when the ball got there. never the qb (trick-play routes happen),
        # never the guy who was thrown to. the line rule keeps a wide open 2-yard
        # checkdown from "beating" a downfield completion: across weeks 1-9 it cuts
        # flags from ~18% to ~11% of throws, and flagged throws complete 64% vs 84%.
        alts = offense[offense.route.eq(1) & offense.target.eq(0)
                       & ~offense.position.astype(str).str.upper().eq("QB")
                       & (offense.px >= offense.los)]
        dpx, dpy = defense.px.to_numpy(), defense.py.to_numpy()
        target = targets.iloc[0]
        t_sep = float(_separation(np.array([target.px]), np.array([target.py]), dpx, dpy)[0])
        t_real = float(_separation(np.array([target.x_arr]), np.array([target.y_arr]),
                                   defense.x_arr.to_numpy(), defense.y_arr.to_numpy())[0])
        best, a_sep, a_real = None, None, None
        if not alts.empty:
            seps = _separation(alts.px.to_numpy(), alts.py.to_numpy(), dpx, dpy)
            i = int(seps.argmax())
            best, a_sep = alts.iloc[i], float(seps[i])
            a_real = float(_separation(np.array([best.x_arr]), np.array([best.y_arr]),
                                       defense.x_arr.to_numpy(), defense.y_arr.to_numpy())[0])
        gain = a_sep - t_sep if a_sep is not None else None
        best_sep = max(v for v in (t_sep, a_sep) if v is not None)
        ctx = _context(plays, game_id, play_id)

        if best_sep < OPEN_SEPARATION_YARDS:
            choice = "suggest_punt" if ctx.get("down") == 4 else "suggest_run_play"
            status = "no_receiver_open_on_fourth_down" if choice == "suggest_punt" else "no_receiver_open"
            reason = (f"No receiver was projected to be open when the ball arrived: the most open was "
                      f"{best_sep:.1f} yards from the nearest defender, under the {OPEN_SEPARATION_YARDS:.0f}-yard bar.")
        elif gain is not None and gain >= advantage_yards:
            choice, status = "consider_alternative", "projected_alternative_more_open"
            reason = (f"The model projects {_name(best)} to be {a_sep:.1f} yards from the nearest defender when "
                      f"the ball arrives, vs {t_sep:.1f} for {_name(target)}, who was targeted.")
        else:
            choice, status = "target_was_reasonable", "no_clear_projected_improvement"
            reason = (f"No route runner was projected to be clearly more open than {_name(target)} "
                      f"({t_sep:.1f} yards from the nearest defender when the ball arrives).")

        report = {
            "game_id": int(game_id), "play_id": int(play_id), "decision": choice, "status": status,
            "probabilities": dict(PASS_PROBABILITIES[choice]),
            "confidence": "moderate" if len(alts) >= 2 else "low",
            "evidence": {
                "targeted_receiver_id": int(target.nflId),
                "best_alternative_receiver_id": int(best.nflId) if best is not None else None,
                "targeted_receiver_predicted_separation_yards": round(t_sep, 2),
                "best_alternative_predicted_separation_yards": round(a_sep, 2) if a_sep is not None else None,
                "alternative_separation_advantage_yards": round(gain, 2) if gain is not None else None,
                "best_predicted_receiver_separation_yards": round(best_sep, 2),
                "targeted_receiver_actual_separation_yards": round(t_real, 2),
                "best_alternative_actual_separation_yards": round(a_real, 2) if a_real is not None else None,
                "eligible_alternatives": int(len(alts)),
                "open_threshold_yards": OPEN_SEPARATION_YARDS,
                "advantage_threshold_yards": advantage_yards,
                "throw_frame_id": int(target.throw), "arrival_frame_id": int(target.arrival),
                "tracked_player_count": int(len(d)), "roster_complete": bool(len(d) >= 22),
                "measurement": ("nearest-defender distance at the ball's arrival, from positions forecast at the "
                                "throw by a model that never saw this week"),
            },
            "recommendation_reason": reason,
            "caveat": PASS_CAVEAT,
        }
        report.update(ctx)
        reports.append(report)
    return reports


def review_runs(plays: pd.DataFrame, game_ids: set[int] | None = None) -> list[dict]:
    """flag runs that failed (lost yards, gained nothing, or came up short on 3rd/4th
    down). play-by-play only, no tracking, same rule as the original engine."""
    runs = plays[~plays.isDropback.astype(bool) & ~plays.qbKneel.astype(bool)
                 & ~plays.qbSpike.fillna(False).astype(bool)].copy()
    if game_ids is not None:
        runs = runs[runs.gameId.isin(game_ids)]
    bad = runs.yardsGained.le(0) | (runs.down.ge(3) & runs.yardsGained.lt(runs.yardsToGo))
    reports = []
    for r in runs[bad].itertuples(index=False):
        report = {
            "game_id": int(r.gameId), "play_id": int(r.playId),
            "decision": "suggest_pass_next_time", "status": "unsuccessful_running_play",
            "probabilities": {"run_again": 0.2, "pass_next_time": 0.8}, "confidence": "low",
            "evidence": {"yards_gained": float(r.yardsGained), "down": int(r.down), "yards_to_go": float(r.yardsToGo)},
            "recommendation_reason": (f"The run gained {r.yardsGained:g} yards on {int(r.down)} and {int(r.yardsToGo)}, "
                                      "short of what the down needed."),
            "caveat": ("Rule-based post-play suggestion from play-by-play only. A single failed run does not "
                       "establish that passing would have been better."),
        }
        report.update(_context(plays, int(r.gameId), int(r.playId)))
        reports.append(report)
    return reports


def _path(p0: np.ndarray, v0: np.ndarray, p_end: np.ndarray, tau: float, horizon: float) -> np.ndarray:
    # leaves at the speed he had at the throw and lands on the forecast spot
    return p0 + v0 * tau + (p_end - p0 - v0 * horizon) * (tau / horizon) ** 2


def suggested_play(data: dict[str, pd.DataFrame], window: pd.Series, predicted: pd.DataFrame,
                   report: dict) -> dict | None:
    """a replay of one pass for the frontend: recorded tracking until the throw, then
    the forecast, with the ball flying to the receiver the model recommends."""
    game_id, play_id = int(window.gameId), int(window.playId)
    t = data["tracking"]
    play = t[(t.gameId == game_id) & (t.playId == play_id)]
    snap = int(window.snap) if pd.notna(window.snap) else int(window.throw)
    start = max(int(play.frameId.min()), snap - PRE_SNAP_FRAMES)
    throw, arrival = int(window.throw), int(window.arrival)
    recorded = play[(play.frameId >= start) & (play.frameId <= throw)]
    if sorted(recorded.frameId.unique()) != list(range(start, throw + 1)):
        return None  # needs every frame up to the throw (the subset only has them for some games)

    left = str(play.playDirection.iloc[0]).lower() == "left"
    plays, games = data["plays"], data["games"]
    ctx = plays[(plays.gameId == game_id) & (plays.playId == play_id)].iloc[0]
    game = games[games.gameId == game_id].iloc[0]
    roster = predicted.set_index("nflId")
    rec = report["evidence"]
    recommended = rec["best_alternative_receiver_id"] if report["decision"] == "consider_alternative" else rec["targeted_receiver_id"]

    def num(v):
        # a missing angle or speed in the tracking would make invalid json
        return round(float(v), 2) if pd.notna(v) else 0.0

    def role(info) -> str:
        if info.target == 1:
            return "Targeted Receiver"
        if info.route == 1:
            return "Route Runner"
        if str(info.position).upper() == "QB" and info.offense == 1:
            return "Passer"
        return "Offense" if info.offense == 1 else "Defense"

    # who's who goes in once per play; frames only carry what changes
    people_info = [{
        "nfl_id": int(nid), "player_name": info.displayName,
        "jersey_number": None if pd.isna(info.jerseyNumber) else int(info.jerseyNumber),
        "club": info.club, "player_position": info.position if isinstance(info.position, str) else None,
        "player_side": "Offense" if info.offense == 1 else "Defense", "player_role": role(info),
    } for nid, info in roster.iterrows()]

    def person(row, record_type, x, y, s, a, direction, o, o_source):
        return {"nfl_id": int(row), "x": num(x), "y": num(y), "s": num(s), "a": num(a),
                "dir": num(direction), "o": num(o), "o_source": o_source if pd.notna(o) else "assumed",
                "record_type": record_type}

    frames = []
    for frame_id in range(start, throw + 1):
        rows = recorded[recorded.frameId == frame_id]
        people = rows[rows.nflId.notna()]
        ball = rows[rows.nflId.isna()]
        frames.append({
            "frame_id": frame_id,
            "players": [person(int(r.nflId), "observed_player", r.x, r.y, r.s, r.a, r.dir, r.o, "measured")
                        for r in people.itertuples(index=False)],
            "ball": None if ball.empty else {"x": round(float(ball.x.iloc[0]), 2), "y": round(float(ball.y.iloc[0]), 2),
                                             "data_origin": "recorded"},
            "events": [],
        })

    # after the throw: everyone follows a smooth path to their forecast spot
    horizon = (arrival - throw) / 10.0
    p0 = predicted[["x", "y"]].to_numpy(dtype=float)
    v0 = predicted[["vx", "vy"]].to_numpy(dtype=float)
    p_end = predicted[["px", "py"]].to_numpy(dtype=float)
    ids = predicted.nflId.astype(int).to_numpy()
    ball_start = frames[-1]["ball"]
    if ball_start is None or recommended not in roster.index:
        return None
    bx0, by0 = (ball_start["x"], ball_start["y"])
    rx, ry = denormalize_xy(roster.loc[recommended].px, roster.loc[recommended].py, left)
    prev = p0
    for step in range(1, arrival - throw + 1):
        tau = step / 10.0
        cur = np.array([_path(p0[i], v0[i], p_end[i], tau, horizon) for i in range(len(ids))])
        delta = cur - prev
        speed = np.hypot(delta[:, 0], delta[:, 1]) * 10.0
        bearing = np.degrees(np.arctan2(delta[:, 0], delta[:, 1])) % 360.0
        if left:
            bearing = (bearing + 180.0) % 360.0
        players = []
        for i, nid in enumerate(ids):
            x, y = denormalize_xy(cur[i, 0], cur[i, 1], left)
            # the forecast has no body angle, so facing follows the path
            players.append(person(nid, "simulated_player", x, y, speed[i], 0.0, bearing[i], bearing[i], "from_velocity"))
        frac = step / (arrival - throw)
        frames.append({
            "frame_id": throw + step, "players": players,
            "ball": {"x": round(float(bx0 + (rx - bx0) * frac), 2), "y": round(float(by0 + (ry - by0) * frac), 2),
                     "data_origin": "simulated_straight_line"},
            "events": [],
        })
        prev = cur

    by_id = {f["frame_id"]: f for f in frames}
    if snap in by_id:
        by_id[snap]["events"].append({"name": "ball_snap", "data_origin": "recorded"})
    by_id[throw]["events"].append({"name": "pass_forward", "data_origin": "recorded"})
    by_id[arrival]["events"].append({"name": "pass_arrived", "data_origin": "simulated"})

    return {
        "schema_version": 1, "data_release": "nfl-big-data-bowl-2025",
        "game_id": game_id, "play_id": play_id, "play_start_frame_id": snap,
        "source_frame_id": throw, "play_direction": "left" if left else "right",
        "coordinates": "release coordinates (not flipped); flip plays going left so the offense attacks +x",
        "home_team": game.homeTeamAbbr, "visitor_team": game.visitorTeamAbbr,
        "possession_team": ctx.possessionTeam, "defensive_team": ctx.defensiveTeam,
        "quarter": int(ctx.quarter), "game_clock": ctx.gameClock, "down": int(ctx.down),
        "yards_to_go": int(ctx.yardsToGo), "absolute_yardline_number": float(ctx.absoluteYardlineNumber),
        "play_description": ctx.playDescription,
        "pass_result": ctx.passResult if isinstance(ctx.passResult, str) else "",
        "recommendation": {
            "decision": report["decision"], "target_nfl_id": int(recommended),
            "targeted_nfl_id": int(rec["targeted_receiver_id"]), "reason": report["recommendation_reason"],
            "evidence": rec, "caveat": report["caveat"],
        },
        "roster": people_info,
        "frames": frames,
    }


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description="Forecast every player at the catch from the throw (Big Data Bowl 2025) "
                                                 "and review each pass.")
    default_data = next((p for p in (here / "data" / "2025", here.parents[1] / "data" / "kaggle" / "big-data-bowl-2025")
                         if p.exists()), here / "data" / "2025")
    parser.add_argument("--data-dir", type=Path, default=default_data,
                        help="2025 Kaggle download, or the folder make_bdb_subset.py wrote")
    parser.add_argument("--weeks", type=int, nargs="+", help="only these weeks (default: all)")
    parser.add_argument("--game-id", type=int, default=DEMO_GAME_ID, help="game to write reviews and replays for")
    parser.add_argument("--advantage-yards", type=float, default=ALTERNATIVE_ADVANTAGE_YARDS)
    parser.add_argument("--reports-dir", type=Path, default=here / "reports")
    parser.add_argument("--model-output", type=Path, default=here / "models" / "forecast_2025.joblib")
    args = parser.parse_args()

    data = read_2025(args.data_dir, args.weeks)
    windows = pass_windows(data["tracking"])
    samples = forecast_samples(data, windows)
    predicted, evaluation = cross_fit_forecast(samples)
    print(json.dumps(evaluation, indent=2))

    bundle = train_forecast(samples)
    bundle["evaluation"] = evaluation
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_output)
    args.reports_dir.mkdir(parents=True, exist_ok=True)
    (args.reports_dir / "forecast_2025_evaluation.json").write_text(json.dumps(evaluation, indent=2) + "\n")

    game = predicted[predicted.gameId == args.game_id]
    if game.empty:
        raise ValueError(f"game {args.game_id} has no reviewable passes in this data")
    reports = review_passes(game, data["plays"], args.advantage_yards) + review_runs(data["plays"], {args.game_id})
    reports.sort(key=lambda r: r["play_id"])
    (args.reports_dir / "decisions.jsonl").write_text("\n".join(json.dumps(r) for r in reports) + "\n")
    flat = pd.DataFrame([{**{k: v for k, v in r.items() if k != "evidence"}, **r["evidence"]} for r in reports])
    flat.to_csv(args.reports_dir / "decisions.csv", index=False)

    game[["gameId", "playId", "nflId", "displayName", "position", "offense", "route", "target",
          "throw", "arrival", "x", "y", "px", "py", "x_arr", "y_arr"]].to_csv(
        args.reports_dir / "forecast_2025.csv", index=False)

    by_play = {(r["game_id"], r["play_id"]): r for r in reports}
    replays = []
    for w in windows[windows.gameId == args.game_id].itertuples(index=False):
        report = by_play.get((int(w.gameId), int(w.playId)))
        if report is None or report["decision"] not in {"consider_alternative", "target_was_reasonable"}:
            continue
        payload = suggested_play(data, pd.Series(w._asdict()), game[game.playId == w.playId], report)
        if payload is not None:
            replays.append(payload)
    (args.reports_dir / "suggested_plays.jsonl").write_text("\n".join(json.dumps(p, allow_nan=False) for p in replays) + "\n")

    counts = pd.Series([r["decision"] for r in reports]).value_counts().to_dict()
    print(json.dumps({"game": args.game_id, "decisions": counts, "suggested_plays": len(replays),
                      "model": str(args.model_output)}, indent=2))


# ---------------------------------------------------------------------------
# 2026 release (used by gridiron_counterfactual's whatif command; unchanged)
# ---------------------------------------------------------------------------

def decisions(state: pd.DataFrame, predictions: pd.DataFrame, data_dir: Path) -> list[dict]:
    role_col = "player_role" if "player_role" in state else ("role" if "role" in state else None)
    if not role_col:
        raise ValueError("Need player_role/role in input data to identify receivers and defenders.")
    p = predictions.sort_values("frame_id").groupby(KEYS, as_index=False).tail(1)
    # Keep the identity/position from the source state as a second guard. A
    # mistaken role label must not make a quarterback eligible for a suggestion.
    position_col = next((c for c in ("player_position", "position") if c in state), None)
    quarterback_ids = set()
    if position_col:
        quarterback_ids = set(state.loc[
            state[position_col].astype(str).str.strip().str.upper().eq("QB"), "nfl_id"
        ].dropna())
    reports = []
    supplementary_paths = list(data_dir.glob("**/supplementary_data.csv"))
    supplementary = pd.read_csv(supplementary_paths[0], low_memory=False) if supplementary_paths else pd.DataFrame()
    for (game_id, play_id), play in p.groupby(["game_id", "play_id"]):
        offense = play[play.player_role.isin(["Targeted Receiver", "Other Route Runner"])
                       & ~play.nfl_id.isin(quarterback_ids)]
        defense = play[play.player_role.eq("Defensive Coverage")]
        if offense.empty or defense.empty:
            continue
        d = np.sqrt((offense.pred_x.to_numpy()[:, None] - defense.pred_x.to_numpy()[None, :]) ** 2 +
                    (offense.pred_y.to_numpy()[:, None] - defense.pred_y.to_numpy()[None, :]) ** 2)
        offense = offense.copy()
        offense["predicted_separation_yards"] = d.min(axis=1)
        target = offense[offense.player_role.eq("Targeted Receiver")]
        if target.empty:
            continue
        actual = target.iloc[0]
        alternatives = offense[offense.player_role.eq("Other Route Runner")].sort_values("predicted_separation_yards", ascending=False)
        best = alternatives.iloc[0] if len(alternatives) else None
        actual_sep = float(actual.predicted_separation_yards)
        alt_sep = float(best.predicted_separation_yards) if best is not None else None
        gain = alt_sep - actual_sep if alt_sep is not None else None
        # Typed judgment schema inspired by lightweight decision engines: bounded choice,
        # probabilities, confidence, and concise evidence instead of free-form claims.
        best_sep = max([value for value in (actual_sep, alt_sep) if value is not None])
        down = None
        if not supplementary.empty:
            ctx = supplementary[(supplementary.game_id == game_id) & (supplementary.play_id == play_id)]
            if not ctx.empty and "down" in ctx:
                down = pd.to_numeric(ctx.iloc[0].down, errors="coerce")
        if best_sep < OPEN_SEPARATION_YARDS:
            if down == 4:
                choice, status = "suggest_punt", "no_receiver_open_on_fourth_down"
                probs = {"pass": 0.1, "run": 0.2, "punt": 0.7}
            else:
                choice, status = "suggest_run_play", "no_receiver_open"
                probs = {"pass": 0.2, "run": 0.8}
        elif gain is not None and gain >= 1.5:
            choice, status = "consider_alternative", "modeled_alternative_more_open"
            probs = {"target_was_reasonable": 0.25, "consider_alternative": 0.75}
        else:
            choice, status = "target_was_reasonable", "no_clear_modeled_improvement"
            probs = {"target_was_reasonable": 0.72, "consider_alternative": 0.28}
        report = {
            "game_id": int(game_id), "play_id": int(play_id), "decision": choice,
            "status": status, "probabilities": probs,
            "confidence": "low" if len(alternatives) < 2 else "moderate",
            "evidence": {
                "targeted_receiver_id": int(actual.nfl_id),
                "tracked_player_count": int(len(play)),
                "roster_complete": False,
                "roster_coverage": "modeled_players_with_assigned_roles",
                "targeted_receiver_predicted_separation_yards": round(actual_sep, 2),
                "best_alternative_receiver_id": int(best.nfl_id) if best is not None else None,
                "best_alternative_predicted_separation_yards": round(alt_sep, 2) if alt_sep is not None else None,
                "alternative_separation_advantage_yards": round(gain, 2) if gain is not None else None,
                "best_predicted_receiver_separation_yards": round(best_sep, 2),
                "open_threshold_yards": OPEN_SEPARATION_YARDS,
                "measurement": "predicted nearest-defender distance at the last tracked airborne frame"
            },
            "caveat": "Retrospective heuristic over modeled players with assigned roles, not a causal estimate of the best throw; it does not model quarterback reads, pressure, throw difficulty, coverage responsibility, or catch skill."
        }
        if best is not None and gain is not None and gain >= 1.5:
            report["recommendation_reason"] = (
                f"The model projects receiver {int(best.nfl_id)} to finish {gain:.1f} yards farther "
                f"from the nearest defender than the actual target ({alt_sep:.1f} vs {actual_sep:.1f} yards)."
            )
        else:
            report["recommendation_reason"] = (
                "The model found no alternative receiver with a clear projected separation advantage; "
                "the actual target is retained as the suggested target."
            )
        if not supplementary.empty:
            context = supplementary[(supplementary.game_id == game_id) & (supplementary.play_id == play_id)]
            if not context.empty:
                play_context = context.iloc[0]
                for col in ("play_description", "pass_result", "yards_gained", "down", "yards_to_go", "expected_points_added"):
                    if col in context.columns and pd.notna(play_context[col]):
                        report[col] = play_context[col].item() if hasattr(play_context[col], "item") else play_context[col]
        reports.append(report)
    return reports


if __name__ == "__main__":
    main()
