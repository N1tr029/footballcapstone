"""Small-data NFL pass-play movement model and typed post-play decision report.

This uses the NFL Big Data Bowl 2026 tracking files. It forecasts player positions
while the ball is airborne, then compares the targeted receiver with other route
runners using predicted separation at the final tracked frame.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

KEYS = ["game_id", "play_id", "nfl_id"]
ID_FRAME = KEYS + ["frame_id"]
NUMERIC = ["x", "y", "s", "a", "dir", "o", "ball_land_x", "ball_land_y", "absolute_yardline_number"]
ROLES = {"Passer": 0, "Targeted Receiver": 1, "Other Route Runner": 2, "Defensive Coverage": 3}
OPEN_SEPARATION_YARDS = 3.0


def read_weeks(data_dir: Path, weeks: list[int]) -> tuple[pd.DataFrame, pd.DataFrame]:
    inputs, outputs = [], []
    for week in weeks:
        input_name = f"input_2023_w{week:02d}.csv"
        output_name = f"output_2023_w{week:02d}.csv"
        input_path = data_dir / "train" / input_name
        output_path = data_dir / "train" / output_name
        # KaggleHub and the Kaggle download often preserve an outer archive folder.
        if not input_path.exists():
            matches = list(data_dir.glob(f"**/train/{input_name}"))
            if matches:
                input_path = matches[0]
                output_path = input_path.with_name(output_name)
        if not input_path.exists() or not output_path.exists():
            raise FileNotFoundError(f"Could not find {input_name} and {output_name} under {data_dir}; check the Kaggle download.")
        inputs.append(pd.read_csv(input_path))
        outputs.append(pd.read_csv(output_path))
    return pd.concat(inputs, ignore_index=True), pd.concat(outputs, ignore_index=True)


def last_states(inputs: pd.DataFrame) -> pd.DataFrame:
    required = set(KEYS + ["frame_id", "x", "y"])
    missing = required - set(inputs.columns)
    if missing:
        raise ValueError(f"Input tracking file is missing columns: {sorted(missing)}")
    state = inputs.sort_values("frame_id").groupby(KEYS, as_index=False).tail(1).copy()
    for col in NUMERIC:
        if col not in state:
            state[col] = np.nan
        state[col] = pd.to_numeric(state[col], errors="coerce")
    state[NUMERIC] = state[NUMERIC].fillna(0)
    role_col = "player_role" if "player_role" in state else ("role" if "role" in state else None)
    state["role_code"] = state[role_col].map(ROLES).fillna(-1) if role_col else -1
    state["play_direction_code"] = state.get("play_direction", pd.Series("right", index=state.index)).map({"right": 1, "left": -1}).fillna(0)
    return state


def feature_rows(state: pd.DataFrame, frames: pd.DataFrame) -> pd.DataFrame:
    rows = frames.merge(state.drop(columns=["frame_id"]), on=KEYS, how="inner")
    # Output frame_id is a future-step counter (1..num_frames_output), not the
    # pre-throw frame_id sequence in the input file.
    rows["horizon"] = rows["frame_id"]
    theta = np.deg2rad(rows["dir"])
    rows["vx"] = rows["s"] * np.sin(theta)
    rows["vy"] = rows["s"] * np.cos(theta)
    cols = ["x", "y", "s", "a", "dir", "o", "ball_land_x", "ball_land_y", "absolute_yardline_number",
            "role_code", "play_direction_code", "vx", "vy", "horizon"]
    return rows, cols


def train(data_dir: Path, weeks: list[int], max_games: int, model_dir: Path):
    inputs, outputs = read_weeks(data_dir, weeks)
    games = sorted(inputs.game_id.dropna().unique())[:max_games]
    inputs = inputs[inputs.game_id.isin(games)]
    outputs = outputs[outputs.game_id.isin(games)]
    state = last_states(inputs)
    # A held-out portion of the selected games gives a quick diagnostic, not a season-level estimate.
    all_games = sorted(state.game_id.unique())
    valid_games = set(all_games[-max(1, len(all_games) // 5):]) if len(all_games) >= 4 else set()
    train_games = set(all_games) - valid_games
    if not train_games:
        train_games = set(all_games)
        valid_games = set()
    frames = outputs[ID_FRAME].drop_duplicates()
    training_rows, cols = feature_rows(state, frames)
    training_rows = training_rows.merge(outputs[ID_FRAME + ["x", "y"]].rename(columns={"x": "target_x", "y": "target_y"}), on=ID_FRAME, how="inner")
    training_rows = training_rows.replace([np.inf, -np.inf], np.nan).dropna(subset=cols + ["target_x", "target_y"])
    fit_rows = training_rows[training_rows.game_id.isin(train_games)]
    if fit_rows.empty:
        raise ValueError("No matching input/output player-frame rows found. Check that files are from the same competition download.")
    models = {}
    for axis in ("x", "y"):
        model = HistGradientBoostingRegressor(max_iter=120, max_leaf_nodes=15, l2_regularization=1.0, random_state=17)
        model.fit(fit_rows[cols], fit_rows[f"target_{axis}"])
        models[axis] = model
    model_dir.mkdir(parents=True, exist_ok=True)
    joblib.dump({"models": models, "feature_columns": cols, "training_games": games}, model_dir / "coordinate_model.joblib")
    score = None
    if valid_games:
        check = training_rows[training_rows.game_id.isin(valid_games)]
        if len(check):
            px, py = models["x"].predict(check[cols]), models["y"].predict(check[cols])
            score = float(np.mean([mean_absolute_error(check.target_x, px), mean_absolute_error(check.target_y, py)]))
    # Forecast all tracked players, including non-targeted route runners which
    # are not present in the competition's output labels.
    horizons = outputs.groupby(["game_id", "play_id"], as_index=False).frame_id.max()
    state_keys = state[KEYS].drop_duplicates()
    forecast_frames = []
    for row in horizons.itertuples(index=False):
        players = state_keys[(state_keys.game_id == row.game_id) & (state_keys.play_id == row.play_id)]
        if players.empty:
            continue
        frame_ids = pd.DataFrame({"frame_id": np.arange(1, int(row.frame_id) + 1)})
        forecast_frames.append(players.merge(frame_ids, how="cross"))
    forecast_index = pd.concat(forecast_frames, ignore_index=True)
    forecast_rows, _ = feature_rows(state, forecast_index)
    predicted = forecast_rows.copy()
    predicted["pred_x"] = models["x"].predict(predicted[cols])
    predicted["pred_y"] = models["y"].predict(predicted[cols])
    return inputs, state, outputs, predicted, score, len(fit_rows), len(games)


def decisions(state: pd.DataFrame, predictions: pd.DataFrame, data_dir: Path) -> list[dict]:
    role_col = "player_role" if "player_role" in state else ("role" if "role" in state else None)
    if not role_col:
        raise ValueError("Need player_role/role in input data to identify receivers and defenders.")
    p = predictions.sort_values("frame_id").groupby(KEYS, as_index=False).tail(1)
    reports = []
    supplementary_paths = list(data_dir.glob("**/supplementary_data.csv"))
    supplementary = pd.read_csv(supplementary_paths[0], low_memory=False) if supplementary_paths else pd.DataFrame()
    for (game_id, play_id), play in p.groupby(["game_id", "play_id"]):
        offense = play[play.player_role.isin(["Targeted Receiver", "Other Route Runner"])]
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
                "targeted_receiver_predicted_separation_yards": round(actual_sep, 2),
                "best_alternative_receiver_id": int(best.nfl_id) if best is not None else None,
                "best_alternative_predicted_separation_yards": round(alt_sep, 2) if alt_sep is not None else None,
                "alternative_separation_advantage_yards": round(gain, 2) if gain is not None else None,
                "best_predicted_receiver_separation_yards": round(best_sep, 2),
                "open_threshold_yards": OPEN_SEPARATION_YARDS,
                "measurement": "predicted nearest-defender distance at the last tracked airborne frame"
            },
            "caveat": "Retrospective tracking-based heuristic, not a causal estimate of the best throw; ignores QB pressure, timing, throw difficulty, coverage assignments, and catch skill."
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


def suggested_play_rows(inputs: pd.DataFrame, predictions: pd.DataFrame, reports: list[dict]) -> pd.DataFrame:
    """Build replay-like records from the observed input and modeled future.

    The competition data has player tracking only. Ball positions and event times
    are explicitly labeled simulations/inferences, rather than presented as
    measured tracking values.
    """
    report_by_play = {
        (r["game_id"], r["play_id"]): r for r in reports
        if "recommendation_reason" in r and r["decision"] in {"consider_alternative", "target_was_reasonable"}
    }
    output = []
    tracked_columns = list(inputs.columns)
    for (game_id, play_id), report in report_by_play.items():
        observed = inputs[(inputs.game_id == game_id) & (inputs.play_id == play_id)].copy()
        future = predictions[(predictions.game_id == game_id) & (predictions.play_id == play_id)].copy()
        if observed.empty or future.empty:
            continue
        observed["record_type"] = "observed_player"
        observed["event_name"] = ""
        observed["ball_x"] = np.nan
        observed["ball_y"] = np.nan
        observed["data_origin"] = "recorded_input"
        observed["source_frame_id"] = observed["frame_id"]
        observed["play_start_frame_id"] = pd.to_numeric(observed["frame_id"]).min()
        observed["recommendation_reason"] = report["recommendation_reason"]
        observed["recommended_target_nfl_id"] = report.get("best_alternative_receiver_id") if report["decision"] == "consider_alternative" else report["evidence"].get("targeted_receiver_id")
        output.append(observed)

        # Keep the same tracking columns as the source data for Unity/import use.
        last = observed.sort_values("frame_id").groupby("nfl_id", as_index=False).tail(1).set_index("nfl_id")
        target_id = observed[observed.player_role.eq("Targeted Receiver")].nfl_id.iloc[0]
        rec_id = report.get("best_alternative_receiver_id") if report["decision"] == "consider_alternative" else target_id
        if rec_id not in last.index or rec_id is None:
            rec_id = target_id
        rec_future = future[future.nfl_id == rec_id].sort_values("frame_id")
        if rec_future.empty:
            continue
        passer_rows = observed[observed.player_role.eq("Passer")].sort_values("frame_id")
        passer = passer_rows.iloc[-1] if not passer_rows.empty else observed.sort_values("frame_id").iloc[-1]
        max_input_frame = int(pd.to_numeric(observed.frame_id).max())
        play_start_frame = int(pd.to_numeric(observed.frame_id).min())
        max_future_horizon = int(pd.to_numeric(future.frame_id).max())
        frames = []
        for nfl_id, trajectory in future.sort_values("frame_id").groupby("nfl_id"):
            if nfl_id not in last.index:
                continue
            base = last.loc[nfl_id]
            for row in trajectory.itertuples(index=False):
                d = {c: base.get(c, np.nan) for c in tracked_columns}
                frame_id = max_input_frame + int(row.frame_id)
                d.update(game_id=game_id, play_id=play_id, nfl_id=nfl_id, frame_id=frame_id,
                         x=float(row.pred_x), y=float(row.pred_y), player_to_predict=False,
                         record_type="simulated_player", event_name="", ball_x=np.nan, ball_y=np.nan,
                         data_origin="model_prediction", source_frame_id=max_input_frame,
                         play_start_frame_id=int(pd.to_numeric(observed.frame_id).min()),
                         recommendation_reason=report["recommendation_reason"],
                         recommended_target_nfl_id=rec_id)
                frames.append(d)
        future_rows = pd.DataFrame(frames)
        if not future_rows.empty:
            # Derive movement fields from the generated path instead of copying
            # stale speed/direction values from the last observed frame.
            for _, idx in future_rows.groupby("nfl_id").groups.items():
                part = future_rows.loc[idx].sort_values("frame_id")
                anchor = last.loc[part.nfl_id.iloc[0]]
                x = np.r_[float(anchor.x), part.x.to_numpy(dtype=float)]
                y = np.r_[float(anchor.y), part.y.to_numpy(dtype=float)]
                dx, dy = np.diff(x), np.diff(y)
                speed = np.hypot(dx, dy) * 10.0  # 10 Hz tracking
                prior_speed = pd.to_numeric(pd.Series([anchor.get("s", 0)]), errors="coerce").fillna(0).iloc[0]
                acceleration = np.r_[prior_speed, speed]
                acceleration = np.diff(acceleration) * 10.0
                future_rows.loc[part.index, "s"] = speed
                future_rows.loc[part.index, "a"] = acceleration
                future_rows.loc[part.index, "dir"] = np.degrees(np.arctan2(dx, dy)) % 360
            output.append(future_rows)

        # A straight interpolated flight is a visual placeholder; dataset has
        # ball landing coordinates but no measured ball trajectory.
        receiver_end = rec_future.iloc[-1]
        event_specs = [(play_start_frame, "ball_snap", "inferred"),
                       (max_input_frame + 1, "pass_forward", "inferred"),
                       (max_input_frame + max_future_horizon, "pass_arrived", "simulated")]
        for event_frame, event, origin in event_specs:
            d = {c: np.nan for c in tracked_columns}
            d.update(game_id=game_id, play_id=play_id, frame_id=event_frame,
                     record_type="event", event_name=event, data_origin=origin,
                     source_frame_id=max_input_frame, play_start_frame_id=int(pd.to_numeric(observed.frame_id).min()),
                     recommendation_reason=report["recommendation_reason"],
                     recommended_target_nfl_id=rec_id)
            output.append(pd.DataFrame([d]))
        for step in range(1, max_future_horizon + 1):
            t = step / max_future_horizon
            d = {c: np.nan for c in tracked_columns}
            d.update(game_id=game_id, play_id=play_id, frame_id=max_input_frame + step,
                     nfl_id=0, player_name="Football", player_role="Ball", record_type="simulated_ball",
                     event_name="", ball_x=float(passer.x + (receiver_end.pred_x - passer.x) * t),
                     ball_y=float(passer.y + (receiver_end.pred_y - passer.y) * t),
                     data_origin="simulated_straight_line", source_frame_id=max_input_frame,
                     play_start_frame_id=int(pd.to_numeric(observed.frame_id).min()),
                     recommendation_reason=report["recommendation_reason"], recommended_target_nfl_id=rec_id)
            output.append(pd.DataFrame([d]))
    return pd.concat(output, ignore_index=True, sort=False) if output else pd.DataFrame()


def build_suggested_play_payload(
    inputs: pd.DataFrame,
    predictions: pd.DataFrame,
    reports: list[dict],
    game_id: int,
    play_id: int,
) -> dict:
    """Return one suggested play as a JSON-ready object for an integration.

    Call this after ``train`` and ``decisions`` when those results are already
    available. Player records retain the source tracking fields; frame-level
    events and ball coordinates are grouped into a Unity-friendly structure.
    """
    play_report = next((r for r in reports if int(r["game_id"]) == int(game_id)
                        and int(r["play_id"]) == int(play_id)), None)
    if play_report is None:
        raise KeyError(f"No decision report for game_id={game_id}, play_id={play_id}")
    rows = suggested_play_rows(inputs, predictions, [play_report])
    if rows.empty:
        raise ValueError(f"No suggested pass replay for game_id={game_id}, play_id={play_id}")

    def clean(value):
        if value is None:
            return None
        try:
            if pd.isna(value):
                return None
        except (TypeError, ValueError):
            pass
        if hasattr(value, "item"):
            return value.item()
        return value

    frames = []
    for frame_id, group in rows.groupby("frame_id", sort=True, dropna=True):
        player_rows = group[group.record_type.isin(["observed_player", "simulated_player"])]
        ball_rows = group[group.record_type.eq("simulated_ball")]
        event_rows = group[group.record_type.eq("event")]
        players = []
        for record in player_rows.to_dict(orient="records"):
            players.append({key: clean(value) for key, value in record.items()
                            if key not in {"ball_x", "ball_y", "event_name"}})
        ball = None
        if not ball_rows.empty:
            b = ball_rows.iloc[0]
            ball = {"x": clean(b.ball_x), "y": clean(b.ball_y),
                    "nfl_id": clean(b.nfl_id), "data_origin": clean(b.data_origin)}
        events = [{"name": clean(r.event_name), "data_origin": clean(r.data_origin)}
                  for r in event_rows.itertuples(index=False)]
        frames.append({"frame_id": int(frame_id), "players": players, "ball": ball, "events": events})

    target_id = (play_report.get("best_alternative_receiver_id")
                 if play_report["decision"] == "consider_alternative"
                 else play_report.get("evidence", {}).get("targeted_receiver_id"))
    return {
        "schema_version": 1,
        "game_id": int(game_id),
        "play_id": int(play_id),
        "play_start_frame_id": int(pd.to_numeric(rows.play_start_frame_id).min()),
        "recommendation": {
            "decision": play_report["decision"],
            "target_nfl_id": clean(target_id),
            "reason": play_report.get("recommendation_reason"),
            "evidence": play_report.get("evidence", {}),
            "caveat": play_report.get("caveat"),
        },
        "frames": frames,
    }


def running_play_reports(data_dir: Path) -> list[dict]:
    """Use optional full-season nflverse play-by-play data to review rushing plays.

    The Big Data Bowl supplementary table contains pass plays only. To include
    rushing-play reviews, place nflverse's play_by_play_2023.csv under data/.
    """
    paths = list(data_dir.glob("**/play_by_play_2023.csv*"))
    if not paths:
        return []
    pbp = pd.read_csv(paths[0], low_memory=False)
    # Big Data Bowl game_id maps to nflverse old_game_id; support already-mapped CSVs too.
    game_col = "old_game_id" if "old_game_id" in pbp.columns else "game_id"
    if game_col not in pbp or "play_id" not in pbp:
        raise ValueError("play_by_play_2023.csv needs old_game_id (or game_id) and play_id columns")
    pbp["game_id"] = pd.to_numeric(pbp[game_col], errors="coerce")
    pbp["play_id"] = pd.to_numeric(pbp.play_id, errors="coerce")
    if "play_type" in pbp:
        is_run = pbp.play_type.astype(str).str.lower().eq("run")
    elif "rush_attempt" in pbp:
        is_run = pbp.rush_attempt.astype(str).str.lower().isin(["true", "1"])
    else:
        raise ValueError("play-by-play CSV needs a play_type or rush_attempt field")
    runs = pbp[is_run].copy()
    if "yards_gained" not in runs:
        raise ValueError("play-by-play CSV needs yards_gained to review run outcomes")
    runs["yards_gained"] = pd.to_numeric(runs.yards_gained, errors="coerce")
    down_col = "down" if "down" in runs else None
    to_go_col = "ydstogo" if "ydstogo" in runs else ("yards_to_go" if "yards_to_go" in runs else None)
    if down_col:
        runs[down_col] = pd.to_numeric(runs[down_col], errors="coerce")
    if to_go_col:
        runs[to_go_col] = pd.to_numeric(runs[to_go_col], errors="coerce")
    bad = runs.yards_gained.le(0)
    if down_col and to_go_col:
        bad |= runs[down_col].ge(3) & runs[to_go_col].notna() & runs.yards_gained.lt(runs[to_go_col])
    if "fumble_lost" in runs:
        bad |= runs.fumble_lost.astype(str).str.lower().isin(["true", "1"])
    reports = []
    for row in runs[bad & runs.game_id.notna() & runs.play_id.notna()].itertuples(index=False):
        data = row._asdict()
        game_id, play_id = int(data["game_id"]), int(data["play_id"])
        detail = {"yards_gained": float(data["yards_gained"])}
        if down_col and pd.notna(data.get(down_col)):
            detail["down"] = int(data[down_col])
        if to_go_col and pd.notna(data.get(to_go_col)):
            detail["yards_to_go"] = float(data[to_go_col])
        reports.append({
            "game_id": game_id, "play_id": play_id,
            "decision": "suggest_pass_next_time", "status": "unsuccessful_running_play",
            "probabilities": {"run_again": 0.2, "pass_next_time": 0.8}, "confidence": "low",
            "evidence": detail,
            "caveat": "Rule-based post-play suggestion. A single failed run does not establish that passing would have been better."
        })
    return reports


def main():
    parser = argparse.ArgumentParser(description="Train a small NFL tracking model and write typed per-play recommendations.")
    parser.add_argument("--data-dir", type=Path, default=Path("data"), help="Kaggle download root containing train/input_2023_wXX.csv")
    parser.add_argument("--weeks", type=int, nargs="+", default=[1], help="Training weeks to use; start small, e.g. --weeks 1 2")
    parser.add_argument("--max-games", type=int, default=4, help="Use only this many games for a quick prototype")
    parser.add_argument("--game-id", type=int, help="Show only this game in the report (it must be among the selected training games)")
    parser.add_argument("--output", type=Path, default=Path("reports/decisions.jsonl"))
    parser.add_argument("--suggested-play-output", type=Path, default=Path("reports/suggested_plays.csv"),
                        help="CSV containing observed frames plus simulated player, ball, and event records")
    parser.add_argument("--suggested-play-jsonl", type=Path, default=Path("reports/suggested_plays.jsonl"),
                        help="Structured JSONL: one suggested-play object per line")
    args = parser.parse_args()
    inputs, state, _, preds, mae, nrows, ngames = train(args.data_dir, args.weeks, args.max_games, Path("models"))
    pass_reports = decisions(state, preds, args.data_dir)
    run_reports = running_play_reports(args.data_dir)
    if not list(args.data_dir.glob("**/play_by_play_2023.csv*")):
        print("Note: Big Data Bowl files contain pass plays only. Add nflverse play_by_play_2023.csv to data/ to include run-play reviews.")
    reports = pass_reports + run_reports
    selected_game = args.game_id if args.game_id is not None else min(state.game_id.unique())
    reports = [row for row in reports if row["game_id"] == selected_game]
    if not reports:
        available = sorted(int(g) for g in state.game_id.unique())
        raise ValueError(f"No reports for game {selected_game}. Games included in this run: {available}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row) for row in reports) + "\n")
    csv_path = args.output.with_suffix(".csv")
    display = pd.DataFrame([{**{k: v for k, v in row.items() if k != "evidence"}, **row["evidence"]} for row in reports])
    display.to_csv(csv_path, index=False)
    suggested = suggested_play_rows(inputs, preds, reports)
    if not suggested.empty:
        args.suggested_play_output.parent.mkdir(parents=True, exist_ok=True)
        suggested.to_csv(args.suggested_play_output, index=False)
    payloads = []
    for row in reports:
        if row.get("decision") not in {"consider_alternative", "target_was_reasonable"}:
            continue
        payloads.append(build_suggested_play_payload(
            inputs, preds, reports, row["game_id"], row["play_id"]
        ))
    args.suggested_play_jsonl.parent.mkdir(parents=True, exist_ok=True)
    args.suggested_play_jsonl.write_text("\n".join(json.dumps(payload, allow_nan=False) for payload in payloads) + ("\n" if payloads else ""))
    print(json.dumps({"trained_rows": nrows, "games_used": ngames, "heldout_coordinate_mae_yards": mae,
                      "game_shown": int(selected_game), "plays_scored": len(reports),
                      "reports": [str(args.output), str(csv_path), str(args.suggested_play_output),
                                  str(args.suggested_play_jsonl)]}, indent=2))
    columns = [c for c in ("play_id", "decision", "pass_result", "yards_gained", "targeted_receiver_predicted_separation_yards",
                            "best_alternative_predicted_separation_yards", "alternative_separation_advantage_yards")
               if c in display.columns]
    print(display[columns].to_string(index=False))


if __name__ == "__main__":
    main()
