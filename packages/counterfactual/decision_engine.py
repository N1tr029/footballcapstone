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
    return state, outputs, predicted, score, len(fit_rows), len(games)


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
        if not supplementary.empty:
            context = supplementary[(supplementary.game_id == game_id) & (supplementary.play_id == play_id)]
            if not context.empty:
                play_context = context.iloc[0]
                for col in ("play_description", "pass_result", "yards_gained", "down", "yards_to_go", "expected_points_added"):
                    if col in context.columns and pd.notna(play_context[col]):
                        report[col] = play_context[col].item() if hasattr(play_context[col], "item") else play_context[col]
        reports.append(report)
    return reports


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
    args = parser.parse_args()
    state, _, preds, mae, nrows, ngames = train(args.data_dir, args.weeks, args.max_games, Path("models"))
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
    print(json.dumps({"trained_rows": nrows, "games_used": ngames, "heldout_coordinate_mae_yards": mae,
                      "game_shown": int(selected_game), "plays_scored": len(reports),
                      "reports": [str(args.output), str(csv_path)]}, indent=2))
    columns = [c for c in ("play_id", "decision", "pass_result", "yards_gained", "targeted_receiver_predicted_separation_yards",
                            "best_alternative_predicted_separation_yards", "alternative_separation_advantage_yards")
               if c in display.columns]
    print(display[columns].to_string(index=False))


if __name__ == "__main__":
    main()
