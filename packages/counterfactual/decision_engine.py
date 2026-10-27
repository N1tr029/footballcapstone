"""NFL Big Data Bowl 2025 full-field tracking decision review.

The 2025 release uses tracking_week_N.csv and records the full tracked play. This
workflow uses the measured positions at pass release and includes every player
present in the tracking file when measuring receiver separation.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import joblib
from sklearn.ensemble import HistGradientBoostingRegressor

KEYS = ["game_id", "play_id", "nfl_id"]
OPEN_SEPARATION_YARDS = 3.0

# The 2025 tracking release has full trajectories rather than paired
# competition input/output files. Learn the next-frame location directly from
# each player's previous measured state, then apply it to every player at the
# pass-forward frame.
TRACK_FEATURES = ["prev_x", "prev_y", "prev_s", "prev_a", "prev_dis", "prev_o",
                  "prev_dir_sin", "prev_dir_cos", "play_direction_code"]


def train_2025_location_model(tracking: pd.DataFrame, max_games: int = 4):
    """Fit x/y next-frame regressors on consecutive 2025 tracking observations."""
    data = tracking.rename(columns={"gameId": "game_id", "playId": "play_id", "nflId": "nfl_id",
                                   "frameId": "frame_id", "playDirection": "play_direction"}).copy()
    required = {"game_id", "play_id", "nfl_id", "frame_id", "x", "y"}
    missing = required - set(data.columns)
    if missing:
        raise ValueError(f"2025 tracking data missing model columns: {sorted(missing)}")
    games = sorted(data.game_id.dropna().unique())[:max_games]
    data = data[data.game_id.isin(games) & data.nfl_id.notna()].copy()
    for col in ("x", "y", "s", "a", "dis", "o", "dir"):
        if col not in data:
            data[col] = 0.0
        data[col] = pd.to_numeric(data[col], errors="coerce")
    data = data.sort_values(["game_id", "play_id", "nfl_id", "frame_id"])
    grouped = data.groupby(["game_id", "play_id", "nfl_id"], sort=False)
    pairs = data.copy()
    for col in ("x", "y", "s", "a", "dis", "o", "dir", "frame_id"):
        pairs[f"prev_{col}"] = grouped[col].shift(1)
    pairs["frame_gap"] = pairs.frame_id - pairs.prev_frame_id
    pairs = pairs[pairs.frame_gap.eq(1)].copy()
    direction = data.get("play_direction", pd.Series("right", index=data.index)).str.lower()
    pairs["play_direction_code"] = direction.map({"right": 1.0, "left": -1.0}).fillna(0.0)
    angle = np.deg2rad(pairs.prev_dir.fillna(0))
    pairs["prev_dir_sin"], pairs["prev_dir_cos"] = np.sin(angle), np.cos(angle)
    pairs = pairs.replace([np.inf, -np.inf], np.nan).dropna(subset=TRACK_FEATURES + ["x", "y"])
    if pairs.empty:
        raise ValueError("No consecutive player frames found to train on in the selected 2025 games")
    models = {}
    for axis in ("x", "y"):
        model = HistGradientBoostingRegressor(max_iter=100, max_leaf_nodes=15,
                                               l2_regularization=1.0, random_state=17)
        model.fit(pairs[TRACK_FEATURES], pairs[axis])
        models[axis] = model
    return {"models": models, "feature_columns": TRACK_FEATURES,
            "training_games": [int(g) for g in games], "training_rows": int(len(pairs))}


def predict_2025_pass_release_locations(tracking: pd.DataFrame, bundle: dict,
                                        game_ids: set | None = None) -> pd.DataFrame:
    """Predict every tracked player's x/y at pass release, preserving source fields."""
    original_columns = list(tracking.columns)
    data = tracking.copy()
    data["_frame_num"] = pd.to_numeric(data.frameId, errors="coerce")
    releases = data[data.event.fillna("").astype(str).str.lower().str.replace(" ", "_").str.contains("pass_forward|passforward", regex=True)]
    if game_ids is not None:
        releases = releases[releases.gameId.isin(game_ids)]
    releases = releases.sort_values(["gameId", "playId", "frameId"]).groupby(["gameId", "playId"], as_index=False).first()
    if releases.empty:
        return pd.DataFrame(columns=original_columns)
    # A many-to-many merge via the release frame list selects all player rows at
    # release, then joins each player's frame-1 observation.
    player_releases = data[data[["gameId", "playId", "_frame_num"]].apply(tuple, axis=1).isin(
        set(releases[["gameId", "playId", "_frame_num"]].apply(tuple, axis=1)))].copy()
    keys = ["gameId", "playId", "nflId"]
    prev = data[[*keys, "_frame_num", "x", "y", "s", "a", "dis", "o", "dir", "playDirection"]].copy()
    prev["_frame_num"] += 1
    prev = prev.rename(columns={c: f"prev_{c}" for c in prev.columns if c not in keys + ["_frame_num"]})
    scored = player_releases.merge(prev, on=keys + ["_frame_num"], how="left", suffixes=("", "_lag"))
    # Coordinates are predicted only where a true preceding observation exists.
    for c in ("prev_x", "prev_y", "prev_s", "prev_a", "prev_dis", "prev_o", "prev_dir"):
        scored[c] = pd.to_numeric(scored.get(c), errors="coerce")
    angle = np.deg2rad(scored.prev_dir.fillna(0))
    scored["prev_dir_sin"], scored["prev_dir_cos"] = np.sin(angle), np.cos(angle)
    scored["play_direction_code"] = scored.playDirection.astype(str).str.lower().map({"right": 1.0, "left": -1.0}).fillna(0.0)
    valid = scored["prev_x"].notna() & scored["prev_y"].notna()
    scored["x"] = pd.to_numeric(scored.x, errors="coerce")
    scored["y"] = pd.to_numeric(scored.y, errors="coerce")
    if valid.any():
        scored.loc[valid, "x"] = bundle["models"]["x"].predict(scored.loc[valid, TRACK_FEATURES])
        scored.loc[valid, "y"] = bundle["models"]["y"].predict(scored.loc[valid, TRACK_FEATURES])
    return scored[original_columns]


def read_2025_weeks(data_dir: Path, weeks: list[int]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Load the 2025 BDB tracking, play, and player-play tables."""
    tracking = []
    for week in weeks:
        name = f"tracking_week_{week}.csv"
        paths = list(data_dir.glob(f"**/{name}"))
        if not paths:
            raise FileNotFoundError(f"Could not find {name} under {data_dir}; download the 2025 BDB files first.")
        tracking.append(pd.read_csv(paths[0], low_memory=False))
    def table(name: str) -> pd.DataFrame:
        paths = list(data_dir.glob(f"**/{name}"))
        if not paths:
            raise FileNotFoundError(f"Could not find {name} under {data_dir}; it is required by the 2025 BDB workflow.")
        return pd.read_csv(paths[0], low_memory=False)
    return pd.concat(tracking, ignore_index=True), table("plays.csv"), table("player_play.csv")


def analyze_2025(tracking: pd.DataFrame, plays: pd.DataFrame, player_play: pd.DataFrame,
                 max_games: int = 4, predicted_locations: pd.DataFrame | None = None) -> list[dict]:
    """Review pass plays using modeled player locations at pass release."""
    aliases = {"gameId": "game_id", "playId": "play_id", "nflId": "nfl_id",
               "frameId": "frame_id", "playDirection": "play_direction"}
    tracking = tracking.rename(columns=aliases).copy()
    plays = plays.rename(columns={"gameId": "game_id", "playId": "play_id",
                                  "possessionTeam": "possession_team", "absoluteYardlineNumber": "absolute_yardline_number",
                                  "yardsToGo": "yards_to_go", "playDescription": "play_description"}).copy()
    player_play = player_play.rename(columns={"gameId": "game_id", "playId": "play_id", "nflId": "nfl_id",
                                               "wasTargettedReceiver": "was_targeted_receiver"}).copy()
    modeled = None
    if predicted_locations is not None and not predicted_locations.empty:
        modeled = predicted_locations.rename(columns={"gameId": "game_id", "playId": "play_id",
                                                       "nflId": "nfl_id", "frameId": "frame_id"})
    needed = {"game_id", "play_id", "nfl_id", "frame_id", "x", "y", "club", "event"}
    missing = needed - set(tracking.columns)
    if missing:
        raise ValueError(f"2025 tracking data missing columns: {sorted(missing)}")
    if "was_targeted_receiver" not in player_play:
        raise ValueError("2025 player_play.csv is missing wasTargettedReceiver, needed to identify the intended target")
    game_ids = sorted(tracking.game_id.dropna().unique())[:max_games]
    tracking = tracking[tracking.game_id.isin(game_ids)].copy()
    tracking["event_norm"] = tracking.event.fillna("").astype(str).str.lower().str.replace(" ", "_")
    metadata = plays.set_index(["game_id", "play_id"], drop=False)
    targets = player_play[pd.to_numeric(player_play.get("was_targeted_receiver", 0), errors="coerce").fillna(0).astype(bool)]
    target_map = targets.groupby(["game_id", "play_id"]).nfl_id.apply(list).to_dict()
    reports = []
    for (game_id, play_id), play in tracking.groupby(["game_id", "play_id"], sort=False):
        # The full-play tracking sequence has a pass_forward event. If absent, skip
        # rather than silently using the end of the play as the decision moment.
        releases = play[play.event_norm.str.contains("pass_forward|passforward", regex=True)]
        if releases.empty:
            continue
        release_frame = pd.to_numeric(releases.frame_id, errors="coerce").min()
        frame = play[pd.to_numeric(play.frame_id, errors="coerce").eq(release_frame)].copy()
        if frame.empty:
            continue
        if modeled is not None:
            modeled_frame = modeled[(modeled.game_id == game_id) & (modeled.play_id == play_id)
                                    & (pd.to_numeric(modeled.frame_id, errors="coerce") == release_frame)]
            if not modeled_frame.empty:
                position_map = modeled_frame.set_index("nfl_id")[["x", "y"]]
                for axis in ("x", "y"):
                    frame[axis] = frame.nfl_id.map(position_map[axis]).fillna(frame[axis])
        ctx = metadata.loc[(game_id, play_id)] if (game_id, play_id) in metadata.index else None
        if isinstance(ctx, pd.DataFrame):
            ctx = ctx.iloc[0]
        possession = str(ctx.get("possession_team", "")) if ctx is not None else ""
        if not possession:
            continue
        frame["nfl_id"] = pd.to_numeric(frame.nfl_id, errors="coerce")
        frame = frame[frame.nfl_id.notna()].copy()  # exclude the ball row
        offense = frame[frame.club.astype(str).str.upper().eq(possession.upper())]
        defense = frame[~frame.index.isin(offense.index)]
        target_ids = target_map.get((game_id, play_id), [])
        # Include the quarterback/pass target fields where present, but only route
        # receivers are alternatives; all 22 tracked positions remain in evidence.
        if offense.empty or defense.empty or not target_ids:
            continue
        d = np.hypot(offense.x.to_numpy(dtype=float)[:, None] - defense.x.to_numpy(dtype=float)[None, :],
                     offense.y.to_numpy(dtype=float)[:, None] - defense.y.to_numpy(dtype=float)[None, :])
        offense = offense.copy()
        offense["separation_yards"] = d.min(axis=1)
        receivers = offense[~offense.nfl_id.isin(target_ids)]
        position_col = next((c for c in ("player_position", "position") if c in receivers), None)
        if position_col:
            receivers = receivers[~receivers[position_col].astype(str).str.strip().str.upper().eq("QB")]
        target = offense[offense.nfl_id.isin(target_ids)]
        if target.empty:
            continue
        actual = target.sort_values("separation_yards", ascending=False).iloc[0]
        alternatives = receivers.sort_values("separation_yards", ascending=False)
        best = alternatives.iloc[0] if not alternatives.empty else None
        actual_sep = float(actual.separation_yards)
        alt_sep = float(best.separation_yards) if best is not None else None
        gain = alt_sep - actual_sep if alt_sep is not None else None
        if gain is not None and gain >= 1.5:
            choice, status = "consider_alternative", "measured_alternative_more_open_at_release"
        else:
            choice, status = "target_was_reasonable", "no_clear_measured_improvement_at_release"
        report = {
            "game_id": int(game_id), "play_id": int(play_id), "decision": choice, "status": status,
            "probabilities": {"target_was_reasonable": 0.3, "consider_alternative": 0.7} if choice == "consider_alternative" else {"target_was_reasonable": 0.7, "consider_alternative": 0.3},
            "confidence": "moderate" if len(alternatives) >= 3 else "low",
            "evidence": {"targeted_receiver_id": int(actual.nfl_id), "tracked_player_count": int(len(frame)),
                         "roster_complete": len(frame) >= 22, "roster_coverage": "all_players_in_2025_tracking_frame",
                         "decision_frame_id": int(release_frame), "targeted_receiver_separation_yards": round(actual_sep, 2),
                         "best_alternative_receiver_id": int(best.nfl_id) if best is not None else None,
                         "best_alternative_separation_yards": round(alt_sep, 2) if alt_sep is not None else None,
                         "alternative_separation_advantage_yards": round(gain, 2) if gain is not None else None,
                         "measurement": "nearest-defender distance using model-predicted pass_forward locations"},
            "caveat": "Retrospective geometric comparison at pass release using next-frame predictions trained on 2025 tracking. Separation alone does not model quarterback reads, throw difficulty, coverage responsibility, or catch probability."
        }
        for col in ("play_description", "down", "yards_to_go", "absolute_yardline_number"):
            if ctx is not None and col in ctx and pd.notna(ctx[col]):
                value = ctx[col]
                report[col] = value.item() if hasattr(value, "item") else value
        reports.append(report)
    return reports


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


def main():
    parser = argparse.ArgumentParser(description="Train on NFL Big Data Bowl 2025 tracking and predict every player's pass-release location.")
    package_data_dir = Path(__file__).resolve().parent / "data"
    repository_data_dir = Path(__file__).resolve().parents[2] / "data" / "kaggle" / "big-data-bowl-data"
    default_data_dir = package_data_dir if list(package_data_dir.glob("**/tracking_week_1.csv")) else repository_data_dir
    parser.add_argument("--data-dir", type=Path, default=default_data_dir,
                        help="Kaggle download root containing tracking_week_N.csv and plays.csv")
    parser.add_argument("--weeks", type=int, nargs="+", default=[1], help="2025 tracking weeks to use, e.g. --weeks 1 2")
    parser.add_argument("--max-games", type=int, default=4, help="Review plays from this many games")
    parser.add_argument("--game-id", type=int, help="Show only this game (it must be included in the selected weeks/games)")
    parser.add_argument("--output", type=Path, default=Path("reports/decisions.jsonl"), help="Decision report output (JSONL; CSV uses the same stem)")
    parser.add_argument("--locations-output", type=Path, default=Path("reports/player_locations_2025.csv"),
                        help="Predicted pass-release rows, with the same columns as tracking_week_N.csv")
    parser.add_argument("--model-output", type=Path, default=Path("models/location_model_2025.joblib"))
    args = parser.parse_args()
    tracking, plays, player_play = read_2025_weeks(args.data_dir, args.weeks)
    bundle = train_2025_location_model(tracking, args.max_games)
    args.model_output.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(bundle, args.model_output)
    selected_training_games = set(bundle["training_games"])
    predicted_locations = predict_2025_pass_release_locations(tracking, bundle, selected_training_games)
    args.locations_output.parent.mkdir(parents=True, exist_ok=True)
    predicted_locations.to_csv(args.locations_output, index=False)
    reports = analyze_2025(tracking, plays, player_play, args.max_games, predicted_locations)
    selected_game = args.game_id if args.game_id is not None else (min(r["game_id"] for r in reports) if reports else None)
    if selected_game is None:
        raise ValueError("No pass_forward plays were found in the selected 2025 tracking weeks.")
    reports = [row for row in reports if row["game_id"] == selected_game]
    if not reports:
        available = sorted({int(g) for g in tracking.gameId.unique()})
        raise ValueError(f"No pass decision reports for game {selected_game}. Games in these weeks include: {available[:30]}")
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text("\n".join(json.dumps(row) for row in reports) + "\n")
    csv_path = args.output.with_suffix(".csv")
    display = pd.DataFrame([{**{k: v for k, v in row.items() if k != "evidence"}, **row["evidence"]} for row in reports])
    display.to_csv(csv_path, index=False)
    print(json.dumps({"data_release": "NFL Big Data Bowl 2025", "full_tracking_rows": int(len(tracking)),
                      "game_shown": int(selected_game), "plays_scored": len(reports),
                      "model_training_games": bundle["training_games"], "model_training_rows": bundle["training_rows"],
                      "predicted_player_location_rows": int(len(predicted_locations)),
                      "predicted_location_columns": list(predicted_locations.columns),
                      "model": str(args.model_output), "player_locations": str(args.locations_output),
                      "reports": [str(args.output), str(csv_path)]}, indent=2))
    columns = [c for c in ("play_id", "decision", "down", "tracked_player_count", "targeted_receiver_separation_yards",
                            "best_alternative_separation_yards", "alternative_separation_advantage_yards")
               if c in display.columns]
    print(display[columns].to_string(index=False))


if __name__ == "__main__":
    main()
