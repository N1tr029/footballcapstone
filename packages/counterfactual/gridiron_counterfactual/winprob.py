"""Expected points to win probability: what a throw was worth to the result of the game.

A point is not always worth the same. Half a point of field position in a tied fourth
quarter moves the game; the same half point down 24 does nothing. The what-if is ranked
on expected points because that is what one throw can be measured in, but it should be
*reported* in win probability, which is what anyone watching cares about.

The release carries the league's own win probability before every play and how much each
play moved it. So rather than build a win-probability model from scratch, this learns the
exchange rate from those 16,124 plays: given how many expected points a play added, and
the game it was added in — the win probability beforehand, the time left, the score — how
far did the win probability move?

What that buys and what it costs. It needs nothing beyond the release, and it inherits the
league model's judgement about game state. But it is a conversion of an estimate, so every
caveat on the expected-points gap carries straight through, and it knows nothing a play's
expected points do not already say — it cannot tell a first down that ends the game from
one that does not, unless the expected points already did.
"""

from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.metrics import mean_absolute_error

from . import bdb2025

STATE = ["wp_before", "seconds_left", "score_diff", "down", "yards_to_go", "yards_to_goal"]
FEATURES = ["epa", *STATE]

TRAIN_WEEKS = range(1, 8)
TEST_WEEKS = range(8, 10)


def situations(plays: pd.DataFrame | None = None, games: pd.DataFrame | None = None) -> pd.DataFrame:
    """Every play's game state, from the side of the team with the ball."""
    plays = bdb2025.plays() if plays is None else plays
    games = bdb2025.games() if games is None else games
    p = plays.merge(games[["gameId", "week", "homeTeamAbbr"]], on="gameId")
    home = p["possessionTeam"] == p["homeTeamAbbr"]
    clock = p["gameClock"].str.split(":", expand=True).astype(float)
    in_quarter = clock[0] * 60 + clock[1]
    own_side = p["yardlineSide"] == p["possessionTeam"]            # NaN side means the 50
    return pd.DataFrame({
        "gameId": p["gameId"], "playId": p["playId"], "week": p["week"],
        "epa": p["expectedPointsAdded"],
        "wpa": np.where(home, p["homeTeamWinProbabilityAdded"], p["visitorTeamWinProbilityAdded"]),
        "wp_before": np.where(home, p["preSnapHomeTeamWinProbability"], p["preSnapVisitorTeamWinProbability"]),
        # Regulation counts down from 3600; overtime is its own ten minutes.
        "seconds_left": np.where(p["quarter"] <= 4, (4 - p["quarter"]) * 900 + in_quarter, in_quarter),
        "score_diff": np.where(home, p["preSnapHomeScore"] - p["preSnapVisitorScore"],
                               p["preSnapVisitorScore"] - p["preSnapHomeScore"]),
        "down": p["down"], "yards_to_go": p["yardsToGo"],
        "yards_to_goal": np.where(own_side, 100 - p["yardlineNumber"], p["yardlineNumber"]),
    })


def train(table: pd.DataFrame) -> HistGradientBoostingRegressor:
    """Win probability added, from expected points added and the state of the game.

    Constrained so that more expected points never means less win probability. Without
    it the conversion of a *gap* could come out negative for a positive gap, which would
    be a model contradicting the thing it is converting.
    """
    table = table.dropna(subset=["epa", "wpa"])
    model = HistGradientBoostingRegressor(
        max_iter=400, learning_rate=0.05, max_depth=5, min_samples_leaf=30, random_state=0,
        monotonic_cst=[1] + [0] * len(STATE))
    return model.fit(table[FEATURES], table["wpa"])


def evaluate(table: pd.DataFrame) -> dict:
    """Weeks 1-7 against 8-9, beside the obvious alternative of one fixed exchange rate."""
    table = table.dropna(subset=["epa", "wpa"])
    tr, te = table[table["week"].isin(TRAIN_WEEKS)], table[table["week"].isin(TEST_WEEKS)]
    pred = train(tr).predict(te[FEATURES])
    rate = float((tr["epa"] * tr["wpa"]).sum() / (tr["epa"] ** 2).sum())
    return {
        "train_plays": len(tr), "test_plays": len(te),
        "fixed_rate_wp_per_point": round(rate, 4),
        "mae_fixed_rate": round(mean_absolute_error(te["wpa"], rate * te["epa"]), 4),
        "mae_model": round(mean_absolute_error(te["wpa"], pred), 4),
        "correlation_fixed_rate": round(float(np.corrcoef(te["wpa"], rate * te["epa"])[0, 1]), 3),
        "correlation_model": round(float(np.corrcoef(te["wpa"], pred)[0, 1]), 3),
    }


def path() -> Path:
    return bdb2025.cache_dir().parent / "models" / "winprob.joblib"


def load_or_train() -> HistGradientBoostingRegressor:
    if path().exists():
        return joblib.load(path())
    model = train(situations())
    path().parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path())
    return model


def added(model: HistGradientBoostingRegressor, epa: np.ndarray, state: pd.DataFrame) -> np.ndarray:
    """Win probability added by a play worth ``epa`` points in each row of ``state``."""
    frame = state[STATE].copy()
    frame.insert(0, "epa", np.asarray(epa, dtype=float))
    return model.predict(frame[FEATURES])
