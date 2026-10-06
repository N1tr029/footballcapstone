"""Points to win probability, on games invented so the exchange rate is known."""

import numpy as np
import pandas as pd
import pytest

from gridiron_counterfactual import winprob


def _plays_and_games():
    plays = pd.DataFrame([
        # home team (KC) has the ball, up 7, on its own 30, 10:00 left in the 4th
        dict(gameId=1, playId=1, possessionTeam="KC", quarter=4, gameClock="10:00", down=1, yardsToGo=10,
             yardlineSide="KC", yardlineNumber=30, preSnapHomeScore=14, preSnapVisitorScore=7,
             preSnapHomeTeamWinProbability=0.8, preSnapVisitorTeamWinProbability=0.2,
             homeTeamWinProbabilityAdded=0.03, visitorTeamWinProbilityAdded=-0.03, expectedPointsAdded=1.0),
        # visitors (LV) have the ball on KC's 20 in the 1st
        dict(gameId=1, playId=2, possessionTeam="LV", quarter=1, gameClock="05:30", down=3, yardsToGo=4,
             yardlineSide="KC", yardlineNumber=20, preSnapHomeScore=0, preSnapVisitorScore=3,
             preSnapHomeTeamWinProbability=0.45, preSnapVisitorTeamWinProbability=0.55,
             homeTeamWinProbabilityAdded=0.02, visitorTeamWinProbilityAdded=-0.02, expectedPointsAdded=-0.7),
    ])
    games = pd.DataFrame([dict(gameId=1, week=1, homeTeamAbbr="KC")])
    return plays, games


def test_state_is_from_the_side_of_the_team_with_the_ball():
    s = winprob.situations(*_plays_and_games()).set_index("playId")
    assert (s.loc[1, "wp_before"], s.loc[1, "wpa"], s.loc[1, "score_diff"]) == (0.8, 0.03, 7)
    assert (s.loc[2, "wp_before"], s.loc[2, "wpa"], s.loc[2, "score_diff"]) == (0.55, -0.02, 3)
    assert s.loc[1, "seconds_left"] == 600 and s.loc[2, "seconds_left"] == 3 * 900 + 330
    assert s.loc[1, "yards_to_goal"] == 70 and s.loc[2, "yards_to_goal"] == 20


def _table(n=3000, seed=0):
    """A point is worth 2% early and 8% in the last five minutes."""
    rng = np.random.default_rng(seed)
    t = pd.DataFrame({
        "epa": rng.normal(0, 1.5, n), "wp_before": rng.uniform(0.1, 0.9, n),
        "seconds_left": rng.uniform(0, 3600, n), "score_diff": rng.integers(-14, 15, n),
        "down": rng.integers(1, 5, n), "yards_to_go": rng.integers(1, 15, n),
        "yards_to_goal": rng.integers(1, 99, n), "week": rng.integers(1, 10, n)})
    t["wpa"] = t["epa"] * np.where(t["seconds_left"] < 300, 0.08, 0.02)
    return t


def test_a_point_is_worth_more_late_in_a_game():
    model = winprob.train(_table())
    state = pd.DataFrame({"wp_before": [0.5, 0.5], "seconds_left": [2000, 100], "score_diff": [0, 0],
                          "down": [1, 1], "yards_to_go": [10, 10], "yards_to_goal": [50, 50]})
    early, late = winprob.added(model, [1.0, 1.0], state)
    assert early == pytest.approx(0.02, abs=0.01) and late == pytest.approx(0.08, abs=0.02)


def test_more_points_never_means_less_win_probability():
    model = winprob.train(_table())
    state = pd.DataFrame({"wp_before": [0.5] * 30, "seconds_left": [900] * 30, "score_diff": [0] * 30,
                          "down": [2] * 30, "yards_to_go": [7] * 30, "yards_to_goal": [40] * 30})
    assert (np.diff(winprob.added(model, np.linspace(-4, 4, 30), state)) >= -1e-9).all()


def test_the_model_beats_one_fixed_rate_when_the_rate_really_varies():
    report = winprob.evaluate(_table())
    assert report["mae_model"] < report["mae_fixed_rate"]
