"""The value model, on throws invented so the right answer is known."""

import numpy as np
import pandas as pd
import pytest

from gridiron_counterfactual.receiver import value
from gridiron_counterfactual.receiver.targets import FEATURES, SITUATION


def _table(n_plays=600, seed=0):
    """Three receivers a play, one targeted. Completion depends only on separation."""
    rng = np.random.default_rng(seed)
    rows = []
    for play in range(n_plays):
        target = rng.integers(3)
        for r in range(3):
            row = {f: rng.normal() for f in FEATURES + SITUATION}
            row.update(gameId=play // 50, week=1 + play * 9 // n_plays, playId=play, nflId=r,
                       separation=rng.uniform(0, 10), lane_defenders=rng.integers(0, 4),
                       down=rng.integers(1, 5), targeted=r == target)
            caught = rng.random() < 0.1 + 0.08 * row["separation"]
            row.update(complete=caught, yards_gained=8.0 if caught else 0.0, epa=1.0 if caught else -0.5)
            rows.append(row)
    return pd.DataFrame(rows)


@pytest.fixture(scope="module")
def fitted():
    table = _table()
    tr, te = value.split(table)
    return value.train(tr), tr, te


def test_weeks_do_not_leak_between_train_and_test(fitted):
    _, tr, te = fitted
    assert set(tr["week"]) <= set(value.TRAIN_WEEKS) and set(te["week"]) <= set(value.TEST_WEEKS)
    assert len(tr) and len(te)


def test_it_learns_that_open_receivers_catch_the_ball(fitted):
    model, tr, te = fitted
    report = value.evaluate(model, tr, te)
    assert report["complete"]["model"]["log_loss"] < report["complete"]["always_the_average"]["log_loss"]
    assert report["complete"]["model"]["auc"] > 0.65


def test_more_separation_never_lowers_the_completion_chance(fitted):
    model, _, te = fitted
    row = te.iloc[[0]].copy()
    sweep = pd.concat([row.assign(separation=s) for s in np.linspace(0, 12, 25)])
    p = model.score(sweep)["p_complete"].to_numpy()
    assert (np.diff(p) >= -1e-9).all()


def test_every_receiver_is_scored_not_just_the_target(fitted):
    model, _, te = fitted
    scored = model.score(te)
    assert len(scored) == len(te)
    assert scored.loc[~scored["targeted"], ["p_complete", "exp_yards_if_caught", "exp_epa"]].notna().all().all()
