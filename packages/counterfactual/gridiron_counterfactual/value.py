"""What a throw to this receiver was worth, before anyone knew how it turned out.

Three models over the receiver-at-throw table, all trained only on the receiver who was
actually targeted, because he is the only one with an outcome:

``complete``   the chance the pass is caught
``yards``      yards gained if it is (air yards plus the run after)
``epa``        expected points added by the throw, complete or not

``epa`` is the one a what-if is ranked on. A completion is not the goal — five yards on
third and eight is a completion and a failure — so the value of a throw has to know the
down, the distance and the field position, and ``complete`` deliberately does not.

Weeks 1-7 train, weeks 8-9 test. Splitting by week rather than at random keeps a game
entirely on one side, so a model cannot score well by recognising a matchup.

**The limit, stated where it is easy to find.** These models learn from throws that were
made. A quarterback does not throw to a man he can see is covered, so the covered men in
the training set are the ones he judged worth it anyway, and the model is therefore too
generous to covered receivers nobody threw to. The monotonic constraints below are a
partial guard — more separation can never *lower* a completion chance — and
:func:`evaluate` reports how far outside the training range the untargeted receivers sit.
It is an estimate against similar throws, and anything built on it has to say so.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import brier_score_loss, log_loss, mean_absolute_error, roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from . import bdb2025
from .targets import FEATURES, SITUATION

TRAIN_WEEKS = range(1, 8)
TEST_WEEKS = range(8, 10)

COMPLETE_FEATURES = FEATURES
VALUE_FEATURES = FEATURES + SITUATION

MONOTONIC = {"separation": 1, "lane_defenders": -1}
"""More open never hurts; another defender in the throwing lane never helps. Without
these a tree is free to learn a dip from a handful of odd throws, and the dip is exactly
where an untargeted receiver would be scored."""

_GBM = dict(max_iter=300, learning_rate=0.05, max_depth=4, min_samples_leaf=40,
            l2_regularization=1.0, early_stopping=True, validation_fraction=0.15, random_state=0)


@dataclass
class ValueModel:
    complete: HistGradientBoostingClassifier
    yards: HistGradientBoostingRegressor
    epa: HistGradientBoostingRegressor

    def score(self, table: pd.DataFrame) -> pd.DataFrame:
        """Every receiver in ``table`` (targeted or not), with the three predictions."""
        out = table.copy()
        out["p_complete"] = self.complete.predict_proba(table[COMPLETE_FEATURES])[:, 1]
        out["exp_yards_if_caught"] = self.yards.predict(table[VALUE_FEATURES])
        out["exp_epa"] = self.epa.predict(table[VALUE_FEATURES])
        return out

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        joblib.dump(self, path)

    @staticmethod
    def load(path: Path) -> "ValueModel":
        return joblib.load(path)


def _cst(features: list[str]) -> list[int]:
    return [MONOTONIC.get(f, 0) for f in features]


def train(table: pd.DataFrame) -> ValueModel:
    """Fit all three on the targeted rows of ``table``."""
    thrown = table[table["targeted"]]
    caught = thrown[thrown["complete"]]
    complete = HistGradientBoostingClassifier(monotonic_cst=_cst(COMPLETE_FEATURES), **_GBM)
    complete.fit(thrown[COMPLETE_FEATURES], thrown["complete"])
    yards = HistGradientBoostingRegressor(**_GBM).fit(caught[VALUE_FEATURES], caught["yards_gained"])
    epa = HistGradientBoostingRegressor(monotonic_cst=_cst(VALUE_FEATURES), **_GBM)
    epa.fit(thrown[VALUE_FEATURES], thrown["epa"])
    return ValueModel(complete, yards, epa)


def split(table: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    return table[table["week"].isin(TRAIN_WEEKS)], table[table["week"].isin(TEST_WEEKS)]


def _calibration(y: np.ndarray, p: np.ndarray, bins: int = 5) -> list[dict]:
    """Predicted against actual in equal-sized groups. The check that matters: a model
    that says 80% should be right about 80% of the time, whatever its accuracy."""
    order = np.argsort(p)
    return [{"predicted": round(float(p[i].mean()), 3), "actual": round(float(y[i].mean()), 3), "n": len(i)}
            for i in np.array_split(order, bins)]


def evaluate(model: ValueModel, train_table: pd.DataFrame, test_table: pd.DataFrame) -> dict:
    """How good it is on throws it has not seen, against two simpler things to beat."""
    tr, te = train_table[train_table["targeted"]], test_table[test_table["targeted"]]
    y = te["complete"].to_numpy(dtype=int)

    p = model.complete.predict_proba(te[COMPLETE_FEATURES])[:, 1]
    base = np.full(len(te), tr["complete"].mean())
    simple = make_pipeline(StandardScaler(), LogisticRegression()).fit(
        tr[["separation", "depth"]], tr["complete"])
    p_simple = simple.predict_proba(te[["separation", "depth"]])[:, 1]

    def scores(pred):
        return {"log_loss": round(log_loss(y, pred), 4), "brier": round(brier_score_loss(y, pred), 4),
                "auc": round(roc_auc_score(y, pred), 4) if pred.std() > 0 else 0.5}

    caught = te[te["complete"]]
    epa_pred = model.epa.predict(te[VALUE_FEATURES])

    # How far outside what the model was trained on are the receivers nobody threw to?
    scored = model.score(test_table)
    lo = tr["separation"].quantile(0.05)
    untargeted = scored[~scored["targeted"]]

    return {
        "train_throws": len(tr), "test_throws": len(te),
        "complete": {"always_the_average": scores(base), "separation_and_depth_only": scores(p_simple),
                     "model": scores(p), "calibration": _calibration(y, p)},
        "yards_if_caught": {
            "mae_always_the_average": round(mean_absolute_error(
                caught["yards_gained"], np.full(len(caught), tr.loc[tr["complete"], "yards_gained"].mean())), 2),
            "mae_model": round(mean_absolute_error(
                caught["yards_gained"], model.yards.predict(caught[VALUE_FEATURES])), 2)},
        "epa": {
            "mae_always_the_average": round(mean_absolute_error(te["epa"], np.full(len(te), tr["epa"].mean())), 3),
            "mae_model": round(mean_absolute_error(te["epa"], epa_pred), 3),
            "correlation": round(float(np.corrcoef(te["epa"], epa_pred)[0, 1]), 3),
            "calibration": [{"predicted": round(float(epa_pred[i].mean()), 3),
                             "actual": round(float(te["epa"].to_numpy()[i].mean()), 3), "n": len(i)}
                            for i in np.array_split(np.argsort(epa_pred), 5)]},
        "untargeted": {
            "receivers": len(untargeted),
            "mean_p_complete": round(float(untargeted["p_complete"].mean()), 3),
            "mean_p_complete_targeted": round(float(scored.loc[scored["targeted"], "p_complete"].mean()), 3),
            "share_tighter_than_95pct_of_training_throws": round(float((untargeted["separation"] < lo).mean()), 3),
        },
        "choice": _choice(scored),
    }


def _choice(scored: pd.DataFrame) -> dict:
    """On each throw, where did the receiver he chose rank among the options?"""
    g = scored.groupby(bdb2025.KEY)
    best = g["exp_epa"].transform("max")
    chosen = scored[scored["targeted"]].assign(gap=lambda d: best[d.index] - d["exp_epa"])
    rank = g["exp_epa"].rank(ascending=False, method="min")[chosen.index]
    return {
        "threw_to_the_models_best_option": round(float((rank == 1).mean()), 3),
        "threw_to_a_top_two_option": round(float((rank <= 2).mean()), 3),
        "median_epa_left_on_the_field": round(float(chosen["gap"].median()), 3),
        "throws_with_over_half_a_point_available": round(float((chosen["gap"] > 0.5).mean()), 3),
    }


def model_path() -> Path:
    return bdb2025.cache_dir().parent / "models" / "receiver_value.joblib"


def train_and_report(table: pd.DataFrame, log=print) -> dict:
    """Fit on weeks 1-7, report on 8-9, then refit on everything and save that."""
    tr, te = split(table)
    report = evaluate(train(tr), tr, te)
    train(table).save(model_path())
    model_path().with_suffix(".report.json").write_text(json.dumps(report, indent=2))
    log(json.dumps(report, indent=2))
    return report
