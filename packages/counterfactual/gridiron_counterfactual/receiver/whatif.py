"""The what-if: on this throw, was there a better receiver, and how sure are we?

The value model scores a receiver. Scoring every receiver and calling the highest one
"the right read" does not work — it prefers men running deep with their backs turned,
because nobody throws to them and so it never learned what happens when somebody does.
Three things stand between a score and a claim, and this module is those three things.

**Availability.** An alternative only counts if he looks like a receiver quarterbacks
actually throw to. A second model learns that directly: the chance a receiver in this
position gets targeted at all. An alternative below the floor — set at the bottom tenth
of real targets — is one the value model has no evidence about, and is left out.

**Uncertainty.** The value model is refit on resampled throws, several times. Each refit
gives its own answer for the gap between the alternative and the man who was thrown to.
A play is flagged only when nearly all of them agree there was a gap, and the typical gap
is big enough to matter. Picking the best of several noisy scores overstates the winner;
asking the low end of the range to clear zero is what pays that back.

**No hindsight.** A model that has seen how a throw turned out will mark the incompletions
as mistakes. Every play is scored by models fitted on *other games* — cross-fitting — so
nothing here knows what happened on the play it is judging. That is also what makes
:func:`check` a fair test: if flagged throws really were the worse choice, they should
have turned out worse than the rest, and the model was never told that they did.

What this cannot know: the play call. A receiver can be open and not in the progression,
and a quarterback can be right to leave him. These are *likely* missed options.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, HistGradientBoostingRegressor
from sklearn.model_selection import GroupKFold

from . import bdb2025, winprob
from .targets import FEATURES
from .value import COMPLETE_FEATURES, VALUE_FEATURES, _GBM, _cst

FOLDS = 5
REFITS = 12
"""Resampled fits per fold. Enough that the tenth percentile of the gap is a stable number."""

AVAILABLE_FLOOR = 0.10
"""An alternative must be at least as targetable as the bottom tenth of real targets."""

GAP_LOW_Q = 0.10
MIN_GAP = 0.80
"""Expected points. Deliberately high: taking the best of several noisy alternatives
overstates the winner, so a small gap is mostly that. At 0.8 about 9% of throws are
flagged; at 0.4 it is 20%, which is more "mistakes" than anyone watching would agree to."""

MAX_TIME_TO_THROW = 5.0
"""Seconds from the snap. Past this the play has broken down and the throw is as likely a
throwaway as a read. From review: both flagged throws later than this were judged
throwaways and none of the eight earlier ones were; across the season they are 2.5% of
throws, completed 42% of the time against 67%."""

ASSUMPTIONS = [
    "Value is expected points added, estimated from similar throws in the 2022 season, weeks 1-9.",
    "Receivers are compared at the instant of the throw; nothing later in the play is used.",
    "The play call and the quarterback's progression are unknown; an open receiver may not have been a read.",
    "An alternative is only considered if receivers in his position do get thrown to.",
    "Scored by models fitted on other games, so the outcome of this play was never seen.",
    "Throws more than 5 seconds after the snap are never flagged: by then the play has broken down and the throw may be a throwaway.",
    "Win probability is converted from expected points using the league's own win probability for this game state.",
]


@dataclass
class Fitted:
    available: HistGradientBoostingClassifier
    floor: float
    complete: HistGradientBoostingClassifier
    epa: list[HistGradientBoostingRegressor]


def fit(table: pd.DataFrame, refits: int = REFITS, seed: int = 0) -> Fitted:
    """Fit availability, completion, and ``refits`` resampled value models on ``table``."""
    thrown = table[table["targeted"]]
    available = HistGradientBoostingClassifier(**_GBM).fit(table[FEATURES], table["targeted"])
    floor = float(np.quantile(available.predict_proba(thrown[FEATURES])[:, 1], AVAILABLE_FLOOR))
    complete = HistGradientBoostingClassifier(monotonic_cst=_cst(COMPLETE_FEATURES), **_GBM)
    complete.fit(thrown[COMPLETE_FEATURES], thrown["complete"])

    rng = np.random.default_rng(seed)
    epa = []
    for _ in range(refits):
        sample = thrown.iloc[rng.integers(0, len(thrown), len(thrown))]
        m = HistGradientBoostingRegressor(monotonic_cst=_cst(VALUE_FEATURES), **_GBM)
        epa.append(m.fit(sample[VALUE_FEATURES], sample["epa"]))
    return Fitted(available, floor, complete, epa)


def score(fitted: Fitted, table: pd.DataFrame) -> pd.DataFrame:
    """Every receiver, with availability, completion chance, and one EPA per refit."""
    out = table.copy()
    out["p_targeted"] = fitted.available.predict_proba(table[FEATURES])[:, 1]
    out["available"] = out["targeted"] | (out["p_targeted"] >= fitted.floor)
    out["p_complete"] = fitted.complete.predict_proba(table[COMPLETE_FEATURES])[:, 1]
    draws = np.column_stack([m.predict(table[VALUE_FEATURES]) for m in fitted.epa])
    for k in range(draws.shape[1]):
        out[f"epa_{k}"] = draws[:, k]
    out["exp_epa"] = draws.mean(axis=1)
    return out


def cross_score(table: pd.DataFrame, folds: int = FOLDS, refits: int = REFITS, log=None) -> pd.DataFrame:
    """Score every play with models that never saw its game."""
    parts = []
    splitter = GroupKFold(n_splits=folds)
    for k, (tr, te) in enumerate(splitter.split(table, groups=table["gameId"])):
        parts.append(score(fit(table.iloc[tr], refits=refits, seed=k), table.iloc[te]))
        if log:
            log(f"fold {k + 1}/{folds}: scored {table.iloc[te].groupby(bdb2025.KEY).ngroups} throws")
    return pd.concat(parts).sort_index()


def decide(scored: pd.DataFrame, min_gap: float = MIN_GAP, wp_model=None) -> pd.DataFrame:
    """One row per throw: who he threw to, the best available alternative, and the gap.

    The gap is in expected points, and that is what a flag is decided on. With ``wp_model``
    (see :mod:`winprob`) it is also converted to win probability for the game it was in.
    """
    draws = [c for c in scored.columns if c.startswith("epa_")]
    rows, chosen_draws, alt_draws = [], [], []
    for (gid, pid), play in scored.groupby(bdb2025.KEY, sort=False):
        chosen = play[play["targeted"]].iloc[0]
        others = play[~play["targeted"] & play["available"]]
        row = {
            "gameId": gid, "playId": pid, "week": chosen["week"], "passer": chosen["passer"],
            "chosen": chosen["nflId"], "chosen_p_complete": chosen["p_complete"],
            "chosen_epa": chosen["exp_epa"], "n_alternatives": len(others),
            "alternative": pd.NA, "alt_p_complete": np.nan, "alt_epa": np.nan,
            "gap": 0.0, "gap_low": 0.0, "gap_high": 0.0, "agreement": 0.0, "flagged": False,
            "late_throw": False,
            "pass_result": chosen["pass_result"], "actual_epa": chosen["epa"],
            "time_to_throw": chosen["time_to_throw"],
        }
        if len(others):
            alt = others.loc[others["exp_epa"].idxmax()]
            gaps = alt[draws].to_numpy(float) - chosen[draws].to_numpy(float)
            row.update(
                alternative=alt["nflId"], alt_p_complete=alt["p_complete"], alt_epa=alt["exp_epa"],
                gap=float(gaps.mean()), gap_low=float(np.quantile(gaps, GAP_LOW_Q)),
                gap_high=float(np.quantile(gaps, 1 - GAP_LOW_Q)), agreement=float((gaps > 0).mean()),
            )
            row["late_throw"] = bool(chosen["time_to_throw"] > MAX_TIME_TO_THROW)
            row["flagged"] = bool(row["gap_low"] > 0 and row["gap"] >= min_gap and not row["late_throw"])
            alt_draws.append(alt[draws].to_numpy(float))
        else:
            alt_draws.append(chosen[draws].to_numpy(float))       # no alternative: a gap of zero
        chosen_draws.append(chosen[draws].to_numpy(float))
        row.update({c: chosen[c] for c in winprob.STATE if c in chosen.index})
        rows.append(row)
    out = pd.DataFrame(rows)
    if wp_model is not None and len(out) and set(winprob.STATE) <= set(out.columns):
        # Convert each refit's pair of values in its own game state, then take the spread.
        c, a = np.array(chosen_draws), np.array(alt_draws)        # throws x refits
        state = out.loc[out.index.repeat(c.shape[1]), winprob.STATE]
        gaps = (winprob.added(wp_model, a.ravel(), state) - winprob.added(wp_model, c.ravel(), state)).reshape(c.shape)
        out["wp_gap"] = gaps.mean(axis=1)
        out["wp_gap_low"] = np.quantile(gaps, GAP_LOW_Q, axis=1)
        out["wp_gap_high"] = np.quantile(gaps, 1 - GAP_LOW_Q, axis=1)
    return out


def check(decisions: pd.DataFrame) -> dict:
    """Did the flagged throws actually turn out worse? The model never saw the outcomes."""
    f, rest = decisions[decisions["flagged"]], decisions[~decisions["flagged"]]

    def side(d):
        return {"throws": len(d), "actual_epa": round(float(d["actual_epa"].mean()), 3),
                "predicted_epa": round(float(d["chosen_epa"].mean()), 3),
                "complete": round(float((d["pass_result"] == "C").mean()), 3),
                "intercepted": round(float((d["pass_result"] == "IN").mean()), 3)}

    # Standard error of the difference in mean actual EPA, to say whether the split is real.
    se = float(np.sqrt(f["actual_epa"].var() / max(len(f), 1) + rest["actual_epa"].var() / max(len(rest), 1)))
    diff = float(f["actual_epa"].mean() - rest["actual_epa"].mean()) if len(f) else float("nan")
    return {
        "throws": len(decisions),
        "flagged_share": round(len(f) / len(decisions), 3),
        "flagged": side(f) if len(f) else None,
        "not_flagged": side(rest),
        "actual_epa_difference": round(diff, 3),
        "difference_in_standard_errors": round(diff / se, 1) if se > 0 else None,
        "mean_gap_when_flagged": round(float(f["gap"].mean()), 3) if len(f) else None,
        "mean_wp_gap_when_flagged": round(float(f["wp_gap"].mean()), 4) if len(f) and "wp_gap" in f else None,
        "median_wp_gap_when_flagged": round(float(f["wp_gap"].median()), 4) if len(f) and "wp_gap" in f else None,
        "no_available_alternative_share": round(float((decisions["n_alternatives"] == 0).mean()), 3),
    }


def run(table: pd.DataFrame, log=None) -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """The whole thing: cross-score every throw, decide, and check against what happened."""
    scored = cross_score(table, log=log)
    decisions = decide(scored, wp_model=winprob.load_or_train())
    return scored, decisions, check(decisions)


def describe(decision: pd.Series, scored_play: pd.DataFrame) -> dict:
    """One throw as the record the rest of the project consumes."""
    names = dict(zip(bdb2025.players()["nflId"], bdb2025.players()["displayName"]))

    def receiver(nfl_id):
        r = scored_play[scored_play["nflId"] == nfl_id].iloc[0]
        return {"player_id": str(int(nfl_id)), "name": names.get(int(nfl_id)), "route": r["route"],
                "depth": round(float(r["depth"]), 1), "separation": round(float(r["separation"]), 1),
                "p_complete": round(float(r["p_complete"]), 3), "exp_epa": round(float(r["exp_epa"]), 2),
                "available": bool(r["available"])}

    has_alt = pd.notna(decision["alternative"])
    return {
        "play_id": f"{int(decision['gameId'])}-{int(decision['playId'])}",
        "kind": "target_choice",
        "t": float(decision["time_to_throw"]),
        "player": {"player_id": str(int(decision["passer"])), "name": names.get(int(decision["passer"]))},
        "flagged": bool(decision["flagged"]),
        "chosen": receiver(decision["chosen"]),
        "alternative": receiver(decision["alternative"]) if has_alt else None,
        "epa_gap": {"expected": round(float(decision["gap"]), 2), "low": round(float(decision["gap_low"]), 2),
                    "high": round(float(decision["gap_high"]), 2)},
        "win_prob_gap": ({"expected": round(float(decision["wp_gap"]), 4), "low": round(float(decision["wp_gap_low"]), 4),
                          "high": round(float(decision["wp_gap_high"]), 4)}
                         if "wp_gap" in decision.index and pd.notna(decision["wp_gap"]) else None),
        "confidence": round(float(decision["agreement"]), 2),
        "options": [receiver(i) for i in scored_play.sort_values("exp_epa", ascending=False)["nflId"]],
        "what_happened": {"pass_result": decision["pass_result"],
                          "epa": round(float(decision["actual_epa"]), 2)},
        "assumptions": ASSUMPTIONS,
    }
