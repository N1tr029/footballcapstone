"""The what-if decision, on scored throws written by hand.

`decide` and `check` take a scored table, so the rules — who counts as an alternative,
when a gap is a flag — can be tested without fitting anything.
"""

import numpy as np
import pandas as pd
import pytest

from gridiron_counterfactual import whatif
from gridiron_counterfactual.targets import FEATURES, SITUATION


def _receiver(play, nfl_id, targeted, draws, available=True, epa=0.0, result="C", time_to_throw=2.5):
    row = dict(gameId=1, playId=play, week=1, nflId=nfl_id, passer=99, targeted=targeted,
               available=available or targeted, p_complete=0.6, pass_result=result, epa=epa,
               time_to_throw=time_to_throw, exp_epa=float(np.mean(draws)))
    row.update({f"epa_{k}": v for k, v in enumerate(draws)})
    return row


def _decide(rows, **kw):
    return whatif.decide(pd.DataFrame(rows), **kw).set_index("playId")


def test_a_clear_and_agreed_gap_is_flagged():
    d = _decide([_receiver(1, 10, True, [-0.5, -0.4, -0.6]), _receiver(1, 11, False, [0.9, 1.0, 0.8])])
    assert d.loc[1, "flagged"] and d.loc[1, "alternative"] == 11
    assert d.loc[1, "gap"] == pytest.approx(1.4) and d.loc[1, "agreement"] == 1.0


def test_a_throw_long_after_the_snap_is_not_flagged_however_big_the_gap():
    d = _decide([_receiver(1, 10, True, [-0.5, -0.4, -0.6], time_to_throw=6.0),
                 _receiver(1, 11, False, [0.9, 1.0, 0.8], time_to_throw=6.0)])
    assert d.loc[1, "gap"] == pytest.approx(1.4) and d.loc[1, "late_throw"] and not d.loc[1, "flagged"]


def test_a_small_gap_is_not_a_mistake():
    d = _decide([_receiver(1, 10, True, [0.0, 0.0, 0.0]), _receiver(1, 11, False, [0.3, 0.3, 0.3])])
    assert not d.loc[1, "flagged"] and d.loc[1, "gap"] == pytest.approx(0.3)


def test_a_big_gap_the_refits_disagree_on_is_not_flagged():
    # Mean gap of 1.0, but one refit in three says the chosen man was the better throw.
    d = _decide([_receiver(1, 10, True, [0.0, 0.0, 0.0]), _receiver(1, 11, False, [2.0, 2.0, -1.0])])
    assert d.loc[1, "gap"] == pytest.approx(1.0) and d.loc[1, "gap_low"] < 0
    assert not d.loc[1, "flagged"]


def test_an_unavailable_receiver_is_never_the_alternative():
    d = _decide([_receiver(1, 10, True, [0.0, 0.0, 0.0]),
                 _receiver(1, 11, False, [3.0, 3.0, 3.0], available=False),
                 _receiver(1, 12, False, [0.2, 0.2, 0.2])])
    assert d.loc[1, "alternative"] == 12 and not d.loc[1, "flagged"]


def test_a_throw_with_nobody_else_available_has_no_alternative():
    d = _decide([_receiver(1, 10, True, [0.0, 0.0, 0.0]),
                 _receiver(1, 11, False, [3.0, 3.0, 3.0], available=False)])
    assert pd.isna(d.loc[1, "alternative"]) and d.loc[1, "n_alternatives"] == 0 and not d.loc[1, "flagged"]


def test_check_compares_what_happened_on_flagged_and_unflagged_throws():
    rows = []
    for play in range(20):
        bad = play < 5
        rows.append(_receiver(play, 10, True, [-1.0] * 3 if bad else [0.5] * 3,
                              epa=-1.0 if bad else 0.5, result="I" if bad else "C"))
        rows.append(_receiver(play, 11, False, [0.6] * 3))
    report = whatif.check(whatif.decide(pd.DataFrame(rows)))
    assert report["flagged_share"] == 0.25
    assert report["flagged"]["actual_epa"] == -1.0 and report["not_flagged"]["actual_epa"] == 0.5
    assert report["flagged"]["complete"] == 0.0


def test_scoring_never_sees_the_game_it_scores():
    """Cross-fitting: give one game an outcome no other game has, and it must not be learned."""
    rng = np.random.default_rng(0)
    rows = []
    for play in range(400):
        game = play % 5
        for r in range(2):
            row = {f: rng.normal() for f in FEATURES + SITUATION}
            row.update(gameId=game, playId=play, week=1, nflId=r, passer=99, targeted=r == 0,
                       complete=True, pass_result="C", yards_gained=5.0, route="GO",
                       time_to_throw=2.5, epa=50.0 if game == 0 else 0.0)
            rows.append(row)
    scored = whatif.cross_score(pd.DataFrame(rows), folds=5, refits=2)
    assert scored.loc[scored["gameId"] == 0, "exp_epa"].abs().max() < 1.0
