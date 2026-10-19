"""The report mapping, against the real reports committed with Landon's model.

``reports/decisions.jsonl`` came with the repository and is genuine output from the
real Kaggle data — 56 plays, four decision types, three different probability
key-sets. So this suite is not checking the mapping against a fixture built to
please it; it is checking it against the thing it will actually be handed.
"""

import json

import pytest

from cf_support import REPORTS
from gridiron_counterfactual import recommendation as rc

pytestmark = pytest.mark.skipif(
    not REPORTS.exists(), reason="reports/decisions.jsonl is not in the tree"
)


def reports():
    with open(REPORTS) as fh:
        return [json.loads(line) for line in fh if line.strip()]


def test_every_real_report_maps_and_validates():
    rows = reports()
    assert len(rows) >= 50, "the fixture has shrunk; this test is no longer meaningful"

    for row in rows:
        rec = rc.from_report(row)
        v = rc.validate(rec)
        assert v.ok, f"play {row['play_id']}: {v.problems}"


def test_all_four_decision_types_are_exercised():
    seen = {row["decision"] for row in reports()}
    assert seen == set(rc._CALLS) - {"suggest_punt"}, (
        f"the real reports cover {sorted(seen)}; suggest_punt needs a hand-built case"
    )
    # Fourth down with nobody open is in decision_engine's logic but absent from
    # this game, so it is covered explicitly rather than left untested.
    rec = rc.from_report({
        "game_id": 1, "play_id": 2, "decision": "suggest_punt",
        "status": "no_receiver_open_on_fourth_down", "confidence": "low",
        "probabilities": {"pass": 0.1, "run": 0.2, "punt": 0.7},
        "evidence": {"open_threshold_yards": 3.0},
    })
    assert rec.call.type == "punt"
    assert rc.validate(rec).ok


def test_epa_is_never_presented_as_the_recommendation_s_expected_value():
    """The mistake that would make the What-If view argue with itself.

    ``expected_points_added`` is the EPA of the play as actually called. Copied into
    ``expected_value`` it would read as the model's prediction for the alternative —
    so the panel would claim the recommendation's value is exactly the result of
    ignoring it.
    """
    withepa = [r for r in reports() if r.get("expected_points_added") is not None]
    assert withepa, "no report carries EPA; this test is no longer meaningful"

    for row in withepa:
        rec = rc.from_report(row)
        assert rec.expected_value is None
        assert rec.review["expected_points_added"] == row["expected_points_added"]
        assert "not a prediction" in rec.review["expected_points_added_note"]


def test_unset_numbers_always_carry_an_assumption_saying_why():
    """A UI cannot tell "not modelled" from "modelled as zero" unless told."""
    for row in reports():
        rec = rc.from_report(row)
        joined = " ".join(rec.assumptions)
        assert "win_prob_delta" in joined
        assert "expected_value" in joined
        assert "outcomes" in joined


def test_validate_catches_a_silently_empty_field():
    rec = rc.from_report(reports()[0])
    rec.assumptions = ["something, but not the disclosure"]
    v = rc.validate(rec)
    assert not v.ok
    assert any("win_prob_delta is unset" in p for p in v.problems)


def test_validate_catches_probabilities_that_do_not_sum_to_one():
    rec = rc.from_report(reports()[0])
    rec.choice_probabilities = {"a": 0.3, "b": 0.3}
    v = rc.validate(rec)
    assert not v.ok
    assert any("sum to" in p for p in v.problems)


def test_a_new_decision_string_raises_rather_than_guessing():
    """A capability the model grows must reach the renderer deliberately."""
    with pytest.raises(ValueError, match="unknown decision"):
        rc.from_report({"decision": "suggest_trick_play", "game_id": 1, "play_id": 1})


def test_the_choice_distribution_is_not_filed_as_an_outcome_distribution():
    """Landon's probabilities are over calls; the contract's outcomes are over
    defensive responses. Same shape, different object."""
    for row in reports():
        rec = rc.from_report(row)
        assert rec.outcomes == []
        assert rec.choice_probabilities == row["probabilities"]


def test_a_play_by_play_only_review_says_it_never_looked_at_tracking():
    runs = [r for r in reports() if r["decision"] == "suggest_pass_next_time"]
    assert runs
    rec = rc.from_report(runs[0])
    assert any("No tracking" in a for a in rec.assumptions)


def test_the_json_shape_carries_every_contract_field():
    d = rc.from_report(reports()[0]).to_json()
    for key in ("call", "expected_value", "win_prob_delta", "outcomes",
                "assumptions", "confidence", "choice_probabilities"):
        assert key in d
    assert set(d["call"]) == {"type", "concept", "direction"}
