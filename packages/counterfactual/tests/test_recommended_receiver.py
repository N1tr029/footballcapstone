"""Which receiver the report recommends, read from where the report puts it.

``decisions`` nests the identifier inside ``evidence``. Three readers looked for it
at the top level, where ``dict.get`` returns None without complaint, so every
``consider_alternative`` play reached the API recommending nobody — 39 of 52 on
week 1, and exactly the plays whose point is naming the receiver the ball should
have gone to.

It survived review twice because the ``reason`` string beside it is built from the
receiver object directly and named him correctly, so the output read as right.
"""

import importlib.util
import pathlib

import pytest

_spec = importlib.util.spec_from_file_location(
    "decision_engine",
    pathlib.Path(__file__).resolve().parents[1] / "decision_engine.py",
)
decision_engine = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(decision_engine)


def report(decision, **evidence):
    return {"game_id": 1, "play_id": 2, "decision": decision, "evidence": evidence}


def test_the_alternative_is_read_from_evidence_where_decisions_writes_it():
    r = report("consider_alternative",
               targeted_receiver_id=11, best_alternative_receiver_id=47856)
    assert decision_engine.recommended_receiver_id(r) == 47856


def test_a_reasonable_target_is_the_one_actually_thrown_to():
    r = report("target_was_reasonable",
               targeted_receiver_id=11, best_alternative_receiver_id=47856)
    assert decision_engine.recommended_receiver_id(r) == 11


def test_a_flattened_report_still_resolves():
    """Older reports put it at the top level; those must not start returning None."""
    r = {"game_id": 1, "play_id": 2, "decision": "consider_alternative",
         "best_alternative_receiver_id": 47856, "evidence": {}}
    assert decision_engine.recommended_receiver_id(r) == 47856


def test_nobody_recommended_is_still_nobody():
    r = report("consider_alternative", targeted_receiver_id=11)
    assert decision_engine.recommended_receiver_id(r) is None


@pytest.mark.parametrize("decision", ["consider_alternative", "target_was_reasonable"])
def test_the_payload_names_the_receiver_its_reason_names(decision):
    """The two must agree. They disagreeing silently is the whole bug."""
    r = report(decision, targeted_receiver_id=11, best_alternative_receiver_id=47856)
    expected = 47856 if decision == "consider_alternative" else 11
    assert decision_engine.recommended_receiver_id(r) == expected
