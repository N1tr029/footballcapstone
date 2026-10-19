"""A ``decision_engine`` report, in the shape ``docs/roles.md`` promised everyone.

The contract stream 5 builds its What-If view against is::

    call { type, concept, direction }, expected_value, win_prob_delta,
    outcomes[] (one per defensive response, with probability),
    assumptions[], confidence

The model next door produces something different: a bounded choice over calls, a
probability on each, numeric evidence, and a caveat. Translating one into the other
is mostly mechanical, and the interesting part is the three fields it *cannot* fill.

**``win_prob_delta`` is None.** roles.md asks for the recommendation "in
win-probability terms, not yards", and nothing in this model computes win
probability. There is no win-probability model in the repository yet. The field
stays empty rather than being filled with a plausible number, because a filled
field is one stream 5 will draw a chart from.

**``expected_value`` is None, and ``expected_points_added`` is not it.** The report
carries an `expected_points_added` for most plays. That is the EPA of the play that
actually happened, lifted from nflverse play-by-play — a measurement of the past,
not a prediction about the alternative. Copying it into ``expected_value`` would
produce a What-If panel in which the recommendation's expected value is always
exactly the result of ignoring the recommendation. It is kept, under ``review``,
where it belongs.

**``outcomes`` is empty, and ``choice_probabilities`` is new.** The contract wants
one outcome per defensive response. This model does not vary the defense; it
reports a distribution over *which call to make*, which is a different object with
the same shape, and putting it in ``outcomes`` would silently mislabel a choice
distribution as a response distribution. So ``outcomes`` stays empty until an
outcome model lands, and the distribution the model does produce gets its own
field. Per roles.md — "if you need a new field, it goes in the contract and
everyone gets it" — that is a contract change, recorded here and in
``docs/architecture.md``, not a local convenience.

Everything the model does not know is listed in ``assumptions``, which is the field
that makes the other three safe to leave empty.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any

CALL_TYPES = ("pass", "run", "punt", "field_goal", "unchanged")
"""What a call can be. ``unchanged`` is a real recommendation: the model reviewed
the play and is not asking for anything different."""

CONFIDENCE = ("low", "moderate", "high")

# decision -> (call type, concept, what the model is actually claiming)
_CALLS: dict[str, tuple[str, str | None, str]] = {
    "consider_alternative": (
        "pass", "alternative_receiver",
        "A different route runner was predicted to be more separated at the last "
        "tracked airborne frame than the receiver who was targeted.",
    ),
    "target_was_reasonable": (
        "unchanged", None,
        "No modelled alternative was meaningfully more separated than the targeted "
        "receiver. This is not a claim that the throw was good.",
    ),
    "suggest_run_play": (
        "run", None,
        "Every modelled receiver was predicted inside the open-separation threshold, "
        "so the pass had no modelled window.",
    ),
    "suggest_punt": (
        "punt", None,
        "Fourth down with no modelled receiver window.",
    ),
    "suggest_pass_next_time": (
        "pass", None,
        "A rule over play-by-play flagged the run as unsuccessful. No tracking was "
        "modelled for this play at all.",
    ),
}

# Said once here rather than on every report, because it applies to all of them.
_STANDING_ASSUMPTIONS = (
    "win_prob_delta is unset: no win-probability model exists in this repository yet.",
    "expected_value is unset: this model reports a choice distribution, not an "
    "expected value in yards or points.",
    "outcomes is empty: the defence is not varied, so there is no per-response "
    "distribution. choice_probabilities holds what the model does report.",
)


@dataclass
class Call:
    type: str
    concept: str | None = None
    direction: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {"type": self.type, "concept": self.concept, "direction": self.direction}


@dataclass
class Recommendation:
    """What stream 3 hands back, for one play."""

    play_id: Any
    game_id: Any
    call: Call
    confidence: str
    expected_value: float | None = None
    win_prob_delta: float | None = None
    outcomes: list[dict[str, Any]] = dc_field(default_factory=list)
    assumptions: list[str] = dc_field(default_factory=list)
    choice_probabilities: dict[str, float] = dc_field(default_factory=dict)
    evidence: dict[str, Any] = dc_field(default_factory=dict)
    review: dict[str, Any] = dc_field(default_factory=dict)
    """What actually happened, when play-by-play says. Deliberately a separate
    block: it is the outcome of the play as called, and it neither confirms nor
    refutes the recommendation."""

    def to_json(self) -> dict[str, Any]:
        return {
            "game_id": self.game_id,
            "play_id": self.play_id,
            "call": self.call.to_json(),
            "expected_value": self.expected_value,
            "win_prob_delta": self.win_prob_delta,
            "outcomes": list(self.outcomes),
            "choice_probabilities": dict(self.choice_probabilities),
            "confidence": self.confidence,
            "assumptions": list(self.assumptions),
            "evidence": dict(self.evidence),
            "review": dict(self.review),
        }


def from_report(report: dict[str, Any]) -> Recommendation:
    """One line of ``reports/decisions.jsonl`` as a :class:`Recommendation`.

    Raises on a decision this mapping has never seen, rather than guessing a call
    type: a new decision string is the model gaining a capability, and it should
    reach the renderer deliberately.
    """
    decision = report.get("decision")
    if decision not in _CALLS:
        raise ValueError(
            f"unknown decision {decision!r}; decision_engine has grown a new one. "
            f"Add it to _CALLS with its call type, or the What-If view will draw a "
            f"call nobody chose. Known: {sorted(_CALLS)}"
        )

    call_type, concept, claim = _CALLS[decision]
    confidence = report.get("confidence", "low")
    if confidence not in CONFIDENCE:
        confidence = "low"

    probs = {k: float(v) for k, v in (report.get("probabilities") or {}).items()}

    assumptions = [claim, *_STANDING_ASSUMPTIONS]
    if report.get("caveat"):
        assumptions.append(str(report["caveat"]))
    threshold = (report.get("evidence") or {}).get("open_threshold_yards")
    if threshold is not None:
        assumptions.append(
            f"The open/not-open boundary is a flat {threshold} yards of separation, "
            f"chosen by hand, not learned."
        )
    if decision == "suggest_pass_next_time":
        assumptions.append(
            "This play was reviewed from play-by-play only. No tracking, no "
            "forecast, and no receiver was examined."
        )

    review: dict[str, Any] = {}
    for key in ("play_description", "pass_result", "yards_gained", "down",
                "yards_to_go", "expected_points_added"):
        if report.get(key) is not None:
            review[key] = report[key]
    if "expected_points_added" in review:
        review["expected_points_added_note"] = (
            "EPA of the play as actually called. A measurement of what happened, "
            "not a prediction about the recommendation."
        )

    return Recommendation(
        game_id=report.get("game_id"),
        play_id=report.get("play_id"),
        call=Call(type=call_type, concept=concept),
        confidence=confidence,
        expected_value=None,
        win_prob_delta=None,
        outcomes=[],
        assumptions=assumptions,
        choice_probabilities=probs,
        evidence=dict(report.get("evidence") or {}),
        review=review,
    )


def read_reports(path: str) -> list[Recommendation]:
    """A whole ``decisions.jsonl``, in contract shape."""
    import json

    with open(path) as fh:
        return [from_report(json.loads(line)) for line in fh if line.strip()]


@dataclass
class Validation:
    ok: bool
    problems: list[str]


def validate(rec: Recommendation | dict[str, Any]) -> Validation:
    """Worth running before anything draws this.

    The strict rule here is the one that protects the viewer: a recommendation may
    leave a number unset, but it may not leave it unset *silently*. An empty
    ``expected_value`` with no assumption explaining the gap is a bug, because the
    UI has no way to tell "not modelled" from "modelled as zero".
    """
    d = rec.to_json() if isinstance(rec, Recommendation) else rec
    problems: list[str] = []

    call = d.get("call") or {}
    if call.get("type") not in CALL_TYPES:
        problems.append(f"call type {call.get('type')!r} is not one of {list(CALL_TYPES)}")
    if d.get("confidence") not in CONFIDENCE:
        problems.append(f"confidence {d.get('confidence')!r} is not one of {list(CONFIDENCE)}")

    probs = d.get("choice_probabilities") or {}
    if probs:
        total = sum(probs.values())
        if abs(total - 1.0) > 1e-6:
            problems.append(f"choice probabilities sum to {total}, not 1")
        if any(v < 0.0 or v > 1.0 for v in probs.values()):
            problems.append("a choice probability is outside [0, 1]")

    assumptions = d.get("assumptions") or []
    if not assumptions:
        problems.append("no assumptions listed")
    joined = " ".join(assumptions)
    for name in ("expected_value", "win_prob_delta"):
        if d.get(name) is None and name not in joined:
            problems.append(f"{name} is unset and no assumption says why")
    if not d.get("outcomes") and "outcomes" not in joined:
        problems.append("outcomes is empty and no assumption says why")

    for outcome in d.get("outcomes") or []:
        if "probability" not in outcome:
            problems.append("an outcome carries no probability")

    return Validation(not problems, problems)
