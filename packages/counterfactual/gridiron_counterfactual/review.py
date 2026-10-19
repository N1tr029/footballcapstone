"""One play, reviewed: the model's forecast, its judgement, and the alternative.

This is the top of the stack — the function the app and the CLI call, and the only
place the whole chain appears in one piece:

    a PlayTrack  ->  roles  ->  features  ->  the fitted regressors
                 ->  a forecast PlayTrack (SIMULATED)
                 ->  decision_engine's own decision logic
                 ->  a Recommendation

The decision itself is **not reimplemented here**. ``decision_engine.decisions`` is
imported and called, so its thresholds, its probabilities and its wording stay the
single source of truth; if Landon changes the open-separation boundary, this changes
with it. That module lives beside the package rather than inside it — it is a script
with an argparse ``main`` — so it is loaded by path rather than by import name,
which works whether or not anything has been pip-installed.

What this adds is everything around that call: the roles it needs and no dataset
outside the 2026 competition file carries, the play-level inputs it fills with zeros
when they are missing, and the conversion of its output into a shape the renderer
and the contract both recognise.
"""

from __future__ import annotations

import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import pandas as pd

from gridiron_tracking.contract import PlayTrack

from . import features, forecast, roles
from .recommendation import Recommendation, from_report

PACKAGE_ROOT = Path(__file__).resolve().parents[1]
DEFAULT_MODEL = PACKAGE_ROOT / "models/coordinate_model.joblib"


def decision_engine():
    """Landon's script, loaded from beside the package.

    By path and not by name: ``decision_engine.py`` is not part of the installed
    package, and copying its decision rules in here would mean two definitions of
    "open" that drift apart silently.
    """
    if "decision_engine" in sys.modules:
        return sys.modules["decision_engine"]
    path = PACKAGE_ROOT / "decision_engine.py"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} is missing. It is the play model itself, merged from "
            f"landonDuba/capstone-ML-model; this module is only its adapter."
        )
    spec = importlib.util.spec_from_file_location("decision_engine", path)
    module = importlib.util.module_from_spec(spec)
    sys.modules["decision_engine"] = module
    spec.loader.exec_module(module)
    return module


@dataclass
class Review:
    """What came of looking at one play."""

    track: PlayTrack
    """The real play, with roles attached."""
    forecast: PlayTrack
    """The simulated continuation. ``meta['source']`` is ``simulated``."""
    recommendation: Recommendation | None
    """None when the play is not one this model can judge — a run, a sack, a
    scramble. The model reviews the airborne part of a pass and there is no
    airborne part to review."""
    report: dict[str, Any] | None
    """``decision_engine``'s own output, unconverted, for anyone who wants it."""
    notes: list[str]

    def to_json(self) -> dict[str, Any]:
        return {
            "game_id": self.track.meta.get("game_id"),
            "play_id": self.track.meta.get("play_id"),
            "description": self.track.meta.get("description"),
            "recommendation": self.recommendation.to_json() if self.recommendation else None,
            "notes": list(self.notes),
        }


def review_play(
    track: PlayTrack,
    model_path: str | Path = DEFAULT_MODEL,
    horizons: int | None = None,
) -> Review:
    """Score one play and produce its counterfactual.

    ``track`` is any play in the contract — a release play, one of the Unity app's,
    or the vision pipeline's output once it carries a ball and the play state.
    Raises only on things that make the answer meaningless; a play the model simply
    cannot judge comes back with ``recommendation=None`` and a note saying why.
    """
    notes: list[str] = []

    roles.apply(track)
    assigned = track.meta["roles"]
    target = next((pid for pid, r in assigned.items() if r == roles.TARGET), None)

    # The instant the forecast starts from. The model was fitted on input that
    # ends at the throw, so the throw is where it has to start; the end of a whole
    # play is a state it never saw and the play is over by then.
    throw = next((t for name in roles.THROW_EVENTS
                  if (t := track.event_time(name)) is not None), None)
    arrival = next((t for name in roles.ARRIVAL_EVENTS
                    if (t := track.event_time(name)) is not None), None)

    # --- the play-level inputs video does not contain -----------------------
    if track.meta.get("ball_land_x") is None:
        ball = track.frame_at(arrival).ball if arrival is not None else None
        if ball is not None:
            features.with_play_state(
                track,
                ball_land_x=ball.x, ball_land_y=ball.y,
                yardline=float(track.meta.get("lineOfScrimmage") or ball.x),
                # Run the forecast to where the ball actually arrived, so the
                # separation it is judged on is separation at the catch point
                # rather than at an arbitrary horizon.
                frames_output=horizons or _steps(throw, arrival) or 15,
            )
            notes.append(
                "ball_land taken from the ball's position when the pass arrived — "
                "this play's own data, not a prediction"
            )
        else:
            raise ValueError(
                "this play has no pass arrival and no ball_land in meta, so the "
                "model has nothing to key its forecast on. Supply it with "
                "features.with_play_state, or review a pass."
            )

    # --- the forecast -------------------------------------------------------
    if throw is None:
        notes.append(
            "no pass_forward on this play, so the forecast starts from the last "
            "frame of the track rather than from a throw"
        )
    predictions = forecast.predict(track, str(model_path), horizons=horizons, at=throw)
    simulated = forecast.to_play_track(predictions, track, t0=throw)
    if throw is not None:
        notes.append(
            f"forecast made from the throw at t={throw}s, which is the state the "
            f"model was fitted on — not from the end of the play"
        )

    # --- the judgement, from decision_engine itself -------------------------
    if target is None:
        notes.append(
            f"no targeted receiver on this play (passResult "
            f"{track.meta.get('passResult')!r}), so there is no throw to review. "
            f"The forecast still stands."
        )
        return Review(track, simulated, None, None, notes)

    reports = _decide(track, predictions, assigned)
    if not reports:
        notes.append(
            "decision_engine returned no report for this play — it needs a "
            "targeted receiver and at least one defender in the forecast"
        )
        return Review(track, simulated, None, None, notes)

    report = reports[0]
    notes.append(
        "decision and thresholds are decision_engine's own; this module only "
        "supplied the roles and the play state it needed"
    )
    return Review(track, simulated, from_report(report), report, notes)


def _decide(track: PlayTrack, predictions: pd.DataFrame,
            assigned: dict[str, str]) -> list[dict[str, Any]]:
    """Call ``decision_engine.decisions`` with the frames it expects.

    It takes two DataFrames keyed on ``(game_id, play_id, nfl_id)``: a state frame,
    which it uses only to find the role column, and a predictions frame it reduces
    to each player's last forecast sample. Both are built here rather than letting
    it read a CSV, which is the whole point of the adapter.
    """
    engine = decision_engine()

    game_id = _as_int(track.meta.get("game_id"), 0)
    play_id = _as_int(track.meta.get("play_id"), 0)

    rows = predictions.copy()
    rows["game_id"] = game_id
    rows["play_id"] = play_id
    rows["nfl_id"] = rows["player_id"].map(lambda p: _as_int(p, 0))
    rows["frame_id"] = rows["horizon"]
    rows["player_role"] = rows["player_id"].map(assigned)
    rows = rows.dropna(subset=["player_role"])

    state = rows.drop_duplicates(subset=["nfl_id"])[
        ["game_id", "play_id", "nfl_id", "player_role"]
    ].copy()

    # data_dir is only globbed for supplementary_data.csv; pointing it at a
    # directory without one leaves `down` unset, which is correct here — the down
    # lives in the play's own meta and decision_engine only uses it to choose
    # between recommending a run and recommending a punt.
    reports = engine.decisions(state, rows, PACKAGE_ROOT / "models")

    for report in reports:
        for key, meta_key in (("down", "down"), ("yards_to_go", "yardsToGo"),
                              ("pass_result", "passResult"),
                              ("yards_gained", "yardsGained"),
                              ("play_description", "description")):
            if report.get(key) is None and track.meta.get(meta_key) is not None:
                report[key] = track.meta[meta_key]
    return reports


def _steps(throw: float | None, arrival: float | None) -> int | None:
    """Forecast steps from the throw to the ball arriving, at the contract rate."""
    if throw is None or arrival is None or arrival <= throw:
        return None
    from gridiron_tracking.contract import RATE

    return max(1, round((arrival - throw) * RATE))


def _as_int(v: Any, default: int) -> int:
    try:
        return int(v)
    except (TypeError, ValueError):
        return default
