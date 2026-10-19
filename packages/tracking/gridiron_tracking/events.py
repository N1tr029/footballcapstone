"""When the ball was thrown and when it arrived, from where the ball was.

The play model is a review of the airborne part of a pass: it forecasts from the
last pre-throw frame to the frame the ball arrives, and compares the targeted
receiver's separation against the alternatives at that instant. Both ends of that
window are events — ``pass_forward`` and one of the ``pass_*`` arrivals — and a
public release labels them by hand. A clip does not. Without them the forecast
starts at the end of the play, which is the failure
``gridiron_counterfactual.review`` was built to prevent: the model returns numbers
without complaint and 61 of 67 throws read as mistakes.

The derivation is geometry over the ball, not vision. A thrown ball does one thing
nothing else on the field does: it leaves everybody. Held or carried, it is within
about a yard of a person for the whole play; in the air it is five, ten, twenty
yards from the nearest of twenty-two people, and it covers ground faster than any
of them can run.

So a flight is a run of frames where the ball is far from everyone and moving fast.
``pass_forward`` is the last frame before it; the arrival is the first frame after,
when the ball is back within reach of somebody.

**What this will also catch, and why the thresholds are what they are.** A handoff
and a pitch also take the ball away from one player and give it to another, and a
short one looks like a short pass. The separation is duration and distance rather
than kind: a handoff is over in a tenth of a second and travels a yard or two, so a
flight has to last and has to go somewhere before it counts. A shovel pass sits
exactly on that boundary and will be missed; that is the right error, because
calling a handoff a throw puts the forecast window in the wrong place entirely.

**And what it cannot do.** On a ball track that is mostly coasted, the "speed" being
measured is the interpolation's, not the ball's. :func:`derive` refuses below a
floor for that reason, and the refusal is the point: a throw instant invented from
a smoothed line through two sightings is worse than no throw instant at all.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Any

from .contract import RATE, PlayTrack

HELD_YARDS = 1.8
"""Within this of a person, the ball is in somebody's hands. Generous: the ball is
carried at the hip and a track's position is an estimate."""

FLIGHT_YARDS = 3.0
"""Beyond this from everybody, nobody is holding it."""

MIN_FLIGHT_SECONDS = 0.3
"""A handoff is over inside this. A forward pass is not."""

MIN_FLIGHT_DISTANCE = 6.0
"""And it has to go somewhere. A pitch travels a couple of yards."""

MIN_MEASURED_FRACTION = 0.25
"""How much of the ball track has to be real detections. Below this the speeds
being measured belong to the interpolation rather than to the ball."""


@dataclass
class PassEvents:
    thrown_at: float | None = None
    arrived_at: float | None = None
    flight_yards: float = 0.0
    confidence: float = 0.0
    notes: list[str] = dc_field(default_factory=list)

    @property
    def found(self) -> bool:
        return self.thrown_at is not None and self.arrived_at is not None

    def to_json(self) -> dict[str, Any]:
        return {
            "thrown_at": self.thrown_at, "arrived_at": self.arrived_at,
            "flight_yards": round(self.flight_yards, 2),
            "confidence": self.confidence, "notes": list(self.notes),
            "source": "derived from the ball track, not labelled",
        }


def derive(track: PlayTrack) -> PassEvents:
    """Find the throw and the arrival, or say why not.

    Returns an empty result rather than a guess on a run play — there is no throw
    on one, and inventing a window would hand the model a receiver nobody threw to.
    """
    out = PassEvents()

    stats = track.meta.get("ball") or {}
    measured = float(stats.get("measured_fraction") or 0.0)
    if measured and measured < MIN_MEASURED_FRACTION:
        out.notes.append(
            f"the ball was actually seen on {measured:.0%} of samples; below "
            f"{MIN_MEASURED_FRACTION:.0%} a throw instant would be read off the "
            f"interpolation rather than off the ball"
        )
        return out

    # Distance from the ball to the nearest person, frame by frame.
    loose: list[tuple[float, float, float]] = []   # (t, x, y) while nobody holds it
    series: list[tuple[float, float, float, float]] = []  # (t, x, y, nearest)
    for f in track.frames:
        if f.ball is None:
            continue
        near = math.inf
        for p in f.players.values():
            d = math.hypot(p.x - f.ball.x, p.y - f.ball.y)
            if d < near:
                near = d
        series.append((f.t, f.ball.x, f.ball.y, near))

    if len(series) < 4:
        out.notes.append("too few frames carry a ball position to find a throw")
        return out

    # The longest run of frames where the ball is away from everybody.
    runs: list[list[tuple[float, float, float, float]]] = []
    current: list[tuple[float, float, float, float]] = []
    for sample in series:
        if sample[3] > FLIGHT_YARDS:
            current.append(sample)
        else:
            if current:
                runs.append(current)
            current = []
    if current:
        runs.append(current)

    if not runs:
        out.notes.append(
            "the ball is within reach of a player on every frame — this looks like "
            "a run, and there is no throw to find"
        )
        return out

    best = max(runs, key=lambda r: (r[-1][0] - r[0][0]))
    duration = best[-1][0] - best[0][0]
    distance = math.hypot(best[-1][1] - best[0][1], best[-1][2] - best[0][2])
    out.flight_yards = distance

    if duration < MIN_FLIGHT_SECONDS or distance < MIN_FLIGHT_DISTANCE:
        out.notes.append(
            f"the longest time the ball spent away from everybody is {duration:.2f} s "
            f"over {distance:.1f} yards, which is a handoff or a pitch rather than a "
            f"throw (a throw must clear {MIN_FLIGHT_SECONDS} s and "
            f"{MIN_FLIGHT_DISTANCE} yards)"
        )
        return out

    # The throw is the last frame the ball was still in someone's hands; the arrival
    # is the first frame it is back in them. Falling back to the flight's own ends
    # when the run touches the edge of the clip.
    times = [s[0] for s in series]
    start_i = times.index(best[0][0])
    end_i = times.index(best[-1][0])
    out.thrown_at = series[start_i - 1][0] if start_i > 0 else best[0][0]
    out.arrived_at = series[end_i + 1][0] if end_i + 1 < len(series) else best[-1][0]

    out.confidence = round(min(0.9, 0.4 + measured), 2) if measured else 0.5
    out.notes.append(
        f"ball away from every player for {duration:.2f} s over {distance:.1f} yards"
    )
    if start_i == 0:
        out.notes.append("the flight starts at the first ball frame; the throw may "
                         "be before the clip opens")
    return out


def apply(track: PlayTrack) -> PassEvents:
    """Derive the events and write them onto the frames, in place.

    Named after the tracking vocabulary the contract already uses, so a derived
    play and a labelled release play are read by exactly the same code downstream.
    """
    found = derive(track)
    if not found.found:
        track.meta["pass_events"] = found.to_json()
        return found

    for f in track.frames:
        if abs(f.t - found.thrown_at) < 0.5 / RATE and "pass_forward" not in f.events:
            f.events.append("pass_forward")
        if abs(f.t - found.arrived_at) < 0.5 / RATE and "pass_arrived" not in f.events:
            f.events.append("pass_arrived")

    track.meta["pass_events"] = found.to_json()
    return found
