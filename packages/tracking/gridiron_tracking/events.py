"""When the ball was thrown and when it arrived, from where the ball was.

The play model is a review of the airborne part of a pass: it forecasts from the
last pre-throw frame to the frame the ball arrives, and compares the targeted
receiver's separation against the alternatives at that instant. Both ends of that
window are events — ``pass_forward`` and one of the ``pass_*`` arrivals — and a
public release labels them by hand. A clip does not. Without them the forecast
starts at the end of the play, which is the failure
``gridiron_counterfactual.review`` was built to prevent: the model returns numbers
without complaint and 61 of 67 throws read as mistakes.

The derivation is geometry over the ball, not vision, and it keys on **speed**. A
thrown ball does one thing nothing else on the field does: it travels faster than a
person can run. A fast professional covers about ten yards a second flat out; a
forward pass leaves the hand at fifteen to twenty-five. There is no overlap, and
unlike every other signal available here it does not depend on who is near what.

Distance from the nearest player is the obvious alternative and it is wrong, which
is worth writing down because it looks right. The ball flies *over* people: in field
coordinates — which are two-dimensional, because a ground-plane homography has no
height — a pass passes directly through the defenders it clears, and a flight
defined as "far from everybody" is chopped into pieces by exactly the players the
quarterback was throwing over. Distance is kept only as a sanity check on where the
ball ended up, never to find the window.

So a flight is a run of frames where the ball is moving faster than a person can.
``pass_forward`` is the last frame before it; the arrival is the first frame after.

**What this will also catch, and why the thresholds are what they are.** A handoff
and a pitch also move the ball between players, and a hard pitch can briefly clear
the speed floor. The separation is duration and distance rather
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

FLIGHT_SPEED = 13.0
"""Yards per second above which nobody is carrying it.

The fastest humans on a football field reach about ten and a half yards a second,
briefly and in a straight line; a forward pass leaves the hand at fifteen to
twenty-five. This sits in the gap rather than against either edge. Set at eleven —
a whisker above the human limit — a ball carried by a receiver at full speed on a
breakaway brushes it, and the "flight" then runs on through the catch and down the
field, putting the arrival wherever he was tackled."""

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

    series = [(f.t, f.ball.x, f.ball.y) for f in track.frames if f.ball is not None]
    if len(series) < 4:
        out.notes.append("too few frames carry a ball position to find a throw")
        return out

    # Speed between consecutive ball samples. Gaps are spanned rather than skipped:
    # a ball lost for two frames mid-flight is still in flight, and the average over
    # the gap is the right estimate of how fast it was going.
    fast: list[bool] = [False]
    for (t0, x0, y0), (t1, x1, y1) in zip(series, series[1:]):
        dt = max(t1 - t0, 1e-6)
        fast.append(math.hypot(x1 - x0, y1 - y0) / dt > FLIGHT_SPEED)

    runs: list[list[int]] = []
    current: list[int] = []
    for i, is_fast in enumerate(fast):
        if is_fast:
            if not current:
                current = [i - 1] if i else [0]
            current.append(i)
        else:
            if current:
                runs.append(current)
            current = []
    if current:
        runs.append(current)

    if not runs:
        peak = max(
            (math.hypot(b[1] - a[1], b[2] - a[2]) / max(b[0] - a[0], 1e-6)
             for a, b in zip(series, series[1:])), default=0.0)
        out.notes.append(
            f"the ball never exceeded {peak:.1f} yards a second, and a person can "
            f"run {FLIGHT_SPEED:.0f} — nothing was thrown. This looks like a run, "
            f"or the track is of something that is not the ball."
        )
        return out

    best = max(runs, key=lambda r: series[r[-1]][0] - series[r[0]][0])
    first, last = series[best[0]], series[best[-1]]
    duration = last[0] - first[0]
    distance = math.hypot(last[1] - first[1], last[2] - first[2])
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
    start_i, end_i = best[0], best[-1]
    out.thrown_at = series[start_i][0]
    out.arrived_at = series[end_i][0]

    out.confidence = round(min(0.9, 0.4 + measured), 2) if measured else 0.5
    out.notes.append(
        f"ball above {FLIGHT_SPEED:.0f} yd/s for {duration:.2f} s over "
        f"{distance:.1f} yards"
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
