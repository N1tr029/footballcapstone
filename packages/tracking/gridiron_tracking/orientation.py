"""Which way the body is pointing — and how much to believe the answer.

This is the module docs/animation.md is a post-mortem about. The renderer used to
take a player's direction of travel and call it his facing, which is wrong for
exactly the players a play is about: a receiver drifting to the corner with his head
back at the ball, a defensive back in a backpedal, a quarterback retreating. Foles
spent the Philly Special looking at grass because of it.

A camera cannot measure ``o`` without a body model, so this pipeline cannot honestly
report ``measured`` until pose estimation lands. What it can do is be explicit:
produce the best inference available, label it with how it was arrived at, and let
the renderer badge a drawing as a drawing. That is the entire purpose of
``o_source``, and an inference that admits what it is beats a confident number that
happens to be wrong.

The rule here is deliberately the same one ``prototype/engine/adapters.js`` applies,
so that the pipeline and the renderer never disagree about a player's facing:

* above walking pace, take the direction of travel — ``from_velocity``
* below it, hold the last good value, because the direction of a near-zero velocity
  is numerically meaningless and a lineman in his stance would otherwise spin
* with nothing to hold, fall back to the resting default for his side of the
  ball — ``assumed``
* and cap how fast the answer may swing, because a human neck and hips have a rate
  limit and an uncapped derived angle does not
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contract import shortest_deg, wrap_deg

SLOW_YPS = 0.6
"""Below walking pace, direction of travel carries no information about facing."""

MAX_TURN_DEG_PER_S = 250.0
"""A person can turn about this fast at a sprint and rather less standing still.
Uncapped, a derived orientation flips 180 degrees between frames when a player
shuffles, and the renderer draws a man spinning on the spot."""


@dataclass
class Orientation:
    o: np.ndarray
    source: list[str]


def from_motion(
    t: np.ndarray,
    speed: np.ndarray,
    direction: np.ndarray,
    default: float = 0.0,
) -> Orientation:
    """Infer facing from a path, labelling every sample with its provenance.

    ``default`` is the resting facing for this player's side of the ball — zero
    (downfield) for the offense, 180 for the defense, which is at least right for
    twenty-two men standing on a line before the snap.
    """
    t = np.asarray(t, dtype=float)
    speed = np.asarray(speed, dtype=float)
    direction = np.asarray(direction, dtype=float)
    n = len(t)

    o = np.zeros(n)
    src: list[str] = []
    last: float | None = None

    for i in range(n):
        if speed[i] > SLOW_YPS:
            target, s = float(direction[i]), "from_velocity"
        elif last is not None:
            target, s = last, "from_velocity"
        else:
            target, s = float(default), "assumed"

        if last is not None:
            dt = (t[i] - t[i - 1]) if i > 0 else 0.0
            cap = MAX_TURN_DEG_PER_S * max(dt, 1e-6)
            delta = shortest_deg(last, target)
            delta = max(-cap, min(cap, delta))
            target = last + delta

        o[i] = wrap_deg(target)
        last = o[i]
        src.append(s)

    return Orientation(o=o, source=src)


def resting_default(team: str | None) -> float:
    """Which way a player faces when nothing else is known.

    Offense downfield, defense back at them. Anything else — an unassigned track, an
    official — gets downfield, and is marked ``assumed`` like everything else here.
    """
    if team and team.lower().startswith("def"):
        return 180.0
    return 0.0
