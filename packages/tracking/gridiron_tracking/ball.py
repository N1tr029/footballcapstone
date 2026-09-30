"""Tracking the ball, which is the hardest object on the field.

A football is ten pixels of brown against brown-and-white players on green, occluded
for most of every running play, and a general-purpose detector reports it at around a
tenth of the confidence it gives a person. Thresholding those detections frame by frame
gives what it gave here on the first attempt: the ball on 43% of frames and blinking.

The fix is not a better threshold, it is refusing to treat frames independently. A ball
obeys physics. It cannot jump twenty yards between frames, it does not change direction
instantly while carried, and when it vanishes it is almost always behind somebody rather
than genuinely gone. So this keeps a filtered estimate and asks of each candidate "could
the ball I was tracking have got here?" — which lets a 0.07-confidence detection be
accepted when it lands where the ball was going, and a 0.3 one be rejected when it does
not.

Three states, and they are reported rather than blended, on the same principle as
``o_source``:

``detected``  a candidate was accepted this frame.
``coasted``   no candidate; the filter carried the estimate forward. Good for a fraction
              of a second, meaningless after a couple.
``carrier``   the ball is attributed to whoever possession says has it. This is not a
              sighting and must never be drawn as one.

A renderer that shows all three identically is lying about two of them.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Literal

import numpy as np

MAX_SPEED_YPS = 30.0
"""A thrown football leaves the hand at roughly 20 yd/s and a punt is faster. Thirty is
generous; anything above it is a detection on something that is not the ball."""

MAX_CARRY_SPEED_YPS = 11.0
"""While carried, the ball moves at running speed, not throwing speed."""

GATE_YARDS = 4.0
"""How far from the prediction a candidate may land and still be believed."""

MAX_COAST_S = 0.8
"""Past this with no sighting, the estimate is fiction and the state stops being
``coasted``."""

NEAR_PLAYER_YARDS = 5.0
"""A candidate further than this from every player, while nobody has thrown it, is
noise — grass, a helmet, a line marking."""


@dataclass
class BallSample:
    t: float
    x: float
    y: float
    source: Literal["detected", "coasted", "carrier", "none"]
    confidence: float = 0.0

    @property
    def measured(self) -> bool:
        return self.source == "detected"


@dataclass
class BallTrack:
    samples: list[BallSample] = dc_field(default_factory=list)

    def counts(self) -> dict[str, int]:
        out: dict[str, int] = {}
        for s in self.samples:
            out[s.source] = out.get(s.source, 0) + 1
        return out

    def at(self, t: float, tol: float = 0.06) -> BallSample | None:
        best = min(self.samples, key=lambda s: abs(s.t - t), default=None)
        return best if best and abs(best.t - t) <= tol else None

    def summary(self) -> str:
        c = self.counts()
        n = len(self.samples) or 1
        return (f"ball: {c.get('detected', 0)} detected, {c.get('coasted', 0)} coasted, "
                f"{c.get('carrier', 0)} from possession, {c.get('none', 0)} unknown "
                f"({100 * c.get('detected', 0) / n:.0f}% measured)")


class BallTracker:
    """Constant-velocity filter in field yards, gated on what a ball can do.

    Deliberately simple. The value is not in the filter, it is in refusing candidates
    that imply impossible motion and in recording how each sample was arrived at.
    """

    def __init__(
        self,
        dt: float,
        gate_yards: float = GATE_YARDS,
        process_noise: float = 12.0,
        measurement_noise: float = 1.2,
    ) -> None:
        self.dt = dt
        self.gate = gate_yards
        self.x: np.ndarray | None = None
        self.P = np.eye(4) * 100.0
        self.F = np.array([[1, 0, dt, 0], [0, 1, 0, dt], [0, 0, 1, 0], [0, 0, 0, 1]], float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], float)
        self.R = np.eye(2) * measurement_noise ** 2
        q = process_noise ** 2
        t2, t3, t4 = dt ** 2, dt ** 3, dt ** 4
        self.Q = q * np.array([[t4 / 4, 0, t3 / 2, 0], [0, t4 / 4, 0, t3 / 2],
                               [t3 / 2, 0, t2, 0], [0, t3 / 2, 0, t2]], float)
        self.since_seen = 0.0

    def _predict(self) -> np.ndarray | None:
        if self.x is None:
            return None
        self.x = self.F @ self.x
        self.P = self.F @ self.P @ self.F.T + self.Q
        v = float(np.linalg.norm(self.x[2:]))
        if v > MAX_SPEED_YPS:
            self.x[2:] *= MAX_SPEED_YPS / v
        return self.x[:2].copy()

    def _update(self, z: np.ndarray) -> None:
        S = self.H @ self.P @ self.H.T + self.R
        K = self.P @ self.H.T @ np.linalg.inv(S)
        self.x = self.x + K @ (z - self.H @ self.x)
        self.P = (np.eye(4) - K @ self.H) @ self.P

    def step(
        self,
        t: float,
        candidates: list[tuple[float, float, float]],
        players: np.ndarray | None = None,
        carrier_xy: tuple[float, float] | None = None,
    ) -> BallSample:
        """One frame. ``candidates`` are (x, y, confidence) in field yards."""
        pred = self._predict()

        # A candidate has to be near somebody, unless the ball is clearly in flight.
        usable = []
        for cx, cy, conf in candidates:
            if not np.isfinite([cx, cy]).all():
                continue
            if players is not None and len(players):
                d = float(np.min(np.linalg.norm(players - np.array([cx, cy]), axis=1)))
                in_flight = pred is not None and float(np.linalg.norm(self.x[2:])) > MAX_CARRY_SPEED_YPS
                if d > NEAR_PLAYER_YARDS and not in_flight:
                    continue
            usable.append((cx, cy, conf))

        pick = None
        if usable:
            if pred is None:
                pick = max(usable, key=lambda c: c[2])
            else:
                gated = [c for c in usable
                         if np.hypot(c[0] - pred[0], c[1] - pred[1]) <= self.gate]
                if gated:
                    # Nearest the prediction, not the most confident: a ball that is
                    # where it should be is the ball, whatever the detector thinks.
                    pick = min(gated, key=lambda c: np.hypot(c[0] - pred[0], c[1] - pred[1]))

        if pick is not None:
            z = np.array([pick[0], pick[1]])
            if self.x is None:
                self.x = np.array([z[0], z[1], 0.0, 0.0])
                self.P = np.diag([4.0, 4.0, MAX_SPEED_YPS ** 2, MAX_SPEED_YPS ** 2])
            else:
                self._update(z)
            self.since_seen = 0.0
            return BallSample(t, float(self.x[0]), float(self.x[1]), "detected", pick[2])

        self.since_seen += self.dt
        if self.x is not None and self.since_seen <= MAX_COAST_S:
            return BallSample(t, float(self.x[0]), float(self.x[1]), "coasted")

        if carrier_xy is not None:
            # Possession says who has it; that is an attribution, not a sighting, and
            # the filter is re-seeded there so a later sighting can be gated sensibly.
            self.x = np.array([carrier_xy[0], carrier_xy[1], 0.0, 0.0])
            self.P = np.diag([9.0, 9.0, MAX_CARRY_SPEED_YPS ** 2, MAX_CARRY_SPEED_YPS ** 2])
            return BallSample(t, float(carrier_xy[0]), float(carrier_xy[1]), "carrier")

        return BallSample(t, float("nan"), float("nan"), "none")


def track_ball(
    times: list[float],
    candidates_by_t: dict[float, list[tuple[float, float, float]]],
    players_by_t: dict[float, np.ndarray] | None = None,
    carrier_by_t: dict[float, tuple[float, float]] | None = None,
    dt: float | None = None,
) -> BallTrack:
    """Run the tracker over a play."""
    if dt is None:
        dt = (times[1] - times[0]) if len(times) > 1 else 0.1
    tr = BallTracker(dt=dt)
    out = BallTrack()
    for t in times:
        key = round(t, 3)
        out.samples.append(tr.step(
            t,
            candidates_by_t.get(key, []),
            players=(players_by_t or {}).get(key),
            carrier_xy=(carrier_by_t or {}).get(key),
        ))
    return out
