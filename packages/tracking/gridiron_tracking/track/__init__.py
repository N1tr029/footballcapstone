"""Multi-object tracking, done in yards rather than in pixels.

Almost every tutorial tracks in image space and projects at the end. That is the
wrong way round for this problem, for three reasons that all bite on football film:

*A panning camera moves every box.* In image space a pan is indistinguishable from
twenty-two players sprinting sideways, so the motion model spends the play fighting
the camera operator. In field space the pan is absorbed by the homography and stops
existing.

*Constant velocity is only true in yards.* A player running at a steady 8 yd/s
crosses wildly different numbers of pixels per frame depending on where he is in the
frame, so a pixel-space velocity model is wrong in a way that varies across the
image — worst at the far sideline, which is where the receivers are.

*Yards have limits that pixels do not.* Nobody runs faster than about 11 yd/s or
accelerates harder than about 9 yd/s². Those are strong, physical, free constraints
on which associations are even possible, and they are unusable until the coordinates
mean something.

The association itself is ordinary: gate, Hungarian, birth and death. The unusual
choice is the space it happens in.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np
from scipy.optimize import linear_sum_assignment

MAX_SPEED_YPS = 11.0
"""A fast NFL player covers 40 yards in about 4.3 seconds from a standstill and tops
out near 10.5 yd/s. Eleven leaves headroom without admitting teleportation."""

MAX_ACCEL_YPS2 = 9.0


@dataclass
class KalmanTrack:
    """Constant-velocity state in field yards: [x, y, vx, vy]."""

    id: int
    x: np.ndarray
    P: np.ndarray
    born: int
    last_seen: int
    hits: int = 1
    observations: dict[int, np.ndarray] = dc_field(default_factory=dict)
    label_votes: dict[str, int] = dc_field(default_factory=dict)
    team_votes: dict[str, int] = dc_field(default_factory=dict)

    @property
    def position(self) -> np.ndarray:
        return self.x[:2]

    @property
    def velocity(self) -> np.ndarray:
        return self.x[2:]

    def label(self) -> str | None:
        """The identity most of this track's detections claimed, if any did.

        Only the oracle and synthetic detectors supply labels. Voting rather than
        taking the first one means one mislabelled frame in the pile does not rename
        a whole track.
        """
        if not self.label_votes:
            return None
        return max(self.label_votes.items(), key=lambda kv: kv[1])[0]

    def team(self) -> str | None:
        if not self.team_votes:
            return None
        return max(self.team_votes.items(), key=lambda kv: kv[1])[0]


class FieldTracker:
    """Associate per-frame field positions into tracks.

    ``process_noise`` is in yards per second squared and is the knob that trades
    smoothness against responsiveness: too low and the filter refuses to believe a
    receiver's break, too high and it follows detection jitter into the stands.
    """

    def __init__(
        self,
        dt: float,
        gate_yards: float = 2.5,
        max_age: int = 8,
        min_hits: int = 3,
        process_noise: float = 6.0,
        measurement_noise: float = 0.35,
        oracle_association: bool = False,
    ) -> None:
        self.dt = dt
        # When on, detections that carry a ground-truth label are matched to the
        # track holding that label instead of to the nearest one. It is cheating,
        # deliberately: running the benchmark with it on and off isolates what
        # association costs from what registration and smoothing cost, and there is
        # no other way to separate those two.
        self.oracle_association = oracle_association
        self.gate = gate_yards
        self.max_age = max_age
        self.min_hits = min_hits
        self.q = process_noise
        self.r = measurement_noise
        self.tracks: list[KalmanTrack] = []
        # Live tracks are pruned as they die; _all keeps every track ever born,
        # because the output of a play is its whole history, not its survivors.
        self._all: list[KalmanTrack] = []
        self._next_id = 0

        self.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1],
        ], dtype=float)
        self.H = np.array([[1, 0, 0, 0], [0, 1, 0, 0]], dtype=float)
        self.R = np.eye(2) * (measurement_noise ** 2)

        # Piecewise-constant white acceleration.
        q = process_noise ** 2
        t2, t3, t4 = dt ** 2, dt ** 3, dt ** 4
        self.Q = q * np.array([
            [t4 / 4, 0, t3 / 2, 0],
            [0, t4 / 4, 0, t3 / 2],
            [t3 / 2, 0, t2, 0],
            [0, t3 / 2, 0, t2],
        ], dtype=float)

    # ------------------------------------------------------------------- step

    def update(
        self,
        frame: int,
        points: np.ndarray,
        labels: list[str | None] | None = None,
        teams: list[str | None] | None = None,
        scores: np.ndarray | None = None,
    ) -> None:
        """One frame's worth of field positions."""
        points = np.atleast_2d(np.asarray(points, dtype=float))
        if points.size == 0:
            points = np.empty((0, 2))
        good = np.isfinite(points).all(axis=1) if len(points) else np.zeros(0, dtype=bool)
        keep = np.flatnonzero(good)
        points = points[good]
        labels = [labels[i] for i in keep] if labels else [None] * len(points)
        teams = [teams[i] for i in keep] if teams else [None] * len(points)

        for t in self.tracks:
            t.x = self.F @ t.x
            t.P = self.F @ t.P @ self.F.T + self.Q
            # A filter that has coasted through an occlusion can accumulate a
            # velocity no human has. Clip it rather than let it fly off the field.
            v = np.linalg.norm(t.x[2:])
            if v > MAX_SPEED_YPS:
                t.x[2:] *= MAX_SPEED_YPS / v

        matched_tracks, matched_dets = self._associate(points, labels)

        for ti, di in zip(matched_tracks, matched_dets):
            t = self.tracks[ti]
            z = points[di]
            S = self.H @ t.P @ self.H.T + self.R
            K = t.P @ self.H.T @ np.linalg.inv(S)
            t.x = t.x + K @ (z - self.H @ t.x)
            t.P = (np.eye(4) - K @ self.H) @ t.P
            t.last_seen = frame
            t.hits += 1
            t.observations[frame] = t.position.copy()
            if labels[di]:
                t.label_votes[labels[di]] = t.label_votes.get(labels[di], 0) + 1
            if teams[di]:
                t.team_votes[teams[di]] = t.team_votes.get(teams[di], 0) + 1

        unmatched = set(range(len(points))) - set(matched_dets)
        for di in sorted(unmatched):
            self._birth(frame, points[di], labels[di], teams[di])

        self.tracks = [t for t in self.tracks if frame - t.last_seen <= self.max_age]

    def _associate(
        self, points: np.ndarray, labels: list[str | None] | None = None
    ) -> tuple[list[int], list[int]]:
        if not self.tracks or not len(points):
            return [], []
        pred = np.array([t.position for t in self.tracks])
        cost = np.linalg.norm(pred[:, None, :] - points[None, :, :], axis=-1)
        gated = np.where(cost <= self.gate, cost, 1e6)

        if self.oracle_association and labels is not None:
            known = np.array([[t.label()] for t in self.tracks], dtype=object)
            want = np.array([labels], dtype=object)
            # Identity decides; distance only breaks ties among equals.
            agrees = (known == want) & (known != None)  # noqa: E711
            named = np.array([lb is not None for lb in labels])
            claimed = np.array([[t.label() is not None] for t in self.tracks])
            both_named = claimed & named[None, :]
            gated = np.where(both_named, np.where(agrees, cost, 1e6), gated)
        ri, ci = linear_sum_assignment(gated)
        out_t, out_d = [], []
        for r, c in zip(ri, ci):
            if gated[r, c] < 1e6:
                out_t.append(int(r))
                out_d.append(int(c))
        return out_t, out_d

    def _birth(self, frame: int, z: np.ndarray, label: str | None, team: str | None) -> None:
        x = np.array([z[0], z[1], 0.0, 0.0])
        P = np.diag([self.r ** 2, self.r ** 2, MAX_SPEED_YPS ** 2, MAX_SPEED_YPS ** 2])
        t = KalmanTrack(id=self._next_id, x=x, P=P, born=frame, last_seen=frame)
        t.observations[frame] = z.copy()
        if label:
            t.label_votes[label] = 1
        if team:
            t.team_votes[team] = 1
        self.tracks.append(t)
        self._next_id += 1
        self._all.append(t)

    # ------------------------------------------------------------------ output

    def confirmed(self) -> list[KalmanTrack]:
        """Tracks seen often enough to be a person rather than a flicker."""
        return [t for t in self._all if t.hits >= self.min_hits]
