"""Boxes computed from tracking that is already known to be true.

The point is attribution. When a play comes out three yards wrong, the question is
always *which stage* — the detector missed people, the tracker swapped two of them,
the registration was off, or the smoother ate a cut. Running the back half of the
pipeline on detections that were *derived* from ground truth answers that, because
any error left over has to belong to the back half.

It also means the whole accuracy loop can be written, tested and tuned with no video
on disk at all, which is the difference between a benchmark that gets built and one
that waits on a download.

The noise model is not decoration. Real detections are jittery, occasionally absent,
and much worse in the pile than in space — turning those knobs one at a time is how
you find out which of them the tracker actually cannot survive, and that is a more
useful thing to know than a single number from real video.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contract import PlayTrack
from ..registration.camera import Camera
from . import Detection

PLAYER_HEIGHT_YARDS = 2.05
PLAYER_WIDTH_YARDS = 0.75
HELMET_SIZE_YARDS = 0.30


@dataclass
class NoiseModel:
    """How unlike ground truth the fake detections should be.

    Defaults are roughly what a decent detector does on a 720p sideline frame.
    """

    box_jitter_px: float = 2.0
    """Gaussian noise on each box edge."""

    dropout: float = 0.02
    """Chance a visible player simply is not detected on a given frame."""

    false_positives_per_frame: float = 0.1
    """Boxes on nobody — a referee, a coach, a bench player over the line."""

    crowd_radius_yards: float = 1.2
    """Inside this distance of another player, a detection is 'in the pile'."""

    crowd_dropout: float = 0.35
    """Dropout for players in the pile, where occlusion actually happens."""

    seed: int | None = 0


class SyntheticDetector:
    """Projects a known :class:`PlayTrack` through a known camera.

    ``kind`` picks what the boxes are meant to be around, which decides the anchor
    and the height correction downstream: ``helmet`` mimics the Kaggle oracle,
    ``body`` mimics a person detector.
    """

    def __init__(
        self,
        truth: PlayTrack,
        camera: Camera,
        image_size: tuple[int, int],
        kind: str = "body",
        noise: NoiseModel | None = None,
        label_boxes: bool = True,
    ) -> None:
        self.truth = truth
        self.camera = camera
        self.image_size = image_size
        self.kind = kind
        self.noise = noise or NoiseModel()
        self.label_boxes = label_boxes
        self._rng = np.random.default_rng(self.noise.seed)
        self._frames = {i: f for i, f in enumerate(truth.frames)}

    # ------------------------------------------------------------------ protocol

    def detect(self, frame_index: int, image=None) -> list[Detection]:
        frame = self._frames.get(frame_index)
        if frame is None:
            return []

        ids = list(frame.players.keys())
        pos = np.array([[frame.players[i].x, frame.players[i].y] for i in ids], dtype=float)
        out: list[Detection] = []

        crowded = self._crowding(pos)

        for k, pid in enumerate(ids):
            p = frame.players[pid]
            drop = self.noise.crowd_dropout if crowded[k] else self.noise.dropout
            if self._rng.random() < drop:
                continue
            box = self._box(p.x, p.y)
            if box is None:
                continue
            out.append(
                Detection(
                    frame=frame_index, x1=box[0], y1=box[1], x2=box[2], y2=box[3],
                    score=float(np.clip(self._rng.normal(0.85, 0.08), 0.05, 1.0)),
                    kind=self.kind,
                    label=pid if self.label_boxes else None,
                    team=self.truth.roster[pid].team if pid in self.truth.roster else None,
                    meta={"crowded": bool(crowded[k])},
                )
            )

        out.extend(self._false_positives(frame_index))
        return out

    # ------------------------------------------------------------------ internals

    def _crowding(self, pos: np.ndarray) -> np.ndarray:
        if len(pos) < 2:
            return np.zeros(len(pos), dtype=bool)
        d = np.linalg.norm(pos[:, None, :] - pos[None, :, :], axis=-1)
        np.fill_diagonal(d, np.inf)
        return d.min(axis=1) < self.noise.crowd_radius_yards

    def _box(self, x: float, y: float) -> tuple[float, float, float, float] | None:
        """Project a standing player to a box, sized by his actual projected height.

        Corners come from two world points rather than an assumed pixel size, so a
        player at the far hash is correctly smaller than one on the near sideline —
        which matters, because a box scale that does not shrink with distance makes
        a broken height correction look right.
        """
        if self.kind == "helmet":
            top = self.camera.project([[x, y, 2.05]])[0]
            half = self._half_size(x, y, HELMET_SIZE_YARDS)
            if not np.isfinite(top).all() or half is None:
                return None
            cx, cy = float(top[0]), float(top[1])
            box = (cx - half, cy - half, cx + half, cy + half)
        else:
            foot = self.camera.project([[x, y, 0.0]])[0]
            head = self.camera.project([[x, y, PLAYER_HEIGHT_YARDS]])[0]
            if not (np.isfinite(foot).all() and np.isfinite(head).all()):
                return None
            half_w = abs(float(foot[1] - head[1])) * (PLAYER_WIDTH_YARDS / PLAYER_HEIGHT_YARDS) / 2.0
            cx = float((foot[0] + head[0]) / 2.0)
            box = (cx - half_w, float(head[1]), cx + half_w, float(foot[1]))

        j = self.noise.box_jitter_px
        if j > 0:
            box = tuple(float(v + self._rng.normal(0.0, j)) for v in box)  # type: ignore[assignment]

        w, h = self.image_size
        # Off-frame players are genuinely not detected; partially visible ones are.
        cx = (box[0] + box[2]) / 2.0
        cy = (box[1] + box[3]) / 2.0
        if not (0 <= cx < w and 0 <= cy < h):
            return None
        return box  # type: ignore[return-value]

    def _half_size(self, x: float, y: float, size_yards: float) -> float | None:
        a = self.camera.project([[x, y, 2.05]])[0]
        b = self.camera.project([[x, y, 2.05 + size_yards]])[0]
        if not (np.isfinite(a).all() and np.isfinite(b).all()):
            return None
        return max(abs(float(a[1] - b[1])) / 2.0, 1.0)

    def _false_positives(self, frame_index: int) -> list[Detection]:
        n = self._rng.poisson(self.noise.false_positives_per_frame)
        w, h = self.image_size
        out = []
        for _ in range(int(n)):
            cx = float(self._rng.uniform(0, w))
            cy = float(self._rng.uniform(h * 0.4, h))
            s = float(self._rng.uniform(10, 40))
            out.append(
                Detection(
                    frame=frame_index, x1=cx - s / 3, y1=cy - s, x2=cx + s / 3, y2=cy,
                    score=float(self._rng.uniform(0.2, 0.6)), kind=self.kind,
                    label=None, meta={"false_positive": True},
                )
            )
        return out

    # --------------------------------------------------------------------- bulk

    def all(self) -> list[Detection]:
        out: list[Detection] = []
        for i in range(len(self.truth.frames)):
            out.extend(self.detect(i))
        return out


def sideline_camera(
    image_size: tuple[int, int] = (1280, 720),
    x: float = 60.0,
    y: float = -25.0,
    height: float = 12.0,
    look_at: tuple[float, float] = (60.0, 26.65),
    focal_px: float = 1400.0,
) -> Camera:
    """A plausible tripod, for tests and for the synthetic benchmark.

    Defaults put it at midfield, 25 yards back of the sideline and 12 yards up,
    aimed at the middle of the field — a high-school press box, near enough.
    """
    from ..registration.homography import Homography

    C = np.array([x, y, height], dtype=float)
    target = np.array([look_at[0], look_at[1], 0.0], dtype=float)
    z = target - C
    z /= np.linalg.norm(z)
    xa = np.cross(z, np.array([0.0, 0.0, 1.0]))
    xa /= np.linalg.norm(xa)
    ya = np.cross(z, xa)
    R = np.vstack([xa, ya, z])
    K = np.array([[focal_px, 0, image_size[0] / 2], [0, focal_px, image_size[1] / 2], [0, 0, 1.0]])

    # The ground-plane homography is the projection with the Z column dropped.
    P = K @ np.hstack([R, (-R @ C).reshape(3, 1)])
    H = P[:, [0, 1, 3]]
    return Camera(K=K, R=R, C=C, homography=Homography(H / H[2, 2]), principal_point_assumed=False)
