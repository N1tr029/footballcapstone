"""Detections, a registration and a clock in; a :class:`PlayTrack` out.

Everything it touches is behind an interface — the detector is a protocol, the
registration is a protocol — so the same function runs the synthetic round trip, the
Kaggle oracle and a real YOLO pass over a coach's upload. That is what makes the
benchmark able to attribute error: swap one component, rerun, read the difference.

The order matters and is not the obvious one. Detections are projected into field
coordinates *before* tracking, not after, for the reasons in
:mod:`gridiron_tracking.track`; and the resampling onto the contract's 10 Hz grid
happens *after* tracking, not before, so the filter sees every frame the camera
actually shot rather than a decimated ninth of them.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from . import kinematics, orientation
from .contract import RATE, PlayTrack, PlayerFrame, RosterEntry, TrackingFrame
from .detect import Detection, Detector
from .field import NFL, FieldSpec
from .track import FieldTracker


@dataclass
class PipelineConfig:
    fps: float
    out_rate: int = RATE
    snap_frame: int = 0
    """Which video frame is t = 0. Everything in the contract is relative to the
    snap, so a play whose snap is not located is a play whose clock is arbitrary."""

    spec: FieldSpec = NFL
    image_size: tuple[int, int] | None = None
    gate_yards: float = 2.5
    max_age: int = 8
    min_hits: int = 3
    process_noise: float = 6.0
    measurement_noise: float = 0.35
    smooth_window: int = 7
    oracle_association: bool = False
    """Match detections to tracks by their ground-truth label rather than by
    distance. Benchmark-only, and the whole point of it is to be turned off again."""

    keep_unlabelled: bool = True
    """Emit tracks the detector could not name. Off, the output is cleaner and the
    coverage figure is a lie."""


@dataclass
class PipelineResult:
    track: PlayTrack
    detections: int
    frames_registered: int
    frames_total: int
    tracks_born: int
    tracks_kept: int

    @property
    def registration_coverage(self) -> float:
        return self.frames_registered / self.frames_total if self.frames_total else 0.0


def run(
    detector: Detector,
    registration,
    n_frames: int,
    config: PipelineConfig,
    images=None,
) -> PipelineResult:
    """Run the back half of the stream over ``n_frames`` of one play.

    ``images`` is an optional sequence indexable by frame for detectors that need
    pixels; the synthetic and CSV-backed ones ignore it.
    """
    dt = 1.0 / config.fps
    tracker = FieldTracker(
        dt=dt,
        gate_yards=config.gate_yards,
        max_age=config.max_age,
        min_hits=config.min_hits,
        process_noise=config.process_noise,
        measurement_noise=config.measurement_noise,
        oracle_association=config.oracle_association,
    )

    n_detections = 0
    frames_registered = 0

    for i in range(n_frames):
        image = images[i] if images is not None else None
        dets: list[Detection] = detector.detect(i, image)
        n_detections += len(dets)
        if registration.at(i) is None:
            continue
        frames_registered += 1
        if not dets:
            tracker.update(i, np.empty((0, 2)))
            continue

        pts, labels, teams = _project(dets, registration, i, config.image_size)
        tracker.update(i, pts, labels=labels, teams=teams)

    kept = tracker.confirmed()
    track = _assemble(kept, config, n_frames)

    return PipelineResult(
        track=track,
        detections=n_detections,
        frames_registered=frames_registered,
        frames_total=n_frames,
        tracks_born=len(tracker._all),
        tracks_kept=len(kept),
    )


def _project(dets: list[Detection], registration, frame: int, image_size):
    """Anchor pixels to field yards, grouped by how high off the ground they are.

    Boxes of different kinds can arrive on one frame — a helmet model and a body
    model running together — and each needs its own plane intersection, so they are
    projected in groups rather than one call.
    """
    by_height: dict[float, list[int]] = {}
    anchors: list[tuple[float, float]] = []
    for k, d in enumerate(dets):
        pt, h = d.anchor()
        anchors.append(pt)
        by_height.setdefault(h, []).append(k)

    pts = np.full((len(dets), 2), np.nan)
    for height, idx in by_height.items():
        sub = np.array([anchors[k] for k in idx], dtype=float)
        got = registration.to_field(frame, sub, height_yards=height)
        pts[idx] = np.atleast_2d(got)

    return pts, [d.label for d in dets], [d.team for d in dets]


def _assemble(tracks, config: PipelineConfig, n_frames: int) -> PlayTrack:
    """Tracks to contract frames: resample, smooth, differentiate, orient."""
    dt_in = 1.0 / config.fps
    out_dt = 1.0 / config.out_rate

    # The output clock, relative to the snap, covering the whole play.
    t_start = (0 - config.snap_frame) * dt_in
    t_end = (n_frames - 1 - config.snap_frame) * dt_in
    grid = np.round(np.arange(t_start, t_end + 1e-9, out_dt), 3)

    frames: dict[float, TrackingFrame] = {t: TrackingFrame(t=float(t)) for t in grid}
    roster: dict[str, RosterEntry] = {}

    for tr in tracks:
        pid = tr.label()
        if pid is None:
            if not config.keep_unlabelled:
                continue
            pid = f"t{tr.id}"
        # Two tracks voting for one identity means the tracker lost him and started
        # again. Keeping both under distinct ids preserves the evidence; merging them
        # here would hide an ID switch from the scorer that exists to count it.
        if pid in roster:
            pid = f"{pid}#{tr.id}"

        obs_frames = np.array(sorted(tr.observations.keys()), dtype=float)
        if len(obs_frames) < 2:
            continue
        obs_xy = np.array([tr.observations[int(f)] for f in obs_frames], dtype=float)
        obs_t = (obs_frames - config.snap_frame) * dt_in

        t_out, xy = kinematics.resample(obs_t, obs_xy, grid)
        if not len(t_out):
            continue
        xy = kinematics.smooth(xy, window=config.smooth_window)
        k = kinematics.derive(t_out, xy)

        team = tr.team()
        ori = orientation.from_motion(
            t_out, k["s"], k["dir"], default=orientation.resting_default(team)
        )

        roster[pid] = RosterEntry(id=pid, team=team or "unknown", label=pid)
        for j, t in enumerate(t_out):
            tf = frames.get(round(float(t), 3))
            if tf is None:
                continue
            tf.players[pid] = PlayerFrame(
                x=float(xy[j, 0]), y=float(xy[j, 1]),
                s=float(k["s"][j]), a=float(k["a"][j]),
                dir=float(k["dir"][j]), o=float(ori.o[j]),
                o_source=ori.source[j],
            )

    ordered = [frames[t] for t in grid if frames[t].players]
    snap = min(ordered, key=lambda f: abs(f.t)) if ordered else None
    if snap is not None:
        # The pipeline was told where the snap was; say so in the output, because a
        # PlayTrack with no ball_snap fails the contract's own validator.
        snap.events.append("ball_snap")

    return PlayTrack(
        meta={
            "source": "vision",
            "rate": config.out_rate,
            "fps_in": config.fps,
            "snap_frame": config.snap_frame,
            "field": config.spec.name,
        },
        roster=roster,
        frames=ordered,
    )
