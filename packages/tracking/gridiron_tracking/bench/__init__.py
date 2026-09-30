"""The tuning loop.

One number, in yards, that changes when you change the pipeline. Everything else in
this package exists to make this module's output trustworthy.

:func:`synthetic_roundtrip` is the version that needs no video: take a real NGS play,
film it with a virtual camera, hand the pipeline nothing but the boxes and a set of
clicked landmarks, and see how much of the original it gets back. Because the truth
is known exactly, a regression anywhere in registration, tracking, resampling or
smoothing shows up here immediately and attributably — and because it runs in a
second, it can run on every change rather than once a week.

It is not a substitute for real footage. It cannot tell you whether the detector
finds players in a November drizzle. It tells you whether everything *after* the
detector is sound, which is the half of the stream you cannot debug by looking at it.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ..contract import PlayTrack
from ..detect.synthetic import NoiseModel, SyntheticDetector, sideline_camera
from ..field import NFL, FieldSpec, catalog
from ..pipeline import PipelineConfig, PipelineResult, run
from ..registration import Anchor, Correspondence, PointClickRegistration
from .score import GATE_YARDS, PlayerScore, Score, score

__all__ = [
    "Score", "PlayerScore", "score", "GATE_YARDS",
    "synthetic_roundtrip", "RoundTrip", "clicked_anchor", "frame_the_play",
    "DEFAULT_LANDMARKS",
]

DEFAULT_LANDMARKS = (
    "y20A/sideline_y0", "y50/sideline_y0", "y20B/sideline_y0",
    "y20A/sideline_ymax", "y50/sideline_ymax", "y20B/sideline_ymax",
)
"""Six points spanning both sidelines and sixty yards of field.

Chosen to be *spread*, not to be convenient. Four points clicked near each other fit
a homography perfectly and then place a receiver in the car park, because the fit has
no evidence about the direction nobody clicked in — which is why
:meth:`PointClickRegistration.assess` checks the span rather than only the residual.
"""


def frame_the_play(truth: PlayTrack, image_size=(1280, 720), height: float = 12.0,
                   setback: float = 25.0, focal_px: float = 1400.0):
    """A virtual camera aimed where the play actually happens.

    The first version of this harness bolted the camera to midfield and left it
    there, and plays near either end zone scored 65% coverage — which looked like a
    tracking failure and was really a framing one: the players were off the edge of
    a frame nobody had pointed at them. A real operator pans to follow the ball, so
    the benchmark's camera has to as well, or every number it produces is a mixture
    of pipeline quality and how close the play happened to be to the 50.

    It stays on one sideline and at one height — panning *within* a play is a
    separate problem that belongs to homography propagation, not to this.
    """
    xs = [p.x for f in truth.frames for p in f.players.values()]
    ys = [p.y for f in truth.frames for p in f.players.values()]
    cx = float(np.mean(xs)) if xs else 60.0
    cy = float(np.mean(ys)) if ys else 26.65
    return sideline_camera(
        image_size=image_size, x=cx, y=-setback, height=height,
        look_at=(cx, cy), focal_px=focal_px,
    )


@dataclass
class RoundTrip:
    score: Score
    result: PipelineResult
    truth: PlayTrack
    registration_rms_yards: float


def visible_landmarks(camera, image_size, spec: FieldSpec = NFL, want: int = 6):
    """Pick well-spread landmarks that are actually inside this frame.

    An operator can only click what they can see. Choosing the points for them by
    taking the widest-spread visible ones is the best case a real interface could
    achieve, which is the right thing for a benchmark to assume.
    """
    w, h = image_size
    cat = catalog(spec)
    seen = []
    for lid, lm in cat.items():
        if not lid.startswith(("y", "goal")):
            continue
        px = camera.project([[lm.x, lm.y, 0.0]])[0]
        if np.isfinite(px).all() and 0 <= px[0] < w and 0 <= px[1] < h:
            seen.append((lid, lm))
    if len(seen) < 4:
        return [lid for lid, _ in seen]

    # Greedy farthest-point selection in field coordinates: spread is what the fit
    # needs, and it is spread in yards that matters, not in pixels.
    chosen = [max(seen, key=lambda s: s[1].x)]
    while len(chosen) < min(want, len(seen)):
        best = max(
            (s for s in seen if s[0] not in {c[0] for c in chosen}),
            key=lambda s: min((s[1].x - c[1].x) ** 2 + (s[1].y - c[1].y) ** 2 for c in chosen),
        )
        chosen.append(best)
    return [lid for lid, _ in chosen]


def clicked_anchor(
    camera,
    landmark_ids=DEFAULT_LANDMARKS,
    click_noise_px: float = 0.0,
    spec: FieldSpec = NFL,
    frame: int = 0,
    seed: int | None = 0,
) -> Anchor:
    """Simulate an operator clicking landmarks on a frame.

    ``click_noise_px`` is the interesting knob: it is how precisely a human hits a
    painted intersection on a 720p frame, and turning it up is how you find out how
    much of the end-to-end error is the human rather than the algorithm. Two or three
    pixels is realistic; zero is the ideal the maths deserves to be tested against.
    """
    rng = np.random.default_rng(seed)
    cat = catalog(spec)
    cors = []
    for lid in landmark_ids:
        lm = cat[lid]
        px = camera.project([[lm.x, lm.y, 0.0]])[0]
        if not np.isfinite(px).all():
            continue
        if click_noise_px:
            px = px + rng.normal(0.0, click_noise_px, size=2)
        cors.append(Correspondence(lid, (float(px[0]), float(px[1]))))
    return Anchor(frame_index=frame, correspondences=cors, spec=spec)


def synthetic_roundtrip(
    truth: PlayTrack,
    fps: float = 30.0,
    image_size: tuple[int, int] = (1280, 720),
    camera=None,
    noise: NoiseModel | None = None,
    click_noise_px: float = 0.0,
    box_kind: str = "body",
    landmark_ids=None,
    labelled: bool = True,
    spec: FieldSpec = NFL,
    config: PipelineConfig | None = None,
) -> RoundTrip:
    """Film a known play, recover it, and report how far off the recovery was.

    ``labelled`` decides whether the detector tells the tracker who each box is. With
    it on, association is free and the number measures registration, resampling and
    smoothing alone. With it off, the tracker has to work identities out for itself
    and the difference between the two runs is exactly what association costs — which
    is the single most useful comparison this harness produces.
    """
    cam = camera or frame_the_play(truth, image_size=image_size)

    # The play is resampled to the camera's frame rate first: 10 Hz truth filmed at
    # 30 fps is what a real capture looks like, and giving the tracker the contract's
    # own grid would flatter it.
    filmed = _upsample(truth, fps)

    detector = SyntheticDetector(
        truth=filmed, camera=cam, image_size=image_size,
        kind=box_kind, noise=noise or NoiseModel(), label_boxes=labelled,
    )

    ids = landmark_ids or visible_landmarks(cam, image_size, spec=spec)
    if not set(ids) <= set(catalog(spec)):
        raise KeyError(f"unknown landmark in {ids}")
    anchor = clicked_anchor(
        cam, landmark_ids=ids, click_noise_px=click_noise_px, spec=spec
    )
    reg = PointClickRegistration([anchor], spec=spec, image_size=image_size)
    quality = reg.assess()

    snap_index = _snap_index(filmed)
    cfg = config or PipelineConfig(fps=fps, spec=spec, image_size=image_size, snap_frame=snap_index)
    cfg.fps, cfg.image_size, cfg.snap_frame = fps, image_size, snap_index
    # Labels are only worth supplying if the tracker is allowed to associate on them;
    # otherwise they name tracks after the fact and change nothing, which is what
    # made the first version of the ablation report two identical rows.
    cfg.oracle_association = labelled

    result = run(detector, reg, n_frames=len(filmed.frames), config=cfg)
    sc = score(result.track, truth, play=str(truth.meta.get("play_id", "synthetic")))

    if quality.warnings:
        sc.notes.extend(f"registration: {w}" for w in quality.warnings)

    return RoundTrip(
        score=sc, result=result, truth=truth,
        registration_rms_yards=quality.rms_yards,
    )


# --------------------------------------------------------------------- helpers


def _snap_index(filmed: PlayTrack) -> int:
    for i, f in enumerate(filmed.frames):
        if "ball_snap" in f.events:
            return i
    return int(np.argmin([abs(f.t) for f in filmed.frames])) if filmed.frames else 0


def _upsample(track: PlayTrack, fps: float) -> PlayTrack:
    """Interpolate a 10 Hz release up to the camera's frame rate.

    Only positions and the snap event are carried; the filmed copy exists solely to
    be projected into boxes, and a synthetic detector has no use for a speed column.
    """
    from ..contract import PlayerFrame, TrackingFrame

    if not track.frames:
        return track
    t0, t1 = track.frames[0].t, track.frames[-1].t
    grid = np.round(np.arange(t0, t1 + 1e-9, 1.0 / fps), 4)
    src_t = np.array([f.t for f in track.frames])

    # Each player's series is built once and interpolated as a whole; rebuilding it
    # per output frame turns a linear job into a quadratic one, and a play is a few
    # thousand samples before it is anything else.
    interp: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for pid in track.roster:
        series = track.series(pid)
        if len(series) < 2:
            continue
        interp[pid] = (
            np.array([s[0] for s in series]),
            np.array([s[1].x for s in series]),
            np.array([s[1].y for s in series]),
        )

    frames = []
    for t in grid:
        players = {}
        for pid, (st, sx, sy) in interp.items():
            if t < st[0] or t > st[-1]:
                continue
            players[pid] = PlayerFrame(
                x=float(np.interp(t, st, sx)), y=float(np.interp(t, st, sy))
            )
        frames.append(TrackingFrame(t=float(t), players=players))

    snap_t = track.event_time("ball_snap")
    if snap_t is not None and frames:
        k = int(np.argmin(np.abs(grid - snap_t)))
        frames[k].events.append("ball_snap")

    return PlayTrack(meta=dict(track.meta), roster=dict(track.roster), frames=frames)
