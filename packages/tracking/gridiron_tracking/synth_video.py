"""Render a known play as an actual video file.

Not a demo. This is how the video path gets tested before any football footage
exists: take real tracking, film it with a virtual camera, write real pixels to an
mp4, and then run the *real* pipeline over that file — decode, register, track,
export. Every stage is exercised against a frame buffer instead of against a list of
numbers, and because the tracking that went in is known exactly, the tracking that
comes out can be scored in yards.

What it does not test is the detector. Painted figures are not photographs, and a
model's behaviour on them says nothing about its behaviour on a Friday night in
November. Detection quality is the one thing that genuinely requires real footage —
everything downstream of it can be, and now is, proven without any.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from .contract import PlayTrack
from .detect.synthetic import sideline_camera
from .field import NFL, FieldSpec, catalog

# BGR, because OpenCV.
TURF = (48, 92, 40)
TURF_DARK = (40, 78, 34)
PAINT = (232, 238, 236)
SKY = (120, 104, 88)


def _cv2():
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise ImportError("rendering video needs opencv: pip install opencv-python") from e
    return cv2


def _draw_field(cv2, img, cam, spec: FieldSpec) -> None:
    """Paint the turf and the lines, in perspective, from the field model."""
    h, w = img.shape[:2]
    img[:] = SKY

    # Turf as one filled quad through the camera.
    corners = cam.project([[0, 0, 0], [spec.length, 0, 0], [spec.length, spec.width, 0], [0, spec.width, 0]])
    if np.isfinite(corners).all():
        cv2.fillConvexPoly(img, corners.astype(np.int32), TURF)

    # Alternating five-yard bands, so the turf is not a flat colour a detector can
    # trivially key on and so a human can see the perspective.
    x = spec.goal_a
    band = False
    while x < spec.goal_b:
        x2 = min(x + 5.0, spec.goal_b)
        quad = cam.project([[x, 0, 0], [x2, 0, 0], [x2, spec.width, 0], [x, spec.width, 0]])
        if band and np.isfinite(quad).all():
            cv2.fillConvexPoly(img, quad.astype(np.int32), TURF_DARK)
        band = not band
        x = x2

    def line(p0, p1, thick=2):
        a = cam.project([p0])[0]
        b = cam.project([p1])[0]
        if np.isfinite(a).all() and np.isfinite(b).all():
            cv2.line(img, tuple(a.astype(int)), tuple(b.astype(int)), PAINT, thick, cv2.LINE_AA)

    for xv in np.arange(spec.goal_a, spec.goal_b + 0.01, 5.0):
        line([xv, 0, 0], [xv, spec.width, 0], 3 if abs(xv % 10 - (spec.goal_a % 10)) < 0.01 else 2)
    line([0, 0, 0], [spec.length, 0, 0], 3)
    line([0, spec.width, 0], [spec.length, spec.width, 0], 3)
    line([spec.goal_a, 0, 0], [spec.goal_a, spec.width, 0], 4)
    line([spec.goal_b, 0, 0], [spec.goal_b, spec.width, 0], 4)

    lo, hi = spec.hashes
    for xv in np.arange(spec.goal_a + 1, spec.goal_b, 1.0):
        for hy in (lo, hi):
            line([xv, hy - 0.35, 0], [xv, hy + 0.35, 0], 2)


def _draw_player(cv2, img, cam, x: float, y: float, colour, jersey: str | None = None) -> None:
    """A standing figure: legs, torso, head, drawn to his real projected height.

    Crude on purpose. The figure exists so that decoding, registration and tracking
    have something to find at a known place — not so that a detector trained on
    photographs will recognise it.
    """
    foot = cam.project([[x, y, 0.0]])[0]
    head = cam.project([[x, y, 2.05]])[0]
    hip = cam.project([[x, y, 1.05]])[0]
    shoulder = cam.project([[x, y, 1.75]])[0]
    if not all(np.isfinite(p).all() for p in (foot, head, hip, shoulder)):
        return

    px_h = abs(float(foot[1] - head[1]))
    if px_h < 4:
        return
    half = max(1, int(px_h * 0.16))

    f = tuple(foot.astype(int))
    hp = tuple(hip.astype(int))
    sh = tuple(shoulder.astype(int))

    cv2.line(img, (f[0] - half // 2, f[1]), hp, (28, 28, 32), max(2, half // 2), cv2.LINE_AA)
    cv2.line(img, (f[0] + half // 2, f[1]), hp, (28, 28, 32), max(2, half // 2), cv2.LINE_AA)
    cv2.line(img, hp, sh, colour, max(3, half), cv2.LINE_AA)
    cv2.circle(img, (sh[0], sh[1] - max(2, int(px_h * 0.09))), max(2, int(px_h * 0.10)), colour, -1, cv2.LINE_AA)

    if jersey and px_h > 40:
        cv2.putText(img, jersey, (sh[0] - 8, (sh[1] + hp[1]) // 2),
                    cv2.FONT_HERSHEY_SIMPLEX, px_h / 140.0, PAINT, 1, cv2.LINE_AA)


def render(
    track: PlayTrack,
    out_path: str | Path,
    fps: float = 30.0,
    size: tuple[int, int] = (1280, 720),
    camera=None,
    spec: FieldSpec = NFL,
    offense_colour=(60, 70, 200),
    defense_colour=(190, 150, 60),
    sides=None,
) -> Path:
    """Write the play to an mp4, and return the path."""
    cv2 = _cv2()
    from .bench import frame_the_play, _upsample
    from .sides import infer_sides

    cam = camera or frame_the_play(track, image_size=size)
    s = sides or infer_sides(track)
    filmed = _upsample(track, fps)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    writer = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), fps, size)
    if not writer.isOpened():
        raise RuntimeError(f"could not open a writer for {out_path}")

    base = np.zeros((size[1], size[0], 3), dtype=np.uint8)
    _draw_field(cv2, base, cam, spec)

    try:
        for f in filmed.frames:
            img = base.copy()
            # Far players first, so nearer ones occlude them the way they really would.
            order = sorted(f.players.items(), key=lambda kv: -kv[1].y)
            for pid, p in order:
                side = s.team_of(track, pid)
                colour = defense_colour if side == "defense" else offense_colour
                entry = track.roster.get(pid)
                _draw_player(cv2, img, cam, p.x, p.y, colour,
                             str(entry.num) if entry and entry.num else None)
            if f.ball is not None:
                b = cam.project([[f.ball.x, f.ball.y, 1.0]])[0]
                if np.isfinite(b).all():
                    cv2.circle(img, tuple(b.astype(int)), 4, (40, 60, 140), -1, cv2.LINE_AA)
            writer.write(img)
    finally:
        writer.release()

    return out_path
