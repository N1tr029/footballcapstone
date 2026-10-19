"""Continuous footage in, candidate plays out.

Everything else in this package starts from a clip somebody already chose: a spec names
a video, a first frame and a last frame, and the pipeline runs between them. That is the
right shape for measuring accuracy and the wrong shape for footage, because nobody hands
you a five-second window — they hand you a quarter, and a quarter is mostly not football.

Two facts decide how this has to work.

**The footage cannot be held in memory.** :func:`~gridiron_tracking.render._decode`
builds ``dict[int, np.ndarray]``, which is fine for the 150 frames of a reference clip
(414 MB) and impossible for the ~70,000 frames of a quarter (193 GB). So this pass holds
one frame at a time, reduces it to a few numbers, and throws the pixels away. Everything
here is O(1) in memory and linear in runtime no matter how long the input is.

**A broadcast is a sequence of shots, not a video.** The camera cuts — to a replay, to
the coach, to the crowd, to a graphic — and a homography propagated across a cut is not
drifting, it is describing a camera that no longer exists. Registration therefore has to
be per shot, which means the shots have to be found first. This is the step that was
missing, and without it a quarter is not one hard problem but several hundred.

What this does *not* do is decide which shots are plays. It scores them, with features
that are named and measurable, and leaves the call to whoever reads the list — the same
arrangement as registration, for the same reason: a confident wrong answer about which
shots matter is more expensive than an honest shortlist.

The turf fraction is the feature that earns its place. A live play is framed on grass
and almost nothing else is: a crowd shot is a wall of people, a coach is a close-up, a
graphic is flat colour. It does not separate a live play from a *replay* of one — both
are grass — which is why the score is a shortlist and not a verdict.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np

# OpenCV hue is 0-179, so green turf sits around 35-85. The saturation and value floors
# matter more than the hue window: they are what reject the grey of a stadium roof and
# the near-black of a shadowed tunnel, both of which can land in the green hue bin when
# they are almost colourless and the hue is therefore meaningless.
TURF_HUE = (30, 90)
TURF_SAT_MIN = 40
TURF_VAL_MIN = 40

# A cut changes the whole frame at once. Measured on the two FOX reels, within-shot
# histogram distance sits well under 0.2 even through a hard pan, and a cut lands above
# 0.5; 0.35 is the gap between them. A *dissolve* spreads the change over a dozen frames
# and will be missed — it reads as motion, not as a cut. Dissolves into replays are the
# known hole here and they are reported rather than papered over.
CUT_DISTANCE = 0.35

# Below this a "shot" is a flash frame, a graphic sting, or the tail of a dissolve.
MIN_SHOT_FRAMES = 8


@dataclass
class Shot:
    """One continuous camera, described by numbers cheap enough to compute streaming."""

    index: int
    start_frame: int
    end_frame: int          # inclusive
    fps: float

    turf: float = 0.0       # mean fraction of the frame that is grass
    turf_sd: float = 0.0    # how much that wandered; a pan across a field is steady
    grid: float = 0.0       # of 16 cells, the fraction that are majority turf
    blob: float = 0.0       # largest single non-turf object, as a fraction of the frame
    motion: float = 0.0     # mean absolute inter-frame change, 0-1
    brightness: float = 0.0
    cut_in: float = 0.0     # how hard the cut into this shot was

    @property
    def n_frames(self) -> int:
        return self.end_frame - self.start_frame + 1

    @property
    def seconds(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    @property
    def start_t(self) -> float:
        return self.start_frame / self.fps if self.fps else 0.0

    def playlike(self) -> float:
        """How much this looks like a wide shot of live football, 0 to 1.

        Not the turf fraction, which was the obvious feature and the wrong one. A
        close-up of a quarterback is half grass — it is all background — so it scored
        0.65 here, above a real play, while a genuine 20-second play scored 0.00 because
        a length veto cut it off. Both failures were found by looking at the frames
        rather than at the numbers, which is the only way either would have surfaced.

        What separates a wide field shot is that the grass is *everywhere* and nothing
        large is in front of it:

        ``grid``  of sixteen cells, how many are mostly turf. A close-up has grass in
                  the background cells only; a wide shot has it in nearly all of them.
        ``blob``  the largest single non-turf object. A torso filling the frame is one
                  big blob; twenty-two players at distance are twenty-two small ones.

        Length is now a floor, not a veto. A play is at least a couple of seconds, but
        the upper bound was wrong: broadcasts hold a wide shot through the tackle and
        the aftermath, and a 20-second shot is routinely one play.

        Calibrated against six frames the author labelled by eye. That is enough to
        place a threshold and nowhere near enough to trust the exact value of one — the
        margins below are a frame or two wide. It is a shortlist, not a verdict.

        It still cannot separate a live play from a replay of one. Both are wide, both
        are grass, and what distinguishes them is a graphic wipe and a frame rate that
        are not in these features. Expect replays in the hits.
        """
        spread = np.clip((self.grid - 0.55) / 0.20, 0.0, 1.0)
        clear = np.clip((0.45 - self.blob) / 0.20, 0.0, 1.0)
        moving = np.clip(self.motion / 0.04, 0.0, 1.0)
        # Long is fine; a play plus its aftermath runs 20s. A minute is a stoppage.
        length = np.clip((self.seconds - 2.0) / 2.0, 0.0, 1.0) * np.clip(
            (60.0 - self.seconds) / 20.0, 0.0, 1.0)
        # Geometric mean, not a product. The four are each a veto — a zero anywhere is
        # still a zero — but multiplying them outright compounds mediocrity: a genuine
        # play that is merely acceptable on three features scored 0.5 x 0.5 x 0.5 = 0.12
        # and was thrown away. The root keeps "any zero kills it" while letting a shot
        # that is middling everywhere score middling rather than near nothing.
        parts = np.array([spread, clear, moving, length], dtype=float)
        if (parts <= 0).any():
            return 0.0
        return float(np.exp(np.log(parts).mean()))

    def why(self) -> str:
        return (f"turf {self.turf:.2f}  grid {self.grid:.2f}  blob {self.blob:.2f}  "
                f"{self.seconds:4.1f}s  motion {self.motion:.3f}  score {self.playlike():.2f}")


def _descriptors(path: str | Path, stride: int = 1) -> Iterator[tuple[int, np.ndarray, float, float, float]]:
    """Frame index, hue histogram, turf fraction, brightness, motion — one frame at a time.

    The frame itself never leaves this function. That is the whole design constraint.
    """
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - depends on the install
        raise ImportError("shot segmentation needs opencv: pip install opencv-python") from e

    cap = cv2.VideoCapture(str(path))
    if not cap.isOpened():
        raise FileNotFoundError(f"cannot open {path}")
    prev_small = None
    i = -1
    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                break
            i += 1
            if stride > 1 and i % stride:
                continue
            small = cv2.resize(frame, (160, 90), interpolation=cv2.INTER_AREA)
            hsv = cv2.cvtColor(small, cv2.COLOR_BGR2HSV)
            h, s, v = hsv[..., 0], hsv[..., 1], hsv[..., 2]

            is_turf = ((h >= TURF_HUE[0]) & (h <= TURF_HUE[1])
                       & (s >= TURF_SAT_MIN) & (v >= TURF_VAL_MIN))
            turf = float(is_turf.mean())
            cells = [is_turf[r * 22:(r + 1) * 22, c * 40:(c + 1) * 40].mean()
                     for r in range(4) for c in range(4)]
            grid = float(np.mean([c > 0.5 for c in cells]))
            n_lab, _, stats, _ = cv2.connectedComponentsWithStats(
                (~is_turf).astype(np.uint8), 8)
            blob = (float(stats[1:, cv2.CC_STAT_AREA].max() / is_turf.size)
                    if n_lab > 1 else 0.0)
            hist = cv2.calcHist([hsv], [0, 1], None, [24, 8], [0, 180, 0, 256])
            hist = (hist / (hist.sum() or 1.0)).flatten()
            bright = float(np.mean(v) / 255.0)
            grey = small.mean(axis=2)
            motion = 0.0 if prev_small is None else float(np.mean(np.abs(grey - prev_small)) / 255.0)
            prev_small = grey
            yield i, hist, turf, bright, motion, grid, blob
    finally:
        cap.release()


def segment(path: str | Path, fps: float | None = None,
            cut_distance: float = CUT_DISTANCE,
            min_frames: int = MIN_SHOT_FRAMES) -> list[Shot]:
    """Split continuous footage into shots. One pass, constant memory."""
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise ImportError("shot segmentation needs opencv") from e

    if fps is None:
        cap = cv2.VideoCapture(str(path))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        cap.release()

    shots: list[Shot] = []
    cur: list[tuple[float, float, float, float, float]] = []  # turf, bright, motion, grid, blob
    start = 0
    prev_hist = None
    cut_in = 0.0
    last_i = 0

    def close(end_i: int, next_cut: float) -> None:
        nonlocal cur, start, cut_in
        if cur and (end_i - start + 1) >= min_frames:
            a = np.array(cur, dtype=float)
            shots.append(Shot(
                index=len(shots), start_frame=start, end_frame=end_i, fps=fps,
                turf=float(a[:, 0].mean()), turf_sd=float(a[:, 0].std()),
                grid=float(a[:, 3].mean()), blob=float(a[:, 4].mean()),
                brightness=float(a[:, 1].mean()),
                # the first frame of a shot has no predecessor inside it
                motion=float(a[1:, 2].mean()) if len(a) > 1 else 0.0,
                cut_in=cut_in,
            ))
        cur = []
        start = end_i + 1
        cut_in = next_cut

    for i, hist, turf, bright, motion, grid, blob in _descriptors(path):
        last_i = i
        if prev_hist is not None:
            # Bhattacharyya: 0 identical, 1 disjoint. Robust to the brightness swings a
            # correlation metric reads as a cut.
            d = float(cv2.compareHist(prev_hist.astype("float32"),
                                      hist.astype("float32"), cv2.HISTCMP_BHATTACHARYYA))
            if d >= cut_distance:
                close(i - 1, d)
        prev_hist = hist
        cur.append((turf, bright, motion, grid, blob))
    close(last_i, 0.0)
    return shots


def census(shots: list[Shot], threshold: float = 0.35) -> str:
    """What a piece of footage actually contains, said in one block."""
    if not shots:
        return "no shots found"
    total = sum(s.seconds for s in shots)
    hits = [s for s in shots if s.playlike() >= threshold]
    play_t = sum(s.seconds for s in hits)
    lines = [
        f"  shots              {len(shots)} over {total / 60:.1f} min",
        f"  median shot        {np.median([s.seconds for s in shots]):.1f}s",
        f"  candidate plays    {len(hits)}  ({play_t / 60:.1f} min, "
        f"{play_t / total * 100:.0f}% of the footage)",
        f"  on grass at all    {sum(1 for s in shots if s.turf > 0.4)} shots",
        "",
        "  best candidates:",
    ]
    for s in sorted(hits, key=lambda s: -s.playlike())[:12]:
        lines.append(f"    {s.start_t / 60:5.2f}min  frames {s.start_frame}-{s.end_frame}  {s.why()}")
    return "\n".join(lines)
