"""One clip, described as data.

The Northwestern-Michigan run that produced the first side-by-side existed only as a
session of typed-in Python. It scored 90.3% and nobody could run it again, which makes
the number an anecdote rather than a measurement. This module is the fix: everything
that decides what comes out — which frames, which detector settings, how the field was
registered, how the teams were split — lives in a JSON file under ``data/clips/specs``,
and improving a clip means editing that file rather than editing the pipeline.

The division is the point. The *code* is what runs on every video; the *spec* is what
differs between them. When a clip renders badly, the first question is which of the two
is at fault, and that question is only answerable if they are separate.

Two fields deserve explanation.

``homography`` pins a registration that has already been checked, so the vision model is
asked once per clip rather than once per run. A registration is expensive, slightly
non-deterministic, and the thing most likely to be wrong; freezing a good one is how a
rendering loop stays fast and how a regression becomes visible instead of being absorbed
by a re-roll.

``snap_frame`` is a source frame index, not a window offset. Every time in the contract
is measured from the snap, so a spec that gets this wrong produces a play whose clock is
silently shifted and whose events all agree with each other about the wrong instant.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import asdict, dataclass, field as dc_field
from pathlib import Path
from typing import Any

SPEC_DIR = Path("data/clips/specs")
RUN_DIR = Path("data/runs")


@dataclass
class DetectSpec:
    """How to find people in a frame."""

    weights: str = "yolo11x.pt"
    conf: float = 0.25
    imgsz: int = 1280
    device: str | None = None
    max_players: int | None = None
    field_margin: float = 4.0
    tiles: int = 1
    """Split each frame into an NxN grid and detect in each tile as well as whole.

    A broadcast wide shot puts a far-hash player across about twenty pixels, and a
    detector run on the downscaled whole frame simply does not see him. Running the same
    model over overlapping crops costs N² times as much and is the single change that
    moves the unclaimed-detections count, which is the largest gap between the
    recreation and the clip.
    """

    tile_overlap: float = 0.2
    sideline_cut: bool = True
    """Find where the crowd starts from the detections themselves, instead of trusting
    the paint's own sideline. See :func:`~gridiron_tracking.render.sideline_cuts` — a
    bench is fifteen detections a frame on this clip, and a margin cannot remove it."""

    merge_iou: float = 0.55
    """How much two boxes must overlap before they are judged the same person. Tiled
    inference finds the same player twice wherever crops overlap; this is what removes
    the duplicate rather than letting it become a second track."""


@dataclass
class RegisterSpec:
    """How pixels become yards."""

    mode: str = "auto"
    """``auto`` asks the vision model, ``pinned`` uses ``homography`` verbatim."""

    frame: int | None = None
    """Which source frame to read the field from. Defaults to the middle of the window,
    which is usually the cleanest look at the paint."""

    level: str | None = None
    """``nfl`` / ``college`` / ``high_school``, when it is known. The model is asked
    anyway and the answer is compared, because hash spacing differs by nearly six yards
    between levels and a wrong one moves every player sideways."""

    homography: list[list[float]] | None = None
    model: str | None = None
    propagate: bool = True
    """Carry the anchor's map across the clip by measuring the camera's own motion.

    Every real clip pans — four freely-licensed game videos were checked and all four
    tracked the ball carrier. One homography held over a panning shot is not an
    approximation, it is wrong everywhere except the frame it was fitted on, and it
    shows up as track fragmentation rather than as an obvious error: a player standing
    still slides across the field as the camera moves, the gate breaks, and one man
    becomes six."""

    min_paint: float = 1.6
    """How much more paint the drawn yard lines must find than empty turf before a fit is
    believed. Around 1.0 the lines are landing on grass at chance — which is what the
    fit that scored 90.3% on its own reprojection was doing."""

    resnap: bool = True
    """After propagating, pull each frame's map back onto that frame's own painted lines.
    Propagation drifts — 3.53 alignment at the anchor down to 1.35 five seconds later on
    the reference clip — and the paint is there on every frame to correct it."""

    min_inliers: int = 25
    horizon_frac: float = 0.35
    """Ignore this fraction of the image from the top when measuring camera motion. The
    crowd is the other big moving thing in frame and none of its motion is the
    camera's."""
    scan_frames: int = 8
    """How many frames across the window to try when ``candidate_frames`` is empty.
    Registration is the stage that fails, it fails on the picture rather than on the
    clip, and the next second of footage is usually a different picture."""

    candidate_frames: list[int] = dc_field(default_factory=list)
    """Frames to try when ``auto`` fails on the first one. A broadcast cut, a replay
    wipe or a shot of the crowd is unreadable; the next second usually is not."""


@dataclass
class TrackSpec:
    """How detections become players."""

    gate_yards: float = 2.5
    max_age: int = 8
    min_hits: int = 3
    process_noise: float = 6.0
    measurement_noise: float = 0.35
    smooth_window: int = 7
    keep_unlabelled: bool = True
    teams: bool = True
    chroma_only: bool = False
    """Cluster jersey colour with the lightness channel dropped.

    It was believed this would fix a team split that was following the sun rather than
    the kit, and it does the opposite: measured over 1034 torso samples on the reference
    clip, full Lab separates the two teams by 90.8 units and chroma alone by 16.7. White
    against purple is almost entirely a lightness difference, so dropping that channel
    throws away the signal. Left here as an option because a clip with two similarly
    bright kits may still want it — but it is off, and turning it on should be justified
    by a separation number on that clip, not by the argument."""

    team_clusters: int = 2
    """How many kits to cluster into. Three was used to give the officials a home; it
    splits one team in two instead — 90.8 units of separation at k=2, 69.2 at k=3 on the
    same samples. The officials cost less as a handful of misassigned tracks than a
    halved team does."""

    team_cost: float = 0.0
    """Yards of penalty for matching a detection to a track of the other team. Needs
    ``pre_teams``; without a team on the detections there is nothing to disagree with."""

    pre_teams: bool = False
    """Cluster jersey colour before tracking as well as after, so the tracker can use it
    to tell two crossing players apart."""

    stitch: bool = True
    """Rejoin tracks that are one player, after the play. See
    :mod:`~gridiron_tracking.stitch`; the online tracker cannot do this and parameter
    tuning does not substitute for it."""

    stitch_gap_s: float = 1.2
    stitch_gate: float = 3.5

    drop_static_sideline: bool = True
    """Drop tracks that never move and sit on a sideline.

    The bench is the largest source of phantom players, and geometry cannot remove it:
    measured on the reference frame, people standing behind the far sideline project to
    field y between 51 and 54 — inside the field, because their feet are hidden and the
    detector's box bottom lands on the chalk. Their apparent height does not give them
    away either; at that depth a standing coach and a cornerback are the same number of
    pixels tall, 0.92 against 0.95 of the height the geometry predicts.

    What separates them is that they are not playing. Over five seconds a player covers
    yards and a man on the bench covers inches, so a track that stays within a yard of
    where it started, within two yards of a sideline, is furniture."""

    static_yards: float = 1.0
    sideline_band: float = 2.0

    ball: bool = True
    ball_tiles: int = 3
    """Crops per side for ball detection, as ``detect.tiles`` is for players.

    A football in a broadcast wide shot is a handful of pixels, and after the
    letterboxing ``imgsz`` does it is fewer. The same argument that makes tiling
    work for a far-hash player applies to the ball several times over, so this
    defaults high where the player tiling defaults to 1."""
    ball_conf: float = 0.05
    """Detection floor for the ball. Lower than the players': a missed ball costs
    the throw instant and there is a tracker downstream whose whole job is to throw
    out the ones that do not move like a ball."""
    ball_imgsz: int = 1536
    ball_weights: str | None = None
    """Weights for the ball, when they are not the player detector's.

    ``None`` uses ``detect.weights`` and COCO's generic sports-ball class, which
    cannot find a football in broadcast footage — see the package README. Point this
    at the output of ``train-ball`` once there is one."""
    ball_class: int = 32
    """Class index to keep. 32 is COCO's sports ball; a detector fine-tuned by
    ``train-ball`` has one class and it is 0."""


@dataclass
class ClipSpec:
    """Everything needed to turn one video into one side-by-side."""

    name: str
    video: str
    start: int = 0
    stop: int | None = None
    step: int = 1
    """Decimation. **Only 1 works** — the renderer refuses anything else, in a
    second rather than after the detector has run.

    It is kept at 1 rather than removed because everything downstream turns a
    contract time into a source frame with ``t * fps + snap``, which is only a
    frame index when every frame in the window was decoded. The temptation is a 60
    fps clip, where decoding every other frame would halve the detection time for
    output that is resampled to 10 Hz anyway; doing that properly means carrying
    the decimation through the registration and the verifier, which is a real
    change and not a spec field."""
    snap_frame: int | None = None
    fps: float | None = None
    out_fps: int = 12
    distance_yards: float | None = None
    """The distance the down-and-distance graphic states, e.g. 6 for "2nd & 6".

    The only external ground truth a broadcast clip carries. See
    :mod:`~gridiron_tracking.broadcast`: the virtual line of scrimmage and the virtual
    first-down line are this far apart, so measuring them through the registration checks
    the scale in yards against somebody else's chain."""

    title: str = ""
    notes: str = ""
    detect: DetectSpec = dc_field(default_factory=DetectSpec)
    register: RegisterSpec = dc_field(default_factory=RegisterSpec)
    track: TrackSpec = dc_field(default_factory=TrackSpec)

    # ------------------------------------------------------------------ io

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "ClipSpec":
        d = dict(d)
        return cls(
            detect=DetectSpec(**d.pop("detect", {}) or {}),
            register=RegisterSpec(**d.pop("register", {}) or {}),
            track=TrackSpec(**d.pop("track", {}) or {}),
            **d,
        )

    def to_json(self) -> dict[str, Any]:
        return asdict(self)

    @classmethod
    def load(cls, name_or_path: str | Path, spec_dir: Path = SPEC_DIR) -> "ClipSpec":
        p = Path(name_or_path)
        if not p.exists():
            p = Path(spec_dir) / f"{name_or_path}.json"
        if not p.exists():
            raise FileNotFoundError(
                f"no clip spec {name_or_path!r}; looked in {spec_dir}. "
                f"Known clips: {', '.join(available(spec_dir)) or '(none)'}"
            )
        return cls.from_json(json.loads(p.read_text()))

    def save(self, spec_dir: Path = SPEC_DIR) -> Path:
        p = Path(spec_dir) / f"{self.name}.json"
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(self.to_json(), indent=2) + "\n")
        return p

    # --------------------------------------------------------------- windows

    def window(self, n_frames: int) -> tuple[int, int]:
        stop = self.stop if self.stop is not None else n_frames
        return self.start, min(stop, n_frames or stop)

    def snap(self, n_frames: int) -> int:
        """Snap as an index into the *window*, which is what the pipeline wants."""
        start, _ = self.window(n_frames)
        return 0 if self.snap_frame is None else max(0, self.snap_frame - start)

    def read_frame(self, n_frames: int) -> int:
        start, stop = self.window(n_frames)
        return self.register.frame if self.register.frame is not None else (start + stop) // 2

    # ----------------------------------------------------------------- keys

    def key(self, stage: str) -> str:
        """A short hash of exactly what this stage depends on.

        Stage caching is worth nothing if the key is the whole spec: changing a
        rendering colour would then re-run the detector for twenty minutes. Each stage
        names its own inputs and inherits the ones before it, so editing the tracker
        re-tracks without re-detecting, and editing the detector re-does both.
        """
        parts: list[Any] = [self.video, self.start, self.stop, self.step]
        if stage in ("detect", "register", "track", "verify"):
            parts.append(asdict(self.detect) if stage != "register" else None)
        if stage in ("register", "track", "verify"):
            parts.append(asdict(self.register))
        if stage in ("track", "verify"):
            parts.append([asdict(self.track), self.snap_frame, self.fps])
        blob = json.dumps(parts, sort_keys=True, default=str)
        return hashlib.sha1(blob.encode()).hexdigest()[:12]

    def run_dir(self, root: Path = RUN_DIR) -> Path:
        return Path(root) / self.name


def available(spec_dir: Path = SPEC_DIR) -> list[str]:
    d = Path(spec_dir)
    return sorted(p.stem for p in d.glob("*.json")) if d.is_dir() else []


def load_all(spec_dir: Path = SPEC_DIR) -> list[ClipSpec]:
    return [ClipSpec.load(n, spec_dir) for n in available(spec_dir)]
