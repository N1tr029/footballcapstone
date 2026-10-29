"""One video in, one side-by-side out. The same command for every clip.

This is the loop the whole stream is tuned in. Point it at a clip spec and it decodes,
detects, registers, tracks, assigns teams, finds the ball, scores the result against
the clip's own pixels, and writes the two-panel video — the footage with every recovered
position drawn back onto it, above the top-down recreation of the same instant.

Three properties matter more than what it does.

**It always renders.** A stage that fails degrades rather than aborting: no registration
means no top-down panel, but the detections still draw, and a video you can look at beats
a traceback every time. The report says what was missing.

**It caches by stage, not by run.** Each stage's key covers exactly its own inputs, so
re-tracking after a tuning change is seconds and re-detecting is minutes, and the two do
not get confused. Detection on a 150-frame window with a tiled 'x' model is the expensive
step by an order of magnitude; nothing should re-run it because a colour changed.

**Every clip gets the same treatment.** The spec holds what differs between videos; the
code holds what does not. A change that helps Northwestern-Michigan and quietly breaks
the high-school clip shows up the moment ``--all`` is run, which is the only way a
pipeline that is meant to work on any upload ever gets there.
"""

from __future__ import annotations

import json
import re
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

import numpy as np

from . import ball as ballmod
from . import carrier as carriermod
from . import field as fieldmod
from . import verify as verifymod
from .clips import RUN_DIR, ClipSpec
from .contract import Ball, PlayTrack
from .detect import Detection
from .pipeline import PipelineConfig, run as run_pipeline
from .video import VideoSource

TEAM_COLORS = {"teamA": (59, 169, 242), "teamB": (232, 168, 79), "other": (150, 150, 150)}


@dataclass
class RenderResult:
    spec: ClipSpec
    out_dir: Path
    video: Path | None = None
    track: PlayTrack | None = None
    report: verifymod.VerifyReport | None = None
    registration: Any = None
    n_frames: int = 0
    n_boxes: int = 0
    team_separation: float = 0.0
    ball: Any = None
    markers: Any = None
    stages: dict[str, str] = dc_field(default_factory=dict)
    problems: list[str] = dc_field(default_factory=list)
    seconds: float = 0.0

    @property
    def agreement(self) -> float:
        return self.report.agreement if self.report else float("nan")

    def summary(self) -> str:
        s = self.spec
        head = f"{s.name}  ({s.title or Path(s.video).name})"
        lines = [head, "-" * len(head)]
        lines.append(f"  frames            {self.n_frames} decoded, {self.n_boxes} boxes "
                     f"({self.n_boxes / self.n_frames:.1f} per frame)" if self.n_frames else "  frames  none")
        if self.track is not None:
            ppf = np.median([len(f.players) for f in self.track.frames]) if self.track.frames else 0
            churn = len(self.track.roster) / ppf if ppf else float("nan")
            lines.append(f"  tracks            {len(self.track.roster)} for "
                         f"{len(self.track.frames)} output frames")
            # The number that says how badly identity is being lost. One man should leave
            # one track; anything above about 1.2 means the tracker is dropping people and
            # starting them again under a new name, which no amount of per-frame accuracy
            # fixes and which the agreement figure cannot see.
            lines.append(f"  people on field   {ppf:.0f} per frame, {churn:.1f} tracks each"
                         f"   (1.0 is one track per player)")
        if self.report is not None:
            r = self.report
            lines.append(f"  agreement         {r.agreement * 100:.1f}% of tracked players hold a person of their own")
            lines.append(f"  coverage          {np.mean([f.on_player for f in r.frames]):.1f} of 22 accounted for")
            lines.append(f"  duplicate / frame {np.mean([f.duplicate for f in r.frames]):.1f}")
            lines.append(f"  stray / frame     {np.mean([f.stray for f in r.frames]):.1f}")
            lines.append(f"  unclaimed / frame {np.mean([f.unclaimed for f in r.frames]):.1f}")
        if self.team_separation:
            lines.append(f"  team separation   {self.team_separation:.0f} Lab units")
        if self.ball is not None:
            lines.append(f"  {self.ball.summary()}")
        if self.markers is not None and self.markers.samples:
            m = self.markers
            lines.append(f"  yards, vs truth   {m.measured:.2f} measured against "
                         f"{m.stated_yards:.0f} stated by the broadcast "
                         f"({m.error_yards:+.2f} yd)")
        lines.append(f"  stages            " + ", ".join(f"{k}:{v}" for k, v in self.stages.items()))
        if self.video:
            lines.append(f"  side by side      {self.video}")
        for p in self.problems:
            lines.append(f"  ! {p}")
        lines.append(f"  took              {self.seconds:.0f}s")
        return "\n".join(lines)


# --------------------------------------------------------------------- stages


def _decode(spec: ClipSpec, log) -> tuple[dict[int, np.ndarray], float, int]:
    src = VideoSource(spec.video)
    fps = spec.fps or src.info.fps
    start, stop = spec.window(src.info.n_frames)
    frames: dict[int, np.ndarray] = {}
    for i, img in src.frames(start, stop, spec.step):
        frames[i] = img
    src.close()
    log(f"    decoded {len(frames)} frames [{start}:{stop}:{spec.step}] at {fps:.2f} fps")
    return frames, fps, src.info.n_frames


def _detect(spec: ClipSpec, frames: dict[int, np.ndarray], out_dir: Path, force: set, log):
    """Boxes per source frame, cached. The expensive stage, so it is cached hardest."""
    cache = out_dir / f"boxes-{spec.key('detect')}.json"
    if cache.exists() and "detect" not in force:
        raw = json.loads(cache.read_text())
        boxes = {int(k): [Detection(frame=int(k), **b) for b in v] for k, v in raw.items()}
        log(f"    detections: cached ({sum(len(v) for v in boxes.values())} boxes)")
        return boxes, "cached"

    from .detect.yolo import YoloDetector

    det = YoloDetector(
        weights=spec.detect.weights,
        conf=spec.detect.conf,
        imgsz=spec.detect.imgsz,
        device=spec.detect.device,
        max_players=spec.detect.max_players,
        tiles=spec.detect.tiles,
        tile_overlap=spec.detect.tile_overlap,
        merge_iou=spec.detect.merge_iou,
    )
    boxes: dict[int, list[Detection]] = {}
    t0 = time.time()
    for n, (i, img) in enumerate(sorted(frames.items())):
        boxes[i] = det.detect(i, img)
        if n and n % 25 == 0:
            log(f"      {n}/{len(frames)} frames, {sum(len(v) for v in boxes.values())} boxes, "
                f"{(time.time() - t0) / n:.2f}s/frame")
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        str(k): [{"x1": d.x1, "y1": d.y1, "x2": d.x2, "y2": d.y2, "score": d.score, "kind": d.kind}
                 for d in v]
        for k, v in boxes.items()
    }))
    log(f"    detections: {sum(len(v) for v in boxes.values())} boxes in {time.time() - t0:.0f}s")
    return boxes, "computed"


def _register(spec: ClipSpec, frames: dict[int, np.ndarray], n_src: int,
              out_dir: Path, force: set, log, boxes=None):
    """A homography per frame: one anchor, read off the paint, then carried by the camera.

    Two steps, and the second is the one that decides whether the output is usable. The
    anchor answers "where is the field" once, from a frame a vision model could read.
    Propagation answers "where is it now" on every other frame, by measuring the camera's
    own motion off the turf with the players masked out.

    Skipping the second step does not produce a slightly worse registration. It produces
    a stationary field under a moving camera, so a player standing still travels several
    yards a second, the tracker's gate breaks, and one man leaves six tracks behind him.
    """
    from .registration.homography import Homography
    from .registration import vision

    base, field_spec, stage, problems = _anchor(spec, frames, n_src, out_dir, force, log)
    if base is None or not spec.register.propagate:
        return base, field_spec, stage, problems, None

    from .registration.propagate import PropagatedRegistration

    ordered = sorted(frames)
    anchor = spec.read_frame(n_src)
    anchor = min(ordered, key=lambda k: abs(k - anchor))
    any_frame = frames[ordered[0]]
    size = (any_frame.shape[1], any_frame.shape[0])

    # The players are masked out of the motion estimate. Twenty-two men running is the
    # most salient motion in the frame and none of it belongs to the camera; without the
    # mask the solved "camera" follows the ball carrier.
    mask = {k: [(d.x1, d.y1, d.x2, d.y2) for d in v] for k, v in (boxes or {}).items()}
    prop = PropagatedRegistration(
        base=base, anchor_frame=anchor, spec=field_spec, image_size=size,
        min_inliers=spec.register.min_inliers,
        horizon_frac=spec.register.horizon_frac,
        mask_boxes=mask,
    )
    rep = prop.build(_FrameBag(frames), ordered[0], ordered[-1] + 1, spec.step)
    log(f"    propagation: {rep.summary()}")
    if spec.register.resnap:
        prop.resnap(frames, field_spec, log=log)
    if rep.coverage < 0.9:
        problems.append(f"camera motion only solved on {rep.coverage * 100:.0f}% of frames")
    return base, field_spec, stage, problems, prop


def _scan_frames(spec: ClipSpec, frames, n_src: int) -> list[int]:
    """Frames to try registering on, best guess first, spread across the window."""
    ordered = sorted(frames)
    if not ordered:
        return []
    mid = spec.read_frame(n_src)
    rest = ordered[::max(1, len(ordered) // spec.register.scan_frames)]
    seen, out = set(), []
    for f in [min(ordered, key=lambda k: abs(k - mid)), *rest]:
        if f not in seen:
            seen.add(f)
            out.append(f)
    return out[: spec.register.scan_frames + 1]


def _anchor(spec: ClipSpec, frames, n_src: int, out_dir: Path, force: set, log):
    """The one frame whose field position is established, and how."""
    from .registration.homography import Homography
    from .registration import vision

    if spec.register.mode == "pinned" and spec.register.homography:
        H = Homography(np.array(spec.register.homography, dtype=float))
        sp = vision.spec_for(spec.register.level or "unclear")
        log(f"    anchor: pinned ({sp.name}, frame {spec.read_frame(n_src)})")
        return H, sp, "pinned", []

    cache = out_dir / f"registration-{spec.key('register')}.json"
    if cache.exists() and "register" not in force:
        d = json.loads(cache.read_text())
        H = Homography(np.array(d["H"], dtype=float))
        sp = vision.spec_for(d.get("level", "unclear"))
        log(f"    anchor: cached ({sp.name}, frame {d.get('frame')})")
        return H, sp, "cached", list(d.get("problems", []))

    # Several frames, and the best one wins on a number rather than on the first that
    # does not obviously fail. A reading varies run to run — the same frame gave 14, 19
    # and 20 correspondences on three consecutive asks — so "the first fit that passed"
    # is a coin toss dressed as a decision. Paint alignment is an objective score that
    # needs no ground truth, so the loop can just take the highest.
    # Scan the window rather than trusting one frame. A broadcast clip is a sequence of
    # very different pictures — a wide shot with four numbers legible, a tight shot with
    # none, a replay wipe — and which of them the middle of the window happens to land on
    # is luck. Reading several and keeping the best by paint alignment turns that luck
    # into a search, and it is cached, so a clip pays for it once.
    wanted = spec.register.candidate_frames or _scan_frames(spec, frames, n_src)
    tried: list[str] = []
    best = None
    for fi in wanted:
        img = frames.get(fi)
        if img is None:
            if not frames:
                break
            fi = min(frames, key=lambda k: abs(k - fi))
            img = frames[fi]
        log(f"    anchor: reading frame {fi} with the vision model...")
        kw = {"model": spec.register.model} if spec.register.model else {}
        auto = vision.auto_register(img, verify=False, **kw)
        log(f"      {auto.summary()}")
        if auto.homography is None:
            tried.append(f"frame {fi}: {auto.summary()}")
            continue
        score = auto.paint
        if score != score:
            score = -1.0
        if best is None or score > best[0]:
            best = (score, auto, fi)
        if score >= spec.register.min_paint:
            break

    if best is None or best[0] < spec.register.min_paint:
        if best is not None:
            tried.append(f"best attempt (frame {best[2]}) aligned to the paint at only "
                         f"{best[0]:.2f}x; {spec.register.min_paint:.2f} is the bar")
        return None, fieldmod.NFL, "failed", tried

    score, auto, fi = best
    cache.parent.mkdir(parents=True, exist_ok=True)
    cache.write_text(json.dumps({
        "H": auto.homography.H.tolist(), "level": auto.reading.level,
        "frame": fi, "paint": score, "problems": auto.problems,
        "correspondences": len(auto.correspondences),
    }))
    log(f"    anchor: fitted on frame {fi} — {auto.summary()}")
    return auto.homography, auto.spec, "computed", list(auto.problems)


class _FrameBag:
    """Already-decoded frames, wearing a VideoSource's ``frames()`` method.

    Propagation wants a video source; the renderer has the frames in memory already and
    decoding them twice costs a minute per clip for nothing.
    """

    def __init__(self, frames: dict[int, np.ndarray]):
        self._f = frames

    def frames(self, start: int, stop: int, step: int = 1):
        for i in sorted(self._f):
            if start <= i < stop and (i - start) % step == 0:
                yield i, self._f[i]


def _label_kits(spec: ClipSpec, frames, boxes, log) -> float:
    """Put a team on every detection, before tracking, so the tracker can use it.

    The same clustering that names the sides afterwards, run early and on detections
    rather than tracks. It is cruder — a detection votes once and there is no pooling
    across a play to steady it — and it is not what the output's team labels come from.
    Its only job is to make two crossing players distinguishable to the matcher, which
    distance alone cannot do.
    """
    from .detect.color import cluster_teams, separation, torso_color

    samples, index = {}, []
    for fi, dets in boxes.items():
        img = frames.get(fi)
        if img is None:
            continue
        for k, d in enumerate(dets):
            c = torso_color(img, d)
            if c is None:
                continue
            key = f"{fi}:{k}"
            samples[key] = [c[1:] if spec.track.chroma_only else c]
            index.append((fi, k, key))
    if len(samples) < 2:
        return 0.0
    assign, centres = cluster_teams(samples, k=spec.track.team_clusters)
    sizes = {c: sum(1 for v in assign.values() if v == c) for c in set(assign.values())}
    ranked = sorted(sizes, key=lambda c: -sizes[c])
    naming = {c: n for c, n in zip(ranked, ["kitA", "kitB", "kitC"])}
    for fi, k, key in index:
        boxes[fi][k].team = naming.get(assign.get(key))
    sep = separation(np.array([centres[c] for c in ranked[:2]])) if len(ranked) >= 2 else 0.0
    log(f"    kits before tracking: {dict((naming[c], sizes[c]) for c in ranked)}, "
        f"separation {sep:.0f} Lab units")
    return sep


def _teams(spec: ClipSpec, track: PlayTrack, frames, rects, reg, fps: float,
           offset: int, log) -> float:
    """Split the tracked players into two sides by the colour of their kit.

    Done here, after tracking and after the furniture has been thrown out, and that
    ordering is the whole thing. Clustering raw detections instead puts the entire
    Northwestern bench into the sample — thirty people in one kit, standing in a row,
    outnumbering the players — and k-means obligingly separates *crowd from field* and
    reports 90 Lab units of separation for it. The number looked like a triumph and the
    overlay showed one team.

    Clustering tracks fixes it twice over. The bench is already gone, and a track votes
    once however many frames it was seen in, so a lineman held in shot for the whole play
    does not outvote a receiver who ran out of frame.

    The colour itself is pooled in Lab across the play — one frame of a man in shadow or
    turned away cannot flip him when forty others disagree.
    """
    from .detect.color import cluster_teams, separation, torso_color
    from .detect import Detection

    samples: dict[str, list] = {}
    for f in track.frames:
        src = int(round(f.t * fps)) + offset
        img = frames.get(src)
        boxes = rects.get(src)
        if img is None or not boxes:
            continue
        h = reg.at(src)
        if h is None or not f.players:
            continue
        ids = list(f.players)
        px = np.atleast_2d(h.to_image([[f.players[i].x, f.players[i].y] for i in ids]))
        for pid, p in zip(ids, px):
            if not np.isfinite(p).all():
                continue
            # The box this player is standing in, if any. A position that reprojects onto
            # nobody has no kit to read, and guessing from the grass underneath is worse
            # than not voting.
            hit = next((b for b in boxes if b[0] <= p[0] <= b[2] and b[1] <= p[1] <= b[3]), None)
            if hit is None:
                continue
            d = Detection(frame=src, x1=hit[0], y1=hit[1], x2=hit[2], y2=hit[3])
            c = torso_color(img, d)
            if c is None:
                continue
            samples.setdefault(pid, []).append(c[1:] if spec.track.chroma_only else c)

    if len(samples) < 2:
        log("    teams: not enough torso samples to split")
        return 0.0

    assign, centres = cluster_teams(samples, k=spec.track.team_clusters)
    sizes = {c: sum(1 for v in assign.values() if v == c) for c in set(assign.values())}
    ranked = sorted(sizes, key=lambda c: -sizes[c])
    naming = {c: n for c, n in zip(ranked, ["teamA", "teamB", "other"])}
    for pid, cl in assign.items():
        if pid in track.roster:
            track.roster[pid].team = naming.get(cl, "other")

    # The separation that matters is between the two *teams*. Taking the closest of all
    # three pairs reports how near the officials sit to a kit, which is not the question.
    sep = separation(np.array([centres[c] for c in ranked[:2]])) if len(ranked) >= 2 else 0.0
    counts = {n: sum(1 for p in track.roster.values() if p.team == n) for n in set(naming.values())}
    log(f"    teams: {counts} from {len(samples)} tracks, separation {sep:.0f} Lab units")
    return sep


def _ball_boxes(model, image, spec: ClipSpec):
    """Every sports-ball box in one frame, over overlapping crops.

    The same trick ``detect.yolo._tiled`` uses for far-hash players, and for the
    same reason one step further: a football is the smallest thing on the field.
    Un-tiled on the reference clip the generic class finds it on 3 frames of 65.

    No merging of duplicates across overlaps, deliberately — the ball tracker
    downstream takes a list of candidates per frame and picks the one that moves
    like a ball, so a few extra candidates cost it nothing and a missed one costs
    the throw instant.
    """
    n = max(1, int(spec.track.ball_tiles))
    kw = dict(conf=spec.track.ball_conf, imgsz=spec.track.ball_imgsz,
              classes=[spec.track.ball_class], verbose=False)
    h, w = image.shape[:2]
    out = []

    def run(crop, dx, dy):
        res = model.predict(crop, **kw)[0]
        if res.boxes is None or not len(res.boxes):
            return
        for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.cpu().numpy(),
                                       res.boxes.conf.cpu().numpy()):
            out.append(((float(x1) + dx, float(y1) + dy,
                         float(x2) + dx, float(y2) + dy), float(c)))

    run(image, 0.0, 0.0)
    if n > 1:
        tw, th = w / n, h / n
        ox, oy = tw * spec.detect.tile_overlap, th * spec.detect.tile_overlap
        for r in range(n):
            for c in range(n):
                x0, y0 = int(max(0, c * tw - ox)), int(max(0, r * th - oy))
                x1, y1 = int(min(w, (c + 1) * tw + ox)), int(min(h, (r + 1) * th + oy))
                if x1 - x0 >= 32 and y1 - y0 >= 32:
                    run(image[y0:y1, x0:x1], x0, y0)
    return out


def _drop_static_candidates(cand: dict, log, radius: float = 1.5,
                            min_frames: int = 8) -> dict:
    """Throw away ball candidates that sit in one place all play.

    The generic sports-ball class has an easy wrong answer on a televised game: the
    down-and-distance graphic is rendered onto the field near the line of scrimmage,
    it is about the right shape, and it is in every frame. Tiling the detector made
    it nine times better at finding it — 3 candidates became 27, and the resulting
    "ball" travelled 2.4 yards in 2.6 seconds.

    A football is the one object on the field guaranteed to move. So a cluster of
    candidates that appears at the same field position across a large part of the
    play is furniture, whatever the detector's confidence, and dropping it lets the
    tracker consider whatever else was found. This is a filter on persistence and
    not on appearance: it has no opinion about what the thing is, only that a ball
    does not spend a play standing still.
    """
    points = [(t, x, y) for t, got in cand.items() for (x, y, _c) in got]
    if len(points) < min_frames:
        return cand

    static: list[tuple[float, float]] = []
    used = [False] * len(points)
    for i, (_t, x, y) in enumerate(points):
        if used[i]:
            continue
        members = [j for j, (_tj, xj, yj) in enumerate(points)
                   if not used[j] and np.hypot(xj - x, yj - y) <= radius]
        frames_seen = {points[j][0] for j in members}
        if len(frames_seen) >= min_frames:
            for j in members:
                used[j] = True
            cx = float(np.mean([points[j][1] for j in members]))
            cy = float(np.mean([points[j][2] for j in members]))
            static.append((cx, cy))

    if not static:
        return cand

    out, dropped = {}, 0
    for t, got in cand.items():
        keep = []
        for (x, y, c) in got:
            if any(np.hypot(x - sx, y - sy) <= radius for sx, sy in static):
                dropped += 1
            else:
                keep.append((x, y, c))
        out[t] = keep
    log(f"    dropped {dropped} static ball candidates at "
        + ", ".join(f"({sx:.0f}, {sy:.0f})" for sx, sy in static[:3])
        + " — a ball does not stand still")
    return out


_FATAL_FIT = re.compile(
    r"collinear|folded|not enough to fix scale|no registration|self-crossing",
    re.I,
)
"""Problems that mean the coordinates are not coordinates. These are the checks'
own words; matching on them rather than restructuring vision.py's warning lists,
because the wording is the thing a person reads and it should stay one string."""


MIN_BALL_TRAVEL = 10.0
"""Yards a real ball covers in a play, at the very least. Below this the track is
of something that does not move, which on a televised game means a graphic."""


def registration_verdict(problems, stage) -> dict:
    """Whether anything in this file is a coordinate, and why not when it is not.

    Pulled out of ``render`` so it can be tested: the rest of that function needs a
    video, a detector and a vision model, so the one line deciding whether a play's
    numbers mean anything had no test at all. It is matched on the checks' own
    wording rather than a structured flag, which makes the pairing of wording and
    match something that has to be kept together — see test_prepare.
    """
    return {
        "stage": stage,
        "trustworthy": not any(_FATAL_FIT.search(p) for p in problems),
        "note": (
            "A homography can fit its own points perfectly and still place the "
            "field a hundred yards away. Read render_problems before believing a "
            "single coordinate in this file."
        ),
    }


def _merge_ball(track: PlayTrack, ball, log) -> None:
    """Put the tracked ball on the frames, and say how much of it was seen.

    It was previously computed and used only to badge the side-by-side video, so
    track.json came out with no ball however well the ball had been tracked —
    and everything downstream that needs one (possession, the passer, the throw,
    the targeted receiver) had nothing to work with and no way to tell whether
    that was because the ball was untracked or because the merge was missing.

    Coasted samples are written as well as detected ones, because a ball between
    two sightings really was somewhere between them. The contract's Ball has no
    field for how a position was arrived at — unlike a player's o_source — so the
    proportions go in meta, and a reader who does not check them is trusting a
    5%-measured ball as though it were a 100%-measured one.
    """
    counts = ball.counts()
    written = 0
    for f in track.frames:
        s = ball.at(f.t)
        if s is None or s.source == "none":
            continue
        f.ball = Ball(x=float(s.x), y=float(s.y), source=s.source)
        written += 1

    measured = counts.get("detected", 0)
    total = len(ball.samples) or 1

    # How far it actually went. A detection rate says how often something was found
    # and nothing at all about whether that something was a football: on the
    # reference clip, tiling raised the rate from 5% to 42% by finding the
    # down-and-distance graphic, which sits on the field near the line of scrimmage
    # and never moves. A ball that travels two yards in a whole play is furniture.
    positions = [(f.t, f.ball.x, f.ball.y) for f in track.frames if f.ball is not None]
    span = 0.0
    if len(positions) >= 2:
        xs = [p[1] for p in positions]
        ys = [p[2] for p in positions]
        span = float(np.hypot(max(xs) - min(xs), max(ys) - min(ys)))

    track.meta["ball"] = {
        "frames_with_a_position": written,
        "frames_total": len(track.frames),
        "by_source": counts,
        "measured_fraction": round(measured / total, 3),
        "travelled_yards": round(span, 2),
        "note": (
            "A football in a wide broadcast shot is a few pixels and is behind "
            "somebody most of the play. Read measured_fraction AND travelled_yards "
            "before trusting any possession, passer or throw derived from this: a "
            "high detection rate on a static object is a confident wrong answer, "
            "and the generic sports-ball class finds broadcast graphics."
        ),
    }
    log(f"    ball merged onto {written}/{len(track.frames)} frames "
        f"({100 * measured / total:.0f}% seen, travelled {span:.1f} yd"
        f"{' — STATIC, this is not a ball' if span < MIN_BALL_TRAVEL else ''})")


def _carrier(spec: ClipSpec, track: PlayTrack, log):
    """Where the ball is because somebody is holding it, rather than because it was seen.

    No pixels and no detector. The ball is within a yard of a player on 83% of
    frames and airborne for about a second a play, so for most of a play finding it
    is a question of which tracked man has it — see carrier.py.
    """
    if not spec.track.carrier:
        return {}
    model = carriermod.load(spec.track.carrier_weights)
    if model is None:
        log(f"    no carrier model at {spec.track.carrier_weights} "
            f"— the ball will come from the detector alone")
        return {}
    xy, conf = carriermod.carrier_by_t(track, model)
    if xy:
        log(f"    carrier: {len(xy)}/{len(track.frames)} frames attributed "
            f"({100 * len(xy) / max(len(track.frames), 1):.0f}%), "
            f"median confidence {np.median(list(conf.values())):.2f}")
    else:
        log("    carrier: nobody held it with enough confidence on any frame")
    return xy


def _ball(spec: ClipSpec, track: PlayTrack, frames, reg, fps, offset: int, log):
    """The ball: a sighting where there is one, the man holding it where there is not."""
    carrier_xy = _carrier(spec, track, log)

    players: dict[float, np.ndarray] = {}
    for f in track.frames:
        players[round(f.t, 3)] = (np.array([[p.x, p.y] for p in f.players.values()])
                                  if f.players else np.zeros((0, 2)))

    cand: dict[float, list] = {}
    if spec.track.ball:
        cand = _ball_candidates(spec, track, frames, reg, fps, offset, log)

    if not cand and not carrier_xy:
        return None

    bt = ballmod.track_ball([round(f.t, 3) for f in track.frames], cand,
                            players_by_t=players, carrier_by_t=carrier_xy)
    log(f"    {bt.summary()}")
    return bt


def _ball_candidates(spec: ClipSpec, track: PlayTrack, frames, reg, fps, offset: int, log):
    """Sightings from the detector, gated to the field and to a player."""
    try:
        from ultralytics import YOLO
    except ImportError:
        return {}
    model = YOLO(spec.track.ball_weights or spec.detect.weights)
    cand: dict[float, list] = {}
    seen = 0
    for f in track.frames:
        src = int(round(f.t * fps)) + offset
        img = frames.get(src)
        key = round(f.t, 3)
        if img is None:
            continue
        boxes = _ball_boxes(model, img, spec)
        seen += bool(boxes)
        got = []
        for (x1, y1, x2, y2), c in boxes:
            pt = np.atleast_2d(reg.to_field(src, [[(x1 + x2) / 2, (y1 + y2) / 2]], height_yards=1.0))[0]
            if np.isfinite(pt).all():
                got.append((float(pt[0]), float(pt[1]), float(c)))
        cand[key] = got
    log(f"    ball candidates on {seen}/{len(track.frames)} frames "
        f"({spec.track.ball_tiles}x{spec.track.ball_tiles} tiles, "
        f"conf {spec.track.ball_conf}, imgsz {spec.track.ball_imgsz}, "
        f"{'fine-tuned weights' if spec.track.ball_weights else 'generic sports-ball class'})")
    return _drop_static_candidates(cand, log)


# ------------------------------------------------------------------ the thing


class _Static:
    """One homography, presented as the per-frame protocol everything else expects."""

    def __init__(self, H, spec: fieldmod.FieldSpec, image_size):
        self._H, self.spec, self.image_size = H, spec, image_size

    def at(self, frame_index: int):
        return self._H

    def to_field(self, frame_index: int, image_pts, height_yards: float = 0.0):
        from .registration.camera import ground_points_with_height

        if self._H is None:
            return np.full((len(np.atleast_2d(image_pts)), 2), np.nan)
        return ground_points_with_height(
            self._H, image_pts, height_yards=height_yards, image_size=self.image_size
        )


class _ByWindowIndex:
    """A source-frame-keyed registration, addressed by position in the decoded window.

    The pipeline counts frames from zero; the registration knows them by their index in
    the file. Keeping the translation in one small adapter is what lets a static
    homography and a propagated one be the same thing to everything downstream.
    """

    def __init__(self, reg, ordered: list[int]):
        self._r, self._o = reg, ordered
        self.spec = reg.spec
        self.image_size = reg.image_size

    def _src(self, i: int) -> int:
        return self._o[i] if 0 <= i < len(self._o) else -1

    def at(self, i: int):
        return self._r.at(self._src(i))

    def to_field(self, i: int, pts, height_yards: float = 0.0):
        return self._r.to_field(self._src(i), pts, height_yards=height_yards)


class _ByTime:
    """The registration addressed by contract time, which is what the checker has.

    ``verify`` holds a :class:`PlayTrack`, and a PlayTrack knows seconds from the snap,
    not frames. Translating here rather than inside verify keeps the one place that knows
    the clock — snap frame plus frame rate — next to the spec that sets it.
    """

    def __init__(self, reg, fps: float, offset: int):
        self._r, self._fps, self._off = reg, fps, offset
        self.spec = reg.spec
        self.image_size = reg.image_size

    def _src(self, t: float) -> int:
        return int(round(t * self._fps)) + self._off

    def at(self, src: int):
        return self._r.at(src)

    def to_image(self, t: float, field_pts):
        h = self._r.at(self._src(t))
        if h is None:
            return np.full((len(np.atleast_2d(field_pts)), 2), np.nan)
        return h.to_image(field_pts)

    def to_field(self, src: int, pts, height_yards: float = 0.0):
        return self._r.to_field(src, pts, height_yards=height_yards)


def render(spec: ClipSpec, root: Path = RUN_DIR, force=(), log=print) -> RenderResult:
    """Run one clip all the way to the two-panel video."""
    t0 = time.time()
    force = set(force)
    out_dir = spec.run_dir(root)
    out_dir.mkdir(parents=True, exist_ok=True)
    res = RenderResult(spec=spec, out_dir=out_dir)
    log(f"  {spec.name}: {spec.title or Path(spec.video).name}")

    if spec.step != 1:
        # Everything downstream turns a contract time into a source frame with
        # t * fps + snap, which is only a frame index when every frame was decoded.
        # Rather than half-support a decimated window, say so.
        res.problems.append(f"step={spec.step} is not supported by the renderer; use step 1")
        return res

    frames, fps, n_src = _decode(spec, log)
    res.n_frames = len(frames)
    if not frames:
        res.problems.append("no frames decoded — check the window")
        return res
    ordered = sorted(frames)
    any_frame = frames[ordered[0]]
    size = (any_frame.shape[1], any_frame.shape[0])

    # Detection runs first, and not only because it is cached. Measuring the camera's
    # motion needs the players painted out of the frame, so the boxes are an input to
    # registration rather than a consumer of it.
    boxes, det_stage = _detect(spec, frames, out_dir, force, log)
    res.stages["detect"] = det_stage

    base, field_spec, reg_stage, reg_problems, prop = _register(
        spec, frames, n_src, out_dir, force, log, boxes=boxes)
    res.stages["register"] = reg_stage + ("+propagated" if prop is not None else "")
    res.problems.extend(reg_problems)
    if base is None:
        res.problems.append("no registration — nothing can be placed in yards")
        res.seconds = time.time() - t0
        return res

    src_reg = prop if prop is not None else _Static(base, field_spec, size)
    if prop is not None:
        prop.spec, prop.image_size = field_spec, size
    res.registration = src_reg

    projected = {}
    for k, v in boxes.items():
        if not v:
            continue
        p = np.atleast_2d(src_reg.to_field(k, np.array([d.anchor()[0] for d in v]), 0.0))
        projected[k] = p[np.isfinite(p).all(axis=1)]
    cuts = sideline_cuts(projected, field_spec, spec.detect.field_margin)
    if spec.detect.sideline_cut:
        log(f"    sidelines: keeping field y {cuts[0]:.1f} to {cuts[1]:.1f} "
            f"(paint is 0 to {field_spec.width:.1f})")
    else:
        cuts = (-spec.detect.field_margin, field_spec.width + spec.detect.field_margin)
    boxes = {k: _on_field(v, src_reg, k, field_spec, spec.detect.field_margin, cuts)
             for k, v in boxes.items()}
    res.n_boxes = sum(len(v) for v in boxes.values())

    if spec.track.pre_teams:
        _label_kits(spec, frames, boxes, log)

    by_index = {i: boxes.get(f, []) for i, f in enumerate(ordered)}

    class _Seq:
        def detect(self, i, image=None):
            return [
                Detection(frame=i, x1=d.x1, y1=d.y1, x2=d.x2, y2=d.y2,
                          score=d.score, kind=d.kind, team=d.team)
                for d in by_index.get(i, [])
            ]

    cfg = PipelineConfig(
        fps=fps,
        snap_frame=spec.snap(n_src),
        spec=field_spec,
        image_size=size,
        gate_yards=spec.track.gate_yards,
        max_age=spec.track.max_age,
        min_hits=spec.track.min_hits,
        process_noise=spec.track.process_noise,
        measurement_noise=spec.track.measurement_noise,
        smooth_window=spec.track.smooth_window,
        keep_unlabelled=spec.track.keep_unlabelled,
        team_cost=spec.track.team_cost,
    )
    out = run_pipeline(_Seq(), _ByWindowIndex(src_reg, ordered), len(ordered), cfg)
    if spec.track.stitch:
        from .stitch import stitch as _stitch
        rep = _stitch(out.track, max_gap_s=spec.track.stitch_gap_s,
                      gate_yards=spec.track.stitch_gate)
        if rep.merged:
            log(f"    {rep.summary()}")
    if spec.track.drop_static_sideline:
        n = _drop_furniture(out.track, field_spec, spec.track.static_yards,
                            spec.track.sideline_band)
        if n:
            log(f"    dropped {n} tracks that never moved off a sideline")
    res.track = out.track
    res.stages["track"] = "computed"
    log(f"    tracked: {len(out.track.roster)} players over {len(out.track.frames)} frames "
        f"({out.tracks_born} born, {out.tracks_kept} kept)")

    # t = 0 is the snap, so the frame a contract time names is snap + t * fps — not
    # window-start + t * fps. They coincide only when the snap is the first frame, which
    # is how an off-by-a-second clock hides.
    offset = spec.snap_frame if spec.snap_frame is not None else ordered[0]
    # verify and the renderer want plain rectangles; they are checking geometry and have
    # no business knowing what a Detection is.
    rects = {k: [(d.x1, d.y1, d.x2, d.y2) for d in v] for k, v in boxes.items()}
    timed = _ByTime(src_reg, fps, offset)

    if spec.track.teams:
        res.team_separation = _teams(spec, out.track, frames, rects, src_reg, fps, offset, log)

    res.report = verifymod.verify(out.track, timed, rects, fps=fps, frame_offset=offset)

    if spec.distance_yards:
        from .broadcast import check_markers
        res.markers = check_markers(frames, src_reg, spec.distance_yards)
        if res.markers.samples:
            log(f"    broadcast markers: {res.markers.measured:.2f} yd measured against "
                f"{spec.distance_yards:.0f} stated "
                f"({res.markers.error_yards:+.2f}, scatter {res.markers.straightness:.2f})")
            # The two lines are read here to check the registration's scale, and
            # they establish three more things nothing downstream can recover
            # without the video again: the yardline, the distance, and — since the
            # first-down line is always downfield — which way the offense is going.
            # A clip cut near the snap with no ball has no other automatic answer
            # to that last one, and every other inference depends on it.
            out.track.meta["broadcast"] = res.markers.situation()
            log(f"    situation: line of scrimmage x={res.markers.line_of_scrimmage:.1f}, "
                f"offense attacks {res.markers.attacks}")

    if spec.track.ball or spec.track.carrier:
        try:
            res.ball = _ball(spec, out.track, frames, src_reg, fps, offset, log)
        except Exception as e:  # noqa: BLE001 - a missing ball must not cost the video
            res.problems.append(f"ball tracking failed: {type(e).__name__}: {e}")
        if res.ball is not None:
            _merge_ball(out.track, res.ball, log)

    res.video = verifymod.side_by_side(
        out_dir / "sidebyside.mp4", out.track, timed, frames, rects, field_spec,
        fps=fps, frame_offset=offset, out_fps=spec.out_fps,
        team_colors=TEAM_COLORS,
        ball_seen_at=(lambda t: (res.ball.at(t) or ballmod.BallSample(t, 0, 0, "none")).measured)
        if res.ball else None,
    )
    res.stages["render"] = "computed"

    # Everything the render wants to complain about, written into the file rather
    # than only onto the console. devin_td registered with a fit the checks called
    # folded, collinear and "a hundred yards wrong and still perfect on its own
    # points" — and track.json came out carrying 41 players positioned in yards
    # with nothing in it to say the yards were fiction. A consumer reading the file
    # had no way to know, which is the whole failure mode this package is arranged
    # against.
    out.track.meta["render_problems"] = list(res.problems)
    out.track.meta["registration"] = registration_verdict(
        res.problems, res.stages.get("register"))
    (out_dir / "track.json").write_text(json.dumps(out.track.to_json()))
    report_txt = res.report.table()
    if res.markers is not None and res.markers.samples:
        report_txt += "\n\n" + res.markers.table()
    (out_dir / "report.txt").write_text(report_txt + "\n")
    (out_dir / "spec.json").write_text(json.dumps(spec.to_json(), indent=2) + "\n")
    res.seconds = time.time() - t0
    return res


def _drop_furniture(track: PlayTrack, spec: fieldmod.FieldSpec,
                    static_yards: float, band: float) -> int:
    """Remove tracks that stand on a sideline and never move.

    Everyone on a bench is a person, is standing on a football field as far as the
    geometry is concerned, and is the wrong answer. This is the only property that
    actually distinguishes them from a player: the play happens and they do not move.
    """
    drop = []
    for pid in list(track.roster):
        pts = np.array([[f.players[pid].x, f.players[pid].y]
                        for f in track.frames if pid in f.players])
        if len(pts) < 2:
            continue
        moved = float(np.linalg.norm(pts.max(axis=0) - pts.min(axis=0)))
        near_edge = float(np.min([pts[:, 1].min(), spec.width - pts[:, 1].max()]))
        if moved <= static_yards and near_edge <= band:
            drop.append(pid)
    for pid in drop:
        track.roster.pop(pid, None)
        for f in track.frames:
            f.players.pop(pid, None)
    return len(drop)


def sideline_cuts(pos_by_frame: dict[int, np.ndarray], spec: fieldmod.FieldSpec,
                  margin: float, min_crowd: float = 2.0) -> tuple[float, float]:
    """Where the field of play ends and the people watching it begin, read off the data.

    A margin around the sidelines does not remove a bench, and the reason is worth
    stating: a coach standing behind the far sideline has his feet hidden by the people
    in front of him, so the detector's box bottom lands on the chalk and the geometry
    places him *inside* the field, at y between 51 and 54 on this clip. No margin can
    exclude him without also excluding a receiver on the sideline, and his apparent size
    does not give him away either — at that depth a standing coach and a cornerback are
    within a few pixels of the same height.

    What gives him away is that he is not alone. Fifteen detections per frame pile into
    the two yards outside the numbers, a play puts one or two there, and between the two
    populations sits an empty band: on the reference clip, 0.03 detections per frame
    across y 48 to 50, against 0.7 just inside it and 4.8 just outside. That gap is the
    sideline as the pixels see it rather than as the paint claims it, and cutting at the
    emptiest bin removes fourteen bench detections a frame and keeps every player.

    The cut is only made where there is something to cut: a crowd beyond the line, at
    least ``min_crowd`` detections a frame. A clip filmed from a tripod with nobody on
    the touchline keeps the plain margin, because there the gap means nothing.
    """
    n = len(pos_by_frame) or 1
    ys = np.concatenate([p[:, 1] for p in pos_by_frame.values() if len(p)]) if pos_by_frame else np.zeros(0)
    if not len(ys):
        return (-margin, spec.width + margin)

    def cut(edge: float, outward: int) -> float:
        # Candidate lines from well inside the paint to a little beyond it. The crowd
        # does not politely stay outside the sideline — on this clip it straddles it,
        # because occluded feet put a standing coach two yards inside the chalk — so the
        # search has to start inside the field, not at its edge.
        cands = edge + outward * np.arange(-8.5, margin + 0.01, 1.0)
        best, best_density = None, np.inf
        for c in cands:
            lo, hi_ = min(c - 0.5, c + 0.5), max(c - 0.5, c + 0.5)
            density = float(((ys >= lo) & (ys < hi_)).sum()) / n
            beyond = float((ys >= c).sum() if outward > 0 else (ys <= c).sum()) / n
            if beyond < min_crowd:
                continue
            # Ties go to the line that keeps the most field: a cut is a cost, and the
            # emptiest band is usually several bins wide.
            if density < best_density - 1e-9 or (abs(density - best_density) <= 1e-9
                                                and best is not None
                                                and outward * c > outward * best):
                best, best_density = float(c), density
        return best if best is not None else edge + outward * margin

    return (cut(0.0, -1), cut(spec.width, +1))


def _on_field(dets, reg, frame, spec: fieldmod.FieldSpec, margin: float,
              cuts: tuple[float, float] | None = None):
    """Drop what is confidently a person and confidently not on the field.

    This is geometry, not confidence. A steward on the touchline and a fan in row three
    are both unmistakably people; both land outside the sidelines once the map is
    applied, and that is a far better filter than raising the detector's threshold until
    they go away and take the far-hash players with them.
    """
    if not dets:
        return []
    lo, hi = cuts if cuts is not None else (-margin, spec.width + margin)
    pts = np.array([d.anchor()[0] for d in dets], dtype=float)
    got = np.atleast_2d(reg.to_field(frame, pts, height_yards=0.0))
    keep = []
    for d, (x, y) in zip(dets, got):
        if np.isfinite(x) and np.isfinite(y) and -margin <= x <= spec.length + margin \
                and lo <= y <= hi:
            keep.append(d)
    return keep
