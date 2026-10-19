"""Does the recreation match the clip? Checked frame by frame, player by player.

There is no ground truth for a clip pulled off the internet, which seems to make
"is each player in the right place" unanswerable. It is not. The registration is a
two-way map: push a recovered field position *back* through it and you get the pixel
that position claims to be. If that pixel lands on a player in the actual frame, the
coordinate is right. If it lands on grass, on a coach, or in the crowd, it is wrong —
and you can see which, on which frame, for which player.

That turns a vague "it looks off" into four numbers per frame. The assignment behind
them is *exclusive* — a box is one person, so one track may hold it — and that is the
whole difference between a number that can grade this pipeline and one that cannot:

**on_player** — the recovered position reprojects inside a detection box that no other
track holds. The coordinate is consistent with what the detector saw, and it accounts
for exactly one person.

**duplicate** — it reprojects onto a real person who is already held by another track.
One man wearing two tracks, which is this pipeline's most common defect and the reason
for the exclusivity: scored permissively, every duplicate reads as a second success, and
the measurement gets better precisely as the tracking gets worse. The reference clip
read 92.1% that way and reads 80.0% honestly; the entire 12-point gap was duplication.

**stray** — it reprojects onto nothing. Either the tracker is coasting a player who is
no longer there, or the position has drifted off the man it belongs to.

**unclaimed** — a detection with no track on it. Somebody on the field is not being
tracked.

What none of them can see is *who* a box contains. An official working the play is a
person, on the field, detected — so a track on him scores as cleanly as a track on a
safety. Furniture has to be kept out upstream; this check will never report it.

None of this proves the *field* coordinates are correct in yards; a wrong-but-
self-consistent registration would still score well, which is why it sits alongside the
camera and turf checks rather than replacing them. What it does prove is that the
tracking, the smoothing and the resampling did not move anybody away from where the
pixels say they are — and that is the layer the eye notices first.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np
from scipy.optimize import linear_sum_assignment

from .contract import PlayTrack

PANEL_RULE_PX = 4
"""A hairline between the two panels. Without it the clip's own dark edge and the top of
the board run together and the pair reads as one picture."""


@dataclass
class FrameCheck:
    t: float
    frame_index: int
    on_player: int = 0
    duplicate: int = 0
    stray: int = 0
    unclaimed: int = 0
    detections: int = 0
    tracked: int = 0
    worst_px: float = 0.0

    @property
    def agreement(self) -> float:
        return self.on_player / self.tracked if self.tracked else 0.0


@dataclass
class PlayerCheck:
    player_id: str
    frames: int = 0
    on_player: int = 0
    duplicate: int = 0
    mean_px: float = 0.0

    @property
    def agreement(self) -> float:
        return self.on_player / self.frames if self.frames else 0.0


@dataclass
class VerifyReport:
    frames: list[FrameCheck] = dc_field(default_factory=list)
    players: dict[str, PlayerCheck] = dc_field(default_factory=dict)

    @property
    def agreement(self) -> float:
        tot = sum(f.tracked for f in self.frames)
        return sum(f.on_player for f in self.frames) / tot if tot else 0.0

    def worst_frames(self, n: int = 5) -> list[FrameCheck]:
        return sorted(self.frames, key=lambda f: f.agreement)[:n]

    def worst_players(self, n: int = 5) -> list[PlayerCheck]:
        return sorted(self.players.values(), key=lambda p: p.agreement)[:n]

    def table(self) -> str:
        lines = [
            "frame-by-frame agreement between the recreation and the clip",
            "",
            f"  overall            {self.agreement * 100:.1f}% of tracked players hold a person of their own",
            f"  frames checked     {len(self.frames)}",
            f"  coverage           {np.mean([f.on_player for f in self.frames]):.1f} of 22"
            f"   (people accounted for, one track each)",
            f"  duplicate          {np.mean([f.duplicate for f in self.frames]):.1f}"
            f"   (one person wearing two tracks)",
            f"  stray per frame    {np.mean([f.stray for f in self.frames]):.1f}"
            f"   (tracks sitting on nobody)",
            f"  unclaimed          {np.mean([f.unclaimed for f in self.frames]):.1f}"
            f"   (people on the field with no track)",
            "",
            "  worst frames:",
        ]
        for f in self.worst_frames():
            lines.append(f"    t {f.t:+.1f}s  frame {f.frame_index}: "
                         f"{f.on_player}/{f.tracked} on a player, {f.duplicate} duplicate, "
                         f"{f.stray} stray, {f.unclaimed} unclaimed")
        lines += ["", "  worst players:"]
        for p in self.worst_players():
            lines.append(f"    {p.player_id:<10} {p.agreement * 100:5.1f}% over {p.frames} frames, "
                         f"{p.duplicate} duplicate, mean miss {p.mean_px:.0f} px")
        return "\n".join(lines)


def _inside(px, box, pad: float = 6.0) -> bool:
    x1, y1, x2, y2 = box
    return (x1 - pad) <= px[0] <= (x2 + pad) and (y1 - pad) <= px[1] <= (y2 + pad)


def _distance_to(px, box) -> float:
    x1, y1, x2, y2 = box
    dx = max(x1 - px[0], 0.0, px[0] - x2)
    dy = max(y1 - px[1], 0.0, px[1] - y2)
    return float(np.hypot(dx, dy))


def _foot(box) -> tuple[float, float]:
    """Where a box says its person is standing."""
    x1, _, x2, y2 = box
    return ((x1 + x2) / 2.0, y2)


def assign(px, boxes, pad: float = 6.0):
    """One box, one track. Returns ``(matched, duplicate, stray)`` by track index.

    The reason this is not a loop over tracks each taking its nearest box: a box is
    one person, so it can be claimed once. Two tracks inside one box is the most
    common defect this pipeline has, and a scorer that lets both claim it reports two
    successes where the truth is one success and one duplicate — which is how a frame
    carrying four officials, a duplicate and two rings on bare grass scored 19 of 19.

    So the assignment is exclusive, and what does not fit is *named* rather than
    quietly absorbed:

    ``matched``    track -> box, one to one. The only thing that counts as agreement.
    ``duplicate``  the track is inside a box another track already holds. That is one
                   person wearing two tracks — a defect of its own kind, never a hit.
    ``stray``      the track is inside no box at all.

    Containment is still the gate, exactly as before. The cost that breaks ties is the
    distance to the box's *foot*, because every box inside which a point lies scores
    zero on :func:`_distance_to` and the choice between them would otherwise be
    whichever one the detector happened to emit first.
    """
    n, m = len(px), len(boxes)
    matched: dict[int, int] = {}
    duplicate: set[int] = set()
    stray: set[int] = set()
    if n == 0:
        return matched, duplicate, stray

    finite = [i for i in range(n) if np.isfinite(px[i]).all()]
    stray.update(set(range(n)) - set(finite))
    if not finite or m == 0:
        stray.update(finite)
        return matched, duplicate, stray

    inside = np.zeros((len(finite), m), dtype=bool)
    cost = np.full((len(finite), m), 1e6)
    for r, i in enumerate(finite):
        for c, b in enumerate(boxes):
            if _inside(px[i], b, pad):
                inside[r, c] = True
                fx, fy = _foot(b)
                cost[r, c] = float(np.hypot(px[i][0] - fx, px[i][1] - fy))

    for r, c in zip(*linear_sum_assignment(cost)):
        if inside[r, c]:
            matched[finite[r]] = int(c)

    for r, i in enumerate(finite):
        if i in matched:
            continue
        (duplicate if inside[r].any() else stray).add(i)
    return matched, duplicate, stray


def _player_radius(to_image, player, fallback: int = 11) -> int:
    """How wide a player is in pixels, at the place he is standing.

    Taken from the geometry, not from whichever detector box the point fell into. A
    box was the wrong source twice over: in a pile the boxes are wide and tangled, so
    the ring swelled to two or three times a player's width exactly where the tracking
    most needed auditing, and when several boxes overlapped the first one won by
    accident. A shoulder span is about a yard anywhere on the field, so projecting one
    yard at the player's own feet gives a ring that shrinks with distance the way it
    should.
    """
    pts = np.atleast_2d(to_image([[player.x - 0.5, player.y], [player.x + 0.5, player.y]]))
    if pts.shape[0] < 2 or not np.isfinite(pts).all():
        return fallback
    return int(np.clip(round(float(np.hypot(*(pts[1] - pts[0]))) / 2.0), 5, 60))


def verify(
    track: PlayTrack,
    registration,
    boxes_by_frame: dict[int, list],
    fps: float,
    frame_offset: int = 0,
    pad: float = 6.0,
) -> VerifyReport:
    """Reproject every tracked player and check it lands on somebody.

    ``boxes_by_frame`` is the detector's own output keyed by source frame index —
    the same boxes the pipeline consumed, so this asks whether the chain moved anyone
    rather than whether the detector was right.
    """
    rep = VerifyReport()

    for f in track.frames:
        src = int(round(f.t * fps)) + frame_offset
        boxes = boxes_by_frame.get(src)
        if not boxes:
            continue

        ids = list(f.players.keys())
        if not ids:
            continue
        field_pts = np.array([[f.players[i].x, f.players[i].y] for i in ids])
        px = np.atleast_2d(registration.to_image(f.t, field_pts)
                           if hasattr(registration, "to_image")
                           else _to_image(registration, src - frame_offset, field_pts))

        fc = FrameCheck(t=f.t, frame_index=src, detections=len(boxes), tracked=len(ids))
        matched, duplicate, stray = assign(px, boxes, pad)

        for i, pid in enumerate(ids):
            pc = rep.players.setdefault(pid, PlayerCheck(player_id=pid))
            pc.frames += 1
            if i in matched:
                fc.on_player += 1
                pc.on_player += 1
            elif i in duplicate:
                fc.duplicate += 1
                pc.duplicate += 1
            else:
                fc.stray += 1
                miss = min((_distance_to(px[i], b) for b in boxes), default=0.0) \
                    if np.isfinite(px[i]).all() else 0.0
                pc.mean_px += miss
                fc.worst_px = max(fc.worst_px, miss)

        fc.unclaimed = len(boxes) - len(matched)
        rep.frames.append(fc)

    for pc in rep.players.values():
        miss = pc.frames - pc.on_player
        pc.mean_px = pc.mean_px / miss if miss else 0.0
    return rep


def _to_image(registration, frame_index: int, field_pts):
    h = registration.at(frame_index)
    if h is None:
        return np.full((len(field_pts), 2), np.nan)
    return h.to_image(field_pts)


def side_by_side(
    out_path,
    track: PlayTrack,
    registration,
    frames: dict,
    boxes_by_frame: dict[int, list],
    spec,
    fps: float = 30.0,
    frame_offset: int = 0,
    out_fps: int = 12,
    pad: float = 6.0,
    team_colors: dict | None = None,
    ball_seen_at=None,
):
    """The clip with the recreation drawn on it, above the top-down view.

    This is the artifact worth producing on every run. The numbers above say *how much*
    agrees; this says *which player, on which frame*, and it needs no ground truth to
    read — a filled dot sitting on a player is right, a red ring on grass is wrong, and
    anyone can see the difference without being told what to look for.

    The two panels share a clock and an orientation: the top-down view is drawn with
    the far sideline at the top, matching how the camera sees it, so a player in one
    panel is in the same place in the other. Flipping one relative to the other makes
    the whole thing unreadable, which is a mistake worth not repeating.
    """
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise ImportError("this needs opencv") from e
    from pathlib import Path

    COL = team_colors or {"teamA": (59, 169, 242), "teamB": (232, 168, 79)}
    any_frame = next(iter(frames.values()))
    fh, fw = any_frame.shape[:2]

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    SEP = PANEL_RULE_PX
    vw = cv2.VideoWriter(str(out_path), cv2.VideoWriter_fourcc(*"mp4v"), out_fps,
                         (fw, fh * 2 + SEP))
    rule = np.full((SEP, fw, 3), (60, 60, 60), np.uint8)

    PADP = 30
    sx = lambda x: int(PADP + (x / spec.length) * (fw - 2 * PADP))          # noqa: E731
    sy = lambda y: int(PADP + ((spec.width - y) / spec.width) * (fh - 2 * PADP))  # noqa: E731

    board = np.full((fh, fw, 3), (22, 26, 20), np.uint8)
    cv2.rectangle(board, (sx(0), sy(spec.width)), (sx(spec.length), sy(0)), (38, 64, 34), -1)
    for a, b in ((0, spec.goal_a), (spec.goal_b, spec.length)):
        cv2.rectangle(board, (sx(a), sy(spec.width)), (sx(b), sy(0)), (28, 50, 26), -1)
    for x in np.arange(spec.goal_a, spec.goal_b + 0.01, 5.0):
        major = int(x) % 10 == 0
        cv2.line(board, (sx(x), sy(0)), (sx(x), sy(spec.width)),
                 (150, 170, 150) if major else (92, 112, 92), 2 if major else 1)
        if major and spec.goal_a < x < spec.goal_b:
            # The painted number, so a position in this panel can be read off against the
            # same number in the frame above it rather than counted out in stripes.
            n = int(round(50.0 - abs(x - 60.0)))
            for yy in (spec.width - 9.0, 9.0):
                t = str(n)
                w_ = cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.5, 1)[0][0]
                cv2.putText(board, t, (sx(x) - w_ // 2, sy(yy)),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (128, 148, 128), 1, cv2.LINE_AA)
    for x in (0, spec.goal_a, spec.goal_b, spec.length):
        cv2.line(board, (sx(x), sy(0)), (sx(x), sy(spec.width)), (215, 230, 215), 2)
    lo, hi = spec.hashes
    for x in np.arange(spec.goal_a + 1, spec.goal_b, 1.0):
        for hy in (lo, hi):
            cv2.line(board, (sx(x), sy(hy) - 3), (sx(x), sy(hy) + 3), (108, 128, 108), 1)

    def label(img, lines_, corner="tl"):
        """A caption block on a strip of its own, so it never sits on the play.

        The first version wrote three captions at fixed heights over the picture and they
        landed on top of each other and on the players. A panel you have to squint past is
        a panel nobody checks.
        """
        pad, lh = 10, 20
        wid = max(cv2.getTextSize(t, cv2.FONT_HERSHEY_SIMPLEX, 0.48, 1)[0][0] for t in lines_) + 2 * pad
        hgt_ = lh * len(lines_) + pad
        x0 = 8 if corner.endswith("l") else img.shape[1] - wid - 8
        y0 = 8
        strip = img[y0:y0 + hgt_, x0:x0 + wid]
        if strip.size:
            img[y0:y0 + hgt_, x0:x0 + wid] = (0.35 * strip).astype(np.uint8)
        for i, t in enumerate(lines_):
            cv2.putText(img, t, (x0 + pad, y0 + lh * (i + 1) - 4),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.48, (238, 238, 238), 1, cv2.LINE_AA)

    try:
        for f in track.frames:
            src = int(round(f.t * fps)) + frame_offset
            if src not in frames:
                continue
            left = frames[src].copy()
            boxes = boxes_by_frame.get(src, [])
            for b in boxes:
                cv2.rectangle(left, (int(b[0]), int(b[1])), (int(b[2]), int(b[3])), (70, 70, 70), 1)

            h = registration.at(src) if hasattr(registration, "at") else None
            on = off = 0
            if h is not None and f.players:
                ids = list(f.players)
                px = np.atleast_2d(h.to_image([[f.players[i].x, f.players[i].y] for i in ids]))
                # The same assignment the scorer uses. Drawing it a second way is how a
                # picture ends up disagreeing with the number printed under it.
                matched, duplicate, stray = assign(px, boxes, pad)
                dupes = len(duplicate)
                for i, (pid, p) in enumerate(zip(ids, px)):
                    if not np.isfinite(p).all():
                        continue
                    team = track.roster[pid].team if pid in track.roster else "other"
                    c = COL.get(team, (150, 150, 150))
                    hit = i in matched
                    on, off = on + int(hit), off + int(not hit)
                    # A ring on the ground the player is standing on, not a dot floating
                    # at his feet. The position being drawn *is* the ground point, and a
                    # flat ellipse says so — a filled circle there reads as a mislaid
                    # marker even when it is exactly right, which cost an hour of looking
                    # for a vertical bias that was not there (measured: 0.1 px).
                    rx = _player_radius(h.to_image, f.players[pid])
                    ctr = tuple(p.astype(int))
                    if hit:
                        cv2.ellipse(left, ctr, (rx, max(3, rx // 3)), 0, 0, 360, c, 2, cv2.LINE_AA)
                        cv2.circle(left, ctr, 3, c, -1, cv2.LINE_AA)
                    elif i in duplicate:
                        # On a real person, but one already held by another track. Drawn
                        # in the team colour so it is findable, crossed in magenta so it
                        # is never mistaken for a player who is genuinely there.
                        cv2.ellipse(left, ctr, (rx, max(3, rx // 3)), 0, 0, 360, c, 1, cv2.LINE_AA)
                        cv2.line(left, (ctr[0] - rx, ctr[1]), (ctr[0] + rx, ctr[1]),
                                 (255, 0, 255), 2, cv2.LINE_AA)
                    else:
                        cv2.ellipse(left, ctr, (rx, max(3, rx // 3)), 0, 0, 360,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.line(left, (ctr[0] - rx, ctr[1] - rx), (ctr[0] + rx, ctr[1] + rx),
                                 (0, 0, 255), 2, cv2.LINE_AA)
            label(left, [
                "THE CLIP, with the recreation drawn back onto it",
                "team ring = holds a player of its own    magenta bar = a second track on "
                "somebody already held    red cross = lands on nobody",
                f"{on} of {on + off} players held, {dupes} duplicate"
                + f"   t {f.t:+.1f}s",
            ])

            right = board.copy()
            for pid, p in f.players.items():
                team = track.roster[pid].team if pid in track.roster else "other"
                cv2.circle(right, (sx(p.x), sy(p.y)), 8, COL.get(team, (150, 150, 150)), -1, cv2.LINE_AA)
            if f.ball is not None:
                seen = ball_seen_at(f.t) if callable(ball_seen_at) else True
                cv2.circle(right, (sx(f.ball.x), sy(f.ball.y)), 9, (25, 35, 55), -1, cv2.LINE_AA)
                cv2.circle(right, (sx(f.ball.x), sy(f.ball.y)), 6,
                           (235, 245, 250) if seen else (120, 185, 215), -1, cv2.LINE_AA)
            label(right, [
                "THE RECREATION, same instant, seen from above",
                f"t {f.t:+.1f}s   {len(f.players)} players   far sideline at top, as filmed",
            ])

            vw.write(np.vstack([left, rule, right]))
    finally:
        vw.release()
    return out_path
