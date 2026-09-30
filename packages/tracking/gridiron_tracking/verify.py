"""Does the recreation match the clip? Checked frame by frame, player by player.

There is no ground truth for a clip pulled off the internet, which seems to make
"is each player in the right place" unanswerable. It is not. The registration is a
two-way map: push a recovered field position *back* through it and you get the pixel
that position claims to be. If that pixel lands on a player in the actual frame, the
coordinate is right. If it lands on grass, on a coach, or in the crowd, it is wrong —
and you can see which, on which frame, for which player.

That turns a vague "it looks off" into three numbers per frame:

**on_player** — the recovered position reprojects inside a detection box. The
coordinate is consistent with what the detector saw.

**stray** — it reprojects onto nothing. Either the tracker is coasting a player who is
no longer there, or the position has drifted off the man it belongs to.

**unclaimed** — a detection with no track on it. Somebody on the field is not being
tracked.

None of this proves the *field* coordinates are correct in yards; a wrong-but-
self-consistent registration would still score well, which is why it sits alongside the
camera and turf checks rather than replacing them. What it does prove is that the
tracking, the smoothing and the resampling did not move anybody away from where the
pixels say they are — and that is the layer the eye notices first.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field

import numpy as np

from .contract import PlayTrack

PANEL_RULE_PX = 4
"""A hairline between the two panels. Without it the clip's own dark edge and the top of
the board run together and the pair reads as one picture."""


@dataclass
class FrameCheck:
    t: float
    frame_index: int
    on_player: int = 0
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
            f"  overall            {self.agreement * 100:.1f}% of tracked players land on a real person",
            f"  frames checked     {len(self.frames)}",
            f"  stray per frame    {np.mean([f.stray for f in self.frames]):.1f}"
            f"   (tracks sitting on nobody)",
            f"  unclaimed          {np.mean([f.unclaimed for f in self.frames]):.1f}"
            f"   (people on the field with no track)",
            "",
            "  worst frames:",
        ]
        for f in self.worst_frames():
            lines.append(f"    t {f.t:+.1f}s  frame {f.frame_index}: "
                         f"{f.on_player}/{f.tracked} on a player, {f.stray} stray, {f.unclaimed} unclaimed")
        lines += ["", "  worst players:"]
        for p in self.worst_players():
            lines.append(f"    {p.player_id:<10} {p.agreement * 100:5.1f}% over {p.frames} frames, "
                         f"mean miss {p.mean_px:.0f} px")
        return "\n".join(lines)


def _inside(px, box, pad: float = 6.0) -> bool:
    x1, y1, x2, y2 = box
    return (x1 - pad) <= px[0] <= (x2 + pad) and (y1 - pad) <= px[1] <= (y2 + pad)


def _distance_to(px, box) -> float:
    x1, y1, x2, y2 = box
    dx = max(x1 - px[0], 0.0, px[0] - x2)
    dy = max(y1 - px[1], 0.0, px[1] - y2)
    return float(np.hypot(dx, dy))


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
        claimed: set[int] = set()

        for pid, p in zip(ids, px):
            if not np.isfinite(p).all():
                fc.stray += 1
                continue
            hit = None
            best = np.inf
            for bi, b in enumerate(boxes):
                d = _distance_to(p, b)
                if d < best:
                    best, hit = d, bi
            pc = rep.players.setdefault(pid, PlayerCheck(player_id=pid))
            pc.frames += 1
            if hit is not None and _inside(p, boxes[hit], pad):
                fc.on_player += 1
                pc.on_player += 1
                claimed.add(hit)
            else:
                fc.stray += 1
                pc.mean_px += best
                fc.worst_px = max(fc.worst_px, best)

        fc.unclaimed = len(boxes) - len(claimed)
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
                for pid, p in zip(ids, px):
                    if not np.isfinite(p).all():
                        continue
                    team = track.roster[pid].team if pid in track.roster else "other"
                    c = COL.get(team, (150, 150, 150))
                    hit = any(_inside(p, b, pad) for b in boxes)
                    on, off = on + int(hit), off + int(not hit)
                    # A ring on the ground the player is standing on, not a dot floating
                    # at his feet. The position being drawn *is* the ground point, and a
                    # flat ellipse says so — a filled circle there reads as a mislaid
                    # marker even when it is exactly right, which cost an hour of looking
                    # for a vertical bias that was not there (measured: 0.1 px).
                    rx = 11
                    for b in boxes:
                        if _inside(p, b, pad):
                            rx = max(7, int((b[2] - b[0]) * 0.55))
                            break
                    ctr = tuple(p.astype(int))
                    if hit:
                        cv2.ellipse(left, ctr, (rx, max(3, rx // 3)), 0, 0, 360, c, 2, cv2.LINE_AA)
                        cv2.circle(left, ctr, 3, c, -1, cv2.LINE_AA)
                    else:
                        cv2.ellipse(left, ctr, (rx, max(3, rx // 3)), 0, 0, 360,
                                    (0, 0, 255), 2, cv2.LINE_AA)
                        cv2.line(left, (ctr[0] - rx, ctr[1] - rx), (ctr[0] + rx, ctr[1] + rx),
                                 (0, 0, 255), 2, cv2.LINE_AA)
            label(left, [
                "THE CLIP, with the recreation drawn back onto it",
                "team ring = lands on a detected player    red cross = lands on nobody",
                f"{on} of {on + off} agree" + (f"   t {f.t:+.1f}s" if True else ""),
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
