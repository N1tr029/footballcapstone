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
