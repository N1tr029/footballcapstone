"""Two tracks that are one player, joined back together.

The tracker is online: it sees each frame once, and when a player disappears — into a
pile, behind an official, out of the detector's reach for a third of a second — it has no
choice but to let him go and to call whoever emerges a new man. On the reference clip that
turns roughly twenty-four people into fifty-odd tracks, and no amount of gate or age
tuning fixes it: the grid was swept and identity churn never fell below about 1.9 tracks
per player without agreement collapsing, because the information needed to join them is
not available until the play is over.

Afterwards it is. A track that ends and a track that starts a moment later, where the
first one was heading, at a speed a person can run, is one player. That is a judgement
that needs the future, which is exactly what an online tracker does not have and an
offline pass does.

Two rules keep it honest.

**Overlapping tracks are never merged.** If both are on the field at the same instant they
are two people, however close together, and joining them would invent a teleport and lose
a real player.

**The prediction has to be met, not merely approached.** The gap is bridged with the first
track's own final velocity, and the second track has to start within a few yards of where
that puts it. Joining on proximity alone glues together whoever happens to stand near a
pile, which is worse than leaving the fragments: a fragment is visibly a fragment, and a
wrong merge is a confident lie about who ran where.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .contract import PlayTrack


@dataclass
class StitchReport:
    merged: int = 0
    before: int = 0
    after: int = 0
    joins: list[tuple[str, str, float, float]] = None  # (kept, absorbed, gap_s, miss_yd)

    def summary(self) -> str:
        return (f"stitched {self.before} tracks into {self.after} "
                f"({self.merged} join{'' if self.merged == 1 else 's'})")


def _series(track: PlayTrack, pid: str):
    ts, xy = [], []
    for f in track.frames:
        p = f.players.get(pid)
        if p is not None:
            ts.append(f.t)
            xy.append((p.x, p.y))
    return np.array(ts), np.array(xy)


def stitch(
    track: PlayTrack,
    max_gap_s: float = 1.2,
    gate_yards: float = 3.5,
    max_speed: float = 11.0,
) -> StitchReport:
    """Join fragments of the same player. Mutates ``track``; returns what it did.

    ``gate_yards`` is measured against the *predicted* position, not the last seen one, so
    a receiver who kept running through the gap is joined and a man standing three yards
    from where somebody vanished is not.

    ``max_speed`` is a sanity bound in yards per second — comfortably above a fast human
    and well below a tracking error — and it rejects a join that would require sprinting
    the length of the gap.
    """
    rep = StitchReport(joins=[])
    ids = list(track.roster)
    rep.before = len(ids)
    if len(ids) < 2:
        rep.after = rep.before
        return rep

    span = {}
    for pid in ids:
        ts, xy = _series(track, pid)
        if len(ts) >= 2:
            span[pid] = (ts, xy)
    order = sorted(span, key=lambda p: span[p][0][0])

    # Where a track ends up, so a chain of three fragments joins into one rather than two.
    alias: dict[str, str] = {}

    def root(p: str) -> str:
        while p in alias:
            p = alias[p]
        return p

    for bi, b in enumerate(order):
        tb, xb = span[b]
        best, best_miss = None, np.inf
        for a in order[:bi]:
            a = root(a)
            if a == b or a not in span:
                continue
            ta, xa = span[a]
            gap = tb[0] - ta[-1]
            if not (0 < gap <= max_gap_s):
                continue
            # Both on the field at once means two people, whatever the distance.
            if ta[-1] >= tb[0]:
                continue
            n = min(3, len(ta) - 1)
            v = (xa[-1] - xa[-1 - n]) / max(ta[-1] - ta[-1 - n], 1e-6)
            sp = float(np.linalg.norm(v))
            if sp > max_speed:
                v = v * (max_speed / sp)
            pred = xa[-1] + v * gap
            miss = float(np.linalg.norm(xb[0] - pred))
            need = float(np.linalg.norm(xb[0] - xa[-1])) / max(gap, 1e-6)
            if miss <= gate_yards and need <= max_speed and miss < best_miss:
                best, best_miss = a, miss

        if best is None:
            continue
        keep = root(best)
        # Fold b into keep, and extend keep's span so a third fragment can chain on.
        for f in track.frames:
            p = f.players.pop(b, None)
            if p is not None and keep not in f.players:
                f.players[keep] = p
        track.roster.pop(b, None)
        alias[b] = keep
        ts, xy = _series(track, keep)
        span[keep] = (ts, xy)
        span.pop(b, None)
        rep.merged += 1
        rep.joins.append((keep, b, float(tb[0] - span[keep][0][0]), best_miss))

    rep.after = len(track.roster)
    return rep
