"""Fifty-odd tracks down to twenty-two people, and an honest account of the loss.

``stitch`` joins a track that ended to a track that started where the first one was
heading: fragments of one player, in sequence. It refuses to touch tracks that
overlap in time, and that refusal is right — two tracks on the field at the same
instant are usually two people, and merging them invents a teleport and loses a man.

What it leaves behind is the other failure, which is the one that breaks the play
model. On the reference clip the pipeline ends with **46 tracks for 22 players**:
some are duplicates, one person wearing two tracks at once; some are strays, a track
sitting on nobody for four frames; and the kit classifier abstains on a few, so the
roster carries a third team label. The verification report measures all of it —
17.2 of 22 people accounted for, 2.6 duplicate, 1.7 stray per frame.

A renderer survives that. It draws 46 dots instead of 22 and looks slightly wrong.
The play model does not survive it: it measures each receiver's separation from the
nearest defender, so a stray track standing on nobody is a defender who is not
there, and every receiver reads as covered. Worse, a third team label makes
``sides.infer_sides`` report zero confidence, and a caller who ignores that gets 44
"defenders" and 2 route runners.

So this is the pass that makes a vision track reviewable. Four steps, in order:

1. **Two teams.** Keep the two kit labels with the most samples; drop the rest.
2. **Strays.** Drop tracks that live for a small fraction of the play.
3. **Duplicates.** Merge same-team tracks that coexist within a yard of each other.
4. **Cap.** Keep the longest-lived eleven a side.

**Why same-team only, and why a yard.** Two tracks a yard apart for half a second
are one person — unless they are opposite numbers in coverage, who spend whole plays
that close. Restricting merges to one team removes that case. What it does not remove
is two linemen, who line up about a yard apart and stay there, so the threshold is
tighter than it looks and the overlap has to be sustained. Getting this wrong deletes
a real player, which is worse than leaving a duplicate: a duplicate is visible, and a
deleted guard is not.

**What this cannot do.** It cannot invent the five people the detector never found.
A 22-man roster assembled from 17 tracked people has five holes in it, and the model
measuring separation against nine defenders instead of eleven will report receivers
as more open than they were. :class:`RosterReport` says how many it actually ended
with so that bias is visible rather than inferred from a clean-looking animation.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Any

from .contract import PlayTrack, RosterEntry, TrackingFrame

PER_SIDE = 11
MIN_LIFE = 0.18
"""Drop a track present in less than this fraction of the play. The reference clip's
strays live for 1 to 6 frames of 50; real players who were only picked up late run to
twenty and more."""

DUP_YARDS = 0.9
"""How close two same-team tracks must stay to be called one person. Interior
linemen line up about a yard apart, so this is deliberately inside that."""

DUP_FRACTION = 0.6
"""And for what proportion of the frames they share."""

DUP_MIN_OVERLAP = 4
"""Frames they must share before the question is even asked. Two tracks that cross
for an instant are two people passing each other."""


@dataclass
class RosterReport:
    """What consolidation did, and what it could not do."""

    before: int
    after: int
    merged: int = 0
    dropped_unlabelled: int = 0
    dropped_short: int = 0
    dropped_surplus: int = 0
    per_side: dict[str, int] = dc_field(default_factory=dict)
    coverage: float = 0.0
    """Mean fraction of frames a kept player actually appears in."""
    notes: list[str] = dc_field(default_factory=list)

    @property
    def complete(self) -> bool:
        """Eleven a side. Rarely true on broadcast video, and worth knowing when."""
        return len(self.per_side) == 2 and all(n == PER_SIDE for n in self.per_side.values())

    def table(self) -> str:
        lines = [
            "roster consolidation",
            "",
            f"  tracks in         {self.before}",
            f"  players out       {self.after}" + ("" if self.complete else "   (not 22)"),
            f"  merged            {self.merged}   duplicate tracks folded into one person",
            f"  dropped           {self.dropped_unlabelled} unlabelled kit, "
            f"{self.dropped_short} too short, {self.dropped_surplus} surplus to eleven",
            f"  per side          " + ", ".join(f"{k}={v}" for k, v in sorted(self.per_side.items())),
            f"  mean coverage     {self.coverage:.0%} of frames per kept player",
        ]
        if not self.complete:
            missing = {k: PER_SIDE - v for k, v in self.per_side.items() if v < PER_SIDE}
            if missing:
                lines += [
                    "",
                    "  SHORT OF A FULL SIDE — " + ", ".join(
                        f"{n} missing from {k}" for k, n in sorted(missing.items())),
                    "  Separation to the nearest defender is measured against whoever was",
                    "  tracked, so a receiver will read as more open than he was.",
                ]
        if self.notes:
            lines += ["", *(f"  {n}" for n in self.notes)]
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        return {
            "before": self.before, "after": self.after, "merged": self.merged,
            "dropped_unlabelled": self.dropped_unlabelled,
            "dropped_short": self.dropped_short,
            "dropped_surplus": self.dropped_surplus,
            "per_side": dict(self.per_side), "coverage": round(self.coverage, 3),
            "complete": self.complete, "notes": list(self.notes),
        }


def consolidate(
    track: PlayTrack,
    per_side: int = PER_SIDE,
    min_life: float = MIN_LIFE,
    dup_yards: float = DUP_YARDS,
    dup_fraction: float = DUP_FRACTION,
) -> tuple[PlayTrack, RosterReport]:
    """A vision track with a roster the play model can reason about.

    Returns a new track and the report; the input is not modified. Safe to call on a
    track that is already clean — a measured release play comes back unchanged with
    a report saying nothing happened.
    """
    n_frames = len(track.frames)
    report = RosterReport(before=len(track.roster), after=len(track.roster))
    if not n_frames or not track.roster:
        report.notes.append("empty track; nothing to consolidate")
        return track, report

    life = {pid: len(track.series(pid)) for pid in track.roster}
    keep = set(track.roster)

    # --- 1. two teams --------------------------------------------------------
    samples: dict[str, int] = {}
    for pid in keep:
        samples[str(track.roster[pid].team)] = samples.get(str(track.roster[pid].team), 0) + life[pid]
    if len(samples) > 2:
        main = {t for t, _ in sorted(samples.items(), key=lambda kv: kv[1], reverse=True)[:2]}
        doomed = {pid for pid in keep if str(track.roster[pid].team) not in main}
        report.dropped_unlabelled = len(doomed)
        report.notes.append(
            f"roster carried {len(samples)} kit labels "
            f"({', '.join(sorted(samples))}); kept {', '.join(sorted(main))}. A third "
            f"label makes sides.infer_sides report zero confidence, which is how a "
            f"caller ends up with one stray track as the whole offense."
        )
        keep -= doomed

    # --- 2. strays -----------------------------------------------------------
    short = {pid for pid in keep if life[pid] < max(2, round(min_life * n_frames))}
    report.dropped_short = len(short)
    keep -= short

    # --- 3. duplicates -------------------------------------------------------
    merges = _duplicate_pairs(track, keep, dup_yards, dup_fraction)
    absorbed: dict[str, str] = {}
    for loser, winner in merges.items():
        root = winner
        while root in absorbed:
            root = absorbed[root]
        if root != loser:
            absorbed[loser] = root
    report.merged = len(absorbed)
    keep -= set(absorbed)

    # --- 4. eleven a side ----------------------------------------------------
    by_team: dict[str, list[str]] = {}
    for pid in keep:
        by_team.setdefault(str(track.roster[pid].team), []).append(pid)
    for team, members in by_team.items():
        members.sort(key=lambda p: life[p], reverse=True)
        if len(members) > per_side:
            report.dropped_surplus += len(members) - per_side
            for pid in members[per_side:]:
                keep.discard(pid)
            del members[per_side:]

    # --- rebuild -------------------------------------------------------------
    roster = {pid: RosterEntry(**vars(track.roster[pid])) for pid in sorted(keep, key=lambda p: -life[p])}
    frames = []
    for f in track.frames:
        players = {}
        for pid, p in f.players.items():
            target = pid
            while target in absorbed:
                target = absorbed[target]
            if target not in roster:
                continue
            # The surviving track's own sample wins; an absorbed one only fills a gap,
            # so a merge never moves a player who was already seen that frame.
            if target == pid or target not in players:
                players[target] = p
        frames.append(TrackingFrame(t=f.t, players=players, ball=f.ball, events=list(f.events)))

    out = PlayTrack(meta=dict(track.meta), roster=roster, frames=frames)
    out.meta["roster_consolidated"] = True

    report.after = len(roster)
    report.per_side = {t: len(m) for t, m in by_team.items() if m}
    present = [len(out.series(pid)) / n_frames for pid in roster]
    report.coverage = sum(present) / len(present) if present else 0.0
    out.meta["roster_report"] = report.to_json()
    return out, report


def _duplicate_pairs(
    track: PlayTrack, candidates: set[str], dup_yards: float, dup_fraction: float
) -> dict[str, str]:
    """``loser -> winner`` for same-team tracks that are one person.

    Both conditions have to hold over the frames the two actually share: close, and
    close for most of them. A pair that is close for a moment is two people passing,
    and the whole point of the fraction is to not merge them.
    """
    series = {pid: dict(track.series(pid)) for pid in candidates}
    order = sorted(candidates, key=lambda p: len(series[p]), reverse=True)
    merges: dict[str, str] = {}

    for i, a in enumerate(order):
        if a in merges:
            continue
        for b in order[i + 1:]:
            if b in merges:
                continue
            if str(track.roster[a].team) != str(track.roster[b].team):
                continue

            shared = set(series[a]) & set(series[b])
            if len(shared) < DUP_MIN_OVERLAP:
                continue
            close = sum(
                1 for t in shared
                if math.hypot(series[a][t].x - series[b][t].x,
                              series[a][t].y - series[b][t].y) <= dup_yards
            )
            if close / len(shared) >= dup_fraction:
                merges[b] = a
    return merges
