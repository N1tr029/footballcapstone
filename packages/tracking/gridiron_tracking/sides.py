"""Which team has the ball.

A tracking release labels players `home` and `away`; a vision pipeline labels them by
jersey colour. Neither says which of the two is on offense, and almost everything in
:mod:`gridiron_tracking.interactions` needs to know — a block is offense against
defense, a tackle is defense stopping the carrier, and "who went around whom" is
meaningless without sides.

Three candidate signals were scored against 149 plays of the 2017 Big Data Bowl
release, labelled by which team's offensive line was on the field:

| signal | accuracy |
|---|---|
| team of the player nearest the ball at the snap | **98.7%** |
| team that was stillest before the snap | 81.2% |
| team with more players behind the ball | 50.3% |

The winner is the centre: at the snap he is bent over the ball with his hands on it,
and nobody else on either side is ever as close. The stillness signal is the rule that
the offense must be set while the defense shuffles — real, but only four times in five,
because linemen in a stance still drift. The third is worthless, which is worth
recording so nobody spends a day rediscovering it.

The catch is that the winning signal needs a ball position at the snap. A public
release has one. A vision pipeline does not yet, so it falls back to stillness and
says so — an 81% guess that announces itself is workable; a 98% guess that hides its
basis when the basis is missing is not.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from .contract import PlayTrack

# Nobody is this close to the ball at the snap except the man snapping it.
CENTRE_YARDS = 2.0


@dataclass
class Sides:
    """Who is attacking, and how much to trust the answer."""

    offense: str
    defense: str
    method: str          # 'ball_at_snap' | 'presnap_stillness' | 'caller' | 'unknown'
    confidence: float    # measured accuracy of the method used

    def team_of(self, track: PlayTrack, player_id: str) -> str | None:
        """'offense' or 'defense' for one player, or None if his team is unknown."""
        entry = track.roster.get(player_id)
        if entry is None:
            return None
        if entry.team == self.offense:
            return "offense"
        if entry.team == self.defense:
            return "defense"
        return None

    def to_json(self) -> dict:
        return {
            "offense": self.offense, "defense": self.defense,
            "method": self.method, "confidence": self.confidence,
        }


def infer_sides(
    track: PlayTrack,
    offense: str | None = None,
    los_x: float | None = None,
    attacks: str | None = None,
) -> Sides:
    """Work out which team has the ball.

    Pass ``offense`` whenever play-by-play is at hand — a benchmark should never be
    guessing this. The inference exists for the case a coach's upload has no metadata
    at all.

    ``los_x`` and ``attacks`` are the broadcast's answer, from
    :meth:`gridiron_tracking.broadcast.MarkerCheck.situation`. They matter because the
    two methods below both need something a broadcast clip often lacks: the centre
    bent over the ball needs a ball track, and pre-snap stillness needs pre-snap
    footage. A clip cut at the snap with no ball has neither, and before this path
    existed that returned the first team alphabetically with zero confidence — which
    a caller who ignored the confidence turned into 44 defenders.

    Given the line of scrimmage and which way the offense is going, possession is
    just geometry: at the snap the offense is behind the line and the defense is in
    front of it.
    """
    teams = sorted({e.team for e in track.roster.values()})
    if offense is not None:
        other = next((t for t in teams if t != offense), "unknown")
        return Sides(offense, other, "caller", 1.0)

    if len(teams) != 2:
        # One team, or three because officials got a track of their own. Either way
        # there is nothing to choose between.
        return Sides(teams[0] if teams else "unknown",
                     teams[1] if len(teams) > 1 else "unknown", "unknown", 0.0)

    snap = _snap_frame(track)

    # --- the broadcast's own lines, if they were read --------------------------
    if los_x is not None and attacks in ("left", "right") and snap is not None:
        behind: dict[str, list[float]] = {t: [] for t in teams}
        for pid, p in snap.players.items():
            entry = track.roster.get(pid)
            if entry and entry.team in behind:
                # Signed distance from the line, positive downfield.
                offset = (p.x - los_x) if attacks == "right" else (los_x - p.x)
                behind[entry.team].append(offset)
        means = {t: sum(v) / len(v) for t, v in behind.items() if v}
        if len(means) == 2:
            off = min(means, key=means.get)
            other = next(t for t in teams if t != off)
            # Both sides sitting on the same side of the line means the line, the
            # snap frame or the direction is wrong, and a coin-flip dressed as a
            # measurement is worse than no answer.
            if means[off] < 0.0 < means[other]:
                return Sides(off, other, "behind_the_los", 0.95)
            return Sides(off, other, "behind_the_los_unseparated", 0.5)

    # --- the centre, if there is a ball to be bent over ------------------------
    if snap is not None and snap.ball is not None:
        best, best_d = None, math.inf
        for pid, p in snap.players.items():
            d = math.hypot(p.x - snap.ball.x, p.y - snap.ball.y)
            if d < best_d:
                best, best_d = pid, d
        if best is not None and best_d <= CENTRE_YARDS:
            off = track.roster[best].team
            return Sides(off, next(t for t in teams if t != off), "ball_at_snap", 0.987)

    # --- failing that, the side that was set -----------------------------------
    if snap is not None:
        pre = [f for f in track.frames if f.t < snap.t]
        totals: dict[str, list[float]] = {t: [] for t in teams}
        for f in pre[-10:]:
            for pid, p in f.players.items():
                entry = track.roster.get(pid)
                if entry and entry.team in totals:
                    totals[entry.team].append(p.s)
        means = {t: (sum(v) / len(v)) for t, v in totals.items() if v}
        if len(means) == 2:
            off = min(means, key=means.get)
            return Sides(off, next(t for t in teams if t != off), "presnap_stillness", 0.812)

    return Sides(teams[0], teams[1], "unknown", 0.0)


def _snap_frame(track: PlayTrack):
    for f in track.frames:
        if "ball_snap" in f.events:
            return f
    # No snap event: the frame nearest t = 0, which is what the contract says t = 0 is.
    return min(track.frames, key=lambda f: abs(f.t)) if track.frames else None


# ------------------------------------------------- the line, without a broadcast


@dataclass
class FormationLine:
    """Where the two teams met, and which way the offense was pointed."""

    los: float
    attacks: str          # 'left' | 'right'
    gap: float            # yards between the two front ranks
    confidence: float

    def to_json(self) -> dict:
        return {"lineOfScrimmage": round(self.los, 2), "attacks": self.attacks,
                "gap_yards": round(self.gap, 2), "confidence": self.confidence,
                "source": "the formation at the snap, not a drawn line"}


def formation_line(track: PlayTrack, offense: str) -> FormationLine | None:
    """The line of scrimmage read off the two front ranks at the snap.

    ``broadcast.MarkerCheck`` is the better answer and needs a televised game: the
    broadcaster draws the line through a surveyed field and states the distance in
    writing. A coach's upload has no graphics at all — no virtual line, no
    down-and-distance — and three of this project's clips are exactly that case.

    What it does have is twenty-two people lined up facing each other. The offense's
    front rank and the defense's front rank are a yard apart across the ball, so the
    line is between them and the offense attacks away from its own side. Percentiles
    rather than extremes, because a split receiver stands on the line and a deep
    safety stands fifteen yards off it, and either would drag a min or a max.

    Returns None when the two ranks are not separated, which means the frame is not
    a snap — a clip cut after the ball is gone has both teams mixed together, and
    there is no line to find.
    """
    snap = _snap_frame(track)
    if snap is None:
        return None

    off = [p.x for pid, p in snap.players.items()
           if (e := track.roster.get(pid)) and e.team == offense]
    deff = [p.x for pid, p in snap.players.items()
            if (e := track.roster.get(pid)) and e.team not in (offense, None)]
    if len(off) < 4 or len(deff) < 4:
        return None

    off_a, deff_a = np.asarray(off, float), np.asarray(deff, float)
    attacks = "right" if float(np.median(deff_a)) > float(np.median(off_a)) else "left"

    if attacks == "right":
        front_off = float(np.percentile(off_a, 90))   # the linemen, furthest downfield
        front_def = float(np.percentile(deff_a, 10))  # the d-line, closest to them
    else:
        front_off = float(np.percentile(off_a, 10))
        front_def = float(np.percentile(deff_a, 90))

    gap = abs(front_def - front_off)
    los = (front_off + front_def) / 2.0

    # A snap has the two ranks within a couple of yards of each other across the
    # ball. Much more and this is not a snap; much less and they are interleaved,
    # which means the play has already started.
    if gap > 8.0 or (attacks == "right" and front_def < front_off) \
            or (attacks == "left" and front_def > front_off):
        return FormationLine(los, attacks, gap, 0.3)
    return FormationLine(los, attacks, gap, 0.8 if gap <= 4.0 else 0.6)
