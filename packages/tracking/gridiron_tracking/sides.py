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


def infer_sides(track: PlayTrack, offense: str | None = None) -> Sides:
    """Work out which team has the ball.

    Pass ``offense`` whenever play-by-play is at hand — a benchmark should never be
    guessing this. The inference exists for the case a coach's upload has no metadata
    at all.
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
