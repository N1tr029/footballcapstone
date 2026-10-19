"""A vision track, made into something the play model can be pointed at.

``render`` produces coordinates. The play model needs rather more than coordinates,
and none of it is in the video in any direct sense: a roster of twenty-two rather
than fifty-odd tracks, a statement of which side has the ball, a field oriented the
way the contract says, and the situation at the snap. Each of those has its own
module; this is the order they go in, which is not obvious and matters.

**Consolidate before deciding sides.** ``sides`` needs exactly two kit labels, and a
raw vision track carries three whenever the classifier abstained on a few
detections. With three it reports zero confidence and still returns a team name.

**Decide sides before mirroring.** Mirroring moves everybody; the line of scrimmage
has to move with them, and the possession test is "who is behind the line", so doing
it afterwards means doing it against a line that has not been moved yet.

**Mirror before anything reads a coordinate.** The contract fixes the offense
attacking +x. A broadcast clip has no ``playDirection`` column, so the pipeline's
output runs whichever way the camera saw it, and the play model takes direction as a
*feature* — a mirrored play is scored against the wrong half of what it learnt.

What this cannot supply is the ball. Without a ball track there is no passer, no
targeted receiver, no throw and no arrival, and the review has nothing to key a
forecast on. :class:`PrepareReport` says so rather than letting the chain fail three
modules later with something less specific.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any

from . import field as fieldmod
from . import roster as rostermod
from . import sides as sidesmod
from .contract import PlayTrack


@dataclass
class PrepareReport:
    roster: rostermod.RosterReport
    sides: sidesmod.Sides | None = None
    mirrored: bool = False
    line_of_scrimmage: float | None = None
    has_ball: bool = False
    reviewable: bool = False
    blockers: list[str] = dc_field(default_factory=list)

    def table(self) -> str:
        lines = [self.roster.table(), ""]
        if self.sides is not None:
            lines.append(
                f"  possession        {self.sides.offense} has the ball "
                f"({self.sides.method}, confidence {self.sides.confidence})"
            )
        if self.line_of_scrimmage is not None:
            lines.append(f"  line of scrimmage x = {self.line_of_scrimmage:.1f} yd")
        lines.append(f"  orientation       {'mirrored so the offense attacks +x' if self.mirrored else 'already attacking +x'}")
        lines.append(f"  ball track        {'present' if self.has_ball else 'ABSENT'}")
        lines.append("")
        lines.append(f"  reviewable        {'yes' if self.reviewable else 'NO'}")
        for b in self.blockers:
            lines.append(f"    - {b}")
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        return {
            "roster": self.roster.to_json(),
            "sides": self.sides.to_json() if self.sides else None,
            "mirrored": self.mirrored,
            "line_of_scrimmage": self.line_of_scrimmage,
            "has_ball": self.has_ball,
            "reviewable": self.reviewable,
            "blockers": list(self.blockers),
        }


def prepare(track: PlayTrack, offense: str | None = None) -> tuple[PlayTrack, PrepareReport]:
    """Consolidate, side, orient and situate. Returns a new track and the report.

    ``offense`` overrides the inference, as the team is spelled in the roster. Pass
    it whenever a human knows; the inference exists for an upload that came with no
    metadata at all.

    A track that is already a release play passes through almost untouched: its
    roster is clean, it is already oriented, and its meta already carries the
    situation.
    """
    out, roster_report = rostermod.consolidate(track)
    report = PrepareReport(roster=roster_report)

    situation = out.meta.get("broadcast") or {}
    los = situation.get("lineOfScrimmage")
    attacks = situation.get("attacks") or out.meta.get("play_direction")

    # --- who has the ball ----------------------------------------------------
    report.sides = sidesmod.infer_sides(out, offense=offense, los_x=los, attacks=attacks)
    if report.sides.confidence <= 0.0:
        report.blockers.append(
            "which side has the ball could not be established. With no ball track "
            "and no pre-snap frames there is nothing to decide it on — check that "
            "the clip's window opens before the snap, and that snap_frame is the "
            "snap rather than the first frame of the window."
        )
    elif report.sides.confidence < 0.9:
        report.blockers.append(
            f"possession is a weak call ({report.sides.method}, "
            f"{report.sides.confidence}); everything downstream inherits that."
        )

    # --- which way is downfield ----------------------------------------------
    if attacks == "left":
        fieldmod.mirror_track(out)
        report.mirrored = True
        if los is not None:
            los = fieldmod.LENGTH - los
    elif attacks != "right":
        report.blockers.append(
            "which way the offense attacks is unknown, so the play may be mirrored. "
            "It comes free from the broadcast's own lines when the clip spec sets "
            "distance_yards — the first-down line is always downfield of the "
            "line of scrimmage."
        )
    out.meta.setdefault("play_direction", "right" if attacks == "right" else attacks)

    # --- the situation --------------------------------------------------------
    if report.sides is not None and report.sides.confidence > 0.0:
        out.meta["possessionTeam"] = report.sides.offense
        out.meta["defensiveTeam"] = report.sides.defense
    if los is not None:
        out.meta["lineOfScrimmage"] = round(los, 2)
        report.line_of_scrimmage = los
    if situation.get("yardsToGo") is not None:
        out.meta["yardsToGo"] = situation["yardsToGo"]
        if los is not None:
            out.meta["firstDownLine"] = round(min(los + situation["yardsToGo"], 110.0), 2)
    else:
        report.blockers.append(
            "no distance to go; set distance_yards in the clip spec from the "
            "down-and-distance graphic"
        )

    # --- the ball -------------------------------------------------------------
    report.has_ball = any(f.ball is not None for f in out.frames)
    if not report.has_ball:
        report.blockers.append(
            "no ball track, so there is no passer, no targeted receiver, no throw "
            "and no arrival. The review has nothing to key a forecast on. Set "
            "track.ball = true in the clip spec."
        )

    report.reviewable = not report.blockers
    out.meta["prepare_report"] = report.to_json()
    return out, report
