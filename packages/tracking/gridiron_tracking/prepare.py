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

The last step needs the first four: the throw is found by asking when the ball was
away from *every* player, and fifty-odd tracks include strays standing in the middle
of the field that a thrown ball would never escape.

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
from . import events as eventsmod
from . import sides as sidesmod
from .contract import PlayTrack

MIN_BALL_TRAVEL = 10.0
"""Yards a real ball covers in a play, at the very least."""

MIN_BALL_MEASURED = 0.25
"""How much of the ball track has to be a real detection rather than a coast.

Not tuned — there is one clip to tune on and it scores 5%, which is not near any
threshold. It is set where a reasonable person would stop believing a derived throw
instant, and it should be revisited the first time a detector clears it."""


@dataclass
class PrepareReport:
    roster: rostermod.RosterReport
    sides: sidesmod.Sides | None = None
    mirrored: bool = False
    line_of_scrimmage: float | None = None
    line_source: str | None = None
    """``broadcast`` (the drawn lines) or ``formation`` (where the teams lined up).
    The first is a surveyed line; the second is an estimate from the players."""
    has_ball: bool = False
    ball_measured: float = 0.0
    """Fraction of ball samples that were actually detected rather than coasted."""
    ball_coverage: float = 0.0
    events: Any = None
    """The derived pass window, or None when one was not looked for."""
    ball_travelled: float = 0.0
    """Yards between the extremes of the ball track. A ball that never moved is
    not a ball, however often it was detected."""
    reviewable: bool = False
    blockers: list[str] = dc_field(default_factory=list)
    notes: list[str] = dc_field(default_factory=list)
    """Things worth knowing that do not stop the review."""

    def table(self) -> str:
        lines = [self.roster.table(), ""]
        if self.sides is not None:
            lines.append(
                f"  possession        {self.sides.offense} has the ball "
                f"({self.sides.method}, confidence {self.sides.confidence})"
            )
        if self.line_of_scrimmage is not None:
            lines.append(f"  line of scrimmage x = {self.line_of_scrimmage:.1f} yd"
                         f"   (from the {self.line_source})")
        lines.append(f"  orientation       {'mirrored so the offense attacks +x' if self.mirrored else 'already attacking +x'}")
        lines.append(
            f"  ball track        " + (
                "ABSENT" if not self.has_ball else
                f"{self.ball_coverage:.0%} of frames, {self.ball_measured:.0%} seen, "
                f"travelled {self.ball_travelled:.1f} yd"
            )
        )
        if self.events is not None and self.events.found:
            lines.append(
                f"  pass window       thrown {self.events.thrown_at:.2f}s, arrived "
                f"{self.events.arrived_at:.2f}s, {self.events.flight_yards:.1f} yd "
                f"in the air"
            )
        lines.append("")
        lines.append(f"  reviewable        {'yes' if self.reviewable else 'NO'}")
        for b in self.blockers:
            lines.append(f"    - {b}")
        for n in self.notes:
            lines.append(f"  note: {n}")
        return "\n".join(lines)

    def to_json(self) -> dict[str, Any]:
        return {
            "roster": self.roster.to_json(),
            "sides": self.sides.to_json() if self.sides else None,
            "mirrored": self.mirrored,
            "line_of_scrimmage": self.line_of_scrimmage,
            "line_source": self.line_source,
            "has_ball": self.has_ball,
            "ball_measured": round(self.ball_measured, 3),
            "ball_coverage": round(self.ball_coverage, 3),
            "ball_travelled": round(self.ball_travelled, 2),
            "reviewable": self.reviewable,
            "events": self.events.to_json() if self.events is not None else None,
            "blockers": list(self.blockers),
            "notes": list(self.notes),
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

    # --- and if there was no broadcast to read ---------------------------------
    # A coach's upload has no virtual lines and no down-and-distance, so the line
    # has to come off the formation instead. It needs the offense named first, which
    # is why it runs after infer_sides rather than feeding it: pre-snap stillness
    # does not need a line, and the line does need a side.
    if los is None and report.sides.confidence > 0.0:
        formation = sidesmod.formation_line(out, report.sides.offense)
        if formation is not None and formation.confidence >= 0.5:
            los, attacks = formation.los, formation.attacks
            report.line_source = "formation"
            out.meta["formation_line"] = formation.to_json()
        elif formation is not None:
            report.blockers.append(
                f"the two front ranks are {formation.gap:.1f} yards apart at the "
                f"frame marked as the snap, which is not a snap. Check snap_frame "
                f"in the clip spec — if it equals start, nothing has identified one."
            )
    elif los is not None:
        report.line_source = "broadcast"
    if report.sides.confidence <= 0.0:
        report.blockers.append(
            "which side has the ball could not be established. With no ball track "
            "and no pre-snap frames there is nothing to decide it on — check that "
            "the clip's window opens before the snap, and that snap_frame is the "
            "snap rather than the first frame of the window."
        )
    elif report.sides.confidence < 0.9:
        # A note, not a blocker. presnap_stillness is a real method with a measured
        # accuracy of 0.812, and it is the only one available on footage with no
        # graphics — blocking on it means a coach's upload can never be reviewed.
        # It does mean roughly one play in five has the sides the wrong way round,
        # so it rides through meta into the review's assumptions rather than being
        # swallowed here.
        report.notes.append(
            f"possession is a {report.sides.confidence:.0%} call "
            f"({report.sides.method}); the sides being swapped would make the whole "
            f"review wrong, and nothing downstream can detect that on its own."
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
        report.notes.append(
            "no distance to go: this clip has no down-and-distance graphic to read "
            "it from. Set distance_yards in the spec if you know it. The review does "
            "not need it — decision_engine only uses the down to choose between "
            "recommending a run and recommending a punt."
        )

    # --- the ball -------------------------------------------------------------
    # Presence is not the test. A ball on a handful of frames is worse than no ball:
    # every derivation downstream will run, return a number, and be wrong, where an
    # absent ball at least stops the chain. The passer is read at the throw and the
    # targeted receiver at the arrival, so what matters is whether the ball was
    # actually SEEN often enough to locate those two instants.
    positioned = sum(1 for f in out.frames if f.ball is not None)
    report.has_ball = positioned > 0
    stats = out.meta.get("ball") or {}
    report.ball_measured = float(stats.get("measured_fraction") or 0.0)
    report.ball_coverage = positioned / len(out.frames) if out.frames else 0.0
    report.ball_travelled = float(stats.get("travelled_yards") or 0.0)

    if not report.has_ball:
        report.blockers.append(
            "no ball track, so there is no passer, no targeted receiver, no throw "
            "and no arrival. The review has nothing to key a forecast on. Set "
            "track.ball = true in the clip spec."
        )
    elif report.ball_travelled < MIN_BALL_TRAVEL:
        report.blockers.append(
            f"the ball track covers {report.ball_travelled:.1f} yards over the whole "
            f"play, which nothing thrown or carried does. It is a track of something "
            f"that does not move — on a televised game, the down-and-distance "
            f"graphic sits on the field near the line of scrimmage and the generic "
            f"sports-ball class finds it readily. A detection rate "
            f"({report.ball_measured:.0%} here) says how often something was found "
            f"and nothing about what."
        )
    elif report.ball_measured < MIN_BALL_MEASURED:
        report.blockers.append(
            f"the ball was actually seen on {report.ball_measured:.0%} of samples "
            f"({report.ball_coverage:.0%} of frames carry a position, the rest "
            f"coasted). Below {MIN_BALL_MEASURED:.0%} the throw and the arrival "
            f"cannot be located, and a review built on that would be confident and "
            f"meaningless. A football in a wide broadcast shot is a few pixels and "
            f"is behind somebody most of the play; the generic sports-ball class "
            f"was not trained on one."
        )

    # --- the throw and the arrival --------------------------------------------
    # Last, because it needs the ball and a consolidated roster: the flight test is
    # "far from every player", and fifty-odd tracks include strays standing in the
    # middle of the field that a thrown ball would never get away from.
    if report.has_ball and not report.blockers:
        found = eventsmod.apply(out)
        report.events = found
        if not found.found:
            report.blockers.append(
                "no throw could be found in the ball track: "
                + (found.notes[-1] if found.notes else "no reason given")
                + ". The review needs a throw and an arrival to forecast between."
            )

    report.reviewable = not report.blockers
    out.meta["prepare_report"] = report.to_json()
    return out, report
