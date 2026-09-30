"""What happened between the players, derived from where they were.

The thing worth understanding about this module: **it does not need video.** Once
you have twenty-two trajectories, "who hit whom", "who went around whom" and "who
had the ball" stop being computer-vision problems and become geometry over the
coordinates. So this layer can be built, tested and trusted against real public
tracking today, and when the vision pipeline lands it inherits all of it for free.

That split is the point. Layer one is coordinates and is hard and uncertain. Layer
two is everything a coach would actually say about the play, and it is deterministic.
Getting the layers the wrong way round — trying to detect a tackle in pixels —
is how this kind of project runs out of semester.

One honest limit, stated where it is easy to find. :func:`ball_in_view` answers
"when did he see the ball", and it is an **inference, not a measurement**. No public
football dataset contains head or eye tracking; body orientation is the closest
available proxy and a man can look somewhere his chest is not pointing. Every result
carries the ``o_source`` it was computed from so a consumer can tell how much of the
answer is data. See docs/animation.md for what happens when that distinction is lost.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field as dc_field
from typing import Any, Iterable

from .contract import PlayTrack, TrackingFrame, shortest_deg

# --- thresholds, all in yards and seconds -------------------------------------

CONTACT_ENTER = 1.50
"""Centre-to-centre distance at which two players are touching. A man in pads is
about three-quarters of a yard across the shoulders, and linemen engage with their
arms extended, so contact starts further out than two bodies touching would suggest."""

CONTACT_EXIT = 2.20
"""Hysteresis. A single threshold makes a wrestling match on the line flicker into
forty separate contacts as the distance jitters across it."""

GRACE_S = 0.35
"""How long a pair must stay apart before their engagement is considered over.
Without it, one jittery frame ends a block and starts a new one."""

DOWN_YPS = 1.4
"""Below this the carrier is going down rather than being carried backwards.

Contact and the tackle are not the same instant, and conflating them costs most of a
second. Measured on 63 plays of the 2017 release, the gap between its own
``first_contact`` and ``tackle`` labels has a median of 0.90 s — a man wrapped up
keeps his feet for the best part of a second. So an engagement records both: it starts
at contact, and ``t_down`` is when the carrier's speed collapses. Measured against
those labels over 68 plays: contact found on 100%, ``t_start`` a median 0.20 s from
``first_contact``, ``t_down`` a median 0.30 s from ``tackle``."""

MIN_CONTACT_S = 0.2
"""Shorter than this is two men passing close, not an engagement."""

BLOCK_S = 0.6
"""A sustained engagement. Below this it is a collision; above it, someone is being
held up."""

CARRIER_YARDS = 1.6
"""Within this of the ball, a player is carrying it. Beyond it for everyone, the
ball is in the air or on the ground."""

EVADE_YARDS = 3.5
"""How close the carrier has to come to a defender for going past him to be a move
rather than a coincidence."""

EVADE_TURN_DEG = 25.0
"""And how much he has to change direction for it to have been deliberate."""

VIEW_CONE_DEG = 100.0
"""Half-angle of the cone a player is treated as able to see. Human vision is about
100 degrees either side of straight ahead including the periphery; the head can turn
further, which is precisely the error this module refuses to make silently."""


# --- results ------------------------------------------------------------------


@dataclass
class Engagement:
    """Two players in contact, for a while."""

    a: str
    b: str
    t_start: float
    t_end: float
    closest_yards: float
    kind: str                    # 'block' | 'collision' | 'tackle' | 'tackle_attempt'
    involved_carrier: bool
    a_speed_drop: float          # yd/s lost across the engagement
    b_speed_drop: float
    t_down: float | None = None
    """When the carrier's speed collapsed, for a tackle. ``t_start`` is first contact;
    this is the tackle itself, and they are typically most of a second apart."""

    @property
    def duration(self) -> float:
        return self.t_end - self.t_start

    def to_json(self) -> dict[str, Any]:
        return {
            "a": self.a, "b": self.b, "t_start": round(self.t_start, 3),
            "t_end": round(self.t_end, 3), "duration": round(self.duration, 3),
            "closest_yards": round(self.closest_yards, 2), "kind": self.kind,
            "involved_carrier": self.involved_carrier,
            "a_speed_drop": round(self.a_speed_drop, 2),
            "b_speed_drop": round(self.b_speed_drop, 2),
            "t_first_contact": round(self.t_start, 3),
            "t_down": None if self.t_down is None else round(self.t_down, 3),
        }


@dataclass
class Evade:
    """The carrier got past someone without being touched."""

    carrier: str
    beaten: str
    t: float
    closest_yards: float
    turn_deg: float

    def to_json(self) -> dict[str, Any]:
        return {
            "carrier": self.carrier, "beaten": self.beaten, "t": round(self.t, 3),
            "closest_yards": round(self.closest_yards, 2),
            "turn_deg": round(self.turn_deg, 1),
        }


@dataclass
class Possession:
    """A stretch of the play in which one player held the ball."""

    player: str | None           # None means the ball was in the air or loose
    t_start: float
    t_end: float

    def to_json(self) -> dict[str, Any]:
        return {
            "player": self.player,
            "t_start": round(self.t_start, 3),
            "t_end": round(self.t_end, 3),
        }


@dataclass
class ViewWindow:
    """A stretch in which the ball was plausibly inside a player's field of view.

    ``confidence`` is not a probability. It is the weakest ``o_source`` that went
    into the window, propagated so that a consumer cannot mistake an inference built
    on a guessed body angle for one built on a measured column.
    """

    player: str
    t_start: float
    t_end: float
    confidence: str              # the o_source this was derived from
    max_offset_deg: float

    def to_json(self) -> dict[str, Any]:
        return {
            "player": self.player, "t_start": round(self.t_start, 3),
            "t_end": round(self.t_end, 3), "basis": self.confidence,
            "max_offset_deg": round(self.max_offset_deg, 1),
            "inferred": True,
        }


@dataclass
class Interactions:
    """Everything layer two knows about a play."""

    possessions: list[Possession] = dc_field(default_factory=list)
    engagements: list[Engagement] = dc_field(default_factory=list)
    evades: list[Evade] = dc_field(default_factory=list)
    views: list[ViewWindow] = dc_field(default_factory=list)
    sides: Any = None
    notes: list[str] = dc_field(default_factory=list)

    def carrier_at(self, t: float) -> str | None:
        for p in self.possessions:
            if p.t_start <= t <= p.t_end:
                return p.player
        return None

    def to_json(self) -> dict[str, Any]:
        return {
            "sides": self.sides.to_json() if self.sides is not None else None,
            "possessions": [p.to_json() for p in self.possessions],
            "engagements": [e.to_json() for e in self.engagements],
            "evades": [e.to_json() for e in self.evades],
            "ball_in_view": [v.to_json() for v in self.views],
            "notes": self.notes,
            "caveats": {
                "ball_in_view": (
                    "INFERRED from body orientation, not measured. No public football "
                    "dataset contains head or eye tracking. Check each window's "
                    "'basis' field before relying on it."
                ),
                "engagements": (
                    "Derived from proximity and deceleration, not from seeing contact. "
                    "Contact that leaves no trace in the coordinates — a hand on a "
                    "jersey while both men keep running — is missed. 'block' vs "
                    "'collision' is decided by duration, which is a heuristic; "
                    "'tackle' vs 'tackle_attempt' by whether the carrier lost speed. "
                    "t_start is FIRST CONTACT and t_down is the tackle itself: on the "
                    "2017 release those labels are a median 0.90 s apart, so do not "
                    "treat them as one moment."
                ),
            },
        }


# --- derivation ---------------------------------------------------------------


def derive(track: PlayTrack, offense: str | None = None) -> Interactions:
    """Everything, in one pass over a play.

    Pass ``offense`` (the team abbreviation or 'home'/'away') whenever play-by-play is
    available. Without it the sides are inferred, and everything that depends on
    knowing them inherits that uncertainty — which is recorded in ``sides``.
    """
    from .sides import infer_sides

    out = Interactions()
    if not track.frames:
        out.notes.append("empty play")
        return out

    out.sides = infer_sides(track, offense=offense)
    if out.sides.confidence < 0.9:
        out.notes.append(
            f"which team had the ball was inferred by {out.sides.method} "
            f"(~{out.sides.confidence * 100:.0f}% accurate); pass offense= to remove the guess"
        )

    has_ball = any(f.ball for f in track.frames)
    if has_ball:
        ok, why = ball_track_usable(track, sides=out.sides)
        out.notes.append(("ball track used: " if ok else "ball track rejected: ") + why)
        if not ok:
            out.notes.append("possession inferred from defensive convergence instead")
    else:
        out.notes.append(
            "no ball position on any frame, so possession was inferred from defensive "
            "convergence. The vision pipeline does not yet produce a ball track."
        )

    out.possessions = possessions(track, sides=out.sides)
    out.engagements = engagements(track, out.possessions, sides=out.sides)
    out.evades = evades(track, out.possessions, out.engagements)
    out.views = ball_in_view(track) if has_ball else []
    if not has_ball:
        out.notes.append("ball_in_view not derived: it needs a ball position")
    return out


def ball_track_usable(
    track: PlayTrack, threshold: float = CARRIER_YARDS, sides: Any = None,
) -> tuple[bool, str]:
    """Is the ball column actually following the ball?

    Worth asking, because in the 2017 Big Data Bowl release it is not. Its ``dis``
    column is all zeros and after the handoff the ball sits four to five yards from
    every player on the field — so nearest-player-to-ball names the centre before the
    snap, a penetrating defensive tackle during it, and nobody at all for the second
    half of the play. Trusting it produced a possession timeline in which a Patriots
    centre and a Chiefs defensive tackle shared the carries.

    The test is simple: through the live part of the play, somebody should almost
    always be holding the ball. If nobody is, the column is a spot marker or an
    artifact, not a track, and possession has to be inferred from the players instead.
    """
    live = [f for f in track.frames if f.t >= 0.0 and f.ball is not None]
    if not live:
        return False, "no ball position on any frame after the snap"

    held = 0
    offensive = 0
    for f in live:
        best, best_d = None, math.inf
        for pid, p in f.players.items():
            d = math.hypot(p.x - f.ball.x, p.y - f.ball.y)
            if d < best_d:
                best, best_d = pid, d
        if best_d <= threshold:
            held += 1
            if sides is not None and sides.team_of(track, best) == "offense":
                offensive += 1

    frac = held / len(live)
    if frac < 0.75:
        return False, (
            f"the ball is within {threshold} yd of a player on only {frac * 100:.0f}% "
            "of post-snap frames; this column is not tracking the ball"
        )

    # The sharper test, and the one that caught the 2017 release: whoever the ball
    # column implies is carrying should be on the team that has the ball. A column
    # that names a penetrating defensive tackle as the carrier is not a ball track,
    # however often somebody happens to be standing near it.
    if sides is not None and held:
        off_frac = offensive / held
        if off_frac < 0.85:
            return False, (
                f"the nearest player to the ball is on defense for {(1 - off_frac) * 100:.0f}% "
                "of the frames where anyone is near it; this column is a spot marker, "
                "not a ball track"
            )
    return True, f"ball is held by an offensive player on {frac * 100:.0f}% of post-snap frames"


def possessions(
    track: PlayTrack,
    threshold: float = CARRIER_YARDS,
    sides: Any = None,
) -> list[Possession]:
    """Who had the ball, when.

    Prefers the ball track — nearest player to it, with a gap meaning the ball is in
    flight, which is how a pass shows up. But only if the ball track survives
    :func:`ball_track_usable`; otherwise possession is inferred from the players via
    :func:`infer_carrier`, which is also the path the vision pipeline takes, since it
    produces no ball position at all.
    """
    ok, _ = (
        ball_track_usable(track, threshold, sides=sides)
        if any(f.ball for f in track.frames) else (False, "")
    )
    if not ok:
        return infer_carrier(track, sides=sides)

    raw: list[tuple[float, str | None]] = []
    for f in track.frames:
        if f.ball is None:
            raw.append((f.t, None))
            continue
        best, best_d = None, math.inf
        for pid, p in f.players.items():
            d = math.hypot(p.x - f.ball.x, p.y - f.ball.y)
            if d < best_d:
                best, best_d = pid, d
        raw.append((f.t, best if best_d <= threshold else None))

    return _runs(raw)


def infer_carrier(
    track: PlayTrack,
    sides: Any = None,
    radius: float = 15.0,
    switch_margin: float = 1.5,
) -> list[Possession]:
    """Who has the ball, worked out from everyone else's behaviour.

    No ball track required, which is the point: the vision pipeline has none, and the
    2017 release's is unusable.

    The signal is that eleven defenders are all trying to get to the same man. For
    each offensive player, sum the rate at which nearby defenders are closing on him;
    the carrier is the one they are converging on hardest. It is a genuinely strong
    signal on a running play and a weak one on a deep pass, where the defense is
    covering receivers rather than converging on the quarterback — so the result is a
    best estimate, not a fact, and consumers should treat a possession timeline from
    here as such.

    ``switch_margin`` stops the answer flickering between two players standing next to
    each other: a new candidate has to beat the incumbent by this much closing speed
    before possession changes hands.
    """
    from .sides import infer_sides

    s = sides or infer_sides(track)
    snap_t = track.event_time("ball_snap")
    snap_t = 0.0 if snap_t is None else snap_t

    raw: list[tuple[float, str | None]] = []
    current: str | None = None

    for f in track.frames:
        if f.t < snap_t:
            raw.append((f.t, None))
            continue

        scores: dict[str, float] = {}
        for pid, p in f.players.items():
            if s.team_of(track, pid) != "offense":
                continue
            total = 0.0
            for did, d in f.players.items():
                if s.team_of(track, did) != "defense":
                    continue
                dx, dy = d.x - p.x, d.y - p.y
                dist = math.hypot(dx, dy)
                if dist > radius or dist < 1e-6:
                    continue
                dvx = _vx(d) - _vx(p)
                dvy = _vy(d) - _vy(p)
                closing = -(dvx * dx + dvy * dy) / dist
                if closing > 0:
                    total += closing
            scores[pid] = total

        if not scores:
            raw.append((f.t, current))
            continue

        best = max(scores, key=scores.get)
        if current is None or best == current:
            current = best
        elif scores[best] - scores.get(current, 0.0) > switch_margin:
            current = best
        raw.append((f.t, current))

    return _runs(raw)


def _vx(p) -> float:
    return p.s * math.cos(math.radians(p.dir))


def _vy(p) -> float:
    return p.s * math.sin(math.radians(p.dir))


def engagements(
    track: PlayTrack,
    poss: list[Possession] | None = None,
    enter: float = CONTACT_ENTER,
    exit_: float = CONTACT_EXIT,
    sides: Any = None,
) -> list[Engagement]:
    """Who hit whom.

    Proximity with hysteresis, then classified by how long it lasted and what it did
    to the people in it. A tackle is an engagement that involves whoever had the ball
    and takes his speed away; a block is a long one between opposite sides; anything
    else is a collision.

    What this cannot see is contact that leaves no trace in the coordinates — a hand
    on a jersey while both men keep running. Those are missed, and they are missed
    quietly, which is the honest limit of deriving contact from positions.
    """
    from .sides import infer_sides

    s = sides or infer_sides(track)
    poss = poss if poss is not None else possessions(track, sides=s)
    open_pairs: dict[tuple[str, str], dict] = {}
    done: list[Engagement] = []

    for f in track.frames:
        # Sorted, because a frame's player dict is not guaranteed to be in the same
        # order every frame — and an unsorted key made (Mason, Andrews) and
        # (Andrews, Mason) two different engagements between the same two men.
        ids = sorted(f.players.keys())
        carrier = _carrier_at(poss, f.t)
        seen: set[tuple[str, str]] = set()

        for i in range(len(ids)):
            for j in range(i + 1, len(ids)):
                a, b = ids[i], ids[j]
                # Only opposite sides. Two guards standing shoulder to shoulder are a
                # yard and a half apart all play, which is inside any contact
                # threshold wide enough to catch a real block — so without this the
                # output is mostly the offensive line noticing itself, and Tom Brady
                # gets credited with blocking his own running back.
                ta, tb = s.team_of(track, a), s.team_of(track, b)
                if ta is not None and ta == tb:
                    continue
                pa, pb = f.players[a], f.players[b]
                d = math.hypot(pa.x - pb.x, pa.y - pb.y)
                key = (a, b)
                live = key in open_pairs
                if d <= (exit_ if live else enter):
                    seen.add(key)
                    rec = open_pairs.setdefault(key, {
                        "t_start": f.t, "closest": d,
                        "a_s0": pa.s, "b_s0": pb.s,
                        "a_smin": pa.s, "b_smin": pb.s,
                        "carrier": False,
                    })
                    rec["t_end"] = f.t
                    rec["closest"] = min(rec["closest"], d)
                    rec["a_smin"] = min(rec["a_smin"], pa.s)
                    rec["b_smin"] = min(rec["b_smin"], pb.s)
                    if carrier in (a, b):
                        rec["carrier"] = True
                        rec.setdefault("held_id", carrier)
                    # Follow whoever held the ball when this engagement began, not
                    # whoever holds it now. Once a carrier is wrapped up, defensive
                    # convergence drifts onto the next man, and a per-frame check
                    # therefore stopped watching the tackled player at the exact moment
                    # he went down — leaving t_down unset on the plays it mattered most.
                    held_id = rec.get("held_id")
                    if rec.get("t_down") is None and held_id in (a, b):
                        held = pa if held_id == a else pb
                        if held.s < DOWN_YPS:
                            rec["t_down"] = f.t

        # A grace period, not an immediate close. Two linemen wrestling drift across
        # the exit threshold for a frame at a time, and closing on the first frame
        # apart turned sixteen real engagements into thirty-five fragments — one pair
        # appearing three separate times in a four-second play. The pair has to stay
        # apart for GRACE_S before the engagement is over.
        for key in list(open_pairs):
            if key not in seen:
                rec = open_pairs[key]
                if f.t - rec["t_end"] >= GRACE_S:
                    done.extend(_close(key, open_pairs.pop(key)))

    for key in list(open_pairs):
        done.extend(_close(key, open_pairs.pop(key)))

    return sorted(done, key=lambda e: e.t_start)


def evades(
    track: PlayTrack,
    poss: list[Possession] | None = None,
    engs: list[Engagement] | None = None,
    radius: float = EVADE_YARDS,
    turn: float = EVADE_TURN_DEG,
) -> list[Evade]:
    """Who the carrier went around.

    Three conditions, all necessary. He came close to a defender; he did not touch
    him; and he changed direction while doing it. Drop the third and every defender
    the carrier merely ran past gets credited as beaten, which makes the output
    flattering and useless.
    """
    poss = poss if poss is not None else possessions(track)
    engs = engs if engs is not None else engagements(track, poss)
    if not poss:
        return []

    touched = {tuple(sorted((e.a, e.b))) for e in engs}
    out: list[Evade] = []
    claimed: set[tuple[str, str]] = set()

    frames = track.frames
    for k, f in enumerate(frames):
        carrier = _carrier_at(poss, f.t)
        if carrier is None or carrier not in f.players:
            continue
        c = f.players[carrier]
        c_team = track.roster[carrier].team if carrier in track.roster else None

        for pid, p in f.players.items():
            if pid == carrier:
                continue
            if c_team and pid in track.roster and track.roster[pid].team == c_team:
                continue
            pair = tuple(sorted((carrier, pid)))
            if pair in touched or pair in claimed:
                continue
            d = math.hypot(c.x - p.x, c.y - p.y)
            if d > radius:
                continue

            # Direction change measured across a window either side of the pass.
            lo = frames[max(0, k - 5)].players.get(carrier)
            hi = frames[min(len(frames) - 1, k + 5)].players.get(carrier)
            if lo is None or hi is None:
                continue
            swing = abs(shortest_deg(lo.dir, hi.dir))
            if swing < turn:
                continue

            # And he has to have actually got past him. Without this the first version
            # credited a defensive tackle with beating four offensive linemen at the
            # same instant, because they were close and he was turning — which is a
            # description of being blocked, not of beating anybody. Projecting the
            # defender onto the carrier's direction of travel a moment later settles
            # it: if he is not behind, nothing was evaded.
            later = frames[min(len(frames) - 1, k + 5)]
            c2 = later.players.get(carrier)
            d2 = later.players.get(pid)
            if c2 is None or d2 is None:
                continue
            heading = math.radians(c2.dir)
            ahead = (d2.x - c2.x) * math.cos(heading) + (d2.y - c2.y) * math.sin(heading)
            if ahead > -0.5:
                continue

            claimed.add(pair)
            out.append(Evade(carrier=carrier, beaten=pid, t=f.t,
                             closest_yards=d, turn_deg=swing))

    return sorted(out, key=lambda e: e.t)


def ball_in_view(
    track: PlayTrack,
    cone_deg: float = VIEW_CONE_DEG,
    min_s: float = 0.2,
) -> list[ViewWindow]:
    """When each player could plausibly see the ball. **Inference, not measurement.**

    The test is geometric: is the ball inside a cone around the direction the body is
    pointing. That is the best a coordinate stream supports, and it is wrong in both
    directions — a man can turn his head to see something outside the cone, and a man
    facing the ball can be looking at the man in front of him.

    So every window reports the ``o_source`` it was built on. A window whose basis is
    ``measured`` came from a real orientation column. One whose basis is
    ``from_velocity`` says only that the ball was roughly ahead of where he was
    running, which is a much weaker claim, and one based on ``assumed`` is very nearly
    no claim at all. Consumers must read that field.
    """
    if not any(f.ball for f in track.frames):
        return []

    rank = {"measured": 0, "from_dir": 1, "from_velocity": 2, "assumed": 3, "authored": 4}
    per_player: dict[str, list[tuple[float, bool, float, str]]] = {}

    for f in track.frames:
        if f.ball is None:
            continue
        for pid, p in f.players.items():
            bearing = math.degrees(math.atan2(f.ball.y - p.y, f.ball.x - p.x))
            offset = abs(shortest_deg(p.o, bearing))
            per_player.setdefault(pid, []).append(
                (f.t, offset <= cone_deg, offset, p.o_source)
            )

    out: list[ViewWindow] = []
    for pid, samples in per_player.items():
        start = None
        worst_src = "measured"
        worst_off = 0.0
        for t, visible, offset, src in samples + [(samples[-1][0], False, 0.0, "measured")]:
            if visible and start is None:
                start, worst_src, worst_off = t, src, offset
            elif visible:
                if rank.get(src, 9) > rank.get(worst_src, 9):
                    worst_src = src
                worst_off = max(worst_off, offset)
            elif start is not None:
                if t - start >= min_s:
                    out.append(ViewWindow(pid, start, t, worst_src, worst_off))
                start = None
    return sorted(out, key=lambda v: (v.player, v.t_start))


# --- helpers ------------------------------------------------------------------


def _runs(raw: list[tuple[float, str | None]]) -> list[Possession]:
    out: list[Possession] = []
    if not raw:
        return out
    cur, start, last = raw[0][1], raw[0][0], raw[0][0]
    for t, who in raw[1:]:
        if who != cur:
            out.append(Possession(cur, start, last))
            cur, start = who, t
        last = t
    out.append(Possession(cur, start, last))
    return out


def _carrier_at(poss: list[Possession], t: float) -> str | None:
    for p in poss:
        if p.t_start <= t <= p.t_end:
            return p.player
    return None


def _close(key: tuple[str, str], rec: dict) -> Iterable[Engagement]:
    dur = rec["t_end"] - rec["t_start"]
    if dur < MIN_CONTACT_S:
        return ()
    a_drop = rec["a_s0"] - rec["a_smin"]
    b_drop = rec["b_s0"] - rec["b_smin"]

    # Whoever is holding the ball is not being blocked. Classifying by speed drop
    # alone put a safety engaging the ball carrier into the 'block' bucket because
    # both men lost 1.9 yd/s and the threshold was 2.0 — which is a tackle attempt
    # described as its opposite. Possession decides the category; the speed drop only
    # decides whether the attempt worked.
    if rec["carrier"]:
        kind = "tackle" if max(a_drop, b_drop) >= 1.5 else "tackle_attempt"
    elif dur >= BLOCK_S:
        kind = "block"
    else:
        kind = "collision"

    return (Engagement(
        a=key[0], b=key[1], t_start=rec["t_start"], t_end=rec["t_end"],
        closest_yards=rec["closest"], kind=kind,
        involved_carrier=rec["carrier"],
        a_speed_drop=a_drop, b_speed_drop=b_drop,
        t_down=rec.get("t_down"),
    ),)
