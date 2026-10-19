"""Who was doing what on this snap, worked out from the play rather than read off it.

The fitted model keys on ``player_role`` — ``Passer``, ``Targeted Receiver``,
``Other Route Runner``, ``Defensive Coverage`` — and exactly one dataset carries
that column: the 2026 competition file it was trained on. Every other play in this
repository lacks it. The Big Data Bowl 2025 release the Unity app ships has
positions and sides but no per-snap roles, and the vision pipeline has neither.

Without this module the model can only ever score Kaggle's own file, which makes it
a competition entry rather than a part of this project. With it, any play that has
twenty-two trajectories and a ball can be scored.

Nothing here is clever: the roles are geometry over the ball.

**The passer** is the offensive player closest to the ball **when the pass leaves**,
not at the snap. At the snap the nearest man to the ball is the centre, who has his
hands on it — ``sides.py`` relies on exactly that — so snapping-instant proximity
identifies the passer on 1 play in 115 and the centre on the other 114. Measured at
``pass_forward`` it is the quarterback on 64 of 67 pass plays. The difference
matters beyond a label: a centre marked ``Passer`` leaves the quarterback marked as
a route runner, and the recommendation then offers him as an open alternative.

**The targeted receiver** is the offensive player closest to the ball when the pass
arrives. This is the one inference worth checking rather than trusting, because the
whole recommendation is a comparison against this player — so
:func:`check_against_descriptions` measures it against the play-by-play text, which
names the target, and reports a hit rate instead of asking anyone to believe it.

**The other route runners** are the remaining eligible receivers — WR, TE, RB, FB.
Linemen are left with no role at all, which the model already handles: it maps roles
through ``fillna(-1)``, so "none of the four" is a code it was trained with. Marking
a tackle as a route runner puts him in the alternatives list, and on one real play
it won it — the inference offered Rob Havenstein, a right tackle, as the man who
should have been thrown to.

Where no position column exists — the vision pipeline's output — every remaining
offensive player becomes a route runner, because the alternative is no candidates
at all. ``roles_source`` in meta records which of the two happened.

**The defense** is all ``Defensive Coverage``. The model measures separation to the
nearest defender, so a pass rusher is a defender who happens to be far from
everyone; it does not need to know he was rushing.
"""

from __future__ import annotations

import math
import re
from typing import Any

from gridiron_tracking.contract import PlayTrack

PASSER = "Passer"
TARGET = "Targeted Receiver"
ROUTE = "Other Route Runner"
COVERAGE = "Defensive Coverage"

ARRIVAL_EVENTS = ("pass_arrived", "pass_outcome_caught", "pass_tipped",
                  "pass_outcome_incomplete", "pass_outcome_interception")
"""When the ball got there, best first. The 2025 release labels these per frame."""

THROW_EVENTS = ("pass_forward", "pass_shovel")
"""When the ball left the passer's hand. The only instant at which the ball
identifies the thrower."""

SNAP_EVENTS = ("ball_snap",)

ELIGIBLE = ("WR", "TE", "RB", "FB")
"""Positions that run routes. Used only when the release carries positions at all."""

LINE = ("T", "G", "C", "OL", "OT", "OG")


def infer_roles(track: PlayTrack, offense: str | None = None) -> dict[str, str]:
    """A ``player_id -> role`` map for :mod:`gridiron_counterfactual.features`.

    ``offense`` names the offensive side as the roster spells it. When omitted it is
    taken from the play's own ``possessionTeam`` if the meta carries one, and
    otherwise inferred by ``gridiron_tracking.sides``, which is right on 98.7% of
    labelled plays.

    Raises if the offense cannot be established. A roles map with the sides swapped
    scores the defense's separation against the offense and returns numbers that
    look entirely reasonable, so guessing here is worse than stopping.
    """
    off = offense or _offense_of(track)
    if off is None:
        raise ValueError(
            "cannot tell which side had the ball, so roles cannot be assigned. "
            "Pass offense= with the team as the roster spells it."
        )

    offensive = [pid for pid, e in track.roster.items() if _same_side(e.team, off)]
    defensive = [pid for pid in track.roster if pid not in offensive]
    if not offensive or not defensive:
        raise ValueError(
            f"offense {off!r} matched {len(offensive)} of {len(track.roster)} "
            f"players; it does not partition this roster"
        )

    positions = {str(track.roster[pid].role or "") for pid in offensive}
    known_positions = bool(positions - {""})

    roles: dict[str, str] = {pid: COVERAGE for pid in defensive}
    for pid in offensive:
        position = str(track.roster[pid].role or "")
        if not known_positions or position in ELIGIBLE:
            roles[pid] = ROUTE
        # Linemen and anyone else get no role. The model maps unknown roles to -1,
        # which is a code it was trained with, and it keeps a tackle out of the
        # list of men who should have been thrown to.

    throw = _event_time(track, THROW_EVENTS)
    passer = _nearest_to_ball(track, offensive, throw)
    if passer is None:
        # No throw on this play, so nobody passed it. Falling back to the snap
        # would name the centre; see the module docstring.
        passer_at_snap = None
    else:
        passer_at_snap = passer
        roles[passer] = PASSER

    arrival = _event_time(track, ARRIVAL_EVENTS)
    if arrival is not None:
        candidates = [p for p in offensive
                      if p != passer_at_snap and roles.get(p) == ROUTE]
        target = _nearest_to_ball(track, candidates, arrival)
        if target is not None:
            roles[target] = TARGET

    return roles


def apply(track: PlayTrack, offense: str | None = None) -> PlayTrack:
    """Infer the roles and record them where ``features`` looks for them.

    Also stars the targeted receiver, because he is the player the What-If panel is
    arguing about and the renderer already knows how to draw a starred man.
    """
    roles = infer_roles(track, offense)
    track.meta["roles"] = roles
    offensive_positions = {
        str(e.role or "") for pid, e in track.roster.items()
        if roles.get(pid) != COVERAGE
    } - {""}
    track.meta["roles_source"] = (
        "inferred: passer from ball proximity at the throw, target at pass "
        "arrival, route runners " + (
            "filtered to eligible positions" if offensive_positions
            else "taken as all remaining offense (no position column)"
        ) + " — not read from a role column"
    )
    for pid, entry in track.roster.items():
        entry.star = roles.get(pid) == TARGET
    return track


def check_against_descriptions(tracks: list[PlayTrack]) -> dict[str, Any]:
    """Measure the targeted-receiver inference against the play-by-play text.

    The description names the target — "pass short left to J.Chase" — in the
    ``F.Lastname`` form, and the roster carries full names, so the two can be
    compared without a lookup table. Plays whose description names nobody (a sack,
    a scramble, a throwaway) are reported separately rather than counted as misses.

    This is the only claim in this module that can be checked, so it is the only
    number worth quoting about it.
    """
    hits = misses = unnamed = no_target = 0
    examples: list[dict[str, Any]] = []

    for track in tracks:
        description = str(track.meta.get("description") or "")
        named = _target_from_description(description)
        try:
            roles = infer_roles(track)
        except ValueError:
            no_target += 1
            continue

        chosen = next((pid for pid, r in roles.items() if r == TARGET), None)
        if chosen is None:
            no_target += 1
            continue
        if named is None:
            unnamed += 1
            continue

        name = track.roster[chosen].name or track.roster[chosen].label or ""
        ok = _name_matches(named, name)
        hits += ok
        misses += not ok
        if not ok and len(examples) < 6:
            examples.append({
                "play": track.meta.get("play_id"),
                "description_says": named,
                "inference_says": name,
            })

    checked = hits + misses
    return {
        "plays": len(tracks),
        "checked": checked,
        "hits": hits,
        "misses": misses,
        "hit_rate": round(hits / checked, 3) if checked else None,
        "description_named_nobody": unnamed,
        "no_arrival_or_no_offense": no_target,
        "misses_sample": examples,
    }


# ------------------------------------------------------------------- internals


def _offense_of(track: PlayTrack) -> str | None:
    possession = track.meta.get("possessionTeam")
    if possession:
        teams = {str(e.team) for e in track.roster.values()}
        if str(possession) in teams:
            return str(possession)
        # The 2025 release's roster is sided rather than clubbed once it has been
        # through the app's exporter, so a club name will not match; fall through.
    if any(str(e.team).lower().startswith("off") for e in track.roster.values()):
        return "offense"

    from gridiron_tracking import sides

    try:
        inferred = sides.infer_sides(track)
    except Exception:  # noqa: BLE001 - an unusable track is not an error here
        return None
    return getattr(inferred, "offense", None)


def _same_side(team: Any, offense: str) -> bool:
    return str(team).lower() == str(offense).lower()


def _event_time(track: PlayTrack, names: tuple[str, ...],
                default: float | None = None) -> float | None:
    for name in names:
        t = track.event_time(name)
        if t is not None:
            return t
    return default


def _nearest_to_ball(track: PlayTrack, candidates: list[str], t: float | None) -> str | None:
    if t is None or not candidates:
        return None
    frame = track.frame_at(t)
    if frame is None or frame.ball is None:
        # No ball at that instant is a real gap, not something to paper over with
        # a fallback to "whoever is nearest the middle of the field".
        return None

    best, best_d = None, math.inf
    for pid in candidates:
        p = frame.players.get(pid)
        if p is None:
            continue
        d = math.hypot(p.x - frame.ball.x, p.y - frame.ball.y)
        if d < best_d:
            best, best_d = pid, d
    return best


_TARGET_RE = re.compile(
    r"(?:pass|complete|incomplete)[^.]*?\bto\s+([A-Z][A-Za-z'’.-]*\.\s?[A-Z][A-Za-z'’-]+)"
)


def _target_from_description(description: str) -> str | None:
    """The ``F.Lastname`` the description says the ball went to, if it says one."""
    m = _TARGET_RE.search(description)
    return m.group(1).replace(" ", "") if m else None


def _name_matches(abbreviated: str, full: str) -> bool:
    """``J.Chase`` against ``Ja'Marr Chase``: initial and surname, loosely.

    Surnames in play-by-play are shortened and punctuated inconsistently, so the
    comparison strips non-letters and allows the recorded surname to be a prefix of
    the full one.
    """
    if not abbreviated or not full:
        return False
    initial, _, surname = abbreviated.partition(".")
    parts = full.split()
    if not parts or not surname:
        return False

    def letters(s: str) -> str:
        return re.sub(r"[^a-z]", "", s.lower())

    want, got = letters(surname), letters(parts[-1])
    if not (got.startswith(want) or want.startswith(got)):
        return False
    return letters(initial)[:1] == letters(parts[0])[:1]
