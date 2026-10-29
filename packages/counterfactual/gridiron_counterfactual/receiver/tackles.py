"""Missed tackles: a defender got to the ball carrier and the carrier kept going.

The second kind of mistake, and a different sort of evidence from target choice. A missed
read is an estimate of something that did not happen. A missed tackle happened: the
defender was in contact with the man carrying the ball, the carrier was not brought down,
and the tracking shows how far he went afterwards. No model is needed to say what it cost.

All of the hard part is stream 2's interaction layer (``gridiron_tracking.interactions``),
which decides from the coordinates who had the ball, who was in contact with him, and
whether that contact ended with him going down. This module asks it one question and adds
the cost:

    yards_after   how far downfield the carrier got after the contact ended

**What it can and cannot be checked against.** This release does not label missed tackles,
so there is no direct test. What it does label is who *made* each tackle, and the same
layer produces those. :func:`check` compares the two: if the layer credits the right
tacklers, its notion of contact-with-the-carrier is sound, and a contact that did not end
the play is a miss by the same logic. That is support, not proof — a defender who was
blocked into the carrier, or who slowed him so a teammate could finish, is counted here
and a scout might not count him.
"""

from __future__ import annotations

import pandas as pd

from gridiron_tracking import interactions
from gridiron_tracking.contract import PlayTrack

from . import bdb2025

MIN_YARDS_AFTER = 3.0
"""A miss that cost less than this is a tackle the next man made a step later."""

MAX_CONTACT_YARDS = 1.0
"""Centre to centre. The layer calls anything inside a yard and a half contact, which is
right for a lineman's extended arms and too generous for "he had him": at that range it
credits three tacklers for every one the release does."""


def missed(track: PlayTrack, derived: interactions.Interactions | None = None,
           carriers: dict[str, float] | None = None) -> list[dict]:
    """Every close contact with the ball carrier, after the snap, that did not bring him down.

    ``carriers`` maps the men who really carried the ball on this play (from play-by-play)
    to the time each got it. Given, a miss only counts against one of them and only after
    he had the ball — which keeps a wrongly inferred carrier, most often on a pass still in
    the air, from producing a miss that never happened.
    """
    derived = derived or interactions.derive(track, offense=track.meta.get("offense"))
    end_t = track.frames[-1].t
    out = []
    for e in derived.engagements:
        if e.kind != "tackle_attempt" or not e.involved_carrier or e.t_start <= 0:
            continue
        if e.closest_yards > MAX_CONTACT_YARDS:
            continue
        carrier = derived.carrier_at(e.t_start)
        if carrier not in (e.a, e.b):
            continue
        if carriers is not None and (carrier not in carriers or e.t_start < carriers[carrier]):
            continue
        defender = e.b if carrier == e.a else e.a
        # Where the carrier was when the contact ended, and where his carry ended.
        still = [p for p in derived.possessions if p.player == carrier and p.t_end >= e.t_end]
        carry_end = min(still[0].t_end, end_t) if still else None
        at_miss, at_end = track.frame_at(float(e.t_end)), carry_end is not None and track.frame_at(float(carry_end))
        if not (at_miss and at_end and carrier in at_miss.players and carrier in at_end.players):
            continue
        out.append({
            "defender": defender, "carrier": carrier, "t": float(e.t_start), "t_end": float(e.t_end),
            "closest_yards": float(e.closest_yards),
            "yards_after": float(at_end.players[carrier].x - at_miss.players[carrier].x),
        })
    return out


def tacklers(derived: interactions.Interactions) -> set[str]:
    """Defenders the layer says brought the carrier down, at close contact."""
    out = set()
    for e in derived.engagements:
        if e.kind == "tackle" and e.involved_carrier and e.closest_yards <= MAX_CONTACT_YARDS:
            carrier = derived.carrier_at(e.t_down if e.t_down is not None else e.t_start)
            out |= {p for p in (e.a, e.b) if p != carrier}
    return out


def scan(game_ids, log=None) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Missed tackles, and the layer's tackle credits beside the release's, over whole games.

    Returns ``(misses, credits)``. ``credits`` has one row per (play, defender) that either
    the layer or the release credited with a tackle, for :func:`check`.
    """
    pp = bdb2025.player_play()
    labelled = pp[(pp["soloTackle"] == 1) | (pp["tackleAssist"] == 1) | (pp["assistedTackle"] == 1)]
    labelled = labelled.groupby(bdb2025.KEY)["nflId"].agg(lambda s: {str(int(i)) for i in s})
    ran = pp[pp["hadRushAttempt"] == 1].groupby(bdb2025.KEY)["nflId"].agg(lambda s: [str(int(i)) for i in s])
    caught = pp[pp["hadPassReception"] == 1].groupby(bdb2025.KEY)["nflId"].agg(lambda s: [str(int(i)) for i in s])
    names = dict(zip(bdb2025.players()["nflId"].astype(str), bdb2025.players()["displayName"]))
    plays = bdb2025.plays().set_index(bdb2025.KEY)

    misses, credits = [], []
    for gid in game_ids:
        raw = bdb2025.raw_tracking(gid)
        for pid in sorted(raw["playId"].unique()):
            if (gid, pid) not in plays.index:
                continue
            try:
                track = bdb2025.play_track(gid, int(pid), raw=raw)
                derived = interactions.derive(track, offense=track.meta.get("offense"))
            except Exception as err:               # one odd play must not stop a season
                if log:
                    log(f"skipped {gid}-{pid}: {err}")
                continue
            # Who really carried it, and from when: a runner from the snap, a receiver from the catch.
            t_catch = track.event_time("pass_outcome_caught")
            carriers = {i: 0.0 for i in ran.get((gid, pid), [])}
            if t_catch is not None:
                carriers.update({i: t_catch for i in caught.get((gid, pid), [])})
            for m in missed(track, derived, carriers):
                misses.append({"gameId": gid, "playId": int(pid), **m,
                               "defender_name": names.get(m["defender"]), "carrier_name": names.get(m["carrier"]),
                               "yards_gained": plays.loc[(gid, pid), "yardsGained"],
                               "description": plays.loc[(gid, pid), "playDescription"]})
            ours, theirs = tacklers(derived), labelled.get((gid, pid), set())
            for d in ours | theirs:
                credits.append({"gameId": gid, "playId": int(pid), "defender": d,
                                "layer": d in ours, "release": d in theirs})
        if log:
            log(f"game {gid}: {len(misses)} misses so far")
    return pd.DataFrame(misses), pd.DataFrame(credits)


def check(misses: pd.DataFrame, credits: pd.DataFrame) -> dict:
    """How well the layer's tackle credits match the release's, and what the misses cost."""
    both = int((credits["layer"] & credits["release"]).sum())
    layer, release = int(credits["layer"].sum()), int(credits["release"].sum())
    costly = misses[misses["yards_after"] >= MIN_YARDS_AFTER] if len(misses) else misses
    return {
        "plays": int(credits.groupby(bdb2025.KEY).ngroups),
        "tackle_credits": {
            "release": release, "layer": layer, "agreed": both,
            "layer_credits_the_release_confirms": round(both / layer, 3) if layer else None,
            "release_credits_the_layer_found": round(both / release, 3) if release else None,
        },
        "missed_tackles": {
            "contacts_that_did_not_end_the_carry": len(misses),
            f"cost_at_least_{MIN_YARDS_AFTER:.0f}_yards": len(costly),
            "median_yards_after": round(float(costly["yards_after"].median()), 1) if len(costly) else None,
            "share_of_plays_with_a_costly_miss": round(
                costly.groupby(bdb2025.KEY).ngroups / max(credits.groupby(bdb2025.KEY).ngroups, 1), 3),
        },
    }
