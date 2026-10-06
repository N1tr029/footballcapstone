"""Missed tackles, with the interaction layer's answer written by hand.

The layer itself is stream 2's and has its own tests. What is tested here is the question
asked of it: which contacts count as a miss, against whom, and what it cost.
"""

import pytest

from gridiron_tracking.contract import Ball, PlayerFrame, PlayTrack, RosterEntry, TrackingFrame
from gridiron_tracking.interactions import Engagement, Interactions, Possession

from gridiron_counterfactual import tackles


def _track():
    """A runner going 5 yards a second downfield from x = 30 for four seconds."""
    roster = {i: RosterEntry(id=i, team="offense" if i == "rb" else "defense") for i in ("rb", "lb", "fs")}
    frames = []
    for k in range(41):
        t = round(k * 0.1, 1)
        players = {"rb": PlayerFrame(30.0 + 5.0 * t, 26.0), "lb": PlayerFrame(35.0, 26.0), "fs": PlayerFrame(50.0, 26.0)}
        frames.append(TrackingFrame(t=t, players=players, ball=Ball(30.0 + 5.0 * t, 26.0),
                                    events=["ball_snap"] if k == 0 else []))
    return PlayTrack(meta={"offense": "offense"}, roster=roster, frames=frames)


def _engagement(defender, t0, t1, kind, closest=0.5):
    return Engagement(a="rb", b=defender, t_start=t0, t_end=t1, closest_yards=closest, kind=kind,
                      involved_carrier=True, a_speed_drop=0.0, b_speed_drop=0.0)


def _derived(*engagements):
    d = Interactions()
    d.possessions = [Possession(player="rb", t_start=0.0, t_end=4.0)]
    d.engagements = list(engagements)
    return d


def test_a_contact_the_carrier_ran_through_is_a_miss_and_costs_what_he_gained_after():
    d = _derived(_engagement("lb", 1.0, 1.4, "tackle_attempt"), _engagement("fs", 3.8, 4.0, "tackle"))
    (miss,) = tackles.missed(_track(), d)
    assert (miss["defender"], miss["carrier"]) == ("lb", "rb")
    assert miss["yards_after"] == pytest.approx(5.0 * (4.0 - 1.4))      # 13 yards after he got free
    assert tackles.tacklers(d) == {"fs"}


def test_the_man_who_made_the_tackle_is_not_a_miss():
    assert tackles.missed(_track(), _derived(_engagement("fs", 3.8, 4.0, "tackle"))) == []


def test_contact_before_the_snap_or_at_arms_length_does_not_count():
    early = _engagement("lb", -2.0, -1.0, "tackle_attempt")
    loose = _engagement("lb", 1.0, 1.4, "tackle_attempt", closest=1.4)
    assert tackles.missed(_track(), _derived(early, loose)) == []


def test_a_miss_only_counts_against_a_man_who_really_had_the_ball():
    d = _derived(_engagement("lb", 1.0, 1.4, "tackle_attempt"))
    assert tackles.missed(_track(), d, carriers={"someone_else": 0.0}) == []
    assert tackles.missed(_track(), d, carriers={"rb": 2.0}) == []       # contact came before the catch
    assert len(tackles.missed(_track(), d, carriers={"rb": 0.0})) == 1
