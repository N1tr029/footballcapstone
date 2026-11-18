"""Stable jersey assignments subject to the number of athletes on the field."""

from __future__ import annotations

import numpy as np
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import lil_matrix


def constrained_teams(probabilities, visible, capacity=11, full_frame=None):
    """Assign each identity once, allowing unknown when evidence conflicts.

    ``probabilities[pid]`` contains appearance probabilities for A and B. Each
    team's simultaneous count is bounded, rather than its count of fragmented IDs
    over an entire play. Equal counts are required only for an inspected full
    formation. No player's location on either side of the field determines a team.
    """
    if capacity < 1:
        raise ValueError("team capacity must be positive")
    ids = sorted(probabilities)
    if not ids:
        return {}
    index = {pid: i for i, pid in enumerate(ids)}
    rows, lower, upper = [], [], []
    for pid in ids:
        rows.append({2 * index[pid]: 1, 2 * index[pid] + 1: 1})
        lower.append(0)
        upper.append(1)
    for fi, players in enumerate(visible):
        players = set(players) & index.keys()
        complete = fi == full_frame
        if complete and len(players) != 2 * capacity:
            raise ValueError("full formation must contain exactly twice the team capacity")
        for team in (0, 1):
            rows.append({2 * index[pid] + team: 1 for pid in players})
            lower.append(capacity if complete else 0)
            upper.append(capacity)
    matrix = lil_matrix((len(rows), 2 * len(ids)))
    for i, row in enumerate(rows):
        for j, value in row.items():
            matrix[i, j] = value
    # Unknown is preferable to assigning a clearly wrong jersey merely to fill a
    # slot. Strong appearances earn a larger reward, with a tiny deterministic tie
    # break. A full formation can resolve a genuinely marginal jersey.
    costs = []
    for i, pid in enumerate(ids):
        p = np.clip(probabilities[pid], 1e-5, 1 - 1e-5)
        costs.extend(np.log(0.60 / p) + i * 1e-9)
    result = milp(np.array(costs), integrality=np.ones(2 * len(ids)),
                  bounds=Bounds(0, 1),
                  constraints=LinearConstraint(matrix.tocsr(), lower, upper),
                  options={"time_limit": 10})
    if result.x is None:
        raise ValueError(f"team roster constraints could not be satisfied: {result.message}")
    return {pid: ("teamA" if result.x[2 * i] > .5 else
                  "teamB" if result.x[2 * i + 1] > .5 else None)
            for i, pid in enumerate(ids)}


def split_team_changes(track, labels_by_frame):
    """A persistent jersey change means the identity switched people, not teams.

    Preserve coordinates, but give each contiguous team segment its own identity.
    This exposes the remaining fragmentation instead of colouring a whole mixed
    identity according to whichever jersey it wore at the beginning.
    """
    from dataclasses import replace

    changed = 0
    for pid in list(track.roster):
        appearances = [(i, labels_by_frame[i].get(pid) or "unknown")
                       for i, f in enumerate(track.frames) if pid in f.players]
        if not appearances:
            continue
        if len({label for _, label in appearances}) == 1:
            track.roster[pid].team = appearances[0][1]
            continue
        original = track.roster.pop(pid)
        previous, segment = None, 0
        known_id = None
        for i, label in appearances:
            if label == "unknown":
                new_id = f"{pid}_unknown"
                track.roster.setdefault(new_id, replace(original, id=new_id, team=label))
            elif label != previous:
                segment += 1
                new_id = f"{pid}_s{segment}"
                track.roster[new_id] = replace(original, id=new_id, team=label)
                previous = label
                known_id = new_id
            else:
                new_id = known_id
            frame = track.frames[i]
            frame.players[new_id] = frame.players.pop(pid)
        changed += max(0, segment - 1)
    return changed
