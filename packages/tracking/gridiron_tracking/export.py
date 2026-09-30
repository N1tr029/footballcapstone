"""The data file the rest of the team consumes.

Two files per play, because two different things want it.

``<play>.csv`` is one row per player per frame — the shape every public tracking
release uses, which means anything already written against Big Data Bowl data reads
it without changes. This is what a model trains on.

``<play>.json`` is the whole play as one object: roster, frames, events, and the
interaction layer (possession, blocks, tackles, evades). This is what a renderer
animates and what a person reads.

Both carry a **provenance block**, and it is not decoration. A file that says a player
was at (48.2, 31.7) facing 212 degrees looks equally confident whether that came from
a measured tracking column or from a homography fitted to six clicks and a body angle
guessed from direction of travel. The consumer cannot tell by looking, so the file has
to say. Anything derived from an assumption is labelled with the assumption.

The one thing this module will not do is emit a field it cannot source. There is no
gaze column, because nothing measures gaze; the nearest honest thing is
``ball_in_view``, which lives under interactions with ``inferred: true`` on every
record.
"""

from __future__ import annotations

import json
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd

from .contract import PlayTrack, validate
from .interactions import Interactions, derive

SCHEMA_VERSION = "1.0"

CSV_COLUMNS = [
    "play_id", "t", "player_id", "name", "jersey", "team", "side",
    "x", "y", "s", "a", "dis", "dir", "o", "o_source", "has_ball", "events",
]


def to_dataframe(track: PlayTrack, interactions: Interactions | None = None) -> pd.DataFrame:
    """One row per player per frame, in the column order public releases use.

    ``side`` and ``has_ball`` are the two columns a raw release does not have and
    every consumer immediately wants: which way this player was going, and whether he
    was the one carrying it.
    """
    ix = interactions
    play_id = track.meta.get("play_id", "play")
    rows: list[dict[str, Any]] = []

    for f in track.frames:
        carrier = ix.carrier_at(f.t) if ix else None
        ev = "|".join(f.events) if f.events else ""
        for pid, p in f.players.items():
            entry = track.roster.get(pid)
            side = ix.sides.team_of(track, pid) if (ix and ix.sides) else None
            rows.append({
                "play_id": play_id,
                "t": round(f.t, 3),
                "player_id": pid,
                "name": (entry.name or entry.label) if entry else pid,
                "jersey": entry.num if entry else None,
                "team": entry.team if entry else None,
                "side": side,
                "x": round(p.x, 3), "y": round(p.y, 3),
                "s": round(p.s, 3), "a": round(p.a, 3),
                "dis": round(p.s / 10.0, 4),
                "dir": round(p.dir, 2), "o": round(p.o, 2),
                "o_source": p.o_source,
                "has_ball": bool(carrier == pid),
                "events": ev,
            })

        if f.ball is not None:
            rows.append({
                "play_id": play_id, "t": round(f.t, 3), "player_id": "ball",
                "name": "football", "jersey": None, "team": "ball", "side": None,
                "x": round(f.ball.x, 3), "y": round(f.ball.y, 3),
                "s": None, "a": None, "dis": None, "dir": None, "o": None,
                "o_source": None, "has_ball": True, "events": ev,
            })

    return pd.DataFrame(rows, columns=CSV_COLUMNS)


def provenance(track: PlayTrack, interactions: Interactions | None = None) -> dict[str, Any]:
    """Where every part of this file came from, and what was assumed.

    Read this before trusting anything else in the file.
    """
    sources = Counter(
        p.o_source for f in track.frames for p in f.players.values()
    )
    total = sum(sources.values()) or 1
    v = validate(track)

    coord_source = track.meta.get("source", "unknown")
    measured_coords = coord_source == "ngs"

    out: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "coordinates": {
            "source": coord_source,
            "measured": measured_coords,
            "note": (
                "Measured by the league's tracking system; treat as ground truth."
                if measured_coords else
                "ESTIMATED from video. Accuracy depends on the field registration — "
                "see the 'registration' block if present. These are not exact positions."
            ),
            "units": "yards; x downfield 0-120 with the offense attacking +x, y across 0-53.3",
            "rate_hz": track.meta.get("rate"),
        },
        "orientation": {
            "measured_fraction": round(sources.get("measured", 0) / total, 3),
            "by_source": dict(sources),
            "note": (
                "o is body orientation, dir is direction of travel; they are different "
                "and must not be derived from each other. o_source on every row says "
                "how that value was arrived at."
            ),
        },
        "gaze": {
            "available": False,
            "note": (
                "Where a player was LOOKING is not measured by anything, in this or any "
                "public football dataset. No such column exists here. The closest "
                "available signal is interactions.ball_in_view, which is inferred from "
                "body orientation and marked as inferred on every record."
            ),
        },
        "contract_valid": v.ok,
        "contract_problems": v.problems,
    }

    if interactions is not None:
        out["interactions"] = {
            "possession": (
                "from the ball track" if any(f.ball for f in track.frames)
                and "ball track used" in " ".join(interactions.notes)
                else "INFERRED from defensive convergence — no usable ball track"
            ),
            "sides": interactions.sides.to_json() if interactions.sides else None,
            "accuracy_notes": [
                "Validated against the 2017 Big Data Bowl's own event labels over 68 "
                "plays that carry both first_contact and tackle.",
                "Carrier contact was found on 100% of them.",
                "t_start vs first_contact: median 0.20 s, 90% within 0.5 s.",
                "t_down vs tackle: median 0.30 s, 73% within 0.5 s, present on 97%.",
                "Which team had the ball: correct on 98.7% of 149 labelled plays.",
                "These figures are from MEASURED tracking. Running the same derivation "
                "on coordinates estimated from video will be worse by an amount not yet "
                "measured.",
            ],
            "notes": interactions.notes,
        }
    return out


def to_json(
    track: PlayTrack,
    interactions: Interactions | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """The whole play, as one object."""
    ix = interactions if interactions is not None else derive(track)
    doc = {
        "meta": dict(track.meta),
        "provenance": provenance(track, ix),
        "roster": {k: v.to_json() for k, v in track.roster.items()},
        "frames": [f.to_json() for f in track.frames],
        "interactions": ix.to_json(),
    }
    if extra:
        doc["meta"].update(extra)
    return doc


def write_play(
    out_dir: str | Path,
    track: PlayTrack,
    interactions: Interactions | None = None,
    stem: str | None = None,
) -> dict[str, Path]:
    """Write both files, and return where they went."""
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    ix = interactions if interactions is not None else derive(track)
    stem = stem or f"play_{track.meta.get('play_id', 'unknown')}"

    csv_path = out_dir / f"{stem}.csv"
    json_path = out_dir / f"{stem}.json"

    to_dataframe(track, ix).to_csv(csv_path, index=False)
    json_path.write_text(json.dumps(to_json(track, ix), indent=2, default=str))
    return {"csv": csv_path, "json": json_path}
