"""The Unity app's play JSON, both directions.

``packages/replay-engine/hindsight`` is Jeremy's Unity 6.6 replay app, merged with
its own history. It reads a schema its own exporter writes — ``schemaVersion 1``,
documented in its ``CLAUDE.md`` — and it was designed independently of this
repository's contract. The two agree on more than they disagree, which is the
dangerous ratio, so the differences are worth stating before the code:

**Angles are the other convention.** Its ``o`` and ``dir`` are compass bearings:
zero along +y, increasing clockwise, exactly as the raw NGS columns give them. The
contract is the mathematical convention, zero along +x increasing toward +y. The
conversion is the same reflection ``(90 - a) % 360`` that ``sources.ngs`` applies,
and it is imported from there rather than retyped. Hand one schema's angle to the
other and every player is rotated — a quarterback reading the far hash ends up
watching the near sideline, and the play still animates smoothly while he does it.

**It has no ``o_source`` and no acceleration.** Its exporter drops ``a``, and its
player frame has nowhere to say how an orientation was arrived at. Reading, that is
fine and the values are honest: the plays in ``Assets/StreamingAssets/plays`` are
the real Big Data Bowl 2025 release, so ``o`` is measured and is marked as such.
Writing is the problem — a simulated play exported to Unity would draw identically
to a measured one. So :func:`to_play_json` emits ``oSource`` on every player frame
and a top-level ``provenance`` block anyway. Unity's ``JsonUtility`` ignores fields
it does not declare, so the extra keys cannot break the client; they are there for
when it grows a badge, and `apps/desktop/README.md` records that it has not yet.

**Both normalize left-direction plays the same way**, so nothing here mirrors
anything. Its field is 160/3 yards wide where ``field.WIDTH`` is 53.3, a third of an
inch apart; the validator's tolerance swallows it and no conversion is applied.

**Its clock is a frame index, not a time.** ``frameId`` is 1-based and contiguous,
and ``snapFrameId`` says which one is the snap, so contract time is
``(frameId - snapFrameId) / frameRate`` and the snap is t = 0 by construction.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from .contract import (
    RATE, Ball, PlayTrack, PlayerFrame, RosterEntry, TrackingFrame, wrap_deg,
)
from .sources.ngs import (
    contract_angle_to_ngs, ngs_angle_to_contract, player_key as _key,
)

SCHEMA_VERSION = 1
"""The version this module reads and writes. Jeremy's notes say to bump it on a
breaking change, so a file claiming a higher number is refused rather than guessed
at."""

# Play-level fields that belong to the situation rather than the tracking. The
# contract has no PlayState yet (stream 4 owns it), so they ride through meta and
# are written back out untouched.
PLAY_STATE_KEYS = (
    "homeTeam", "visitorTeam", "description", "quarter", "gameClock",
    "driveNumber", "down", "yardsToGo", "possessionTeam", "defensiveTeam",
    "lineOfScrimmage", "firstDownLine", "passResult",
)


# ------------------------------------------------------------------- reading


def read_play(source: str | Path | dict[str, Any]) -> PlayTrack:
    """One of the Unity app's play files as a :class:`PlayTrack`.

    This is how the 115 real plays committed in ``Assets/StreamingAssets/plays``
    become available to everything on the Python side — the benchmark, the
    interaction layer, the play model. They are the real Big Data Bowl 2025
    release, already registered, so they are measured tracking and are marked as
    such.
    """
    d = source if isinstance(source, dict) else json.loads(Path(source).read_text())

    version = d.get("schemaVersion")
    if version is not None and version > SCHEMA_VERSION:
        raise ValueError(
            f"this play is schemaVersion {version}; this module reads "
            f"{SCHEMA_VERSION}. The Unity app's CLAUDE.md says a bump means a "
            f"breaking change, so read it before widening this check."
        )

    rate = int(d.get("frameRate") or RATE)
    snap = d.get("snapFrameId")
    raw_frames = d.get("frames") or []
    if not raw_frames:
        raise ValueError("this play has no frames")
    if snap is None:
        # Without a snap there is no zero for the clock. Falling back to the first
        # frame is better than refusing, but it makes every time in the play
        # relative to an arbitrary instant, so it is recorded in meta.
        snap = raw_frames[0].get("frameId", 1)

    roster: dict[str, RosterEntry] = {}
    for p in d.get("players") or []:
        pid = _key(p.get("nflId"))
        side = str(p.get("side") or "?")
        roster[pid] = RosterEntry(
            id=pid,
            team=side,
            label=str(p.get("name") or pid),
            name=p.get("name"),
            num=_maybe_int(p.get("jerseyNumber")),
            role=p.get("position"),
        )

    frames: list[TrackingFrame] = []
    for f in raw_frames:
        fid = f.get("frameId")
        players: dict[str, PlayerFrame] = {}
        for p in f.get("players") or []:
            pid = _key(p.get("nflId"))
            o = p.get("o")
            players[pid] = PlayerFrame(
                x=float(p["x"]), y=float(p["y"]),
                s=float(p.get("s") or 0.0),
                a=0.0,  # differenced below; this exporter does not carry it
                dir=float(ngs_angle_to_contract(float(p.get("dir") or 0.0))),
                o=(float(ngs_angle_to_contract(float(o))) if o is not None
                   else float(ngs_angle_to_contract(float(p.get("dir") or 0.0)))),
                o_source="measured" if o is not None else "from_dir",
            )

        ball = None
        if f.get("ball"):
            b = f["ball"]
            ball = Ball(x=float(b["x"]), y=float(b["y"]))

        event = f.get("event")
        frames.append(TrackingFrame(
            t=round((fid - snap) / rate, 3),
            players=players,
            ball=ball,
            events=[str(event)] if event else [],
        ))

    _difference_acceleration(frames, 1.0 / rate)

    # A play this package wrote as a forecast must not read back as measurement.
    # The flag is in the file; dropping it on the way in meant a simulated play
    # became an ordinary one the moment it round-tripped, and the counterfactual
    # CLI then reviewed its own output and wrote forecasts of forecasts.
    simulated = bool(d.get("simulated"))

    meta: dict[str, Any] = {
        "game_id": d.get("gameId"),
        "play_id": d.get("playId"),
        "source": "simulated" if simulated else "ngs",
        "simulated": simulated,
        "source_detail": (
            "a forecast written by packages/counterfactual; this play did not happen"
            if simulated else
            "NFL Big Data Bowl 2025, via the Unity app's exporter"
        ),
        "rate": rate,
        # Its exporter has already normalized the play, which is why nothing here
        # mirrors. Recorded so a reader knows it was not simply right-handed.
        "play_direction": "right",
        "play_direction_source": "normalized by the hindsight exporter",
        "orientation_measured": any(
            p.o_source == "measured" for f in frames for p in f.players.values()
        ),
        "acceleration_measured": False,
        "snap_frame_id": snap,
        "schema_version": version,
    }
    if d.get("snapFrameId") is None:
        meta["clock_warning"] = (
            "no snapFrameId in the source file; t is relative to the first frame, "
            "not to the snap"
        )
    for key in PLAY_STATE_KEYS:
        if d.get(key) is not None:
            meta[key] = d[key]
    if d.get("simulatedFrom") is not None:
        meta["simulated_from_play"] = d["simulatedFrom"]

    return PlayTrack(meta=meta, roster=roster, frames=frames)


def read_index(source: str | Path) -> dict[str, Any]:
    """The app's ``index.json`` — one game's plays, in order."""
    return json.loads(Path(source).read_text())


def plays_in(directory: str | Path) -> list[Path]:
    """Every play file in a ``StreamingAssets/plays`` folder, index excluded."""
    return sorted(p for p in Path(directory).glob("*.json") if p.name != "index.json")


# ------------------------------------------------------------------- writing


def to_play_json(
    track: PlayTrack,
    play_state: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """A :class:`PlayTrack` in the Unity app's schema.

    This is the seam that lets the app draw anything this repository produces — the
    vision pipeline's output and the play model's counterfactual, not only its own
    exporter's files.

    ``play_state`` supplies the situation fields the contract has no home for yet
    (down, distance, the line of scrimmage, the description). Anything absent is
    left out of the document rather than zeroed: ``JsonUtility`` will default it,
    and a defaulted ``lineOfScrimmage`` of 0 draws the marker at the back of the end
    zone — visibly wrong, which is better than plausibly wrong. Where the ball is
    known at the snap, the line of scrimmage is taken from it rather than left out;
    Jeremy's own notes put the two within half a yard.
    """
    state = {**{k: v for k, v in track.meta.items() if k in PLAY_STATE_KEYS},
             **(play_state or {})}

    rate = int(track.meta.get("rate") or RATE)
    snap_t = track.event_time("ball_snap")
    if snap_t is None:
        snap_t = 0.0 if any(abs(t) < 1e-9 for t in track.times) else track.times[0]

    # frameId is 1-based and contiguous; snapFrameId points at the snap.
    snap_index = min(range(len(track.frames)), key=lambda i: abs(track.frames[i].t - snap_t))

    if "lineOfScrimmage" not in state:
        ball = track.frames[snap_index].ball
        if ball is not None:
            state["lineOfScrimmage"] = round(ball.x, 2)
            if state.get("yardsToGo") is not None:
                state["firstDownLine"] = round(
                    min(ball.x + float(state["yardsToGo"]), 110.0), 2
                )

    ids = nfl_ids(track)
    players = [
        {
            "nflId": ids[pid],
            "name": entry.name or entry.label or pid,
            "jerseyNumber": entry.num if entry.num is not None else 0,
            "club": entry.team,
            "position": entry.role or "",
            "side": _side(entry.team),
            # Outside schemaVersion 1, and the only way back to the track this came
            # from once the id has been synthesised.
            "trackId": pid,
        }
        for pid, entry in track.roster.items()
    ]

    frames = []
    for i, f in enumerate(track.frames):
        frame: dict[str, Any] = {
            "frameId": i + 1,
            "frameType": ("SNAP" if i == snap_index
                          else "BEFORE_SNAP" if i < snap_index else "AFTER_SNAP"),
            "event": f.events[0] if f.events else None,
            "players": [
                {
                    "nflId": ids[pid],
                    "x": round(p.x, 2),
                    "y": round(p.y, 2),
                    "o": round(float(contract_angle_to_ngs(p.o)), 2),
                    "dir": round(float(contract_angle_to_ngs(p.dir)), 2),
                    "s": round(p.s, 2),
                    # Not in schemaVersion 1. JsonUtility ignores fields it does
                    # not declare, so this is safe to send and is the only thing
                    # that would let the client badge a drawing as a drawing.
                    "oSource": p.o_source,
                }
                for pid, p in f.players.items()
            ],
        }
        if f.ball is not None:
            frame["ball"] = {"x": round(f.ball.x, 2), "y": round(f.ball.y, 2), "s": 0.0}
            if f.ball.source is not None:
                # Same reasoning as oSource above, and the same field the API uses
                # for a ball it got from the play model, so a client has one place
                # to look whatever produced the play: a ball measured off footage
                # and one interpolated between two people are both a position, and
                # only this says which. See contract.Ball.
                frame["ball"]["dataOrigin"] = f.ball.source
        frames.append(frame)

    doc: dict[str, Any] = {
        "schemaVersion": SCHEMA_VERSION,
        "gameId": _maybe_int(track.meta.get("game_id")) or 0,
        "playId": _maybe_int(track.meta.get("play_id")) or 0,
        "frameRate": rate,
        "snapFrameId": snap_index + 1,
    }
    for key in PLAY_STATE_KEYS:
        if state.get(key) is not None:
            doc[key] = state[key]
    doc["players"] = players
    doc["frames"] = frames

    # Also outside schemaVersion 1, and the reason this function is not a plain
    # field rename: a simulated play must not reach a renderer looking measured.
    from .export import provenance

    doc["provenance"] = provenance(track)
    doc["simulated"] = track.meta.get("source") == "simulated"
    if track.meta.get("simulated_from_play") is not None:
        # Which real play this forecast argues with. Outside schemaVersion 1 and
        # ignored by the client, but it is what lets a second run replace a
        # simulation rather than write another one beside it.
        doc["simulatedFrom"] = track.meta["simulated_from_play"]
    return doc


def write_play(
    out_dir: str | Path,
    track: PlayTrack,
    play_state: dict[str, Any] | None = None,
) -> Path:
    """Write one play where the Unity app looks for it.

    ``out_dir`` is a ``StreamingAssets/plays`` folder. The filename is
    ``{gameId}_{playId}.json``, which is what ``index.json`` points at.
    """
    doc = to_play_json(track, play_state)
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / f"{doc['gameId']}_{doc['playId']}.json"
    path.write_text(json.dumps(doc))
    return path


def write_index(
    out_dir: str | Path,
    tracks: list[PlayTrack],
    game_id: Any = None,
    home_team: str = "",
    visitor_team: str = "",
) -> Path:
    """The ``index.json`` the app's play navigator reads.

    The client requires plays grouped contiguously by quarter and drive, which
    holds when they are sorted by ``playId``, so they are.
    """
    rows = []
    for t in sorted(tracks, key=lambda t: _maybe_int(t.meta.get("play_id")) or 0):
        gid = _maybe_int(t.meta.get("game_id")) or 0
        pid = _maybe_int(t.meta.get("play_id")) or 0
        row = {"id": f"{gid}_{pid}", "gameId": gid, "playId": pid}
        for key in ("quarter", "gameClock", "driveNumber", "possessionTeam",
                    "down", "yardsToGo", "passResult", "description"):
            if t.meta.get(key) is not None:
                row[key] = t.meta[key]
        rows.append(row)

    doc = {
        "schemaVersion": SCHEMA_VERSION,
        "gameId": _maybe_int(game_id) or (rows[0]["gameId"] if rows else 0),
        "homeTeam": home_team,
        "visitorTeam": visitor_team,
        "plays": rows,
    }
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    path = out / "index.json"
    path.write_text(json.dumps(doc))
    return path


# ------------------------------------------------------------------- helpers


def nfl_ids(track: PlayTrack) -> dict[str, int]:
    """A stable integer id per player, because the Unity client keys on one.

    Its ``CLAUDE.md`` is explicit that players are matched by ``nflId`` and not by
    array position, so every player needs one and no two may share it. A track from
    a public release already has one. A track from the vision pipeline does not: its
    ids are ``t0``, ``t1``, ``t17`` — and mapping each of those to 0 puts the whole
    roster onto one player, which is what happened before this function existed.

    Synthetic ids start at :data:`SYNTHETIC_ID_BASE`, which is far above the real
    nflId range, so a vision track can never be mistaken for identified players.
    They are assigned in sorted order, so exporting the same track twice gives the
    same ids and a re-export does not shuffle the roster.
    """
    out: dict[str, int] = {}
    synthetic = 0
    for pid in sorted(track.roster):
        real = _maybe_int(pid)
        if real is not None and 0 < real < SYNTHETIC_ID_BASE:
            out[pid] = real
        else:
            synthetic += 1
            out[pid] = SYNTHETIC_ID_BASE + synthetic
    return out


SYNTHETIC_ID_BASE = 900_000
"""Where made-up player ids start. Real nflIds are five digits, so anything at or
above this is this repository's invention and says so by its magnitude."""


def _side(team: str | None) -> str:
    """``offense``/``defense``, which is what the app colours by."""
    t = (team or "").lower()
    if t.startswith("def"):
        return "defense"
    if t.startswith("off"):
        return "offense"
    return t or "?"


def _difference_acceleration(frames: list[TrackingFrame], dt: float) -> None:
    """Acceleration from the speed column, since this exporter drops it.

    Marked ``acceleration_measured: False`` in meta by the caller. Differencing a
    measured speed is honest; inventing the column would not be.
    """
    series: dict[str, list[tuple[int, PlayerFrame]]] = {}
    for i, f in enumerate(frames):
        for pid, p in f.players.items():
            series.setdefault(pid, []).append((i, p))
    for samples in series.values():
        for j, (_, p) in enumerate(samples):
            if len(samples) < 2:
                p.a = 0.0
            elif j == 0:
                p.a = abs(samples[1][1].s - p.s) / dt
            else:
                p.a = abs(p.s - samples[j - 1][1].s) / dt





def _maybe_int(v: Any) -> int | None:
    try:
        return int(v)
    except (TypeError, ValueError):
        return None
