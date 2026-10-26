"""
Landon's suggested plays -> Hindsight play JSON.

decision_engine.py (build_suggested_play_payload) writes one object per line to
reports/suggested_plays.jsonl: the recorded frames up to the throw, then the
model's predicted frames, a straight-line ball to the recommended receiver,
inferred/simulated events, and the reason for the recommendation.

That payload speaks the Big Data Bowl 2026 dialect: snake_case columns, raw
coordinates that are NOT flipped for plays going left, and a list of events per
frame. Unity reads Hindsight's schemaVersion 1. This module is the translation,
using the same normalization Hindsight's exporter applies:

    playDirection == "left":  x = 120 - x,  y = 160/3 - y,
                              o = (o + 180) % 360,  dir = (dir + 180) % 360

Angles already share Hindsight's convention (0 = +y, clockwise), so a play going
right passes through unchanged.

What the 2026 input does not carry is left honest rather than invented: no jersey
numbers (0), no team abbreviations (clubs are OFF and DEF), no quarter or clock.
"""
import math
from typing import Any, Dict, List, Optional

FIELD_LENGTH = 120.0
FIELD_WIDTH = 160.0 / 3.0
OFFENSE, DEFENSE = "OFF", "DEF"


def _num(value, default: Optional[float] = None) -> Optional[float]:
    if value is None:
        return default
    try:
        f = float(value)
    except (TypeError, ValueError):
        return default
    return default if math.isnan(f) else f


def _r(value: float) -> float:
    return round(value, 2)


def to_hindsight(payload: Dict[str, Any], report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One suggested-play payload as a Hindsight play. `report` is the matching
    line of decisions.jsonl, when there is one; it supplies down, distance and the
    real play's description, which the payload leaves out."""
    report = report or {}
    game_id, play_id = int(payload["game_id"]), int(payload["play_id"])
    frames_in = sorted(payload.get("frames") or [], key=lambda f: int(f["frame_id"]))
    if not frames_in:
        raise ValueError(f"suggested play {game_id}_{play_id} has no frames")

    first_player = next((p for f in frames_in for p in f.get("players") or []), {})
    flip = str(first_player.get("play_direction", "right")).lower() == "left"

    def xy(x, y):
        x, y = _num(x, 0.0), _num(y, 0.0)
        return (FIELD_LENGTH - x, FIELD_WIDTH - y) if flip else (x, y)

    def angle(a):
        a = _num(a)
        if a is None:
            return None
        return (a + 180.0) % 360.0 if flip else a % 360.0

    snap = int(_num(payload.get("play_start_frame_id"), frames_in[0]["frame_id"]))
    roster: Dict[int, Dict[str, Any]] = {}
    frames: List[Dict[str, Any]] = []
    sim_start: Optional[int] = None
    source_frame: Optional[int] = None
    last_ball = None

    for f in frames_in:
        frame_id = int(f["frame_id"])
        players, passer = [], None
        for p in f.get("players") or []:
            nfl_id = int(_num(p.get("nfl_id"), 0))
            simulated = p.get("record_type") == "simulated_player"
            if simulated and sim_start is None:
                sim_start = frame_id
            # On recorded rows source_frame_id is just the row's own frame; on
            # predicted rows it is the last recorded frame the forecast started from.
            if simulated and source_frame is None and _num(p.get("source_frame_id")) is not None:
                source_frame = int(_num(p["source_frame_id"]))
            side = str(p.get("player_side") or "").lower()
            if nfl_id not in roster:
                roster[nfl_id] = {
                    "nflId": nfl_id,
                    "name": p.get("player_name") or "",
                    "jerseyNumber": 0,
                    "club": OFFENSE if side == "offense" else DEFENSE,
                    "position": p.get("player_position") or "",
                    "side": "offense" if side == "offense" else "defense",
                }
            x, y = xy(p.get("x"), p.get("y"))
            o, d = angle(p.get("o")), angle(p.get("dir"))
            # A forecast row carries o from the last observed frame; only dir is
            # recomputed from the predicted path, so o is assumed, not measured.
            if o is None:
                o_source = "from_dir" if d is not None else "assumed"
                o = d if d is not None else 0.0
            else:
                o_source = "assumed" if simulated else "measured"
            players.append({
                "nflId": nfl_id, "x": _r(x), "y": _r(y),
                "o": _r(o), "dir": _r(d if d is not None else 0.0),
                "s": _r(_num(p.get("s"), 0.0)), "oSource": o_source,
            })
            if p.get("player_role") == "Passer":
                passer = (x, y)

        # Unity draws a missing ball at the field's corner, so every frame gets one:
        # Landon's straight-line flight where it exists, the passer's hands before
        # the throw, and wherever it last was after it.
        b = f.get("ball")
        if b and _num(b.get("x")) is not None:
            bx, by = xy(b["x"], b["y"])
            ball_origin = b.get("data_origin") or "simulated_straight_line"
        elif passer is not None and sim_start is None:
            bx, by = passer
            ball_origin = "inferred_passer_position"
        elif last_ball is not None:
            bx, by, ball_origin = last_ball
        else:
            bx, by = xy(first_player.get("absolute_yardline_number"), FIELD_WIDTH / 2)
            ball_origin = "assumed"
        last_ball = (bx, by, ball_origin)

        events = [e.get("name") for e in f.get("events") or [] if e.get("name")]
        frames.append({
            "frameId": frame_id,
            "frameType": "BEFORE_SNAP" if frame_id < snap else ("SNAP" if frame_id == snap else "AFTER_SNAP"),
            "event": events[0] if events else None,
            "events": events,
            "ball": {"x": _r(bx), "y": _r(by), "s": 0.0, "dataOrigin": ball_origin},
            "players": players,
        })

    los = _num(first_player.get("absolute_yardline_number"))
    if los is not None and flip:
        los = FIELD_LENGTH - los
    to_go = _num(report.get("yards_to_go"))
    rec = payload.get("recommendation") or {}
    evidence = rec.get("evidence") or {}
    reason = rec.get("reason") or "No reason given by the model."

    return {
        "schemaVersion": 1,
        "gameId": game_id,
        "playId": play_id,
        "homeTeam": OFFENSE,
        "visitorTeam": DEFENSE,
        "description": f"SUGGESTED (model) — {reason}",
        "quarter": 0,
        "gameClock": "",
        "driveNumber": 0,
        "down": int(_num(report.get("down"), 0)),
        "yardsToGo": int(to_go or 0),
        "possessionTeam": OFFENSE,
        "defensiveTeam": DEFENSE,
        "lineOfScrimmage": _r(los or 0.0),
        "firstDownLine": _r(min((los or 0.0) + to_go, 110.0) if to_go is not None else (los or 0.0)),
        "passResult": report.get("pass_result") or "",
        "frameRate": 10,
        "snapFrameId": snap,
        "simulated": True,
        # Fields beyond schemaVersion 1. Unity ignores them until PlayData declares them.
        "suggested": True,
        "simulatedFrom": play_id,
        "simulationStartFrameId": sim_start,
        "sourceFrameId": source_frame,
        "realPlayDescription": report.get("play_description") or "",
        "recommendation": {
            "decision": rec.get("decision"),
            "targetNflId": rec.get("target_nfl_id"),
            "targetedNflId": evidence.get("targeted_receiver_id"),
            "reason": reason,
            "separationAdvantageYards": evidence.get("alternative_separation_advantage_yards"),
            "caveat": rec.get("caveat"),
            "evidence": evidence,
        },
        "dataNotes": [
            "Frames before simulationStartFrameId are recorded tracking; frames from it on are the model's prediction.",
            "The ball's flight is a straight line from the passer to the recommended receiver, not measured.",
            "ball_snap and pass_forward are inferred; pass_arrived is simulated.",
            "Jersey numbers, team abbreviations, quarter and clock are not in the source data.",
        ],
        "players": list(roster.values()),
        "frames": frames,
    }
