"""
The play model's suggested plays -> Hindsight play JSON.

decision_engine.py writes one object per line to reports/suggested_plays.jsonl:
the recorded frames up to the throw, then the model's forecast frames, a
straight-line ball to the recommended receiver, events, and the reason for the
recommendation.

That payload speaks the Big Data Bowl dialect: snake_case columns, raw
coordinates that are NOT flipped for plays going left, and a list of events per
frame. Unity reads Hindsight's schemaVersion 1. This module is the translation,
using the same normalization Hindsight's exporter applies:

    playDirection == "left":  x = 120 - x,  y = 160/3 - y,
                              o = (o + 180) % 360,  dir = (dir + 180) % 360

Angles already share Hindsight's convention (0 = +y, clockwise), so a play going
right passes through unchanged.

Two payload shapes are read. The 2025 one carries real clubs, jersey numbers and
the play's context, with a `roster` block once per play. The older 2026 one has
none of that, which is left honest rather than invented: jersey numbers 0, clubs
OFF and DEF, no quarter or clock.
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


def _int(value) -> Optional[int]:
    n = _num(value)
    return None if n is None else int(n)


def to_hindsight(payload: Dict[str, Any], report: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """One suggested-play payload as a Hindsight play. `report` is the matching
    line of decisions.jsonl, when there is one; it fills down, distance and the
    real play's description when the payload leaves them out.

    Reads both payloads decision_engine has written: the 2026 one (every field on
    every player row, no teams or jersey numbers) and the 2025 one (a `roster`
    block once per play, real clubs and jerseys, play context at the top)."""
    report = report or {}
    game_id, play_id = int(payload["game_id"]), int(payload["play_id"])
    frames_in = sorted(payload.get("frames") or [], key=lambda f: int(f["frame_id"]))
    if not frames_in:
        raise ValueError(f"suggested play {game_id}_{play_id} has no frames")

    first_player = next((p for f in frames_in for p in f.get("players") or []), {})
    flip = str(payload.get("play_direction") or first_player.get("play_direction", "right")).lower() == "left"
    possession = payload.get("possession_team") or None
    info_by_id = {int(r["nfl_id"]): r for r in payload.get("roster") or [] if r.get("nfl_id") is not None}

    def xy(x, y):
        x, y = _num(x, 0.0), _num(y, 0.0)
        return (FIELD_LENGTH - x, FIELD_WIDTH - y) if flip else (x, y)

    def angle(a):
        a = _num(a)
        if a is None:
            return None
        return (a + 180.0) % 360.0 if flip else a % 360.0

    def roster_entry(nfl_id: int, row: Dict[str, Any]) -> Dict[str, Any]:
        info = {**row, **info_by_id.get(nfl_id, {})}
        side = str(info.get("player_side") or "").lower()
        club = info.get("club")
        if side not in ("offense", "defense") and club and possession:
            side = "offense" if club == possession else "defense"
        return {
            "nflId": nfl_id,
            "name": info.get("player_name") or "",
            "jerseyNumber": _int(info.get("jersey_number")) or 0,
            "club": club or (OFFENSE if side == "offense" else DEFENSE),
            "position": info.get("player_position") or "",
            "side": "offense" if side == "offense" else "defense",
            "_role": info.get("player_role"),
        }

    snap = int(_num(payload.get("play_start_frame_id"), frames_in[0]["frame_id"]))
    roster: Dict[int, Dict[str, Any]] = {}
    frames: List[Dict[str, Any]] = []
    sim_start: Optional[int] = None
    source_frame: Optional[int] = _int(payload.get("source_frame_id"))
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
            if nfl_id not in roster:
                roster[nfl_id] = roster_entry(nfl_id, p)
            x, y = xy(p.get("x"), p.get("y"))
            o, d = angle(p.get("o")), angle(p.get("dir"))
            if p.get("o_source") and o is not None:
                o_source = p["o_source"]
            elif o is None:
                o_source = "from_dir" if d is not None else "assumed"
                o = d if d is not None else 0.0
            else:
                # A 2026 forecast row carries o from the last observed frame; only
                # dir is recomputed from the predicted path, so o is assumed.
                o_source = "assumed" if simulated else "measured"
            players.append({
                "nflId": nfl_id, "x": _r(x), "y": _r(y),
                "o": _r(o), "dir": _r(d if d is not None else 0.0),
                "s": _r(_num(p.get("s"), 0.0)), "oSource": o_source,
            })
            if roster[nfl_id]["_role"] == "Passer":
                passer = (x, y)

        # Unity draws a missing ball at the field's corner, so every frame gets one:
        # the payload's ball where it exists, the passer's hands before the throw,
        # and wherever it last was after it.
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

    def context(key):
        value = payload.get(key)
        return value if value not in (None, "") else report.get(key)

    los = _num(payload.get("absolute_yardline_number"), _num(first_player.get("absolute_yardline_number")))
    if los is not None and flip:
        los = FIELD_LENGTH - los
    to_go = _num(context("yards_to_go"))
    rec = payload.get("recommendation") or {}
    evidence = rec.get("evidence") or {}
    reason = rec.get("reason") or "No reason given by the model."
    has_teams = bool(possession)
    home = payload.get("home_team") or (possession if has_teams else OFFENSE)

    notes = ["Frames before simulationStartFrameId are recorded tracking; frames from it on are the model's prediction.",
             "The ball's flight after the throw is a straight line to the recommended receiver, not measured."]
    if payload.get("data_release") == "nfl-big-data-bowl-2025":
        notes.append("After the throw each player follows a smooth path to where the model forecast him at the catch; "
                     "body angles there follow the path (oSource from_velocity).")
        notes.append("ball_snap and pass_forward are recorded; pass_arrived is simulated.")
    else:
        notes.append("ball_snap and pass_forward are inferred; pass_arrived is simulated.")
        notes.append("Jersey numbers, team abbreviations, quarter and clock are not in the source data.")

    for entry in roster.values():
        entry.pop("_role", None)

    return {
        "schemaVersion": 1,
        "gameId": game_id,
        "playId": play_id,
        "homeTeam": home,
        "visitorTeam": payload.get("visitor_team") or (payload.get("defensive_team") or DEFENSE),
        "description": f"SUGGESTED (model) — {reason}",
        "quarter": _int(context("quarter")) or 0,
        "gameClock": context("game_clock") or "",
        "driveNumber": 0,
        "down": _int(context("down")) or 0,
        "yardsToGo": int(to_go or 0),
        "possessionTeam": possession or OFFENSE,
        "defensiveTeam": payload.get("defensive_team") or DEFENSE,
        "lineOfScrimmage": _r(los or 0.0),
        "firstDownLine": _r(min((los or 0.0) + to_go, 110.0) if to_go is not None else (los or 0.0)),
        "passResult": context("pass_result") or "",
        "frameRate": 10,
        "snapFrameId": snap,
        "simulated": True,
        # Fields beyond schemaVersion 1. Unity ignores them until PlayData declares them.
        "suggested": True,
        "simulatedFrom": play_id,
        "simulationStartFrameId": sim_start,
        "sourceFrameId": source_frame,
        "realPlayDescription": context("play_description") or "",
        "recommendation": {
            "decision": rec.get("decision"),
            "targetNflId": _int(rec.get("target_nfl_id")),
            "targetedNflId": _int(rec.get("targeted_nfl_id", evidence.get("targeted_receiver_id"))),
            "reason": reason,
            "separationAdvantageYards": evidence.get("alternative_separation_advantage_yards"),
            "caveat": rec.get("caveat"),
            "evidence": evidence,
        },
        "dataNotes": notes,
        "players": list(roster.values()),
        "frames": frames,
    }
