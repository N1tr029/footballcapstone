"""Big Data Bowl 2025 tracking rows, with uncertainty in a separate manifest.

Column order is checked against tracking_week_1.csv from the 2025 release. These
are schema-compatible video estimates, not league tracking measurements. Unknown
names, jerseys and wall-clock timestamps are never filled with invented values.
"""
from __future__ import annotations

import csv
import hashlib
import io
import json
import math
from datetime import datetime, timedelta, timezone
from pathlib import Path

from .contract import PlayTrack
from .hindsight import SYNTHETIC_ID_BASE, nfl_ids
from .sources.ngs import contract_angle_to_ngs

COLUMNS = [
    "gameId", "playId", "nflId", "displayName", "frameId", "frameType", "time",
    "jerseyNumber", "club", "playDirection", "x", "y", "s", "a", "dis", "o", "dir", "event",
]
SCHEMA = "nfl-big-data-bowl-2025"
LOCAL_GAME_ID = 999_000_001


def _integer(value):
    try:
        return int(value) if value is not None and str(value).isdigit() else None
    except (TypeError, ValueError):
        return None


def _number(value):
    return round(float(value), 6) if value is not None and math.isfinite(value) else None


def convert(track: PlayTrack, options: dict | None = None) -> tuple[list[dict], dict]:
    """Return the exact tracking columns and a manifest of the video-specific gaps.

    ``startTime``, when known, timestamps the first exported frame. Coordinates
    from an NGS import that was mirrored are restored to its original direction;
    unmirrored video coordinates retain their calibrated field orientation.
    """
    config = {**track.meta.get("bdb", {}), **(options or {})}
    game_id = _integer(config.get("gameId", track.meta.get("game_id")))
    synthetic_game = game_id is None or game_id == LOCAL_GAME_ID
    game_id = game_id if game_id is not None else LOCAL_GAME_ID
    play_id = _integer(config.get("playId", track.meta.get("play_id")))
    synthetic_play = play_id is None or synthetic_game
    if play_id is None:
        name = str(track.meta.get("play_id") or track.meta.get("clip", {}).get("name") or "video")
        play_id = int.from_bytes(hashlib.sha256(name.encode()).digest()[:4], "big") % 1_000_000_000 + 1
    direction = (config.get("playDirection") or track.meta.get("play_direction")
                 or track.meta.get("broadcast", {}).get("attacks"))
    if direction not in (None, "left", "right"):
        raise ValueError("playDirection must be left or right")
    clubs = config.get("teamClubs", {})
    mirror = bool(track.meta.get("mirrored", False))
    start = config.get("startTime")
    if start:
        start = datetime.fromisoformat(str(start).replace("Z", "+00:00"))
        if start.tzinfo is not None:
            start = start.astimezone(timezone.utc).replace(tzinfo=None)
    rate = float(track.meta.get("rate") or 10)
    if rate <= 0:
        raise ValueError("tracking rate must be positive")
    ids = nfl_ids(track)
    snap_t = track.event_time("ball_snap")
    snap = (min(range(len(track.frames)), key=lambda i: abs(track.frames[i].t - (snap_t or 0)))
            if track.frames else None)
    rows, previous = [], {}
    ball_sources, orientation_sources, events = {}, {}, {}
    for index, frame in enumerate(track.frames):
        fid = index + 1
        clock = ((start + timedelta(seconds=index / rate)).strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                 if start else None)
        common = {
            "gameId": game_id, "playId": play_id, "frameId": fid,
            "frameType": "SNAP" if index == snap else "BEFORE_SNAP" if index < snap else "AFTER_SNAP",
            "time": clock, "playDirection": direction,
            "event": frame.events[0] if frame.events else None,
        }
        if frame.events:
            events[str(fid)] = list(frame.events)
        orientation_sources[str(fid)] = {}
        for pid, sample in frame.players.items():
            entry = track.roster[pid]
            x, y = (120 - sample.x, 53.3 - sample.y) if mirror else (sample.x, sample.y)
            old = previous.get(pid)
            distance = (math.hypot(x-old[1], y-old[2]) if old and old[0] == index-1 else None)
            previous[pid] = (index, x, y)
            bearing = lambda a: float((contract_angle_to_ngs(a) + (180 if mirror else 0)) % 360)
            row = {**common, "nflId": ids[pid], "displayName": entry.name or entry.label or pid,
                   "jerseyNumber": entry.num, "club": clubs.get(entry.team, entry.team or "unknown"),
                   "x": _number(x), "y": _number(y), "s": _number(sample.s), "a": _number(sample.a),
                   "dis": _number(distance), "o": _number(bearing(sample.o)), "dir": _number(bearing(sample.dir))}
            rows.append({column: row[column] for column in COLUMNS})
            orientation_sources[str(fid)][str(ids[pid])] = sample.o_source
        if frame.ball is not None:
            ball = frame.ball
            x, y = (120-ball.x, 53.3-ball.y) if mirror else (ball.x, ball.y)
            old = previous.get("ball")
            distance = (math.hypot(x-old[1], y-old[2]) if old and old[0] == index-1 else None)
            speed = distance * rate if distance is not None else None
            acceleration = (speed-old[3]) * rate if speed is not None and old[3] is not None else None
            previous["ball"] = (index, x, y, speed)
            angle = ((90-math.degrees(math.atan2(y-old[2], x-old[1]))) % 360
                     if distance else None)
            row = {**common, "nflId": None, "displayName": "football", "jerseyNumber": None,
                   "club": "football", "x": _number(x), "y": _number(y), "s": _number(speed),
                   "a": _number(acceleration), "dis": _number(distance), "o": None, "dir": _number(angle)}
            rows.append({column: row[column] for column in COLUMNS})
            ball_sources[str(fid)] = ball.source or "unverified"
    metadata = {
        "schema": SCHEMA, "columns": COLUMNS, "frameRate": rate,
        "snapFrameId": snap + 1 if snap is not None else None,
        "gameId": game_id, "playId": play_id, "syntheticGameId": synthetic_game,
        "syntheticPlayId": synthetic_play, "wallClockTimeKnown": start is not None,
        "playDirectionKnown": direction is not None, "coordinatesMirroredBack": mirror,
        "source": track.meta.get("source", "unknown"),
        "coordinatesMeasured": track.meta.get("source") == "ngs",
        "playerIdMap": {str(ids[pid]): {"trackId": pid, "synthetic": ids[pid] >= SYNTHETIC_ID_BASE,
                                       "team": entry.team} for pid, entry in track.roster.items()},
        "teamClubs": clubs, "ballSourcesByFrame": ball_sources,
        "frameTimesSeconds": {str(i+1): f.t for i, f in enumerate(track.frames)},
        "sourceFps": track.meta.get("fps_in"),
        "snapSourceFrame": track.meta.get("clip", {}).get("snap_source_frame"),
        "ballImageObservations": track.meta.get("ball", {}).get("image_observations", {}),
        "orientationSourcesByFrame": orientation_sources, "eventsByFrame": events,
        "note": "Video coordinates are estimates; coordinatesMeasured identifies league measurements. "
                "Synthetic IDs are local identifiers, not verified NFL identities. "
                "Null timestamps, jerseys and absent ball rows mean unknown; no missing positions are filled.",
    }
    return rows, metadata


def to_csv(rows: list[dict]) -> str:
    output = io.StringIO(newline="")
    writer = csv.DictWriter(output, fieldnames=COLUMNS, lineterminator="\n")
    writer.writeheader()
    writer.writerows(rows)
    return output.getvalue()


def write_play(directory: str | Path, track: PlayTrack, options: dict | None = None) -> dict[str, Path]:
    rows, metadata = convert(track, options)
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    paths = {"json": directory / "tracking.json", "csv": directory / "tracking.csv",
             "metadata": directory / "tracking-metadata.json"}
    paths["json"].write_text(json.dumps(rows, allow_nan=False))
    paths["csv"].write_text(to_csv(rows))
    paths["metadata"].write_text(json.dumps(metadata, indent=2, allow_nan=False) + "\n")
    return paths
