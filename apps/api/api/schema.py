"""
Pydantic models for Hindsight's play JSON (schemaVersion 1, see the Hindsight README).

The API serves the JSON that Hindsight's tools/export_plays.py already writes, so
these models document the shape and validate index files. Extra fields are allowed
so the exporter can add fields without breaking the API.

Conventions (from Hindsight): 1 unit = 1 yard, offense always moves toward +x,
angles in degrees with 0 = +y, clockwise (90 = +x).
"""
from typing import List, Optional

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = 1


class _Loose(BaseModel):
    model_config = ConfigDict(extra="allow")


class PlayIndexEntry(_Loose):
    id: str                       # "{gameId}_{playId}", used to fetch the play
    gameId: int
    playId: int
    quarter: int
    gameClock: str
    driveNumber: int
    possessionTeam: str
    down: int
    yardsToGo: int
    playType: str                 # pass | run | sack | scramble | kneel | spike
    passResult: str = ""
    yardsGained: int = 0
    touchdown: bool = False
    description: str = ""


class PlayIndex(_Loose):
    schemaVersion: int
    gameId: int
    homeTeam: str
    visitorTeam: str
    plays: List[PlayIndexEntry]


class GameSummary(BaseModel):
    gameId: int
    homeTeam: str
    visitorTeam: str
    playCount: int


class GameList(BaseModel):
    schemaVersion: int = SCHEMA_VERSION
    games: List[GameSummary]


# Play detail models, for the /docs page. The endpoint returns the exporter's file as-is.

class BallSample(_Loose):
    x: float
    y: float
    s: float = 0.0


class PlayerSample(_Loose):
    nflId: int
    x: float
    y: float
    o: Optional[float] = None
    dir: Optional[float] = None
    s: float = 0.0


class Frame(_Loose):
    frameId: int
    frameType: str
    event: Optional[str] = None
    ball: BallSample
    players: List[PlayerSample]


class PlayerInfo(_Loose):
    nflId: int
    name: str
    jerseyNumber: int
    club: str
    position: str
    side: str


class PlayDetail(_Loose):
    schemaVersion: int
    gameId: int
    playId: int
    homeTeam: str
    visitorTeam: str
    description: str
    quarter: int
    gameClock: str
    driveNumber: int
    down: int
    yardsToGo: int
    possessionTeam: str
    defensiveTeam: str
    lineOfScrimmage: float
    firstDownLine: float
    passResult: str
    frameRate: int
    snapFrameId: int
    players: List[PlayerInfo]
    frames: List[Frame]
