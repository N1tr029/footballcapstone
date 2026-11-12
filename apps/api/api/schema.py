"""
Pydantic models for Hindsight's play JSON (schemaVersion 1, see the Hindsight README).

The API serves the JSON that Hindsight's tools/export_plays.py already writes, so
these models document the shape and validate index files. Extra fields are allowed
so the exporter can add fields without breaking the API.

Conventions (from Hindsight): 1 unit = 1 yard, offense always moves toward +x,
angles in degrees with 0 = +y, clockwise (90 = +x).
"""
from typing import List, Literal, Optional

from pydantic import BaseModel, ConfigDict

SCHEMA_VERSION = 1


class _Loose(BaseModel):
    model_config = ConfigDict(extra="allow")


class PlayIndexEntry(_Loose):
    # Only the id fields are required. Producers (the exporter, the whatif command,
    # the vision pipeline) fill different subsets, and the API must not refuse to
    # start because one of them leaves a display field out.
    id: str                       # "{gameId}_{playId}", used to fetch the play
    gameId: int
    playId: int
    quarter: Optional[int] = None
    gameClock: Optional[str] = None
    driveNumber: Optional[int] = None
    possessionTeam: Optional[str] = None
    down: Optional[int] = None
    yardsToGo: Optional[int] = None
    playType: Optional[str] = None  # pass | run | sack | scramble | kneel | spike
    passResult: Optional[str] = None
    yardsGained: Optional[int] = None
    touchdown: Optional[bool] = None
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
    ball: Optional[BallSample] = None
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
    homeTeam: Optional[str] = None
    visitorTeam: Optional[str] = None
    description: Optional[str] = None
    quarter: Optional[int] = None
    gameClock: Optional[str] = None
    driveNumber: Optional[int] = None
    down: Optional[int] = None
    yardsToGo: Optional[int] = None
    possessionTeam: Optional[str] = None
    defensiveTeam: Optional[str] = None
    lineOfScrimmage: Optional[float] = None
    firstDownLine: Optional[float] = None
    passResult: Optional[str] = None
    frameRate: int
    snapFrameId: int
    players: List[PlayerInfo]
    frames: List[Frame]


class BDBTrackingRow(BaseModel):
    """Exact Big Data Bowl 2025 tracking_week CSV fields, also served as JSON rows.

    Video exports keep unknown fields null. Generated player/game/play identifiers
    and observation provenance are described by the companion metadata endpoint.
    """
    model_config = ConfigDict(allow_inf_nan=False)

    gameId: int
    playId: int
    nflId: Optional[int]
    displayName: str
    frameId: int
    frameType: Literal["BEFORE_SNAP", "SNAP", "AFTER_SNAP"]
    time: Optional[str]
    jerseyNumber: Optional[int]
    club: str
    playDirection: Optional[Literal["left", "right"]]
    x: float
    y: float
    s: Optional[float]
    a: Optional[float]
    dis: Optional[float]
    o: Optional[float]
    dir: Optional[float]
    event: Optional[str]
