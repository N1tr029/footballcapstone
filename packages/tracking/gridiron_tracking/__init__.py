"""Footage to tracking data.

Video in, players in field coordinates out, plus an honest account of how much of
that to believe. See docs/roles.md, stream 2.
"""

from .contract import (
    EVENTS, O_SOURCES, RATE, Ball, PlayTrack, PlayerFrame, RosterEntry,
    TrackingFrame, validate, wrap_deg, shortest_deg,
)
from .field import HIGH_SCHOOL, NCAA, NFL, FieldSpec, Landmark, catalog, landmark

__all__ = [
    "EVENTS", "O_SOURCES", "RATE", "Ball", "PlayTrack", "PlayerFrame",
    "RosterEntry", "TrackingFrame", "validate", "wrap_deg", "shortest_deg",
    "FieldSpec", "Landmark", "NFL", "NCAA", "HIGH_SCHOOL", "catalog", "landmark",
]
