"""The shared shape, in Python.

This is a deliberate mirror of ``packages/replay-engine/prototype/engine/contract.js``.
Everything that draws a play draws one of these, whoever produced it — the
hand-authored fixture, this vision pipeline, or the play model's simulated
alternative — so the two definitions have to agree down to the wording of the
complaints. ``tests/test_contract_parity.py`` runs both validators over the same
plays and fails if they ever disagree; when you change one, change the other in the
same commit.

Field coordinates (docs/roles.md):

    x   0-120 yards downfield. The offense attacks +x. Goal line at x = 110.
    y   0-53.3 yards across, increasing toward the OFFENSE'S RIGHT.
    t   seconds relative to the snap.

Angles are DEGREES, 0 = facing +x (downfield), increasing toward +y. Two of them,
and they are not the same thing: ``dir`` is the direction a player is MOVING, ``o``
is the direction his body is POINTING. A vision pipeline measures neither directly,
which is exactly why ``o_source`` exists — see :mod:`gridiron_tracking.orientation`.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from math import isfinite
from typing import Any, Iterable

RATE = 10
"""Frames per second of the output. Public tracking releases are 10 Hz, so
everything this package produces is resampled onto that grid no matter what the
camera shot at."""

O_SOURCES = ("measured", "from_dir", "from_velocity", "assumed", "authored")
"""How an orientation was arrived at, worst-trusted last. ``authored`` means a
human drew it and it is never to be claimed by a pipeline."""

EVENTS = (
    "ball_snap", "handoff", "lateral", "pass_forward", "pass_arrived",
    "pass_outcome_caught", "tackle", "touchdown", "out_of_bounds",
)
"""The events a play is cut into, named after the tracking-data vocabulary so a
real play needs no translation."""

# The renderer's tolerance, not the field's dimensions: a player a couple of yards
# into the bench area is a real observation, a player at x = 400 is a broken
# homography. Kept identical to contract.js.
_X_MIN, _X_MAX = -5.0, 125.0
_Y_MIN, _Y_MAX = -5.0, 58.0


def _is_num(v: Any) -> bool:
    return isinstance(v, (int, float)) and not isinstance(v, bool) and isfinite(v)


@dataclass
class PlayerFrame:
    """One player at one instant."""

    x: float
    y: float
    s: float = 0.0
    a: float = 0.0
    dir: float = 0.0
    o: float = 0.0
    o_source: str = "assumed"

    def to_json(self) -> dict[str, Any]:
        return {
            "x": self.x, "y": self.y, "s": self.s, "a": self.a,
            "dir": self.dir, "o": self.o, "o_source": self.o_source,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "PlayerFrame":
        return cls(
            x=d["x"], y=d["y"], s=d.get("s", 0.0), a=d.get("a", 0.0),
            dir=d.get("dir", 0.0), o=d.get("o", 0.0),
            o_source=d.get("o_source", "assumed"),
        )


@dataclass
class Ball:
    x: float
    y: float
    z: float | None = None

    def to_json(self) -> dict[str, Any]:
        d: dict[str, Any] = {"x": self.x, "y": self.y}
        if self.z is not None:
            d["z"] = self.z
        return d

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Ball":
        return cls(x=d["x"], y=d["y"], z=d.get("z"))


@dataclass
class TrackingFrame:
    """One instant of one play."""

    t: float
    players: dict[str, PlayerFrame] = dc_field(default_factory=dict)
    ball: Ball | None = None
    events: list[str] = dc_field(default_factory=list)

    def to_json(self) -> dict[str, Any]:
        return {
            "t": self.t,
            "players": {k: v.to_json() for k, v in self.players.items()},
            "ball": self.ball.to_json() if self.ball else None,
            "events": list(self.events),
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "TrackingFrame":
        return cls(
            t=d["t"],
            players={k: PlayerFrame.from_json(v) for k, v in (d.get("players") or {}).items()},
            ball=Ball.from_json(d["ball"]) if d.get("ball") else None,
            events=list(d.get("events") or []),
        )


@dataclass
class RosterEntry:
    id: str
    team: str                      # 'offense' | 'defense', or a club abbreviation
    label: str | None = None
    name: str | None = None
    num: int | None = None
    role: str | None = None
    star: bool = False

    def to_json(self) -> dict[str, Any]:
        return {
            "id": self.id, "team": self.team, "label": self.label or self.id,
            "name": self.name, "num": self.num, "role": self.role, "star": self.star,
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "RosterEntry":
        return cls(
            id=d["id"], team=d["team"], label=d.get("label"), name=d.get("name"),
            num=d.get("num"), role=d.get("role"), star=bool(d.get("star")),
        )


@dataclass
class PlayTrack:
    """A whole play: who was on the field, and where they all were, when."""

    meta: dict[str, Any] = dc_field(default_factory=dict)
    roster: dict[str, RosterEntry] = dc_field(default_factory=dict)
    frames: list[TrackingFrame] = dc_field(default_factory=list)

    # ------------------------------------------------------------------ access

    @property
    def times(self) -> list[float]:
        return [f.t for f in self.frames]

    def player_ids(self) -> list[str]:
        return list(self.roster.keys())

    def series(self, player_id: str) -> list[tuple[float, PlayerFrame]]:
        """Every sample this player appears in, with its timestamp. Tracks from a
        vision pipeline are ragged — a man swallowed by the pile is genuinely
        absent rather than at (0, 0) — so this skips rather than interpolates."""
        out = []
        for f in self.frames:
            p = f.players.get(player_id)
            if p is not None:
                out.append((f.t, p))
        return out

    def frame_at(self, t: float) -> TrackingFrame | None:
        """Nearest frame, or None if the play does not cover that instant."""
        if not self.frames:
            return None
        best = min(self.frames, key=lambda f: abs(f.t - t))
        return best if abs(best.t - t) <= (0.5 / RATE) else None

    def event_time(self, name: str) -> float | None:
        for f in self.frames:
            if name in f.events:
                return f.t
        return None

    # --------------------------------------------------------------------- io

    def to_json(self) -> dict[str, Any]:
        return {
            "meta": self.meta,
            "roster": {k: v.to_json() for k, v in self.roster.items()},
            "frames": [f.to_json() for f in self.frames],
        }

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "PlayTrack":
        return cls(
            meta=d.get("meta") or {},
            roster={k: RosterEntry.from_json(v) for k, v in (d.get("roster") or {}).items()},
            frames=[TrackingFrame.from_json(f) for f in (d.get("frames") or [])],
        )


@dataclass
class Validation:
    ok: bool
    problems: list[str]
    events: dict[str, int]


def validate(track: PlayTrack | dict[str, Any]) -> Validation:
    """Worth running on anything before it reaches the renderer: a play that is
    wrong here is a play that is wrong in every view at once.

    Line-for-line the same checks, in the same order, with the same wording as
    ``GRID.validate`` in contract.js.
    """
    d = track.to_json() if isinstance(track, PlayTrack) else track
    problems: list[str] = []

    frames = (d or {}).get("frames") or []
    if not d or not frames:
        return Validation(False, ["no frames"], {})

    ids = list((d.get("roster") or {}).keys())
    if not ids:
        problems.append("no roster")

    prev: float | None = None
    bad_angles = missing = off_field = 0
    for f in frames:
        t = f.get("t")
        if not _is_num(t):
            problems.append("frame with no time")
        if prev is not None and _is_num(t) and t <= prev:
            problems.append(f"time runs backwards at t={_js_num(t)}")
        if _is_num(t):
            prev = t
        players = f.get("players") or {}
        for pid in ids:
            p = players.get(pid)
            if not p or not _is_num(p.get("x")) or not _is_num(p.get("y")):
                missing += 1
                continue
            if p["x"] < _X_MIN or p["x"] > _X_MAX or p["y"] < _Y_MIN or p["y"] > _Y_MAX:
                off_field += 1
            o = p.get("o")
            if not _is_num(o) or o < 0 or o >= 360:
                bad_angles += 1
            if p.get("o_source") not in O_SOURCES:
                bad_angles += 1

    if missing:
        problems.append(f"{missing} player samples missing position")
    if off_field:
        problems.append(f"{off_field} samples outside the field")
    if bad_angles:
        problems.append(f"{bad_angles} samples with a bad orientation")
    if not any(f.get("ball") for f in frames):
        problems.append("no ball track")

    events: dict[str, int] = {}
    for f in frames:
        for e in f.get("events") or []:
            events[e] = events.get(e, 0) + 1
    if not events.get("ball_snap"):
        problems.append("no ball_snap event")

    return Validation(not problems, problems, events)


def _js_num(v: float) -> str:
    """Format a number the way JavaScript's string coercion does, so the parity
    test compares the same message rather than ``1.0`` against ``1``."""
    if isinstance(v, float) and v.is_integer():
        return str(int(v))
    return repr(float(v))


# ------------------------------------------------------------------ angle helpers
# Everything downstream works in radians in world space; these are the only places
# the conversion happens, and they match contract.js exactly.


def wrap_deg(d: float) -> float:
    """Into [0, 360) — and the half-open end of that interval is load-bearing.

    Python's modulo returns exactly ``360.0`` for a small enough negative input:
    an ulp of 360 is about 5.7e-14, so anything in roughly ``(-2.8e-14, 0)`` rounds up
    to the excluded end of the interval. An angle of 360 then fails this module's own
    validator, which requires ``0 <= o < 360``. It is reachable in practice — a player
    running dead straight downfield has a differenced y velocity of a few times
    1e-14 either side of zero — and it presents as a play the renderer refuses for no
    visible reason. Larger negatives are fine: they wrap to 359.999..., which is a
    legal angle, and comparing angles anywhere downstream must use
    :func:`shortest_deg` rather than subtraction for exactly that reason.

    contract.js has the same expression and so the same latent bug; it has not bitten
    there because the renderer's angles come from interpolation rather than from
    differenced positions.
    """
    d = d % 360.0
    if d < 0:
        d += 360.0
    return 0.0 if d >= 360.0 else d


def shortest_deg(a: float, b: float) -> float:
    """Signed shortest turn from ``a`` to ``b``, in (-180, 180]. Angles have to be
    interpolated and differenced the short way round, or a man turning past due
    north spins all the way back through his own shoulder."""
    d = wrap_deg(b - a)
    return d - 360.0 if d > 180.0 else d


def mean_abs_angle_error(pairs: Iterable[tuple[float, float]]) -> float:
    """Mean absolute angular error in degrees, taken the short way round. The
    honest way to score ``o`` and ``dir``: a prediction of 359 against a truth of
    1 is two degrees out, not three hundred and fifty-eight."""
    errs = [abs(shortest_deg(a, b)) for a, b in pairs]
    return sum(errs) / len(errs) if errs else 0.0
