"""Field geometry, and the catalog of points a human can point at.

Registration needs correspondences: a pixel in the image, and the place on the
field it is a picture of. The second half of that pair has to come from somewhere,
and painted lines are the only thing on a football field whose position is known
to the inch without measuring it. This module is that list.

The one number that is not the same everywhere is the hash inset, and it is not a
small difference: NFL hashes sit 23.6 yards from the sideline, college 20, high
school 17.8. Registering a Friday-night high-school field against NFL hash
positions puts every player about six yards sideways with a perfectly clean-looking
reprojection error, because the homography will happily absorb the mistake. Pick
the right :class:`FieldSpec`.
"""

from __future__ import annotations

from dataclasses import dataclass

# The renderer (contract.js) uses 53.3. The true width is 160 feet, or 53 and a
# third yards; the four-centimetre difference is an order of magnitude below the
# error floor of anything here, and agreeing with the renderer is worth more than
# the third decimal place.
WIDTH = 53.3
LENGTH = 120.0
GOAL_A = 10.0
GOAL_B = 110.0


@dataclass(frozen=True)
class FieldSpec:
    """A level of football, and the two paint dimensions that vary with it."""

    name: str
    hash_inset: float       # yards from each sideline to the hash marks
    number_inset: float     # yards from each sideline to the bottom of the numbers
    width: float = WIDTH
    length: float = LENGTH
    goal_a: float = GOAL_A
    goal_b: float = GOAL_B

    @property
    def hashes(self) -> tuple[float, float]:
        return (self.hash_inset, self.width - self.hash_inset)

    @property
    def mid_y(self) -> float:
        return self.width / 2.0


# 70 feet 9 inches from each sideline; matches GRID.FIELD.hashes in contract.js.
NFL = FieldSpec("NFL", hash_inset=23.58, number_inset=12.0)
# Hashes 40 feet apart, so 60 feet from each sideline.
NCAA = FieldSpec("NCAA", hash_inset=20.0, number_inset=9.0)
# NFHS: hashes 53 feet 4 inches apart — a third of the width, and much wider than
# either code above. This is the one that matters for the coach-uploads-film case.
HIGH_SCHOOL = FieldSpec("NFHS", hash_inset=17.78, number_inset=9.0)

SPECS = {"nfl": NFL, "ncaa": NCAA, "hs": HIGH_SCHOOL, "nfhs": HIGH_SCHOOL}


@dataclass(frozen=True)
class Landmark:
    """A point on the field whose position is known because it is painted there."""

    id: str
    x: float
    y: float
    label: str

    @property
    def xy(self) -> tuple[float, float]:
        return (self.x, self.y)


def painted_number(x: float) -> int:
    """The number painted on the yard line at ``x`` — what an operator reads off
    the grass. Both 25-yard lines are painted "25"; ``x`` is what tells them apart."""
    return int(round(50.0 - abs(x - 60.0)))


def _line_id(x: float) -> tuple[str, str]:
    n = painted_number(x)
    if x == GOAL_A:
        return "goalA", "A goal line"
    if x == GOAL_B:
        return "goalB", "B goal line"
    if x == 0.0:
        return "backA", "A back of end zone"
    if x == LENGTH:
        return "backB", "B back of end zone"
    if n == 50:
        return "y50", "50 yard line"
    side = "A" if x < 60.0 else "B"
    return f"y{n}{side}", f"{n} yard line ({side} side)"


_LANES = (
    ("sideline_y0", "at the y=0 sideline"),
    ("hash_y0", "at the y=0 hash"),
    ("hash_ymax", "at the far hash"),
    ("sideline_ymax", "at the far sideline"),
)


def catalog(spec: FieldSpec = NFL) -> dict[str, Landmark]:
    """Every point worth clicking, keyed by id.

    Four lanes across (both sidelines, both hash rows) at every five-yard line,
    plus the goal lines and the back lines of both end zones. Around 170 points —
    an operator uses four to six of them, but which four depends entirely on where
    the camera is, so the whole grid has to be offerable.
    """
    lo_hash, hi_hash = spec.hashes
    lane_y = {
        "sideline_y0": 0.0,
        "hash_y0": lo_hash,
        "hash_ymax": hi_hash,
        "sideline_ymax": spec.width,
    }

    xs: list[float] = [0.0, spec.length]
    x = spec.goal_a
    while x <= spec.goal_b + 1e-9:
        xs.append(round(x, 3))
        x += 5.0

    out: dict[str, Landmark] = {}
    for xv in sorted(set(xs)):
        line_id, line_label = _line_id(xv)
        for lane, lane_label in _LANES:
            # End-zone back lines have no hash marks painted across them.
            if xv in (0.0, spec.length) and lane.startswith("hash"):
                continue
            lid = f"{line_id}/{lane}"
            out[lid] = Landmark(lid, xv, lane_y[lane], f"{line_label} {lane_label}")
    return out


def landmark(lid: str, spec: FieldSpec = NFL) -> Landmark:
    try:
        return catalog(spec)[lid]
    except KeyError:
        raise KeyError(
            f"no such landmark {lid!r}; ids look like 'y30A/hash_y0' or 'goalB/sideline_ymax'"
        ) from None


def on_field(x: float, y: float, spec: FieldSpec = NFL, margin: float = 0.0) -> bool:
    return -margin <= x <= spec.length + margin and -margin <= y <= spec.width + margin


def clamp_to_field(x: float, y: float, spec: FieldSpec = NFL, margin: float = 5.0) -> tuple[float, float]:
    """Pull a point back to the renderer's tolerated box.

    Use it on export, never on measurement: clamping a bad estimate makes a broken
    homography look like a player standing politely on the sideline, and the
    quality report is the place that failure belongs.
    """
    return (
        min(max(x, -margin), spec.length + margin),
        min(max(y, -margin), spec.width + margin),
    )
