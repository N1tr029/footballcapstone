"""What a detector hands back, and the one thing every detector must answer.

The protocol is per-frame and deliberately ignorant of video: a detector is asked
for frame N and may or may not be given pixels. That is what lets the Kaggle oracle
(boxes from a CSV), the synthetic generator (boxes computed from known tracking) and
a real YOLO run all sit behind the same call, so the rest of the pipeline never
learns which one it is talking to and the benchmark can swap them to attribute error.

The part worth reading is :meth:`Detection.anchor`. A box is not a position — it is
a rectangle around some part of a person, and *which* part decides both the pixel you
project and how far off the ground it is. Get that pairing wrong and the height
correction in :mod:`~gridiron_tracking.registration.camera` makes things worse rather
than better, confidently.
"""

from __future__ import annotations

from dataclasses import dataclass, field as dc_field
from typing import Any, Protocol, runtime_checkable

from ..registration.camera import HELMET_HEIGHT_YARDS, TORSO_HEIGHT_YARDS

__all__ = ["Detection", "Detector", "BOX_KINDS", "by_frame"]

# kind -> (where in the box the reference pixel is, how high that is off the turf)
BOX_KINDS: dict[str, tuple[str, float]] = {
    # A whole-body box: the bottom edge is where the player meets the ground, which
    # is the one part of him whose height is known exactly.
    "body": ("bottom_center", 0.0),
    # A helmet box has no ground contact anywhere in it, so the centre is used and
    # the height is carried instead.
    "helmet": ("center", HELMET_HEIGHT_YARDS),
    # A torso/numbers box, for detectors trained on shoulder pads.
    "torso": ("center", TORSO_HEIGHT_YARDS),
}


@dataclass
class Detection:
    """One box, on one frame."""

    frame: int
    x1: float
    y1: float
    x2: float
    y2: float
    score: float = 1.0
    kind: str = "body"
    label: str | None = None        # ground-truth identity, when a source knows it
    team: str | None = None
    meta: dict[str, Any] = dc_field(default_factory=dict)

    def __post_init__(self) -> None:
        if self.kind not in BOX_KINDS:
            raise ValueError(f"unknown box kind {self.kind!r}; expected one of {list(BOX_KINDS)}")

    @property
    def width(self) -> float:
        return self.x2 - self.x1

    @property
    def height(self) -> float:
        return self.y2 - self.y1

    @property
    def center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, (self.y1 + self.y2) / 2.0)

    @property
    def bottom_center(self) -> tuple[float, float]:
        return ((self.x1 + self.x2) / 2.0, self.y2)

    def anchor(self) -> tuple[tuple[float, float], float]:
        """The pixel to project, and how far above the turf that pixel is.

        Returned together on purpose. They are two halves of one fact, and every bug
        in this area comes from a call site that took one and assumed the other.
        """
        where, height = BOX_KINDS[self.kind]
        return (getattr(self, where), height)


@runtime_checkable
class Detector(Protocol):
    """Frame index in, boxes out.

    ``image`` is optional because two of the three implementations do not need it.
    A detector that does need pixels should raise if handed None rather than
    returning an empty list, so a missing frame reads as a failure and not as an
    empty field.
    """

    def detect(self, frame_index: int, image: Any = None) -> list[Detection]: ...


def by_frame(detections: list[Detection]) -> dict[int, list[Detection]]:
    out: dict[int, list[Detection]] = {}
    for d in detections:
        out.setdefault(d.frame, []).append(d)
    return out
