"""Finding people in a frame.

A COCO-pretrained detector already knows what a person looks like, and a football
player is a person, so this works on the first video it sees without any training.
It will not be *good* — a general model on a wide sideline shot misses players in the
pile, loses the ones at the far hash when they get small, and cheerfully boxes the
referee, a coach on the sideline and anyone in the front row of the stand.

That is the expected starting point, not a failure, and the pipeline is arranged so
it can be measured rather than argued about: every box is projected into field
coordinates before tracking, so a detection in the stands lands off the field and can
be dropped on geometry instead of on a confidence threshold.

The upgrades, in the order worth doing them:

1. fine-tune on football frames — the single biggest win, and the public helmet and
   player datasets already exist for it
2. a pose model, which gives shoulder keypoints and so a *measured* ``o`` instead of
   one inferred from direction of travel
3. a helmet detector, which survives the pile far better than a body box, at the cost
   of needing the height correction in ``registration/camera.py`` to be right
"""

from __future__ import annotations

from typing import Any

import numpy as np

from . import Detection

PERSON_CLASS = 0


class YoloDetector:
    """Ultralytics YOLO, restricted to people.

    ``on_field`` is the useful filter and it is not a confidence threshold: a box is
    kept if its ground point projects to somewhere on the football field. A steward
    on the touchline and a spectator in row three are both confidently people, and
    both land outside the sidelines once the homography is applied.
    """

    def __init__(
        self,
        weights: str = "yolo11n.pt",
        conf: float = 0.25,
        imgsz: int = 1280,
        device: str | None = None,
        registration: Any = None,
        field_margin: float = 4.0,
        max_players: int | None = None,
    ) -> None:
        try:
            from ultralytics import YOLO
        except ImportError as e:  # pragma: no cover - depends on the install
            raise ImportError(
                "detection needs ultralytics: pip install 'gridiron-tracking[detect]'"
            ) from e
        self.model = YOLO(weights)
        self.conf = conf
        self.imgsz = imgsz
        self.device = device
        self.registration = registration
        self.field_margin = field_margin
        self.max_players = max_players

    def detect(self, frame_index: int, image: Any = None) -> list[Detection]:
        if image is None:
            raise ValueError(
                f"YoloDetector needs pixels for frame {frame_index}; it was handed None. "
                "Pass images= to the pipeline."
            )
        kw: dict[str, Any] = dict(
            conf=self.conf, imgsz=self.imgsz, classes=[PERSON_CLASS], verbose=False
        )
        if self.device:
            kw["device"] = self.device
        res = self.model.predict(image, **kw)[0]

        out: list[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out

        xyxy = res.boxes.xyxy.cpu().numpy()
        confs = res.boxes.conf.cpu().numpy()
        for (x1, y1, x2, y2), c in zip(xyxy, confs):
            out.append(Detection(
                frame=frame_index, x1=float(x1), y1=float(y1), x2=float(x2), y2=float(y2),
                score=float(c), kind="body",
            ))

        if self.registration is not None:
            out = self._keep_on_field(out, frame_index)
        if self.max_players is not None and len(out) > self.max_players:
            out.sort(key=lambda d: -d.score)
            out = out[: self.max_players]
        return out

    def _keep_on_field(self, dets: list[Detection], frame_index: int) -> list[Detection]:
        from ..field import NFL

        pts = np.array([d.anchor()[0] for d in dets], dtype=float)
        got = np.atleast_2d(self.registration.to_field(frame_index, pts, height_yards=0.0))
        spec = getattr(self.registration, "spec", NFL)
        m = self.field_margin
        keep = []
        for d, (x, y) in zip(dets, got):
            if not (np.isfinite(x) and np.isfinite(y)):
                continue
            if -m <= x <= spec.length + m and -m <= y <= spec.width + m:
                keep.append(d)
        return keep
