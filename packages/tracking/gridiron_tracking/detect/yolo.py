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
        tiles: int = 1,
        tile_overlap: float = 0.2,
        merge_iou: float = 0.55,
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
        self.tiles = max(1, int(tiles))
        self.tile_overlap = tile_overlap
        self.merge_iou = merge_iou

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
        out = self._predict(image, frame_index, kw)
        if self.tiles > 1:
            out = merge_boxes(out + self._tiled(image, frame_index, kw), iou=self.merge_iou)

        if self.registration is not None:
            out = self._keep_on_field(out, frame_index)
        if self.max_players is not None and len(out) > self.max_players:
            out.sort(key=lambda d: -d.score)
            out = out[: self.max_players]
        return out

    # ------------------------------------------------------------------ tiling

    def _predict(self, image, frame_index: int, kw, dx: float = 0.0, dy: float = 0.0):
        res = self.model.predict(image, **kw)[0]
        out: list[Detection] = []
        if res.boxes is None or len(res.boxes) == 0:
            return out
        for (x1, y1, x2, y2), c in zip(res.boxes.xyxy.cpu().numpy(), res.boxes.conf.cpu().numpy()):
            out.append(Detection(
                frame=frame_index, x1=float(x1) + dx, y1=float(y1) + dy,
                x2=float(x2) + dx, y2=float(y2) + dy, score=float(c), kind="body",
            ))
        return out

    def _tiled(self, image, frame_index: int, kw) -> list[Detection]:
        """The same model again, over overlapping crops.

        A broadcast wide shot gives a far-hash player about twenty pixels of height, and
        after the letterboxing that ``imgsz`` does he is a handful. He is not a hard
        detection — he is a small one, and running the identical model on a crop where
        he is four times bigger finds him without any retraining. The cost is N² forward
        passes per frame and duplicate boxes wherever crops overlap, which is what
        :func:`merge_boxes` is for.

        The crops overlap on purpose: a player split across a tile boundary is detected
        as two halves or not at all, and an overlap wider than a player means every
        player is whole in at least one tile.
        """
        h, w = image.shape[:2]
        n = self.tiles
        tw, th = w / n, h / n
        ox, oy = tw * self.tile_overlap, th * self.tile_overlap
        out: list[Detection] = []
        for r in range(n):
            for c in range(n):
                x0 = int(max(0, c * tw - ox))
                y0 = int(max(0, r * th - oy))
                x1 = int(min(w, (c + 1) * tw + ox))
                y1 = int(min(h, (r + 1) * th + oy))
                if x1 - x0 < 32 or y1 - y0 < 32:
                    continue
                out.extend(self._predict(image[y0:y1, x0:x1], frame_index, kw, dx=x0, dy=y0))
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


def merge_boxes(dets: list[Detection], iou: float = 0.55) -> list[Detection]:
    """One box per person, when several passes found the same one.

    Plain non-maximum suppression, with one choice worth naming: boxes are kept in
    score order and a lower-scoring box survives only if it overlaps nothing already
    kept. Two players genuinely standing on each other in a pile overlap far less than
    two detections of one player, so the threshold separates duplicates from a crowd
    rather than thinning the crowd.
    """
    if len(dets) < 2:
        return list(dets)
    order = sorted(dets, key=lambda d: -d.score)
    kept: list[Detection] = []
    for d in order:
        if all(_iou(d, k) < iou for k in kept):
            kept.append(d)
    return kept


def _iou(a: Detection, b: Detection) -> float:
    x1, y1 = max(a.x1, b.x1), max(a.y1, b.y1)
    x2, y2 = min(a.x2, b.x2), min(a.y2, b.y2)
    inter = max(0.0, x2 - x1) * max(0.0, y2 - y1)
    if inter <= 0:
        return 0.0
    union = a.width * a.height + b.width * b.height - inter
    return inter / union if union > 0 else 0.0
