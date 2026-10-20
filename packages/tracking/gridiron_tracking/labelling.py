"""Labelling footballs, and turning the result into a detector.

The generic COCO sports-ball class cannot find a football in this footage. Untiled
it finds one on 5% of frames; tiled it finds the down-and-distance graphic on 42%;
tiled with static clutter rejected it finds a player. That is written up in the
package README with all three measurements. Nothing else in the chain from video to
a counterfactual is waiting on research — this is the only piece.

The fix is an ordinary supervised one: a few hundred boxes drawn on footballs,
fine-tuned onto the detector that is already in the repository. This module is the
machinery around that so the only part a person does is the part only a person can.

    python -m gridiron_tracking label              # extract frames, open the UI
    python -m gridiron_tracking train-ball         # export, fine-tune, measure

**Frames are sampled, not taken whole.** Consecutive frames of a 30 Hz clip are
nearly identical and a model learns nothing from the second one; every fourth frame
across eight clips is a few hundred images covering eight camera setups, four
levels of football and two resolutions, which is a far better few hundred than one
clip's worth.

**Frames with no visible ball are kept.** A detector trained only on frames
containing the thing it is looking for learns to find it everywhere. The ball
spends much of a play in somebody's hands or behind a pile, and those frames are
exactly the negatives that stop a fine-tune from re-learning the graphic.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any

FRAME_DIR = Path("data/labels/frames")
LABEL_FILE = Path("data/labels/labels.json")
DATASET_DIR = Path("data/labels/dataset")
WEIGHTS_DIR = Path("data/labels/weights")

BOX_PIXELS = 22
"""Default box side, in source pixels. A football in a broadcast wide shot is
roughly this across; the UI lets it be changed per label and records what was
used, because a high-school clip at 640x360 is a different proposition."""


@dataclass
class Shot:
    """One frame offered for labelling."""

    key: str              # "<clip>/<frame index>"
    clip: str
    frame: int
    path: str             # relative to the repo root
    width: int
    height: int

    def to_json(self) -> dict[str, Any]:
        return {"key": self.key, "clip": self.clip, "frame": self.frame,
                "path": self.path, "width": self.width, "height": self.height}


@dataclass
class Label:
    """Where the ball is in one frame, or that it is not visible."""

    key: str
    x: float | None = None     # centre, source pixels
    y: float | None = None
    size: float = BOX_PIXELS

    @property
    def visible(self) -> bool:
        return self.x is not None and self.y is not None

    def to_json(self) -> dict[str, Any]:
        return {"key": self.key, "x": self.x, "y": self.y, "size": self.size}

    @classmethod
    def from_json(cls, d: dict[str, Any]) -> "Label":
        return cls(key=d["key"], x=d.get("x"), y=d.get("y"),
                   size=float(d.get("size") or BOX_PIXELS))


@dataclass
class LabelSet:
    labels: dict[str, Label] = dc_field(default_factory=dict)

    @classmethod
    def load(cls, path: Path = LABEL_FILE) -> "LabelSet":
        if not Path(path).exists():
            return cls()
        raw = json.loads(Path(path).read_text())
        return cls({k: Label.from_json(v) for k, v in raw.items()})

    def save(self, path: Path = LABEL_FILE) -> Path:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(
            {k: v.to_json() for k, v in sorted(self.labels.items())}, indent=1))
        return path

    def counts(self) -> dict[str, int]:
        visible = sum(1 for v in self.labels.values() if v.visible)
        return {"labelled": len(self.labels), "with_a_ball": visible,
                "without": len(self.labels) - visible}

    def summary(self) -> str:
        c = self.counts()
        return (f"{c['labelled']} frames labelled — {c['with_a_ball']} with a ball, "
                f"{c['without']} without")


# ------------------------------------------------------------------ extraction


def extract(specs, every: int = 4, out_dir: Path = FRAME_DIR, log=print) -> list[Shot]:
    """Sample frames from each clip's own window and write them as jpgs.

    The clip's window, not the whole video: a spec's ``start``/``stop`` is the play,
    and frames outside it are a different play or a replay or a crowd shot.
    """
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise ImportError("extracting frames needs opencv") from e

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    shots: list[Shot] = []

    for spec in specs:
        cap = cv2.VideoCapture(spec.video)
        if not cap.isOpened():
            log(f"  {spec.name}: cannot open {spec.video}")
            continue
        stop = spec.stop if spec.stop is not None else int(1e9)
        i = written = 0
        while i <= stop:
            ok, img = cap.read()
            if not ok:
                break
            if i >= spec.start and (i - spec.start) % every == 0:
                name = f"{spec.name}_{i:05d}.jpg"
                path = out_dir / name
                if not path.exists():
                    cv2.imwrite(str(path), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                h, w = img.shape[:2]
                shots.append(Shot(key=f"{spec.name}/{i}", clip=spec.name, frame=i,
                                  path=str(path), width=w, height=h))
                written += 1
            i += 1
        cap.release()
        log(f"  {spec.name:<18} {written} frames every {every}")

    (out_dir / "manifest.json").write_text(
        json.dumps([s.to_json() for s in shots], indent=1))
    return shots


def load_manifest(out_dir: Path = FRAME_DIR) -> list[Shot]:
    path = Path(out_dir) / "manifest.json"
    if not path.exists():
        return []
    return [Shot(**d) for d in json.loads(path.read_text())]


# --------------------------------------------------------------------- dataset


def to_yolo(
    shots: list[Shot],
    labels: LabelSet,
    out_dir: Path = DATASET_DIR,
    val_fraction: float = 0.2,
    log=print,
) -> Path:
    """Write a YOLO dataset and return the path to its data.yaml.

    **Split by clip, not by frame.** Frames four apart from one camera are nearly
    the same picture, so a random split puts near-duplicates in both halves and the
    validation score measures memorisation. Holding out whole clips measures the
    only thing worth knowing: whether it finds the ball on footage it has not seen.
    """
    import shutil

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)

    by_key = {s.key: s for s in shots}
    usable = [(by_key[k], v) for k, v in labels.labels.items() if k in by_key]
    if not usable:
        raise ValueError("nothing labelled yet")

    clips = sorted({s.clip for s, _ in usable})
    n_val = max(1, round(len(clips) * val_fraction)) if len(clips) > 1 else 0
    val_clips = set(clips[-n_val:]) if n_val else set()

    counts = {"train": 0, "val": 0, "boxes": 0}
    for shot, label in usable:
        split = "val" if shot.clip in val_clips else "train"
        images = out_dir / "images" / split
        texts = out_dir / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        texts.mkdir(parents=True, exist_ok=True)

        stem = f"{shot.clip}_{shot.frame:05d}"
        shutil.copy2(shot.path, images / f"{stem}.jpg")

        lines = []
        if label.visible:
            half = label.size / 2.0
            lines.append(
                f"0 {label.x / shot.width:.6f} {label.y / shot.height:.6f} "
                f"{2 * half / shot.width:.6f} {2 * half / shot.height:.6f}"
            )
            counts["boxes"] += 1
        # An empty file is a valid YOLO label: this frame has no ball in it, and
        # those are the negatives that stop the model re-learning the graphic.
        (texts / f"{stem}.txt").write_text("\n".join(lines))
        counts[split] += 1

    yaml = out_dir / "data.yaml"
    yaml.write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: football\n"
    )
    log(f"  train {counts['train']} images, val {counts['val']} "
        f"({', '.join(sorted(val_clips)) or 'none held out'}), "
        f"{counts['boxes']} boxes")
    if counts["val"] == 0:
        log("  no validation split — label a second clip before trusting a score")
    return yaml


def train(
    data_yaml: Path,
    base_weights: str = "yolo11n.pt",
    epochs: int = 60,
    imgsz: int = 1280,
    out_dir: Path = WEIGHTS_DIR,
    log=print,
) -> Path:
    """Fine-tune a detector on the labelled footballs.

    ``yolo11n`` rather than the ``yolo11x`` used for players: one class, a few
    hundred images, and a small model fine-tunes faster and overfits a small set
    less. Swap it if the measurement says otherwise — and measure, because a
    detection rate on this task has twice looked like progress and been furniture.
    """
    from ultralytics import YOLO

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    model = YOLO(base_weights)
    model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz,
                project=str(out_dir), name="football", exist_ok=True,
                patience=15, verbose=True)
    best = out_dir / "football" / "weights" / "best.pt"
    log(f"  weights: {best}")
    log("  point a clip at it with  track.ball_weights  in its spec")
    return best
