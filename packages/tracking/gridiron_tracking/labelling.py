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
TILED_DIR = Path("data/labels/dataset_tiled")
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

    def without_stale_repeats(self) -> tuple["LabelSet", dict[str, int]]:
        """Drop labels that repeat the previous labelled frame's coordinates exactly.

        The UI keeps the last click on screen while the frame advances, so pressing
        next without clicking again saves the same point for a frame the ball has
        already left. A ball at the *identical* subpixel coordinate two frames apart
        is that, not a measurement: a real ball either moves or is occluded, and even
        a stationary one jitters by a pixel under a hand-placed cursor.

        Keeping the first of each run keeps the frame the click was actually made on
        and drops the ones it drifted across. Measured on this set it removes 20 of
        baker_kickoff's 26, which were a ball in flight recorded at one fixed point —
        a box of grass a ball-width behind the ball, taught as the ball.

        Not-visible labels are negatives and are never repeats.
        """
        kept: dict[str, Label] = {}
        dropped: dict[str, int] = {}
        by_clip: dict[str, list[tuple[int, str, Label]]] = {}
        for key, label in self.labels.items():
            clip, _, frame = key.rpartition("/")
            by_clip.setdefault(clip, []).append((int(frame), key, label))
        for clip, rows in by_clip.items():
            last: tuple[float | None, float | None] | None = None
            for _, key, label in sorted(rows):
                if label.visible:
                    here = (label.x, label.y)
                    if here == last:
                        dropped[clip] = dropped.get(clip, 0) + 1
                        continue
                    last = here
                else:
                    last = None
                kept[key] = label
        return LabelSet(kept), dropped


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
    holdout: list[str] | None = None,
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
    if holdout:
        val_clips = {c for c in holdout if c in clips}
        missing = set(holdout) - val_clips
        if missing:
            raise ValueError(f"nothing labelled for {sorted(missing)}")
    else:
        n_val = max(1, round(len(clips) * val_fraction)) if len(clips) > 1 else 0
        val_clips = set(clips[-n_val:]) if n_val else set()

    # A held-out clip with almost no ball in it cannot measure anything: a model
    # that predicts nothing at all scores well on it. Say so rather than reporting
    # a number that means nothing.
    positives = {c: sum(1 for s, l in usable if s.clip == c and l.visible) for c in clips}
    thin = {c: positives[c] for c in val_clips if positives[c] < 5}
    if thin:
        log(f"  WARNING: held-out {thin} — too few balls to measure against. "
            f"Pass holdout= to choose a clip with more.")

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


def to_yolo_tiled(
    shots: list[Shot],
    labels: LabelSet,
    out_dir: Path = TILED_DIR,
    tile: int = 640,
    holdout: list[str] | None = None,
    crops_per_ball: int = 4,
    negatives_per_frame: int = 2,
    seed: int = 0,
    log=print,
) -> Path:
    """Write a YOLO dataset of native-resolution crops instead of whole frames.

    **Why this exists.** Training on whole frames resized to the model's input is
    what the first fine-tune did, and it is a scale mismatch with how the ball is
    actually looked for. A 1920x1080 frame at ``imgsz 960`` halves everything, so a
    twelve-pixel ball arrives as six pixels — at or under the floor of a stride-8
    detection head. Inference does the opposite: ``detect/yolo.py`` cuts the frame
    into tiles and *magnifies* each one, so the ball it is asked about is two to
    three times larger than any ball the model was trained on. It scored
    mAP50 0.107 at the size it was trained at and 0.001 at the size it is used at,
    which is that mismatch and not a fact about footballs.

    Cropping at native resolution removes the resize from training: the ball is the
    same handful of pixels it is in the source, and a 640 tile shown to a 1280-input
    model is magnified the way a tile is at inference.

    Training crops are centred near the ball with jitter, so it is seen against
    different backgrounds and at different offsets rather than always in the middle.
    Held-out clips are cut on a fixed overlapping grid instead — that is the sweep
    inference performs, and a score over it is a score over the real task.

    A tile holding only *part* of the ball is written for neither: it is not a
    negative, and teaching a clipped box is how a detector learns half a ball. The
    grid overlaps by half a tile so every ball falls whole inside at least one.
    """
    import random
    import shutil

    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise ImportError("tiling needs opencv") from e

    out_dir = Path(out_dir)
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    by_key = {s.key: s for s in shots}
    usable = [(by_key[k], v) for k, v in labels.labels.items() if k in by_key]
    if not usable:
        raise ValueError("nothing labelled yet")

    clips = sorted({s.clip for s, _ in usable})
    val_clips = set(holdout or []) & set(clips)
    if holdout and not val_clips:
        raise ValueError(f"nothing labelled for {sorted(holdout)}")
    positives = {c: sum(1 for s, l in usable if s.clip == c and l.visible) for c in clips}
    thin = {c: positives[c] for c in val_clips if positives[c] < 5}
    if thin:
        log(f"  WARNING: held-out {thin} — too few balls to measure against.")

    rng = random.Random(seed)
    counts = {"train": 0, "val": 0, "train_boxes": 0, "val_boxes": 0,
              "skipped_partial": 0, "unreadable": 0}

    def write(split: str, stem: str, img, box: tuple[float, float, float] | None) -> None:
        images = out_dir / "images" / split
        texts = out_dir / "labels" / split
        images.mkdir(parents=True, exist_ok=True)
        texts.mkdir(parents=True, exist_ok=True)
        cv2.imwrite(str(images / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 95])
        lines = []
        if box is not None:
            bx, by, bs = box
            h, w = img.shape[:2]
            lines.append(f"0 {bx / w:.6f} {by / h:.6f} {bs / w:.6f} {bs / h:.6f}")
            counts[f"{split}_boxes"] += 1
        (texts / f"{stem}.txt").write_text("\n".join(lines) + "\n")
        counts[split] += 1

    def crop(img, ox: int, oy: int, tw: int, th: int):
        return img[oy:oy + th, ox:ox + tw]

    for shot, label in sorted(usable, key=lambda sl: sl[0].key):
        split = "val" if shot.clip in val_clips else "train"
        img = cv2.imread(shot.path)
        if img is None:
            counts["unreadable"] += 1
            continue
        h, w = img.shape[:2]
        # A frame smaller than the tile becomes one tile of itself rather than
        # nothing. A 640x360 high-school clip is shorter than a 640 tile in one
        # axis only, and skipping it would drop the whole clip silently.
        tw, th = min(tile, w), min(tile, h)
        half = label.size / 2.0
        stem = f"{shot.clip}_{shot.frame:05d}"

        if split == "val":
            # The sweep inference performs: a fixed grid, overlapping by half a tile
            # so a ball on a seam still lands whole inside a neighbour.
            xs = sorted({min(x, w - tw) for x in range(0, w, max(1, tw // 2))})
            ys = sorted({min(y, h - th) for y in range(0, h, max(1, th // 2))})
            for ox in xs:
                for oy in ys:
                    box = None
                    if label.visible:
                        inside_whole = (ox + half <= label.x <= ox + tw - half
                                        and oy + half <= label.y <= oy + th - half)
                        touches = (ox - half < label.x < ox + tw + half
                                   and oy - half < label.y < oy + th + half)
                        if inside_whole:
                            box = (label.x - ox, label.y - oy, label.size)
                        elif touches:
                            counts["skipped_partial"] += 1
                            continue
                    write("val", f"{stem}_{ox}_{oy}", crop(img, ox, oy, tw, th), box)
            continue

        if label.visible:
            for n in range(crops_per_ball):
                lo_x = max(0, int(label.x + half - tw))
                hi_x = min(w - tw, int(label.x - half))
                lo_y = max(0, int(label.y + half - th))
                hi_y = min(h - th, int(label.y - half))
                if lo_x > hi_x or lo_y > hi_y:
                    break
                ox = rng.randint(lo_x, hi_x)
                oy = rng.randint(lo_y, hi_y)
                write("train", f"{stem}_p{n}", crop(img, ox, oy, tw, th),
                      (label.x - ox, label.y - oy, label.size))

        for n in range(negatives_per_frame):
            for _ in range(20):
                ox = rng.randint(0, w - tw)
                oy = rng.randint(0, h - th)
                if label.visible and (ox - half < label.x < ox + tw + half
                                      and oy - half < label.y < oy + th + half):
                    continue
                write("train", f"{stem}_n{n}", crop(img, ox, oy, tw, th), None)
                break

    if counts["unreadable"]:
        log(f"  WARNING: {counts['unreadable']} frames could not be read as images "
            f"and produced no tiles")
    if not counts["train"] and not counts["val"]:
        raise ValueError(
            f"no tiles written from {len(usable)} labelled frames "
            f"({counts['unreadable']} unreadable). Tiling reads the pixels, unlike "
            f"the whole-frame export, which only copies the files."
        )

    yaml = out_dir / "data.yaml"
    yaml.write_text(
        f"path: {out_dir.resolve()}\n"
        f"train: images/train\n"
        f"val: images/val\n"
        f"names:\n  0: football\n"
    )
    log(f"  tiles {tile}px — train {counts['train']} ({counts['train_boxes']} boxes), "
        f"val {counts['val']} ({counts['val_boxes']} boxes"
        f"{', ' + ', '.join(sorted(val_clips)) if val_clips else ', none held out'}), "
        f"{counts['skipped_partial']} partial tiles skipped")
    return yaml


def train(
    data_yaml: Path,
    base_weights: str = "yolo11n.pt",
    epochs: int = 60,
    imgsz: int = 640,
    out_dir: Path = WEIGHTS_DIR,
    batch: int = 16,
    device: str | None = None,
    small_object: bool = True,
    log=print,
) -> Path:
    """Fine-tune a detector on the labelled footballs.

    ``yolo11n`` rather than the ``yolo11x`` used for players: one class, a few
    hundred images, and a small model fine-tunes faster and overfits a small set
    less. Swap it if the measurement says otherwise — and measure, because a
    detection rate on this task has twice looked like progress and been furniture.
    """
    from ultralytics import YOLO

    # Resolved, because ultralytics treats a *relative* project path as relative to
    # its own runs directory rather than to the caller: the first fine-tune asked
    # for data/labels/weights and the weights landed in
    # runs/detect/data/labels/weights, so the path this returns — and the one the
    # README tells you to point a clip at — did not exist.
    out_dir = Path(out_dir).resolve()
    out_dir.mkdir(parents=True, exist_ok=True)
    # Mosaic stitches four images into one and downscales it. On whole frames that
    # is ordinary augmentation; on tiles cropped specifically so the ball arrives
    # at its real size it undoes the crop, and the ball goes back to the six pixels
    # that made the first fine-tune fail. The same goes for wide scale jitter.
    # Measured: with mosaic on, classification loss sat at 147 falling to 10 and
    # mAP50 never left 0.005; with it off, loss starts near 1.5 and the model
    # learns. Everything here serves one thing -- the ball reaches the model the
    # size the source has it.
    aug = dict(mosaic=0.0, mixup=0.0, copy_paste=0.0, scale=0.2, erasing=0.0) \
        if small_object else {}

    model = YOLO(base_weights)
    model.train(data=str(data_yaml), epochs=epochs, imgsz=imgsz, batch=batch,
                project=str(out_dir), name="football", exist_ok=True,
                patience=15, verbose=True, **({"device": device} if device else {}),
                **aug)
    best = out_dir / "football" / "weights" / "best.pt"
    log(f"  weights: {best}")
    log("  point a clip at it with  track.ball_weights  in its spec")
    return best
