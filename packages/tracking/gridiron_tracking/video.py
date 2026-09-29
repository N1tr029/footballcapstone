"""Reading frames, and the one step a human has to do.

Two jobs. Decoding is the boring one. The interesting one is
:func:`calibration_sheet`: a frame written out with a labelled pixel grid over it,
so whoever is registering the video can read the coordinates of a yard-line
intersection straight off the image and type them in. That is the whole of the
operator-assisted registration workflow — no GUI, no click handler, one PNG and six
numbers — and it is what makes this runnable on any field on the first try.

OpenCV is imported lazily. Everything else in this package works without it, and a
teammate who only wants the contract or the benchmark should not need a CV stack.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

import numpy as np


def _cv2():
    try:
        import cv2
    except ImportError as e:  # pragma: no cover - depends on the install
        raise ImportError(
            "reading video needs opencv: pip install 'gridiron-tracking[video]' "
            "(or pip install opencv-python)"
        ) from e
    return cv2


@dataclass
class VideoInfo:
    path: str
    fps: float
    width: int
    height: int
    n_frames: int

    @property
    def size(self) -> tuple[int, int]:
        return (self.width, self.height)

    @property
    def duration(self) -> float:
        return self.n_frames / self.fps if self.fps else 0.0

    def describe(self) -> str:
        return (
            f"{Path(self.path).name}: {self.width}x{self.height}, "
            f"{self.fps:.2f} fps, {self.n_frames} frames ({self.duration:.1f}s)"
        )


class VideoSource:
    """A video file, opened once and read by frame index."""

    def __init__(self, path: str | Path) -> None:
        cv2 = _cv2()
        self.path = str(path)
        if not Path(self.path).exists():
            raise FileNotFoundError(self.path)
        self._cap = cv2.VideoCapture(self.path)
        if not self._cap.isOpened():
            raise RuntimeError(f"could not open {self.path} — unsupported codec?")

        fps = float(self._cap.get(cv2.CAP_PROP_FPS)) or 0.0
        n = int(self._cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.info = VideoInfo(
            path=self.path,
            # A container with a broken fps field is common; 30 is the safest guess
            # and the caller is told so it can be overridden.
            fps=fps if 1.0 < fps < 1000.0 else 30.0,
            width=int(self._cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            height=int(self._cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            n_frames=n if n > 0 else 0,
        )
        self._pos = -1

    # ------------------------------------------------------------------ reading

    def frame(self, index: int) -> np.ndarray | None:
        """One frame as BGR. Sequential reads avoid a seek, which matters: seeking
        per frame on a long-GOP file is slower than decoding the whole thing."""
        cv2 = _cv2()
        if index != self._pos + 1:
            self._cap.set(cv2.CAP_PROP_POS_FRAMES, index)
        ok, img = self._cap.read()
        self._pos = index if ok else -1
        return img if ok else None

    def frames(self, start: int = 0, stop: int | None = None, step: int = 1) -> Iterator[tuple[int, np.ndarray]]:
        i = start
        stop = stop if stop is not None else (self.info.n_frames or 10 ** 9)
        while i < stop:
            img = self.frame(i)
            if img is None:
                return
            yield i, img
            if step > 1:
                for _ in range(step - 1):
                    if self.frame(self._pos + 1) is None:
                        return
            i += step

    def close(self) -> None:
        self._cap.release()

    def __enter__(self) -> "VideoSource":
        return self

    def __exit__(self, *exc) -> None:
        self.close()


# ------------------------------------------------------------------ calibration


def calibration_sheet(
    source: VideoSource | str | Path,
    out_path: str | Path,
    frame_index: int = 0,
    grid: int = 100,
) -> Path:
    """Write a frame with a labelled pixel grid over it.

    This is the registration UI. Open the PNG, find a point on the field you can
    name — the 50 where it meets the near sideline, the corner of the end zone, a
    yard line crossing a hash — read its pixel coordinates off the grid, and pass it
    as ``landmark=x,y``. Six of those, spread wide, and the video is registered.

    Spread matters more than precision: four points clicked close together fit
    perfectly and then put a receiver in the car park, because the fit has no
    evidence about the direction nobody marked. Take points from both sidelines and
    from opposite ends of the visible field.
    """
    cv2 = _cv2()
    src = source if isinstance(source, VideoSource) else VideoSource(source)
    img = src.frame(frame_index)
    if img is None:
        raise ValueError(f"no frame {frame_index} in {src.info.path}")

    canvas = img.copy()
    h, w = canvas.shape[:2]

    for x in range(0, w, grid):
        heavy = x % (grid * 5) == 0
        cv2.line(canvas, (x, 0), (x, h), (0, 255, 255) if heavy else (0, 160, 160), 2 if heavy else 1)
        if heavy:
            cv2.putText(canvas, str(x), (x + 4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(canvas, str(x), (x + 4, 22), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)
    for y in range(0, h, grid):
        heavy = y % (grid * 5) == 0
        cv2.line(canvas, (0, y), (w, y), (0, 255, 255) if heavy else (0, 160, 160), 2 if heavy else 1)
        if heavy:
            cv2.putText(canvas, str(y), (6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 0), 4)
            cv2.putText(canvas, str(y), (6, y - 6), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255), 1)

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out_path), canvas)
    if not isinstance(source, VideoSource):
        src.close()
    return out_path


def motion_energy(source: VideoSource, start: int = 0, stop: int | None = None, step: int = 1) -> np.ndarray:
    """Mean absolute frame difference, per frame.

    The snap is the sharpest rise in this signal: twenty-two men who were nearly
    still all start moving inside a tenth of a second, and nothing else in a play
    looks like that. Used by :func:`find_snap` when nobody says where the snap is.
    """
    cv2 = _cv2()
    prev = None
    out: list[float] = []
    for _, img in source.frames(start, stop, step):
        small = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (160, 90))
        if prev is not None:
            out.append(float(np.mean(cv2.absdiff(small, prev))))
        else:
            out.append(0.0)
        prev = small
    return np.asarray(out)


def find_snap(energy: np.ndarray, smooth: int = 5) -> int:
    """Index of the largest sustained jump in motion.

    A heuristic, and it fails on film that starts mid-play or holds a long pre-snap
    shot with the camera drifting. It reports an index, not a certainty — pass
    ``--snap-frame`` whenever you know it, because everything in the contract is
    timed from the snap and an offset clock makes every event wrong together.
    """
    if len(energy) < smooth * 2 + 2:
        return 0
    k = np.ones(smooth) / smooth
    sm = np.convolve(energy, k, mode="same")
    rise = np.diff(sm)
    return int(np.argmax(rise)) + 1
