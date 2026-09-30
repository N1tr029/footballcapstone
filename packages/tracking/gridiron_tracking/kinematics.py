"""From a path to the columns a tracking release publishes.

``s``, ``a``, ``dis`` and ``dir`` are all derivatives of position, which means they
amplify whatever noise the position has. A tenth of a yard of jitter at 10 Hz is one
yard per second of speed error, and differencing that again for acceleration turns
it into ten. Differencing raw positions therefore produces an ``a`` column that is
almost entirely noise, and it will look plausible while being worthless.

So: smooth first, on the position, with a filter that preserves the shape of a cut —
Savitzky-Golay fits a low-order polynomial over a short window, which passes a
receiver's break through largely intact where a moving average would round it off.
Then difference the smoothed path, centrally, so nothing ends up half a frame late.

Resampling happens here too, because video is not 10 Hz and the contract is. It is
strictly interpolation *within* each track's observed span — a track that vanished
into the pile for a second is left absent rather than bridged, because inventing
frames is how a quality report starts lying about coverage.
"""

from __future__ import annotations

import numpy as np
from scipy.signal import savgol_filter

from .contract import wrap_deg


def resample(times: np.ndarray, xy: np.ndarray, out_times: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Linear interpolation onto ``out_times``, clipped to the observed span.

    Returns the output times that fall inside the track and the positions there.
    """
    times = np.asarray(times, dtype=float)
    xy = np.asarray(xy, dtype=float)
    if len(times) < 2:
        return np.empty(0), np.empty((0, 2))

    inside = (out_times >= times[0]) & (out_times <= times[-1])
    t_out = out_times[inside]
    if not len(t_out):
        return np.empty(0), np.empty((0, 2))

    x = np.interp(t_out, times, xy[:, 0])
    y = np.interp(t_out, times, xy[:, 1])
    return t_out, np.column_stack([x, y])


def smooth(xy: np.ndarray, window: int = 7, poly: int = 2) -> np.ndarray:
    """Savitzky-Golay over the path, degrading gracefully on short tracks."""
    n = len(xy)
    if n < 5:
        return np.asarray(xy, dtype=float)
    w = min(window, n if n % 2 == 1 else n - 1)
    if w <= poly:
        return np.asarray(xy, dtype=float)
    return np.column_stack([
        savgol_filter(xy[:, 0], w, poly),
        savgol_filter(xy[:, 1], w, poly),
    ])


def derive(t: np.ndarray, xy: np.ndarray) -> dict[str, np.ndarray]:
    """Speed, acceleration, per-frame distance and direction of travel.

    ``dir`` is held through near-stillness rather than recomputed: the direction of
    a velocity vector with almost no magnitude is numerically meaningless, and a
    lineman set in his stance would otherwise spin on the spot. This is the same
    rule the renderer's adapter already applies, deliberately — two different
    answers to "which way is he going" in two parts of one system is how the
    quarterback ended up watching grass.
    """
    t = np.asarray(t, dtype=float)
    xy = np.asarray(xy, dtype=float)
    n = len(t)
    if n == 0:
        empty = np.empty(0)
        return {"s": empty, "a": empty, "dis": empty, "dir": empty}
    if n == 1:
        z = np.zeros(1)
        return {"s": z, "a": z, "dis": z, "dir": z}

    # Central differences on the interior, one-sided at the ends.
    vx = np.gradient(xy[:, 0], t, edge_order=1)
    vy = np.gradient(xy[:, 1], t, edge_order=1)
    s = np.hypot(vx, vy)
    a = np.gradient(s, t, edge_order=1)

    step = np.zeros(n)
    step[1:] = np.hypot(np.diff(xy[:, 0]), np.diff(xy[:, 1]))

    SLOW = 0.6
    direction = np.zeros(n)
    last = None
    for i in range(n):
        if s[i] > SLOW:
            last = wrap_deg(np.degrees(np.arctan2(vy[i], vx[i])))
        direction[i] = last if last is not None else 0.0

    # A track that never exceeded walking pace has no meaningful direction at all;
    # point it the way its net displacement went rather than at zero, which would
    # silently mean "downfield".
    if last is None:
        d = xy[-1] - xy[0]
        if np.hypot(*d) > 1e-6:
            direction[:] = wrap_deg(np.degrees(np.arctan2(d[1], d[0])))

    return {"s": s, "a": np.abs(a), "dis": step, "dir": direction}
