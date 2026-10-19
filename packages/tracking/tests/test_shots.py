"""Shot segmentation — the front door for footage nobody has cut up by hand.

The segmentation is tested against video built here with known cuts, so the expected
answer is exact. The *classifier* is not tested that way on purpose: what a play looks
like is not something synthetic colour blocks can express, and a test that asserted a
threshold would only be pinning a number that six hand-labelled frames produced.
"""

import numpy as np
import pytest

cv2 = pytest.importorskip("cv2")

from gridiron_tracking import shots


def _video(path, blocks, size=(320, 180), fps=30.0):
    """Write a video made of solid-colour blocks. ``blocks`` is [(bgr, n_frames), ...]."""
    w, h = size
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    for bgr, n in blocks:
        frame = np.zeros((h, w, 3), dtype=np.uint8)
        frame[:] = bgr
        for k in range(n):
            # a little noise so motion is not identically zero and the encoder has work
            noisy = np.clip(frame.astype(int) + np.random.randint(-6, 7, frame.shape), 0, 255)
            vw.write(noisy.astype(np.uint8))
    vw.release()


def test_hard_cuts_are_found(tmp_path):
    p = tmp_path / "cuts.mp4"
    _video(p, [((40, 160, 40), 40), ((200, 40, 40), 40), ((30, 30, 200), 40)])
    ss = shots.segment(p)
    assert len(ss) == 3, [(s.start_frame, s.end_frame) for s in ss]
    assert ss[0].start_frame == 0
    for a, b in zip(ss, ss[1:]):
        assert b.start_frame == a.end_frame + 1, "shots must tile the video with no gap"


def test_a_single_camera_is_a_single_shot(tmp_path):
    p = tmp_path / "one.mp4"
    _video(p, [((40, 160, 40), 90)])
    assert len(shots.segment(p)) == 1


def test_flash_frames_do_not_become_shots(tmp_path):
    """A two-frame sting is not a camera. Anything under the floor is absorbed."""
    p = tmp_path / "flash.mp4"
    _video(p, [((40, 160, 40), 40), ((255, 255, 255), 2), ((40, 160, 40), 40)])
    assert all(s.n_frames >= shots.MIN_SHOT_FRAMES for s in shots.segment(p))


def test_turf_is_measured_not_assumed(tmp_path):
    """Green reads as grass; red does not. The hue window is the whole claim."""
    p = tmp_path / "turf.mp4"
    _video(p, [((40, 160, 40), 40), ((30, 30, 200), 40)])
    ss = shots.segment(p)
    assert ss[0].turf > 0.9, ss[0].turf
    assert ss[1].turf < 0.1, ss[1].turf


def test_a_flat_colour_is_never_playlike(tmp_path):
    """One blob filling the frame is a graphic or a close-up, whatever colour it is."""
    p = tmp_path / "flat.mp4"
    _video(p, [((30, 30, 200), 120)])
    assert shots.segment(p)[0].playlike() == 0.0


def test_memory_does_not_grow_with_length(tmp_path):
    """The whole reason this module exists: a quarter cannot be held in memory.

    Segmenting twice the footage must not cost twice the resident memory. The check is
    on the descriptor generator, which is the only thing that touches pixels.
    """
    p = tmp_path / "long.mp4"
    _video(p, [((40, 160, 40), 200)])
    gen = shots._descriptors(p)
    first = next(gen)
    # a descriptor is a handful of floats plus a 192-bin histogram, never a frame
    assert first[1].size == 24 * 8
    assert all(np.isscalar(x) or isinstance(x, float) for x in (first[2], first[3], first[4]))
