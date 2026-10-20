"""``pin``: calibrator clicks in, a checked registration out.

The point of the command is that the verdict arrives before the twenty minutes of
detection rather than after, and that a fit below the paint bar is not written at
all. Both of those are what this tests; the fitting itself is Homography.fit's.
"""

import json

import numpy as np
import pytest

from tracking_support import REPO
from gridiron_tracking import __main__ as cli
from gridiron_tracking.clips import ClipSpec


def points_from(h, field_pts):
    """Project field points through a known homography to make honest clicks."""
    out = []
    for x, y in field_pts:
        v = h @ np.array([x, y, 1.0])
        out.append({"field": [x, y], "pixel": [float(v[0] / v[2]), float(v[1] / v[2])]})
    return out


@pytest.fixture
def clip(tmp_path):
    """A spec pointing at a tiny synthetic video, in a writable spec dir."""
    cv2 = pytest.importorskip("cv2")
    video = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(video), cv2.VideoWriter_fourcc(*"mp4v"), 30.0, (320, 180))
    assert w.isOpened()
    for _ in range(12):
        w.write(np.full((180, 320, 3), 60, np.uint8))
    w.release()

    specs = tmp_path / "specs"
    specs.mkdir()
    spec = ClipSpec(name="synth", video=str(video), start=0, stop=10, snap_frame=4)
    spec.save(specs)
    return specs, spec


def test_too_few_points_is_refused(clip, tmp_path, capsys):
    specs, _ = clip
    pts = tmp_path / "p.json"
    pts.write_text(json.dumps({"points": [
        {"field": [10, 10], "pixel": [1, 2]},
        {"field": [20, 10], "pixel": [3, 4]},
        {"field": [10, 20], "pixel": [5, 6]}]}))

    rc = cli.main(["--spec-dir", str(specs), "pin", "synth", str(pts)])
    assert rc == 1
    assert "at least" in capsys.readouterr().err


def test_two_points_naming_the_same_place_are_refused(clip, tmp_path, capsys):
    specs, _ = clip
    pts = tmp_path / "p.json"
    pts.write_text(json.dumps({"points": [
        {"field": [10, 10], "pixel": [1, 2]},
        {"field": [10, 10], "pixel": [9, 9]},
        {"field": [20, 10], "pixel": [3, 4]},
        {"field": [10, 20], "pixel": [5, 6]}]}))

    rc = cli.main(["--spec-dir", str(specs), "pin", "synth", str(pts)])
    assert rc == 1
    assert "same place" in capsys.readouterr().err


def test_a_fit_below_the_paint_bar_is_not_written(clip, tmp_path, capsys):
    """A blank green frame has no paint, so the check cannot be computed at all.

    It comes back NaN, and every comparison against NaN is False — so testing
    `score < bar` alone wrote the worst case straight into the spec. Not knowing is
    not the same as passing, and this is the second time that exact trap has turned
    up in this repository.
    """
    specs, _ = clip
    h = np.array([[2.0, 0.0, 10.0], [0.0, 2.0, 5.0], [0.0, 0.0, 1.0]])
    pts = tmp_path / "p.json"
    pts.write_text(json.dumps({"points": points_from(
        h, [(10, 5), (60, 5), (10, 45), (60, 45), (35, 25), (90, 30)])}))

    rc = cli.main(["--spec-dir", str(specs), "pin", "synth", str(pts), "--frame", "4"])
    assert rc == 1
    err = capsys.readouterr().err
    assert "no more paint than chance" in err or "could not be computed" in err
    assert ClipSpec.load("synth", specs).register.mode == "auto", "it wrote anyway"


def test_force_writes_it_and_says_what_it_scored(clip, tmp_path, capsys):
    specs, _ = clip
    h = np.array([[2.0, 0.0, 10.0], [0.0, 2.0, 5.0], [0.0, 0.0, 1.0]])
    pts = tmp_path / "p.json"
    pts.write_text(json.dumps({"points": points_from(
        h, [(10, 5), (60, 5), (10, 45), (60, 45), (35, 25), (90, 30)])}))

    rc = cli.main(["--spec-dir", str(specs), "pin", "synth", str(pts),
                   "--frame", "4", "--force"])
    assert rc == 0
    out = capsys.readouterr().out
    assert "paint alignment" in out

    saved = ClipSpec.load("synth", specs)
    assert saved.register.mode == "pinned"
    assert saved.register.frame == 4
    fitted = np.array(saved.register.homography, dtype=float)
    # Recovered up to scale: a homography is only defined projectively.
    assert np.allclose(fitted / fitted[2, 2], h / h[2, 2], atol=1e-6)


def test_calframe_refuses_a_frame_outside_the_window(clip, capsys):
    """The registration has to come from a frame that actually gets decoded."""
    specs, _ = clip
    rc = cli.main(["--spec-dir", str(specs), "calframe", "synth", "--frame", "500"])
    assert rc == 1
    assert "outside" in capsys.readouterr().err


def test_calframe_writes_a_frame_and_says_what_to_do_next(clip, tmp_path, capsys):
    specs, _ = clip
    out = tmp_path / "cal.jpg"
    rc = cli.main(["--spec-dir", str(specs), "calframe", "synth",
                   "--frame", "4", "--out", str(out)])
    assert rc == 0
    assert out.exists()
    printed = capsys.readouterr().out
    assert "calibrate.html" in printed
    assert "pin synth" in printed
