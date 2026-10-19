"""Reprojection agreement — the check that needs no ground truth."""

import numpy as np
import pytest

from gridiron_tracking import verify as VER
from gridiron_tracking.contract import PlayTrack, PlayerFrame, RosterEntry, TrackingFrame
from gridiron_tracking.detect.synthetic import sideline_camera


class _Reg:
    def __init__(self, cam):
        self.cam = cam

    def at(self, i):
        return self.cam.homography


def _track(n=10, players=4, dx=0.0):
    roster = {f"p{i}": RosterEntry(id=f"p{i}", team="offense") for i in range(players)}
    frames = []
    for k in range(n):
        frames.append(TrackingFrame(
            t=round(k / 10.0, 3),
            players={f"p{i}": PlayerFrame(x=40.0 + 8 * i + dx, y=26.0) for i in range(players)},
        ))
    return PlayTrack(roster=roster, frames=frames)


def _boxes_from(track, cam, n=10, fps=10.0, offset=0):
    out = {}
    for f in track.frames:
        px = cam.project([[p.x, p.y, 0.0] for p in f.players.values()])
        out[int(round(f.t * fps)) + offset] = [
            [q[0] - 12, q[1] - 40, q[0] + 12, q[1]] for q in px if np.isfinite(q).all()
        ]
    return out


def test_a_perfect_recreation_agrees_completely():
    cam = sideline_camera(image_size=(1280, 720), height=30.0, y=-40.0)
    t = _track()
    rep = VER.verify(t, _Reg(cam), _boxes_from(t, cam), fps=10.0)
    assert rep.agreement == pytest.approx(1.0)
    assert all(f.stray == 0 for f in rep.frames)


def test_positions_shifted_off_the_players_are_caught():
    """Five yards of error puts a dot on grass, and the check must say so."""
    cam = sideline_camera(image_size=(1280, 720), height=30.0, y=-40.0)
    truth = _track()
    boxes = _boxes_from(truth, cam)
    rep = VER.verify(_track(dx=5.0), _Reg(cam), boxes, fps=10.0)
    assert rep.agreement < 0.5, rep.agreement
    assert np.mean([f.stray for f in rep.frames]) > 1


def test_people_nobody_is_tracking_are_reported():
    cam = sideline_camera(image_size=(1280, 720), height=30.0, y=-40.0)
    t = _track(players=2)
    boxes = _boxes_from(_track(players=6), cam)
    rep = VER.verify(t, _Reg(cam), boxes, fps=10.0)
    assert np.mean([f.unclaimed for f in rep.frames]) >= 3


def test_the_worst_player_is_identifiable():
    """The point of the per-player breakdown: knowing which one to look at."""
    cam = sideline_camera(image_size=(1280, 720), height=30.0, y=-40.0)
    truth = _track()
    boxes = _boxes_from(truth, cam)
    bad = _track()
    for f in bad.frames:
        f.players["p2"].x += 6.0
    rep = VER.verify(bad, _Reg(cam), boxes, fps=10.0)
    assert rep.worst_players(1)[0].player_id == "p2"


def test_side_by_side_writes_a_playable_file(tmp_path):
    """The validation artifact is worth a test of its own — it is the thing a person
    actually looks at, and a renderer that silently writes nothing is worse than one
    that fails."""
    import cv2
    from gridiron_tracking import field as F

    cam = sideline_camera(image_size=(320, 180), height=30.0, y=-40.0)
    t = _track(n=6, players=3)
    boxes = _boxes_from(t, cam, fps=10.0)
    frames = {int(round(f.t * 10.0)): np.zeros((180, 320, 3), np.uint8) for f in t.frames}
    out = VER.side_by_side(tmp_path / "sbs.mp4", t, _Reg(cam), frames, boxes,
                           spec=F.NCAA, fps=10.0, out_fps=6)
    assert out.exists() and out.stat().st_size > 0
    cap = cv2.VideoCapture(str(out))
    ok, img = cap.read()
    cap.release()
    assert ok, "the file must be readable"
    assert img.shape[0] == 180 * 2 + VER.PANEL_RULE_PX, \
        "both panels must be stacked, separated by the hairline rule"


def _two_tracks_on_one_body(n=10):
    """Two ids on one man — the defect the old scorer counted twice as a success."""
    roster = {"p0": RosterEntry(id="p0", team="offense"),
              "p0b": RosterEntry(id="p0b", team="defense")}
    return PlayTrack(roster=roster, frames=[
        TrackingFrame(t=round(k / 10.0, 3), players={
            "p0": PlayerFrame(x=40.0, y=26.0),
            "p0b": PlayerFrame(x=40.1, y=26.0),
        }) for k in range(n)
    ])


def test_two_tracks_on_one_body_are_one_hit_and_one_duplicate():
    """A box is one person, so it can be claimed once.

    This is the whole reason the assignment is exclusive. Scored the old way — every
    track taking whatever box contained it — this frame reads 2 of 2 agree, which is
    how the reference clip reported 92.1% while one man in three wore two tracks.
    """
    cam = sideline_camera(image_size=(1280, 720), height=30.0, y=-40.0)
    boxes = _boxes_from(_track(players=1), cam)
    rep = VER.verify(_two_tracks_on_one_body(), _Reg(cam), boxes, fps=10.0)

    assert all(f.on_player == 1 for f in rep.frames)
    assert all(f.duplicate == 1 for f in rep.frames)
    assert all(f.stray == 0 for f in rep.frames)
    assert rep.agreement == pytest.approx(0.5)


def test_a_clean_recreation_has_no_duplicates():
    cam = sideline_camera(image_size=(1280, 720), height=30.0, y=-40.0)
    t = _track()
    rep = VER.verify(t, _Reg(cam), _boxes_from(t, cam), fps=10.0)
    assert all(f.duplicate == 0 for f in rep.frames)
    assert all(f.unclaimed == 0 for f in rep.frames)


def test_the_duplicate_is_named_on_the_player_that_caused_it():
    cam = sideline_camera(image_size=(1280, 720), height=30.0, y=-40.0)
    boxes = _boxes_from(_track(players=1), cam)
    rep = VER.verify(_two_tracks_on_one_body(), _Reg(cam), boxes, fps=10.0)
    dupes = {p.player_id: p.duplicate for p in rep.players.values()}
    assert sum(dupes.values()) == 10          # one per frame
    assert sorted(dupes.values()) == [0, 10]  # and all of it on one of the two ids


def test_the_ring_shrinks_with_distance():
    """Ring size is geometry, not the width of whatever box it landed in."""
    cam = sideline_camera(image_size=(1280, 720), height=30.0, y=-40.0)

    def to_image(pts):
        return np.asarray(cam.project([[p[0], p[1], 0.0] for p in pts]))[:, :2]

    near = VER._player_radius(to_image, PlayerFrame(x=60.0, y=5.0))
    far = VER._player_radius(to_image, PlayerFrame(x=60.0, y=50.0))
    assert near > far, f"near {near} should be wider than far {far}"
