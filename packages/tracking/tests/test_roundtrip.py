"""The whole back half, end to end, against tracking that is known to be true."""

import numpy as np
import pytest

from gridiron_tracking import bench
from gridiron_tracking.contract import PlayTrack, PlayerFrame, RosterEntry, TrackingFrame, validate
from gridiron_tracking.detect.synthetic import NoiseModel


def _play(n=40, players=12, seed=1):
    rng = np.random.default_rng(seed)
    roster = {f"p{i}": RosterEntry(id=f"p{i}", team="offense" if i < 6 else "defense")
              for i in range(players)}
    start = np.column_stack([
        rng.uniform(55, 65, players), rng.uniform(12, 42, players),
    ])
    vel = rng.uniform(-3.5, 3.5, size=(players, 2))
    frames = []
    for k in range(n):
        t = round(-1.0 + 0.1 * k, 3)
        pf = {}
        for i in range(players):
            p = start[i] + vel[i] * max(t, 0.0)
            pf[f"p{i}"] = PlayerFrame(x=float(p[0]), y=float(p[1]), s=float(np.hypot(*vel[i])))
        frames.append(TrackingFrame(t=t, players=pf, events=["ball_snap"] if k == 10 else []))
    return PlayTrack(meta={"play_id": "unit"}, roster=roster, frames=frames)


CLEAN = NoiseModel(dropout=0.0, crowd_dropout=0.0, false_positives_per_frame=0.0, box_jitter_px=0.0)


def test_a_clean_round_trip_lands_well_inside_the_bar():
    rt = bench.synthetic_roundtrip(_play(), fps=30.0, click_noise_px=0.0, noise=CLEAN)
    assert rt.registration_rms_yards < 1e-6
    assert rt.score.median_yards < 0.25
    assert rt.score.coverage > 0.95
    assert rt.score.meets()


def test_the_output_satisfies_the_renderer_contract():
    rt = bench.synthetic_roundtrip(_play(), fps=30.0, click_noise_px=0.0, noise=CLEAN)
    v = validate(rt.result.track)
    # The vision pipeline does not produce a ball track yet, so that is the one
    # complaint expected here; anything else is a real defect.
    assert v.problems in ([], ["no ball track"]), v.problems
    assert v.events.get("ball_snap") == 1


def test_helmet_boxes_are_corrected_for_height_not_left_yards_out():
    helmet = bench.synthetic_roundtrip(_play(), fps=30.0, click_noise_px=0.0,
                                       noise=CLEAN, box_kind="helmet")
    # Without the height correction this lands 5+ yards out; with it, under a yard.
    assert helmet.score.median_yards < 1.0


def test_operator_click_error_is_the_dominant_term():
    clean = bench.synthetic_roundtrip(_play(), fps=30.0, click_noise_px=0.0, noise=CLEAN)
    sloppy = bench.synthetic_roundtrip(_play(), fps=30.0, click_noise_px=3.0, noise=CLEAN)
    assert sloppy.score.median_yards > clean.score.median_yards
    assert sloppy.registration_rms_yards > clean.registration_rms_yards


def test_badly_spread_clicks_are_warned_about_even_when_they_fit_perfectly():
    from gridiron_tracking.registration import PointClickRegistration

    cam = bench.frame_the_play(_play())
    anchor = bench.clicked_anchor(
        cam, landmark_ids=["y45A/sideline_y0", "y50/sideline_y0",
                           "y45A/hash_y0", "y50/hash_y0"],
        click_noise_px=0.0,
    )
    reg = PointClickRegistration([anchor], image_size=(1280, 720))
    q = reg.assess()
    assert q.rms_yards < 1e-6, "these clicks fit perfectly"
    assert q.warnings, "and are still a bad registration, which is the point"
    assert any("span" in w for w in q.warnings)
