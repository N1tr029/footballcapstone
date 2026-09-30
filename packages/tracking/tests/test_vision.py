"""The vision-assisted registration: schema, prompt shape, and the CV half.

No API calls here. What is testable without one is the contract between the model's
answer and the geometry — which is where the bugs were.
"""

import numpy as np
import pytest

from gridiron_tracking import field as F
from gridiron_tracking.registration import vision as V
from gridiron_tracking.registration.homography import Homography

pytestmark = pytest.mark.skipif(not V.HAVE_PYDANTIC, reason="needs pydantic")


def test_the_schema_always_offers_a_way_to_say_i_cannot_tell():
    """A field with no honest 'unknown' value is a field that gets filled dishonestly."""
    r = V.FieldReading.model_fields
    assert "unclear" in str(r["level"].annotation)
    assert "unclear" in str(r["numbers_increase_toward"].annotation)
    assert "none_visible" in str(r["end_zone_at"].annotation)
    assert "not_visible" in str(r["far_sideline_at"].annotation)
    assert r["down_marker_number"].default is None


def test_the_prompt_asks_for_structure_before_numbers():
    """Tested on a real night game, not one painted number was legible while the
    sidelines and end zone were obvious. Structure has to come first."""
    p = V.READ_PROMPT
    assert p.index("structural") < p.index("painted yard numbers")
    assert "empty list is the right answer" in p
    assert "down marker" in p


def test_the_system_prompt_tells_it_not_to_invent_precision():
    assert "Do not try to be precise" in V.SYSTEM
    assert "cannot tell" in V.SYSTEM


def test_a_painted_number_needs_to_know_which_half_it_is_in():
    """Every number but the 50 appears twice. Guessing the half is a 60-yard error."""
    assert V.field_x_for(30, "left_half", "unclear") == 40.0
    assert V.field_x_for(30, "right_half", "unclear") == 80.0
    assert V.field_x_for(50, "unclear", "unclear") == 60.0
    assert V.field_x_for(30, "unclear", "unclear") is None


def test_level_sets_the_hash_spacing():
    assert V.spec_for("nfl").hash_inset == F.NFL.hash_inset
    assert V.spec_for("college").hash_inset == F.NCAA.hash_inset
    assert V.spec_for("high_school").hash_inset == F.HIGH_SCHOOL.hash_inset


def _fit(pairs):
    fld = np.array([p[0] for p in pairs], float)
    px = np.array([p[1] for p in pairs], float)
    return Homography.fit(fld, px, threshold_px=25.0)


def test_a_fit_from_too_few_yard_lines_is_rejected():
    """The bug this catches: points on two yard lines fitted a homography that passed
    every camera check while being a hundred yards wrong down the field."""
    pairs = [((60.0, y), (900.0 + y * 2, 300.0 + y * 8)) for y in (0.0, 20.0, 53.3)]
    pairs += [((110.0, y), (1500.0 + y * 2, 320.0 + y * 8)) for y in (0.0, 53.3)]
    probs = V.sanity_problems(_fit(pairs), (1920, 1080), F.NFL, correspondences=pairs)
    assert any("yard line" in p for p in probs), probs


def test_a_well_spread_fit_is_not_flagged():
    from gridiron_tracking.detect.synthetic import sideline_camera
    cam = sideline_camera(image_size=(1920, 1080), height=30.0, y=-40.0)
    pairs = []
    for fx in (30.0, 60.0, 90.0):
        for fy in (0.0, 26.65, 53.3):
            px = cam.project([[fx, fy, 0.0]])[0]
            if np.isfinite(px).all():
                pairs.append(((fx, fy), (float(px[0]), float(px[1]))))
    assert len(pairs) >= 6
    assert V.sanity_problems(_fit(pairs), (1920, 1080), F.NFL, correspondences=pairs) == []


def test_a_folded_field_is_caught():
    pairs = [((0.0, 0.0), (100.0, 100.0)), ((120.0, 0.0), (900.0, 120.0)),
             ((120.0, 53.3), (150.0, 400.0)), ((0.0, 53.3), (880.0, 420.0))]
    probs = V.sanity_problems(_fit(pairs), (1920, 1080), F.NFL, correspondences=pairs)
    assert probs, "a self-crossing field should not pass"


def test_reading_with_nothing_readable_yields_no_correspondences():
    """The honest empty answer must produce an honest refusal, not a guess."""
    img = np.zeros((720, 1280, 3), np.uint8)
    r = V.FieldReading(
        landmarks=[], down_marker_number=None, yard_numbers=[],
        numbers_increase_toward="unclear", end_zone_at="none_visible",
        far_sideline_at="not_visible", near_sideline_visible=False,
        camera_view="unclear", level="unclear", team_colors=[], notes="unreadable",
    )
    pairs, _ = V.to_correspondences(r, img, F.NFL, refine=False)
    assert pairs == []


def test_guessed_landmarks_are_dropped():
    img = np.zeros((720, 1280, 3), np.uint8)
    r = V.FieldReading(
        landmarks=[V.Landmark(kind="far_sideline", p1=[0.1, 0.4], p2=[0.9, 0.38], confidence="guess")],
        down_marker_number=None, yard_numbers=[], numbers_increase_toward="unclear",
        end_zone_at="none_visible", far_sideline_at="top", near_sideline_visible=False,
        camera_view="unclear", level="nfl", team_colors=[], notes="",
    )
    pairs, notes = V.to_correspondences(r, img, F.NFL, refine=False)
    assert pairs == []
    assert any("guessed" in n for n in notes)
