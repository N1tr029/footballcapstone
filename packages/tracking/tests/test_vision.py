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
    assert "unclear" in str(r["numbers_decrease_toward"].annotation)
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
        numbers_decrease_toward="unclear", end_zone_at="none_visible",
        far_sideline_at="not_visible", near_sideline_visible=False,
        camera_view="unclear", level="unclear", team_colors=[], notes="unreadable",
    )
    pairs, _ = V.to_correspondences(r, img, F.NFL, refine=False)
    assert pairs == []


def test_guessed_landmarks_are_dropped():
    img = np.zeros((720, 1280, 3), np.uint8)
    r = V.FieldReading(
        landmarks=[V.Landmark(kind="far_sideline", p1=[0.1, 0.4], p2=[0.9, 0.38], confidence="guess")],
        down_marker_number=None, yard_numbers=[], numbers_decrease_toward="unclear",
        end_zone_at="none_visible", far_sideline_at="top", near_sideline_visible=False,
        camera_view="unclear", level="nfl", team_colors=[], notes="",
    )
    pairs, notes = V.to_correspondences(r, img, F.NFL, refine=False)
    assert pairs == []
    assert any("guessed" in n for n in notes)


def test_turf_overlap_rejects_a_fit_that_left_the_field():
    """The check that replaced asking the model to grade itself.

    Shown a fit whose sidelines ran through the crowd, both gemini-3.5-flash and
    flash-lite answered "good, off by 0.0 yards". Green pixels do not.
    """
    import cv2
    from gridiron_tracking.detect.synthetic import sideline_camera

    # A frame that is turf in the lower half and crowd-dark in the upper half.
    img = np.zeros((720, 1280, 3), np.uint8)
    img[:, :] = (40, 40, 40)
    img[360:, :] = (40, 140, 60)          # BGR green

    cam = sideline_camera(image_size=(1280, 720), height=25.0, y=-35.0)
    good = cam.homography
    on_grass = V.turf_overlap(img, good, F.NFL)

    # The same fit pushed upward lands the field in the crowd.
    import copy
    drifted = copy.deepcopy(good)
    drifted.H = good.H.copy()
    drifted.H[1, 2] -= 260.0
    off_grass = V.turf_overlap(img, drifted, F.NFL)

    assert on_grass > off_grass, (on_grass, off_grass)
    assert off_grass < 0.75, "a field projected into the crowd must fail the bar"


# --------------------------------------------------------- retrying and failing over
# The failure that cost a whole clip was not a bad reading, it was a busy endpoint:
# nine of nine anchor frames on a 1080p NFL broadcast died to 503s and truncated
# answers. These pin the policy that fixed it, without making a call.


def _reader(monkeypatch, answers):
    """Drive _ask with a scripted sequence of per-model answers.

    ``answers`` maps a model name to a list of VisionResults to return in order;
    a model missing from it keeps returning its last answer.
    """
    seen = []

    def fake(client, img, prompt, schema, max_tokens, model, **extra):
        seen.append(model)
        queue = answers.get(model) or [V.VisionResult(None, "", "ClientError: 404 NOT_FOUND")]
        return queue.pop(0) if len(queue) > 1 else queue[0]

    monkeypatch.setattr(V, "_ask_once", fake)
    monkeypatch.setattr(V, "_nap", lambda s: None)
    return seen


def test_a_busy_model_is_asked_again_rather_than_given_up_on(monkeypatch):
    busy = V.VisionResult(None, "", "ServerError: 503 UNAVAILABLE high demand")
    good = V.VisionResult(object(), "{}")
    seen = _reader(monkeypatch, {"m": [busy, busy, good]})

    res = V._ask(None, None, "p", V.FieldReading, model="m", model_chain=())
    assert res.reading is not None
    assert seen == ["m", "m", "m"]


def test_a_model_that_stays_busy_falls_back_to_its_sibling(monkeypatch):
    busy = V.VisionResult(None, "", "ServerError: 503 UNAVAILABLE")
    good = V.VisionResult(object(), "{}")
    seen = _reader(monkeypatch, {"m": [busy], "sib": [good]})

    res = V._ask(None, None, "p", V.FieldReading, model="m", model_chain=("sib",))
    assert res.reading is not None
    assert seen == ["m", "m", "m", "sib"]


def test_a_permanent_failure_is_not_retried_on_the_same_model(monkeypatch):
    """A 404 will still be a 404 in two seconds. The sibling is still worth a try."""
    gone = V.VisionResult(None, "", "ClientError: 404 NOT_FOUND")
    good = V.VisionResult(object(), "{}")
    seen = _reader(monkeypatch, {"m": [gone], "sib": [good]})

    res = V._ask(None, None, "p", V.FieldReading, model="m", model_chain=("sib",))
    assert res.reading is not None
    assert seen == ["m", "sib"]


def test_when_everything_fails_the_last_real_diagnosis_survives(monkeypatch):
    """Not a generic 'registration failed' — the caller has to be able to tell a busy
    endpoint from a missing key from a frame with no field in it."""
    _reader(monkeypatch, {
        "m": [V.VisionResult(None, "", "ServerError: 503 UNAVAILABLE")],
        "sib": [V.VisionResult(None, "", "GEMINI_API_KEY is not set")],
    })
    res = V._ask(None, None, "p", V.FieldReading, model="m", model_chain=("sib",))
    assert res.reading is None
    assert "GEMINI_API_KEY is not set" in res.error


def test_transient_and_permanent_are_told_apart():
    assert V._is_transient("ServerError: 503 UNAVAILABLE")
    assert V._is_transient("429 rate limit exceeded")
    assert V._is_transient("this model is currently experiencing high demand")
    assert not V._is_transient("ClientError: 404 NOT_FOUND")
    assert not V._is_transient("GEMINI_API_KEY is not set")
    assert not V._is_transient("the model declined this request")


def test_running_out_of_output_budget_is_reported_as_itself():
    """It arrived as a JSONDecodeError, which reads like a prompt problem. It was a
    thinking model spending the answer's token budget on thinking."""
    class Resp:
        candidates = [type("C", (), {"finish_reason": "MAX_TOKENS"})()]

    assert V._hit_the_token_cap(Resp())
    assert not V._hit_the_token_cap(type("R", (), {"candidates": []})())


def test_every_backend_has_a_name_that_routes_to_it():
    assert V.backend_for(V.GEMINI_MODEL) == "gemini"
    assert V.backend_for(V.QWEN_MODEL) == "qwen"
    assert V.backend_for(V.MODEL) == "anthropic"


# --------------------------------------------------------- halves around the 50
# Every rule in infer_halves except this one assumes the whole frame is in one half
# of the field. A sideline camera sits at midfield, so the shot is 30-50-30 and it
# is not — and forcing one half onto that puts two yard lines forty yards apart
# onto the same line.


def _reading(numbers, end_zone="none_visible"):
    """A FieldReading carrying just the painted numbers, as (number, x_frac)."""
    return V.FieldReading(
        lines=[], landmarks=[], down_marker_number=None,
        yard_numbers=[
            V.YardNumber(number=n, x_frac=x, y_frac=0.25,
                         side="unclear", confidence="certain")
            for n, x in numbers
        ],
        numbers_decrease_toward="unclear", end_zone_at=end_zone,
        far_sideline_at="top", near_sideline_visible=False,
        camera_view="sideline_elevated", level="nfl", team_colors=[], notes="",
    )


def test_numbers_either_side_of_the_50_land_in_different_halves():
    """The failure that cost every anchor frame of an NFL broadcast: one half was
    forced onto a frame that straddles the 50, putting yard lines forty yards
    apart onto the same line."""
    r = _reading([(30, 0.2), (40, 0.35), (50, 0.51), (40, 0.68), (30, 0.82)])
    notes = V.infer_halves(r)

    by_x = sorted(r.yard_numbers, key=lambda n: n.x_frac)
    assert [n.side for n in by_x] == [
        "left_half", "left_half", "unclear", "right_half", "right_half"], notes

    # Which is the whole point: five distinct yard lines, forty yards apart.
    xs = [V.field_x_for(n.number, n.side) for n in by_x]
    assert xs == [40.0, 50.0, 60.0, 70.0, 80.0]
    assert max(xs) - min(xs) == 40.0


def test_a_symmetric_pair_is_no_longer_collapsed_onto_one_line():
    """Without the 50 to split on, [40, 40] is neither rising nor falling, and the
    sequence rule gives up. With it, the pair is twenty yards apart."""
    r = _reading([(40, 0.3), (50, 0.5), (40, 0.7)])
    V.infer_halves(r)
    xs = sorted(V.field_x_for(n.number, n.side)
                for n in r.yard_numbers if n.number != 50)
    assert xs == [50.0, 70.0], "the two 40s are the same yard line again"


def test_the_50_split_beats_a_visible_end_zone():
    """An end zone says the frame is in one half. Numbers straddling the 50 say it
    is not, and that is the more direct evidence — it is about these very numbers."""
    r = _reading([(40, 0.3), (50, 0.5), (40, 0.7)], end_zone="left")
    notes = V.infer_halves(r)
    assert "split at the 50" in notes[0]
    by_x = sorted(r.yard_numbers, key=lambda n: n.x_frac)
    assert by_x[-1].side == "right_half"


def test_numbers_all_on_one_side_of_the_50_still_use_the_old_rule():
    """The 50-split must not fire when the frame really is in one half: a 50 at the
    edge with 40 and 30 running away from it is one half, and both rules agree."""
    r = _reading([(50, 0.2), (40, 0.5), (30, 0.8)])
    V.infer_halves(r)
    assert [n.side for n in r.yard_numbers if n.number != 50] == ["right_half"] * 2
    xs = sorted(V.field_x_for(n.number, n.side) for n in r.yard_numbers)
    assert xs == [60.0, 70.0, 80.0]


def test_a_frame_with_no_50_is_untouched_by_the_split():
    r = _reading([(20, 0.3), (30, 0.6)])
    notes = V.infer_halves(r)
    assert not any("split at the 50" in n for n in notes)
    assert all(n.side == "left_half" for n in r.yard_numbers)


def _two_row_reading(far, near):
    """Painted numbers in both sideline rows, as (number, x_frac) lists."""
    return V.FieldReading(
        lines=[], landmarks=[], down_marker_number=None,
        yard_numbers=(
            [V.YardNumber(number=n, x_frac=x, y_frac=0.25, side="unclear",
                          confidence="certain") for n, x in far]
            + [V.YardNumber(number=n, x_frac=x, y_frac=0.80, side="unclear",
                            confidence="certain") for n, x in near]
        ),
        numbers_decrease_toward="unclear", end_zone_at="none_visible",
        far_sideline_at="top", near_sideline_visible=True,
        camera_view="sideline_elevated", level="nfl", team_colors=[], notes="",
    )


def test_two_sideline_rows_are_not_read_as_one_interleaved_sequence():
    """A field paints each number twice. [20, 30, 40] along the far sideline with a
    single 30 near the camera sorts by x into [20, 30, 40, 30] — which rises and
    falls, and cost a whole frame on the NFL broadcast."""
    r = _two_row_reading(far=[(20, 0.2), (30, 0.5), (40, 0.8)], near=[(30, 0.6)])
    notes = V.infer_halves(r)

    assert not any("do not run in one direction" in n for n in notes), notes
    assert all(n.side == "left_half" for n in r.yard_numbers), notes


def test_a_row_that_genuinely_doubles_back_is_still_refused():
    """The guard has to keep working: a single row that rises then falls is a bad
    read, not two rows, and forcing a half onto it would be an invention."""
    r = _two_row_reading(far=[(20, 0.2), (40, 0.5), (30, 0.8)], near=[])
    notes = V.infer_halves(r)
    assert any("do not run in one direction" in n for n in notes), notes
    assert all(n.side == "unclear" for n in r.yard_numbers)


def test_rows_are_split_the_same_way_the_field_y_split_does_it():
    """Two functions disagreeing about how many rows a frame has would be worse
    than either answer, so they share the threshold."""
    one = _two_row_reading(far=[(20, 0.2), (30, 0.5)], near=[])
    assert len(V._number_rows(one.yard_numbers)) == 1
    two = _two_row_reading(far=[(20, 0.2)], near=[(20, 0.3)])
    assert len(V._number_rows(two.yard_numbers)) == 2


def test_a_50_with_no_40_beside_it_is_not_trusted_to_split_the_halves():
    """On this project's NFL clip the model returned a 50 at the middle of the
    frame, "certain", in three reads out of four, and there is no 50 on that field
    — it was reading the "1ST & 10" caption. Acted on, that scatters real numbers
    across both halves, forty yards apart. Walking away from a real 50 the next
    number is always a 40, so a 30 directly beside it gives the lie away."""
    r = _reading([(30, 0.27), (50, 0.51), (30, 0.72)])
    # 30 immediately either side of the claimed 50: impossible on a real field.
    notes = V.infer_halves(r)
    assert not any("split at the 50" in n for n in notes), notes


def test_a_50_flanked_by_a_40_is_still_trusted():
    """The guard must not throw away the real case it was built for."""
    r = _reading([(30, 0.2), (40, 0.35), (50, 0.51), (40, 0.68), (30, 0.82)])
    notes = V.infer_halves(r)
    assert any("split at the 50" in n for n in notes), notes
    by_x = sorted(r.yard_numbers, key=lambda n: n.x_frac)
    assert [n.side for n in by_x] == [
        "left_half", "left_half", "unclear", "right_half", "right_half"]
    xs = [V.field_x_for(n.number, n.side) for n in by_x]
    assert xs == [40.0, 50.0, 60.0, 70.0, 80.0]
