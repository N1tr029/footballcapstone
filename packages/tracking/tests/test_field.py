from gridiron_tracking import field as F


def test_hash_insets_differ_by_level():
    """The one dimension that varies, and the one that silently ruins a
    registration if it is taken from the wrong code."""
    assert F.NFL.hash_inset > F.NCAA.hash_inset > F.HIGH_SCHOOL.hash_inset
    # An NFL hash is nearly six yards further from the sideline than a school one.
    assert F.NFL.hash_inset - F.HIGH_SCHOOL.hash_inset > 5.0


def test_nfl_hashes_match_the_renderer():
    lo, hi = F.NFL.hashes
    assert round(lo, 2) == 23.58 and round(hi, 2) == 29.72


def test_painted_numbers_are_symmetric():
    assert F.painted_number(60.0) == 50
    assert F.painted_number(35.0) == F.painted_number(85.0) == 25
    assert F.painted_number(F.GOAL_A) == 0


def test_catalog_spans_the_field_and_names_both_sides():
    cat = F.catalog(F.NFL)
    assert "y50/sideline_y0" in cat and "goalB/sideline_ymax" in cat
    # Two different 25-yard lines, sixty yards apart, distinguishable by id.
    assert cat["y25A/hash_y0"].x != cat["y25B/hash_y0"].x
    # End-zone back lines carry no hash marks.
    assert "backA/hash_y0" not in cat


def test_hash_landmarks_follow_the_spec():
    nfl = F.catalog(F.NFL)["y50/hash_y0"]
    hs = F.catalog(F.HIGH_SCHOOL)["y50/hash_y0"]
    assert nfl.y != hs.y
    assert hs.y == F.HIGH_SCHOOL.hash_inset
