import pytest

from gridiron_tracking.teams import constrained_teams


def test_full_formation_resolves_one_ambiguous_jersey_and_stays_stable():
    p = {f"a{i}": [.95, .05] for i in range(11)}
    p.update({f"b{i}": [.05, .95] for i in range(11)})
    p["a0"] = [.48, .52]
    got = constrained_teams(p, [list(p), ["a0", "b0"]], full_frame=0)
    assert all(got[f"a{i}"] == "teamA" for i in range(11))
    assert all(got[f"b{i}"] == "teamB" for i in range(11))


def test_cropped_shots_do_not_force_equal_counts():
    p = {f"a{i}": [.99, .01] for i in range(8)}
    p.update({f"b{i}": [.01, .99] for i in range(3)})
    got = constrained_teams(p, [list(p)])
    assert list(got.values()).count("teamA") == 8
    assert list(got.values()).count("teamB") == 3


def test_over_capacity_prefers_unknown_to_a_wrong_kit():
    p = {str(i): [.99, .01] for i in range(12)}
    got = constrained_teams(p, [list(p)])
    assert list(got.values()).count("teamA") == 11
    assert list(got.values()).count(None) == 1
    assert "teamB" not in got.values()


def test_inconclusive_jersey_stays_unknown_without_a_full_formation_hint():
    assert constrained_teams({"p": [.51, .49]}, [["p"]])["p"] is None


def test_capacity_is_simultaneous_not_the_number_of_track_fragments():
    p = {str(i): [.99, .01] for i in range(22)}
    got = constrained_teams(p, [[str(i) for i in range(11)],
                                 [str(i) for i in range(11, 22)]])
    assert list(got.values()).count("teamA") == 22


def test_an_unverified_22_person_count_does_not_impose_equal_teams():
    p = {str(i): [.99, .01] for i in range(22)}
    got = constrained_teams(p, [list(p)])
    assert list(got.values()).count(None) == 11


def test_full_formation_requires_the_inspected_number_of_athletes():
    with pytest.raises(ValueError, match="full formation"):
        constrained_teams({"a": [.9, .1]}, [["a"]], full_frame=0)


def test_jersey_conflict_splits_identity_without_moving_the_player():
    from gridiron_tracking.contract import PlayTrack, RosterEntry, TrackingFrame, PlayerFrame
    from gridiron_tracking.teams import split_team_changes

    track = PlayTrack(roster={"p": RosterEntry(id="p", team="teamA")}, frames=[
        TrackingFrame(t=i*.1, players={"p": PlayerFrame(x=40+i, y=20)}) for i in range(4)])
    n = split_team_changes(track, [{"p": "teamA"}, {"p": "teamA"},
                                    {"p": "teamB"}, {"p": "teamB"}])
    assert n == 1
    assert list(track.frames[0].players) == ["p_s1"]
    assert list(track.frames[3].players) == ["p_s2"]
    assert track.roster["p_s1"].team == "teamA"
    assert track.roster["p_s2"].team == "teamB"
    assert track.frames[3].players["p_s2"].x == 43


def test_an_unknown_interval_does_not_create_a_new_known_identity():
    from gridiron_tracking.contract import PlayTrack, RosterEntry, TrackingFrame, PlayerFrame
    from gridiron_tracking.teams import split_team_changes

    track = PlayTrack(roster={"p": RosterEntry(id="p", team="teamA")}, frames=[
        TrackingFrame(t=i*.1, players={"p": PlayerFrame(x=40+i, y=20)}) for i in range(3)])
    assert split_team_changes(track, [{"p": "teamA"}, {"p": None}, {"p": "teamA"}]) == 0
    assert list(track.frames[0].players) == list(track.frames[2].players)
    assert track.roster[next(iter(track.frames[1].players))].team == "unknown"
