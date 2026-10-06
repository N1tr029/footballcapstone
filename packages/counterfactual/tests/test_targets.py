"""The receiver-at-throw features, on a play small enough to check by hand.

No dataset needed: the rows are written in the 2025 release's own column names, in
stadium coordinates, with the offense going LEFT — so a conversion that forgets to mirror
fails here rather than quietly on half the season.
"""

import numpy as np
import pandas as pd
import pytest

from gridiron_tracking.sources import ngs

from gridiron_counterfactual import bdb2025, targets

# Contract coordinates (offense attacks +x). Line of scrimmage at x = 40.
LOS = 40.0
QB, WR_OPEN, WR_COVERED = 1, 2, 3
CB_TIGHT, CB_OFF, RUSHER = 20, 21, 22
#            id          club   x     y     s    dir(contract)  o(contract)
PLAYERS = [
    (QB,         "KC", 35.0, 26.0, 0.0,   0.0,   0.0),
    (WR_OPEN,    "KC", 50.0, 40.0, 6.0,   0.0, 180.0),   # 10 past the line, looking back
    (WR_COVERED, "KC", 45.0, 10.0, 5.0,   0.0,   0.0),   # back to the passer
    (CB_TIGHT,   "LV", 46.0, 10.0, 5.0,   0.0,   0.0),   # 1 yard from WR_COVERED
    (CB_OFF,     "LV", 56.0, 48.0, 3.0, 180.0, 180.0),   # 10 yards from WR_OPEN
    (RUSHER,     "LV", 38.0, 26.0, 4.0, 180.0, 180.0),   # 3 yards from the QB
]


def _contract_to_ngs(a):
    return float(ngs.contract_angle_to_ngs(a))


def _raw(frame_id=30, event="pass_forward", shift=0.0):
    """The play as the release would publish it, offense going left."""
    rows = []
    for pid, club, x, y, s, d, o in PLAYERS:
        x = x - shift if pid == CB_OFF else x      # CB_OFF was `shift` yards closer earlier
        rows.append(dict(
            gameId=1, playId=7, nflId=pid, displayName=f"P{pid}", frameId=frame_id, jerseyNumber=pid,
            club=club, playDirection="left", x=120.0 - x, y=53.3 - y, s=s, a=0.0, dis=0.0,
            dir=_contract_to_ngs((d + 180) % 360), o=_contract_to_ngs((o + 180) % 360), event=event))
    rows.append(dict(gameId=1, playId=7, nflId=None, displayName="football", frameId=frame_id,
                     jerseyNumber=None, club="football", playDirection="left", x=120.0 - 35.0,
                     y=53.3 - 26.0, s=0.0, a=0.0, dis=0.0, dir=None, o=None, event=event))
    return pd.DataFrame(rows)


@pytest.fixture
def now():
    return bdb2025.contract_frame(_raw())


def test_contract_frame_mirrors_a_play_going_left(now):
    qb = now[now["nflId"] == QB].iloc[0]
    assert (qb["x"], qb["y"]) == pytest.approx((35.0, 26.0))
    wr = now[now["nflId"] == WR_OPEN].iloc[0]
    assert wr["dir"] == pytest.approx(0.0, abs=1e-6) and wr["o"] == pytest.approx(180.0)


def test_contract_frame_agrees_with_the_shared_loader(now):
    raw = _raw().assign(club=lambda d: d["club"].replace("football", "ball"))
    track = ngs.to_play_track(ngs.load_tracking(raw), play_id=7)
    frame = track.frames[0]
    for row in now[now["nflId"].notna()].itertuples():
        p = frame.players[str(int(row.nflId))]
        assert (p.x, p.y) == pytest.approx((row.x, row.y))
        assert p.dir == pytest.approx(row.dir, abs=1e-6) and p.o == pytest.approx(row.o, abs=1e-6)
    assert frame.ball is not None and frame.ball.x == pytest.approx(35.0)


def test_receiver_features(now):
    before = bdb2025.contract_frame(_raw(frame_id=25, event=None, shift=2.0))
    f = targets.receiver_features(now, before, "KC", [WR_OPEN, WR_COVERED], LOS).set_index("nflId")

    assert list(f.index) == [WR_OPEN, WR_COVERED]          # the passer is never his own target
    assert f.loc[WR_OPEN, "depth"] == pytest.approx(10.0)
    assert f.loc[WR_OPEN, "lateral"] == pytest.approx(14.0)
    assert f.loc[WR_COVERED, "sideline"] == pytest.approx(10.0)
    assert f.loc[WR_OPEN, "throw_dist"] == pytest.approx(np.hypot(15, 14))

    assert f.loc[WR_OPEN, "separation"] == pytest.approx(10.0)
    assert f.loc[WR_COVERED, "separation"] == pytest.approx(1.0)
    assert f.loc[WR_OPEN, "separation_trend"] == pytest.approx(10.0 - np.hypot(4, 8))   # opening
    assert f.loc[WR_COVERED, "separation_trend"] == pytest.approx(0.0)
    assert f.loc[WR_OPEN, "defender_speed"] == pytest.approx(3.0)

    assert f.loc[WR_OPEN, "facing_passer"] > 0.7            # looking back toward the QB
    assert f.loc[WR_COVERED, "facing_passer"] < 0           # back turned
    assert f["pressure"].tolist() == pytest.approx([3.0, 3.0])
    assert f["n_routes"].tolist() == [2, 2]


def test_lane_defenders_counts_only_men_between_passer_and_receiver():
    # Passer at the origin, receiver 20 yards straight downfield.
    px, py = 0.0, 0.0
    rx, ry = np.array([20.0]), np.array([0.0])
    dx = np.array([10.0, 10.0, 25.0, -3.0])    # in the lane, wide of it, beyond him, behind the passer
    dy = np.array([1.0, 5.0, 0.0, 0.0])
    assert targets._lane_count(px, py, rx, ry, dx, dy).tolist() == [1]


def test_no_trend_without_an_earlier_frame(now):
    f = targets.receiver_features(now, None, "KC", [WR_OPEN], LOS)
    assert np.isnan(f["separation_trend"].iloc[0])


def test_a_quarterback_near_the_ball_is_the_passer_even_if_a_lineman_is_nearer(now):
    # Move the ball a yard downfield of the QB, right onto the rusher's side: nearest
    # offensive player by distance is still the QB here, so put a lineman on the ball.
    lineman = now[now["nflId"] == QB].assign(nflId=9, x=37.0)
    ball = now["nflId"].isna()
    moved = pd.concat([now.assign(x=now["x"].where(~ball, 37.0)), lineman])
    plain = targets.receiver_features(moved, None, "KC", [WR_OPEN], LOS)
    told = targets.receiver_features(moved, None, "KC", [WR_OPEN], LOS, quarterbacks=[QB])
    assert plain["passer"].iloc[0] == 9 and told["passer"].iloc[0] == QB
