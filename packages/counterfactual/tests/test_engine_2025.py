"""the 2025 path of decision_engine.py, on a tiny made-up game so it runs without
the kaggle data. players move in straight lines, so every answer is checkable by
hand."""

import json

import numpy as np
import pandas as pd
import pytest

import decision_engine as de

W = 160 / 3
LOS = 40.0  # line of scrimmage in normalized coordinates (offense attacks +x)
SNAP, THROW, ARRIVAL, LAST = 11, 25, 33, 36

# nflId -> (position, offense?, route?, target?, start x rel to los, start y, vx, vy) in yards/frame
ROSTER = {
    1: ("QB", 1, 0, 0, -7, 26.7, -0.1, 0.0),
    2: ("WR", 1, 1, 1, -1, 10.0, 0.5, 0.0),   # the target
    3: ("WR", 1, 1, 0, -1, 45.0, 0.6, 0.0),   # route runner on the other side
    4: ("TE", 1, 1, 0, -1, 30.0, 0.3, 0.1),
    5: ("RB", 1, 1, 0, -5, 28.0, 0.2, 0.2),
    **{i: ("G", 1, 0, 0, -1, 22.0 + i, 0.0, 0.0) for i in range(6, 12)},
    12: ("CB", 0, 0, 0, 6, 10.5, 0.5, 0.0),   # sits on the target
    13: ("CB", 0, 0, 0, 7, 44.0, 0.3, 0.0),
    **{i: ("OLB", 0, 0, 0, 3, 18.0 + 1.5 * (i - 14), 0.0, 0.0) for i in range(14, 23)},
}


def make_data(games=((2022090800, 1), (2022091100, 2)), passes_per_game=4) -> dict:
    tracking, plays, player_play = [], [], []
    for game_id, week in games:
        for k in range(passes_per_game):
            play_id = 100 + k
            left = k % 2 == 1
            wobble = 0.05 * k  # a little variety between plays
            for f in range(1, LAST + 1):
                event = {SNAP: "ball_snap", THROW: "pass_forward", ARRIVAL: "pass_arrived"}.get(f)
                moving = max(0, f - SNAP)  # everyone stands still until the snap
                for nid, (pos, off, route, tgt, x0, y0, vx, vy) in ROSTER.items():
                    vy2 = vy + (wobble if off else 0)
                    x, y = LOS + x0 + vx * moving, y0 + vy2 * moving
                    bearing = float(np.degrees(np.arctan2(vx, vy2)) % 360) if (vx or vy2) else 0.0
                    s = float(np.hypot(vx, vy2) * 10) if f > SNAP else 0.0
                    rx, ry = (120 - x, W - y) if left else (x, y)
                    tracking.append(dict(gameId=game_id, playId=play_id, nflId=nid, displayName=f"Player {nid}",
                                         frameId=f, jerseyNumber=nid, club="BUF" if off else "LA",
                                         playDirection="left" if left else "right", x=rx, y=ry, s=s, a=0.0,
                                         dis=s / 10, o=(bearing + 180) % 360 if left else bearing,
                                         dir=(bearing + 180) % 360 if left else bearing, event=event))
                # the ball sits with the qb, then flies to the target
                qx, qy = LOS - 7 - 0.1 * min(moving, THROW - SNAP), 26.7
                if f <= THROW:
                    bx, by = qx, qy
                else:
                    tx, ty = LOS - 1 + 0.5 * (ARRIVAL - SNAP), 10.0
                    frac = min(1.0, (f - THROW) / (ARRIVAL - THROW))
                    bx, by = qx + (tx - qx) * frac, qy + (ty - qy) * frac
                rbx, rby = (120 - bx, W - by) if left else (bx, by)
                tracking.append(dict(gameId=game_id, playId=play_id, nflId=np.nan, displayName="football", frameId=f,
                                     jerseyNumber=np.nan, club="football", playDirection="left" if left else "right",
                                     x=rbx, y=rby, s=0.0, a=0.0, dis=0.0, o=np.nan, dir=np.nan, event=event))
            plays.append(dict(gameId=game_id, playId=play_id, playDescription=f"pass {play_id}", quarter=1,
                              down=2, yardsToGo=7, possessionTeam="BUF", defensiveTeam="LA",
                              absoluteYardlineNumber=120 - LOS if left else LOS, gameClock="10:00",
                              passResult="C", yardsGained=8, expectedPointsAdded=0.5, isDropback=True,
                              qbKneel=0, qbSpike=False))
            for nid, (pos, off, route, tgt, *_rest) in ROSTER.items():
                player_play.append(dict(gameId=game_id, playId=play_id, nflId=nid,
                                        wasRunningRoute=1.0 if route else np.nan, wasTargettedReceiver=tgt))
        # two runs per game: one that failed, one that worked
        for play_id, gained, down in ((900, -1, 1), (901, 6, 1)):
            plays.append(dict(gameId=game_id, playId=play_id, playDescription=f"run {play_id}", quarter=2, down=down,
                              yardsToGo=10, possessionTeam="BUF", defensiveTeam="LA", absoluteYardlineNumber=LOS,
                              gameClock="5:00", passResult=np.nan, yardsGained=gained, expectedPointsAdded=-0.3,
                              isDropback=False, qbKneel=0, qbSpike=np.nan))
    players = pd.DataFrame([dict(nflId=n, position=v[0], displayName=f"Player {n}") for n, v in ROSTER.items()])
    games_df = pd.DataFrame([dict(gameId=g, week=w, homeTeamAbbr="LA", visitorTeamAbbr="BUF") for g, w in games])
    return {"tracking": pd.DataFrame(tracking), "plays": pd.DataFrame(plays),
            "player_play": pd.DataFrame(player_play), "players": players, "games": games_df}


@pytest.fixture(scope="module")
def data():
    return make_data()


@pytest.fixture(scope="module")
def pipeline(data):
    windows = de.pass_windows(data["tracking"])
    samples = de.forecast_samples(data, windows)
    predicted, evaluation = de.cross_fit_forecast(samples)
    return windows, samples, predicted, evaluation


def test_left_plays_are_flipped_like_the_hindsight_exporter(data):
    t = de.normalize(data["tracking"])
    raw = data["tracking"]
    left = raw.playDirection.eq("left") & raw.nflId.notna()
    assert np.allclose(t.loc[left, "x"], 120 - raw.loc[left, "x"])
    assert np.allclose(t.loc[left, "y"], W - raw.loc[left, "y"])
    assert np.allclose(t.loc[left, "dir"], (raw.loc[left, "dir"] + 180) % 360)
    # after the flip every play's target runs toward +x
    target = t[(t.nflId == 2) & (t.frameId == ARRIVAL)]
    start = t[(t.nflId == 2) & (t.frameId == SNAP)]
    assert (target.x.to_numpy() > start.x.to_numpy()).all()


def test_pass_windows_find_the_throw_and_the_catch(data):
    w = de.pass_windows(data["tracking"])
    assert len(w) == 8
    assert (w.snap == SNAP).all() and (w.throw == THROW).all() and (w.arrival == ARRIVAL).all()
    assert (w.h == ARRIVAL - THROW).all()


def test_one_sample_per_player_per_pass_with_the_true_displacement(pipeline):
    _, samples, _, _ = pipeline
    assert len(samples) == 8 * 22
    assert samples[de.FORECAST_FEATURES].notna().all().all()
    target = samples[samples.nflId == 2]
    # the target runs 0.5 yd/frame toward +x on every play, left or right
    assert np.allclose(target.tx, 0.5 * (ARRIVAL - THROW))
    assert (samples.offense.groupby([samples.gameId, samples.playId]).sum() == 11).all()


def test_the_forecast_is_judged_on_weeks_it_never_saw(pipeline):
    _, _, predicted, evaluation = pipeline
    assert evaluation["held_out_by"] == "week" and evaluation["folds"] == 2
    assert predicted[["px", "py"]].notna().all().all()
    errors = evaluation["error_yards_at_arrival"]
    assert set(errors) == {"model", "keep_current_speed", "stand_still"}
    assert errors["model"]["mean"] < errors["stand_still"]["mean"]


def _frame(overrides=None):
    """one pass, hand-placed at the catch: target covered, a WR wide open past the line."""
    rows = [
        dict(nflId=1, position="QB", offense=1, route=0, target=0, px=30, py=26),
        dict(nflId=2, position="WR", offense=1, route=1, target=1, px=50, py=10),
        dict(nflId=3, position="WR", offense=1, route=1, target=0, px=52, py=45),
        dict(nflId=12, position="CB", offense=0, route=0, target=0, px=50.5, py=10),
        dict(nflId=13, position="CB", offense=0, route=0, target=0, px=45, py=30),
    ]
    for row in rows:
        row.update((overrides or {}).get(row["nflId"], {}))
    df = pd.DataFrame(rows)
    df["x_arr"], df["y_arr"] = df.px, df.py
    df["gameId"], df["playId"], df["los"], df["throw"], df["arrival"] = 1, 7, LOS, THROW, ARRIVAL
    df["displayName"] = "Player " + df.nflId.astype(str)
    return df


def _plays(down=2):
    return pd.DataFrame([dict(gameId=1, playId=7, playDescription="a pass", passResult="I", yardsGained=0,
                              down=down, yardsToGo=7, expectedPointsAdded=-0.6, quarter=3, gameClock="2:00")])


def test_a_clearly_more_open_route_runner_is_recommended_by_name():
    [r] = de.review_passes(_frame(), _plays())
    assert r["decision"] == "consider_alternative"
    assert r["evidence"]["best_alternative_receiver_id"] == 3
    assert "Player 3 (WR)" in r["recommendation_reason"] and "Player 2 (WR)" in r["recommendation_reason"]
    assert r["expected_points_added"] == -0.6 and r["pass_result"] == "I"


def test_the_quarterback_is_never_an_alternative():
    [r] = de.review_passes(_frame({1: dict(route=1, px=60, py=30)}), _plays())
    assert r["evidence"]["best_alternative_receiver_id"] != 1


def test_players_not_running_a_route_are_never_alternatives():
    [r] = de.review_passes(_frame({3: dict(route=0)}), _plays())
    assert r["evidence"]["best_alternative_receiver_id"] is None
    assert r["decision"] != "consider_alternative"


def test_a_receiver_behind_the_line_is_not_an_alternative():
    [r] = de.review_passes(_frame({3: dict(px=LOS - 2)}), _plays())
    assert r["evidence"]["best_alternative_receiver_id"] is None


def test_a_small_advantage_keeps_the_throw():
    # target 4 yards clear, alternative 6: open, but not 5 yards more open
    [r] = de.review_passes(_frame({12: dict(px=54, py=10), 13: dict(px=52, py=39)}), _plays())
    assert r["decision"] == "target_was_reasonable"


def test_nobody_open_suggests_a_run_or_a_punt_on_fourth_down():
    covered = {13: dict(px=52, py=46)}
    [r] = de.review_passes(_frame(covered), _plays(down=2))
    assert r["decision"] == "suggest_run_play"
    [r] = de.review_passes(_frame(covered), _plays(down=4))
    assert r["decision"] == "suggest_punt"


def test_only_failed_runs_are_flagged(data):
    runs = de.review_runs(data["plays"], {2022090800})
    assert [r["play_id"] for r in runs] == [900]
    assert runs[0]["decision"] == "suggest_pass_next_time" and runs[0]["recommendation_reason"]


def test_a_suggested_replay_is_real_until_the_throw_then_the_forecast(data, pipeline):
    windows, _, predicted, _ = pipeline
    game = predicted[predicted.gameId == 2022090800]
    reports = de.review_passes(game, data["plays"], advantage_yards=0.0)
    report = next(r for r in reports if r["play_id"] == 101)  # a play going left
    w = windows[(windows.gameId == 2022090800) & (windows.playId == 101)].iloc[0]
    payload = de.suggested_play(data, w, game[game.playId == 101], report)

    json.dumps(payload, allow_nan=False)
    ids = [f["frame_id"] for f in payload["frames"]]
    assert ids == list(range(max(1, SNAP - de.PRE_SNAP_FRAMES), ARRIVAL + 1))
    assert payload["play_direction"] == "left" and payload["source_frame_id"] == THROW
    assert len(payload["roster"]) == 22 and {r["club"] for r in payload["roster"]} == {"BUF", "LA"}

    raw = data["tracking"]
    for f in payload["frames"]:
        kinds = {p["record_type"] for p in f["players"]}
        assert kinds == ({"observed_player"} if f["frame_id"] <= THROW else {"simulated_player"})
        if f["frame_id"] <= THROW:  # recorded frames are the tracking, untouched
            truth = raw[(raw.gameId == 2022090800) & (raw.playId == 101) & (raw.frameId == f["frame_id"])]
            for p in f["players"]:
                row = truth[truth.nflId == p["nfl_id"]].iloc[0]
                assert (p["x"], p["y"]) == (round(row.x, 2), round(row.y, 2))
    events = {e["name"]: f["frame_id"] for f in payload["frames"] for e in f["events"]}
    assert events == {"ball_snap": SNAP, "pass_forward": THROW, "pass_arrived": ARRIVAL}

    last = payload["frames"][-1]
    target = payload["recommendation"]["target_nfl_id"]
    spot = next(p for p in last["players"] if p["nfl_id"] == target)
    assert abs(last["ball"]["x"] - spot["x"]) < 0.02 and abs(last["ball"]["y"] - spot["y"]) < 0.02
    assert last["ball"]["data_origin"] == "simulated_straight_line"
