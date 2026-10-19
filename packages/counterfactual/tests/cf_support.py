"""Fixtures and paths the counterfactual suite needs at import time.

Not in conftest.py: packages/tracking/tests has a conftest.py too, and a test
module that says ``from conftest import ...`` gets whichever of the two landed on
sys.path first — each suite passes alone and the pair fails to collect. A distinct
module name has no such ambiguity. Path setup stays in conftest.py, which is the
one thing pytest does want there.
"""

from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve()
PACKAGE = HERE.parents[1]            # packages/counterfactual
REPO = HERE.parents[3]

REPORTS = PACKAGE / "reports/decisions.jsonl"
MODEL = PACKAGE / "models/coordinate_model.joblib"


def bearing(dx: float, dy: float) -> float:
    """A movement vector as the compass bearing a release would record.

    Zero along +y, increasing clockwise — the inverse of
    ``ngs.ngs_angle_to_contract``. The fixtures build their angle columns from the
    movement they actually generate, so ``check_angle_convention`` is being asked a
    real question rather than one the fixture was built to pass.
    """
    return float(np.mod(90.0 - np.degrees(np.arctan2(dy, dx)), 360.0))


def bdb_play(
    n_frames: int = 20,
    play_direction: str = "right",
    game_id: int = 2023091002,
    play_id: int = 110,
    frames_output: int = 15,
) -> pd.DataFrame:
    """A Big Data Bowl 2026 ``input_2023_wXX.csv`` for one play, in its own dialect.

    Column spellings, roles, compass angles and the pre-throw-only coverage are the
    release's; the trajectories are invented. There is no Kaggle download in this
    repository and this suite must run without one, so the shape is reproduced
    rather than sampled. The real check happens the first time someone runs
    ``round_trip_error`` over a real week, which is a one-liner on the fixture's
    own terms.
    """
    rng = np.random.default_rng(17)
    players = [
        (53434, "Passer", "Offense", "QB"),
        (47965, "Targeted Receiver", "Offense", "WR"),
        (46104, "Other Route Runner", "Offense", "WR"),
        (52595, "Other Route Runner", "Offense", "TE"),
        (41233, "Defensive Coverage", "Defense", "CB"),
        (44871, "Defensive Coverage", "Defense", "FS"),
    ]

    # Which way this offense is attacking, in the release's own coordinates. The
    # release does not mirror its plays — that is what play_direction is for — so a
    # left-direction offense moves toward x = 0. Generating it moving the other way
    # produces a play that cannot happen, and a left-play test built on one is
    # testing the fixture's confusion rather than the conversion.
    attack = 1.0 if play_direction == "right" else -1.0

    rows = []
    for pid, role, side, pos in players:
        # A smooth path per player: enough curvature that dir genuinely changes,
        # slow enough that nobody outruns a human.
        x0 = 60.0 + rng.uniform(-4, 4)
        y0 = 26.65 + rng.uniform(-12, 12)
        vx = rng.uniform(1.5, 6.0) * attack * (1.0 if side == "Offense" else -0.6)
        vy = rng.uniform(-2.0, 2.0)
        curve = rng.uniform(-0.25, 0.25)

        xs, ys = [], []
        for i in range(n_frames):
            t = i / 10.0
            xs.append(x0 + vx * t)
            ys.append(y0 + vy * t + curve * t * t * 10.0)

        for i in range(n_frames):
            dx = xs[i] - xs[i - 1] if i else xs[1] - xs[0]
            dy = ys[i] - ys[i - 1] if i else ys[1] - ys[0]
            speed = float(np.hypot(dx, dy) * 10.0)
            rows.append({
                "game_id": game_id,
                "play_id": play_id,
                "nfl_id": pid,
                "frame_id": i + 1,
                "player_name": f"Player {pid}",
                "player_position": pos,
                "player_side": side,
                "player_role": role,
                "play_direction": play_direction,
                "x": xs[i],
                "y": ys[i],
                "s": speed,
                "a": 0.4,
                "dir": bearing(dx, dy),
                # Body orientation is a measurement in this release and is not the
                # direction of travel: a receiver with his head back at the ball
                # can be a long way from where he is going.
                "o": bearing(dx, dy) + (25.0 if role == "Targeted Receiver" else 5.0),
                "ball_land_x": 58.0,
                "ball_land_y": 31.0,
                "absolute_yardline_number": 38.0,
                "num_frames_output": frames_output,
            })

    df = pd.DataFrame(rows)
    df["o"] = np.mod(df["o"], 360.0)
    return df


def bdb_output(play: pd.DataFrame, frames_output: int = 15) -> pd.DataFrame:
    """The matching ``output_2023_wXX.csv``: where the players actually went."""
    last = play.sort_values("frame_id").groupby("nfl_id").tail(1)
    rows = []
    for _, r in last.iterrows():
        for step in range(1, frames_output + 1):
            t = step / 10.0
            theta = np.deg2rad(r["dir"])
            rows.append({
                "game_id": r["game_id"],
                "play_id": r["play_id"],
                "nfl_id": r["nfl_id"],
                "frame_id": step,
                "x": r["x"] + r["s"] * np.sin(theta) * t,
                "y": r["y"] + r["s"] * np.cos(theta) * t,
            })
    return pd.DataFrame(rows)


def predictions_from(track, horizons: int = 15, jitter: float = 0.0) -> pd.DataFrame:
    """Forecast rows in the release's frame of reference, without a fitted model.

    ``forecast.to_play_track`` must be testable without scikit-learn and without a
    477 KB model file being loadable on whatever Python this runs on, so the rows
    the model would emit are produced by dead reckoning from each player's last
    observed state instead.
    """
    from gridiron_counterfactual import features

    rng = np.random.default_rng(3)
    state = features.last_state(track)
    rows = []
    for _, r in state.iterrows():
        theta = np.deg2rad(r["dir"])
        for step in range(1, horizons + 1):
            t = step / 10.0
            rows.append({
                "player_id": r["player_id"],
                "horizon": step,
                "pred_x": r["x"] + r["s"] * np.sin(theta) * t + rng.normal(0, jitter),
                "pred_y": r["y"] + r["s"] * np.cos(theta) * t + rng.normal(0, jitter),
            })
    return pd.DataFrame(rows)


# The Unity app's real plays, merged in from hindsight-frontend. Suites that use
# them skip when the subtree is absent.
HINDSIGHT_PLAYS = REPO / "packages/replay-engine/hindsight/Assets/StreamingAssets/plays"
