"""The 2025 Big Data Bowl release (2022 season, weeks 1-9), as the what-if model reads it.

Two shapes come out of here, for two different jobs.

:func:`play_track` is one play as the shared ``PlayTrack`` — built by stream 2's own
loader, so a play the model reasons about is the same object the replay engine draws.

:func:`contract_frame` is a whole game as a DataFrame in *contract coordinates*: offense
attacking +x, y toward the offense's right, angles counter-clockwise from +x. Training
needs thousands of plays at once and ``PlayTrack`` is built a row at a time, so the
table path does the same conversion vectorised. The two must agree, and
``tests/test_bdb2025.py`` holds them to it.

The raw release is nine weekly CSVs of about a gigabyte each. :func:`build_cache` splits
them once into one Parquet file per game under ``$FOOTBALL_DATA_DIR/bdb2025/tracking``;
nothing here ever writes to the download itself.
"""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path

import numpy as np
import pandas as pd

from gridiron_tracking.config import find_env, load_env
from gridiron_tracking.contract import PlayTrack
from gridiron_tracking.sources import ngs

WEEKS = range(1, 10)
FIELD_LENGTH, FIELD_WIDTH = 120.0, 53.3
"""53.3, not 160/3, because that is the width stream 2's loader mirrors about."""

KEY = ["gameId", "playId"]


# ------------------------------------------------------------------------ paths


def _root() -> Path:
    env = find_env()
    return env.parent if env else Path.cwd()


def _path(name: str, default: str | None = None) -> Path:
    load_env()
    value = os.environ.get(name, default)
    if not value:
        raise RuntimeError(
            f"{name} is not set. Download the NFL Big Data Bowl 2025 data from Kaggle and "
            f"point {name} at the folder in .env (see packages/counterfactual/README.md)."
        )
    p = Path(value).expanduser()
    return p if p.is_absolute() else _root() / p


def raw_dir() -> Path:
    return _path("FOOTBALL_BDB2025_DIR")


def cache_dir() -> Path:
    return _path("FOOTBALL_DATA_DIR", "./data") / "bdb2025" / "tracking"


# ------------------------------------------------------------------------ cache


def build_cache(weeks=WEEKS, log=print) -> list[int]:
    """Split ``tracking_week_N.csv`` into one Parquet file per game. About a minute."""
    import pyarrow as pa
    import pyarrow.compute as pc
    import pyarrow.csv as pcsv
    import pyarrow.parquet as pq

    out = cache_dir()
    out.mkdir(parents=True, exist_ok=True)
    written: list[int] = []
    for week in weeks:
        # Streamed in batches so a 1 GB file never has to fit in memory. The release
        # writes missing values as the string "NA", including in `event`.
        reader = pcsv.open_csv(
            raw_dir() / f"tracking_week_{week}.csv",
            read_options=pcsv.ReadOptions(block_size=64 << 20),
            convert_options=pcsv.ConvertOptions(
                strings_can_be_null=True,
                column_types={"gameId": pa.int64(), "playId": pa.int32(), "nflId": pa.int32(),
                              "frameId": pa.int16(), "jerseyNumber": pa.int16(), "time": pa.string(),
                              "event": pa.string(),
                              **{c: pa.float32() for c in ("x", "y", "s", "a", "dis", "o", "dir")}},
            ),
        )
        writers: dict[int, pq.ParquetWriter] = {}
        try:
            for batch in reader:
                table = pa.Table.from_batches([batch])
                for gid in pc.unique(table["gameId"]).to_pylist():
                    part = table.filter(pc.equal(table["gameId"], gid))
                    if gid not in writers:
                        writers[gid] = pq.ParquetWriter(out / f"{gid}.parquet.part", part.schema,
                                                        compression="zstd")
                    writers[gid].write_table(part)
        finally:
            for w in writers.values():
                w.close()
        for gid in writers:  # only complete files get the real name
            (out / f"{gid}.parquet.part").rename(out / f"{gid}.parquet")
        log(f"week {week}: {len(writers)} games")
        written += sorted(writers)
    return written


# ----------------------------------------------------------------------- tables


@lru_cache(maxsize=1)
def games() -> pd.DataFrame:
    return pd.read_csv(raw_dir() / "games.csv")


@lru_cache(maxsize=1)
def plays() -> pd.DataFrame:
    return pd.read_csv(raw_dir() / "plays.csv")


@lru_cache(maxsize=1)
def players() -> pd.DataFrame:
    return pd.read_csv(raw_dir() / "players.csv")


@lru_cache(maxsize=1)
def player_play() -> pd.DataFrame:
    return pd.read_csv(raw_dir() / "player_play.csv")


def raw_tracking(game_id: int) -> pd.DataFrame:
    """One game exactly as released: stadium coordinates, compass angles."""
    path = cache_dir() / f"{game_id}.parquet"
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Run `python -m gridiron_counterfactual build-cache` first.")
    return pd.read_parquet(path)


# ------------------------------------------------------------------- conversion


def contract_frame(raw: pd.DataFrame) -> pd.DataFrame:
    """Raw release rows to contract coordinates and angles, vectorised.

    The same three steps ``ngs.to_play_track`` takes a row at a time: mirror the field
    through its centre when the offense was going left, turn compass bearings into
    contract angles, and turn those by 180 degrees on a mirrored play.
    """
    out = raw.copy()
    flip = (out["playDirection"] == "left").to_numpy()
    out["x"] = np.where(flip, FIELD_LENGTH - out["x"], out["x"])
    out["y"] = np.where(flip, FIELD_WIDTH - out["y"], out["y"])
    for col in ("dir", "o"):
        a = ngs.ngs_angle_to_contract(out[col].to_numpy(dtype=float))
        out[col] = np.mod(a + np.where(flip, 180.0, 0.0), 360.0)
    return out


def line_of_scrimmage(play: pd.Series | pd.DataFrame, direction) -> float | np.ndarray:
    """``absoluteYardlineNumber`` in contract x. It is a stadium coordinate, so it mirrors."""
    los = play["absoluteYardlineNumber"]
    return np.where(np.asarray(direction) == "left", FIELD_LENGTH - los, los)


def play_track(game_id: int, play_id: int, raw: pd.DataFrame | None = None) -> PlayTrack:
    """One play as the shared ``PlayTrack``, with what the tracking file alone cannot say.

    Stream 2's loader does the conversion. Added on top: each player's position from
    ``players.csv`` as his roster ``role``, and the offense, defense and line of scrimmage
    from ``plays.csv`` in ``meta``.
    """
    raw = raw_tracking(game_id) if raw is None else raw
    rows = raw[raw["playId"] == play_id]
    if rows.empty:
        raise KeyError(f"no tracking for play {game_id}-{play_id}")

    # The loader recognises the ball by a team of "ball"; this release calls it "football".
    rows = rows.assign(club=rows["club"].replace("football", "ball"))
    track = ngs.to_play_track(ngs.load_tracking(rows), play_id=play_id)

    position = dict(zip(players()["nflId"], players()["position"]))
    for pid, entry in track.roster.items():
        entry.role = position.get(int(pid)) if pid.isdigit() else None

    p = plays()
    p = p[(p["gameId"] == game_id) & (p["playId"] == play_id)]
    track.meta.update(game_id=int(game_id), play_id=int(play_id),
                      orientation_measured=bool(track.meta["orientation_measured"]),
                      acceleration_measured=bool(track.meta["acceleration_measured"]))
    if len(p):
        row = p.iloc[0]
        track.meta.update(
            offense=row["possessionTeam"], defense=row["defensiveTeam"],
            line_of_scrimmage=float(line_of_scrimmage(row, track.meta["play_direction"])),
        )
    return track
