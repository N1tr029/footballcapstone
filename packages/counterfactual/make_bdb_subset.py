"""
Shrink the Big Data Bowl 2025 files to what the forecast model needs, so they can
be attached to a chat.

Keeps:
  - every player row at the snap, the throw (pass_forward / pass_shovel) and the
    catch (pass_arrived) on every play, all weeks: training data for the forecast
  - every player row 0.5 s and 1.0 s before the throw: movement history
  - every row of the demo game (2022090800): full replays for the app
  - plays.csv, players.csv, player_play.csv, games.csv in full (they are small)

Usage (needs pandas):
    python make_bdb_subset.py /path/to/nfl-big-data-bowl-2025

Writes bdb2025_subset.zip next to this script. Unzip it and pass the folder to
decision_engine.py --data-dir.
"""
import sys
import zipfile
from pathlib import Path

import pandas as pd

DEMO_GAME = 2022090800
KEEP_EVENTS = {"pass_forward", "pass_shovel", "pass_arrived", "ball_snap"}
THROW_EVENTS = {"pass_forward", "pass_shovel"}
HISTORY_FRAMES = (5, 10)  # 0.5 s and 1.0 s before the throw, at 10 frames/s

src = Path(sys.argv[1] if len(sys.argv) > 1 else ".")
out_dir = Path(__file__).resolve().parent / "bdb2025_subset"
out_dir.mkdir(exist_ok=True)

weeks = sorted(src.glob("**/tracking_week_*.csv"))
if not weeks:
    sys.exit(f"No tracking_week_*.csv found under {src.resolve()}")

KEYS = ["gameId", "playId", "frameId"]
kept = []
for path in weeks:
    print(f"filtering {path.name} ...", flush=True)
    # Pass 1: find the event frames for the whole file first, so a play split
    # across two chunks still gets its history frames.
    events = []
    for chunk in pd.read_csv(path, chunksize=1_000_000, usecols=KEYS + ["event"], low_memory=False):
        e = chunk["event"].fillna("").astype(str).str.lower()
        events.append(chunk.loc[e.isin(KEEP_EVENTS), KEYS + ["event"]].drop_duplicates())
    events = pd.concat(events).drop_duplicates()
    throws = events.loc[events["event"].str.lower().isin(THROW_EVENTS), KEYS]
    frame_keys = pd.concat([events[KEYS], *[throws.assign(frameId=throws["frameId"] - k) for k in HISTORY_FRAMES]])
    frame_keys = frame_keys.drop_duplicates()
    # Pass 2: keep those frames, plus every row of the demo game.
    for chunk in pd.read_csv(path, chunksize=1_000_000, low_memory=False):
        at_event = chunk.merge(frame_keys, on=KEYS, how="inner")
        demo = chunk[chunk["gameId"] == DEMO_GAME]
        kept.append(pd.concat([at_event, demo]).drop_duplicates())

tracking = pd.concat(kept, ignore_index=True)
tracking.to_csv(out_dir / "tracking_subset.csv", index=False)
print(f"tracking_subset.csv: {len(tracking):,} rows from {len(weeks)} week files")

for name in ("plays.csv", "players.csv", "player_play.csv", "games.csv"):
    found = next(iter(src.glob(f"**/{name}")), None)
    if found is None:
        print(f"warning: {name} not found, skipping")
        continue
    pd.read_csv(found, low_memory=False).to_csv(out_dir / name, index=False)

zip_path = Path(__file__).resolve().parent / "bdb2025_subset.zip"
with zipfile.ZipFile(zip_path, "w", zipfile.ZIP_DEFLATED, compresslevel=9) as z:
    for f in sorted(out_dir.iterdir()):
        z.write(f, f.name)
print(f"\nDone: {zip_path} ({zip_path.stat().st_size / 1e6:.1f} MB).")
