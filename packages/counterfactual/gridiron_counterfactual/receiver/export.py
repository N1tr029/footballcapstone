"""The receiver what-if, written where the API and the Unity app read it.

Two files, the same split the play model already uses for its own output:

``reports/receiver_whatif.jsonl``
    One record per scored throw (:func:`whatif.describe`): who threw, to whom, the best
    available alternative, the gap in expected points and in win probability with its
    range, every receiver's numbers, and the assumptions. ``apps/api`` serves these.

``reports/receiver_whatif_plays/{gameId}_{playId}.json``
    For each flagged throw, the simulated alternative (:func:`simulate.alternative`) in
    the Unity app's play schema via ``gridiron_tracking.hindsight``: the real play up to
    the throw, then the ball to the alternative receiver, ending when it arrives. Marked
    ``"simulated": true`` with ``simulatedFrom`` set to the real play, and carrying
    ``simulationStartFrameId`` so a client knows where measurement ends.

By default only the games the API is serving are written, because those are the plays
anyone can open, and a season of records is too large to commit. ``--all`` writes every
game to ``data/bdb2025/`` instead, which is git-ignored.
"""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd

from gridiron_tracking import hindsight

from . import bdb2025, simulate, whatif

PACKAGE = Path(__file__).resolve().parents[2]          # packages/counterfactual
REPO = PACKAGE.parents[1]
REPORTS = PACKAGE / "reports"
RECORDS = REPORTS / "receiver_whatif.jsonl"
PLAYS = REPORTS / "receiver_whatif_plays"

SERVED_PLAY_DIRS = [
    REPO / "data/unity-plays",
    REPO / "packages/replay-engine/hindsight/Assets/StreamingAssets/plays",
    REPO / "tests/fixtures/hindsight-plays",
]
"""Where the API looks for plays, in its order (apps/api/api/store.py)."""


def served_games() -> list[int]:
    """Game ids with an ``index.json`` in the first folder the API would serve from."""
    for folder in SERVED_PLAY_DIRS:
        indexes = sorted(folder.rglob("index.json")) if folder.is_dir() else []
        if indexes:
            return sorted({int(json.loads(p.read_text())["gameId"]) for p in indexes})
    return []


def play_state(game_id: int, play_id: int, line_of_scrimmage: float | None) -> dict:
    """The situation fields the Unity schema shows, from the release's plays and games."""
    p = bdb2025.plays()
    p = p[(p["gameId"] == game_id) & (p["playId"] == play_id)].iloc[0]
    g = bdb2025.games()
    g = g[g["gameId"] == game_id].iloc[0]
    state = {
        "homeTeam": g["homeTeamAbbr"], "visitorTeam": g["visitorTeamAbbr"],
        "description": "SIMULATED: the ball thrown to the model's alternative receiver. "
                       "This did not happen. Real play: " + str(p["playDescription"]),
        "quarter": int(p["quarter"]), "gameClock": p["gameClock"],
        "down": int(p["down"]), "yardsToGo": int(p["yardsToGo"]),
        "possessionTeam": p["possessionTeam"], "defensiveTeam": p["defensiveTeam"],
        "passResult": "",                      # the simulation stops before any result
    }
    if line_of_scrimmage is not None:
        state["lineOfScrimmage"] = round(float(line_of_scrimmage), 2)
        state["firstDownLine"] = round(min(float(line_of_scrimmage) + int(p["yardsToGo"]), 110.0), 2)
    return state


def _json_safe(record: dict) -> dict:
    return json.loads(json.dumps(record, default=lambda v: v.item() if hasattr(v, "item") else str(v)))


def records(scored: pd.DataFrame, decisions: pd.DataFrame) -> list[dict]:
    """Every throw in ``decisions`` as a what-if record, keyed like the API's play ids."""
    out = []
    by_play = {k: g for k, g in scored.groupby(bdb2025.KEY)}
    for d in decisions.itertuples(index=False):
        row = pd.Series(d._asdict())
        rec = whatif.describe(row, by_play[(row["gameId"], row["playId"])])
        rec.update(game_id=int(row["gameId"]), play_id_number=int(row["playId"]),
                   id=f"{int(row['gameId'])}_{int(row['playId'])}")
        out.append(_json_safe(rec))
    return out


def simulated_play(record: dict) -> dict | None:
    """The flagged throw's alternative as Unity play JSON, or None if it cannot be drawn."""
    if not record.get("alternative"):
        return None
    game_id, play_id = record["game_id"], record["play_id_number"]
    real = bdb2025.play_track(game_id, play_id)
    try:
        alt = simulate.alternative(real, record["player"]["player_id"], record["alternative"]["player_id"],
                                   what_if=record)
    except ValueError:            # the real play ended before the ball could have arrived
        return None
    doc = hindsight.to_play_json(alt, play_state(game_id, play_id, alt.meta.get("line_of_scrimmage")))
    start = next(i for i, f in enumerate(alt.frames) if abs(f.t - alt.meta["simulated_from_t"]) < 1e-6)
    doc.update(simulatedFrom=play_id, simulationStartFrameId=start + 1, receiverWhatIf=record)
    return doc


def write(scored: pd.DataFrame, decisions: pd.DataFrame, game_ids: list[int] | None,
          records_path: Path = RECORDS, plays_dir: Path = PLAYS, log=print) -> dict:
    """Write the records and the flagged throws' simulated plays. Returns what was written."""
    if game_ids is not None:
        decisions = decisions[decisions["gameId"].isin(game_ids)]
        scored = scored[scored["gameId"].isin(game_ids)]
    recs = records(scored, decisions)
    records_path.parent.mkdir(parents=True, exist_ok=True)
    records_path.write_text("".join(json.dumps(r) + "\n" for r in recs))

    plays_dir.mkdir(parents=True, exist_ok=True)
    for old in plays_dir.glob("*.json"):       # a rerun replaces, never accumulates
        old.unlink()
    drawn, skipped = 0, 0
    for rec in recs:
        if not rec["flagged"]:
            continue
        doc = simulated_play(rec)
        if doc is None:
            skipped += 1
            continue
        (plays_dir / f"{rec['id']}.json").write_text(json.dumps(doc))
        drawn += 1
    summary = {"throws": len(recs), "flagged": sum(r["flagged"] for r in recs),
               "simulated_plays": drawn, "could_not_simulate": skipped,
               "records": str(records_path), "plays": str(plays_dir)}
    log(json.dumps(summary, indent=2))
    return summary
