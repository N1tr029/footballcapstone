"""The receiver what-if over HTTP.

A FastAPI ``router`` that serves what ``python -m gridiron_counterfactual.receiver export``
writes, and a standalone ``app`` that serves only that router. It changes nothing in
``apps/api``; it is written so ``apps/api`` can take it with one line:

    from gridiron_counterfactual.receiver.api import router as receiver_router
    app.include_router(receiver_router)

Until then it runs on its own, beside the main API:

    uvicorn gridiron_counterfactual.receiver.api:app --port 8001     # docs at /docs

Play ids are the API's own, ``{gameId}_{playId}``, so a play from ``apps/api``'s
``/plays/{id}`` can be looked up here unchanged. Paths are overridable with
``RECEIVER_WHATIF_PATH`` and ``RECEIVER_WHATIF_PLAYS``.
"""

from __future__ import annotations

import json
import logging
import os
import re
from pathlib import Path

from fastapi import APIRouter, FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse

from .export import PLAYS, RECORDS

log = logging.getLogger("receiver.api")
PLAY_FILE = re.compile(r"^(\d+)_(\d+)\.json$")


class ReceiverStore:
    """The exported records and simulated plays, read from disk and swapped in whole."""

    def __init__(self, records_path: str | Path | None = None, plays_dir: str | Path | None = None):
        self.records_path = Path(records_path or os.environ.get("RECEIVER_WHATIF_PATH") or RECORDS)
        self.plays_dir = Path(plays_dir or os.environ.get("RECEIVER_WHATIF_PLAYS") or PLAYS)
        self.records: dict[str, dict] = {}
        self.plays: dict[str, Path] = {}
        self.reload()

    def reload(self) -> None:
        records: dict[str, dict] = {}
        if self.records_path.is_file():
            for n, line in enumerate(self.records_path.read_text(encoding="utf-8").splitlines(), 1):
                if not line.strip():
                    continue
                try:
                    r = json.loads(line)
                    records[r["id"]] = r
                except (ValueError, KeyError) as e:        # one bad line must not empty the rest
                    log.warning("skipping line %d of %s: %s", n, self.records_path, e)
        plays = {p.stem: p for p in sorted(self.plays_dir.glob("*.json")) if PLAY_FILE.match(p.name)} \
            if self.plays_dir.is_dir() else {}
        self.records, self.plays = records, plays


store = ReceiverStore()
router = APIRouter(tags=["receiver what-if"])


def _strip_json(play_id: str) -> str:
    return play_id[: -len(".json")] if play_id.endswith(".json") else play_id


def _summary(play_id: str, r: dict) -> dict:
    alt = r.get("alternative") or {}
    wp = r.get("win_prob_gap") or {}
    return {
        "playId": play_id, "flagged": r["flagged"], "passer": r["player"]["name"],
        "chosen": r["chosen"]["name"], "alternative": alt.get("name"),
        "gapPoints": r["epa_gap"]["expected"], "gapWinProbability": wp.get("expected"),
        "hasSimulated": play_id in store.plays,
    }


@router.get("/receiver-whatif", summary="Throws the receiver what-if scored, flagged ones first")
def list_receiver_whatif(flagged: bool = Query(False, description="Only throws flagged as a likely better option")):
    """Every scored throw, largest flagged gap first. A flag means "likely a better
    receiver was available", not a proven mistake: see each record's `assumptions`."""
    rows = [_summary(pid, r) for pid, r in store.records.items() if r["flagged"] or not flagged]
    rows.sort(key=lambda x: (not x["flagged"], -x["gapPoints"]))
    return {"plays": rows}


@router.get("/plays/{play_id}/receiver-whatif", summary="The receiver what-if's verdict on one throw")
def get_receiver_whatif(play_id: str):
    """Who threw, to whom, the best available alternative receiver, the gap in expected
    points and win probability with its range, every receiver's numbers, and the
    assumptions behind it."""
    play_id = _strip_json(play_id)
    record = store.records.get(play_id)
    if record is None:
        raise HTTPException(404, f"No receiver what-if for '{play_id}'. Scored throws: GET /receiver-whatif. "
                                 f"Generate with: python -m gridiron_counterfactual.receiver export")
    return {**record, "hasSimulated": play_id in store.plays}


@router.get("/plays/{play_id}/receiver-whatif/simulated", summary="The throw to the alternative, as a play")
def get_receiver_simulated(play_id: str):
    """The real play up to the throw, then the ball to the alternative receiver, ending when
    it arrives, in the Unity app's play JSON (the same shape as the main API's /plays/{id}).
    Players after the throw are still the real tracking; only the ball is simulated. Marked
    `"simulated": true`, with `simulationStartFrameId` where measurement ends and the record
    under `receiverWhatIf`."""
    play_id = _strip_json(play_id)
    path = store.plays.get(play_id)
    if path is None:
        raise HTTPException(404, f"No simulated throw for '{play_id}'. Only flagged throws have one, and only "
                                 f"when the real play lasted long enough for the ball to arrive: "
                                 f"GET /receiver-whatif?flagged=true")
    return FileResponse(path, media_type="application/json")


@router.post("/receiver-whatif/reload", summary="Re-read the exported records and plays")
def reload_receiver_whatif():
    store.reload()
    return {"records": len(store.records), "simulatedPlays": len(store.plays)}


# Standalone, for running beside apps/api until it includes the router.
app = FastAPI(title="Receiver what-if", version="0.1.0")
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
app.add_middleware(GZipMiddleware, minimum_size=1024)
app.include_router(router)


@app.get("/health")
def health():
    return {"status": "ok", "records": len(store.records), "simulatedPlays": len(store.plays)}
