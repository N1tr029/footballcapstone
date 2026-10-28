"""
Football Capstone API.

Serves Hindsight's play JSON at the exact paths its WebPlaySource requests, so the
Unity app switches from StreamingAssets to this backend by setting PlayLoader's
remote base URL to http://<host>:8000/plays. No Unity code changes.

Run from apps/api:   uvicorn api.main:app --reload
Interactive docs:    http://localhost:8000/docs
"""
import json
import sys
from typing import Optional

from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import FileResponse

from .schema import SCHEMA_VERSION, GameList, GameSummary, PlayDetail, PlayIndex
from .store import REPO_ROOT, PlayStore

# The repo's packages aren't installed during development (see pytest.ini), so make
# the play model importable from a bare checkout. recommendation.py is pure Python.
try:
    from gridiron_counterfactual.recommendation import from_report
except ImportError:
    sys.path.insert(0, str(REPO_ROOT / "packages" / "counterfactual"))
    from gridiron_counterfactual.recommendation import from_report

app = FastAPI(title="Football Capstone API", version="0.1.0")
# WebGL builds run in a browser on another origin, so they need CORS.
app.add_middleware(CORSMiddleware, allow_origins=["*"], allow_methods=["GET", "POST"], allow_headers=["*"])
# Play files are large and repetitive; gzip shrinks them several-fold.
app.add_middleware(GZipMiddleware, minimum_size=1024)

store = PlayStore()


def _index(game_id: Optional[int]) -> PlayIndex:
    if game_id is not None:
        if game_id not in store.games:
            raise HTTPException(404, f"No index for game {game_id}. Available: {sorted(store.games)}")
        return store.games[game_id]
    if not store.games:
        raise HTTPException(404, f"No index.json found under {store.plays_dir}. Run tools/export_plays.py first.")
    if len(store.games) > 1:
        raise HTTPException(400, f"Several games are loaded; pass ?gameId=. Available: {sorted(store.games)}")
    return next(iter(store.games.values()))


@app.get("/health")
def health():
    return {
        "status": "ok",
        "schemaVersion": SCHEMA_VERSION,
        "games": len(store.games),
        "plays": len(store.play_files),
        "reviews": len(store.reports),
        "simulated": len(store.simulated_of),
        "suggested": len(store.suggested),
    }


@app.get("/games", response_model=GameList)
def list_games():
    return GameList(games=[
        GameSummary(gameId=g.gameId, homeTeam=g.homeTeam, visitorTeam=g.visitorTeam, playCount=len(g.plays))
        for g in store.games.values()
    ])


@app.get("/plays", response_model=PlayIndex, summary="Play index for one game")
@app.get("/plays/index.json", response_model=PlayIndex, include_in_schema=False)
def get_index(gameId: Optional[int] = Query(None, description="Required only when several games are loaded")):
    # Served as written, like the play files. Re-serializing the validated model
    # would add every optional field the producer left out as null, and Unity's
    # JsonUtility cannot read null into an int or bool.
    return FileResponse(store.index_paths[_index(gameId).gameId], media_type="application/json")


@app.get("/plays/{play_id}/recommendation", summary="Play model's review of one play")
def get_recommendation(play_id: str):
    """The decision_engine report for this play, in the roles.md Recommendation
    shape (via gridiron_counterfactual.recommendation.from_report). Unset fields
    are explained in `assumptions`."""
    report = store.report(play_id)
    if report is None:
        raise HTTPException(404, f"No model review for play '{play_id}'. Reviewed plays: GET /recommendations")
    return from_report(report).to_json()


@app.get("/recommendations", summary="Plays the model has reviewed")
def list_recommendations():
    return {"plays": sorted(store.reports)}


def _strip_json(play_id: str) -> str:
    return play_id[: -len(".json")] if play_id.endswith(".json") else play_id


def _read(path) -> dict:
    with open(path, encoding="utf-8") as f:
        return json.load(f)


@app.get("/plays/{play_id}/optimal", summary="The model's simulated version of a play")
def get_optimal(play_id: str):
    """The play model's forecast of this play, in the same play JSON as the real one
    and flagged `"simulated": true`. It starts at the throw and shows where the model
    predicted everyone would be when the ball arrived.

    Simulated plays are produced offline by the play model:
    `python -m gridiron_counterfactual whatif <playId>` (or `--worst N`)."""
    play_id = _strip_json(play_id)
    if store.play_path(play_id) is None:
        raise HTTPException(404, f"Play '{play_id}' not found")
    path = store.optimal_path(play_id)
    if path is None:
        raise HTTPException(
            404,
            f"No simulated version of '{play_id}' yet. Plays that have one: GET /optimal. "
            f"Generate it with: python -m gridiron_counterfactual whatif {play_id}",
        )
    return FileResponse(path, media_type="application/json")


@app.get("/plays/{play_id}/suggested", summary="The model's suggested play, with ball, events and reason")
def get_suggested(play_id: str):
    """Landon's suggested play (decision_engine's suggested_plays.jsonl), converted
    to Hindsight play JSON. Unlike /optimal it carries:

    - `simulationStartFrameId`: the first frame that is the model's prediction;
      frames before it are recorded tracking.
    - `events` on frames: ball_snap and pass_forward (inferred), pass_arrived (simulated).
    - `ball` on every frame: a straight line from the passer to the recommended receiver.
    - `recommendation`: the decision, the recommended receiver (`targetNflId`), the
      receiver actually targeted, and the model's `reason` in plain English.

    Fields beyond schemaVersion 1 are ignored by Unity until PlayData declares them."""
    play_id = _strip_json(play_id)
    play = store.suggested.get(play_id)
    if play is None:
        raise HTTPException(
            404,
            f"No suggested play for '{play_id}'. Plays that have one: GET /suggested. "
            f"They come from running decision_engine.py, which writes {store.suggested_path.name}.",
        )
    return play


@app.get("/suggested", summary="Plays that have a suggested play, with the model's reason")
def list_suggested():
    return {"plays": [
        {
            "playId": pid,
            "decision": p["recommendation"]["decision"],
            "targetNflId": p["recommendation"]["targetNflId"],
            "reason": p["recommendation"]["reason"],
        }
        for pid, p in sorted(store.suggested.items())
    ]}


@app.get("/plays/{play_id}/compare", summary="The real play and the model's versions, in one response")
def compare(play_id: str):
    """Everything about one play in one call, for drawing them side by side.

    - `current`: the recorded play, or null if it was not exported.
    - `suggested`: the model's suggested play with ball, events, start frame and
      reason (see /plays/{id}/suggested), or null.
    - `optimal`: the whatif command's forecast of the play (see /plays/{id}/optimal), or null.
    - `recommendation`: the model's verdict in the roles.md shape, or null.

    404 only when the API knows nothing about the play at all."""
    play_id = _strip_json(play_id)
    current = store.play_path(play_id)
    optimal = store.optimal_path(play_id)
    suggested = store.suggested.get(play_id)
    report = store.report(play_id)
    if current is None and suggested is None:
        raise HTTPException(404, f"Play '{play_id}' not found")
    return {
        "playId": play_id,
        "hasCurrent": current is not None,
        "hasSuggested": suggested is not None,
        "hasOptimal": optimal is not None,
        "current": _read(current) if current else None,
        "suggested": suggested,
        "optimal": _read(optimal) if optimal else None,
        "recommendation": from_report(report).to_json() if report else None,
    }


@app.get("/optimal", summary="Plays that have a simulated version")
def list_optimal():
    return {"plays": [
        {"playId": real, "optimalPlayId": sim}
        for real, sim in sorted(store.simulated_of.items())
    ]}


@app.get(
    "/plays/{play_id}",
    response_model=PlayDetail,
    summary="One play with all tracking frames",
    responses={404: {"description": "Unknown play"}},
)
def get_play(play_id: str):
    # WebPlaySource requests "{id}.json"; accept both forms.
    play_id = _strip_json(play_id)
    path = store.play_path(play_id)
    if path is None:
        raise HTTPException(404, f"Play '{play_id}' not found")
    # Served as written by the exporter: no re-serialization of 2,000+ frames per request.
    return FileResponse(path, media_type="application/json")


@app.post("/admin/reload", summary="Re-scan the plays folder and model reports")
def reload_data():
    store.reload()
    return health()
