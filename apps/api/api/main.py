"""
Football Capstone API.

Serves Hindsight's play JSON at the exact paths its WebPlaySource requests, so the
Unity app switches from StreamingAssets to this backend by setting PlayLoader's
remote base URL to http://<host>:8000/plays. No Unity code changes.

Run from apps/api:   uvicorn api.main:app --reload
Interactive docs:    http://localhost:8000/docs
"""
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
    return _index(gameId)


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


@app.get(
    "/plays/{play_id}",
    response_model=PlayDetail,
    summary="One play with all tracking frames",
    responses={404: {"description": "Unknown play"}},
)
def get_play(play_id: str):
    # WebPlaySource requests "{id}.json"; accept both forms.
    if play_id.endswith(".json"):
        play_id = play_id[: -len(".json")]
    path = store.play_path(play_id)
    if path is None:
        raise HTTPException(404, f"Play '{play_id}' not found")
    # Served as written by the exporter: no re-serialization of 2,000+ frames per request.
    return FileResponse(path, media_type="application/json")


@app.post("/admin/reload", summary="Re-scan the plays folder and model reports")
def reload_data():
    store.reload()
    return health()
