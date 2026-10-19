# apps/api

HTTP service for the Hindsight replay app. It serves the files other streams
already write and adds no data of its own:

- **Plays**: Hindsight's play JSON (schemaVersion 1, see
  `packages/replay-engine/hindsight` README), from
  `Assets/StreamingAssets/plays/`.
- **Model reviews**: `packages/counterfactual/reports/decisions.jsonl`, converted
  to the roles.md `Recommendation` shape by
  `gridiron_counterfactual.recommendation.from_report`.

## Run

```
cd apps/api
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload
```

Open http://localhost:8000/docs to try every endpoint.

## Point Hindsight at it

No Unity code changes. On the scene's `PlayLoader`, set **Source** to `Remote`
and **Remote Base Url** to `http://localhost:8000/plays`. `WebPlaySource` then
requests `/plays/index.json` and `/plays/{id}.json`, which this API serves.

In a WebGL build the page runs on another origin; CORS is open, so that works.

## Endpoints

| Method | Path | Returns |
| --- | --- | --- |
| GET | `/health` | Counts of games, plays, and reviews loaded |
| GET | `/games` | Games with an index loaded |
| GET | `/plays` (also `/plays/index.json`) | Play index for one game; pass `?gameId=` when several are loaded |
| GET | `/plays/{id}` (also `/plays/{id}.json`) | One play with all frames, exactly as exported |
| GET | `/plays/{id}/recommendation` | The play model's review of that play, or 404 |
| GET | `/recommendations` | IDs of every reviewed play |
| POST | `/admin/reload` | Re-scan files after re-exporting or re-running the model |

Play IDs are `{gameId}_{playId}`, e.g. `2022090800_56`.

## Configuration

| Variable | Default |
| --- | --- |
| `PLAYS_DIR` | `packages/replay-engine/hindsight/Assets/StreamingAssets/plays` |
| `DECISIONS_PATH` | `packages/counterfactual/reports/decisions.jsonl` |

`PLAYS_DIR` is scanned recursively. The exporter clears its output folder on
every run, so to serve several games, export each into its own subfolder.

## Known gap

Hindsight exports Big Data Bowl **2025** (2022 season) and the play model reviews
Big Data Bowl **2026** (2023 season), so no play currently has both a replay and
a review. `/recommendations` lists what is reviewed. Lining these up means one
side moving to the other's release.

## Tests

From the repo root: `pytest apps/api/tests`. The suite builds its own fixture
files, plus one test that loads the real exported plays when they are present.
