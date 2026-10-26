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
| GET | `/health` | Counts of games, plays, reviews, simulated and suggested plays loaded |
| GET | `/games` | Games with an index loaded |
| GET | `/plays` (also `/plays/index.json`) | Play index for one game; pass `?gameId=` when several are loaded |
| GET | `/plays/{id}` (also `/plays/{id}.json`) | One play with all frames, exactly as exported |
| GET | `/plays/{id}/suggested` | The model's suggested play, with start frame, events, ball and reason (see below), or 404 |
| GET | `/suggested` | Every play with a suggested play, with the model's decision, recommended receiver and reason |
| GET | `/plays/{id}/optimal` | The model's simulated version of the play (same play JSON, `"simulated": true`), or 404 with how to generate it |
| GET | `/plays/{id}/compare` | `{ playId, hasCurrent, hasSuggested, hasOptimal, current, suggested, optimal, recommendation }`: everything about one play in one call |
| GET | `/optimal` | Every play that has a simulated version, as `{ playId, optimalPlayId }` |
| GET | `/plays/{id}/recommendation` | The play model's review of that play, or 404 |
| GET | `/recommendations` | IDs of every reviewed play |
| POST | `/admin/reload` | Re-scan files after re-exporting or re-running the model |

Play IDs are `{gameId}_{playId}`, e.g. `2022090800_56`.

## Configuration

| Variable | Default |
| --- | --- |
| `PLAYS_DIR` | The first of `data/unity-plays/`, `packages/replay-engine/hindsight/Assets/StreamingAssets/plays`, `tests/fixtures/hindsight-plays` that holds an `index.json` |
| `DECISIONS_PATH` | `packages/counterfactual/reports/decisions.jsonl` |
| `SUGGESTED_PATH` | `packages/counterfactual/reports/suggested_plays.jsonl` |

The `PLAYS_DIR` fallback means the API keeps working when the Unity app moves to
its own repository: `whatif` then writes to `data/unity-plays/`, and the real
plays are pinned under `tests/fixtures/hindsight-plays/`.

`PLAYS_DIR` is scanned recursively. The exporter clears its output folder on
every run, so to serve several games, export each into its own subfolder.

## Suggested plays

`decision_engine.py` writes `reports/suggested_plays.jsonl`: per play, the recorded
frames up to the throw, then the model's predicted frames, a ball, events, and the
reason for its recommendation. `api/suggested.py` converts each line to Hindsight
play JSON, flipping plays that go left the same way Hindsight's exporter does.
Besides the usual fields, a suggested play has:

| Field | Meaning |
| --- | --- |
| `simulationStartFrameId` | First frame that is the model's prediction. Earlier frames are recorded tracking. |
| `sourceFrameId` | The last recorded frame the prediction starts from. |
| `frames[].event` / `frames[].events` | `ball_snap`, `pass_forward` (inferred), `pass_arrived` (simulated). |
| `frames[].ball` | On every frame. `dataOrigin` says how: the passer's position before the throw, then a straight line to the recommended receiver. |
| `recommendation.targetNflId` | The receiver the model recommends. `targetedNflId` is who actually got the ball. |
| `recommendation.reason` | The model's explanation in plain English. |
| `players[].oSource` on frames | `measured` on recorded frames, `assumed` on predicted ones (body angle is carried from the last recorded frame). |

Unity ignores fields beyond schemaVersion 1 until `PlayData` declares them.
The source data has no jersey numbers, team abbreviations, quarter or clock:
clubs are `OFF` and `DEF`, numbers are 0. Down and distance come from
`decisions.jsonl` when it has the play.

To produce the file, run `decision_engine.py` (Landon's README has the command)
and copy `reports/suggested_plays.jsonl` to `SUGGESTED_PATH`, then `POST /admin/reload`.

## Current play vs. the model's version

`/plays/{id}` is the play as recorded. `/plays/{id}/optimal` is the play model's
forecast of the same play: it starts at the throw and shows where the model
predicted everyone would be when the ball arrived. It is not a different play
call drawn from scratch. It is the model's expectation of this one, and its
verdict on whether another receiver was more open comes from the same run.

Simulated plays are generated offline, not per request:

```
python -m gridiron_counterfactual whatif 2022090800_56   # one play
python -m gridiron_counterfactual whatif --worst 5       # the five the model most disputes
```

then `POST /admin/reload` (or restart) to pick them up.

## Known gaps

Hindsight exports Big Data Bowl **2025** (2022 season) and the play model reviews
Big Data Bowl **2026** (2023 season), so no play currently has both a replay and
a review. `/recommendations` lists what is reviewed. Lining these up means one
side moving to the other's release.

`suggested_plays.jsonl` is not in the repository yet: `suggested_play_rows` and
`build_suggested_play_payload` are in Landon's ML repository, newer than the copy
in `packages/counterfactual`, and the file is produced by running them on the
Kaggle data. Until it exists, `/suggested` is empty and `/plays/{id}/suggested`
returns 404. The tests run the conversion on output from Landon's own code
(`tests/fixtures_suggested.jsonl`).

## Tests

From the repo root: `pytest apps/api/tests`. The suite builds its own fixture
files, plus one test that loads the real exported plays when they are present.
