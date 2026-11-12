# Video analysis API

The FastAPI app is **Football Capstone API**. Video JSON uses the same Hindsight
schemaVersion 1 as `GET /plays/{id}`, produced by `hindsight.to_play_json`.
`GET /video/clips/{name}/track` and `/tracking` both return that play document:
`players[]`, `frames[].players[]`, integer `nflId`, `snapFrameId` and `frameRate`.
The internal `roster`/player maps are available only at `/track/raw`.

**Migration:** `/tracking` previously returned Big Data Bowl rows. Those rows now
live at `/tracking/rows`; the 18-column `/tracking.csv` export remains available.
Use `/tracking` for the replay frontend and `/review` for the play model.

Code is on `carrier-ball`, PR #10. Start a backend host after checking out that
branch; pushing the code does not provision a hosted API.

## Start the backend

From the repository root, with Python 3.11+:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r apps/api/requirements.txt
python -m pip install -e './packages/tracking[video,detect]'
python -m pip install -e ./packages/counterfactual
uvicorn api.main:app --app-dir apps/api --reload
```

Install FFmpeg for browser previews (`brew install ffmpeg` on macOS).
Open `http://localhost:8000/docs` for the API contract. Local frontend CORS is
supported. The two coordinate model files are committed under
`packages/counterfactual/models`; no Kaggle download is required to run a review.

Automatic field reading uses downloaded Qwen through MLX on Apple Silicon:

```bash
python -m pip install -e './packages/tracking[local-vision]'
python scripts/setup_local_vision.py
```

The local reader needs approximately 6 GB once. Reuse an existing download with
`FOOTBALL_LOCAL_MODELS_DIR` in `.env`. Pinned field calibration works without it.
Uploaded footage, detections and generated videos are local artifacts, not in Git.

## Try the checked example

```bash
curl http://localhost:8000/video/examples/nfl_sf_sea/tracking
curl http://localhost:8000/video/examples/nfl_sf_sea/tracking/rows
curl http://localhost:8000/video/examples/nfl_sf_sea/tracking.csv
curl http://localhost:8000/video/examples/nfl_sf_sea/tracking/metadata
```

These work on a fresh checkout without models or source footage. The committed
[play JSON](examples/nfl_sf_sea.play.json) is the replay format; the
[internal track](examples/nfl_sf_sea.track.json) retains diagnostic provenance.

This is an **assisted example**: field calibration, a full 22-athlete formation,
QB/RB identity corrections and throw/catch were reviewed from film. The review is
recorded in [the annotation](calibration/nfl_sf_sea-review.json). It contains:

- 22 roster IDs, 11 SF / 11 SEA; no births after the inspected formation.
- 97 output frames at 10 Hz, with the snap exactly at t=0.
- `pass_forward` at 3.2s and `pass_arrived` + `pass_outcome_caught` at 3.6s.
- Ball positions in 95 frames: 92 attributed carrier positions and 3 interpolated
  flight positions. Two frames remain unknown. This is not 95 visual detections.
- Seven actual source-image sightings at frames 1772–1778, preserved separately.

The broadcast crops players out. Mean observed roster is about 17 players per
frame; same-kit crossings remain uncertain. A 22-person roster does not establish
22 accurate continuous trajectories. Model results on this sample are a partial,
low-confidence demonstration, not validated football advice.

## Upload and analyze

| Step | Request | Response |
| --- | --- | --- |
| Upload | `POST /video/uploads`, multipart `file` | `videoId`, source URL, FPS, frame count, dimensions |
| Find plays | `POST /video/{videoId}/scan` | Job ticket with candidate windows |
| Poll | `GET /video/jobs/{jobId}` | `running`, `done`, `failed`; result/error |
| Create clip | `POST /video/clips` | Named clip spec |
| Inspect image | `GET /video/clips/{name}/frame?f={sourceFrame}` | JPEG with original pixel dimensions in headers |
| Auto-calibrate | `POST /video/clips/{name}/register` | Registration result; may need clicked points |
| Pin calibration | `POST /video/clips/{name}/pin` | Validated field homography |
| Review identities | `POST /video/clips/{name}/identities` | Save formation/identity anchors, then rerun |
| Run tracking | `POST /video/clips/{name}/run` | Job ticket; result has artifact URLs and readiness |
| Replay JSON | `GET /video/clips/{name}/tracking` or `/track` | Hindsight play JSON plus quality/readiness |
| Review a pass | `POST /video/clips/{name}/pass-review` | Completed-pass annotation; returns updated play JSON |
| Run model | `POST /video/clips/{name}/review` | Recommendation, notes and forecast in Hindsight format |
| Debug | `GET /video/clips/{name}/track/raw` | Original sparse tracking and evidence |
| Report | `GET /video/clips/{name}/report` | Geometry checks and assisted-ball summary |
| Replay video | `GET /video/clips/{name}/reconstruction.mp4` | Browser H.264 replay |
| Ball evidence | `GET /video/clips/{name}/ball-tracking.mp4` | Slow motion of actual visible flight, when available |

```json
{
  "videoId": "012345abcdef",
  "name": "practice_001",
  "startFrame": 120,
  "stopFrame": 420,
  "snapFrame": 210,
  "fullRosterFrame": 150,
  "level": "nfl",
  "playDirection": "left",
  "teamAClub": "SF",
  "teamBClub": "SEA"
}
```

Source frame indices are zero-based; `stopFrame` is exclusive. Scan `endFrame` is
inclusive, so add one. `fullRosterFrame` is a manually inspected frame with exactly
22 athlete detections. It enables fixed roster tracking: local IDs `p00..p21`
remain dormant across missed detections instead of spawning new players.
Visible player counts can be unequal in cropped frames. The anchor must really
contain 11 per team; choosing an arbitrary frame is not a roster check.

`level` accepts `nfl`, `college`, `high_school`. Name the actual `snapFrame` for
model analysis. Optional `gameId`, `playId`, `startTime` and club labels annotate
known metadata. Otherwise IDs are local/synthetic and timestamps remain unknown.
Team A is the lighter kit and B the darker kit. Existing clip names return 409.

For pinned registration, submit at least four (preferably six) real field/image
correspondences, spread across sidelines and yard lines:

```json
{"frame":210,"points":[
  {"field":[30,0],"pixel":[125,690]},
  {"field":[50,0],"pixel":[1100,690]},
  {"field":[30,53.3],"pixel":[380,140]},
  {"field":[50,53.3],"pixel":[970,140]}
]}
```

These only illustrate shape. Use actual image measurements. Convert display clicks
to original image pixels. A refused fit preserves the previous calibration.

To correct a same-kit crossing, supply reviewed source-frame field points:

```json
{"fullRosterFrame":150,"anchors":[
  {"frame":300,"playerId":"p02","field":[78.2,29.4]}
]}
```

POST this to `/identities`, then `/run`. Points are yards in the original calibrated
field orientation. An anchor must match an actual detection within three yards.
It is an identity correction, not permission to invent a position. Updated
identity configuration makes the old play endpoint return 409 until rerun.

## Sparse balls and assisted passes

Unreviewed clips may have only a few sightings. They remain replayable but return
`analysisReady:false` and actionable `analysisBlockers`. `/review` returns 422
until the required ball/pass inputs exist. A completed processing job does not
mean a model-ready play; check `hasTrack`, `analysisReady`, `analysisBlockers` and
`trackingQuality` before starting the model.

If a completed pass cannot be resolved automatically, review the film and POST:

```json
{"throwFrame":1769,"catchFrame":1780,"passerId":"p02","receiverId":"p07","centerId":"p18"}
```

This is the actual checked example annotation. Use each clip's own frame indices
and raw `trackId` values from `players[]`. A distinct passer and receiver on the
same team must have observed poses at their respective events. `centerId` is
optional; without it pre-snap ball positions stay unknown. A valid review saves
onto the clip spec and updates the JSON immediately. Rerun to update rendered
videos and raw CSV artifacts as well. Incomplete passes and interceptions are
not supported by this completed-pass annotation route.

The ball follows the reviewed center through the snap, the passer until release,
and receiver after catch, only while those poses exist. During flight its ground
position is linearly interpolated between those endpoints. Ball height is unknown;
this is not 3D flight reconstruction. Source-image pixels remain evidence, never
replaced with interpolated sightings. The timings are rounded to the 10 Hz export;
original source frames remain in provenance. Example review timing is approximate
within about two source frames.

## Frontend contract

- `players[]` is the roster. Key markers by `nflId`, not array index. `trackId` maps
  to raw video identity. `club` is SF/SEA when supplied; `side` is offense/defense.
- `frames[].players[]` contains only observed or briefly interpolated poses. Do not
  freeze missing players or create them at (0,0). Gaps over 0.25s stay unfilled.
- Time is `(frameId - snapFrameId) / frameRate`. Offense attacks +x; angles use
  NFL compass bearings (0 along +y, 90 along +x).
- `event` is the primary event for existing consumers; `events[]` preserves both
  arrival and caught events on the same frame.
- Optional `ball.dataOrigin` distinguishes `detected`, `carrier`, `coasted` and
  `interpolated`. `carrierNflId` explicitly identifies an attributed carrier.
  Hide the ball marker on missing-ball frames.
- `provenance.ball.image_observations` contains original pixel sightings indexed
  by source frame. Only these support rings on the actual footage. Ground-plane
  projection of an airborne pixel is not its real position at ground height.
- `trackingQuality` includes roster/visible counts, ball sources, missing players
  at throw/arrival, assistance and geometry checks. Geometry agreement does not
  establish identity accuracy.
- `/review` uses the real coordinate model and returns a low-confidence video
  recommendation with coverage caveats. The forecast retains original player IDs
  and snap clock. `simulationStartFrameId` marks where model output begins.
  It cannot model an athlete with no recent observed pose. Missing defenders can
  make a receiver appear more open than in reality.

Use `/tracking/rows` or `/tracking.csv` for the optional Big Data Bowl 2025
`tracking_week_*` 18-column interface, and `/tracking/metadata` for its identifier
map and sources. These retain the original field orientation. The normalized
Hindsight JSON is the frontend interface.

Analysis jobs run one at a time and live only in process memory; restart loses job
tickets. Join asset URLs to the backend origin. H.264 replay endpoints support
byte ranges. Missing FFmpeg reports unavailable previews without discarding tracks.
Only show assets returned by a successful job. A refused registration can complete
with `hasTrack:false`; runtime errors set the job to `failed`.

## Validation

```bash
python -m pytest -q
```

Tests cover the replay schema, sparse-ball rejection, reviewed event/carrier
provenance, missing poses, dormant identities, stable forecast IDs and actual
model inference on the checked video fixture. Source video rerenders are a
separate local check; detector/render calls are stubbed in routine API tests.
