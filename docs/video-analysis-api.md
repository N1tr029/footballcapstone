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
`FOOTBALL_LOCAL_MODELS_DIR` in `.env`. Automatic identity tracking downloads
`sam2.1_t.pt` (about 75 MB) to `data/models` on its first run. Ultralytics is bounded
to the tested 8.4 API because SAM's unfiltered masks preserve identity through
occlusion. The trained carrier classifier is included in
`packages/tracking/models/carrier.joblib`; the counterfactual install supplies its
scikit-learn runtime.

OCR uses Apple Vision on macOS when `swiftc` is available, or optional Tesseract
(`brew install tesseract`). Jersey OCR can abstain without blocking tracking.
The local Qwen reader also checks formation crops for players versus officials.
Its weights are released before SAM starts, and detection automatically uses
CUDA or Metal when available. An explicit detector device still takes priority.
Field registration still needs readable field markings and a valid geometry fit.
Uploaded footage, model downloads, detections and generated videos are local
artifacts, not in Git.

## Automatic upload flow

Uploading starts analysis by default. The frontend does not have to supply field
points, snap timing, a formation frame, player corrections or pass annotations:

```bash
curl -F file=@play.mp4 http://localhost:8000/video/uploads
# {"videoId":"...", "analysisJobId":"...", "sourceUrl":"...", "media":{...}}
curl http://localhost:8000/video/jobs/JOB_ID
```

Poll the returned `analysisJobId`. A completed job has
`result.plays[]`, each with `clip`, `hasTrack`, `analysisReady`, `analysisBlockers`,
`trackingQuality` and `assets`. Fetch `assets.tracking` for replay JSON; call
`POST /video/clips/{clip}/review` when that play has `analysisReady:true`.
`progress.stage` and `progress.message` describe the current stage while running.
Join returned asset paths to the API origin.

`POST /video/{videoId}/analyze` starts another automatic job for an existing upload.
Use multipart `autoAnalyze=false` to upload without analysis and use the optional
assisted endpoints below. A job completing does not imply every candidate is
model-ready: unresolved footage returns blockers and any usable replay artifacts.

The automatic worker selects up to ten candidate field shots and bounds each
window to 20 seconds. It then:

- Reads field geometry with the local model. When a single view is unreadable,
  OCR can use yard numerals revealed later in the same camera shot, transfer them
  back through image registration, and fit measured paint intersections. An
  additional Apple Vision fallback fits word boxes on both painted number rows
  when long yard lines are not detected. Labels must agree across two source
  frames. An independent paint/turf check must accept the result.
- Finds the set-to-live motion transition for the snap. A clip already in motion
  does not receive a made-up snap.
- Chooses an early formation allowing for officials in the initial person count,
  checks uniform roles locally, and removes officials and sideline personnel.
  A tiled recheck of that one frame can recover smaller or overlapping athletes;
  it uses the same sideline cuts and excludes body crops whose feet are outside
  the image. Uncertain new crops cannot establish another identity. SAM 2 is
  prompted with the retained athletes. Empty masks keep their original IDs.
  A probable full formation uses an 11-per-team prior; a partial view permits
  unequal counts and unknown team labels. No team can gain extra players just
  to fill a nominal 22-person roster.
- Traces selected same-team crossings backward from later clear observations.
  Both traces must match the earlier known identities before a swap or gap is
  corrected. Failed matches remain unresolved. These checks are bounded to eight
  crossings per clip; they do not recover every occlusion or unseen entrant.
- When an individual identity is lost or drifts onto another body and is found
  clearly later, runs SAM backward with **only that player's box**. Three later
  frames must support a unique detector/mask match. The reverse trace must
  reconnect to that same athlete's earlier known box before its missing or wrong
  positions are repaired. Other player tracks are not rerun. Each separately
  supported disappearance of the same identity is checked in time order. Failed checks
  leave explicit replay estimates, not invented tracking observations.
- Looks for a moving ball flight and associates its extrapolated image endpoints
  with tracked bodies. At least six real source-frame sightings, fit, motion and
  endpoint checks are required. Accepted endpoints produce `pass_forward` and
  `pass_arrived`. Completion stays unknown; arrival does not create a caught event.
- Reads jersey crops across the entire play, including later clearer frames.
  At least three distinct frames must agree, with conflicting same-team claims
  rejected. Unreadable numbers remain unknown. OCR alone does not merge people.

Carried ball positions use the inferred center/passer/receiver while their poses
exist in the evidence. The replay keeps these carrier actors present through
tracking gaps and follows their estimated display positions. Flight positions are interpolated ground coordinates. They are estimates,
not extra visual detections or 3D flight reconstruction. Original image sightings
remain in provenance.

The replay now retains the same roster on every frame. A dropped detection no
longer removes its actor. If a player reappears at C after last being seen at A,
the missing B positions are interpolated from both ends. At an open-ended gap the
actor coasts briefly with damping, then retains its remembered position. Display
positions are smoothed and speed-bounded to avoid jitter and teleports. IDs, clubs
and team colors remain fixed. Each known roster actor has a coordinate on every
frame; an athlete never observed at all is reported explicitly instead of placed
at a made-up origin.

Original observations are retained separately from these display estimates.
Team-locked, one-to-one body matching rejects a SAM mask that drifts onto a
confidently opposing kit. When that happens, the replay fills the gap instead of
moving that ID onto the opponent. The coordinate model continues to use the
original observation subset, so constant replay membership does not claim all
22 athletes are currently measured.

This is batch processing. On an M3 Pro, the uncached 22-object SAM pass for the
9.7-second reference clip took about 20 minutes. The subsequent fully automatic
run, reusing detector/SAM results but computing field calibration, took 269 seconds.
These are local measurements, not an upload latency guarantee. Jobs are serialized
to limit memory use. Longer videos, unfamiliar camera angles, crowded piles,
missing paint and off-camera athletes remain material limits.

To exercise real upload jobs on additional footage and save JSON, previews and
roster/recovery audits:

```bash
python scripts/validate_video_uploads.py --base-url http://localhost:8000 play1.mp4 play2.mp4
```

The script supplies no calibration, snap, identity or pass annotations. It rejects
completed jobs without usable tracks as well as roster changes, team changes,
nonfinite positions, over-capacity rosters and display teleports. These checks
establish output consistency; footage still needs visual inspection for identity
accuracy. Reports and downloaded previews are under `data/runs/upload-validation`.

## Try the examples

The automatic example uses the same video window with no clicked calibration,
annotated snap, checked formation, identity corrections or pass labels:

```bash
curl http://localhost:8000/video/examples/nfl_sf_sea_auto/tracking
curl http://localhost:8000/video/examples/nfl_sf_sea_auto/tracking/metadata
```

The committed [automatic play JSON](examples/nfl_sf_sea_auto.play.json) and
[automatic internal track](examples/nfl_sf_sea_auto.track.json) retain the evidence:

- The same 22 roster IDs on every replay frame, 11 per kit. IDs are local/synthetic.
  The stricter body checks retain about 14 observations per frame; the remainder
  are explicitly estimated replay positions.
- 97 output frames at 10 Hz; snap inferred at source frame 1672.
- Throw inferred at source 1764 (3.1s), arrival at 1779 (3.6s), completion unknown.
- Seven visible source-frame ball sightings, 1772–1778. The exported 97 ball poses
  consist of 93 attributed carrier positions and four interpolated flight poses.
- Eight backward checks: seven matched their earlier seams, one abstained.
  This sample required no swap/gap edits from those checks; it does not establish
  identity accuracy for other footage.
- No jersey numbers passed temporal consensus on this broadcast. They remain
  unknown instead of being assigned guesses.
- The real play model runs on 13 recently observed players at release and returns
  a low-confidence video recommendation. Missing defenders still affect it.

These are results on one clip, not a broad accuracy benchmark. Detector agreement
is about 97.5% after exclusive body matching and individual recovery (previously 81.5%); that measures
geometry overlap, not identity correctness. A fixed roster
does not establish 22 continuously accurate trajectories.

A second automatic example from Northwestern/Michigan is also available:

```bash
curl http://localhost:8000/video/examples/nw_michigan_auto/tracking
```

It retains 21 identities on all 65 frames, with 1,095 observed and 270 estimated
player samples. Nine individual backward checks recover 69 missing source
samples and correct 11 drifted ones, including two separate gaps for `p15`.
One kit remains unknown. This clip has no accepted ball flight and remains
`analysisReady:false`; successful player recovery does not establish pass events.
The [raw track](examples/nw_michigan_auto.track.json) and
[play JSON](examples/nw_michigan_auto.play.json) are genuine automatic outputs.

The older assisted example remains available for comparison:

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
- The original internal track has ball positions in 95 frames: 92 attributed
  carrier positions and three interpolated flight positions. The continuous replay
  also fills its two display gaps. These are estimates, not visual detections.
- Seven actual source-image sightings at frames 1772–1778, preserved separately.

The broadcast crops players out. Mean observed roster is about 17 players per
frame; same-kit crossings remain uncertain. A 22-person roster does not establish
22 accurate continuous trajectories. Model results on this sample are a partial,
low-confidence demonstration, not validated football advice.

## Optional assisted endpoints

| Step | Request | Response |
| --- | --- | --- |
| Upload | `POST /video/uploads`, multipart `file` | `videoId`, `analysisJobId`, source URL and media metadata; `autoAnalyze=false` opts out |
| Analyze automatically | `POST /video/{videoId}/analyze` | Job ticket; no annotations required |
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
| Report | `GET /video/clips/{name}/report` | Geometry checks and ball provenance summary |
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

This JSON is an assisted `POST /video/clips` request, not the automatic upload
requirement. Omitting `fullRosterFrame` selects automatic SAM tracking. Source
frame indices are zero-based; `stopFrame` is exclusive. Scan `endFrame` is
inclusive, so add one. When supplied, `fullRosterFrame` is an inspected frame with exactly
22 athlete detections. It enables fixed roster tracking: local IDs `p00..p21`
remain dormant across missed detections instead of spawning new players.
Visible player counts can be unequal in cropped frames. The anchor must really
contain 11 per team; choosing an arbitrary frame is not a roster check.

`level` accepts `nfl`, `college`, `high_school`. Automatic uploads infer the snap;
an assisted request can supply the actual `snapFrame`. Optional `gameId`, `playId`, `startTime` and club labels annotate
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

## Unresolved footage and assisted passes

Clips whose snap/pass cannot be inferred may have only a few sightings. They remain replayable but return
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
original source frames remain in provenance. Reviewed timing remains an estimate,
not synchronized ground-truth tracking.

## Frontend contract

- `players[]` is the roster. Key markers by `nflId`, not array index. `trackId` maps
  to raw video identity. `club` is SF/SEA when supplied; `side` is offense/defense.
- `jerseyNumberKnown:false` means the number is unknown. The integer
  `jerseyNumber:0` is then a compatibility placeholder; hide it. An accepted actual
  number 0 has `jerseyNumberKnown:true`.
- `frames[].players[]` retains the same known roster IDs on every frame, in roster
  order. Create actors once and key them by `nflId`; do not renumber or recolor by
  array index. Each sample also carries its fixed `club` and `side`.
- `observed:true` identifies an original video-tracking sample. Its displayed
  coordinates are smoothed, while `observation` preserves the original sample.
  `observed:false` means a display estimate. `trackingStatus` is `interpolated`,
  `predicted`, `last_known` or `backfilled`; forecast samples use `simulated`.
  Keep the actor visible and optionally draw it hollow/faded while estimated.
  `positionAgeSeconds` records the distance in time to its remembered observation.
- The internal `/track/raw` and BDB exports remain observation data, with sparse
  positions and short-gap interpolation. `hindsight.read_play` restores the original
  samples from a stabilized document for model analysis. Filled replay markers do
  not become fresh evidence. Never-observed roster IDs appear in
  `provenance.replayStability.neverObservedIds`; no coordinate is invented for them.
- Time is `(frameId - snapFrameId) / frameRate`. Offense attacks +x; angles use
  NFL compass bearings (0 along +y, 90 along +x).
- `event` is the primary event for existing consumers; `events[]` preserves
  co-occurring events. Automatic arrival does not imply a caught event.
- Optional `ball.dataOrigin` distinguishes `detected`, `carrier`, `coasted` and
  `interpolated`. `carrierNflId` explicitly identifies an attributed carrier.
  Accepted pass replays keep the ball with the continuous carrier/flight path.
  A display-filled ball stores its original sample (or null) in `ball.observation`.
  Hide the marker if no pass/ball estimate is available.
- `provenance.ball.image_observations` contains original pixel sightings indexed
  by source frame. Only these support rings on the actual footage. Ground-plane
  projection of an airborne pixel is not its real position at ground height.
- `trackingQuality` includes roster/visible counts, ball sources, missing players
  at throw/arrival, assistance, `identityAutomatic`, `backwardChecks`,
  `jerseyNumbersFound` and geometry checks. Geometry agreement does not establish
  identity accuracy. `provenance.automaticPass` records source frames, endpoint
  errors, heuristic confidence score and unknown outcome. Its score is not a
  calibrated probability. `provenance.identity.backwardChecks` records each seam
  decision and any actual edits.
- `provenance.identity.reacquisitionChecks` records each single-player backward
  pass: player ID, earlier/later source frames, supporting later sightings, seam
  agreement, recovered/corrected frames and any rejection reason. The
  `trackingQuality.reacquisitionChecks` count is separate from paired crossing
  checks. This recovers observations from video rather than merely filling a
  display path between two points.
- `replayRosterStable`, `replayPlayersPerFrame` and `replayEstimatedSamples` describe
  continuous replay membership. `ballFrames`/`ballSources` describe replay output;
  `analysisBallFrames`/`analysisBallSources` retain the counts before display
  filling. `meanObservedPlayers` and `missingAtThrow` still describe actual analysis
  coverage, not the number of visible replay actors.
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

Tests cover the replay schema, automatic job orchestration without annotations,
snap/flight evidence rejection, later-to-earlier identity repair and abstention,
temporal jersey consensus, calibration reference frames, missing poses, stable
forecast IDs and actual model inference on both video fixtures. Source video
rerenders are a separate local check; detector/render calls are stubbed in routine
API tests.

The local end-to-end check also uploaded a lossless 290-frame copy of the reference
window through FastAPI's HTTP test client, ran the real automatic worker, fetched
the Hindsight document and called `/review` with the real coordinate model. It
completed with `analysisReady:true` and no processing problems. All uploaded pixels
were verified against the source before reusing genuine detector/SAM inference
caches; calibration, snap/pass inference, OCR, exports and model review ran on the
upload. This cache-assisted integration check is separate from the uncached SAM
timing above.
