# Video analysis API for the frontend

The FastAPI app exposes the footage pipeline under `/video`. Its public tracking
output matches the **Big Data Bowl 2025 `tracking_week_*.csv` columns**, as requested
for [this archive](https://www.kaggle.com/datasets/alexandermeau/nfl-big-data-bowl-archived-data-2025).
Start with an upload, choose a play window, calibrate the field, then run analysis and poll its job.
OpenAPI is available at `http://localhost:8000/docs`.

This code is on the `carrier-ball` branch. Fetching `main` alone will not include
the video API until that branch is merged. A Git push publishes the backend code;
it does not provision a hosted server. Start the service below on the backend host.

## Start the backend

From the repository root, using Python 3.11 or newer:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r apps/api/requirements.txt
python -m pip install -e './packages/tracking[video,detect]'
uvicorn api.main:app --app-dir apps/api --reload
```

Install `ffmpeg` on the backend host for H.264 browser previews. On macOS:

```bash
brew install ffmpeg
```

Automatic field reading uses downloaded Qwen through MLX on Apple Silicon:

```bash
python -m pip install -e './packages/tracking[local-vision]'
python scripts/setup_local_vision.py
```

This downloads approximately 6 GB once. An existing model directory can be reused
through `FOOTBALL_LOCAL_MODELS_DIR` in `.env`; see
[`packages/tracking/README.md`](../packages/tracking/README.md). Pinned field
calibration works without the local reader. Model weights, uploads, cached
detections and generated videos are local artifacts and are not included in Git.

For UI development without models or video processing, start the server and use:

```bash
curl http://localhost:8000/video
curl http://localhost:8000/video/examples/nfl_sf_sea/tracking.csv
curl http://localhost:8000/video/examples/nfl_sf_sea/tracking
curl http://localhost:8000/video/examples/nfl_sf_sea/tracking/metadata
```

The committed [CSV](examples/nfl_sf_sea.tracking.csv) and
[JSON rows](examples/nfl_sf_sea.tracking.json) contain the checked video output.
These example endpoints work on a fresh checkout, with no uploaded film or model
download. [`nfl_sf_sea.track.json`](examples/nfl_sf_sea.track.json) is the internal
debugging representation; it is not the Big Data Bowl interface.

## Request sequence

All paths below are relative to the backend origin, usually
`http://localhost:8000`. CORS permits GET and POST from a separate local frontend.

| Step | Request | Response/use |
| --- | --- | --- |
| Upload | `POST /video/uploads`, multipart field `file` | `videoId`, `sourceUrl`, `media` with FPS, frame count, width, height and duration |
| Preview source | `GET {sourceUrl}` | Uploaded source video; browser support depends on its original codec |
| Find plays | `POST /video/{videoId}/scan` | `{jobId}`; poll to get ranked candidate shots |
| Poll | `GET /video/jobs/{jobId}` | `state`: `running`, `done` or `failed`; completed result is in `result`, failure text in `error` |
| Create clip | `POST /video/clips` | Saves a named window and returns `clip` and `next`; example body below |
| List clips | `GET /video/clips` | `clip`, title, registration state, frame window and whether a track exists |
| Calibration image | `GET /video/clips/{name}/frame?f={sourceFrame}` | JPEG; `X-Frame`, `X-Width`, `X-Height` headers describe the original coordinates |
| Auto-calibrate | `POST /video/clips/{name}/register` | `registered`, `frame`, verdict and paint score; may take minutes while the local model reads the frame |
| Click calibration | `POST /video/clips/{name}/pin` | `registered`, `paintAlignment`, acceptance `bar` and `next` |
| Analyze | `POST /video/clips/{name}/run` | `{jobId, clip}`; poll for `hasTrack`, quality metadata and artifact URLs |
| Tracking rows | `GET /video/clips/{name}/tracking` | Array of Big Data Bowl 2025 tracking records |
| Tracking CSV | `GET /video/clips/{name}/tracking.csv` | The exact 18-column Big Data Bowl tracking header |
| Export provenance | `GET /video/clips/{name}/tracking/metadata` | Synthetic identity map, missing metadata and observation sources |
| Internal debug track | `GET /video/clips/{name}/track` | Internal `PlayTrack` JSON, retained for diagnostics |
| Report | `GET /video/clips/{name}/report` | Plain-text geometry/tracking checks |
| Replay | `GET /video/clips/{name}/reconstruction.mp4` | H.264 source overlay and top-down reconstruction |
| Ball preview | `GET /video/clips/{name}/ball-tracking.mp4` | H.264 slow-motion crop, when visible ball flight was found |

Shot `endFrame` is inclusive; clip `stopFrame` is exclusive. Set
`stopFrame = candidate.endFrame + 1`. All source frame indices are zero-based.
Clip names accept letters, numbers, `_` and `-`, up to 80 characters. Creating an
existing name returns 409 and preserves its calibration.

```json
{
  "videoId": "012345abcdef",
  "name": "practice_001",
  "startFrame": 120,
  "stopFrame": 420,
  "snapFrame": 210,
  "title": "Practice play 1",
  "level": "nfl",
  "playDirection": "left",
  "teamAClub": "SF",
  "teamBClub": "SEA"
}
```

`level` accepts `nfl`, `college` or `high_school`. `snapFrame` is optional; when
omitted the clip start is used as time zero. Select a single short play: rendering
decodes its frames into memory. Shot scanning streams the full upload.
`gameId`, `playId` and `startTime` are optional verified metadata; `startTime`
timestamps the first exported frame. If identifiers are not supplied, local
synthetic game/play IDs are generated and declared in the metadata endpoint.
Unknown names, jersey numbers and wall-clock time remain unknown. Club inputs map
the lighter (`teamA`) and darker (`teamB`) kit groups to the supplied abbreviations.

When auto-calibration returns `registered: false`, show its `problems` and the
calibration image. Submit at least four well-spread field correspondences to
`/pin`, preferably six across both sidelines and several yard lines:

```json
{
  "frame": 210,
  "points": [
    {"field": [30, 0], "pixel": [125, 690]},
    {"field": [50, 0], "pixel": [1100, 690]},
    {"field": [30, 53.3], "pixel": [380, 140]},
    {"field": [50, 53.3], "pixel": [970, 140]}
  ]
}
```

These illustrate the request shape; use measurements from the actual image.
Convert canvas clicks to original image pixels before submitting. A refused fit
does not overwrite the saved homography. The same paint gate applies to automatic
and clicked calibration.

The current development server runs one analysis job at a time to limit memory.
Jobs are held in process memory and are lost on restart. Treat `done` as job
completion, then check `result.hasTrack` and `result.problems`: a rejected
registration can complete normally with `hasTrack: false`. A thrown runtime error
sets the job to `failed`. Only show video links present in `result.assets`:

```json
{
  "clip": "practice_001",
  "hasTrack": true,
  "assets": {
    "tracking": "/video/clips/practice_001/tracking",
    "trackingCsv": "/video/clips/practice_001/tracking.csv",
    "trackingMetadata": "/video/clips/practice_001/tracking/metadata",
    "track": "/video/clips/practice_001/track",
    "reconstruction": "/video/clips/practice_001/reconstruction.mp4",
    "ballPreview": "/video/clips/practice_001/ball-tracking.mp4"
  }
}
```

The actual result also contains `problems`, `stages`, `seconds`, `teams` and
`ball`. Missing FFmpeg leaves tracking available and reports unavailable browser
previews. Join artifact paths to the backend origin. Replay endpoints support
byte-range requests for seeking.

## Big Data Bowl tracking format

Both public endpoints return one row per player/frame, plus a football row when
a ball position exists. CSV has exactly this ordered header; JSON has the same keys:

```csv
gameId,playId,nflId,displayName,frameId,frameType,time,jerseyNumber,club,playDirection,x,y,s,a,dis,o,dir,event
```

`frameId` is one-based and contiguous at 10 Hz. `frameType` is `BEFORE_SNAP`,
`SNAP` or `AFTER_SNAP`. The ball has `displayName=football`, `club=football` and an
empty/null `nflId`. Missing balls produce no football row. `o` and `dir` use the
NFL compass convention: **0 along +y, 90 along +x**, clockwise. `dis` is actual
distance between consecutive observed positions, not `s/10`; it is null after a
gap. The timestamp is null unless a real start time is supplied. Null CSV fields
are empty cells. All extra quality and provenance information lives outside the
18 columns in the metadata response.

Coordinates are yards in the calibrated field orientation; `playDirection`
records the offense's actual direction. For NGS imports that were normalized by
our reader, the exporter restores the original field orientation. The example
video attacks left and has SF/SEA club labels. Its game/play IDs and tracked player
IDs are explicitly synthetic; this does not identify the actual NFL athletes.
This interface reproduces the tracking-file schema, not the full archive's game,
play, player biography and annotation tables.

## Internal debugging format

`roster` maps identity IDs to entries containing `team`. `frames` contains time
`t` in seconds from the snap, a sparse `players` map, optional `ball` and events.
The internal angles use 0 along +x and differ from the public NFL rows above.
Tracking is sampled at 10 Hz. For display, interpolate positions only between
adjacent frames containing the same identity. Iterate the identities present in
each frame; do not assume the entire roster is visible at every instant.

Coordinates are yards: `x` runs along a 120-yard field including end zones, and
`y` runs across its 53.3-yard width. Field calibration does not by itself rotate
every offense to the same attack direction. The checked example attacks left.

The current colour convention is blue for `teamA` (lighter kit), yellow for
`teamB` (darker kit), gray for `unknown`. These are appearance groups, not official
club IDs. Team assignment has a per-frame limit of 11 per side. Equality is
required only at an explicitly inspected `full_roster_frame`; cropped shots can
show unequal counts. The SF/SEA example has a checked 11/11 opening formation.

Ball sourcing must remain visible in the UI:

- `detected`: accepted visual evidence.
- `coasted`: an extrapolated estimate between sightings.
- `carrier`: an inferred ball position attributed to a player.
- Missing/null ball: no position; hide its marker.

New clips created through the API enable motion-based visible-flight detection
and disable the generic sports-ball detector and carrier estimator. Motion
detection currently targets isolated brown footballs over turf; it has been
visually checked on the SF/SEA pass, not validated broadly across broadcasts.

The checked output contains seven visible source frames (1772–1778), of which two
fall on the 10 Hz grid. Full-rate pixel sightings live in
`meta.ball.image_observations`, keyed by source frame index. Each records `pixel`,
`field`, `bbox`, `confidence` and `frame`. The confidence is a candidate quality
score, not a calibrated probability. A ring on the source video must use that
frame's own pixel sighting. Ground coordinates during flight are projections;
ball height is unknown. Hidden ball frames stay unknown.

Player identities still fragment and can switch during contact. Conflicting
jersey segments are split, so the roster can contain many more than 22 IDs while
visible team counts remain bounded. This output supports a visual analysis UI;
continuous identities and a complete ball path remain open tracking work.

## Verification

```bash
python -m pytest apps/api/tests/test_video.py packages/tracking/tests -q
```

API tests cover uploads, real shot scanning, source/frame serving, registration
success/refusal, paint-gated pinning, job states, artifact links and video ranges.
The expensive model/render boundary is stubbed in endpoint tests; the browser
encoder is tested against a real short video when FFmpeg is installed. Tracking
tests exercise team capacity, partial views, jersey conflicts, camera motion and
ball evidence.
Big Data Bowl export tests check the real release header, angle conversion,
frame numbering, timestamps, true displacement, missing samples, football IDs
and ingestion by the existing NGS reader.
