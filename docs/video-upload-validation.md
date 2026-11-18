# Automatic upload validation — November 17, 2026

These runs exercise the real `POST /video/uploads` pipeline: automatic shot
selection, field calibration, formation selection, SAM identities, backward
checks, jersey OCR and ball processing. The service receives only a video file.
No field clicks, snap frame, player IDs, identity corrections or pass annotations
are supplied. Selecting a short football shot for the test is the only input
preparation. Existing hand-calibrated clip specs are not passed to the worker.

`scripts/validate_video_uploads.py` saves upload/job responses, Hindsight JSON,
browser previews and consistency reports under `data/runs/upload-validation`.
It checks constant membership, fixed clubs, finite coordinates, team capacity,
roster-role verification and bounded display movement. A completed job without a
track fails validation. Visual preview inspection is an additional check; these
invariants alone do not establish identity accuracy.

## Confirmed tracking results

| Footage | Retained IDs / replay frames | Observed / estimated player samples | Accepted individual backward checks | Missing / drifted source samples repaired |
| --- | --- | --- | --- | --- |
| Northwestern / Michigan | 21 / 65 | 1,095 / 270 | 9 of 9 | 69 / 11 |
| Oregon / Washington State | 19 / 62 | 841 / 337 | 6 of 6 | 20 / 1 |

Northwestern retains the same 21 IDs on every replay frame: 11 in one kit, nine
in the other and one unknown. The maximum display step is 1.202 yards at 10 Hz;
there are no membership changes, club changes, nonfinite positions or display
teleports. `p15` disappears twice; two distinct one-player reverse passes repair
both intervals. Backward repairs require a matching earlier seam, and conflicting
bodies already held by another ID remain unresolved. Four additional paired
crossing checks run automatically. One jersey number passes temporal consensus.

The native word-box fallback automatically calibrates this college field from
five number landmarks confirmed across source frames, with independent paint
alignment of 2.80. The local uniform reader excludes officials before roster
seeding, and a tiled check examines additional athletes. This does not establish
that every athlete on the field was recovered.

There is no accepted ball flight on this clip. Its 19 ball poses are 18 carrier
estimates and one coasted position. It remains `analysisReady:false`; player
recovery must not manufacture a pass or make an unsupported model review ready.

The portable [raw track](examples/nw_michigan_auto.track.json) and
[Hindsight play JSON](examples/nw_michigan_auto.play.json) are genuine worker
outputs, with per-source pixel boxes omitted from the committed fixture for size.
The frontend can fetch `/video/examples/nw_michigan_auto/tracking` without running
models or downloading source footage.

Oregon retains 19 IDs on every frame, eight in one kit and 11 in the other. Its
maximum display step is 1.204 yards at 10 Hz, with no consistency-audit issues.
Six one-player backward checks reconnect to their earlier seams, including three
separate repairs for `p00`. Five officials and one other person are excluded from
the initial formation. An over-capacity tiled alternative is rejected rather
than used to invent a complete roster. The uploaded video is 59.94 fps, so this
also exercises source-frame conversion at a different input rate. Its ball poses
are 15 carrier estimates; there is no accepted flight or pass review readiness.
The [raw track](examples/oregon_wsu_auto.track.json) and
[play JSON](examples/oregon_wsu_auto.play.json) are available through
`/video/examples/oregon_wsu_auto/tracking`.

## Views still rejected automatically

| Uploaded footage | Result |
| --- | --- |
| Milton touchdown, low night-game camera | Field shot selected, but no accepted calibration; no yard-coordinate track published. |
| Devin Singletary, low stadium camera | Field shot selected, but no accepted calibration; no yard-coordinate track published. |
| Obstructed spectator clip | No playable field shot found; the job returns a clear failure. |

The two low-camera clips return `hasTrack:false` with calibration problems.
Their original selector scores were zero; after the field-region fix, their
scores are 0.452 and 0.895 respectively. Recognizing a field shot does not by
itself establish where its yard coordinates lie. The obstructed view remains
rejected. The validation harness exits nonzero for these rejected inputs, as
intended; job completion alone is not counted as tracking success.

An additional local-reader check removes the stands using the detected turf
horizon and samples the first, middle and last image of each low-camera clip.
All six field crops still produce zero usable landmarks/correspondences. Cropping
therefore does not resolve these two calibration failures. No permissive fit or
pixel-to-yard substitution is published for them.

## Failures fixed during these runs

- Stadium backgrounds caused low-camera field shots to be rejected. Shot scoring
  now measures the field below its detected horizon while preserving foreground
  occlusions; a thin turf sliver still fails.
- The reader could abstain despite visible yard numerals, and OCR discarded text
  lines containing several numbers. OCR now retains individual word boxes; a
  fallback fits both number rows using independent temporal confirmations and
  paint/turf checks.
- A 22-person formation could contain officials and omit athletes. Formation
  selection allows officials in the initial count, then checks uniform roles.
  The tiled recheck uses the same sideline cuts, rejects truncated body crops at
  the image boundary, and cannot fill the roster with uncertain new people.
- Unknown kit labels rejected every later body with a known kit. Unknown athletes
  can now follow their matching masks while known opposing-kit gates remain.
- Only one disappearance per identity was checked backward. Every separately
  supported recovery interval is now checked in time order.
- Detection defaulted to CPU on the Mac. Available CUDA/Metal acceleration is now
  selected automatically; the inactive local reader is released before SAM and
  Swift's compiler module cache is shared across clips.

## Controls and regression checks

The previously validated SF/SEA automatic example is a control: 22 IDs on 97
frames, 1,378 observed and 756 estimated player samples, nine accepted individual
checks, 34 missing samples recovered and 11 drifted samples corrected. Its seven
actual ball-flight sightings support the existing inferred throw/arrival. Those
cached results are a control, not another newly uploaded test clip.

All 539 tests pass, including real upload/API contracts, repeated recovery,
unknown-kit handling, roster capacity, failed reverse seams, continuous replay
and sparse-observation round trips. `git diff --check` passes. The real HTTP
example endpoint returns schemaVersion 1, matching the existing play API.

Runs reuse genuine content-keyed detector, automatic-calibration and SAM caches
after fixes when their inputs still match. They do not reuse manually pinned
calibration or operator identity anchors. The Mac entered sleep/dark wake during
part of validation, so its wall times should not be treated as a throughput
benchmark. Model weights, uploads and generated videos remain local artifacts.

This is batch processing with explicit uncertainty. A fixed roster includes
estimated coordinates; off-camera athletes, crowded piles and unresolved kit or
field geometry remain limitations. These clips do not justify a guarantee for
every uploaded football video.
