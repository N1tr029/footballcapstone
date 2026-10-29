# packages/tracking — footage → tracking data

Stream 2 of [docs/roles.md](../../docs/roles.md). Video in, players in field
coordinates out, plus an honest account of how much of it to believe.

Output is the shared `PlayTrack` that `packages/replay-engine` already draws, so
anything this produces animates without translation.

## Two layers, and only one of them needs video

**Layer 1 — coordinates.** Video in, x/y/s/a/dir/o per player per frame. Hard,
uncertain, and blocked on real footage.

**Layer 2 — what happened.** Who had the ball, who hit whom, who blocked whom, who
went around whom, when each player could plausibly see the ball. Once you have
twenty-two trajectories this is *geometry over the coordinates*, not computer vision —
so it is built, validated against real public tracking, and will inherit layer 1's
output for free when it lands.

Getting these the wrong way round — trying to spot a tackle in pixels — is how this
kind of project runs out of semester.

## Status

Layer 2 is built and validated against real tracking. Layer 1's back half is built
and measured: registration, field-space tracking, resampling, kinematics, orientation,
export, and the benchmark that scores it in yards. Layer 1's front half — a real
detector on real pixels — has started: players come off a general-purpose detector,
and the ball has a fine-tune of its own, measured and still too weak to use. See
**The ball** below for exactly how weak and why.

Everything here runs on `numpy scipy pandas pyarrow` alone. OpenCV and ultralytics
are optional extras, needed only once real video is involved.

## Run it

```bash
python -m pytest packages/tracking/tests -q
```

```bash
python -m gridiron_tracking.bench ablation --csv tracking_gameId_2017090700.csv --plays 8
```

To produce the data file for one play:

```python
from gridiron_tracking.sources import ngs
from gridiron_tracking import interactions, export

df = ngs.load_tracking("tracking_gameId_2017090700.csv")
track = ngs.to_play_track(df, play_id=139, play_direction="right")
export.write_play("out/", track, interactions.derive(track, offense="home"))
```

`out/play_139.csv` is one row per player per frame in release shape;
`out/play_139.json` is the whole play plus the interaction layer and the provenance
block. **Read `provenance` before trusting any of it.**

Any public NGS release works as the `--csv`; column spellings are resolved by alias.

**Where to get one, without an account.** The release every number below is measured
on is published openly by the NFL — no Kaggle sign-in, no competition entry:

```bash
mkdir -p data/bdb2017 && cd data/bdb2017
for f in tracking_gameId_2017090700.csv plays.csv games.csv players.csv; do
  curl -LO "https://raw.githubusercontent.com/nfl-football-ops/Big-Data-Bowl/master/Data/$f"
done
```

30 MB, 177 plays, and it carries the three things the newer competition files do
not: an `event` column, a `dis` column, and real per-frame football rows. The
2026 competition release loads fine through the same aliases but cannot drive the
benchmark — it has `ball_land_x`/`ball_land_y`, where the ball came down, and no
record of where it was on the way.

## How accuracy is measured

Real tracking → virtual camera → boxes → the pipeline → scored against the tracking
it started from. Because the truth is known exactly, every number is attributable,
and the whole loop runs in seconds with no video on disk.

Measured over 8 real plays from the 2017 Big Data Bowl release:

| condition | median | p95 | id switches |
|---|---|---|---|
| ids given, no detector noise, perfect clicks | 0.08 yd | 0.27 yd | 12 |
| + box jitter and dropout | 0.14 yd | 0.37 yd | 21 |
| + operator clicks 2 px off | **0.40 yd** | 0.66 yd | 39 |
| + tracker associates by distance, not identity | 0.40 yd | 0.68 yd | **115** |
| helmet boxes instead of body boxes | 0.65 yd | 1.10 yd | 176 |

Read downward: each row adds one thing, so a jump is that stage's cost. Two results
worth acting on — **operator click precision dominates position error** (0.14 → 0.40
yd from three pixels), and **association costs identity, not position** (switches
triple while the median does not move).

## Layer 2 accuracy, against the release's own event labels

68 plays of the 2017 Big Data Bowl release carry both `first_contact` and `tackle`.
Nothing in the derivation ever sees those labels, so they are a fair test:

| what | result |
|---|---|
| carrier contact found | **100%** of plays |
| contact time vs `first_contact` | median **0.20 s**, 90% within 0.5 s |
| going-down time vs `tackle` | median **0.30 s**, 73% within 0.5 s, present on 97% |
| which team had the ball | **98.7%** of 149 labelled plays |

Two findings from building it. The gap between first contact and being down is a
median **0.90 s** — a man wrapped up keeps his feet — so an engagement records both
`t_start` and `t_down`, and treating them as one moment costs most of a second. And
the 2017 release's **ball column is not a ball track**: its `dis` is all zeros and
after the handoff the ball sits 4–5 yards from everybody, so possession has to be
inferred from the players. `ball_track_usable()` detects that rather than trusting it.

Possession without a ball track comes from **defensive convergence**: eleven defenders
are all trying to reach the same man. Strong on a run, weak on a deep pass where the
defense is covering receivers instead — so it is reported as an estimate.

## What is not measurable, and is therefore not in the file

**Where a player was looking.** No football dataset contains head or eye tracking.
There is no gaze column and there will not be one. The nearest honest thing is
`interactions.ball_in_view` — was the ball inside a cone around the direction his body
was pointing — and every record carries `inferred: true` plus the `o_source` it was
built from, so a window derived from a measured orientation is distinguishable from one
derived from a guess.

**"Exact" coordinates.** Video → position is an estimate. The file says so in
`provenance.coordinates`, and it says it differently depending on whether the
coordinates came from a tracking system or from a homography.

## Registering a clip the vision model cannot

`register.mode: auto` asks the vision model and is checked against the painted
lines; below 1.60x the drawn lines are finding no more paint than chance and the
clip is refused. It clears that bar on some cameras and not others — a low wide
sideline shot gives it three usable correspondences and it needs more, and three
of the eight clips here fail on it. The fallback is a person naming points, which
is what `data/calibrate.html` is for, and these two commands are the steps either
side of it:

```bash
python -m gridiron_tracking calframe oregon_wsu          # a frame from the window
# open data/calibrate.html, load it, name 4-6 points, press Copy
python -m gridiron_tracking pin oregon_wsu points.json --frame 207
```

`pin` fits the homography, scores it against the paint on that same frame, and
writes it into the spec **only if it clears the bar** — so the answer to "is this
good enough" arrives in a second rather than after twenty minutes of detection.
Name points that are far apart: opposite sidelines, and yard lines well down the
field. A fit from points clustered in one corner is self-consistent and can still
be a hundred yards wrong, which is the whole reason the paint check exists.

## The three decisions worth knowing about

**Registration is operator-assisted.** Someone clicks four to six named field
landmarks once per camera setup; the homography follows. It works on a worn
high-school field in November, where line detection does not. Automatic registration
lands later behind the same protocol.

**Tracking happens in yards, not pixels.** Detections are projected to the field
*before* association. Camera pan stops being a tracking problem, constant velocity
becomes true, and human speed limits become usable evidence.

**Height is corrected, not ignored.** A helmet is two yards off the turf, and pushing
it through a ground-plane homography puts the player 5–12 yards away — always toward
the far sideline. The full camera is recovered from the homography so the ray can be
intersected at the right height. See `registration/camera.py`.

## Layout

```
contract.py        the PlayTrack shape, mirroring the renderer's contract.js
field.py           geometry + the landmark catalog (hash insets differ by level!)
sides.py           which team has the ball (98.7%: the centre is on it at the snap)
interactions.py    layer 2 — possession, blocks, tackles, evades, ball-in-view
export.py          the CSV and JSON the team consumes, with provenance
registration/      homography, camera recovery, operator clicks, quality warnings
detect/            Detection + Detector protocol; synthetic and oracle sources
track/             field-space association and Kalman filtering
kinematics.py      resample to 10 Hz, derive s / a / dis / dir
orientation.py     o, with an honest o_source
pipeline.py        the whole back half, stitched
bench/             the scorer and the tuning loop
sources/ngs.py     public NGS releases → the contract
```

## Two things that will bite you

**NGS angles are compass bearings.** `dir` and `o` are measured clockwise from +y;
the contract is counter-clockwise from +x. The conversion is `(90 - a) % 360`, and
`ngs.check_angle_convention()` proves it empirically against differenced positions
rather than asking you to trust it.

**Hash marks move between levels.** NFL 23.58 yards from the sideline, college 20,
high school 17.78. Register a high-school field against NFL hashes and every player
lands six yards sideways, with a clean-looking reprojection error, because the
homography absorbs the mistake.

## The ball, and why it is the one thing still blocking the chain

Everything the play model needs from a clip is now derived automatically — a roster
of twenty-two, which side has the ball, which way it is going, the line of
scrimmage — except the throw and the arrival, and those need the ball.

The ball stage uses the generic COCO "sports ball" class. It cannot find a football
in this footage, and the way it fails is worth recording because both failures look
like success from a detection rate.

**Un-tiled** (`imgsz 1280`, conf 0.10): 3 detections in 65 frames. 5%.

**Tiled** (full frame plus 3x3 overlapping crops at 1536, conf 0.05 — the same trick
`detect/yolo.py` uses for far-hash players): 27 detections, 42%. Nine times better,
and every one of them the **down-and-distance graphic**. It is rendered onto the
field near the line of scrimmage, it is about the right shape, and it is in every
frame. The resulting "ball" travelled 2.4 yards in 2.6 seconds.

**Tiled, with static candidates rejected** (any cluster of candidates holding one
field position across a large part of the play is furniture): 1146 candidates
dropped at three fixed spots, leaving 12 detections, 18%. The track now travels 23
yards — but at 2 to 11 yards a second in a smooth drift, which are human speeds. With
the graphics gone the detector finds a player instead.

So: static clutter, then a person. Never the ball. COCO class 32 was trained on
basketballs, tennis balls and footballs of the other kind, and a brown spheroid that
is nine pixels across and in someone's hands most of the play is out of its
distribution.

**Two things that were tried instead of labelling, and did not work.** Both are
here so the next person does not spend the afternoon again.

*Reading the ball off the picture by eye, to label it.* At 10x zoom on the frames
where the ball is in somebody's hands there is reliably a tan object near the
hands — and it cannot be told apart from the player's own forearm. Both are tan,
both are about ten pixels, and the arms are extended exactly where the ball is.
Labelling that teaches a detector to find forearms, which would look like progress
and be worse than nothing. A person who knows football can do this; reading it off
a still at this resolution cannot.

*Finding it by motion instead.* A thrown ball is the only small thing on the field
moving faster than a person can run, so cancelling the camera's pan and looking for
small fast blobs should isolate it without any detector. Measured on the reference
clip: after phase-correlation pan cancellation and a turf mask, **101 small on-field
blobs per frame** — limbs, paint texture, shadows — and of 3222 linked chains of
four or more frames the longest ran at 0.1 to 1.4 px/frame, which is static
texture. No fast coherent trajectory at all. The pan cancellation and the turf mask
both work; the signal-to-noise is the problem, and tuning it further against a clip
with no ground truth is how the 42% figure above happened in the first place.

**The fine-tune happened, and the first one was measured wrong.** It scored
mAP50 0.005 and was written off. Re-measured against the holdout it was actually
built with — `devin_td`, 25 balls — `best.pt` scores:

| imgsz | mAP50 | precision | recall |
|---|---|---|---|
| 960 — what it was trained at | **0.107** | 0.478 | 0.120 |
| 1536 — what it is *used* at | 0.001 | 0.003 | 0.120 |

It had learned something: nearly half right when it fires, firing on an eighth of
the balls. Too weak to use, and weak for two reasons that are not about footballs.

**Scale.** It was trained on whole 1920x1080 frames at `imgsz 960`, which halves
everything — a twelve-pixel ball arrives as six, at or under the floor of a
stride-8 detection head. Inference does the *opposite*: it tiles the frame and
magnifies each tile, so the ball it is asked about is two to three times larger
than any ball it was trained on. The collapse from 0.107 to 0.001 between those two
sizes is that mismatch measured. `to_yolo_tiled` crops at native resolution
instead, so training sees the ball the size the source has it, and held-out clips
are cut on the same overlapping grid inference sweeps with.

Mosaic augmentation has to go off with it. Mosaic stitches four images into one and
downscales — on tiles cropped specifically to preserve the ball's size it undoes
the crop and hands back the six-pixel ball. With it on, classification loss sat at
147 falling to 10 and mAP50 never left 0.005; with it off the loss starts near 1.5.

**Stale labels.** 25 of the first 112 labels are exact coordinate repeats — the
same point to the last decimal, which is a click quantised to a screen pixel, not a
hand. A click saves the frame and advances, leaving the next frame under a cursor
that has not moved; click again and it is labelled at the old frame's point. 20 of
the 25 are one kickoff: a ball in flight recorded at one fixed point, so the box
sits a ball-width behind the ball on every frame but the first. That was **30% of
the training positives**. The UI now refuses a click within two pixels of the
previous frame's, and `without_stale_repeats` throws away the ones already
collected.

**What is still needed.** 87 real labels is thin, and the honest next gain is more
of them rather than more epochs. Nothing else in the chain is waiting on research;
this is the one piece, and the machinery around it is built:

```bash
python -m gridiron_tracking label        # samples frames, opens the labelling UI
python -m gridiron_tracking train-ball   # exports a YOLO dataset and fine-tunes
```

`label` samples every fourth frame of each clip's own window — consecutive frames
of a 30 Hz clip are the same picture and a model learns nothing from the second —
and serves a page that magnifies the cursor 5x, because the thing being clicked is
about nine pixels across. Click the ball; press `N` when it is in a pile or behind
somebody. Every click is written to `data/labels/labels.json` as it happens, so a
closed tab costs one frame rather than an afternoon.

`train-ball` holds out **whole clips** rather than random frames. Frames four apart
from one camera are near-duplicates, so a random split scores memorisation; holding
out a clip measures the only thing worth knowing, which is whether it finds a ball
on footage it has not seen. Frames marked `N` become empty label files — the
negatives that stop a fine-tune learning the graphic all over again.

Name the clip with `--holdout`, and name one with balls in it. The first run held
out `ucf_kickoff`, which has **one**: a model that predicts nothing at all scores
well on that, and the resulting number means nothing either way. `to_yolo` warns
when the held-out clip has fewer than five.

Then point a clip at the result and **measure it the same way**:

```jsonc
"track": { "ball": true, "ball_weights": "data/labels/weights/football/weights/best.pt",
           "ball_class": 0, "ball_imgsz": 640, "ball_tiles": 3 }
```

**These two numbers are not free.** `ball_tiles: 3` over a 1920-wide frame cuts
640-wide tiles, and `ball_imgsz: 640` hands each one to the model unscaled — which
is the tile size and the magnification `train-ball` used, so the ball arrives the
size it was learned at.

The rule is `ball_imgsz` = the `--imgsz` the weights were trained with, and
`ball_tiles` chosen so a tile is the `--tile` they were trained with. Change one
and you must change the other. Reading them off `args.yaml` beside the weights is
the reliable way:

```bash
grep -E "^(imgsz|model):" data/labels/weights/football/args.yaml
```

Getting this wrong is not a small miscalibration. The same weights score mAP50
0.107 at the size they were trained at and 0.001 two sizes up — setting these
independently of training is exactly how the first fine-tune came to look
worthless at the size it was being used at.

Read `travelled_yards` beside `measured_fraction`. A detection rate alone has looked
like progress twice on this task and been furniture both times.

**What the pipeline does meanwhile**, and the part worth keeping whatever replaces
the detector: it refuses, three times over and for three different reasons.
`_merge_ball` records `travelled_yards`, so a static track is visible in the file;
`prepare` blocks a ball that covers under ten yards in a play *and* one detected on
under a quarter of samples; and `events.derive` will not call anything a throw that
does not clear 13 yards a second for a third of a second over six yards. The second
of those caught the graphic before anybody looked at a picture.
