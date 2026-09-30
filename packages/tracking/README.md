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
detector on real pixels — is next, and nothing here has yet seen a photograph.

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
