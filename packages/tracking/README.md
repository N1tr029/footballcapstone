# packages/tracking — footage → tracking data

Stream 2 of [docs/roles.md](../../docs/roles.md). Video in, players in field
coordinates out, plus an honest account of how much of it to believe.

Output is the shared `PlayTrack` that `packages/replay-engine` already draws, so
anything this produces animates without translation.

## Status

The back half of the pipeline is built and measured: registration, field-space
tracking, resampling, kinematics, orientation, export, and the benchmark that scores
all of it in yards. The front half — a real detector on real pixels — is next.

Everything here runs on `numpy scipy pandas pyarrow` alone. OpenCV and ultralytics
are optional extras, needed only once real video is involved.

## Run it

```bash
python -m pytest packages/tracking/tests -q
```

```bash
python -m gridiron_tracking.bench ablation --csv tracking_gameId_2017090700.csv --plays 8
```

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
