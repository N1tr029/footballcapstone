# Workstreams

Five people, five workstreams. Your three core roles are here — visualization,
footage→data, and the play model — plus the two pieces that sit between them and
have to be owned by someone or they get owned by nobody: the data platform that
every stream reads and writes, and the app that the work is actually seen in.

The point of the split is that after week 1 nobody is blocked on anybody. That
only holds if the **contracts** below are written down first and mocked
immediately — each stream builds against synthetic data until the real upstream
lands.

---

## 1 — Replay & visualization (Unity or Unreal)

**Owns** `packages/replay-engine`, the 3D field, player models, camera work, and
the side-by-side "real play vs simulated alternative" view.

- Field, hash marks, down marker, yard lines to scale; 22 player models with a
  distinguishable team/position read from broadcast-ish camera heights.
- Plays back a sequence of tracking frames (positions + orientation + speed) at
  real time and scrubbable.
- Renders two plays on one clock: the real one and the simulated one, labelled
  **SIMULATED** on every frame.
- Export a clip to video.

**Recommendation: Unity.** C#, a shallower ramp, and the data-driven "drive 22
transforms from a frame array" job is squarely what it's good at. Unreal wins on
photorealism, which is not what this project is graded on.

**Consumes** a tracking-frame array (contract below). **Produces** rendered
playback + video export.

**Week-1 unblock:** hand-write a 60-frame JSON of a toy play and animate it. No
CV, no model needed.

---

## 2 — Footage → tracking data (computer vision)

**Owns** `packages/tracking`, the hard research problem: broadcast or sideline
video in, player positions on the field out.

- Player detection + multi-object tracking across a play (detector + tracker;
  ByteTrack-class association is the standard starting point).
- **Field registration** — homography from image pixels to field coordinates via
  line/hash keypoints. This is the piece that makes the output *data* rather
  than boxes, and it is where the time goes. Do not leave it for later.
- Team assignment (jersey color clustering), and play segmentation: where does
  the snap start and the whistle end.
- An honest quality report per play: how many players tracked, for what fraction
  of frames, registration error in yards.

**Consumes** video. **Produces** tracking frames in field coordinates + a quality
report.

**Week-1 unblock:** run detection and tracking on a single All-22 clip and get
boxes on screen. Registration next.

---

## 3 — Play model (recommendation + outcome)

**Owns** `packages/counterfactual` and `packages/play-detector`: given the state
at the snap, what should have been called, and what would have happened.

- **Outcome model** — features (personnel, box count, down/distance, field
  position, coverage shell) → distribution over yards gained / turnover / score.
  Trained on public tracking + play-by-play.
- **Recommendation** — search the call space (run/pass, concept, direction) and
  return the call that maximizes expected value, in win-probability terms, not
  yards.
- **Detection** — which plays in a game mattered: win-probability swing and
  situational leverage, not a highlight list.
- Reports a distribution against multiple defensive responses, never a point
  estimate, and logs every assumption.

**Consumes** play state + tracking frames. **Produces** a recommended call, an
outcome distribution, and a simulated tracking-frame array for stream 1 to draw.

**Week-1 unblock — important:** do **not** wait for stream 2. Public tracking
data (NFL Big Data Bowl releases) is real player tracking, already registered,
with play-by-play attached. Build and evaluate the model on that; stream 2's
output is designed to land in the same schema later.

---

## 4 — Data platform & pipeline

**Owns** `packages/shared`, `packages/game-engine`, `apps/api`, `infrastructure`.
The schemas, the storage, the service layer, and the job that runs a game end to
end.

- **Writes the contracts** (below) in week 1 and owns them afterwards. This is
  the role's single most important deliverable.
- Play-by-play ingest for real games; columnar storage for tracking frames, a
  database for the semantic layer (drives, plays, moments, analyses) with
  pointers into it.
- The game engine: drives, series, personnel, field position, situational context
  per snap — the layer that turns rows into meaning.
- API the app calls; queue for anything slow (CV, simulation, rendering).
- Synthetic data generator so every other stream has realistic input on day one.

**Consumes** everything. **Produces** the ground everyone else stands on.

---

## 5 — Application, integration & product

**Owns** `apps/desktop`, `packages/ui`, the demo, and the writing.

- Upload/select a game, watch the pipeline run, see what it found.
- Moment browser: the detected plays ranked, with clip thumbnails.
- The What-If view: real play, the recommendation with its reasoning, the
  simulated alternative, the numbers beside it.
- Owns the **demo path** end to end — the thing that gets shown at the review has
  to work when four streams are mid-refactor, and somebody has to be responsible
  for that being true.
- Owns the README, the architecture docs, and the final write-up.

---

## The contracts

Write these in `packages/shared` in week 1. They are the only reason five people
can work at once.

**TrackingFrame** — one instant of one play.

```
play_id, t (seconds from snap), players[]:
  { player_id, team, jersey?, x, y (yards, field coords), speed, direction, orientation }
```

Field coordinates: origin at the back of one end zone, x downfield 0–120, y
across 0–53.3. Everything — stream 2's output, Big Data Bowl data, stream 3's
simulated plays — is in these units, so stream 1 draws all three the same way.

**PlayState** — the situation at the snap.

```
play_id, game_id, quarter, clock, down, distance, yardline, score_diff,
possession, personnel, box_count, formation?
```

**Recommendation** — what stream 3 hands back.

```
call { type, concept, direction }, expected_value, win_prob_delta,
outcomes[] (one per defensive response, with probability),
assumptions[], confidence
```

Rule: a stream may only depend on these three shapes, never on another stream's
internals. If you need a new field, it goes in the contract and everyone gets it.

---

## Who takes what

| Stream | Owner |
|---|---|
| 1 — Replay & visualization | |
| 2 — Footage → tracking data | |
| 3 — Play model | |
| 4 — Data platform & pipeline | |
| 5 — App, integration & product | |

Streams 2 and 3 are the research risk; 4 is the one that quietly determines
whether the other four ever meet. If someone has to carry two, the safe pairing
is 4+5, and the one to never split across two people is 2.
