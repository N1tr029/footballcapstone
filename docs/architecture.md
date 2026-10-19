# Architecture

Placeholder. To be written as the pipeline lands.

Planned shape, mirroring APEX:

- **Ingest → store** — play-by-play and tracking land in columnar files keyed by
  game; the database holds drives, plays, moments and analyses with pointers.
- **Game engine** — the semantic layer: drives, series, personnel groupings, field
  position, situational context per snap.
- **Detection** — a `Detector` protocol; each rule is a thing a coach would say out
  loud. Detectors that need channels a data source does not publish are *skipped*,
  and the import summary says which.
- **Counterfactual** — reconstruct the state at the snap, apply an alternative call,
  evaluate it against a set of defensive responses, return a distribution.
- **Presentation** — the real play and the simulated alternative on one clock.

---

## What is actually wired, and where the seams are

Written when the separate repositories were merged into this one. The streams are
joined by the three contracts and nothing else; the table below is the real list of
what crosses a boundary today.

```
       video ──▶ packages/tracking ──────────┐
                 (stream 2)                  │
                                             ▼
  Big Data Bowl ──▶ counterfactual.bdb ──▶ PlayTrack ──▶ replay-engine
  release                                    │            (stream 1)
                                             │
                     counterfactual.features ▼
                     (back to the model's frame of reference)
                                             │
                     decision_engine regressors
                                             │
                     counterfactual.forecast ▼
                              simulated PlayTrack ──▶ replay-engine
                                                       labelled SIMULATED

  decision_engine reports ──▶ counterfactual.recommendation ──▶ Recommendation
                                                                 (stream 5)
```

| seam | module | status |
|---|---|---|
| release → `PlayTrack` | `counterfactual.bdb` | done, tested |
| `PlayTrack` → model features | `counterfactual.features` | done, tested, round-trip asserted exact |
| predictions → simulated `PlayTrack` | `counterfactual.forecast` | done, tested against the real fitted model |
| report → `Recommendation` | `counterfactual.recommendation` | done, tested against the real reports |
| vision `PlayTrack` → model features | `features.with_play_state` | **mechanically done, never run on real video.** Needs layer 1's detector, and needs play-by-play for the three play-level inputs video cannot supply |
| `Recommendation` → the What-If view | — | blocked: the frontend is not merged |
| `PlayState` | — | not written. Stream 4 owns it |

### The contract changed once

`Recommendation` gained **`choice_probabilities`**. The specified `outcomes[]` is a
distribution over defensive responses; the play model produces a distribution over
*which call to make*. Those are the same shape and a different object, and filing one
as the other would mislabel it in every view at once. Per the rule in
[roles.md](roles.md) — a new field goes in the contract and everyone gets it — it is
recorded here rather than kept as a local convenience. `outcomes[]` stays in the
contract, and stays empty, until an outcome model that varies the defense exists.

`export.provenance` also grew a third case for `coordinates.source == "simulated"`.
It previously had two: measured by the league's tracking system, or estimated from
video. A counterfactual is neither, and inheriting the video wording would point a
viewer at the registration quality of a play that was never filmed.

### Why the conversions live in exactly one place

Both directions of the coordinate transform are in `counterfactual.features`, and
both are tested as exact inverses on a left-direction play. This is not tidiness.
The model was fitted on the release's own frame of reference — unmirrored, compass
bearings, `play_direction` as a feature — and a `PlayTrack` is mirrored with
mathematical angles. Feeding one to a model trained on the other raises nothing,
returns plausible numbers, and predicts a reflection of the real play.
[animation.md](animation.md) is the post-mortem of that class of bug one layer out.
A second copy of the transform anywhere is a second chance to get it wrong.
