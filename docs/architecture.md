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
