# GRIDIRON

**See the play. Understand the mistake. Relive the alternative.**

GRIDIRON ingests American football game data, finds the plays that decided the
outcome, explains what went wrong in the numbers that were actually on the field —
and then simulates what would have happened if the call had been different.

The flagship feature is **WHAT IF?**

> 3rd & 4 on your own 38, 2:11 left, one timeout. You ran inside zone into a
> six-man box that rotated late; the backside end crashed and you were tackled a
> yard short. The punt gave them the ball on their 41.
>
> *Generate What If* → GRIDIRON reconstructs the game state at the snap, runs the
> alternative call through a play-outcome model against three plausible defensive
> responses, and shows you what the drive looks like on the other side.
>
> **+0.11 win probability. 64% chance the conversion sticks.** Labelled, every
> frame, as a simulation.

This is the football sibling of APEX, which does the same thing for
Formula 1 and sim racing: same pipeline shape, different sport.

---

## Status

**Scaffolding only.** The directory layout and the plan are here; the code is not
written yet. Nothing below is implemented.

---

## The pipeline

1. **Ingest** — play-by-play and (where available) player tracking for a game.
2. **Game engine** — derive drives, series, personnel, field position and a
   situational record for every snap.
3. **Detection** — find the plays that mattered. Not a fixed list of "highlights":
   leverage, win-probability swing, and situational leverage work it out.
4. **Clips** — cut the video for the moments worth looking at.
5. **Analysis** — explain each moment against this team's own tendencies, not a
   league average.
6. **What If** — simulate the alternative call and show it beside the real one.

---

## Layout

```
apps/
  desktop/          The app shell and UI
  api/              Service layer + the game pipeline
packages/
  shared/           Schemas shared between Python and TypeScript
  tracking/         Data providers, player tracking, play-by-play store
  game-engine/      Field geometry, drive/series semantics, team tendencies
  play-detector/    Rules-based detection of decisive plays
  counterfactual/   Play-outcome model, defensive response models
  replay-engine/    Field replay renderer
  ui/               Design system
services/
  video-worker/     Clip cutting and export
  simulation-worker/Counterfactual and render jobs off the request path
infrastructure/     Database, queue, object storage
docs/               Architecture and design notes
tests/
```

---

## Principles carried over from APEX

- **Tracking data never touches a database row.** Columnar files on disk; the
  database holds the semantic result with a pointer.
- **One timeline.** Video frames and tracking frames share a single clock. Get this
  wrong and every explanation attached to a clip is wrong too.
- **Detection is rules-based, on purpose** — with a `Detector` protocol so a learned
  model can drop in beside the rules later.
- **The model recommends; the simulator decides.** The LLM sees structured game
  state, never raw tracking, and proposes a call. Whether it *works* is decided
  afterwards by the outcome model.
- **You are compared against yourself.** References are this team's own tendencies
  and this team's own results in that situation.
- **The simulation is never presented as fact.** Every simulated frame is labelled,
  outcomes are a distribution, and every assumption is logged and shown.

---

## Team

Capstone project.

| | GitHub |
|---|---|
| Jeremy | [@EllisJeremy](https://github.com/EllisJeremy) |
| Landon | [@landonDuba](https://github.com/landonDuba) |
| Abhi | [@chadiveabhi](https://github.com/chadiveabhi) |
| Connor | [@connorh27](https://github.com/connorh27) |

---

## Further reading

- [docs/architecture.md](docs/architecture.md)
