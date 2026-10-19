# Football Capstone

**See the play. Understand the mistake. Relive the alternative.**

Football Capstone ingests American football game data, finds the plays that
decided the outcome, explains what went wrong in the numbers that were actually
on the field — and then simulates what would have happened if the call had been
different.

The flagship feature is **WHAT IF?**

> 3rd & 4 on your own 38, 2:11 left, one timeout. You ran inside zone into a
> six-man box that rotated late; the backside end crashed and you were tackled a
> yard short. The punt gave them the ball on their 41.
>
> *Generate What If* → Football Capstone reconstructs the game state at the
> snap, runs the alternative call through a play-outcome model against three
> plausible defensive responses, and shows you what the drive looks like on the
> other side.
>
> **+0.11 win probability. 64% chance the conversion sticks.** Labelled, every
> frame, as a simulation.

This is the football sibling of APEX, which does the same thing for Formula 1
and sim racing: same pipeline shape, different sport.

---

## Status

Four streams of five are in one tree and the data flows between them. `python -m
pytest` at the root is **150 tests**, green.

| stream | state |
|---|---|
| 1 — replay & visualization | [`packages/replay-engine/hindsight`](packages/replay-engine/hindsight) — Unity 6.6 app: field, 22 players, playback, play navigator, 115 real plays. Plus the original three.js [`prototype`](packages/replay-engine/prototype) |
| 2 — footage → tracking | [`packages/tracking`](packages/tracking) — layer 2 built and validated against real tracking; layer 1's back half built and measured in yards. A real detector on real pixels is next |
| 3 — play model | [`packages/counterfactual`](packages/counterfactual) — outcome forecast and typed decision review, bridged to the contract in all three directions |
| 4 — data platform | `packages/shared`, `apps/api` — **not started.** `PlayState` is unwritten, and the contracts live in the packages that produce them until it is |
| 5 — app & product | [`apps/desktop`](apps/desktop) — **not started**, and possibly unnecessary: the Unity app may simply be the app |

The join is data flow, not co-location. On a real play out of the Unity app:

```
hindsight play JSON ─▶ PlayTrack ─▶ layer 2 derives possession, blocks, tackles
                                 ─▶ feature rows in the model's own frame
                                 ─▶ the committed regressors
                                 ─▶ simulated PlayTrack
                                 ─▶ back to hindsight play JSON, labelled SIMULATED
```

Every arrow is a test, and they run against the real artefacts — Jeremy's 115
committed plays and Landon's fitted `coordinate_model.joblib` — not against fixtures
built to agree with them.

The two conversions that can fail silently are both pinned by round-trip tests, and
both were mutation-checked: drop the coordinate mirror and the feature round trip
reports a 57-yard error; drop the angle reflection and five real plays fail. Neither
is cosmetic — the Unity app's angles are compass bearings (0 = +y, clockwise) and the
contract's are mathematical (0 = +x), so a careless rename rotates every player while
the play still animates perfectly smoothly.

**Not done, and worth saying plainly:** there is no win-probability model, so the
headline "+0.11 win probability" below is an aspiration, not an output. The play
model reports a distribution over calls; `expected_value` and `win_prob_delta` are
`null` with an assumption attached saying why.

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
  shared/           Schemas shared between Python and TypeScript (empty; stream 4)
  tracking/         Data providers, player tracking, play-by-play store
  game-engine/      Field geometry, drive/series semantics, team tendencies
  play-detector/    Rules-based detection of decisive plays
  counterfactual/   Play-outcome model, defensive response models
  replay-engine/    Field replay renderer — hindsight/ (Unity) + prototype/ (three.js)
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

## Run it

```bash
python -m pytest
```

Both Python packages, from a bare checkout — `pytest.ini` puts them on the path, so
nothing needs installing first. Each also runs on its own.

## Prototype

[`packages/replay-engine/prototype`](packages/replay-engine/prototype) — the Philly
Special, Super Bowl LII, rebuilt as a scrubbable 3D replay from timed keyframes in
the shared field coordinates. One play, no build step, hand-authored positions. It
exists to prove the data model animates.

---

## Team

Capstone project.

| | GitHub | Stream |
|---|---|---|
| Hassan | [@N1tr029](https://github.com/N1tr029) | 2 — footage → tracking |
| Landon | [@landonDuba](https://github.com/landonDuba) | 3 — play model |
| Jeremy | [@EllisJeremy](https://github.com/EllisJeremy) | 5 — app & frontend |
| Abhi | [@chadiveabhi](https://github.com/chadiveabhi) | |
| Connor | [@connorh27](https://github.com/connorh27) | |

Merged in from where the work was done, every contributor's history intact:

| from | to |
|---|---|
| [`landonDuba/capstone-ML-model`](https://github.com/landonDuba/capstone-ML-model) | `packages/counterfactual` |
| [`EllisJeremy/hindsight-frontend`](https://github.com/EllisJeremy/hindsight-frontend) | `packages/replay-engine/hindsight` |

Both by `git subtree`, so `git log` still shows who wrote what.

---

## Further reading

- [docs/roles.md](docs/roles.md) — how the work splits across the five of us
- [docs/animation.md](docs/animation.md) — why the player views are wrong, and the plan
- [docs/architecture.md](docs/architecture.md)
