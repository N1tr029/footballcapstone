# apps/desktop — the app, and the seam the frontend lands on

Stream 5 of [docs/roles.md](../../docs/roles.md). Empty on purpose: the frontend is
Jeremy's [`hindsight-frontend`](https://github.com/EllisJeremy/hindsight-frontend),
which is **not merged yet** — it is private, and the account doing the merge is not
a collaborator on it, so nothing here has seen a line of it. This file is what it
has to plug into, written down now so the merge is a merge and not a negotiation.

To bring it in, with its history, the same way the play model came in:

```bash
git subtree add --prefix=apps/desktop/hindsight-frontend \
  https://github.com/EllisJeremy/hindsight-frontend.git main
```

That needs read access first. Until then this directory is a placeholder, and the
`.gitkeep` beside this file is doing the only job it has.

## What it consumes

Three shapes, and nothing else — the rule in roles.md is that a stream may depend on
the contracts and never on another stream's internals.

**A `PlayTrack`**, for anything that draws a play. One JSON object: `meta`, `roster`,
`frames[]`, where each frame is a time, a map of player id to `{x, y, s, a, dir, o,
o_source}`, a ball, and events. Field coordinates are yards — x downfield 0–120 with
the offense attacking +x, y across 0–53.3 — and angles are degrees, 0 along +x.

The validator is already written, in both languages, and they are held identical by
a test: [`packages/replay-engine/prototype/engine/contract.js`](../../packages/replay-engine/prototype/engine/contract.js)
(`GRID.validate`) and [`packages/tracking/gridiron_tracking/contract.py`](../../packages/tracking/gridiron_tracking/contract.py).
**Call it on anything before drawing it.** A play that is wrong there is a play that
is wrong in every view at once.

Three producers, one shape: the hand-authored fixture, the vision pipeline, and the
play model's simulated alternative. The frontend should not be able to tell them
apart except by reading `meta` and `provenance`, which is the point.

**A `Recommendation`**, for the What-If panel: `call {type, concept, direction}`,
`expected_value`, `win_prob_delta`, `outcomes[]`, `choice_probabilities`,
`confidence`, `assumptions[]`, `evidence`, `review`. See
[`packages/counterfactual/gridiron_counterfactual/recommendation.py`](../../packages/counterfactual/gridiron_counterfactual/recommendation.py).

**A `PlayState`**, for the situation at the snap. Not written yet — stream 4 owns it.

## Two things the UI must not do

**Do not draw a simulated play without saying so.** `provenance.coordinates.simulated`
is `true` on every counterfactual, and `meta.simulated_from_t` on a joined play says
the time after which nothing happened. A viewer scrubbing across that join has no
other way to know. Every frame at or after it gets the label.

**Do not render an absent number as a zero.** `expected_value` and `win_prob_delta`
are `null` today, because no win-probability model exists in this repository yet, and
`assumptions[]` says so in words. A chart that plots `null` as zero turns "we did not
model this" into "we modelled this as no change", which is a different and much worse
claim. Read `assumptions` and show it.

`o_source` is the same idea one level down: body orientation is `measured` only in
some releases, and on a forecast it is always inferred. Good animation implies good
data to a viewer, so a drawing built from an inferred angle should be badged as one.
