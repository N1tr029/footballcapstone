# apps/desktop — not started

Stream 5 of [docs/roles.md](../../docs/roles.md): the app shell, the moment browser,
the What-If view, and the demo path. Nobody owns it yet.

It is **not** the Unity replay app. That is
[`packages/replay-engine/hindsight`](../../packages/replay-engine/hindsight) — stream 1
— merged in from `hindsight-frontend`, which is a Unity 6.6 project rather than a web
frontend despite the name. It already has a play navigator and a playback HUD, so the
honest question for stream 5 is whether this directory should exist at all or whether
the app simply *is* the Unity build. That is a decision for whoever picks the stream
up; nothing here presumes it.

## What it would consume

Three shapes, and nothing else — the rule in roles.md is that a stream may depend on
the contracts and never on another stream's internals.

**A `PlayTrack`**, for anything that draws a play: `meta`, `roster`, `frames[]`, each
frame a time plus a map of player id to `{x, y, s, a, dir, o, o_source}`, a ball, and
events. Yards, x downfield 0–120 with the offense attacking +x, y across 0–53.3.
Angles in degrees, **0 along +x**. The validator is written in both languages and a
test holds them identical:
[`contract.js`](../../packages/replay-engine/prototype/engine/contract.js) and
[`contract.py`](../../packages/tracking/gridiron_tracking/contract.py). Call it on
anything before drawing it.

Four producers, one shape: the hand-authored fixture, the vision pipeline, the play
model's counterfactual, and the Unity app's own exporter via
[`gridiron_tracking.hindsight`](../../packages/tracking/gridiron_tracking/hindsight.py).

**A `Recommendation`**, for the What-If panel — see
[`recommendation.py`](../../packages/counterfactual/gridiron_counterfactual/recommendation.py).

**A `PlayState`**, for the situation at the snap. Not written. Stream 4 owns it, and
in the meantime the situation fields ride through `PlayTrack.meta`.

## Two things any UI here must not do

**Do not draw a simulated play without saying so.** `provenance.coordinates.simulated`
is `true` on every counterfactual, `simulated` is a top-level flag in the Unity
document, `oSource` is on every player frame, and `meta.simulated_from_t` on a joined
play says the time after which nothing happened. A viewer scrubbing across that join
has no other way to know.

**Do not render an absent number as a zero.** `expected_value` and `win_prob_delta`
are `null` today — there is no win-probability model in this repository — and
`assumptions[]` says so in words. Plotting `null` as zero turns "we did not model
this" into "we modelled this as no change", which is a different and much worse claim.
