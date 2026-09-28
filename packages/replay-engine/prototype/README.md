# Replay prototype — the Philly Special

The first thing the replay engine has to prove: hand it a play as timed positions
and it draws the play. This is that, with one play in it.

```bash
python3 -m http.server 4173 --directory packages/replay-engine/prototype
# then open http://localhost:4173
```

(or `replay-prototype` from `.claude/launch.json`.)

- [`plays/philly-special.js`](plays/philly-special.js) — the play: 22 players as
  timed keyframes, the ball as a chain of carries and flights, camera keyframes,
  and the beats that drive the captions. All in the field coordinates from
  [docs/roles.md](../../../docs/roles.md): `x` 0–120 downfield, `y` 0–53.3 across,
  `t` in seconds from the snap. Sample it at 10 Hz and it is a TrackingFrame array.
- [`index.html`](index.html) — the renderer. Three.js, no build step. Field painted
  procedurally to a canvas texture, players as blocky primitives, Catmull-Rom
  through the keyframes so paths bend instead of turning corners, and four cameras:
  broadcast (the cinematic keyframe path), offense, defense, and a top-down view
  that draws the designed paths as a diagram. The stage widens from 9:16 to 4:5 for
  the three tactical views, because 24 yards of play does not fit a reel.

**The positions are hand-authored.** The All-22 for this play has never been
released as tracking data, so the paths match the shape of the play and nothing
more. Everything downstream — the vision pipeline's measured positions, the play
model's simulated alternative — lands in the same shape and draws through the same
renderer. That is the whole point of the prototype.

## What it does not do yet

Unity or Unreal, if the visualization workstream goes that way — this is the
cheapest possible proof that the data model animates, not a commitment to a
renderer. Real player models, contact, blocking, a crowd that is more than a
texture, and playing two plays at once (the real one beside the simulated one) are
all still open.
