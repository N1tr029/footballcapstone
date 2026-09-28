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
- Each man has a head that turns toward the ball independently of his body, capped
  at about a hundred degrees off the shoulders — so when the ball is behind him the
  head stops at the limit and you can see he has lost it. He wears a modelled helmet
  ([`models/football-helmet.fbx`](models/football-helmet.fbx)), loaded with
  three.js's FBXLoader, scaled and turned to fit, and painted in kit colours — the
  4K PBR textures that shipped with it are not used, since flat colour suits the
  rest of the scene and keeps the page small. A helmet built from primitives stays
  in the code as the fallback, and is what you see if the model fails to load. The
  facemask is the point: it only exists on the front, so facing reads from any
  angle, with the crest stripe doing the same job from directly overhead and a gaze
  cone backing it up in the top view.

  **Provenance:** the model was supplied locally and arrived without a licence
  file. Confirm its licence and attribution terms before this repository goes any
  further — it is public. Arms carry, throw, reach or block. The ball is held in
  a hand rather than floating at the chest, and it spirals along its flight path.
- [`index.html`](index.html) — the renderer. Three.js, no build step. Field painted
  procedurally to a canvas texture, players as blocky primitives, Catmull-Rom
  through the keyframes so paths bend instead of turning corners, and five ways to
  watch: broadcast (the cinematic keyframe path), offense, defense, a top-down view
  that draws the designed paths as a diagram, and a per-player view — pick any of
  the 22 from the dropdown and the camera rides just behind his head, pointing
  where he is pointing. If the ball is out of frame, that is the answer: at that
  moment he could not see it. Expand takes the stage full size. The stage widens
  from 9:16 to 4:5 for the tactical views, because 24 yards of play does not fit
  a reel.

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
