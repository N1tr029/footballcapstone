# How players move, look and are filmed

> **Status:** all five phases are built. The renderer no longer guesses anything —
> it consumes a `PlayTrack` through `engine/adapters.js`, poses it through
> `engine/rig.js`, and is checked by `engine/qa.js`, which runs headless over any
> play via `node scripts/qa-play.js <play>`. What follows is the diagnosis that
> led there, kept because the reasoning is the point.

The replay prototype animates one hand-authored play. Watching the quarterback's
helmet cam end to end shows that it does not survive contact with a second one —
and the reasons are structural, not cosmetic. This is the diagnosis and the plan.

---

## What the quarterback actually sees now

Foles, helmet cam, across the whole play:

| t | what is in frame |
|---|---|
| −1.5 s | the back of his own left tackle |
| **0.1 s** | **empty grass. No players, no ball, at the snap** |
| 1.2 s | one figure at the far-left edge, otherwise turf |
| 2.0 s | the throw he is about to receive is off-frame |
| 2.6 s | the ball clipped into the top-left corner |
| **3.0 s** | **the stadium roof. The catch is not in frame** |

He is the receiver on the play and he spends it looking at grass and then at the
sky. Nobody would accept this as "what the player saw".

---

## Why — five faults, in order of damage

**1. Body facing is guessed from velocity.** `rawFacing` takes the direction a
player is travelling and calls that the way he is pointing. Those are different
things and the gap is largest for exactly the players that matter: a receiver
drifting to the corner with his head back at the ball, a defensive back running
one way and looking another, a quarterback backpedalling. Foles' body is turned
away from the play for its entire duration because he is *moving* away from it.

**2. The gaze limit is applied on top of that wrong number.** The head can turn
about a hundred degrees off the shoulders, which is right — but the shoulders are
already wrong, so the head hits its limit and stays there. The limit then *locks
in* the error instead of correcting it.

**3. There is no torso.** A man tracking a ball behind him turns his shoulders
first and his neck second. With a single joint, anything past the neck limit is
simply unseeable.

**4. The look point is placed at a fixed distance.** The camera aims at a point
ten yards along the gaze, regardless of whether the thing being watched is two
yards away or thirty. When the ball arrives near his hands, that arithmetic puts
the aim point well above it. This is the sky at 3.0 s.

**5. Smoothing papers over a timing problem.** Heads snap because the gaze target
changes instantly when the ball changes hands. A box filter blurs the snap; it
does not make the motion human, and it costs latency everywhere else.

Underneath all five: **the rig is driven by inference where it should be driven by
measurement**, and the inference has no model behind it.

---

## The thing that changes everything

Most of what is being guessed here is *measured* in real tracking data. Per player
per frame at 10 Hz, NFL Big Data Bowl releases carry:

| field | meaning |
|---|---|
| `x`, `y` | position on the field |
| `s`, `a` | speed and acceleration |
| `dir` | direction of **motion** |
| `o` | **orientation** — which way the body is pointing |
| `event` | `ball_snap`, `handoff`, `lateral`, `pass_forward`, `pass_arrived`, `tackle`, … |

`o` and `dir` are separate columns precisely because a player's body and his path
diverge. Fault 1 is not a modelling problem — it is us not reading a column that
exists. The football is a row in the same table, so the ball's position through a
throw is measured too, which removes the hand-authored carries and flights. And
the `event` column gives the phases of a play without anyone declaring them.

What is genuinely *not* in any public dataset is **where a player is looking**. No
head or eye tracking exists. That stays inferred — so it should be one explicit,
documented model with limits and a label, not heuristics scattered through the
renderer.

---

## Plan

### Phase 0 — Widen the contract

`TrackingFrame` currently carries position only. It becomes:

```
play_id, t, events[]
players[]: { id, team, x, y, s, a, dir, o, o_source }
ball: { x, y, z? }
```

`o_source` is `measured` | `from_dir` | `from_velocity` — the renderer must know
how much to trust what it is drawing. This is a change to the shared contract in
[roles.md](roles.md), so it lands before anyone builds on the current shape.

### Phase 1 — Drive the rig from data

- Body yaw ← `o`, falling back to `dir`, falling back to velocity, recording which.
- Ball ← the measured ball track. Delete the authored carry/flight segments.
- Phases ← `event` rows. Delete the authored beat list.
- Keep the hand-authored play as a *fixture*: it becomes the synthetic test case
  that exercises the renderer without needing a data pull.

Fixes faults 1 and 2 outright, and makes every later play free.

### Phase 2 — A gaze model worth the name

Three changes, all cheap:

- **Two joints.** Torso twist ±45°, neck ±70°. Past their sum, the whole body
  turns over ~0.3 s — a man looks for the ball rather than freezing at a limit.
- **Reaction and rate limits instead of filtering.** A 200 ms reaction delay, and
  a neck that slews at a bounded angular velocity (~400°/s). Motion becomes smooth
  *because it is physical*, not because it was blurred, and the snapping problem
  disappears at its source.
- **Aim at the real point.** Look at the target's actual 3D position, not a point
  projected at a fixed distance. Fixes fault 4.

Target selection stays role-driven — carrier looks up the field or at the man he
is handing to, passer at his receiver, everyone else at the ball — but keyed off
measured events rather than authored ones.

### Phase 3 — Framing you can trust

- Eye rigidly mounted in the helmet, as now.
- **Tell the truth about a bad view.** For each player, compute how much of the
  play's key moments fall inside his frustum. If a POV never sees the ball, the
  page says so instead of quietly showing grass. A lineman who never finds the
  ball is a finding; the same frame unlabelled is a bug.

### Phase 4 — Validation at scale

Thousands of plays cannot be eyeballed. A headless pass per play, per player:

| metric | fails when |
|---|---|
| ball in frame | the carrier's own view loses the ball |
| angular velocity | the head exceeds human limits |
| horizon stability | the frame rolls or pitches beyond bounds |
| camera occlusion | the eye ends up inside another body |
| orientation source | a play falls back to velocity for most players |

Plays that fail get flagged, not published. A handful of golden frames from known
plays become regression tests so a change to the rig cannot silently break plays
nobody is looking at any more.

### Phase 5 — Label what is inferred

Position, orientation, ball and events are measured. Gaze, limbs and camera are
not. The page already stamps the whole reconstruction; per-channel badges are the
honest version of that, and they stop good animation from implying good data.

---

## Order, and what it buys

| | work | buys |
|---|---|---|
| 0–1 | contract + read `o`, ball, events | the QB looks at the play; every play works |
| 2 | gaze model | human heads, no snapping, no sky |
| 3 | framing report | bad views become findings |
| 4 | headless QA | scale without watching everything |
| 5 | badges | nobody mistakes the drawing for the data |

Phases 0 and 1 are most of the win and are mechanical. Phase 2 is the only part
that needs judgement, and it is the only part that will still be inference when
real data is flowing.


---

## What was built

| file | does |
|---|---|
| `engine/contract.js` | the shared shape, the angle convention, and `validate()` |
| `engine/adapters.js` | `fromKeyframes` (the fixture) and `fromTrackingRows` (real releases) |
| `engine/rig.js` | carrier chain, exchanges, the gaze model, limb poses |
| `engine/qa.js` | per-play, per-player checks and a printable report |
| `scripts/qa-play.js` | the same checks headless, non-zero exit on a play not to publish |

The renderer holds no play logic at all now. To add a real tracked play you write
nothing: `GRID.fromTrackingRows(rows)` produces the same `PlayTrack` the fixture
does, and everything downstream — cameras, helmet cam, gaze, arms, the checks —
works unchanged. The one thing to calibrate on first contact with a real release
is the `angle` option, which maps that release's orientation convention onto ours.

### What the checks caught on the first run

Worth recording, because it is the argument for having them:

- **Derived orientation spins on the spot.** Below walking pace, direction of
  travel is noise, so bodies span wildly exactly when a lineman is set. Now held
  below 0.6 yd/s and rate-limited. Real releases have the same failure mode, which
  is why they ship `o`.
- **A neck injury in the fixture.** Foles' authored turn upfield after the catch
  was 130° in half a second. The rate check found it; the keyframes were spread.
- **Linemen cannot see anything.** Their helmet cams are obstructed for a sixth of
  the play by the man across from them. That is true, so it is reported as a
  finding on the view rather than counted as a defect.
