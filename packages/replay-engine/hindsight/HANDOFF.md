# Handoff — the Unity work nobody else can do

This project is merged into the capstone monorepo at
`packages/replay-engine/hindsight`, with its history intact; `git log` still shows
your commits. Your `Assets/`, `Packages/`, `ProjectSettings/` and `tools/` are
untouched except for the four files listed below.

**Everything here was written without Unity and has never been compiled.** There is
no Unity and no C# toolchain on the machine it was written on. It was reviewed by
reading: brace balance, the Input System and TMP/uGUI APIs checked against your
surrounding code, `Mathf.SmoothDampAngle`'s signature, and `JsonUtility`'s
behaviour for absent fields. Treat all of it as a draft that compiles in your head
and nowhere else.

The JSON side *is* tested, from Python, in
`packages/tracking/tests/test_unity_contract.py` — it parses `PlayData.cs` for
declared fields and fails if the exporter writes something the client would
silently drop.

---

## 1. The thing that actually matters

`JsonUtility` ignores any JSON field the target class does not declare. No error,
no warning. The Python exporter had been sending three honesty fields for a week
and all three did nothing, which meant **a model's forecast drew identically to
real league tracking** with only the description to tell them apart.

Two of them are now declared in `PlayData.cs` and read:

| field | where | what it means |
|---|---|---|
| `simulated` | `PlayData` | this play did not happen; the positions are a forecast |
| `oSource` | `PlayerState` | how that body angle was arrived at — `measured`, `from_dir`, `from_velocity`, `assumed`, `authored` |

A missing `simulated` defaults to `false` and a missing `oSource` to `""`, which
`PlayerState.OrientationMeasured` treats as measured. That is correct: the only
files without those fields are your own exporter's, and those are real tracking.

**`provenance` is also in the file and you cannot read it with `JsonUtility`.** It
is a nested object with arbitrary keys, and `JsonUtility` supports neither
dictionaries nor untyped nesting — reading it would mean a DTO per block for no
gain. It is there for humans and for other consumers. Ignore it; `simulated` and
`oSource` carry the same warning in a shape the client can use. (A test pins this
so nobody is told to "just read provenance" and loses an afternoon.)

## 2. What was changed, and what to check

Four files. Each change is small and each has a specific way it can be wrong.

**`Assets/Scripts/Data/PlayData.cs`** — added `simulated` to `PlayData`;
`oSource` and the `OrientationMeasured` helper to `PlayerState`; `trackId` to
`PlayerInfo` (the original track name when a play came from video and the `nflId`
is synthetic).
*Check:* nothing. This is declaration only, and the Python test covers the shapes.

**`Assets/Scripts/Playback/PlaybackController.cs`** — one line:
`public IReadOnlyList<PlayerView> Players => views;` so a camera can ride one.
*Check:* nothing.

**`Assets/Scripts/UI/PlaybackHud.cs`** — a red `SIMULATED — THIS DID NOT HAPPEN`
banner, top-centre, shown whenever `play.simulated` is true **or** any player frame
carries an `oSource` that is not `measured`.
*Check:* that it appears on a simulated play and on a video-tracked one, and does
**not** appear on your 115 Big Data Bowl plays. If it shows on those, the default
for an absent `oSource` is being read as inferred and the logic is inverted.

**`Assets/Scripts/Cameras/CameraRig.cs`** and **`Assets/Scripts/UI/PlaybackInput.cs`**
— the player POV camera. `C` now cycles top-down → broadcast → player; `V` rides
the next player, `Shift+V` the previous; a play loads pre-pointed at the
quarterback so `V` is one keypress rather than needing someone selected first.

## 3. The POV camera, and the three ways it is likely wrong

It is a third camera tagged `MainCamera`, positioned each `LateUpdate` at the
ridden player's position plus `povEyeHeight` (1.05 yd) and nudged
`povForwardOffset` (0.45 yd) along his facing, rotated to that facing.

**You will probably be looking at the inside of his own head.** The offset and a
0.05 near clip are a guess at clearing the player mesh. If the view is full of
jersey, either push the offset out or cull the ridden player's renderers while he
is being ridden — the latter is correct and I did not write it because I could not
see the result.

**The yaw is smoothed with `Mathf.SmoothDampAngle`, not `SmoothDamp`.** That is
deliberate: at 10 Hz a rigid camera snaps, and subtracting angles sends a man
turning past due north the long way round through his own shoulder. If it still
feels rough, raise `povTurnSmoothTime` (0.12) rather than switching function.

**It looks along the body, not the eyes, and it always will.** The camera uses
`o`, the measured body orientation. No public football dataset has a gaze column
and there will not be one. A receiver running a corner with his head turned back
at the ball is drawn facing the corner. That is an honest view of where a man was
and what was around him; it is not what he saw, and the UI should not imply it is.
On any play whose `oSource` is not `measured` it is an inference drawn on top of an
inference — which is exactly when the banner from §1 is on screen.

**There is a working reference.** `packages/replay-engine/prototype` is the
three.js prototype in this same repo and it has a helmet cam: the camera sits at
the face opening and you look out through the bars, with the shell modelled with
wall thickness so sitting further back puts you inside it. If the Unity version
fights you, that code has already solved the same geometry.

## 4. Plays arrive from Python now

`Assets/StreamingAssets/plays/` is no longer only your exporter's output. Two
commands in the monorepo write there, both through
`packages/tracking/gridiron_tracking/hindsight.py`:

```bash
python -m gridiron_tracking unity <clip>            # a clip tracked from video
python -m gridiron_counterfactual whatif --worst 5  # plays + the model's alternative
```

Both keep your 115 plays and re-index around them. Video plays carry `gameId`
999000001; a simulated play sits immediately after the play it argues with, in the
same quarter and drive, so `N` steps from the real throw to the forecast.

**Your angles are not the contract's.** Your schema uses compass bearings — zero
along +y, clockwise — and the shared contract uses the mathematical convention,
zero along +x. The Python side converts in exactly one place and a round-trip test
over five of your real plays asserts it is lossless. Nothing is being asked of you
here; it is written down because the two conventions look interchangeable and are
not, and a play converted without the reflection animates perfectly smoothly with
everybody facing ninety degrees wrong.

## 5. Order to do it in

1. Open the project, let it compile, fix whatever I got wrong. **Expect errors.**
2. `Hindsight > Setup Scene`, press Play, load one of your own 115 plays. Nothing
   should look different and **no banner should appear**.
3. Load a simulated play (run the `whatif` command above first). The banner should
   be on every frame.
4. `C` three times — top-down, broadcast, player. `V` to ride someone.
5. Tell us what the POV actually needs; the numbers in §3 are guesses.
