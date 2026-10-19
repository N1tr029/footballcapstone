# Hindsight (frontend)

Unity 6.6 (6000.6.3f1, URP 17) app that replays NFL plays from player tracking data, viewed
top-down like a chess game review. Nothing is simulated: the app plays back recorded positions,
interpolating between 10 Hz tracking frames. Must stay WebGL compatible.

Data source: NFL Big Data Bowl 2025 (2022 season, weeks 1-9). Raw CSVs live in `Data/raw/`
(git-ignored, never imported into Unity). Plays are currently exported to static JSON in
`Assets/StreamingAssets/plays/`; a backend will later serve the same JSON.

## Layout

```
tools/export_plays.py          CSV -> play JSON exporter (pandas)
Assets/StreamingAssets/plays/  index.json + {gameId}_{playId}.json
Assets/Scripts/
  HindsightApp.cs              composition root, the only scene object
  Data/      PlayData (JSON DTOs), IPlaySource, WebPlaySource (UnityWebRequest), PlayLoader
  Playback/  PlaybackController, PlayTimeline (sampling), ThrowArc, PlayerView, BallView, TeamColors
  Field/     FieldBuilder (field mesh, lines, hashes, TMP yard numbers, LOS / first down markers)
  Cameras/   CameraRig (ortho top-down + perspective broadcast)
  UI/        PlaybackInput (keyboard), PlaybackHud (TMP overlay), PlayNavigator + NavCell
             (quarter > drive > play accordion bar), Billboard
  Editor/    HindsightSetup (menu: generates materials, prefabs, scene object)
Assets/Art/Materials/          Unlit.mat, Ball.mat (generated)
Assets/Prefabs/                Player.prefab, Ball.prefab (generated, safe to customize)
```

## Setup

1. `python3 tools/export_plays.py` (default: every play of game 2022090800, ~28 MB;
   `--game ID` for another game, `--plays ID ...` for a subset). Needs pandas.
   Clears existing JSON in the output folder first.
2. In Unity: Window > TextMeshPro > Import TMP Essential Resources (once).
3. Hindsight > Setup Scene, then save the scene. This creates the materials and prefabs
   (only if missing), adds a `Hindsight` object with `HindsightApp`, and deletes other scene cameras.
4. Press Play.

Controls: Space play/pause, Left/Right step a frame, 1-4 speed (0.25/0.5/1/2x), R restart,
N/] next play, P/[ previous play, C toggle camera.

Play navigator (bottom bar, fixed width): quarters > drives > plays. Only the quarter and drive
containing the current play are open; opening one takes width from its collapsed siblings.
Clicking a quarter or drive loads its first play; clicking a play loads it. Play cells are
colored by result (green complete, gray incomplete, red INT, blue run, orange sack, teal
scramble, gold TD). Hover shows the play description above the bar.

## Conventions

- Build scene content from code. Avoid hand-editing scene/prefab YAML; use the editor menu.
- Colors are set via `MaterialPropertyBlock` / runtime instances of `Unlit.mat`. Don't use
  `Shader.Find` at runtime: shaders only referenced that way get stripped from WebGL builds.
- File/network reads go through `IPlaySource` + `UnityWebRequest`, never `System.IO`.
- Input uses the Input System package (`Keyboard.current`); the legacy Input Manager is disabled.
- After script changes, check `~/Library/Logs/Unity/Editor.log` for compile errors.

## Coordinate system

- 1 Unity unit = 1 yard. Data `(x, y)` maps to Unity `(x, height, y)`.
- Field: x 0..120 (end zones 0..10 and 110..120, goal lines at 10 and 110),
  y 0..53.33 (`160/3`, the exact width used by the data). Hash inner edges at y 23.58 and 29.75.
- **Normalized so the offense always moves toward +x.** For plays with
  `playDirection == "left"` the exporter applies `x = 120 - x`, `y = 160/3 - y`,
  `o = (o + 180) % 360`, `dir = (dir + 180) % 360`.
- Angles (`o` = facing, `dir` = direction of motion): degrees, 0 = +y, clockwise (90 = +x).
  Verified against movement in the data (`dir` matches `atan2(dx, dy)` to within ~0.7 deg).
  This is exactly Unity's yaw about +Y with +y mapped to +z, so `Quaternion.Euler(0, o, 0)`
  is used with no conversion.
- Heights (y): field at 0, players' bodies 0..0.8, jersey number at 0.85, carried ball at 1.2,
  ball on the turf before the snap at 0.15.
- Throw arc: data has no ball height. From `pass_forward` (or `pass_shovel`) to the first of
  `pass_arrived`, `pass_outcome_*`, or `pass_tipped`, the ball gets `+4*H*t*(1-t)` where
  `H = clamp(0.15 * throw distance, 1, 12)` yards. Ball x/y always come from the data.

## Play JSON schema (schemaVersion 1)

`index.json` (later `GET /plays`):

```jsonc
{
  "schemaVersion": 1,
  "gameId": 2022090800,
  "homeTeam": "LA",
  "visitorTeam": "BUF",
  "plays": [                           // one game, ordered by playId (chronological)
    { "id": "2022090800_56",          // used to fetch the play
      "gameId": 2022090800, "playId": 56,
      "quarter": 1,                    // 5 = overtime
      "gameClock": "15:00",
      "driveNumber": 1,                // see "Drives" below
      "possessionTeam": "BUF",
      "down": 1, "yardsToGo": 10,
      "playType": "pass",              // pass | run | sack | scramble | kneel | spike
      "passResult": "C",               // "" for non-pass plays
      "yardsGained": 6,
      "touchdown": false,
      "description": "(15:00) (Shotgun) J.Allen pass short right to S.Diggs ..." }
  ]
}
```

Drives: plays.csv has no drive id. The exporter numbers drives 1..N per game, starting a
new drive when `possessionTeam` changes and at the start of Q3 and OT. Special teams plays
aren't in the data, so this is an approximation. A drive can span two quarters; the
navigator shows it in both. The client requires `plays` to be grouped contiguously by
quarter and drive (true when sorted by playId).

`{gameId}_{playId}.json` (later `GET /plays/{id}`):

```jsonc
{
  "schemaVersion": 1,
  "gameId": 2022090800,          // games.csv / plays.csv
  "playId": 56,
  "homeTeam": "LA",              // games.homeTeamAbbr
  "visitorTeam": "BUF",          // games.visitorTeamAbbr
  "description": "...",          // plays.playDescription
  "quarter": 1,
  "gameClock": "15:00",
  "driveNumber": 1,
  "down": 1,
  "yardsToGo": 10,
  "possessionTeam": "BUF",       // offense
  "defensiveTeam": "LA",
  "lineOfScrimmage": 35.0,       // normalized field x (from plays.absoluteYardlineNumber)
  "firstDownLine": 45.0,         // LOS + yardsToGo, capped at the goal line (110)
  "passResult": "C",             // C complete, I incomplete, IN interception, S sack, R scramble
  "frameRate": 10,               // frames per second
  "snapFrameId": 157,            // frameId with frameType SNAP
  "players": [
    { "nflId": 46076, "name": "Josh Allen", "jerseyNumber": 17,
      "club": "BUF", "position": "QB",
      "side": "offense" }        // "offense" | "defense"
  ],
  "frames": [                    // ordered by frameId, one entry per tracking frame
    { "frameId": 1,              // 1-based, contiguous
      "frameType": "BEFORE_SNAP", // BEFORE_SNAP | SNAP | AFTER_SNAP
      "event": null,             // e.g. "ball_snap", "pass_forward", "pass_arrived"; null if none
      "ball": { "x": 35.0, "y": 26.6, "s": 0.0 },   // no o/dir for the ball
      "players": [
        { "nflId": 46076, "x": 30.1, "y": 26.4,     // normalized yards
          "o": 88.2, "dir": 91.0,                   // degrees, 0 = +y, clockwise
          "s": 0.4 }                                // speed, yards/s
      ] }
  ]
}
```

Notes for producers (exporter or backend):
- Must be deserializable by Unity `JsonUtility`: no top-level arrays, no dictionaries,
  field names exactly as above. Numbers are rounded to 2 decimals.
- Every frame should contain every player in `players`; the client holds the last known
  sample if one is missing. Players are matched by `nflId`, not array position.
- Bump `schemaVersion` on breaking changes.

## Raw data notes (verified)

- tracking_week_N.csv columns: gameId, playId, nflId, displayName, frameId, frameType, time,
  jerseyNumber, club, playDirection, x, y, s, a, dis, o, dir, event. Missing values are `NA`.
- Ball rows: `displayName == "football"`, `club == "football"`, `nflId`/`jerseyNumber`/`o`/`dir` NA.
- Every frame has 23 rows (22 players + ball). Plays include long pre-snap periods
  (from `huddle_break_offense`), so playback starts 1.5 s before the snap by default.
- `plays.absoluteYardlineNumber` is in raw (unflipped) tracking x, within ~0.5 yd of the ball at the snap.
- Only `tracking_week_1.csv` is currently in `Data/raw/`.
