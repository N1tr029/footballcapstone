# packages/counterfactual — the play model

Stream 3 of [docs/roles.md](../../docs/roles.md). Given the state at the snap: what
should have been called, and what would have happened.

Two things live here.

**`decision_engine.py`** is Landon's prototype, merged into this repository with its
own history. It trains on the Big Data Bowl 2026 release, forecasts where everyone
will be while the ball is in the air, and writes a typed judgement per play. Its
documentation is below, unchanged.

**`gridiron_counterfactual/`** is the part that connects it to everything else. The
prototype speaks the competition's dialect — its column names, its coordinates, its
compass bearings, its own report shape — and nothing else in the repository speaks
that dialect. These four modules are the translation, and they are deliberately the
only place it happens:

| module | direction | why it exists |
|---|---|---|
| `bdb` | release → `PlayTrack` | so the model's own training plays can be drawn by the renderer |
| `features` | `PlayTrack` → model rows | so the model can score the **vision pipeline's** output, not only Kaggle's |
| `forecast` | predictions → `PlayTrack` | so a counterfactual is a play stream 1 can animate, labelled SIMULATED |
| `recommendation` | report → `Recommendation` | so stream 5 can draw the What-If panel against the contract |

```bash
python -m pytest packages/counterfactual/tests -q
```

## The one that can fail silently

`features` is the load-bearing one, and the reason is worth reading before changing
it. `coordinate_model.joblib` was fitted on the release exactly as it ships: x and y
unmirrored, with `play_direction_code` as a *feature* so the model learns the two
cases separately, and `dir`/`o` as compass bearings. A `PlayTrack` is neither: it is
mirrored so the offense always attacks +x, and its angles are the mathematical
convention.

Hand a model trained in one frame of reference a row from the other and nothing
raises. The features are all in range, the regressors return plausible yards, and
the predicted play is a **reflection** of the real one. `features.round_trip_error`
exists to prove the conversions are exact inverses — release rows in, `PlayTrack`,
back out, measured against what the file started with — and a test asserts it on a
left-direction play, which is the case that carries the mirroring. Delete the
mirror and that test reports a 57-yard error.

## What the model does not produce, and therefore does not claim

The `Recommendation` contract asks for an expected value, a win-probability delta,
and one outcome per defensive response. This model produces none of the three, so
all three are left empty and every gap is listed in `assumptions` — a UI cannot tell
"not modelled" from "modelled as zero" unless it is told, and `recommendation.validate`
fails a report that leaves a number unset without saying why.

In particular `expected_points_added` is **not** the recommendation's expected value.
It is the EPA of the play as actually called, lifted from play-by-play. Copied into
`expected_value` it would produce a panel claiming the recommendation's value is
exactly the result of ignoring the recommendation. It is kept under `review`, beside
the other things that actually happened.

One field was added to the contract rather than bent: `choice_probabilities`. This
model reports a distribution over *which call to make*; the contract's `outcomes` is
a distribution over *defensive responses*. Same shape, different object, and filing
one as the other would mislabel it in every view at once.

---

## The prototype

Original notes, as merged.

This starter project uses NFL Big Data Bowl 2026 tracking data to forecast player
locations during the airborne part of a pass, then create a **post-play review**:
was the targeted receiver predicted to have a reasonable window, or did another
route runner look more open at the end of the tracked flight?

The decision report uses a small typed schema (`decision`, probabilities,
confidence, and numeric evidence), inspired by Jev-style bounded judgements. The
trajectory learner is an ordinary supervised ML model; it does **not** simulate a
fly connectome. A connectome simulation is not a football decision policy, and
the two ideas are used here as contrasting design inspirations: learned movement
prediction, then a simple inspectable decision layer.

## Set up

1. Open the competition page while signed in to Kaggle and accept its rules.
2. On macOS, create and activate a virtual environment with `python3` (the
   system often has `python3` but no `python` command):

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

3. Download the competition files into `data/` with KaggleHub:

   ```bash
   python -c "import kagglehub; path = kagglehub.competition_download('nfl-big-data-bowl-2026-analytics', output_dir='data'); print('Downloaded to:', path)"
   ```

   KaggleHub may ask you to authenticate. Follow its prompt, or configure a
   Kaggle API token as described in the [KaggleHub authentication guide](https://github.com/Kaggle/kagglehub#authenticate).
   The files may be nested inside an outer Kaggle download folder; the loader
   searches recursively for `train/input_2023_w01.csv` and its matching output.

4. Train on three games from week 1 and view the held-out fourth game
   (Cleveland vs. Cincinnati, game ID `2023091002`):

   ```bash
   python decision_engine.py --data-dir data --weeks 1 --max-games 4 --game-id 2023091002
   ```

   If you don't activate the environment, use `.venv/bin/python` in place of
   `python` for the install and run commands.

   The script prints a per-play summary and saves `reports/decisions.csv` (easy
   to open in your IDE or spreadsheet app) plus the full typed reports in
   `reports/decisions.jsonl`. Use another `--game-id` from the game list to
   inspect a different game. Expand training to more weeks with `--weeks 1 2 3`
   and increase `--max-games` to use more games.

## What the prototype says

For each play, the model forecasts player coordinates from the final observed
pre-throw state to the last airborne tracking frame. It measures predicted
distance from the targeted receiver and each other route runner to the nearest
coverage defender. If an alternative route runner is at least 1.5 yards more
separated, the report suggests considering that alternative. The threshold is a
transparent starter heuristic, not a learned definition of a good decision.

`pass_result` and `yards_gained` in the report come from the supplementary data
and describe what actually happened. `C` means complete, `I` incomplete, and
`IN` interception. These outcomes do not prove whether a recommendation was
correct.

If every modeled receiver is predicted to be closer than 3 yards to a defender,
the report recommends a running play; on fourth down it recommends a punt. This
is a simple threshold. The Kaggle download contains pass plays only, so it cannot
provide examples of run-play reviews by itself. To include those, download the
public 2023 play-by-play file from nflverse into the `data/` folder:

```bash
curl -L "https://github.com/nflverse/nflverse-data/releases/download/pbp/play_by_play_2023.csv" \
  -o data/play_by_play_2023.csv
```

With that file present, a run is flagged as unsuccessful if it loses yards,
gains zero yards, loses a fumble, or fails to reach the line to gain on third or
fourth down. Such plays receive `suggest_pass_next_time`. These are rule-based
prompts for review, not learned counterfactuals.

These reports are exploratory. The competition files focus on player movement,
not labeled quarterback decision quality. Tracking data alone cannot establish
what the quarterback saw, whether the pass was catchable, or what would have
happened after a different throw. Treat `confidence` as a rough sample-coverage
flag, not a calibrated probability. The script does not claim a causal “should
have thrown elsewhere” answer.

## Files

- `decision_engine.py` — load a few games, fit two coordinate regressors, emit
  typed per-play suggestions.
- `requirements.txt` — minimal Python dependencies.
- `data/`, `models/`, and `reports/` — local Kaggle data and generated outputs;
  these are excluded from version control.
