# packages/counterfactual — the play model

Stream 3 of [docs/roles.md](../../docs/roles.md). Given the state at the snap: what
should have been called, and what would have happened.

Two things live here.

**`decision_engine.py`** trains x/y next-frame regressors on the NFL Big Data Bowl
2025 full-play tracking release. At each `pass_forward` frame it predicts a
location for every tracked player and writes the original tracking columns to
`reports/player_locations_2025.csv`, alongside a typed per-play review.

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

## Decision review

The CLI fits and saves `models/location_model_2025.joblib` from consecutive
player frames in the selected 2025 games. It predicts each player's x/y at the
pass-forward frame from that player's immediately preceding tracked state. The
location CSV retains the source tracking schema; x and y are modeled predictions
and the other columns describe the corresponding measured pass-forward row. A
typed per-play review is written separately. The `gridiron_counterfactual` video
adapter remains a separate bridge and expects its existing model bundle/schema.

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
   python -c "import kagglehub; path = kagglehub.competition_download('nfl-big-data-bowl-2025', output_dir='data'); print('Downloaded to:', path)"
   ```

   If this raises `kagglehub.exceptions.UnauthenticatedError`, authenticate
   once in the same environment and then rerun the download:

   ```bash
   python -c "import kagglehub; kagglehub.login()"
   ```

   Enter the API token from [Kaggle account settings](https://www.kaggle.com/settings/api).
   You must be signed in to Kaggle and have accepted the competition rules (step 1).
   Alternatively, configure a token using one of the methods in the
   [KaggleHub authentication guide](https://github.com/Kaggle/kagglehub#authenticate).
   The files may be nested inside an outer Kaggle download folder; the loader
   searches recursively for `tracking_week_1.csv`, `plays.csv`, and `player_play.csv`.

4. Review up to four games from week 1:

   ```bash
   python decision_engine.py --data-dir data --weeks 1 --max-games 4
   ```

   If you don't activate the environment, use `.venv/bin/python` in place of
   `python` for the install and run commands.

   The script saves the model, one predicted location row per player at pass
   release in `reports/player_locations_2025.csv`, a per-play `decisions.csv`,
   and full typed reports in `reports/decisions.jsonl`. Use `--locations-output`
   and `--model-output` to choose other destinations. `--weeks` and `--max-games`
   control which data is loaded and which games train the model.

## What the prototype says

For each pass, the model predicts every player's x/y at the `pass_forward` frame.
The separate review measures each offensive player's distance to the nearest
defender using those predicted positions and compares the intended receiver with
other offensive players. If another player is at least 1.5 yards more separated,
the report suggests considering that alternative. This is a transparent geometric
heuristic, not a learned definition of a good decision.

The report includes play context such as down and yards to go from `plays.csv`.
The 2025 tracking release does not label quarterback decision quality, so these
comparisons do not establish what the quarterback saw or whether another throw
would have succeeded. The separation threshold is a review heuristic, not a
calibrated probability.

These reports are exploratory. The competition files focus on player movement,
not labeled quarterback decision quality. Tracking data alone cannot establish
what the quarterback saw, whether the pass was catchable, or what would have
happened after a different throw. Treat `confidence` as a rough sample-coverage
flag, not a calibrated probability. The script does not claim a causal “should
have thrown elsewhere” answer.

### Player coverage

The 2025 release's `tracking_week_N.csv` files contain player locations across
the play. The model forecasts every player row present at pass release, retaining
the source columns in its CSV. Players absent from the tracking frame cannot be
forecast from this dataset.

## Files

- `decision_engine.py` — train on 2025 tracking, emit per-player predicted
  pass-release locations, and write typed per-play suggestions.
- `requirements.txt` — minimal Python dependencies.
- `data/`, `models/`, and `reports/` — local Kaggle data and generated outputs;
  these are excluded from version control.
