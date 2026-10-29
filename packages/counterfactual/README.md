# packages/counterfactual — the play model

Stream 3 of [docs/roles.md](../../docs/roles.md). Given the state at the snap: what
should have been called, and what would have happened.

Two things live here.

**`decision_engine.py`** reviews pass plays on the NFL Big Data Bowl 2025 release
(all 22 players, whole play). It freezes every player at the moment of the throw,
forecasts where each will be when the ball arrives, and asks whether a route
runner other than the targeted receiver was projected to be clearly more open. It
writes a typed report per play and a replay per pass for the Unity app. Details
under [Decision review](#decision-review). Its 2026 `decisions()` function is kept
unchanged for the `whatif` command below.

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

```
tracking at the throw ──> forecast at the catch ──> separation per receiver ──> verdict + reason
                                                                             └─> replay for Unity
```

### The forecast

For every player on every pass, the inputs are his state at the throw (position
relative to the line, speed, acceleration, direction, facing), where he was 0.5 s
and 1 s earlier, how long the ball will be in the air, which side he's on, whether
he ran a route, his position group, and where the nearest opponent is. The target is
how far he moves before the ball arrives. Two gradient-boosted regressors (x and y)
are saved to `models/forecast_2025.joblib`.

Nothing after the throw goes in, so the forecast judges the throw from the
quarterback's side of it. It does not get the ball's landing spot either: defenders
are forecast from their own momentum, not from where the pass actually went.

Scored on whole weeks the model was not trained on (leave one week out, weeks 1-9,
6,792 passes, 149,424 player positions):

| | mean error at the catch | median | 90th percentile |
|---|---|---|---|
| **forecast** | **0.77 yd** | **0.48 yd** | **1.66 yd** |
| keep running at current speed | 1.34 yd | 0.88 yd | 2.90 yd |
| stand still | 3.05 yd | 2.32 yd | 6.14 yd |

That is 43% less error than the speed baseline. `reports/forecast_2025_evaluation.json`
holds the numbers from the last run.

### The verdict

Separation is a player's distance to the nearest defender, from the forecast
positions at the catch.

| verdict | when |
|---|---|
| `consider_alternative` | a route runner is projected **5+ yards** more open than the target |
| `target_was_reasonable` | nobody is |
| `suggest_run_play` / `suggest_punt` (4th down) | nobody, target included, is projected 3+ yards clear |
| `suggest_pass_next_time` | a run that lost yards, gained nothing, or came up short on 3rd/4th down (play-by-play only) |

Alternatives must have run a route (`player_play.wasRunningRoute`), must not be the
quarterback, and must be past the line of scrimmage when the ball arrives, so a wide
open 2-yard checkdown can't "beat" a downfield completion.

Checked against what actually happened, across weeks 1-9:

- projected separation at the catch tracks measured separation closely (correlation
  0.94 for the target)
- about 11% of throws are flagged; flagged throws completed 64% of the time vs 84%
  for the rest
- but flagged throws' expected points are only slightly lower (+0.35 vs +0.40):
  being open predicts *completion* well, not *value*. Catch probability and what a
  catch is worth (the receiver what-if model in PR #6) are what would close that gap.

The `probabilities` on a report are fixed heuristic values, not fitted, and
`confidence` is a rough flag. None of this says what the quarterback saw.

### Outputs

| file | what |
|---|---|
| `reports/decisions.jsonl` / `.csv` | one report per play of the reviewed game, with the reason in plain English and evidence (projected and actual separation, frames, ids) |
| `reports/suggested_plays.jsonl` | one replay per reviewed pass: recorded tracking from 1 s before the snap to the throw, then the forecast, with the ball flying in a straight line to the recommended receiver. Served by `apps/api` at `/plays/{id}/suggested` |
| `reports/forecast_2025.csv` | each player's position at the throw, forecast at the catch, and actual at the catch, for the reviewed game |
| `reports/forecast_2025_evaluation.json` | held-out error of the forecast vs the baselines |
| `models/forecast_2025.joblib` | the forecast, trained on every week given |

In a replay, frames up to the throw are tracking, untouched. After it, each player
follows a smooth path from his speed at the throw to his forecast spot, with facing
following the path (`o_source: from_velocity`). `ball_snap` and `pass_forward` are
recorded; `pass_arrived` is simulated.

## Set up

1. Get the Big Data Bowl 2025 files from Kaggle (accept the competition rules
   first). Either the whole download, or the smaller subset that
   `make_bdb_subset.py` writes (the throw, the catch and 1 s of history for every
   pass, plus full tracking for game 2022090800) works.
2. Install, with scikit-learn pinned to 1.8 because its saved models don't load
   across minor versions:

   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   python -m pip install -r requirements.txt
   ```

3. Run it (about a minute on all nine weeks):

   ```bash
   python decision_engine.py --data-dir path/to/big-data-bowl-2025
   ```

   `--game-id` picks the game to write reports and replays for (default
   2022090800, the one the Unity app ships with), `--weeks` limits the data,
   `--advantage-yards` changes the 5-yard bar. Outputs go to `reports/` and
   `models/` next to the script.

Keep the Kaggle files out of git: the repository is public.

## Files

- `decision_engine.py` — the 2025 forecast and review, and the 2026 `decisions()`
  the `whatif` command uses.
- `requirements.txt` — dependencies.
- `models/` — `forecast_2025.joblib` (2025) and `coordinate_model.joblib` (2026, for `whatif`).
- `reports/` — generated outputs listed above.
