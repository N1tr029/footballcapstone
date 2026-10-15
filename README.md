# NFL Play Decision Prototype

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
