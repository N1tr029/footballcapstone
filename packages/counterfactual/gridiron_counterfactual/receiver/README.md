# receiver what-if

Part of stream 3 ([docs/roles.md](../../../../docs/roles.md)), beside the play model in
this package. Given where everyone was at the moment of the throw, was there a better
receiver to throw to, and what was it worth?

How it relates to `decision_engine.py`: both ask whether another receiver was the better
throw. The play model forecasts where everyone will be while the ball is in the air, on
the 2026 release. This scores every receiver at the instant of the throw, on the 2025
release, which carries all 22 players and the game situation. What it adds is a value in
expected points and win probability, an availability filter, and scoring by models that
never saw the outcome of the play they judge.

The first question it is being built to answer: **at the moment of the throw, was there a
better receiver to throw to?** It reads the shared `PlayTrack` that `packages/tracking`
produces, so it runs the same on league tracking and on tracking recovered from video.

## Status

Built: data access for the 2025 Big Data Bowl release, the table the model learns from —
every receiver at the moment of every throw — the receiver value model, and the what-if
engine that flags throws where a better receiver was likely available, the conversion of
its gap to win probability, the simulated alternative as a `PlayTrack` the replay can draw,
and a first version of a second mistake, missed tackles. Not done: anyone watching the
flagged plays to say whether they are right, and running any of it on tracking from video.

| | |
|---|---|
| throws | 8,329 (2022 season, weeks 1–9) |
| receivers scored | 38,283 (4.6 per throw) |
| completion rate of the targeted receiver | 67.3% |
| defender within 1 yd → over 8 yd of separation | 27% → 87% complete |
| thrown behind the line → over 20 yd downfield | 84% → 40% complete |
| the target was the most open receiver | 33% of throws |

## The value model

Three predictions for a receiver at the throw: the chance it is caught, yards if it is,
and expected points added (EPA) by the throw. Trained on weeks 1–7, measured on the
throws of weeks 8–9, which it never saw (1,634 of them).

| completion chance | log loss | Brier | AUC |
|---|---|---|---|
| always guess the average | 0.629 | 0.219 | 0.500 |
| separation and depth only | 0.552 | 0.184 | 0.744 |
| **the model** | **0.519** | **0.173** | **0.781** |

It is calibrated: split the held-out throws into five equal groups by predicted chance and
the model said 34 / 58 / 74 / 83 / 90%, and 34 / 57 / 69 / 85 / 94% were caught.

Yards if caught: off by 4.5 yards on average, against 7.2 for guessing the average. EPA is
much noisier — a single throw's EPA swings on a tip or a broken tackle — so its
correlation with what happened is only 0.19, but its five groups are in the right order
and close (predicted −0.32 … +0.69, actual −0.37 … +0.66).

**The model's highest score is not, by itself, the right read.** Scored naively, the
quarterback threw to the model's top receiver on only 29% of throws (random would be 22%).
That is the model being wrong, not the quarterbacks: the receivers it prefers are deeper,
a quarter of them are on go routes, and 58% have their back to the passer at the throw
against 27% of real targets — men who were not yet available to be thrown to. Picking the
highest of four or five noisy scores also overstates the winner. The what-if engine below
exists to deal with both.

## The what-if engine

For each throw: the receiver he chose, the best *available* alternative, and the gap in
expected points between them with a range. A throw is flagged when the gap is at least
0.8 points and nine refits in ten agree it is above zero.

Three rules stand between a score and a flag (`whatif.py` explains each):

- **Availability.** An alternative must look like a receiver quarterbacks do throw to.
  40% of untargeted receivers qualify; 18% of throws have no alternative at all.
- **Uncertainty.** The value model is refit twelve times on resampled throws, and the gap
  has to hold up across them.
- **No hindsight.** Every throw is scored by models fitted on other games, so the model
  never knows how the play it is judging turned out.

Throws more than 5 seconds after the snap are never flagged; see the review below.

Result over all 8,329 throws: **698 flagged (8.4%)**.

| | throws | actual EPA | predicted EPA of the chosen receiver | completed |
|---|---|---|---|---|
| flagged | 698 | -0.17 | -0.29 | 55% |
| not flagged | 7,631 | +0.16 | +0.17 | 68% |

What that does and does not show:

- **Shown:** the flagged throws were the low-value choices. They turned out 0.33 points
  worse than the rest (6 standard errors), and the model was never told the outcomes.
- **Supported, not proven:** that the alternative was better. Nobody threw to him, so
  there is no outcome to check. The nearest test is the 1,786 real throws to receivers
  resembling the flagged alternatives: predicted +0.35, actual +0.38. And the
  alternatives now look like real targets — 37% with their back turned, down from 58%.
- **Overstated:** the size of the gap. The mean flagged gap is 1.16 points, built on
  alternatives scored at +0.85 where similar real throws earn about +0.38. Choosing the
  best of several inflates the winner. Read a flag as "likely a better option", and the
  number as an upper bound.
- **Not known at all:** the play call. An open receiver may not have been in the
  progression.

### Reviewed by eye

The ten largest flags, watched in the tracking viewer and judged one at a time:

| verdict | throws |
|---|---|
| real miss | 5 |
| not really open | 3 |
| throwaway | 2 |

Half held up. Ten is far too few to tune on, and only one thing in it was clear enough to
act on: both throwaways came more than 5 seconds after the snap (5.4 s and 8.2 s) and none
of the other eight did, which is why late throws are now excluded — 30 flags removed.

Nothing measured at the throw separates "real miss" from "not really open" in this sample.
How open the receiver would have been when the ball *arrived* looked promising and does
not do it either: it closes on two of the three "not really open" plays and on one of the
real misses. More reviews are needed before another rule is justified.

## Win probability

A flag is decided in expected points and reported in win probability as well, because a
point is not always worth the same. The conversion is learned from the release itself,
which carries the league's win probability before every play and how far each play moved
it (16,124 plays): given the points a play added and the game it was added in, how far did
win probability move?

| on weeks 8–9 | error | correlation with the league's number |
|---|---|---|
| one fixed rate (2.3% per point) | 0.015 | 0.75 |
| **the model** | **0.011** | **0.87** |

One expected point is worth about 3.4% in a tied game at halftime, 7.2% tied with two
minutes left, and 1.3% up by 21. The median flagged throw is a gap of 2.7% win
probability. Every caveat on the points gap carries through: it is a conversion of an
estimate, and the gap it converts is an upper bound.

## The simulated alternative

`simulate.alternative` returns the play as it would have looked with the ball thrown to
the other receiver, as a `PlayTrack` that passes the shared contract — so the replay
engine can draw it beside the real one.

- Up to the throw it is the real play, unaltered.
- After the throw every player is still the real tracking. Only the ball is invented: it
  leaves the passer at the real release and flies, as a parabola, to where the receiver
  really was when it would have arrived.
- It ends when the ball arrives. The catch did not happen and is not drawn.
- `meta` says `simulated: true`, lists the assumptions, and carries the what-if record.

Of the 150 largest flags, 126 simulate. The other 24 cannot: the real play, and so the
tracking, ended before the ball could have reached the receiver. The assumption that
matters is that defenders do not react to the different throw.

## Missed tackles (first version)

A defender in close contact with the ball carrier who does not bring him down, and how far
the carrier went afterwards. Contact, possession and who went down come from stream 2's
interaction layer; this adds who really carried the ball (from play-by-play) and the cost.

On a 10-game sample (1,069 plays): 279 such contacts, 63 of which cost three yards or more
(5% of plays), median 6.6 yards after the miss. The longest read like missed tackles — a
safety on a 49-yard run, a corner on a 24-yard catch and run.

**Not validated.** This release does not label missed tackles. The nearest check is who
*made* each tackle, which it does label: the layer finds 83% of the credited tacklers but
credits nearly three for every one the release does, so its idea of "had him" is loose
around a pile. Treat the list as candidates to watch, not as a count.

## Run it

The data is not in the repo. On Kaggle, accept the rules for **NFL Big Data Bowl 2025**,
download it somewhere outside the repo, and point `.env` at it:

```
FOOTBALL_BDB2025_DIR=../nfl-big-data-2025
```

```bash
pip install -e packages/tracking -e packages/counterfactual fastapi uvicorn
python -m gridiron_counterfactual.receiver build-cache     # once: weekly CSVs -> one Parquet per game (~1 min)
python -m gridiron_counterfactual.receiver throws          # the receiver-at-throw table (~1 min)
python -m gridiron_counterfactual.receiver train           # fit the value model, print held-out accuracy
python -m gridiron_counterfactual.receiver flag            # score every throw, flag likely missed options (~1 min)
python -m gridiron_counterfactual.receiver top 15          # the largest flagged gaps
python -m gridiron_counterfactual.receiver show 2022092504 2191    # one throw's what-if, as JSON
python -m gridiron_counterfactual.receiver simulate 2022092504 2191    # real + simulated PlayTrack JSON for the replay
python -m gridiron_counterfactual.receiver tackles 10      # missed tackles in 10 games (~20 s a game)
python -m pytest packages/counterfactual/tests -q
```

`FOOTBALL_BDB2025_DIR` goes in your own `.env` (relative paths resolve from the repo root).

Both commands write under `data/bdb2025/`, which is git-ignored.

```python
from gridiron_counterfactual import bdb2025
track = bdb2025.play_track(2022090800, 56)        # a PlayTrack, with offense and positions filled in
```

## The table

One row per (play, receiver running a route) at `pass_forward`, in contract coordinates.

| column | meaning |
|---|---|
| `depth` | yards past the line of scrimmage |
| `lateral` | yards to the passer's right (negative = left) |
| `sideline` | yards to the nearer sideline |
| `throw_dist` | passer to receiver |
| `separation`, `separation_2` | nearest and second-nearest defender |
| `separation_trend` | change in separation over the last 0.5 s; positive = getting more open |
| `defender_speed` | speed of the nearest defender |
| `lane_defenders` | defenders within 2 yd of the straight line from passer to receiver |
| `speed`, `moving_downfield` | the receiver's speed, and the cosine of his direction (1 = straight downfield) |
| `facing_passer` | 1 = looking at the passer, −1 = back turned |
| `pressure` | nearest defender to the passer |
| `time_to_throw`, `n_routes` | seconds from the snap; receivers in the pattern |
| `down`, `yards_to_go`, `yards_to_goal` | the situation; used for value, not for completion chance |
| `targeted`, `complete`, `pass_result`, `yards_gained`, `epa`, `route` | what happened; outcome columns describe the play, so they only mean something on the targeted row |

## Serving it

`python -m gridiron_counterfactual.receiver export` writes what the API serves, for the
games the API is serving (today the one game the Unity app ships, 2022090800):

- `packages/counterfactual/reports/receiver_whatif.jsonl`: one record per scored throw (67).
- `packages/counterfactual/reports/receiver_whatif_plays/{gameId}_{playId}.json`: for each
  flagged throw, the simulated alternative in the Unity app's play schema, marked
  `"simulated": true` with `simulatedFrom` and `simulationStartFrameId` (7 of the 8 flags;
  one ended before the ball could arrive).

`--all` writes every game into `data/bdb2025/` instead, which is git-ignored.

`api.py` serves those files. Play ids are the main API's own (`{gameId}_{playId}`).

| endpoint | returns |
|---|---|
| `GET /receiver-whatif?flagged=true` | scored throws, largest flagged gap first |
| `GET /plays/{id}/receiver-whatif` | one throw's record: passer, receiver, alternative, gap in points and win probability, every receiver, assumptions |
| `GET /plays/{id}/receiver-whatif/simulated` | the throw to the alternative, as a play in the same JSON as the main API's `/plays/{id}` |
| `POST /receiver-whatif/reload` | re-read the files after an export |

It runs on its own beside `apps/api`:

```bash
uvicorn gridiron_counterfactual.receiver.api:app --port 8001     # docs at /docs
```

Or `apps/api` can serve it with one line, which is the API owner's call:

```python
from gridiron_counterfactual.receiver.api import router as receiver_router
app.include_router(receiver_router)
```

Nothing in `apps/api` was changed to make this work; that line is tested against it
in memory.

## Layout

```
bdb2025.py     the 2025 release: cache, tables, PlayTrack, contract-coordinate frames
targets.py     receiver features at the throw, and the table built from them
value.py       completion chance, yards and EPA for a receiver; training and the held-out report
whatif.py      availability, uncertainty, cross-fitting; the flag and the record for one throw
winprob.py     expected points to win probability, learned from the release's own numbers
simulate.py    the alternative as a PlayTrack: real players, an invented ball, ends at the catch point
tackles.py     missed tackles from the interaction layer, and the check against tackle credits
export.py      the records and simulated plays the API serves
api.py         those, over HTTP: a router for apps/api and a standalone app
```

## What will bite you

**Only the targeted receiver has an outcome.** A quarterback does not throw to a man he
can see is covered, so the throws that were made are not a fair sample of the throws that
could have been. A model trained on them will be too generous to receivers nobody threw
to. Any "better option" it reports is an estimate against similar throws, not a fact.

**All of this leans on the ball.** The passer is whoever is at the ball, the throw is an
event in the tracking, and possession for tackles comes from the ball track. Tracking from
video does not have a reliable ball or events yet, so none of this runs on it as it stands.
The throw and the passer would have to be inferred from the players, or supplied.

**The 2025 release calls the ball `football`.** The shared loader looks for a team of
`ball`, so loading this release directly loses the ball track and fails validation.
`bdb2025.play_track` renames it first.

**`hadDropback` is only set on completions** in `player_play.csv`, so the passer is found
from the tracking instead. And **the frame marked `pass_forward` is sometimes after the
release**: taking "nearest to the ball" as the passer named a lineman on 7.5% of throws.
The passer is the quarterback if one is within 8 yards of the ball, else the nearest man.
