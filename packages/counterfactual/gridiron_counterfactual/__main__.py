"""Review plays from a shell, and put the alternative where it can be watched.

    python -m gridiron_counterfactual whatif 2022090800_56     # one play
    python -m gridiron_counterfactual whatif --worst 5         # the five it most disputes
    python -m gridiron_counterfactual whatif --all-passes      # every pass in the set

Each reviewed play writes a second play beside the real one — the model's forecast,
labelled SIMULATED — and re-indexes so the Unity app's navigator shows them next to
each other. Load the real play, watch it, press N, watch what the model thought.

``--worst`` is the one that answers "which plays should we look at": it ranks by how
much more separated the model thinks the best alternative receiver was than the man
who actually got the ball. That is the model's own disagreement with the play, in
yards, and it is the closest thing here to a list of decisions worth reviewing.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from gridiron_tracking import hindsight

from . import review

UNITY_PLAYS = Path("packages/replay-engine/hindsight/Assets/StreamingAssets/plays")
CLIP_GAME_ID = 999_000_001
"""Plays exported from video, which this command skips: they have no ball and no
play state, so there is nothing for the model to key a forecast on."""


def _load(plays_dir: Path, wanted: list[str] | None, passes_only: bool):
    out = []
    for path in hindsight.plays_in(plays_dir):
        if path.name.startswith(f"{CLIP_GAME_ID}_"):
            continue
        if wanted and path.stem not in wanted:
            continue
        track = hindsight.read_play(path)
        # Never review this command's own output. A forecast is not a play that
        # happened, and reviewing one produces a forecast of a forecast — which on
        # a re-run breeds, because each generation is written back into the folder
        # the next run reads.
        if track.meta.get("simulated"):
            continue
        if passes_only and (track.meta.get("passResult") or "") not in ("C", "I", "IN"):
            continue
        out.append((path, track))
    return out


def _whatif(args) -> int:
    plays_dir: Path = args.plays
    if not plays_dir.exists():
        print(f"{plays_dir} does not exist — is the hindsight subtree merged?",
              file=sys.stderr)
        return 1

    wanted = list(args.play) or None
    ranking = args.worst is not None
    loaded = _load(plays_dir, wanted, passes_only=args.all_passes or ranking)
    if not loaded:
        print("no plays matched. Pass a play id like 2022090800_56, or --all-passes.",
              file=sys.stderr)
        return 1

    reviewed = []
    for path, track in loaded:
        try:
            result = review.review_play(track, model_path=args.model)
        except Exception as e:  # noqa: BLE001 - one unreviewable play must not stop the run
            print(f"  {path.stem:<20} skipped — {type(e).__name__}: {e}")
            continue
        reviewed.append(result)

    if not reviewed:
        print("nothing could be reviewed", file=sys.stderr)
        return 1

    if ranking:
        reviewed.sort(key=_advantage, reverse=True)
        reviewed = reviewed[: args.worst]

    taken = {p.stem for p in hindsight.plays_in(plays_dir)}
    # Where this command's previous output lives, keyed by the play it argues
    # with, so a second run replaces a simulation instead of writing a sibling.
    existing = _existing_simulations(plays_dir)
    written = []
    for result in reviewed:
        _print(result)
        if args.no_write:
            continue
        written.append(_write_simulation(result, plays_dir, taken, existing))

    if written:
        kept = [hindsight.read_play(p) for p in hindsight.plays_in(plays_dir)]
        index = hindsight.write_index(plays_dir, kept, game_id=_majority(kept))
        print(f"\n  wrote {len(written)} simulated plays; index.json now lists "
              f"{len(kept)} — {index}")
        print("  In Unity: load the real play, then press N for the alternative.")
    return 0


def _advantage(result: review.Review) -> float:
    if result.recommendation is None:
        return -1.0
    value = result.recommendation.evidence.get(
        "alternative_separation_advantage_yards")
    return float(value) if value is not None else -1.0


def _print(result: review.Review) -> None:
    meta = result.track.meta
    print(f"\n  {meta.get('gameId') or meta.get('game_id')}_{meta.get('play_id')}"
          f"  Q{meta.get('quarter')} {meta.get('gameClock')}"
          f"  {meta.get('down')} & {meta.get('yardsToGo')}")
    print(f"    {str(meta.get('description') or '')[:96]}")

    rec = result.recommendation
    if rec is None:
        print("    no judgement — " + (result.notes[-1] if result.notes else "?"))
        return

    evidence = rec.evidence
    print(f"    call            {rec.call.type}"
          + (f" / {rec.call.concept}" if rec.call.concept else "")
          + f"   ({rec.confidence} confidence)")
    print(f"    separation      target {evidence.get('targeted_receiver_predicted_separation_yards')} yd"
          f"   best alternative {evidence.get('best_alternative_predicted_separation_yards')} yd"
          f"   advantage {evidence.get('alternative_separation_advantage_yards')} yd")
    gained = rec.review.get("yards_gained")
    print(f"    actually        {rec.review.get('pass_result')}"
          + (f"  {gained} yards" if gained is not None else
             "  (yards gained is in index.json, not the play file)"))
    print(f"    expected_value  {rec.expected_value}   win_prob_delta {rec.win_prob_delta}"
          f"      <- unset, see assumptions")


def _existing_simulations(plays_dir: Path) -> dict[tuple, int]:
    """``(game_id, source play id) -> the play id its simulation already has``."""
    out: dict[tuple, int] = {}
    for path in hindsight.plays_in(plays_dir):
        try:
            doc = json.loads(path.read_text())
        except (OSError, ValueError):
            continue
        if doc.get("simulated") and doc.get("simulatedFrom") is not None:
            out[(doc.get("gameId"), int(doc["simulatedFrom"]))] = int(doc["playId"])
    return out


def _write_simulation(result: review.Review, plays_dir: Path, taken: set[str],
                      existing: dict[tuple, int] | None = None) -> Path:
    """The forecast as a play file, sitting beside the real one in the navigator.

    Same quarter and drive so the accordion puts them together, and the next free
    play id so it sorts immediately after. The description leads with SIMULATED,
    because the navigator's hover text is the only thing some viewers will read.
    """
    real, sim = result.track, result.forecast
    game_id = real.meta.get("game_id")
    play_id = int(real.meta.get("play_id") or 0)

    # Reuse the id this play's simulation already has, so re-running replaces it.
    candidate = (existing or {}).get((game_id, play_id))
    if candidate is None:
        candidate = play_id + 1
        while f"{game_id}_{candidate}" in taken:
            candidate += 1
    taken.add(f"{game_id}_{candidate}")

    call = result.recommendation.call.type if result.recommendation else "forecast"
    sim.meta.update({
        "game_id": game_id,
        "play_id": candidate,
        "quarter": real.meta.get("quarter"),
        "driveNumber": real.meta.get("driveNumber"),
        "gameClock": real.meta.get("gameClock"),
        "possessionTeam": real.meta.get("possessionTeam"),
        "defensiveTeam": real.meta.get("defensiveTeam"),
        "down": real.meta.get("down"),
        "yardsToGo": real.meta.get("yardsToGo"),
        "lineOfScrimmage": real.meta.get("lineOfScrimmage"),
        "firstDownLine": real.meta.get("firstDownLine"),
        "description": f"SIMULATED — model's forecast of this throw ({call}). "
                       f"This did not happen.",
        "simulated_from_play": play_id,
    })
    return hindsight.write_play(plays_dir, sim)


def _majority(tracks):
    from collections import Counter

    return Counter(t.meta.get("game_id") for t in tracks).most_common(1)[0][0]


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="python -m gridiron_counterfactual")
    sub = ap.add_subparsers(dest="cmd", required=True)

    w = sub.add_parser("whatif", help="review plays and write the alternatives")
    w.add_argument("play", nargs="*", help="play ids like 2022090800_56")
    w.add_argument("--all-passes", action="store_true")
    w.add_argument("--worst", type=int, metavar="N",
                   help="only the N plays the model most disputes")
    w.add_argument("--plays", type=Path, default=UNITY_PLAYS)
    w.add_argument("--model", type=Path, default=review.DEFAULT_MODEL)
    w.add_argument("--no-write", action="store_true",
                   help="print the reviews without touching the Unity folder")
    w.set_defaults(fn=_whatif)

    args = ap.parse_args(argv)
    if args.cmd == "whatif" and not args.play and not args.all_passes and args.worst is None:
        ap.error("name a play, or pass --all-passes or --worst N")
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
