"""Command line.

    python -m gridiron_counterfactual build-cache [WEEK ...]   split the raw tracking, once
    python -m gridiron_counterfactual throws                   build the receiver-at-throw table
    python -m gridiron_counterfactual train                    fit the receiver value model, report on weeks 8-9
    python -m gridiron_counterfactual flag                     score every throw, flag likely missed options
    python -m gridiron_counterfactual show GAME_ID PLAY_ID     one throw's what-if, as JSON
    python -m gridiron_counterfactual top [N]                  the N largest flagged gaps
    python -m gridiron_counterfactual simulate GAME_ID PLAY_ID the real play and the alternative, for the replay
    python -m gridiron_counterfactual tackles [N]              missed tackles in N games (default 10; ~20 s a game)
"""

import argparse
import json

import pandas as pd

from . import bdb2025, targets


def main() -> None:
    parser = argparse.ArgumentParser(prog="gridiron_counterfactual")
    sub = parser.add_subparsers(dest="cmd", required=True)
    cache = sub.add_parser("build-cache", help="split BDB 2025 tracking into one Parquet per game")
    cache.add_argument("weeks", nargs="*", type=int, default=list(bdb2025.WEEKS))
    sub.add_parser("throws", help="write the receiver-at-throw table")
    sub.add_parser("train", help="fit the receiver value model and report held-out accuracy")
    sub.add_parser("flag", help="score every throw and flag likely missed options")
    show = sub.add_parser("show", help="print one throw's what-if")
    show.add_argument("game_id", type=int)
    show.add_argument("play_id", type=int)
    sim = sub.add_parser("simulate", help="write the real play and the simulated alternative as PlayTrack JSON")
    sim.add_argument("game_id", type=int)
    sim.add_argument("play_id", type=int)
    tk = sub.add_parser("tackles", help="find missed tackles and check tackle credits against the release")
    tk.add_argument("games", nargs="?", type=int, default=10)
    top = sub.add_parser("top", help="list the largest flagged gaps")
    top.add_argument("n", nargs="?", type=int, default=15)
    args = parser.parse_args()

    if args.cmd == "build-cache":
        bdb2025.build_cache(args.weeks)
    elif args.cmd == "throws":
        table = targets.throw_table(log=print)
        out = throws_path()
        table.to_parquet(out, index=False)
        plays = table.groupby(bdb2025.KEY).ngroups
        print(f"{len(table):,} receivers on {plays:,} throws -> {out}")


    elif args.cmd == "train":
        from . import value

        value.train_and_report(pd.read_parquet(throws_path()))
        print(f"model -> {value.model_path()}")


    elif args.cmd == "flag":
        from . import whatif

        scored, decisions, report = whatif.run(pd.read_parquet(throws_path()), log=print)
        scored.to_parquet(out_dir() / "scored.parquet", index=False)
        decisions.to_parquet(out_dir() / "decisions.parquet", index=False)
        (out_dir() / "decisions.report.json").write_text(json.dumps(report, indent=2))
        print(json.dumps(report, indent=2))
    elif args.cmd == "show":
        from . import whatif

        scored, decisions = _results()
        d = decisions[(decisions["gameId"] == args.game_id) & (decisions["playId"] == args.play_id)]
        if d.empty:
            raise SystemExit(f"no scored throw for {args.game_id}-{args.play_id}")
        play = scored[(scored["gameId"] == args.game_id) & (scored["playId"] == args.play_id)]
        print(json.dumps(whatif.describe(d.iloc[0], play), indent=2))
    elif args.cmd == "simulate":
        from gridiron_tracking.contract import validate

        from . import simulate, whatif

        scored, decisions = _results()
        d = decisions[(decisions["gameId"] == args.game_id) & (decisions["playId"] == args.play_id)]
        if d.empty or pd.isna(d.iloc[0]["alternative"]):
            raise SystemExit(f"no alternative for {args.game_id}-{args.play_id}")
        d = d.iloc[0]
        play = scored[(scored["gameId"] == args.game_id) & (scored["playId"] == args.play_id)]
        real = bdb2025.play_track(args.game_id, args.play_id)
        alt = simulate.alternative(real, str(int(d["passer"])), str(int(d["alternative"])),
                                   what_if=whatif.describe(d, play))
        for name, track in (("real", real), ("simulated", alt)):
            check = validate(track)
            print(f"{name}: {len(track.frames)} frames, t {track.frames[0].t} to {track.frames[-1].t}, "
                  f"contract {'ok' if check.ok else check.problems}")
        out = out_dir() / "whatif" / f"{args.game_id}-{args.play_id}.json"
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text(json.dumps({"real": real.to_json(), "simulated": alt.to_json()}))
        print(f"ball {alt.meta['ball_speed']} yd/s ({alt.meta['ball_speed_source']}), "
              f"in the air {alt.meta['flight_seconds']} s -> {out}")
    elif args.cmd == "tackles":
        from . import tackles

        ids = bdb2025.games()["gameId"].tolist()
        ids = ids[:: max(1, len(ids) // args.games)][: args.games]     # spread across the season
        misses, credits = tackles.scan(ids, log=print)
        misses.to_parquet(out_dir() / "missed_tackles.parquet", index=False)
        print(json.dumps(tackles.check(misses, credits), indent=2))
        costly = misses[misses["yards_after"] >= tackles.MIN_YARDS_AFTER].sort_values("yards_after", ascending=False)
        for r in costly.head(10).itertuples():
            print(f"{r.gameId} {r.playId:>5}  {r.defender_name} on {r.carrier_name}, {r.t:.1f} s after the snap, "
                  f"+{r.yards_after:.0f} yd after")
    elif args.cmd == "top":
        _, decisions = _results()
        names = dict(zip(bdb2025.players()["nflId"], bdb2025.players()["displayName"]))
        desc = bdb2025.plays().set_index(bdb2025.KEY)["playDescription"]
        flagged = decisions[decisions["flagged"]].sort_values("gap_low", ascending=False).head(args.n)
        for r in flagged.itertuples():
            print(f"{r.gameId} {r.playId:>5}  {names.get(r.passer, '?'):<18} threw to {names.get(r.chosen, '?'):<20} "
                  f"better: {names.get(int(r.alternative), '?'):<20} gap {r.gap:+.2f} pts ({r.gap_low:+.2f} to {r.gap_high:+.2f})"
                  f"  win prob {r.wp_gap * 100:+.1f}%")
            print(f"      {desc.get((r.gameId, r.playId), '')[:110]}")


def out_dir():
    return bdb2025.cache_dir().parent


def _results():
    try:
        return pd.read_parquet(out_dir() / "scored.parquet"), pd.read_parquet(out_dir() / "decisions.parquet")
    except FileNotFoundError:
        raise SystemExit("run `python -m gridiron_counterfactual flag` first")


def throws_path():
    return bdb2025.cache_dir().parent / "throws.parquet"


if __name__ == "__main__":
    main()
