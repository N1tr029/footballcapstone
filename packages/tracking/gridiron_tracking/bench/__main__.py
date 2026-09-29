"""Run the benchmark.

    python -m gridiron_tracking.bench synthetic --csv tracking.csv --plays 5
    python -m gridiron_tracking.bench ablation  --csv tracking.csv --plays 5

``synthetic`` is the tuning loop: score a change in yards. ``ablation`` is the
diagnostic — the same plays run under conditions that differ by one thing each, so
the cost of every stage is a subtraction rather than a guess. Run it whenever the
headline number moves and you do not know why.

Results append to a leaderboard CSV so a change is a diff, not a memory.
"""

from __future__ import annotations

import argparse
import csv
import sys
from datetime import datetime, timezone
from pathlib import Path

from ..detect.synthetic import NoiseModel
from ..sources import ngs
from . import synthetic_roundtrip


def _load_plays(csv_path: str, limit: int, play_ids: list[str] | None):
    df = ngs.load_tracking(csv_path)
    if play_ids:
        wanted = [type(df["play_id"].iloc[0])(p) for p in play_ids]
    else:
        # Only plays with a real snap: a kickoff has no t = 0 and scoring one
        # measures the clock, not the pipeline.
        snapped = df[df["event"].astype(str) == "ball_snap"]["play_id"].unique()
        wanted = list(snapped[:limit])

    out = []
    for pid in wanted:
        try:
            # Direction is forced rather than inferred. A benchmark that mirrors half
            # its own ground truth is worse than no benchmark, and the 2017 release
            # carries no playDirection to check against.
            out.append((pid, ngs.to_play_track(df, play_id=pid, play_direction="right")))
        except ValueError as e:
            print(f"  skipping play {pid}: {e}", file=sys.stderr)
    return out


def _append(path: Path, rows: list[dict]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    new = not path.exists()
    with path.open("a", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        if new:
            w.writeheader()
        w.writerows(rows)


def _mean(vals):
    vals = [v for v in vals if v == v]
    return sum(vals) / len(vals) if vals else float("nan")


def _run_condition(plays, label: str, **kw):
    scores = []
    for pid, truth in plays:
        rt = synthetic_roundtrip(truth, **kw)
        rt.score.play = str(pid)
        scores.append(rt.score)
    return label, scores


def _summary(label: str, scores) -> dict:
    return {
        "condition": label,
        "plays": len(scores),
        "median_yd": round(_mean([s.median_yards for s in scores]), 3),
        "p95_yd": round(_mean([s.p95_yards for s in scores]), 3),
        "worst_yd": round(max((s.max_yards for s in scores), default=float("nan")), 3),
        "coverage": round(_mean([s.coverage for s in scores]), 3),
        "id_switches": sum(s.id_switches for s in scores),
        "misses": sum(s.misses for s in scores),
        "false_pos": sum(s.false_positives for s in scores),
        "speed_mae": round(_mean([s.speed_mae for s in scores]), 3),
        "dir_moving_deg": round(_mean([s.dir_mae_moving_deg for s in scores]), 1),
        "passes_bar": sum(1 for s in scores if s.meets()),
    }


def _print_table(rows: list[dict]) -> None:
    if not rows:
        return
    cols = list(rows[0].keys())
    widths = {c: max(len(c), *(len(str(r[c])) for r in rows)) for c in cols}
    print("  " + "  ".join(c.ljust(widths[c]) for c in cols))
    print("  " + "  ".join("-" * widths[c] for c in cols))
    for r in rows:
        print("  " + "  ".join(str(r[c]).ljust(widths[c]) for c in cols))


def cmd_synthetic(args) -> int:
    plays = _load_plays(args.csv, args.plays, args.play)
    if not plays:
        print("no plays loaded", file=sys.stderr)
        return 1

    noise = NoiseModel(dropout=args.dropout, crowd_dropout=args.crowd_dropout)
    rows = []
    for pid, truth in plays:
        rt = synthetic_roundtrip(
            truth, fps=args.fps, click_noise_px=args.click_noise,
            box_kind=args.kind, labelled=not args.unlabelled, noise=noise,
        )
        rt.score.play = str(pid)
        if args.verbose:
            print(rt.score.table())
            print()
        row = rt.score.to_row()
        row["registration_rms_yd"] = round(rt.registration_rms_yards, 4)
        rows.append(row)

    _print_table(rows)
    print()
    passed = sum(1 for r in rows if r["median_yd"] <= 1.0 and r["p95_yd"] <= 2.0 and r["id_switches"] <= 5)
    print(f"  {passed}/{len(rows)} plays inside the bar (median <= 1.0 yd, p95 <= 2.0 yd, <= 5 switches)")

    if args.leaderboard:
        stamp = datetime.now(timezone.utc).isoformat(timespec="seconds")
        _append(Path(args.leaderboard), [{"when": stamp, "note": args.note, **r} for r in rows])
        print(f"  appended to {args.leaderboard}")
    return 0


def cmd_ablation(args) -> int:
    plays = _load_plays(args.csv, args.plays, args.play)
    if not plays:
        print("no plays loaded", file=sys.stderr)
        return 1

    clean = NoiseModel(dropout=0.0, crowd_dropout=0.0, false_positives_per_frame=0.0, box_jitter_px=0.0)
    real = NoiseModel()

    conditions = [
        ("ideal: ids given, no noise, perfect clicks",
         dict(labelled=True, noise=clean, click_noise_px=0.0, box_kind="body")),
        ("+ box jitter and dropout",
         dict(labelled=True, noise=real, click_noise_px=0.0, box_kind="body")),
        ("+ operator clicks 2px off",
         dict(labelled=True, noise=real, click_noise_px=2.0, box_kind="body")),
        ("+ tracker associates by distance, not identity",
         dict(labelled=False, noise=real, click_noise_px=2.0, box_kind="body")),
        ("helmet boxes instead of body boxes",
         dict(labelled=False, noise=real, click_noise_px=2.0, box_kind="helmet")),
    ]

    rows = []
    for label, kw in conditions:
        _, scores = _run_condition(plays, label, fps=args.fps, **kw)
        rows.append(_summary(label, scores))

    _print_table(rows)
    print()
    print("  Each row adds one thing to the row above it, so a jump in a column is")
    print("  that stage's cost. Read it downward before changing anything.")
    return 0


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="gridiron_tracking.bench", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)

    def common(p):
        p.add_argument("--csv", required=True, help="an NGS tracking CSV")
        p.add_argument("--plays", type=int, default=5, help="how many snapped plays to score")
        p.add_argument("--play", action="append", help="score a specific play id (repeatable)")
        p.add_argument("--fps", type=float, default=30.0, help="virtual camera frame rate")

    s = sub.add_parser("synthetic", help="score the pipeline against known tracking")
    common(s)
    s.add_argument("--click-noise", type=float, default=2.0, help="pixels of operator click error")
    s.add_argument("--kind", default="body", choices=("body", "helmet", "torso"))
    s.add_argument("--unlabelled", action="store_true", help="make the tracker work out identities")
    s.add_argument("--dropout", type=float, default=0.02)
    s.add_argument("--crowd-dropout", type=float, default=0.35)
    s.add_argument("--leaderboard", default=None, help="append results to this CSV")
    s.add_argument("--note", default="", help="what changed, for the leaderboard")
    s.add_argument("--verbose", action="store_true")
    s.set_defaults(func=cmd_synthetic)

    a = sub.add_parser("ablation", help="attribute the error to each stage")
    common(a)
    a.set_defaults(func=cmd_ablation)

    args = ap.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
