"""The rendering loop, from a shell.

    python -m gridiron_tracking clips                       # what is registered
    python -m gridiron_tracking render nw_michigan          # one clip
    python -m gridiron_tracking render --all                # every clip, one table
    python -m gridiron_tracking render nw_michigan --force detect
    python -m gridiron_tracking add ucf_kickoff data/clips/UCF.webm --start 60 --stop 210
    python -m gridiron_tracking unity nw_michigan            # watch it in the Unity app
    python -m gridiron_tracking unity --all                   # every clip, beside the real plays

``--all`` is the one that matters. A change that lifts one clip and drops three is the
normal outcome of tuning, and the only way to see it is to render them together and read
one table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from .clips import SPEC_DIR, ClipSpec, available, load_all
from .config import load_env


def _render(args) -> int:
    from .render import render

    specs = load_all(args.spec_dir) if args.all else [ClipSpec.load(n, args.spec_dir) for n in args.clip]
    if not specs:
        print(f"no clip specs in {args.spec_dir}; add one with `add`", file=sys.stderr)
        return 1

    results = []
    for spec in specs:
        try:
            results.append(render(spec, root=args.out, force=args.force or ()))
        except Exception as e:  # noqa: BLE001 - one bad clip must not stop the sweep
            print(f"  {spec.name}: FAILED — {type(e).__name__}: {e}", file=sys.stderr)
            if args.raise_errors:
                raise
    print()
    for r in results:
        print(r.summary())
        print()

    if len(results) > 1:
        print(f"{'clip':<18}{'agree':>7}{'stray':>7}{'uncl':>7}{'trks':>6}{'boxes/f':>9}  problems")
        for r in results:
            import numpy as np
            rep = r.report
            print(f"{r.spec.name:<18}"
                  f"{(rep.agreement * 100 if rep else float('nan')):>6.1f}%"
                  f"{(np.mean([f.stray for f in rep.frames]) if rep and rep.frames else float('nan')):>7.1f}"
                  f"{(np.mean([f.unclaimed for f in rep.frames]) if rep and rep.frames else float('nan')):>7.1f}"
                  f"{(len(r.track.roster) if r.track else 0):>6}"
                  f"{(r.n_boxes / r.n_frames if r.n_frames else 0):>9.1f}"
                  f"  {'; '.join(r.problems)[:60]}")
    return 0


def _clips(args) -> int:
    names = available(args.spec_dir)
    if not names:
        print(f"no clip specs in {args.spec_dir}")
        return 0
    for spec in load_all(args.spec_dir):
        win = f"[{spec.start}:{spec.stop if spec.stop is not None else 'end'}]"
        print(f"  {spec.name:<18} {win:<14} {spec.register.mode:<7} "
              f"tiles={spec.detect.tiles} {Path(spec.video).name}")
        if spec.notes:
            print(f"    {spec.notes}")
    return 0


def _add(args) -> int:
    spec = ClipSpec(
        name=args.name, video=args.video, start=args.start, stop=args.stop,
        snap_frame=args.snap_frame, title=args.title or "",
    )
    p = spec.save(args.spec_dir)
    print(f"wrote {p} — edit it, then `render {args.name}`")
    return 0


UNITY_PLAYS = Path("packages/replay-engine/hindsight/Assets/StreamingAssets/plays")
"""Where the Unity app loads plays from. Its PlayLoader reads index.json and then
``{gameId}_{playId}.json`` beside it, so writing here is the whole install step."""


def _unity(args) -> int:
    """A rendered clip, written where the Unity app will play it back.

    This is the join between stream 2 and stream 1: the clip has already become a
    PlayTrack, so showing it is a schema translation and not new science. The
    tracking is what it is — read the clip's report.txt before believing the
    animation, because a smooth replay of a bad track looks exactly as convincing
    as a smooth replay of a good one.
    """
    import json

    from . import hindsight
    from .contract import PlayTrack, validate

    names = available(args.spec_dir) if args.all else args.clip
    if not names:
        print("name a clip, or pass --all", file=sys.stderr)
        return 1

    out = args.unity_dir
    tracks, skipped = [], []

    if not args.replace:
        # The app ships 115 real plays; keeping them means the navigator shows the
        # tracked clips beside measured tracking, which is the comparison worth
        # having in front of anybody judging this.
        for path in hindsight.plays_in(out):
            # Skip this command's own previous output, or re-exporting accumulates
            # a stale copy of every clip under its old play id.
            if path.name.startswith(f"{CLIP_GAME_ID}_"):
                continue
            try:
                tracks.append(hindsight.read_play(path))
            except Exception as e:  # noqa: BLE001
                skipped.append((path.name, f"{type(e).__name__}: {e}"))

    for i, name in enumerate(names):
        track_path = args.runs / name / "track.json"
        if not track_path.exists():
            skipped.append((name, f"no track.json — run `render {name}` first"))
            continue

        track = PlayTrack.from_json(json.loads(track_path.read_text()))
        v = validate(track)

        # The app keys plays on (gameId, playId) and a vision track has neither, so
        # clips are numbered off a base that cannot collide with a real game id.
        track.meta["game_id"] = CLIP_GAME_ID
        # High play ids so clips sort after any real play: the navigator opens on
        # the first play in the index, and that should be measured tracking rather
        # than a clip, unless the clips are all there is (--replace).
        track.meta["play_id"] = CLIP_PLAY_BASE + i
        track.meta.setdefault("description", f"{name} — tracked from video")
        track.meta.setdefault("possessionTeam", "")
        track.meta.setdefault("quarter", 1)
        # Their own drive bucket, so the accordion does not file a clip among the
        # real plays of a drive it had nothing to do with.
        track.meta.setdefault("driveNumber", CLIP_DRIVE)
        track.meta.setdefault("gameClock", "--:--")

        path = hindsight.write_play(out, track)
        tracks.append(track)
        flag = "" if v.ok else f"  [contract: {'; '.join(v.problems)}]"
        print(f"  {name:<18} {len(track.frames):>4} frames  {len(track.roster):>3} tracks"
              f"  -> {path.name}{flag}")

    if not tracks:
        print("nothing written", file=sys.stderr)
        for name, why in skipped:
            print(f"  {name}: {why}", file=sys.stderr)
        return 1

    # One index, possibly two games in it. Its top-level gameId is informational —
    # the client fetches each play by its own id — so it names the majority game
    # rather than whichever happens to sort first.
    from collections import Counter
    majority = Counter(t.meta.get("game_id") for t in tracks).most_common(1)[0][0]
    index = hindsight.write_index(out, tracks, game_id=majority)
    print(f"\n  index.json  {len(tracks)} plays -> {index}")
    for name, why in skipped:
        print(f"  skipped {name}: {why}")
    print("\nIn Unity: Hindsight > Setup Scene (once), then press Play.")
    return 0


CLIP_GAME_ID = 999_000_001
"""Game id for plays that came from a clip rather than a game. Real ids are ten
digits beginning with the date, so this cannot collide with one."""

CLIP_PLAY_BASE = 900_000
CLIP_DRIVE = 99


def main(argv=None) -> int:
    load_env()
    ap = argparse.ArgumentParser(prog="python -m gridiron_tracking")
    ap.add_argument("--spec-dir", type=Path, default=SPEC_DIR)
    sub = ap.add_subparsers(dest="cmd", required=True)

    r = sub.add_parser("render", help="video in, side-by-side out")
    r.add_argument("clip", nargs="*")
    r.add_argument("--all", action="store_true")
    r.add_argument("--out", type=Path, default=Path("data/runs"))
    r.add_argument("--force", nargs="*", choices=["detect", "register"],
                   help="re-run a cached stage")
    r.add_argument("--raise-errors", action="store_true")
    r.set_defaults(fn=_render)

    c = sub.add_parser("clips", help="list the registered clips")
    c.set_defaults(fn=_clips)

    a = sub.add_parser("add", help="register a new clip")
    a.add_argument("name")
    a.add_argument("video")
    a.add_argument("--start", type=int, default=0)
    a.add_argument("--stop", type=int, default=None)
    a.add_argument("--snap-frame", type=int, default=None)
    a.add_argument("--title", default="")
    a.set_defaults(fn=_add)

    u = sub.add_parser("unity", help="write rendered clips where the Unity app reads them")
    u.add_argument("clip", nargs="*")
    u.add_argument("--all", action="store_true")
    u.add_argument("--runs", type=Path, default=Path("data/runs"))
    u.add_argument("--unity-dir", type=Path, default=UNITY_PLAYS)
    u.add_argument("--replace", action="store_true",
                   help="index only these clips, dropping the app's own 115 real "
                        "plays from the navigator (the files stay on disk)")
    u.set_defaults(fn=_unity)

    args = ap.parse_args(argv)
    if args.cmd in ("render", "unity") and not args.all and not args.clip:
        ap.error("name a clip, or pass --all")
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
