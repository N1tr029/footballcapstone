"""The rendering loop, from a shell.

    python -m gridiron_tracking clips                       # what is registered
    python -m gridiron_tracking render nw_michigan          # one clip
    python -m gridiron_tracking render --all                # every clip, one table
    python -m gridiron_tracking render nw_michigan --force detect
    python -m gridiron_tracking add ucf_kickoff data/clips/UCF.webm --start 60 --stop 210

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

    args = ap.parse_args(argv)
    if args.cmd == "render" and not args.all and not args.clip:
        ap.error("name a clip, or pass --all")
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
