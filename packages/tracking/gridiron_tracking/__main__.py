"""The rendering loop, from a shell.

    python -m gridiron_tracking clips                       # what is registered
    python -m gridiron_tracking render nw_michigan          # one clip
    python -m gridiron_tracking render --all                # every clip, one table
    python -m gridiron_tracking render nw_michigan --force detect
    python -m gridiron_tracking add ucf_kickoff data/clips/UCF.webm --start 60 --stop 210
    python -m gridiron_tracking unity nw_michigan            # watch it in the Unity app
    python -m gridiron_tracking calframe oregon_wsu           # a frame to calibrate on
    python -m gridiron_tracking pin oregon_wsu points.json    # the calibrator's output
    python -m gridiron_tracking label                        # click footballs
    python -m gridiron_tracking train-ball                   # fine-tune on them
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


UNITY_PLAYS = Path("data/unity-plays")
"""A StreamingAssets/plays folder for the Unity app, which lives in its own
repository (EllisJeremy/hindsight-frontend) rather than here. Its PlayLoader reads
index.json and then ``{gameId}_{playId}.json`` beside it, so copying this folder into
that checkout's Assets/StreamingAssets/plays is the whole install step — or pass
``--unity-dir`` pointing straight at it."""


def _unity(args) -> int:
    """A rendered clip, written where the Unity app will play it back.

    This is the join between stream 2 and stream 1: the clip has already become a
    PlayTrack, so showing it is a schema translation and not new science. The
    tracking is what it is — read the clip's report.txt before believing the
    animation, because a smooth replay of a bad track looks exactly as convincing
    as a smooth replay of a good one.
    """
    import json

    from . import hindsight, prepare
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

        # Consolidate, side, orient and situate before showing it. The renderer
        # survives 46 tracks — it draws 46 dots — but what it draws is then not a
        # football team, and a play running toward -x looks entirely normal doing it.
        if not args.raw:
            track, prepared = prepare.prepare(track)
            print(f"  {name:<18} {prepared.roster.before} tracks -> "
                  f"{prepared.roster.after} players"
                  + (", mirrored" if prepared.mirrored else "")
                  + (f", {prepared.sides.offense} on offense"
                     if prepared.sides and prepared.sides.confidence > 0 else ""))
            for b in prepared.blockers:
                print(f"      not reviewable: {b[:110]}")

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
        print(f"  {' ':<18} {len(track.frames):>4} frames  {len(track.roster):>3} players"
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


def _calframe(args) -> int:
    """Write one frame of a clip for the field calibrator to work on.

    Registration is the stage that fails on a clip nobody has calibrated, and the
    vision model clears the paint bar on some cameras and not others — a low wide
    sideline shot gives it three usable correspondences and it needs more. The
    fallback is a person naming points, which ``data/calibrate.html`` was built for;
    this is the step before that, so the frame it opens is one from the window that
    is actually going to be rendered.
    """
    import cv2

    spec = ClipSpec.load(args.clip, args.spec_dir)
    frame = args.frame if args.frame is not None else (spec.snap_frame or spec.start)
    if not (spec.start <= frame <= (spec.stop or frame)):
        print(f"frame {frame} is outside {args.clip}'s window "
              f"[{spec.start}:{spec.stop}] — the registration has to come from a "
              f"frame that gets decoded", file=sys.stderr)
        return 1

    cap = cv2.VideoCapture(spec.video)
    got, i = None, 0
    while i <= frame:
        ok, img = cap.read()
        if not ok:
            break
        if i == frame:
            got = img
        i += 1
    cap.release()
    if got is None:
        print(f"could not read frame {frame} of {spec.video}", file=sys.stderr)
        return 1

    out = args.out or Path(f"data/cal_{args.clip}.jpg")
    out.parent.mkdir(parents=True, exist_ok=True)
    cv2.imwrite(str(out), got, [cv2.IMWRITE_JPEG_QUALITY, 95])
    h, w = got.shape[:2]
    print(f"  wrote {out}  ({w}x{h}, frame {frame} of {args.clip})")
    print(f"  open data/calibrate.html, load it, name four to six points, Copy,")
    print(f"  then:  python -m gridiron_tracking pin {args.clip} <that file> "
          f"--frame {frame}")
    return 0


def _pin(args) -> int:
    """Turn the calibrator's clicks into a pinned registration, and score it.

    The calibrator emits field/pixel pairs. This fits the homography, measures it
    against the painted yard lines on the same frame, and only then writes it into
    the spec — so the answer to "is this good enough" arrives before the twenty
    minutes of detection, not after.

    The paint check is the one that cannot be talked into agreeing: every other
    measure of a homography is a round trip through the same map, and a fit that is
    wrong by a factor of two and a half agrees with itself perfectly.
    """
    import json

    import cv2
    import numpy as np

    from .field import SPECS
    from .registration.homography import Homography
    from .registration.vision import paint_alignment

    spec = ClipSpec.load(args.clip, args.spec_dir)
    payload = json.loads(Path(args.points).read_text())
    points = payload.get("points") or []
    if len(points) < 4:
        print(f"{args.points} has {len(points)} points; a homography needs at least "
              f"4, and 6 well-spread ones is the number that behaves", file=sys.stderr)
        return 1

    field_xy = np.array([p["field"] for p in points], dtype=float)
    image_xy = np.array([p["pixel"] for p in points], dtype=float)
    if len(set(map(tuple, field_xy))) != len(field_xy):
        print("two points name the same place on the field", file=sys.stderr)
        return 1

    h = Homography.fit(field_xy, image_xy, threshold_px=args.threshold)

    frame = args.frame if args.frame is not None else (spec.snap_frame or spec.start)
    cap = cv2.VideoCapture(spec.video)
    img, i = None, 0
    while i <= frame:
        ok, f = cap.read()
        if not ok:
            break
        if i == frame:
            img = f
        i += 1
    cap.release()
    if img is None:
        print(f"could not read frame {frame} to check the fit against", file=sys.stderr)
        return 1

    level = spec.register.level or "nfl"
    field_spec = SPECS.get({"college": "ncaa", "high_school": "hs"}.get(level, level),
                           SPECS["nfl"])
    score = paint_alignment(img, h, field_spec)

    print(f"  {args.clip}: {len(points)} points, {field_spec.name} field, frame {frame}")
    shown = f"{score:.2f}x" if np.isfinite(score) else "could not be computed"
    print(f"  paint alignment {shown}   (the bar is {spec.register.min_paint:.2f}; "
          f"near 1.0 is chance)")

    # NaN comes back when there is no paint under the drawn lines at all, and every
    # comparison against NaN is False — so testing `score < bar` alone writes the
    # worst case. Not knowing is not the same as passing.
    if not np.isfinite(score) and not args.force:
        print("\n  NOT written. The paint check could not be computed on this frame: "
              "there is no white under the drawn lines to compare against. Pick a "
              "frame with visible yard lines, or pass --force if you have checked "
              "the overlay yourself.", file=sys.stderr)
        return 1

    if score < spec.register.min_paint and not args.force:
        print("\n  NOT written. Below the bar the drawn lines are finding no more "
              f"paint than chance, and a homography that is self-consistent agrees "
              f"with itself however wrong its scale is. Re-click with points spread "
              f"further apart — opposite sidelines and yard lines far down the field "
              f"— or pass --force if you have checked the overlay yourself.",
              file=sys.stderr)
        return 1

    spec.register.mode = "pinned"
    spec.register.homography = [[float(v) for v in row] for row in np.asarray(h.H)]
    spec.register.frame = frame
    spec.save(args.spec_dir)
    print(f"\n  pinned into {args.spec_dir}/{args.clip}.json")
    print(f"  python -m gridiron_tracking render {args.clip}")
    return 0


def _label(args) -> int:
    """Sample frames from the clips and open the labelling UI.

    The one piece of this project that cannot be automated: COCO's sports-ball
    class finds the down-and-distance graphic, and no amount of tuning changes what
    it was trained on. A few hundred clicks fixes it.
    """
    from . import labelling
    from .label_server import serve

    shots = labelling.load_manifest(args.frames)
    if args.extract or not shots:
        specs = load_all(args.spec_dir) if not args.clip else [
            ClipSpec.load(n, args.spec_dir) for n in args.clip]
        if not specs:
            print(f"no clip specs in {args.spec_dir}", file=sys.stderr)
            return 1
        print(f"sampling every {args.every}th frame of {len(specs)} clips:")
        shots = labelling.extract(specs, every=args.every, out_dir=args.frames)
        if not shots:
            print("no frames extracted", file=sys.stderr)
            return 1

    serve(port=args.port, frame_dir=args.frames, label_file=args.labels)
    return 0


def _train_ball(args) -> int:
    """Export what has been labelled and fine-tune a football detector on it."""
    from . import labelling

    shots = labelling.load_manifest(args.frames)
    labels = labelling.LabelSet.load(args.labels)
    if not labels.labels:
        print(f"nothing labelled in {args.labels}; run `label` first", file=sys.stderr)
        return 1

    print(f"  {labels.summary()}")
    counts = labels.counts()
    if counts["with_a_ball"] < args.min_boxes:
        print(f"  only {counts['with_a_ball']} frames have a ball in them. Below "
              f"{args.min_boxes} a fine-tune learns this clip rather than a football "
              f"— pass --min-boxes to override if you know better.", file=sys.stderr)
        return 1

    data = labelling.to_yolo(shots, labels, out_dir=args.dataset,
                             val_fraction=args.val_fraction,
                             holdout=args.holdout or None)
    if args.export_only:
        print(f"  dataset at {data}")
        return 0

    best = labelling.train(data, base_weights=args.base, epochs=args.epochs,
                           imgsz=args.imgsz, out_dir=args.weights)
    print(f"\n  Now set it on a clip and measure:\n"
          f'    "track": {{"ball": true, "ball_weights": "{best}", "ball_class": 0}}\n'
          f"    python -m gridiron_tracking render <clip>\n"
          f"  and read travelled_yards and measured_fraction, not the detection count "
          f"alone — that has looked like progress twice and been furniture.")
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

    u = sub.add_parser("unity", help="write rendered clips where the Unity app reads them")
    u.add_argument("clip", nargs="*")
    u.add_argument("--all", action="store_true")
    u.add_argument("--runs", type=Path, default=Path("data/runs"))
    u.add_argument("--unity-dir", type=Path, default=UNITY_PLAYS)
    u.add_argument("--raw", action="store_true",
                   help="export the tracker's output as-is, without consolidating "
                        "the roster or orienting the field")
    u.add_argument("--replace", action="store_true",
                   help="index only these clips, dropping the app's own 115 real "
                        "plays from the navigator (the files stay on disk)")
    u.set_defaults(fn=_unity)

    from .labelling import DATASET_DIR, FRAME_DIR, LABEL_FILE, WEIGHTS_DIR

    cf = sub.add_parser("calframe", help="write a frame for the field calibrator")
    cf.add_argument("clip")
    cf.add_argument("--frame", type=int, default=None)
    cf.add_argument("--out", type=Path, default=None)
    cf.set_defaults(fn=_calframe)

    pn = sub.add_parser("pin", help="turn calibrator clicks into a checked registration")
    pn.add_argument("clip")
    pn.add_argument("points", help="the JSON the calibrator's Copy button gives you")
    pn.add_argument("--frame", type=int, default=None)
    pn.add_argument("--threshold", type=float, default=3.0)
    pn.add_argument("--force", action="store_true",
                    help="write it even though it is below the paint bar")
    pn.set_defaults(fn=_pin)

    lb = sub.add_parser("label", help="click footballs so a detector can be trained")
    lb.add_argument("clip", nargs="*", help="which clips to sample; default all")
    lb.add_argument("--extract", action="store_true", help="re-sample the frames")
    lb.add_argument("--every", type=int, default=4,
                    help="sample every Nth frame of each clip's window")
    lb.add_argument("--frames", type=Path, default=FRAME_DIR)
    lb.add_argument("--labels", type=Path, default=LABEL_FILE)
    lb.add_argument("--port", type=int, default=4176)
    lb.set_defaults(fn=_label)

    tb = sub.add_parser("train-ball", help="fine-tune a football detector on the labels")
    tb.add_argument("--frames", type=Path, default=FRAME_DIR)
    tb.add_argument("--labels", type=Path, default=LABEL_FILE)
    tb.add_argument("--dataset", type=Path, default=DATASET_DIR)
    tb.add_argument("--weights", type=Path, default=WEIGHTS_DIR)
    tb.add_argument("--base", default="yolo11n.pt")
    tb.add_argument("--epochs", type=int, default=60)
    tb.add_argument("--imgsz", type=int, default=1280)
    tb.add_argument("--val-fraction", type=float, default=0.2)
    tb.add_argument("--holdout", nargs="*", default=None,
                    help="clips to hold out, by name; choose ones with enough "
                         "labelled balls that the score means something")
    tb.add_argument("--min-boxes", type=int, default=80)
    tb.add_argument("--export-only", action="store_true",
                    help="write the YOLO dataset and stop")
    tb.set_defaults(fn=_train_ball)

    args = ap.parse_args(argv)
    if args.cmd in ("render", "unity") and not args.all and not args.clip:
        ap.error("name a clip, or pass --all")
    return args.fn(args)


if __name__ == "__main__":
    raise SystemExit(main())
