"""Upload, select a play, calibrate the field, and consume tracking over HTTP.

Scans and renders return job tickets. Automatic registration may need clicked
field points; a failed fit is returned with its reasons. Tracking metadata keeps
actual ball sightings separate from estimates and missing observations.
"""
from __future__ import annotations

import json
import math
import re
import shutil
import subprocess
import threading
import time
import uuid
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Any, Literal, Optional

from fastapi import APIRouter, File, HTTPException, UploadFile
from fastapi.responses import FileResponse, Response
from pydantic import BaseModel, Field, model_validator

from .store import REPO_ROOT

router = APIRouter(prefix="/video", tags=["video"])

UPLOADS = REPO_ROOT / "data" / "uploads"
SPEC_DIR = REPO_ROOT / "data" / "clips" / "specs"
RUNS = REPO_ROOT / "data" / "runs"

# Rendering retains a decoded clip and a local vision model in memory. Serialize
# analysis jobs on the development machine; HTTP polling stays responsive.
_pool = ThreadPoolExecutor(max_workers=1)
_jobs: dict[str, dict[str, Any]] = {}
_lock = threading.Lock()


def _tracking():
    """The tracking package, which is not installed in a bare checkout."""
    import sys

    p = str(REPO_ROOT / "packages" / "tracking")
    if p not in sys.path:
        sys.path.insert(0, p)
    import gridiron_tracking  # noqa: F401

    return gridiron_tracking


def _job(kind: str, fn) -> str:
    """Start work on a thread and hand back a ticket.

    Scanning a quarter takes ten seconds and rendering a clip takes minutes, which
    is past where a request should be held open.
    """
    jid = uuid.uuid4().hex[:12]
    with _lock:
        _jobs[jid] = {"id": jid, "kind": kind, "state": "running",
                      "started": time.time(), "result": None, "error": None}

    def run():
        try:
            out = fn()
            with _lock:
                _jobs[jid].update(state="done", result=out, finished=time.time())
        except Exception as e:  # noqa: BLE001 - the ticket carries the failure
            with _lock:
                _jobs[jid].update(state="failed", error=f"{type(e).__name__}: {e}",
                                  finished=time.time())

    _pool.submit(run)
    return jid


# ------------------------------------------------------------------- models

class Candidate(BaseModel):
    startFrame: int
    endFrame: int
    startSeconds: float
    endSeconds: float
    seconds: float
    playlike: float = Field(description="0-1, how much this looks like a wide shot of live play")
    turf: float
    motion: float


class ClipRequest(BaseModel):
    videoId: str = Field(pattern=r"^[a-f0-9]{12}$")
    name: str = Field(pattern=r"^[A-Za-z0-9][A-Za-z0-9_-]{0,79}$")
    startFrame: int = Field(ge=0)
    stopFrame: int = Field(gt=0, description="Exclusive source-frame boundary")
    snapFrame: Optional[int] = Field(default=None, ge=0)
    title: str = ""
    level: Literal["nfl", "college", "high_school"] = "college"

    @model_validator(mode="after")
    def valid_window(self):
        if self.stopFrame <= self.startFrame:
            raise ValueError("stopFrame must be greater than startFrame")
        if self.snapFrame is not None and not self.startFrame <= self.snapFrame < self.stopFrame:
            raise ValueError("snapFrame must be inside the clip window")
        return self


class Point(BaseModel):
    field: tuple[float, float] = Field(description="yards: x downfield 0-120, y across 0-53.3")
    pixel: tuple[float, float]


class PinRequest(BaseModel):
    points: list[Point]
    frame: Optional[int] = Field(default=None, ge=0)


def _name(name: str) -> str:
    if not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_-]{0,79}", name):
        raise HTTPException(400, "invalid clip name")
    return name


def _load_clip(name: str):
    _tracking()
    from gridiron_tracking.clips import ClipSpec

    _name(name)
    path = SPEC_DIR / f"{name}.json"
    if not path.is_file():
        raise HTTPException(404, f"no clip {name}")
    return ClipSpec.from_json(json.loads(path.read_text()))


def _source_path(spec) -> Path:
    path = Path(spec.video)
    return path if path.is_absolute() else REPO_ROOT / path


def _reference_frame(spec) -> int:
    if spec.register.frame is not None:
        return spec.register.frame
    return spec.snap_frame if spec.snap_frame is not None else spec.start


def _finite(value):
    return round(float(value), 3) if math.isfinite(value) else None


# ------------------------------------------------------------------ uploads

@router.post("/uploads", summary="Upload footage")
def upload(file: UploadFile = File(...)) -> dict:
    import cv2

    UPLOADS.mkdir(parents=True, exist_ok=True)
    suffix = Path(file.filename or "clip.mp4").suffix or ".mp4"
    vid = uuid.uuid4().hex[:12]
    dest = UPLOADS / f"{vid}{suffix}"
    with dest.open("wb") as out:
        shutil.copyfileobj(file.file, out)
    cap = cv2.VideoCapture(str(dest))
    try:
        if not cap.isOpened():
            dest.unlink()
            raise HTTPException(400, "the uploaded file is not a readable video")
        fps = float(cap.get(cv2.CAP_PROP_FPS))
        count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    finally:
        cap.release()
    return {"videoId": vid, "filename": file.filename,
            "bytes": dest.stat().st_size, "path": str(dest.relative_to(REPO_ROOT)),
            "sourceUrl": f"/video/uploads/{vid}/source",
            "media": {"fps": _finite(fps), "frameCount": count,
                      "width": width, "height": height,
                      "seconds": _finite(count / fps) if fps > 0 else None}}


@router.get("/uploads", summary="Footage already uploaded")
def uploads() -> dict:
    UPLOADS.mkdir(parents=True, exist_ok=True)
    return {"videos": [{"videoId": p.stem, "bytes": p.stat().st_size}
                       for p in sorted(UPLOADS.iterdir()) if p.is_file()]}


def _video_path(video_id: str) -> Path:
    if not re.fullmatch(r"[a-f0-9]{12}", video_id):
        raise HTTPException(404, "no such upload")
    hits = list(UPLOADS.glob(f"{video_id}.*")) if UPLOADS.exists() else []
    if not hits:
        raise HTTPException(404, f"no upload {video_id}")
    return hits[0]


@router.get("/uploads/{video_id}/source", summary="Play the uploaded source footage")
def source(video_id: str):
    return FileResponse(_video_path(video_id))


# --------------------------------------------------------------------- scan

@router.post("/{video_id}/scan", summary="Find the parts of the footage that look like plays")
def scan(video_id: str) -> dict:
    path = _video_path(video_id)

    def work():
        _tracking()
        from gridiron_tracking import shots as shotsmod

        found = shotsmod.segment(str(path))
        out = []
        for s in found:
            out.append(Candidate(
                startFrame=s.start_frame, endFrame=s.end_frame,
                startSeconds=round(s.start_t, 2), endSeconds=round((s.end_frame + 1) / s.fps, 2),
                seconds=round(s.seconds, 2), playlike=round(s.playlike(), 3),
                turf=round(s.turf, 3), motion=round(s.motion, 4),
            ).model_dump())
        out.sort(key=lambda c: -c["playlike"])
        return {"videoId": video_id, "shots": len(found), "candidates": out}

    return {"jobId": _job("scan", work)}


@router.get("/jobs/{job_id}", summary="How a scan or a run is getting on")
def job(job_id: str) -> dict:
    with _lock:
        j = dict(_jobs[job_id]) if job_id in _jobs else None
    if j is None:
        raise HTTPException(404, f"no job {job_id}")
    return j


# -------------------------------------------------------------------- clips

@router.post("/clips", summary="Name a window of footage as a clip")
def make_clip(req: ClipRequest) -> dict:
    _tracking()
    from gridiron_tracking.clips import ClipSpec

    path = _video_path(req.videoId)
    if (SPEC_DIR / f"{req.name}.json").exists():
        raise HTTPException(409, f"clip {req.name} already exists")
    spec = ClipSpec(name=req.name, video=str(path.relative_to(REPO_ROOT)),
                    start=req.startFrame, stop=req.stopFrame,
                    snap_frame=req.snapFrame, title=req.title)
    spec.register.level = req.level
    spec.track.pre_teams = True
    spec.track.team_cost = .8
    spec.track.ball = False
    spec.track.ball_motion = True
    spec.track.carrier = False
    written = spec.save(SPEC_DIR)
    return {"clip": req.name, "spec": str(written.relative_to(REPO_ROOT)),
            "registered": False,
            "next": f"POST /video/clips/{req.name}/register"}


@router.get("/clips", summary="Clips that exist")
def clips() -> dict:
    _tracking()
    from gridiron_tracking.clips import ClipSpec, available

    out = []
    for name in available(SPEC_DIR):
        spec = ClipSpec.load(name, SPEC_DIR)
        out.append({"clip": name, "title": spec.title,
                    "registered": spec.register.mode == "pinned",
                    "window": [spec.start, spec.stop],
                    "hasTrack": (RUNS / name / "track.json").exists()})
    return {"clips": out}


@router.get("/clips/{name}/frame", summary="A frame, to click field points on")
def frame(name: str, f: Optional[int] = None):
    _tracking()
    import cv2

    spec = _load_clip(name)
    want = f if f is not None else _reference_frame(spec)
    if want < spec.start or (spec.stop is not None and want >= spec.stop):
        raise HTTPException(400, "frame is outside the clip window")
    cap = cv2.VideoCapture(str(_source_path(spec)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, want)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(404, f"frame {want} could not be read from {spec.video}")
    ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    return Response(content=buf.tobytes(), media_type="image/jpeg",
                    headers={"X-Frame": str(want),
                             "X-Width": str(img.shape[1]), "X-Height": str(img.shape[0])})


# ------------------------------------------------------------- registration

@router.post("/clips/{name}/register", summary="Try to find the field automatically")
def register_auto(name: str) -> dict:
    """Ask the vision model. A refusal is a state, not an error.

    Measured over this repository's nine clips this succeeds about one time in
    seven, so the interesting branch is the other one: ``registered: false`` comes
    back with the reasons, and the caller finishes the job by clicking points.
    """
    _tracking()
    import cv2

    from gridiron_tracking.registration import vision

    spec = _load_clip(name)
    want = _reference_frame(spec)
    cap = cv2.VideoCapture(str(_source_path(spec)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, want)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(400, f"frame {want} could not be read")

    res = vision.auto_register(img, **({"model": spec.register.model} if spec.register.model else {}))
    if not res.ok or not math.isfinite(res.paint) or res.paint < spec.register.min_paint:
        return {"clip": name, "registered": False, "frame": want,
                "verdict": res.verdict, "problems": res.problems, "notes": res.notes,
                "correspondences": len(res.correspondences),
                "turf": _finite(res.turf),
                "paint": _finite(res.paint),
                "next": (f"POST /video/clips/{name}/pin with 4-6 points, far apart: "
                         f"both sidelines and yard lines well down the field. "
                         f"GET /video/clips/{name}/frame for the picture.")}

    spec.register.mode = "pinned"
    spec.register.frame = want
    spec.register.homography = res.homography.H.tolist()
    spec.register.level = {"NFL": "nfl", "NCAA": "college", "NFHS": "high_school"}[res.spec.name]
    spec.save(SPEC_DIR)
    return {"clip": name, "registered": True, "frame": want, "verdict": res.verdict,
            "turf": _finite(res.turf), "paint": _finite(res.paint),
            "next": f"POST /video/clips/{name}/run"}


@router.post("/clips/{name}/pin", summary="Register from clicked field points")
def pin(name: str, req: PinRequest) -> dict:
    """Four to six points, and the paint decides whether they are good enough.

    Scored the same way as the automatic fit and written only if it clears the bar,
    because every other measure of a homography is a round trip through the same
    map: a fit that is wrong by a factor of two agrees with itself perfectly.
    """
    _tracking()
    import cv2
    import numpy as np

    from gridiron_tracking.registration.homography import Homography
    from gridiron_tracking.registration.vision import paint_alignment, spec_for

    spec = _load_clip(name)
    if len(req.points) < 4:
        raise HTTPException(400, f"{len(req.points)} points; a homography needs at "
                                 f"least 4, and 6 well-spread ones is the number "
                                 f"that behaves")

    field_xy = np.array([p.field for p in req.points], float)
    image_xy = np.array([p.pixel for p in req.points], float)
    if len({tuple(r) for r in field_xy}) != len(field_xy):
        raise HTTPException(400, "two points name the same place on the field")

    try:
        h = Homography.fit(field_xy, image_xy)
    except (ValueError, np.linalg.LinAlgError) as e:
        raise HTTPException(400, f"field points do not define a homography: {e}") from e
    want = req.frame if req.frame is not None else _reference_frame(spec)
    if want < spec.start or (spec.stop is not None and want >= spec.stop):
        raise HTTPException(400, "frame is outside the clip window")
    cap = cv2.VideoCapture(str(_source_path(spec)))
    cap.set(cv2.CAP_PROP_POS_FRAMES, want)
    ok, img = cap.read()
    cap.release()
    if not ok:
        raise HTTPException(400, f"frame {want} could not be read")

    field_spec = spec_for(spec.register.level or "college")
    paint = float(paint_alignment(img, h, field_spec))
    good = math.isfinite(paint) and paint >= spec.register.min_paint
    if good:
        spec.register.mode = "pinned"
        spec.register.frame = want
        spec.register.homography = h.H.tolist()
        spec.save(SPEC_DIR)
    return {"clip": name, "registered": good, "frame": want,
            "paintAlignment": _finite(paint), "bar": spec.register.min_paint,
            "why": ("the drawn yard lines find this much more paint than empty turf; "
                    f"below {spec.register.min_paint:g} the fit is "
                    "not written"),
            "next": (f"POST /video/clips/{name}/run" if good else
                     "click again, further apart: opposite sidelines, and yard "
                     "lines well down the field")}


# ----------------------------------------------------------------- the data

@router.post("/clips/{name}/run", summary="Turn the clip into tracking data")
def run(name: str) -> dict:
    _tracking()
    _load_clip(name)

    def work():
        from gridiron_tracking import render as rendermod
        spec = _load_clip(name)
        spec.video = str(_source_path(spec))
        for obj, key in ((spec.detect, "weights"), (spec.track, "ball_weights"),
                         (spec.track, "carrier_weights")):
            value = getattr(obj, key)
            if value and not Path(value).is_absolute() and (REPO_ROOT / value).exists():
                setattr(obj, key, str(REPO_ROOT / value))
        res = rendermod.render(spec, root=RUNS, log=lambda *a, **k: None)
        assets = {}
        problems = list(res.problems)
        if res.track is not None:
            assets["track"] = f"/video/clips/{name}/track"
            for source, dest in (("sidebyside.mp4", "reconstruction.mp4"),
                                 ("ball-preview.mp4", "ball-tracking.mp4")):
                path = RUNS / name / source
                if path.exists() and (source != "ball-preview.mp4" or
                                      res.track.meta.get("ball", {}).get("image_observations")):
                    if _browser_video(path, path.with_name(dest)):
                        assets["reconstruction" if dest == "reconstruction.mp4" else "ballPreview"] = (
                            f"/video/clips/{name}/{dest}")
                    else:
                        problems.append("ffmpeg is unavailable; install it for browser-playable previews")
        return {"clip": name, "problems": problems,
                "stages": dict(res.stages), "seconds": round(res.seconds, 1),
                "hasTrack": res.track is not None, "assets": assets,
                "teams": res.track.meta.get("teams") if res.track is not None else None,
                "ball": res.track.meta.get("ball") if res.track is not None else None}

    return {"jobId": _job("run", work), "clip": name}


def _browser_video(source: Path, dest: Path) -> bool:
    ffmpeg = shutil.which("ffmpeg")
    if ffmpeg is None:
        return False
    subprocess.run([ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-i", str(source),
                    "-c:v", "libx264", "-preset", "veryfast", "-crf", "21", "-threads", "2",
                    "-pix_fmt", "yuv420p", "-movflags", "+faststart", str(dest)],
                   check=True, timeout=600, capture_output=True)
    return True


@router.get("/clips/{name}/track", summary="The tracking data for a clip")
def track(name: str):
    _name(name)
    path = RUNS / name / "track.json"
    if not path.exists():
        raise HTTPException(404, f"no track for {name} — run it first, and if the "
                                 f"run refused, read its problems")
    return json.loads(path.read_text())


@router.get("/clips/{name}/report", summary="How much of the clip to believe")
def report(name: str):
    _name(name)
    path = RUNS / name / "report.txt"
    if not path.exists():
        raise HTTPException(404, f"no report for {name}")
    return Response(content=path.read_text(), media_type="text/plain")


@router.get("/clips/{name}/sidebyside.mp4", summary="The clip beside the reconstruction")
def sidebyside(name: str):
    _name(name)
    path = RUNS / name / "sidebyside.mp4"
    if not path.exists():
        raise HTTPException(404, f"no video for {name}")
    return FileResponse(path, media_type="video/mp4")


@router.get("/clips/{name}/reconstruction.mp4", summary="Browser-playable tracking replay")
def reconstruction(name: str):
    return _preview(name, "reconstruction.mp4")


@router.get("/clips/{name}/ball-tracking.mp4", summary="Browser-playable slow-motion ball preview")
def ball_preview(name: str):
    return _preview(name, "ball-tracking.mp4")


def _preview(name: str, filename: str):
    _name(name)
    path = RUNS / name / filename
    if not path.is_file():
        raise HTTPException(404, f"no {filename} for {name}")
    return FileResponse(path, media_type="video/mp4")
