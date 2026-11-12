"""The frontend's upload, calibration, analysis-job and playback contract."""
import json
import shutil
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
from fastapi.testclient import TestClient

import api.video as video
from api.main import app
from gridiron_tracking.clips import ClipSpec
from gridiron_tracking.contract import PlayTrack, TrackingFrame
from gridiron_tracking.registration.homography import Homography
from gridiron_tracking.registration.vision import AutoRegistration
from gridiron_tracking import field


@pytest.fixture
def source_video(tmp_path):
    cv2 = pytest.importorskip("cv2")
    path = tmp_path / "source.mp4"
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 30, (320, 180))
    assert writer.isOpened()
    for _ in range(12):
        writer.write(np.full((180, 320, 3), (40, 110, 40), np.uint8))
    writer.release()
    return path


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setattr(video, "REPO_ROOT", tmp_path)
    monkeypatch.setattr(video, "UPLOADS", tmp_path / "uploads")
    monkeypatch.setattr(video, "SPEC_DIR", tmp_path / "specs")
    monkeypatch.setattr(video, "RUNS", tmp_path / "runs")
    with TestClient(app) as client:
        yield client


def _clip(client, source_video):
    uploaded = client.post("/video/uploads", files={"file": ("play.mp4", source_video.read_bytes(), "video/mp4")})
    assert uploaded.status_code == 200
    request = {"videoId": uploaded.json()["videoId"], "name": "play", "startFrame": 0,
               "stopFrame": 12, "snapFrame": 0, "level": "college"}
    response = client.post("/video/clips", json=request)
    assert response.status_code == 200
    return request, uploaded.json()


def _finished(client, job_id):
    deadline = time.monotonic() + 5
    while time.monotonic() < deadline:
        response = client.get(f"/video/jobs/{job_id}")
        assert response.status_code == 200
        state = response.json()
        if state["state"] != "running":
            return state
        time.sleep(.01)
    pytest.fail("analysis job did not finish")


def test_upload_metadata_source_and_motion_tracking_defaults(client, source_video):
    _, uploaded = _clip(client, source_video)
    assert uploaded["media"] == {"fps": 30, "frameCount": 12, "width": 320, "height": 180, "seconds": .4}
    assert client.get(uploaded["sourceUrl"]).content == source_video.read_bytes()
    spec = ClipSpec.load("play", video.SPEC_DIR)
    assert spec.track.ball_motion and spec.track.pre_teams
    assert not spec.track.ball and not spec.track.carrier
    assert client.get("/video/clips").json()["clips"][0]["hasTrack"] is False


def test_invalid_video_is_refused_and_removed(client):
    response = client.post("/video/uploads", files={"file": ("bad.mp4", b"not a video", "video/mp4")})
    assert response.status_code == 400
    assert list(video.UPLOADS.iterdir()) == []


@pytest.mark.parametrize("change", [{"name": "../escape"}, {"stopFrame": 0},
                                     {"snapFrame": 12}, {"level": "unknown"}])
def test_clip_request_rejects_bad_names_windows_and_levels(client, source_video, change):
    request, _ = _clip(client, source_video)
    assert client.post("/video/clips", json=request | change).status_code == 422


def test_duplicate_clip_does_not_overwrite_calibration(client, source_video):
    request, _ = _clip(client, source_video)
    before = (video.SPEC_DIR / "play.json").read_bytes()
    assert client.post("/video/clips", json=request).status_code == 409
    assert (video.SPEC_DIR / "play.json").read_bytes() == before


def test_scan_uses_real_shot_properties_and_returns_a_pollable_job(client, source_video):
    _, uploaded = _clip(client, source_video)
    response = client.post(f"/video/{uploaded['videoId']}/scan")
    job = _finished(client, response.json()["jobId"])
    assert job["state"] == "done", job
    candidate = job["result"]["candidates"][0]
    assert candidate["startFrame"] == 0 and candidate["endFrame"] == 11
    assert candidate["endSeconds"] == .4


def test_frame_zero_and_cross_origin_calibration_headers(client, source_video, monkeypatch):
    _clip(client, source_video)
    # Starting uvicorn from apps/api must still resolve repository-relative footage.
    monkeypatch.chdir(video.REPO_ROOT / "specs")
    response = client.get("/video/clips/play/frame?f=0", headers={"Origin": "http://localhost:5173"})
    assert response.status_code == 200
    assert response.headers["content-type"] == "image/jpeg"
    assert response.headers["x-frame"] == "0"
    assert response.headers["x-width"] == "320"
    assert "X-Width" in response.headers["access-control-expose-headers"]
    assert client.get("/video/clips/play/frame?f=12").status_code == 400


def test_automatic_registration_saves_the_actual_homography_protocol(client, source_video, monkeypatch):
    _clip(client, source_video)
    from gridiron_tracking.registration import vision
    h = Homography(np.eye(3))
    monkeypatch.setattr(vision, "auto_register", lambda image, **kw: AutoRegistration(
        homography=h, spec=field.NCAA, verdict="good", turf=.9, paint=3.0))
    response = client.post("/video/clips/play/register")
    assert response.status_code == 200 and response.json()["registered"] is True
    spec = ClipSpec.load("play", video.SPEC_DIR)
    assert spec.register.homography == h.H.tolist()
    assert spec.register.level == "college" and spec.register.frame == 0


def test_registration_refusal_is_a_state_with_finite_json(client, source_video, monkeypatch):
    _clip(client, source_video)
    from gridiron_tracking.registration import vision
    monkeypatch.setattr(vision, "auto_register", lambda image, **kw: AutoRegistration(
        problems=["not enough field landmarks"]))
    response = client.post("/video/clips/play/register")
    assert response.status_code == 200
    assert response.json()["registered"] is False
    assert response.json()["paint"] is None
    assert ClipSpec.load("play", video.SPEC_DIR).register.mode == "auto"


@pytest.mark.parametrize("paint,accepted", [(3.0, True), (1.1, False), (float("nan"), False)])
def test_clicked_calibration_obeys_the_paint_gate(client, source_video, monkeypatch, paint, accepted):
    _clip(client, source_video)
    from gridiron_tracking.registration import vision
    seen = []

    def score(image, homography, field_spec):
        seen.append((homography, field_spec))
        return paint

    monkeypatch.setattr(vision, "paint_alignment", score)
    points = [{"field": [x, y], "pixel": [2*x+5, 2*y+8]}
              for x, y in [(10, 5), (60, 5), (10, 45), (60, 45), (35, 25), (90, 30)]]
    response = client.post("/video/clips/play/pin", json={"points": points, "frame": 0})
    assert response.status_code == 200
    assert response.json()["registered"] is accepted
    assert seen[0][1] == field.NCAA
    assert isinstance(seen[0][0], Homography)
    spec = ClipSpec.load("play", video.SPEC_DIR)
    assert (spec.register.mode == "pinned") is accepted


def test_run_calls_current_renderer_and_exposes_frontend_artifacts(client, source_video, monkeypatch):
    _clip(client, source_video)
    from gridiron_tracking import render
    observed = []

    def fake_render(spec, root, log):
        observed.append((spec, root))
        path = root / spec.name
        path.mkdir(parents=True)
        track = PlayTrack(frames=[TrackingFrame(t=0,events=["ball_snap"])],meta={"teams": {"max_players_per_team": 11},
                                "ball": {"image_observations": {"5": {"pixel": [10, 20]}}}})
        (path / "track.json").write_text(json.dumps(track.to_json()))
        (path / "report.txt").write_text("tracking is partial\n")
        for name in ("sidebyside.mp4", "ball-preview.mp4"):
            shutil.copyfile(source_video, path / name)
        return SimpleNamespace(track=track, problems=[], stages={"render": "computed"}, seconds=.5)

    def encode(source, dest):
        shutil.copyfile(source, dest)
        return True

    monkeypatch.setattr(render, "render", fake_render)
    monkeypatch.setattr(video, "_browser_video", encode)
    response = client.post("/video/clips/play/run")
    job = _finished(client, response.json()["jobId"])
    assert job["state"] == "done", job
    assert Path(observed[0][0].video).is_absolute()
    assert observed[0][1] == video.RUNS
    result = job["result"]
    assert result["hasTrack"] and result["teams"]["max_players_per_team"] == 11
    for url in result["assets"].values():
        assert client.get(url).status_code == 200
    assert client.get("/video/clips/play/report").text == "tracking is partial\n"
    ranged = client.get(result["assets"]["reconstruction"], headers={"Range": "bytes=0-31"})
    assert ranged.status_code == 206 and len(ranged.content) == 32


def test_refused_run_does_not_advertise_a_stale_track(client, source_video, monkeypatch):
    _clip(client, source_video)
    from gridiron_tracking import render
    path = video.RUNS / "play"
    path.mkdir(parents=True)
    (path / "track.json").write_text("{}")
    monkeypatch.setattr(render, "render", lambda *a, **kw: SimpleNamespace(
        track=None, problems=["no registration"], stages={"register": "failed"}, seconds=.1))
    job = _finished(client, client.post("/video/clips/play/run").json()["jobId"])
    assert job["state"] == "done"
    assert job["result"]["hasTrack"] is False and job["result"]["assets"] == {}


def test_failed_analysis_is_visible_through_polling(client, source_video, monkeypatch):
    _clip(client, source_video)
    from gridiron_tracking import render

    def fail(*a, **kw):
        raise RuntimeError("model is not installed")

    monkeypatch.setattr(render, "render", fail)
    job = _finished(client, client.post("/video/clips/play/run").json()["jobId"])
    assert job["state"] == "failed" and "model is not installed" in job["error"]


def test_unknown_resources_return_404(client):
    for url in ("/video/jobs/missing", "/video/clips/missing/frame", "/video/clips/missing/track",
                "/video/clips/missing/reconstruction.mp4", "/video/uploads/*/source"):
        assert client.get(url).status_code == 404


def test_bdb_example_is_available_without_local_runs_or_models(client):
    discovery = client.get("/video").json()
    assert discovery["schema"] == "hindsight-play"
    listing = client.get(discovery["examples"]).json()["examples"]
    assert listing and listing[0]["name"] == "nfl_sf_sea"
    response = client.get(listing[0]["tracking"])
    assert response.status_code == 200
    doc = response.json()
    from api.schema import PlayDetail
    PlayDetail.model_validate(doc)
    assert doc["schemaVersion"] == 1 and "roster" not in doc
    assert len(doc["players"]) == 22
    assert doc["analysisReady"] and doc["trackingQuality"]["assisted"]
    assert any(f["event"] == "pass_forward" for f in doc["frames"])
    rows = client.get(listing[0]["tracking"]+"/rows").json()
    assert isinstance(rows, list) and rows[0]["club"] in ("SF", "SEA")
    assert set(rows[0]) == {"gameId", "playId", "nflId", "displayName", "frameId", "frameType",
                          "time", "jerseyNumber", "club", "playDirection", "x", "y", "s", "a", "dis", "o", "dir", "event"}
    assert len([r for r in rows if r["club"] == "football"]) > 60
    csv_response = client.get(listing[0]["csv"])
    assert csv_response.status_code == 200 and csv_response.headers["content-type"].startswith("text/csv")
    metadata = client.get("/video/examples/nfl_sf_sea/tracking/metadata").json()
    assert metadata["syntheticGameId"] and metadata["syntheticPlayId"]
    assert not metadata["coordinatesMeasured"]
    assert not video.RUNS.exists(), "the example must not require analysis artifacts"


def test_bdb_metadata_can_be_supplied_at_clip_creation(client, source_video):
    uploaded = client.post("/video/uploads", files={"file": ("play.mp4", source_video.read_bytes())}).json()
    response = client.post("/video/clips", json={"videoId": uploaded["videoId"], "name": "labeled",
        "startFrame": 0, "stopFrame": 12, "gameId": 2022091200, "playId": 64,
        "playDirection": "right", "teamAClub": "DEN", "teamBClub": "SEA"})
    assert response.status_code == 200
    spec = ClipSpec.load("labeled", video.SPEC_DIR)
    assert spec.bdb["gameId"] == 2022091200
    assert spec.bdb["teamClubs"] == {"teamA": "DEN", "teamB": "SEA"}


def test_bdb_row_schema_is_discoverable_in_openapi(client):
    schema = client.get("/openapi.json").json()
    row = schema["components"]["schemas"]["BDBTrackingRow"]
    assert "frameType" in row["properties"] and "nflId" in row["properties"]
    assert "/video/clips/{name}/tracking.csv" in schema["paths"]


def _checked_track(client):
    raw = json.loads((video.EXAMPLES/"nfl_sf_sea.track.json").read_text())
    dest = video.RUNS/"checked"
    dest.mkdir(parents=True)
    (dest/"track.json").write_text(json.dumps(raw))
    return raw,dest


def test_primary_video_endpoints_share_the_replay_contract(client):
    raw,_ = _checked_track(client)
    doc = client.get("/video/clips/checked/tracking").json()
    assert doc == client.get("/video/clips/checked/track").json()
    from api.schema import PlayDetail
    PlayDetail.model_validate(doc)
    assert len(doc["players"]) == 22 and doc["analysisReady"]
    assert sum("ball" in f for f in doc["frames"]) == 95
    assert client.get("/video/clips/checked/track/raw").json() == raw


def test_incomplete_video_review_returns_actionable_blockers(client):
    raw,path = _checked_track(client)
    raw["meta"].pop("reviewed_pass")
    for f in raw["frames"]:
        f["ball"] = None
        f["events"] = [e for e in f["events"] if not e.startswith("pass_")]
    (path/"track.json").write_text(json.dumps(raw))
    doc = client.get("/video/clips/checked/tracking").json()
    assert not doc["analysisReady"] and doc["analysisBlockers"]
    response = client.post("/video/clips/checked/review")
    assert response.status_code == 422
    assert response.json()["detail"]["analysisBlockers"] == doc["analysisBlockers"]


def test_pass_annotation_is_validated_before_save_and_returns_replay_json(client):
    raw,_ = _checked_track(client)
    spec = ClipSpec(name="checked",video="unused.mp4",start=1575,stop=1865,snap_frame=1672)
    spec.save(video.SPEC_DIR)
    valid = {"throwFrame":1769,"catchFrame":1780,"passerId":"p02","receiverId":"p07","centerId":"p18"}
    invalid = client.post("/video/clips/checked/pass-review",json=valid|{"receiverId":"missing"})
    assert invalid.status_code == 422
    assert not ClipSpec.load("checked",video.SPEC_DIR).reviewed_pass
    response = client.post("/video/clips/checked/pass-review",json=valid)
    assert response.status_code == 200 and response.json()["analysisReady"]
    assert ClipSpec.load("checked",video.SPEC_DIR).reviewed_pass == valid
    assert client.get("/video/clips/checked/tracking").json() == response.json()


def test_identity_update_requires_rerun_before_serving_stale_results(client):
    raw,_ = _checked_track(client)
    spec = ClipSpec(name="checked",video="unused.mp4",start=1575,stop=1865,snap_frame=1672)
    spec.save(video.SPEC_DIR)
    response = client.post("/video/clips/checked/identities",json={"fullRosterFrame":1580,"anchors":[]})
    assert response.status_code == 200 and response.json()["rerunRequired"]
    response = client.get("/video/clips/checked/tracking")
    assert response.status_code == 409 and "rerun" in response.json()["detail"]


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg optional outside the video server")
def test_real_browser_transcode_is_readable(source_video, tmp_path):
    import cv2
    output = tmp_path / "browser.mp4"
    assert video._browser_video(source_video, output)
    cap = cv2.VideoCapture(str(output))
    ok, image = cap.read()
    cap.release()
    assert ok and image.shape[:2] == (180, 320)
