"""The video/replay/model seam, including sparse evidence and reviewed passes."""
from copy import deepcopy

import numpy as np
import pytest

from gridiron_tracking.contract import Ball, PlayTrack, PlayerFrame, RosterEntry, TrackingFrame
from gridiron_tracking.identity import Observation, anchored
from gridiron_tracking.video_play import apply_review, play_json, prepared
from gridiron_tracking import hindsight


def video_track():
    roster = {p:RosterEntry(id=p,team=t) for p,t in
              (("qb","teamA"),("wr","teamA"),("db","teamB"),("gone","teamB"))}
    frames = []
    for i in range(21):
        players = {"qb":PlayerFrame(45+i*.02,20),"wr":PlayerFrame(50+i*.4,25),
                   "db":PlayerFrame(53+i*.4,25)}
        if i == 0:
            players["gone"] = PlayerFrame(55,40)
        frames.append(TrackingFrame(t=i/10,players=players,events=["ball_snap"] if i==0 else []))
    frames[7].ball = Ball(50,23,source="detected")
    return PlayTrack(roster=roster,frames=frames,meta={"source":"vision","rate":10,"fps_in":30,
        "clip":{"start_frame":0,"stop_frame":61,"snap_source_frame":0},
        "identity":{"anchorSourceFrame":0,"method":"checked formation"},
        "broadcast":{"lineOfScrimmage":50,"yardsToGo":10,"attacks":"right"},
        "ball":{"measured_fraction":1/21,"travelled_yards":0,"by_source":{"detected":1}},
        "bdb":{"gameId":999000001,"playId":900000,"playDirection":"right",
               "teamClubs":{"teamA":"SF","teamB":"SEA"}}})


ANNOTATION = {"throwFrame":15,"catchFrame":30,"passerId":"qb","receiverId":"wr"}


def test_sparse_video_is_replayable_but_not_model_ready():
    doc = play_json(video_track())
    assert doc["schemaVersion"] == 1 and "roster" not in doc
    assert not doc["analysisReady"] and doc["analysisBlockers"]
    assert sum("ball" in f for f in doc["frames"]) == 1
    assert not any("pass_forward" in f["events"] for f in doc["frames"])
    assert {p["side"] for p in doc["players"]} == {"offense","defense"}
    assert {p["club"] for p in doc["players"]} == {"SF","SEA"}


def test_review_preserves_evidence_and_publishes_carriers_and_both_catch_events():
    raw = video_track(); before = deepcopy(raw.to_json())
    doc = play_json(raw,annotation=ANNOTATION)
    assert raw.to_json() == before, "API conversion must not mutate the cached track"
    assert doc["analysisReady"] and doc["trackingQuality"]["assisted"]
    assert doc["trackingQuality"]["missingAtThrow"] == ["gone"]
    assert len(doc["players"]) == 4, "confirmed dormant athletes must not be deleted"
    assert doc["frames"][5]["event"] == "pass_forward"
    assert doc["frames"][10]["events"] == ["pass_arrived","pass_outcome_caught"]
    assert doc["frames"][6]["ball"]["dataOrigin"] == "interpolated"
    assert "carrierNflId" not in doc["frames"][6]["ball"]
    carrier = doc["frames"][5]["ball"]["carrierNflId"]
    assert carrier == next(p["nflId"] for p in doc["players"] if p["trackId"] == "qb")
    assert doc["provenance"]["ball"]["raw_projected_observations"]["8"]["source"] == "detected"
    restored = hindsight.read_play(doc)
    assert restored.meta["source"] == "vision"
    assert restored.meta["prepare_report"]["reviewable"]
    assert restored.frames[6].ball.source == "interpolated"
    assert restored.frames[10].events == ["pass_arrived","pass_outcome_caught"]
    assert {p.o_source for f in restored.frames for p in f.players.values()} == {"assumed"}


@pytest.mark.parametrize("change",[{"receiverId":"db"},{"receiverId":"missing"},
    {"receiverId":"qb"},{"catchFrame":15},{"catchFrame":99}])
def test_bad_reviews_cannot_publish_events(change):
    raw = video_track(); before = deepcopy(raw.to_json())
    with pytest.raises(ValueError):
        apply_review(raw,ANNOTATION|change)
    assert raw.to_json() == before


def test_unobserved_carrier_stays_missing_and_missing_catch_refuses_review():
    raw = video_track(); del raw.frames[12].players["wr"]
    doc = play_json(raw,annotation=ANNOTATION)
    assert "ball" not in doc["frames"][12]
    del raw.frames[10].players["wr"]
    with pytest.raises(ValueError,match="observed positions"):
        play_json(raw,annotation=ANNOTATION)


def test_preparation_and_review_are_idempotent_even_when_mirrored():
    raw = video_track(); raw.meta["broadcast"]["attacks"] = "left"
    first,report = prepared(raw,annotation=ANNOTATION)
    again,other = prepared(first,annotation=ANNOTATION)
    assert report.reviewable and other.reviewable
    assert [(f.t,f.ball.to_json() if f.ball else None) for f in first.frames] == [
            (f.t,f.ball.to_json() if f.ball else None) for f in again.frames]
    assert first.meta["lineOfScrimmage"] == again.meta["lineOfScrimmage"]


def test_dormant_identity_returns_without_new_births_or_long_gap_ghosts():
    color_a,color_b = np.zeros(5),np.full(5,100.)
    observations = {}
    for i in range(16):
        points = [Observation(np.array([i*.1,10.]),color_a)]
        if i < 3 or i > 9:
            points.append(Observation(np.array([5.+i*.1,20.]),color_b))
        if i > 5:
            points.append(Observation(np.array([90.,45.]),color_a)) # unrelated detection
        observations[i] = points
    track = anchored(observations,0,["A","B"],fps=10,snap=0)
    assert set(track.roster) == {"p00","p01"}
    assert "p01" not in track.frame_at(.7).players
    assert track.frame_at(1.2).players["p01"].x == pytest.approx(6.2)
    assert all(p.x < 10 for f in track.frames for p in f.players.values())
