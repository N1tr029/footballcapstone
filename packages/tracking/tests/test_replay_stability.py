"""Constant replay membership and reversible estimates across tracking dropouts."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from gridiron_tracking.contract import PlayTrack,RosterEntry,TrackingFrame,PlayerFrame,Ball
from gridiron_tracking.hindsight import to_play_json,read_play
from gridiron_tracking.replay_stability import stabilize_document,continuous_path,display_track
from gridiron_tracking.identity_observations import reconcile
from gridiron_tracking.detect import Detection


def sparse():
    return PlayTrack(meta={"source":"vision","rate":10},roster={
        "a":RosterEntry("a","teamA"),"b":RosterEntry("b","teamB")},frames=[
        TrackingFrame(t=i/10,players={**({"a":PlayerFrame(40+i*.3,20)} if i in (0,1,2,8,9) else {}),
            **({"b":PlayerFrame(50,30)} if 2<=i<=6 else {})}) for i in range(10)])


def test_remembers_a_fills_b_from_c_and_keeps_every_actor_and_team():
    raw=to_play_json(sparse());before=deepcopy(raw);doc=stabilize_document(raw)
    ids={p["nflId"] for p in doc["players"]};clubs={p["nflId"]:p["club"] for p in doc["players"]}
    assert all({p["nflId"] for p in f["players"]}==ids for f in doc["frames"])
    assert all(p["club"]==clubs[p["nflId"]] for f in doc["frames"] for p in f["players"])
    a=doc["players"][0]["nflId"]
    midpoint=next(p for p in doc["frames"][5]["players"] if p["nflId"]==a)
    assert midpoint["x"]==pytest.approx(41.5,abs=.1)
    assert midpoint["trackingStatus"]=="interpolated" and not midpoint["observed"]
    assert doc["trackingQuality"]["replayRosterStable"]
    assert raw==before,"continuous replay must not mutate the sparse evidence"
    assert stabilize_document(doc)==doc,"a cached public document must not be smoothed twice"


def test_model_round_trip_restores_only_original_observations_and_coordinates():
    raw=to_play_json(sparse());plain=read_play(raw);restored=read_play(stabilize_document(raw))
    assert [f.to_json() for f in restored.frames]==[f.to_json() for f in plain.frames]
    assert len(restored.frames[5].players)==1,"a display-filled athlete must not become a fresh model input"


def test_position_smoothing_reduces_jitter_and_never_teleports():
    indices=np.arange(40);truth=np.column_stack([40+indices*.2,np.full(40,20.)])
    noise=np.column_stack([np.where(indices%2,.3,-.3),np.where(indices%2,-.2,.2)])
    smooth=continuous_path(indices,truth+noise,40,10)
    assert np.sqrt(np.mean((smooth-truth)**2))<np.sqrt(np.mean(noise**2))*.5
    jumped=continuous_path([0,1,20,21],[[40,20],[40.3,20],[55,22],[55.3,22]],30,10)
    assert np.max(np.linalg.norm(np.diff(jumped,axis=0),axis=1))<=1.200001
    assert np.linalg.norm(jumped[-1]-jumped[-2])<.1,"an unseen athlete must coast to rest"


def test_never_seen_id_is_reported_without_a_made_up_origin():
    doc=to_play_json(sparse());doc["players"].append({"nflId":999,"club":"teamA","side":"offense"})
    out=stabilize_document(doc)
    assert out["provenance"]["replayStability"]["neverObservedIds"]==[999]
    assert not out["trackingQuality"]["replayRosterStable"]
    assert all(p["nflId"]!=999 for f in out["frames"] for p in f["players"])


def test_video_display_keeps_roster_without_changing_the_evidence():
    track=sparse();shown=display_track(track)
    assert all(set(f.players)==set(track.roster) for f in shown.frames)
    assert len(track.frames[5].players)==1


def test_ball_can_follow_continuous_carriers_without_fabricating_model_sightings():
    track=sparse();track.meta["automatic_pass"]={"passerId":"a","receiverId":"b", "centerId":"a",
        "throwTime":.2,"arrivalTime":.6}
    track.frames[0].ball=Ball(40,20,source="carrier")
    doc=to_play_json(track);doc["provenance"]["automaticPass"]=track.meta["automatic_pass"]
    shown=stabilize_document(doc)
    assert all(f.get("ball") for f in shown["frames"])
    restored=read_play(shown)
    assert sum(f.ball is not None for f in restored.frames)==1
    assert restored.frames[0].ball.x==40


def test_filled_forecast_actors_are_not_labelled_as_camera_observations():
    doc=to_play_json(sparse());doc["simulated"]=True;doc["simulationStartFrameId"]=7
    shown=stabilize_document(doc)
    future=[p for f in shown["frames"][6:] for p in f["players"]]
    assert all(not p["observed"] for p in future)
    assert any(p["trackingStatus"]=="simulated" for p in future)


def test_opponent_masks_cannot_reassign_a_fixed_team_and_bodies_are_exclusive():
    roster={"a":RosterEntry("a","teamA"),"b":RosterEntry("b","teamB")}
    def d(frame,x,team):
        return Detection(frame=frame,x1=x,y1=0,x2=x+2,y2=10,score=.9,team=team)
    detections={0:[d(0,40,"kitA"),d(0,44,"kitB")],3:[d(3,40.3,"kitA"),d(3,44.3,"kitB")]}
    tracked={0:{"a":[40,0,42,10],"b":[44,0,46,10]},
             3:{"a":[44.3,0,46.3,10],"b":[44.3,0,46.3,10]}}
    registration=SimpleNamespace(to_field=lambda frame,points:np.asarray(points))
    fixed,report=reconcile(tracked,detections,registration,roster,0,30)
    assert fixed[3]["a"]==[40.3,0,42.3,10] and fixed[3]["b"]==[44.3,0,46.3,10]
    assert report["opponentMasksRejected"]==1
    assert roster["a"].team=="teamA" and roster["b"].team=="teamB"


def test_a_single_formation_sighting_survives_alignment_to_the_replay_clock():
    from gridiron_tracking.auto_identity import resample
    registration=SimpleNamespace(to_field=lambda frame,points:np.asarray(points))
    roster={"a":RosterEntry("a","teamA")}
    frames=resample(roster,{1:{"a":[39,10,41,20]}},registration,
        fps=30,snap=0,first=0,last=10)
    assert any("a" in f.players for f in frames)
