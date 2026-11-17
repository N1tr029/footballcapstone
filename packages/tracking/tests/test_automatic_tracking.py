"""Automatic inputs, future-to-past identity correction and honest ball estimates."""
from copy import deepcopy
from types import SimpleNamespace

import numpy as np
import pytest

from gridiron_tracking.automatic import snap_from_positions
from gridiron_tracking.auto_identity import reconcile_window,crossing_windows
from gridiron_tracking.auto_pass import infer,apply
from gridiron_tracking.jerseys import consensus
from gridiron_tracking.contract import PlayTrack,RosterEntry,TrackingFrame,PlayerFrame
from gridiron_tracking.sam_video import cache_signature


def test_snap_comes_from_common_motion_without_annotated_frame():
    points={}
    base=np.array([[40+j%11*2,10+(j//11)*4] for j in range(22)],float)
    for f in range(150):
        t=max(0,(f-60)/30)
        shift=min(t*t*4,t*4)
        points[f]=base+[shift,0]
    result=snap_from_positions(points,30)
    assert result.frame is not None and abs(result.frame-60)<=7
    moving={f:p+[f/10,0] for f,p in points.items()}
    assert snap_from_positions(moving,30).frame is None


def test_formation_selection_allows_officials_before_uniform_roles_are_checked(monkeypatch):
    from gridiron_tracking.automatic import choose_formation
    from gridiron_tracking.detect import Detection,color
    monkeypatch.setattr(color,"kit_color",lambda image,d:np.array([d.x1,0.,0.]))
    monkeypatch.setattr(color,"cluster_teams",lambda samples:({k:int(v[0][0]>10) for k,v in samples.items()},
        np.array([[0.,0.,0.],[100.,0.,0.]])))
    def people(n):
        return [Detection(frame=0,x1=0 if j<11 else 100,y1=0,x2=105,y2=10,score=.9) for j in range(n)]
    boxes={0:people(25),1:people(22)};frames={f:np.zeros((20,20,3),np.uint8) for f in boxes}
    assert choose_formation(boxes,frames,None,30,allow_officials=True)==0
    assert choose_formation(boxes,frames,None,30,allow_officials=False)==1


def test_failed_uniform_reader_cannot_publish_25_people_as_players(monkeypatch,tmp_path):
    from gridiron_tracking import automatic,athlete_roles,auto_identity
    from gridiron_tracking.registration import local
    monkeypatch.setattr(automatic,"choose_formation",lambda *a,**k:0)
    def unavailable(*args,**kwargs):raise RuntimeError("model not installed")
    monkeypatch.setattr(athlete_roles,"refine",unavailable);released=[]
    monkeypatch.setattr(local,"release",lambda:released.append(True))
    with pytest.raises(ValueError,match="team capacity"):
        auto_identity.track({0:np.zeros((20,20,3),np.uint8)},{0:[object()]*25},None,
            fps=30,snap=0,weights="unused",cache_dir=tmp_path)
    assert released==[True]


def box(x):
    return [x,0,x+4,10]


def test_later_evidence_repairs_an_earlier_swap_and_a_gap():
    forward={0:{"a":box(0),"b":box(10)},1:{"a":box(3)},
             2:{"a":box(2),"b":box(8)},3:{"a":box(0),"b":box(10)}}
    # The later object called a traces back to the original b, and vice versa.
    backward={0:{"a":box(10),"b":box(0)},1:{"a":box(7),"b":box(3)},
              2:{"a":box(2),"b":box(8)}}
    result=reconcile_window(forward,backward,0,2,["a","b"])
    assert result["accepted"] and result["mapping"]=={"a":"b","b":"a"}
    assert forward[0]["a"]==box(0),"a later correction must not rewrite an established earlier identity"
    assert forward[1]["b"]==box(7) and result["recovered"]
    assert forward[3]["a"]==box(10) and forward[3]["b"]==box(0)


def test_failed_backward_seam_never_merges_unknown_people():
    forward={0:{"a":box(0),"b":box(10)},2:{"a":box(2),"b":box(8)}}
    before=deepcopy(forward)
    result=reconcile_window(forward,{0:{"a":box(40),"b":box(60)}},0,2,["a","b"])
    assert not result["accepted"] and forward==before


def test_backward_work_targets_teammates_who_cross_then_separate():
    positions=[(0,10),(1,9),(3,5),(4,4),(7,1),(8,0),(9,-1)]
    boxes={i:{"a":box(a),"b":box(b),"other":box(3)} for i,(a,b) in enumerate(positions)}
    windows=crossing_windows(boxes,{"a":"A","b":"A","other":"B"},10)
    assert windows and all(w["players"]==["a","b"] for w in windows)
    permanent={i:{"a":box(0),"b":box(1)} for i in range(20)}
    assert not crossing_windows(permanent,{"a":"A","b":"A"},10)


def test_jersey_consensus_uses_distinct_frames_and_abstains_on_conflicts():
    readings=[{"frame":f,"number":23,"confidence":.95} for f in (1,5,10)]
    assert consensus(readings)["number"]==23
    assert consensus([readings[0]]*6) is None
    other=[{"frame":f,"number":13,"confidence":.95} for f in (12,16,20)]
    assert consensus(readings+other) is None


def test_jersey_revealed_later_labels_the_existing_earlier_identity(tmp_path,monkeypatch):
    from gridiron_tracking import jerseys
    monkeypatch.setattr(jerseys.shutil,"which",lambda _:"optional-ocr")
    monkeypatch.setattr(jerseys,"native_binary",lambda _:None)
    def read(crop,*args):
        return [{"number":23,"confidence":.95}] if crop[0,0,0]>=45 else []
    monkeypatch.setattr(jerseys,"read_crop",read)
    frames={f:np.full((100,100,3),f,np.uint8) for f in range(0,76,15)}
    track=PlayTrack(roster={"a":RosterEntry("a","A")},frames=[
        TrackingFrame(t=f/30,players={"a":PlayerFrame(50,20)}) for f in frames])
    boxes={f:{"a":[10,0,80,90]} for f in frames}
    accepted=jerseys.annotate(track,frames,boxes,fps=30,cache_dir=tmp_path)
    assert accepted["a"]["number"]==23 and track.roster["a"].num==23
    assert all(set(f.players)=={"a"} for f in track.frames)
    assert min(r["frame"] for r in track.meta["jerseys"]["observations"]["a"])>=45


def test_tracking_cache_depends_on_image_content_and_model(tmp_path):
    weights=tmp_path/"sam.pt";weights.write_bytes(b"one")
    frames={i:np.zeros((30,50,3),np.uint8) for i in range(6)}
    kw=dict(fps=30,weights=weights,cache_dir=tmp_path)
    first=cache_signature(frames,0,{"a":box(1)},**kw)[0]
    frames[3][:]=255
    assert cache_signature(frames,0,{"a":box(1)},**kw)[0]!=first
    frames[3][:]=0;weights.write_bytes(b"two")
    assert cache_signature(frames,0,{"a":box(1)},**kw)[0]!=first


def pass_fixture():
    roster={"qb":RosterEntry("qb","A"),"wr":RosterEntry("wr","A"),"db":RosterEntry("db","B")}
    frames=[TrackingFrame(t=i/10,players={"qb":PlayerFrame(50,50),"wr":PlayerFrame(70,50),
            "db":PlayerFrame(60,70)},events=["ball_snap"] if i==0 else []) for i in range(16)]
    boxes={str(i):{"qb":[40,10,60,50],"wr":[60,10,80,50],"db":[50,30,70,70]} for i in range(16)}
    track=PlayTrack(roster=roster,frames=frames,meta={"source":"vision","fps_in":10,
        "clip":{"snap_source_frame":0},"identity":{"sourceBoxes":boxes},
        "broadcast":{"lineOfScrimmage":55,"attacks":"right"}})
    observations={i:{"field":[56+(i-6)*2,22],"pixel":[56+(i-6)*2,22]} for i in range(6,12)}
    h=SimpleNamespace(to_image=lambda p:np.asarray(p),to_field=lambda p:np.asarray(p))
    return track,observations,SimpleNamespace(at=lambda f:h)


def test_flight_recovers_endpoints_without_manual_ids_or_event_frames():
    track,observations,registration=pass_fixture()
    hypothesis,reason=infer(track,observations,registration,10,0)
    assert not reason and hypothesis["passerId"]=="qb" and hypothesis["receiverId"]=="wr"
    assert hypothesis["throwFrame"]<min(observations) and hypothesis["catchFrame"]>max(observations)
    accepted,reason=apply(track,hypothesis)
    assert accepted and not reason
    assert track.event_time("pass_forward") is not None and track.event_time("pass_arrived") is not None
    assert track.event_time("pass_outcome_caught") is None,"arrival alone does not establish completion"
    assert "reviewed_pass" not in track.meta and track.meta["automatic_pass"]["validated"]
    assert all(f.ball.source in ("carrier","interpolated") for f in track.frames if f.ball)


def test_static_or_too_sparse_ball_evidence_does_not_invent_a_pass():
    track,observations,registration=pass_fixture()
    assert infer(track,dict(list(observations.items())[:5]),registration,10,0)[0] is None
    for point in observations.values():
        point["field"]=[56,22]
    assert infer(track,observations,registration,10,0)[0] is None


def test_cached_calibration_restores_the_frame_it_was_fitted_on(tmp_path):
    import json
    from gridiron_tracking.clips import ClipSpec
    from gridiron_tracking.render import _anchor
    spec=ClipSpec(name="clip",video="unused.mp4",start=0,stop=100)
    cache=tmp_path/f"registration-{spec.key('register')}.json"
    cache.write_text(json.dumps({"H":np.eye(3).tolist(),"level":"nfl","frame":17}))
    _,_,stage,_=_anchor(spec,{0:np.zeros((20,30,3)),17:np.zeros((20,30,3))},100,tmp_path,set(),lambda *a:None)
    assert stage=="cached" and spec.register.frame==17
    assert spec.read_frame(100)==17,"camera propagation must use the fitted frame, not the window midpoint"


def test_later_paint_labels_calibrate_an_earlier_frame_without_clicked_points(tmp_path,monkeypatch):
    from gridiron_tracking.registration import paint_auto,vision
    from gridiron_tracking.registration.lines import FieldLine
    from gridiron_tracking import jerseys
    lines=[FieldLine(i,i*100,i*100,0,200,600) for i in range(5)]
    def read(image,boxes,native):
        later=bool(image[0,0,0])
        label={"number":40 if later else 30,"pixel":[100.,70.] if later else [300.,70.],
               "gridIndex":1 if later else 3,"confidence":1.}
        return [label],[[100.,70.],[300.,70.]],lines
    monkeypatch.setattr(paint_auto,"number_candidates",read)
    monkeypatch.setattr(paint_auto,"far_sideline",lambda image:(0.,10.))
    monkeypatch.setattr(paint_auto,"image_registration",lambda *a:np.eye(3))
    monkeypatch.setattr(jerseys,"native_binary",lambda cache:None)
    monkeypatch.setattr(vision,"sanity_problems",lambda *a,**k:[])
    monkeypatch.setattr(vision,"snap_to_paint",lambda image,h,spec:(h,2.5))
    monkeypatch.setattr(vision,"turf_overlap",lambda *a:.9)
    frames={f:np.full((200,500,3),int(f==6),np.uint8) for f in range(7)}
    result=paint_auto.calibrate(frames,{},fps=30,cache_dir=tmp_path,level="nfl")
    assert result and result.ok and result.frame==0
    assert {e["frame"] for e in result.evidence["numbers"]}=={0,6}
    assert result.homography.to_field([[300,70]])[0,0]==pytest.approx(80)
    monkeypatch.setattr(vision,"snap_to_paint",lambda image,h,spec:(h,1.1))
    assert paint_auto.calibrate(frames,{},fps=30,cache_dir=tmp_path,level="nfl") is None
