"""A later sighting triggers one-player reverse tracking and a checked repair."""
from copy import deepcopy
import json
import math
from pathlib import Path

from gridiron_tracking.detect import Detection
from gridiron_tracking.reacquisition import recovery_windows,repair_one,recover


def box(x):
    return [x,0,x+5,12]


def fixture():
    masks={f:{"a":box(f/10),"b":box(50)} for f in range(0,43,3)}
    accepted=deepcopy(masks)
    for f in range(9,27,3):accepted[f].pop("a")
    bodies={f:[Detection(frame=f,x1=b[0],y1=b[1],x2=b[2],y2=b[3],score=.95)
        for b in people.values()] for f,people in masks.items()}
    return accepted,masks,bodies


def test_later_reappearance_runs_only_that_player_backward_and_fills_his_gap(tmp_path):
    accepted,masks,bodies=fixture();other={f:p["b"][:] for f,p in accepted.items()};calls=[]
    def backward(frames,anchor,prompts,**kw):
        calls.append((anchor,prompts,kw))
        assert list(prompts)==["a"] and kw["reverse"] is True
        return {f:{"a":people["a"]} for f,people in masks.items() if kw["stop"]<=f<=anchor}
    results=recover(accepted,masks,bodies,{},fps=30,weights="unused.pt",cache_dir=tmp_path,tracker=backward)
    assert len(calls)==1 and results[0]["accepted"]
    assert results[0]["start"]<9 and results[0]["stop"]>=27
    assert all(accepted[f]["a"]==masks[f]["a"] for f in range(9,27,3))
    assert {f:p["b"] for f,p in accepted.items()}==other


def test_wrong_earlier_person_refuses_repair_and_leaves_everything_unchanged():
    accepted,_,_=fixture();before=deepcopy(accepted)
    reverse={6:{"a":box(50)},9:{"a":box(50)}}
    result=repair_one(accepted,reverse,"a",6,30)
    assert not result["accepted"] and accepted==before


def test_one_later_flash_or_no_later_body_does_not_authorize_backward_recovery():
    accepted,masks,bodies=fixture()
    for f in range(9,43,3):accepted[f].pop("a",None)
    for f in range(9,43,3):
        if f!=27:masks[f].pop("a")
    assert recovery_windows(accepted,masks,bodies,30)==[]


def test_repairs_wrong_body_association_without_renaming_or_moving_another_id(tmp_path):
    accepted,masks,bodies=fixture()
    for f in range(9,27,3):accepted[f]["a"]=box(50)
    def backward(frames,anchor,prompts,**kw):
        return {f:{"a":people["a"]} for f,people in masks.items() if kw["stop"]<=f<=anchor}
    results=recover(accepted,masks,bodies,{},fps=30,weights="unused",cache_dir=tmp_path,tracker=backward)
    assert results[0]["accepted"] and results[0]["correctedFrames"]
    assert accepted[15]["a"]==box(1.5) and accepted[15]["b"]==box(50)


def test_reverse_track_cannot_duplicate_a_body_already_assigned_to_another_identity():
    accepted,_,_=fixture()
    reverse={6:{"a":box(.6)},9:{"a":box(50)}}
    result=repair_one(accepted,reverse,"a",6,30)
    assert result["accepted"] and result["conflictedFrames"]==[9]
    assert "a" not in accepted[9] and accepted[9]["b"]==box(50)


def test_each_separate_loss_of_the_same_player_is_recovered(tmp_path):
    accepted,masks,bodies=fixture()
    for f in range(45,91,3):
        masks[f]={"a":box(f/10),"b":box(50)}
        accepted[f]=deepcopy(masks[f])
        bodies[f]=[Detection(frame=f,x1=b[0],y1=b[1],x2=b[2],y2=b[3],score=.95)
            for b in masks[f].values()]
    for f in range(54,72,3):accepted[f].pop("a")
    calls=[]
    def backward(frames,anchor,prompts,**kwargs):
        calls.append((kwargs["stop"],anchor))
        assert list(prompts)==["a"]
        return {f:{"a":p["a"]} for f,p in masks.items() if kwargs["stop"]<=f<=anchor}
    results=recover(accepted,masks,bodies,{},fps=30,weights="unused",cache_dir=tmp_path,tracker=backward)
    assert len(calls)==2 and all(r["accepted"] for r in results)
    assert all("a" in accepted[f] for f in list(range(9,27,3))+list(range(54,72,3)))


def test_real_automatic_fixture_recovers_the_defender_near_the_passer():
    from gridiron_tracking.contract import PlayTrack
    from gridiron_tracking.video_play import play_json
    path=Path(__file__).resolve().parents[3]/"docs/examples/nfl_sf_sea_auto.track.json"
    raw=PlayTrack.from_json(json.loads(path.read_text()))
    check=next(c for c in raw.meta["identity"]["reacquisitionChecks"] if c["playerId"]=="p16")
    assert check["accepted"] and check["earlierSeamIoU"]>=.35
    assert check["recoveredFrames"] and check["correctedFrames"]
    thrown=raw.frame_at(raw.meta["automatic_pass"]["throwTime"])
    defender,qb=thrown.players["p16"],thrown.players["p03"]
    assert math.hypot(defender.x-qb.x,defender.y-qb.y)<8
    doc=play_json(raw)
    assert all(len(f["players"])==22 for f in doc["frames"])
