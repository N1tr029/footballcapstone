"""Automatic fixed roster tracking and later-to-earlier identity reconciliation."""
from __future__ import annotations

import math
from itertools import combinations
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from .contract import PlayTrack,PlayerFrame,RosterEntry,TrackingFrame


def iou(a,b):
    left=max(a[0],b[0]);top=max(a[1],b[1]);right=min(a[2],b[2]);bottom=min(a[3],b[3])
    area=max(0,right-left)*max(0,bottom-top)
    return area/max(1.,(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-area)


def crossing_windows(boxes,teams,fps,maximum=8):
    """Teammates separate, overlap, then become clear again: backward-check there."""
    frames=sorted(boxes);windows=[]
    for a,b in combinations(teams,2):
        if teams[a]!=teams[b]:
            continue
        start=None;last_clear=None;overlap=0;clear=[]
        for frame in frames:
            here=boxes[frame]
            if a not in here or b not in here:
                if start is not None:
                    clear=[]
                continue
            touching=iou(here[a],here[b])>.15
            if touching:
                if start is None and last_clear is not None:
                    start=last_clear;overlap=0
                overlap+=1;clear=[]
            else:
                if start is not None:
                    clear.append(frame)
                    if len(clear)>=3:
                        if overlap>=2 and .1<=(frame-start)/fps<=3.0:
                            windows.append({"start":start,"stop":frame,"players":[a,b]})
                        start=None;overlap=0;clear=[]
                last_clear=frame
            if start is not None and frame-start>fps*3:
                start=None;overlap=0;last_clear=None
    # Prefer isolated crossings, not repeated permanent line-of-scrimmage contact.
    windows.sort(key=lambda w:(w["stop"]-w["start"],w["start"]))
    selected=[]
    for window in windows:
        if any(set(window["players"])&set(w["players"]) and
               not(window["start"]>w["stop"] or w["start"]>window["stop"]) for w in selected):
            continue
        selected.append(window)
        if len(selected)>=maximum:
            break
    return sorted(selected,key=lambda w:w["start"])


def reconcile_window(forward,backward,start,stop,pids,minimum_iou=.3):
    """A later clear observation must trace back to a known earlier athlete.

    Backward masks alone do not authorize a merge. Require a one-to-one match to
    both known pre-crossing identities. A failed seam leaves the ambiguity intact.
    """
    if start not in backward or any(p not in backward[start] or p not in forward[start] for p in pids):
        return {"accepted":False,"reason":"backward trace did not reach both earlier identities"}
    overlap=np.array([[iou(backward[start][a],forward[start][b]) for b in pids] for a in pids])
    rows,cols=linear_sum_assignment(1-overlap)
    if min(overlap[rows,cols])<minimum_iou:
        return {"accepted":False,"reason":"later observations do not match the earlier identity seam"}
    mapping={pids[i]:pids[j] for i,j in zip(rows,cols)}
    corrected=[];recovered=[]
    for frame in sorted(forward):
        if frame<start:
            continue
        old=dict(forward[frame])
        if frame>=stop:
            for p in pids:
                forward[frame].pop(p,None)
            for p in pids:
                if p in old:
                    forward[frame][mapping[p]]=old[p]
        if start<frame<stop and frame in backward:
            for p in pids:
                actual=mapping[p]
                if p not in backward[frame]:
                    continue
                if actual not in forward[frame]:
                    recovered.append({"frame":frame,"playerId":actual})
                elif iou(forward[frame][actual],backward[frame][p])<.3:
                    corrected.append({"frame":frame,"playerId":actual})
                forward[frame][actual]=backward[frame][p]
    return {"accepted":True,"mapping":mapping,"corrected":corrected,"recovered":recovered,
        "laterEvidenceFrame":stop,"earlierSeamFrame":start}


def resample(roster,boxes,registration,*,fps,snap,first,last,rate=10):
    from . import kinematics
    times=np.arange(math.ceil((first-snap)/fps*rate),math.floor((last-snap)/fps*rate)+1)/rate
    output=[TrackingFrame(t=round(float(t),3),events=["ball_snap"] if abs(t)<1e-9 else []) for t in times]
    for pid in roster:
        observed=[]
        for frame,identities in sorted(boxes.items()):
            if pid not in identities:
                continue
            x1,y1,x2,y2=identities[pid]
            xy=registration.to_field(frame,[[(x1+x2)/2,y2]])[0]
            if np.isfinite(xy).all() and -5<=xy[0]<=125 and -5<=xy[1]<=58:
                observed.append(((frame-snap)/fps,xy))
        if not observed:
            continue
        ts=np.array([t for t,_ in observed]);xy=np.array([p for _,p in observed]);indices=[];path=[]
        for i,t in enumerate(times):
            at=int(np.searchsorted(ts,t))
            if at<len(ts) and abs(ts[at]-t)<1e-7:
                indices.append(i);path.append(xy[at]);continue
            if at==0 or at==len(ts):
                nearest=0 if at==0 else len(ts)-1
                if abs(ts[nearest]-t)<=.5/rate+1e-7:
                    indices.append(i);path.append(xy[nearest])
                continue
            if ts[at]-ts[at-1]>.25:
                continue
            a=(t-ts[at-1])/(ts[at]-ts[at-1]);indices.append(i);path.append(xy[at-1]*(1-a)+xy[at]*a)
        cuts=[0]+[i for i in range(1,len(indices)) if indices[i]-indices[i-1]>1]+[len(indices)]
        for lo,hi in zip(cuts,cuts[1:]):
            if lo==hi:
                continue
            at=indices[lo:hi];coords=np.array(path[lo:hi])
            kin=kinematics.derive(times[at],kinematics.smooth(coords,window=5))
            for j,i in enumerate(at):
                output[i].players[pid]=PlayerFrame(float(coords[j,0]),float(coords[j,1]),
                    s=float(kin["s"][j]),a=float(kin["a"][j]),dir=float(kin["dir"][j]),
                    o=float(kin["dir"][j]),o_source="from_velocity")
    return output


def track(frames,boxes,registration,*,fps,snap,weights,cache_dir,capacity=11,
          backward=True,jerseys=True,log=lambda *a:None):
    from .automatic import choose_formation
    from .identity import from_video
    from .sam_video import track_boxes

    anchor=choose_formation(boxes,frames,registration,fps,capacity,before=snap)
    # The existing constrained jersey classifier supplies stable team labels.
    # For a cropped formation use upper bounds, never force equal visible counts.
    if len(boxes[anchor])==capacity*2:
        initial=from_video({anchor:frames[anchor]},{anchor:boxes[anchor]},registration,
            anchor,fps,snap,capacity)
        roster=initial.roster
    else:
        from .detect.color import kit_color,cluster_teams
        samples={str(i):[kit_color(frames[anchor],d)] for i,d in enumerate(boxes[anchor])}
        labels,centers=cluster_teams({k:v for k,v in samples.items() if v[0] is not None})
        light=int(np.argmax(centers[:,0]))
        roster={f"p{i:02d}":RosterEntry(id=f"p{i:02d}",team="teamA" if labels.get(str(i))==light else "teamB")
                for i in range(len(boxes[anchor]))}
    seeds={pid:[d.x1,d.y1,d.x2,d.y2] for pid,d in zip(roster,boxes[anchor])}
    log(f"    automatic formation: source {anchor}, {len(roster)} identities")
    tracked=track_boxes(frames,anchor,seeds,fps=fps,weights=weights,cache_dir=cache_dir,log=log)
    if anchor>min(frames):
        tracked.update(track_boxes(frames,anchor,seeds,fps=fps,weights=weights,cache_dir=cache_dir,reverse=True,log=log))
    teams={pid:e.team for pid,e in roster.items()};reconciliations=[]
    if backward:
        for window in crossing_windows(tracked,teams,fps):
            start,stop=window["start"],window["stop"];pids=window["players"]
            later={pid:tracked[stop][pid] for pid in pids}
            reverse=track_boxes(frames,stop,later,fps=fps,weights=weights,cache_dir=cache_dir,
                                reverse=True,stop=start,log=log)
            result=reconcile_window(tracked,reverse,start,stop,pids)
            reconciliations.append({**window,**result})
    from .identity_observations import reconcile
    original_masks=tracked
    tracked,observation_checks=reconcile(tracked,boxes,registration,roster,anchor,fps)
    log(f"    identity observations: {observation_checks}")
    reacquisitions=[]
    if backward:
        from .reacquisition import recover
        reacquisitions=recover(tracked,original_masks,boxes,frames,fps=fps,weights=weights,
            cache_dir=cache_dir,log=log)
    out=PlayTrack(roster=roster,frames=resample(roster,tracked,registration,fps=fps,snap=snap,
        first=min(frames),last=max(frames)),meta={"source":"vision","rate":10,"fps_in":fps,
        "identity":{"method":"automatic detector-seeded SAM 2 with backward crossing and single-player reacquisition checks",
            "automatic":True,"anchorSourceFrame":anchor,"rosterSize":len(roster),
            "newIdentitiesAfterAnchor":0,"operatorAnchors":[],
            "sourceBoxes":{str(f):b for f,b in tracked.items()},"backwardChecks":reconciliations,
            "observationChecks":observation_checks,"teamLocked":True,
            "reacquisitionChecks":reacquisitions,
            "note":"Fixed IDs and teams persist. Hidden poses are absent from observation data and filled explicitly for replay; backward corrections require a matching earlier identity seam."}})
    if jerseys:
        from .jerseys import annotate
        ocr_boxes={}
        for frame,people in tracked.items():
            detections=boxes.get(frame,[])
            pids=list(people)
            if not detections or not pids:
                continue
            proposals=[[d.x1,d.y1,d.x2,d.y2] for d in detections]
            overlap=np.array([[iou(people[pid],b) for b in proposals] for pid in pids])
            rows,cols=linear_sum_assignment(1-overlap)
            ocr_boxes[frame]={pids[i]:proposals[j] for i,j in zip(rows,cols)
                if overlap[i,j]>.35 and detections[j].score>.5}
        annotate(out,frames,ocr_boxes,fps=fps,cache_dir=Path(cache_dir)/"ocr",log=log)
    return out
