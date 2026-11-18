"""Continuous replay actors from sparse evidence, without changing model inputs."""
from __future__ import annotations

from collections import Counter
from copy import deepcopy

import numpy as np
from scipy.ndimage import median_filter

from . import kinematics
from .sources.ngs import contract_angle_to_ngs


def continuous_path(indices, positions, count, rate):
    """Use both ends of a gap; coast briefly then hold at an open-ended gap."""
    indices=np.asarray(indices);positions=np.asarray(positions,dtype=float)
    grid=np.arange(count)
    path=np.column_stack([np.interp(grid,indices,positions[:,axis]) for axis in (0,1)])
    horizon=.5
    if len(indices)>1:
        for edge,next_at,sign in ((0,1,-1),(-1,-2,1)):
            dt=(indices[edge]-indices[next_at])/rate
            velocity=(positions[edge]-positions[next_at])/dt
            speed=np.linalg.norm(velocity)
            if speed>12:
                velocity*=12/speed
            for i in (range(indices[0]) if sign<0 else range(indices[-1]+1,count)):
                seconds=(i-indices[edge])/rate
                # Integrate a velocity that decays to zero, rather than letting
                # an unseen athlete run forever or snap back after half a second.
                distance_time=np.sign(seconds)*horizon*(1-np.exp(-abs(seconds)/horizon))
                path[i]=positions[edge]+velocity*distance_time
    # Remove isolated foot-box spikes, then smooth positions (not only speed).
    if count>=5:
        path=median_filter(path,size=(3,1),mode="nearest")
        path=kinematics.smooth(path,window=7)
    # A bounded display path avoids teleporting even if an identity was recovered
    # far from its last sighting. These coordinates remain explicitly estimated.
    for order in (range(1,count),range(count-2,-1,-1)):
        for i in order:
            previous=i-1 if order.step>0 else i+1
            delta=path[i]-path[previous];distance=np.linalg.norm(delta)
            if distance>12/rate:
                path[i]=path[previous]+delta*(12/rate/distance)
    path[:,0]=np.clip(path[:,0],-5,125);path[:,1]=np.clip(path[:,1],-5,58)
    return path


def stabilize_document(document):
    """Keep one actor per known roster ID on every replay frame.

    A missing detection is a change in observation status, not actor membership.
    Original samples travel in ``observation`` and are restored by read_play for
    analysis. Interpolated/coasted markers never become fresh model evidence.
    """
    if document.get("provenance",{}).get("replayStability"):
        return document
    doc=deepcopy(document);frames=doc.get("frames",[]);roster=doc.get("players",[])
    if not frames or not roster:
        return doc
    rate=doc["frameRate"];count=len(frames);sources=Counter();missing=[]
    original=[{p["nflId"]:p for p in f["players"]} for f in frames]
    original_ball_frames=sum(bool(f.get("ball")) for f in frames)
    original_ball_sources=dict(Counter(f["ball"].get("dataOrigin","unverified") for f in frames if f.get("ball")))
    dense=[{} for _ in frames]
    for player in roster:
        pid=player["nflId"];indices=[i for i,people in enumerate(original) if pid in people]
        if not indices:
            missing.append(pid)
            continue  # There is no defensible coordinate for a never-seen athlete.
        positions=[[original[i][pid]["x"],original[i][pid]["y"]] for i in indices]
        path=continuous_path(indices,positions,count,rate)
        kin=kinematics.derive(np.arange(count)/rate,path)
        for i in range(count):
            right=int(np.searchsorted(indices,i));seen=pid in original[i]
            if seen:
                status="tracked";at=i;age=0.
            elif i<indices[0]:
                status="backfilled";at=indices[0];age=(at-i)/rate
            elif i>indices[-1]:
                age=(i-indices[-1])/rate;at=indices[-1]
                status="predicted" if age<=.5 else "last_known"
            else:
                status="interpolated";at=indices[right-1];age=(i-at)/rate
            point=deepcopy(original[at][pid]);point.pop("observation",None)
            simulated=seen and doc.get("simulated") and i+1>=doc.get("simulationStartFrameId",1)
            if simulated:
                status="simulated"
            if seen and not simulated:
                point["observation"]=deepcopy(original[i][pid])
            point.update(x=round(float(path[i,0]),2),y=round(float(path[i,1]),2),
                s=round(float(kin["s"][i]),2),observed=bool(seen and not simulated),trackingStatus=status,
                positionAgeSeconds=round(age,3),dataOrigin="smoothed" if seen else status,
                club=player["club"],side=player["side"])
            point["dir"]=round(float(contract_angle_to_ngs(kin["dir"][i])),2)
            if point.get("oSource") in ("from_velocity","from_dir","assumed"):
                point["o"]=point["dir"]
            sources[status]+=1;dense[i][pid]=point
    for i,frame in enumerate(frames):
        frame["players"]=[dense[i][p["nflId"]] for p in roster if p["nflId"] in dense[i]]
    _continuous_ball(doc,dense)
    doc.setdefault("provenance",{})["replayStability"]={
        "method":"fixed roster; future-bracketed interpolation; damped coast and hold; smoothed display positions",
        "sources":dict(sources),"neverObservedIds":missing,"maximumDisplaySpeedYardsPerSecond":12,
        "note":"All roster actors persist. Missing positions are display estimates, not new observations. Original samples are retained in observation."}
    doc.setdefault("trackingQuality",{}).update(
        replayPlayersPerFrame=len(roster)-len(missing),replayRosterStable=not missing,
        replayEstimatedSamples=sum(n for source,n in sources.items() if source!="tracked"),
        replayBallFrames=sum(bool(f.get("ball")) for f in frames))
    quality=doc["trackingQuality"]
    quality["analysisBallFrames"]=quality.get("ballFrames",original_ball_frames)
    quality["analysisBallSources"]=quality.get("ballSources",original_ball_sources)
    quality["ballFrames"]=sum(bool(f.get("ball")) for f in frames)
    quality["ballSources"]=dict(Counter(f["ball"].get("dataOrigin","unverified") for f in frames if f.get("ball")))
    return doc


def _continuous_ball(doc,people):
    """An already accepted pass can follow the same continuous carrier actors."""
    evidence=doc.get("provenance",{}).get("automaticPass") or doc.get("provenance",{}).get("reviewedPass")
    if not evidence:
        return
    ids={p.get("trackId"):p["nflId"] for p in doc["players"]}
    passer=ids.get(evidence.get("passerId"));receiver=ids.get(evidence.get("receiverId"))
    center=ids.get(evidence.get("centerId"));rate=doc["frameRate"];snap=doc["snapFrameId"]-1
    throw=evidence.get("throwTime");arrival=evidence.get("arrivalTime",evidence.get("catchTime"))
    if throw is None or arrival is None:
        return
    ti=int(round(throw*rate))+snap;ci=int(round(arrival*rate))+snap
    if not 0<=ti<ci<len(people) or passer not in people[ti] or receiver not in people[ci]:
        return
    for i,frame in enumerate(doc["frames"]):
        original=deepcopy(frame.get("ball"));carrier=None
        if ti<i<ci:
            a=(i-ti)/(ci-ti);p=people[ti][passer];q=people[ci][receiver]
            point={"x":p["x"]*(1-a)+q["x"]*a,"y":p["y"]*(1-a)+q["y"]*a}
            source="interpolated"
        else:
            carrier=(center if i<=snap else passer) if i<=ti else receiver
            point=people[i].get(carrier);source="carrier"
        if point is not None:
            frame["ball"]={"x":round(point["x"],2),"y":round(point["y"],2),"s":0.,
                "dataOrigin":source,"observed":False,"observation":original}
            if carrier is not None:
                frame["ball"]["carrierNflId"]=carrier


def display_track(track):
    """The video uses the same continuous trajectories as the public replay JSON."""
    from .contract import PlayerFrame,Ball
    from .hindsight import to_play_json,nfl_ids
    doc=to_play_json(track)
    doc["provenance"]["automaticPass"]=track.meta.get("automatic_pass")
    doc["provenance"]["reviewedPass"]=track.meta.get("reviewed_pass")
    doc=stabilize_document(doc);ids={n:pid for pid,n in nfl_ids(track).items()}
    out=deepcopy(track)
    for frame,shown in zip(out.frames,doc["frames"]):
        frame.players={ids[p["nflId"]]:PlayerFrame(x=p["x"],y=p["y"]) for p in shown["players"]}
        if shown.get("ball"):
            b=shown["ball"];frame.ball=Ball(b["x"],b["y"],source=b.get("dataOrigin"))
    out.meta["replay_display"]=doc["provenance"]["replayStability"]
    return out
