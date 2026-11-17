"""Infer pass endpoints from visible flight and tracked image-space bodies.

An airborne pixel and a player's feet lie at different heights. Projecting both
onto the grass and choosing the nearest player is therefore wrong. Endpoint
matching happens in each source image after compensating camera motion.
"""
from __future__ import annotations

import numpy as np


def interpolate_boxes(source_boxes, frame, fps):
    keys=sorted(int(k) for k in source_boxes)
    if frame in source_boxes:
        return source_boxes[frame]
    if str(frame) in source_boxes:
        return source_boxes[str(frame)]
    at=int(np.searchsorted(keys,frame))
    if at==0 or at==len(keys) or keys[at]-keys[at-1]>.25*fps:
        return {}
    left,right=keys[at-1],keys[at]
    a=source_boxes.get(left,source_boxes.get(str(left),{}))
    b=source_boxes.get(right,source_boxes.get(str(right),{}))
    alpha=(frame-left)/(right-left)
    return {pid:(np.asarray(box)*(1-alpha)+np.asarray(b[pid])*alpha).tolist()
            for pid,box in a.items() if pid in b}


def infer(track, observations, registration, fps, offset):
    """Return an evidence-backed pass hypothesis, or an explicit abstention.

    No frame numbers, IDs or jersey labels are supplied by an operator. Six or
    more actual moving-ball sightings and plausible same-team endpoints are
    required. Arrival is inferred; catching/completion is not asserted from an
    endpoint intersection alone.
    """
    source_boxes=track.meta.get("identity",{}).get("sourceBoxes",{})
    if not source_boxes:
        return None,"no image-space identity trajectories for endpoint association"
    sightings=sorted((int(f),p) for f,p in observations.items())
    groups=[]
    for f,point in sightings:
        if not groups or f-groups[-1][-1][0]>2:
            groups.append([])
        groups[-1].append((f,point))
    hypotheses=[]
    quarterback_candidates=set()
    broadcast=track.meta.get("broadcast",{})
    los=broadcast.get("lineOfScrimmage");attacks=broadcast.get("attacks")
    snap_frame=track.frame_at(0)
    if los is not None and attacks in ("left","right") and snap_frame:
        for team in {e.team for e in track.roster.values()}:
            players={pid:p for pid,p in snap_frame.players.items() if track.roster[pid].team==team}
            front={pid:p for pid,p in players.items() if abs(p.x-los)<2.5}
            if len(front)<3:
                continue
            mid=float(np.median([p.y for p in front.values()]))
            central={pid:p for pid,p in front.items() if abs(p.y-mid)<6}
            if not central:
                continue
            center=min(central,key=lambda pid:abs(central[pid].x-los))
            candidates=[]
            for pid,p in players.items():
                depth=p.x-los if attacks=="left" else los-p.x
                if depth>2.5:
                    candidates.append((abs(p.y-central[center].y)+.05*depth,pid))
            if candidates:
                candidates.sort();quarterback_candidates.add(candidates[0][1])
    for group in groups:
        if len(group)<6:
            continue
        first,last=group[0][0],group[-1][0]
        t=np.array([(f-first)/fps for f,_ in group])
        xy=np.array([p["field"] for _,p in group])
        duration=t[-1]-t[0];span=float(np.linalg.norm(xy[-1]-xy[0]))
        if duration<=0 or span<3 or not 10<=span/duration<=45:
            continue
        coeff=np.polyfit(t,xy,2)
        residual=float(np.sqrt(np.mean((np.array([np.polyval(coeff,ti) for ti in t])-xy)**2)))
        if residual>.75:
            continue
        endpoints=[]
        for before in (True,False):
            candidates=[]
            bounds=(first-round(fps*.45),first) if before else (last,last+round(fps*.25))
            for frame in range(*bounds):
                h=registration.at(frame)
                if h is None:
                    continue
                field=np.polyval(coeff,(frame-first)/fps)
                if not np.isfinite(field).all():
                    continue
                pixel=h.to_image([field])[0]
                for pid,box in interpolate_boxes(source_boxes,frame,fps).items():
                    if before and quarterback_candidates and pid not in quarterback_candidates:
                        continue
                    x1,y1,x2,y2=box;w=x2-x1;height=y2-y1
                    if w<=0 or height<=0:
                        continue
                    if not (x1-.25*w<=pixel[0]<=x2+.25*w and y1-.2*height<=pixel[1]<=y1+.65*height):
                        continue
                    center=np.array([(x1+x2)/2,y1+.3*height])
                    error=float(np.linalg.norm((pixel-center)/[max(20.,.75*w),max(20.,.35*height)]))
                    if error>1.5:
                        continue
                    # A normal forward-pass release is behind the line. This is
                    # additional evidence where the broadcast provides direction.
                    broadcast=track.meta.get("broadcast",{})
                    los=broadcast.get("lineOfScrimmage");attacks=broadcast.get("attacks")
                    foot=h.to_field([[(x1+x2)/2,y2]])[0]
                    if before and los is not None and attacks in ("left","right"):
                        depth=(foot[0]-los) if attacks=="left" else (los-foot[0])
                        if depth<1.5:
                            continue
                    candidates.append({"frame":frame,"playerId":pid,"error":error,
                        "pixel":pixel.tolist(),"box":box,"team":track.roster[pid].team})
            endpoints.append(candidates)
        releases,arrivals=endpoints
        if not releases or not arrivals:
            continue
        # The earliest plausible hand/body intersection identifies release;
        # intervening blockers crossed later by the flight cannot become passers.
        releases=sorted(releases,key=lambda c:(round(c["error"]/.2),c["frame"]))
        arrivals=sorted(arrivals,key=lambda c:(c["error"],c["frame"]))
        for release in releases:
            valid=[a for a in arrivals if a["playerId"]!=release["playerId"] and a["team"]==release["team"]]
            if not valid:
                continue
            arrival=valid[0];duration=(arrival["frame"]-release["frame"])/fps
            if not .15<=duration<=2.0:
                continue
            confidence=float(np.clip(.9-.12*release["error"]-.15*arrival["error"]-.1*residual,.0,.95))
            if confidence<.65:
                continue
            hypotheses.append({"throwFrame":release["frame"],"catchFrame":arrival["frame"],
                "passerId":release["playerId"],"receiverId":arrival["playerId"],
                "confidence":round(confidence,3),"visibleSourceFrames":[f for f,_ in group],
                "fitResidualYards":round(residual,3),"releaseImageError":round(release["error"],3),
                "arrivalImageError":round(arrival["error"],3),
                "source":"automatic camera-compensated flight/body association",
                "outcome":"unknown"})
            break
    if not hypotheses:
        return None,"no sufficiently supported release/arrival association from visible flight"
    return max(hypotheses,key=lambda h:h["confidence"]),None


def apply(track, hypothesis):
    """Publish automatically inferred events and an explicitly estimated path."""
    from .contract import Ball
    fps=float(track.meta["fps_in"]);snap=track.meta["clip"]["snap_source_frame"]
    throw=track.frame_at((hypothesis["throwFrame"]-snap)/fps)
    arrival=track.frame_at((hypothesis["catchFrame"]-snap)/fps)
    passer,receiver=hypothesis["passerId"],hypothesis["receiverId"]
    if throw.t>=arrival.t or passer not in throw.players or receiver not in arrival.players:
        return False,"automatic endpoints have no retained player poses at export resolution"
    start,end=throw.players[passer],arrival.players[receiver]
    original=track.meta.get("ball",{});raw={str(i+1):f.ball.to_json() for i,f in enumerate(track.frames) if f.ball}
    snap_frame=track.frame_at(0)
    offense=track.roster[passer].team
    # The central front player aligned with the passer is a centre candidate.
    # Keep the pre-snap ball unknown if that geometry is not sufficiently clear.
    center=None
    if passer in snap_frame.players:
        qb=snap_frame.players[passer];candidates=[]
        for pid,p in snap_frame.players.items():
            if pid==passer or track.roster[pid].team!=offense:
                continue
            dx=abs(p.x-qb.x);dy=abs(p.y-qb.y)
            if 2<=dx<=9 and dy<1.5:
                candidates.append((dy,pid))
        if candidates:
            center=min(candidates)[1]
    counts={};carriers={}
    for i,f in enumerate(track.frames):
        f.events=[e for e in f.events if e not in ("pass_forward","pass_arrived","pass_outcome_caught")]
        f.ball=None
        if throw.t<f.t<arrival.t:
            a=(f.t-throw.t)/(arrival.t-throw.t)
            f.ball=Ball(start.x*(1-a)+end.x*a,start.y*(1-a)+end.y*a,source="interpolated")
        else:
            pid=(center if f.t<=0 else passer) if f.t<=throw.t else receiver
            if pid in f.players:
                p=f.players[pid];f.ball=Ball(p.x,p.y,source="carrier");carriers[str(i+1)]=pid
        if f.ball:
            counts[f.ball.source]=counts.get(f.ball.source,0)+1
    throw.events.append("pass_forward");arrival.events.append("pass_arrived")
    track.meta["automatic_pass"]={**hypothesis,"throwTime":throw.t,"arrivalTime":arrival.t,
        "centerId":center,"validated":True,
        "note":"Automatically inferred endpoints. Carried and airborne ground positions are estimates; completion is unknown."}
    track.meta["possessionTeam"]=offense
    track.meta["ball"]={**original,"by_source":counts,"carriers":carriers,
        "raw_projected_observations":raw,"raw_measured_fraction":original.get("measured_fraction",0),
        "measured_fraction":0.,"frames_with_a_position":sum(counts.values()),
        "coverage":sum(counts.values())/len(track.frames),
        "ground_path_source":"automatic possession continuity and interpolated flight"}
    return True,None
