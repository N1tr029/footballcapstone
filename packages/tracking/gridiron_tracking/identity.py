"""Fixed identities from an inspected formation, dormant across missed detections."""
from __future__ import annotations

from dataclasses import dataclass
import math
import numpy as np
from scipy.optimize import linear_sum_assignment

from .contract import PlayTrack, PlayerFrame, RosterEntry, TrackingFrame


@dataclass
class Observation:
    xy: np.ndarray
    color: np.ndarray | None = None
    box: tuple | None = None


def anchored(observations, anchor, teams, *, fps, snap, rate=10, max_gap=.25, anchors=(), images=None):
    """No births after the checked anchor and no long-gap invented positions.

    Same-kit crossings remain estimates. Optional reviewed field points resolve
    specific identity ambiguities and remain visible in the provenance.
    """
    from . import kinematics

    seeds = observations.get(anchor, [])
    if not seeds or len(seeds) != len(teams):
        raise ValueError("identity anchor requires one team label per seed observation")
    ids = [f"p{i:02d}" for i in range(len(seeds))]
    keys = [(int(a["frame"]), a["playerId"]) for a in anchors]
    if len(keys) != len(set(keys)):
        raise ValueError("duplicate frame/player identity anchors")
    for a in anchors:
        if a["playerId"] not in ids or int(a["frame"]) not in observations or not np.isfinite(a["field"]).all():
            raise ValueError("identity anchor must name an existing formation ID, source frame and finite field point")
    roster = {pid: RosterEntry(id=pid, label=pid, team=team) for pid, team in zip(ids, teams)}
    colors = [p.color for p in seeds]
    centers = {}
    for team in set(teams):
        samples = [p.color for p,t in zip(seeds,teams) if t == team and p.color is not None]
        if samples:
            centers[team] = np.median(samples, axis=0)
    measured = {pid: {anchor: np.array(p.xy, float)} for pid,p in zip(ids,seeds)}
    boxes = {anchor: {pid:p.box for pid,p in zip(ids,seeds) if p.box is not None}}
    overrides = {(int(a["frame"]),a["playerId"]):np.asarray(a["field"],float) for a in anchors}

    for order in ([f for f in sorted(observations) if f > anchor],
                  [f for f in sorted(observations,reverse=True) if f < anchor]):
        last = np.array([p.xy for p in seeds],float)
        velocity = np.zeros_like(last)
        seen = np.full(len(seeds),anchor)
        last_boxes = [p.box for p in seeds]
        previous = anchor
        for frame in order:
            points = observations[frame]
            if not points:
                continue
            predicted_boxes = _flow_boxes(images, previous, frame, last_boxes) if images is not None else last_boxes
            xy = np.array([p.xy for p in points])
            delta = (frame-seen)/fps
            prediction = last+velocity*np.clip(delta[:,None],-.4,.4)
            cost = np.full((len(ids),len(points)+len(ids)),1e6)
            for i,pid in enumerate(ids):
                gap = abs(delta[i]); gate = min(14.,1.+12*gap)
                cost[i,len(points)+i] = min(8.,gate+.75)
                for j,point in enumerate(points):
                    miss = float(np.linalg.norm(point.xy-prediction[i]))
                    if miss > gate or np.linalg.norm(point.xy-last[i]) > 1.+12*gap:
                        continue
                    appearance = 0.
                    if point.color is not None and colors[i] is not None:
                        distances = {t:float(np.linalg.norm(point.color-c)) for t,c in centers.items()}
                        own = distances[teams[i]]
                        other = min((d for t,d in distances.items() if t != teams[i]),default=own)
                        if own-other > 30:
                            continue
                        appearance = min(2.,float(np.linalg.norm(point.color-colors[i]))/65)
                    box_cost = 0.
                    if images is not None and predicted_boxes[i] is not None and point.box is not None:
                        b = np.asarray(predicted_boxes[i]); q = np.asarray(point.box)
                        scale = max(30., b[3]-b[1])
                        center_error = np.linalg.norm((b[:2]+b[2:]-q[:2]-q[2:])/2)/scale
                        if abs(delta[i]) < .15 and center_error > .8:
                            continue
                        box_cost = 5*center_error
                    cost[i,j] = .25*miss+appearance+box_cost
            forced = set()
            for i,pid in enumerate(ids):
                if (frame,pid) in overrides:
                    target = overrides[frame,pid]
                    j = int(np.argmin(np.linalg.norm(xy-target,axis=1)))
                    if np.linalg.norm(xy[j]-target) > 3:
                        raise ValueError(f"identity anchor {pid} frame {frame} has no matching detection")
                    if j in forced:
                        raise ValueError("two identity anchors claim the same detection")
                    forced.add(j)
                    cost[i,:] = 1e6; cost[:,j] = 1e6; cost[i,j] = -100
            for i,j in zip(*linear_sum_assignment(cost)):
                if j >= len(points) or cost[i,j] >= 1e6:
                    continue
                point = points[j]; dt = delta[i]
                if abs(dt) > 1e-6:
                    v = (point.xy-last[i])/dt
                    speed = np.linalg.norm(v)
                    if speed > 12:
                        v *= 12/speed
                    velocity[i] = .65*velocity[i]+.35*v
                last[i] = point.xy; seen[i] = frame
                predicted_boxes[i] = point.box
                measured[ids[i]][frame] = point.xy
                if point.box is not None:
                    boxes.setdefault(frame,{})[ids[i]] = point.box
            last_boxes = predicted_boxes
            previous = frame

    first,last_frame = min(observations),max(observations)
    times = np.arange(math.ceil((first-snap)/fps*rate),math.floor((last_frame-snap)/fps*rate)+1)/rate
    frames = [TrackingFrame(t=round(float(t),3),events=["ball_snap"] if abs(t)<1e-9 else []) for t in times]
    interpolation = {}
    for pid in ids:
        source = sorted(measured[pid]); ts = np.array([(f-snap)/fps for f in source])
        positions = np.array([measured[pid][f] for f in source]); good,xy = [],[]
        for i,t in enumerate(times):
            at = int(round(t*fps))+snap
            if at in measured[pid]:
                good.append(i); xy.append(measured[pid][at]); continue
            right = int(np.searchsorted(ts,t))
            if right == 0 or right == len(ts) or ts[right]-ts[right-1] > max_gap:
                continue
            alpha = (t-ts[right-1])/(ts[right]-ts[right-1])
            good.append(i); xy.append(positions[right-1]*(1-alpha)+positions[right]*alpha)
            interpolation.setdefault(str(i+1),[]).append(pid)
        cuts = [0]+[i for i in range(1,len(good)) if good[i]-good[i-1]>1]+[len(good)]
        for lo,hi in zip(cuts,cuts[1:]):
            if hi <= lo:
                continue
            # Keep measured feet on their detection. Smoothing positions can pull
            # two nearby linemen onto the same body; smooth velocities separately.
            indices = good[lo:hi]; path = np.asarray(xy[lo:hi])
            kin = kinematics.derive(times[indices],path)
            for k,fi in enumerate(indices):
                frames[fi].players[pid] = PlayerFrame(x=float(path[k,0]),y=float(path[k,1]),
                    s=float(kin["s"][k]),a=float(kin["a"][k]),dir=float(kin["dir"][k]),
                    o=float(kin["dir"][k]),o_source="from_velocity")
    return PlayTrack(roster=roster,frames=frames,meta={"source":"vision","rate":rate,"fps_in":fps,
        "identity":{"method":"checked full-formation anchor; fixed dormant identities",
            "anchorSourceFrame":anchor,"rosterSize":len(ids),"newIdentitiesAfterAnchor":0,
            "interpolatedSamples":interpolation,"sourceBoxes":{str(f):b for f,b in boxes.items()},
            "operatorAnchors":list(anchors),
            "note":"Same-kit crossings/reacquisition are estimates; missing poses remain absent."}})


def from_video(frames,boxes,registration,anchor,fps,snap,capacity=11,anchors=()):
    from .detect.color import kit_color,cluster_teams
    from .teams import constrained_teams

    observations = {}
    for fi,detections in boxes.items():
        pts = registration.to_field(fi,[d.anchor()[0] for d in detections]) if detections else []
        observations[fi] = [Observation(np.asarray(p),kit_color(frames[fi],d),(d.x1,d.y1,d.x2,d.y2))
                           for d,p in zip(detections,pts) if np.isfinite(p).all()]
    seed = observations.get(anchor,[])
    if len(seed) != capacity*2:
        raise ValueError(f"inspected formation has {len(seed)} detections, expected {capacity*2}")
    samples = {str(i):[p.color] for i,p in enumerate(seed) if p.color is not None}
    _,centers = cluster_teams(samples)
    if len(centers) != 2:
        raise ValueError("seed frame does not distinguish two kits")
    centers = centers[np.argsort(-centers[:,0])]; probabilities = {}
    for i,p in enumerate(seed):
        d = np.linalg.norm(centers-p.color,axis=1) if p.color is not None else np.ones(2)
        prob = np.exp(-d*d/(2*35**2)); probabilities[str(i)] = prob/prob.sum()
    labels = constrained_teams(probabilities,[list(probabilities)],capacity=capacity,full_frame=0)
    return anchored(observations,anchor,[labels[str(i)] for i in range(len(seed))],
                    fps=fps,snap=snap,anchors=anchors,images=frames)


def _flow_boxes(images, previous, current, boxes):
    """Local image motion supplements field motion through camera pans and turns."""
    import cv2
    a = cv2.cvtColor(images[previous], cv2.COLOR_BGR2GRAY)
    b = cv2.cvtColor(images[current], cv2.COLOR_BGR2GRAY)
    h,w = a.shape
    points,owners = [],[]
    for i,box in enumerate(boxes):
        if box is None:
            continue
        x1,y1,x2,y2 = box
        # Interior features avoid grass and neighbouring players near box edges.
        xa,xb = max(0,int(x1+.25*(x2-x1))),min(w,int(x2-.25*(x2-x1)))
        ya,yb = max(0,int(y1+.15*(y2-y1))),min(h,int(y2-.2*(y2-y1)))
        if xb <= xa or yb <= ya:
            continue
        pts = cv2.goodFeaturesToTrack(a[ya:yb,xa:xb],12,.03,5)
        if pts is not None:
            pts = pts.reshape(-1,2)+[xa,ya]
            points.extend(pts); owners.extend([i]*len(pts))
    out = list(boxes)
    if not points:
        return out
    pts = np.asarray(points,np.float32).reshape(-1,1,2)
    moved,status,_ = cv2.calcOpticalFlowPyrLK(a,b,pts,None,winSize=(25,25),maxLevel=3)
    back,valid,_ = cv2.calcOpticalFlowPyrLK(b,a,moved,None,winSize=(25,25),maxLevel=3)
    good = status.ravel().astype(bool)&valid.ravel().astype(bool)&(np.linalg.norm(back-pts,axis=2).ravel()<2)
    for i,box in enumerate(boxes):
        use = good & (np.asarray(owners)==i)
        if use.sum() >= 3:
            dx,dy = np.median((moved-pts).reshape(-1,2)[use],axis=0)
            out[i] = tuple(np.asarray(box)+[dx,dy,dx,dy])
    return out
