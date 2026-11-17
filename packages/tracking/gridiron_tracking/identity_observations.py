"""One-to-one, team-locked body observations for SAM's persistent identities."""
from __future__ import annotations

from collections import Counter
import numpy as np
from scipy.optimize import linear_sum_assignment

from .auto_identity import iou


def reconcile(tracked, detections, registration, roster, anchor, fps):
    """A SAM channel may drift onto an opponent; its roster team must not follow.

    Match each body once, gated by the fixed kit and physically plausible motion.
    Missing bodies stay missing in evidence; replay interpolation handles the gap.
    """
    teams={pid:e.team for pid,e in roster.items()};ids=list(roster);counts=Counter()
    result={anchor:{pid:[d.x1,d.y1,d.x2,d.y2] for pid,d in zip(ids,detections[anchor])}}
    expected={"teamA":"kitA","teamB":"kitB"}
    def xy(frame,box):
        return registration.to_field(frame,[[(box[0]+box[2])/2,box[3]]])[0]
    seed={pid:xy(anchor,box) for pid,box in result[anchor].items()}
    for order in ([f for f in sorted(tracked) if f>anchor],
                  [f for f in sorted(tracked,reverse=True) if f<anchor]):
        last={pid:np.array(point) for pid,point in seed.items()}
        velocity={pid:np.zeros(2) for pid in seed};seen={pid:anchor for pid in seed}
        for frame in order:
            bodies=detections.get(frame,[]);masks=tracked[frame]
            boxes=[[d.x1,d.y1,d.x2,d.y2] for d in bodies]
            coords=[xy(frame,b) for b in boxes]
            cost=np.full((len(ids),len(bodies)+len(ids)),1e5)
            for i,pid in enumerate(ids):
                cost[i,len(bodies)+i]=4.3
                if pid not in last:
                    continue
                dt=(frame-seen[pid])/fps;gap=abs(dt)
                prediction=last[pid]+velocity[pid]*np.clip(dt,-.5,.5)
                mask=masks.get(pid);wrong_mask=False
                known_team=expected.get(teams[pid])
                if mask is not None and boxes:
                    overlaps=[iou(mask,b) for b in boxes];best=int(np.argmax(overlaps))
                    wrong_mask=known_team is not None and overlaps[best]>.3 and bodies[best].team is not None and bodies[best].team!=known_team
                    if wrong_mask:
                        counts["opponentMasksRejected"]+=1
                for j,(body,box,point) in enumerate(zip(bodies,boxes,coords)):
                    if known_team is not None and body.team is not None and body.team!=known_team:
                        continue
                    if not np.isfinite(point).all():
                        continue
                    distance=float(np.linalg.norm(point-prediction))
                    if np.linalg.norm(point-last[pid])>1.+12*gap:
                        continue
                    overlap=iou(mask,box) if mask is not None and not wrong_mask else 0.
                    # Without a useful SAM mask, only a nearby, confidently
                    # same-kit body can recover an observation.
                    if overlap<.1 and (gap>1. or body.team!=expected.get(teams[pid]) or distance>1.+8*gap):
                        continue
                    cost[i,j]=2.5*(1-overlap)+.35*distance+(0.25 if body.team is None else 0.)
            result[frame]={}
            for i,j in zip(*linear_sum_assignment(cost)):
                if j>=len(bodies) or cost[i,j]>=4.3:
                    continue
                pid=ids[i];point=coords[j];dt=(frame-seen[pid])/fps
                if abs(dt)>1e-7:
                    v=(point-last[pid])/dt;speed=np.linalg.norm(v)
                    if speed>12:v*=12/speed
                    velocity[pid]=.6*velocity[pid]+.4*v
                if pid not in masks or iou(masks[pid],boxes[j])<.1:
                    counts["detectorRecoveries"]+=1
                result[frame][pid]=boxes[j];last[pid]=point;seen[pid]=frame
    counts["observedSamples"]=sum(len(people) for people in result.values())
    return result,{"method":"exclusive detector/SAM association with immutable kit and motion gates",**counts}
