"""Automatic formation and snap selection from camera-compensated detections."""
from __future__ import annotations

from dataclasses import dataclass
import numpy as np
from scipy.optimize import linear_sum_assignment
from scipy.ndimage import median_filter


@dataclass
class SnapEstimate:
    frame: int | None
    confidence: float
    reason: str


def snap_from_positions(points, fps):
    """Find a sustained transition from a set formation to common player motion.

    The input is on the grass plane, so a camera pan does not count as a snap.
    No player IDs or ball annotations are required. A clip already in motion is
    reported as missing its snap rather than assigned a confident arbitrary zero.
    """
    delta=max(1,round(fps*.2));ordered=sorted(points);samples=[]
    for f in ordered:
        if f+delta not in points or len(points[f])<8 or len(points[f+delta])<8:
            continue
        a,b=np.asarray(points[f]),np.asarray(points[f+delta])
        distances=np.linalg.norm(a[:,None,:]-b[None,:,:],axis=2)
        rows,cols=linear_sum_assignment(distances)
        moves=distances[rows,cols]
        samples.append((f+delta/2,float(np.median(moves)/(delta/fps))))
    if len(samples)<fps:
        return SnapEstimate(None,0.,"too little formation footage to establish a snap")
    times=np.array([f for f,_ in samples]);v=median_filter([v for _,v in samples],size=max(3,round(.25*fps)|1))
    baseline=float(np.percentile(v[:max(3,round(fps))],40))
    if baseline>1.0:
        return SnapEstimate(None,0.,"the clip opens with players already moving")
    low=max(.45,baseline+.3);high=max(1.4,baseline*3+.65)
    duration=max(3,round(fps*.3))
    for i in range(max(3,round(fps*.5)),len(v)-duration):
        if np.median(v[i:i+duration])<high or np.mean(v[i:i+duration]>high)<.7:
            continue
        j=i
        while j>0 and v[j-1]>low:
            j-=1
        # Fit the onset of the acceleration ramp; the signal is timestamped at
        # each displacement window's midpoint, rather than its leading edge.
        lo=max(j-1,0);hi=min(i+duration,len(v));x=times[lo:hi]/fps;y=np.asarray(v[lo:hi])
        slope,intercept=np.polyfit(x,y,1)
        onset=(baseline-intercept)/slope if slope>1 else times[j]/fps
        estimate=int(round(np.clip(onset*fps,times[max(0,j-3)],times[i])))
        return SnapEstimate(estimate,.8,"sustained camera-compensated formation motion after a stationary interval")
    return SnapEstimate(None,0.,"no sustained set-to-live transition found")


def choose_formation(boxes, frames, registration, fps, capacity=11, before=None,allow_officials=False):
    """Choose the richest clear early formation; do not demand an invented 11/11.

    A full roster is a candidate inferred from detections and jersey separation,
    not the operator-checked evidence used by the assisted mode.
    """
    from .detect.color import kit_color,cluster_teams
    first=min(frames);end=min(first+2*fps,before) if before is not None else first+2*fps
    candidates=[]
    for frame,detections in boxes.items():
        # Before uniform roles have been checked, a full formation can contain
        # 22 athletes plus officials. Rejecting every >22-person view selects a
        # worse frame with missing athletes simply because its total is 22.
        extra=7 if allow_officials else 0
        if frame>end or not 8<=len(detections)<=capacity*2+extra:
            continue
        colors={str(i):[c] for i,d in enumerate(detections) if (c:=kit_color(frames[frame],d)) is not None}
        if len(colors)<len(detections)*.8:
            continue
        labels,centers=cluster_teams(colors)
        if len(centers)!=2:
            continue
        separation=float(np.linalg.norm(centers[0]-centers[1]))
        if separation<35:
            continue
        counts=[sum(v==k for v in labels.values()) for k in (0,1)]
        # At a probable full formation an occluded torso can put one athlete in
        # the other colour cluster. Treat 11/11 as a soft roster prior, not a
        # requirement that a cropped frame must satisfy.
        limit=capacity+extra if allow_officials else capacity+2 if len(detections)==capacity*2 else capacity
        if max(counts)>limit or min(counts)<3:
            continue
        score=len(detections)*2+np.mean([d.score for d in detections])
        candidates.append((score,frame))
    if not candidates:
        raise ValueError("no sufficiently clear two-team formation was found automatically")
    return max(candidates)[1]


def field_points(boxes,registration):
    result={}
    for frame,detections in boxes.items():
        if not detections:
            result[frame]=np.empty((0,2));continue
        pts=np.asarray(registration.to_field(frame,[d.anchor()[0] for d in detections]))
        result[frame]=pts[np.isfinite(pts).all(axis=1)]
    return result
