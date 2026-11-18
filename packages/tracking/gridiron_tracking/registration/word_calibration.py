"""Calibrate from OCR word boxes on both painted number rows, without long lines."""
from __future__ import annotations

from types import SimpleNamespace
from collections import Counter
import subprocess
import numpy as np


def numbers(image,boxes,native):
    import cv2
    from ..jerseys import read_crop
    from .lines import turf_mask
    h,w=image.shape[:2];turf=turf_mask(image);ys,_=np.where(turf>0)
    if len(ys)<1000:return []
    top,bottom=np.percentile(ys,[2,98]);span=bottom-top
    bands=[("far",int(top),int(top+.29*span)),("near",int(top+.65*span),min(h,int(bottom+.04*span)))]
    proposals=[]
    for row,ya,yb in bands:
        if yb-ya<15:continue
        original=image[ya:yb]
        for angle in (0,180):
            crop=cv2.rotate(original,cv2.ROTATE_180) if angle else original
            for stretch in (2,3,4):
                try:readings=read_crop(cv2.resize(crop,None,fx=1,fy=stretch),native=native)
                except (OSError,subprocess.TimeoutExpired,ValueError):continue
                for r in readings:
                    if r["number"] not in (10,20,30,40,50) or r["confidence"]<.85 or "x" not in r:continue
                    x=r["x"]+r["width"]/2;y=r["y"]+r["height"]/2
                    if angle:x,y=1-x,1-y
                    x*=w;y=ya+y*(yb-ya)
                    if not .025<=r["width"]<=.22:continue
                    if any(a<=x<=c and b<=y<=d for a,b,c,d in boxes):continue
                    xa,xb=max(0,int(x-r["width"]*w/2)),min(w,int(x+r["width"]*w/2))
                    yy=int(np.clip(y,0,h-1))
                    if xb<=xa or turf[max(0,yy-10):min(h,yy+10),xa:xb].mean()/255<.3:continue
                    proposals.append({"number":r["number"],"pixel":[float(x),float(y)],"row":row})
    groups=[]
    for p in proposals:
        group=next((g for g in groups if g[0]["row"]==p["row"] and
            np.linalg.norm((np.asarray(g[0]["pixel"])-p["pixel"])/[w,h])<.04),None)
        if group is None:groups.append([p])
        else:group.append(p)
    result=[]
    for group in groups:
        counts=Counter(p["number"] for p in group);number,votes=counts.most_common(1)[0]
        if votes/len(group)<.75:continue
        points=[p["pixel"] for p in group if p["number"]==number]
        result.append({"number":number,"row":group[0]["row"],"pixel":np.median(points,axis=0).tolist(),"votes":votes})
    return result


def fit(image,labels,level):
    from ..field import NFL,NCAA,HIGH_SCHOOL
    from .homography import Homography
    from .vision import sanity_problems,paint_alignment,snap_to_paint,turf_overlap
    if len(labels)<5 or min(Counter(p["row"] for p in labels).get(r,0) for r in ("near","far"))<2:return None
    spec={"nfl":NFL,"college":NCAA,"high_school":HIGH_SCHOOL}.get(level)
    if spec is None:return None
    best=None
    for direction in (1,-1):
        pairs=[((10+p["number"] if direction==1 else 110-p["number"],
            spec.number_inset+1 if p["row"]=="near" else spec.width-spec.number_inset-1),p["pixel"]) for p in labels]
        if len({p[0][0] for p in pairs})<3:continue
        try:h=Homography.fit(np.array([a for a,_ in pairs]),np.array([b for _,b in pairs]),threshold_px=max(12,image.shape[1]*.02))
        except (ValueError,np.linalg.LinAlgError):continue
        if h.inliers is not None and h.inliers.sum()<5:continue
        if sanity_problems(h,(image.shape[1],image.shape[0]),spec,correspondences=pairs):continue
        h,score=snap_to_paint(image,h,spec);turf=turf_overlap(image,h,spec)
        if np.isfinite(score) and score>=1.6 and turf>=.75 and (best is None or score>best[0]):
            best=score,h,spec,pairs,turf
    return best


def calibrate(frames,boxes,*,fps,cache_dir,level,log=lambda *a:None):
    from ..jerseys import native_binary
    from .paint_auto import image_registration
    from .homography import transform
    from .vision import AutoRegistration
    native=native_binary(cache_dir)
    if native is None:return None
    evidence=[];reference=None
    order=sorted(frames);step=max(1,round(fps*.4))
    for frame in order[::step][:6]:
        labels=numbers(frames[frame],boxes.get(frame,[]),native)
        log(f"    field number OCR: source {frame}, {len(labels)} word landmarks")
        if len(labels)<4:continue
        if reference is None:reference=frame
        mapping=np.eye(3) if frame==reference else image_registration(frames[frame],frames[reference],boxes.get(frame,[]),boxes.get(reference,[]))
        if mapping is None:continue
        for label in labels:
            pixel=transform(mapping,[label["pixel"]])[0]
            evidence.append({**label,"frame":frame,"referencePixel":pixel.tolist()})
        if len({p["frame"] for p in evidence})<2:continue
        groups=[]
        for p in evidence:
            g=next((g for g in groups if g[0]["row"]==p["row"] and g[0]["number"]==p["number"] and
                np.linalg.norm(np.asarray(g[0]["referencePixel"])-p["referencePixel"])<frames[reference].shape[1]*.05),None)
            if g is None:groups.append([p])
            else:g.append(p)
        confirmed=[{"number":g[0]["number"],"row":g[0]["row"],
            "pixel":np.median([p["referencePixel"] for p in g],axis=0).tolist()} for g in groups if len({p["frame"] for p in g})>=2]
        candidate=fit(frames[reference],confirmed,level)
        if candidate:
            score,h,spec,pairs,turf=candidate
            result=AutoRegistration(homography=h,spec=spec,correspondences=pairs,reading=SimpleNamespace(level=level),
                verdict="good",paint=score,turf=turf,notes=["automatic OCR on both number rows, confirmed across frames and checked against paint"])
            result.frame=reference;result.evidence={"method":"temporal word OCR","numbers":evidence,"referenceSourceFrame":reference}
            return result
    return None
