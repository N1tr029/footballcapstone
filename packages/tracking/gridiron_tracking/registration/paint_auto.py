"""Read unobstructed painted numerals across a shot, not a single chosen frame.

The field is a plane. Image-to-image registration lets a numeral revealed later
constrain an earlier view without any clicked field points.
"""
from __future__ import annotations

from collections import Counter,defaultdict
from types import SimpleNamespace
import subprocess
import numpy as np


def number_candidates(image,boxes=(),native=None):
    import cv2
    from .lines import find_yard_lines
    from ..jerseys import read_crop
    height,width=image.shape[:2]
    lines=find_yard_lines(image,min_length_px=width*.02,min_support=width*.25,max_lines=20)
    lines=[l for l in lines if width*.04<l.x_at(height*.25)<width*.96]
    lines.sort(key=lambda l:l.x_at(height*.25))
    if len(lines)<3:
        return [],[],lines
    pitch=float(np.median(np.diff([l.x_at(height*.25) for l in lines])))
    if pitch<width*.04:
        return [],[],lines
    hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    mask=cv2.inRange(hsv,(0,0,140),(179,80,255));mask[:int(height*.14)]=0;mask[int(height*.45):]=0
    for box in boxes:
        x1,y1,x2,y2=map(int,box)
        cv2.rectangle(mask,(max(0,x1-3),max(0,y1-3)),(x2+3,y2+3),0,-1)
    for line in lines:
        cv2.line(mask,(round(line.x_at(0)),0),(round(line.x_at(height)),height),0,max(3,round(width/240)))
    _,_,stats,_=cv2.connectedComponentsWithStats(mask)
    glyphs=[s for s in stats[1:] if width*.006<=s[2]<=width*.09 and height*.015<=s[3]<=height*.07 and s[4]>=width*.09]
    labels=[];rows=[]
    for index,line in enumerate(lines):
        near=[s for s in glyphs if abs(s[0]+s[2]/2-line.x_at(s[1]+s[3]/2))<pitch*.5]
        if len(near)<2:
            continue
        y=float(np.median([s[1]+s[3]/2 for s in near]));near=[s for s in near if abs(s[1]+s[3]/2-y)<height*.025]
        if not near or not any(s[0]+s[2]/2<line.x_at(y) for s in near) or not any(s[0]+s[2]/2>line.x_at(y) for s in near):
            continue
        x=line.x_at(y);rows.append([x,y])
        xa,xb=max(0,round(x-.62*pitch)),min(width,round(x+.62*pitch))
        ya,yb=max(0,round(y-height*.035)),min(height,round(y+height*.035))
        crops=[image[ya:yb,xa:xb]]
        txa=max(0,min(s[0] for s in near)-8);txb=min(width,max(s[0]+s[2] for s in near)+8)
        tya=max(0,min(s[1] for s in near)-8);tyb=min(height,max(s[1]+s[3] for s in near)+8)
        crops.append(image[tya:tyb,txa:txb]);votes=Counter();confidence={}
        for original in crops:
            crop=cv2.rotate(original,cv2.ROTATE_180)
            for stretch in (2,3,4,5):
                upright=cv2.resize(crop,None,fx=1,fy=stretch)
                try:
                    readings=read_crop(upright,native=native)
                except (OSError,subprocess.TimeoutExpired,ValueError):
                    readings=[]
                for reading in readings:
                    n=reading["number"]
                    if n in (10,20,30,40,50) and reading["confidence"]>=.85:
                        votes[n]+=1;confidence[n]=max(confidence.get(n,0),reading["confidence"])
        if votes and len(votes)==1:
            n=votes.most_common(1)[0][0]
            labels.append({"number":n,"pixel":[float(x),float(y)],"gridIndex":index,"confidence":confidence[n]})
    return labels,rows,lines


def far_sideline(image):
    import cv2
    height,width=image.shape[:2];hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV)
    paint=cv2.inRange(hsv,(0,0,140),(179,80,255));paint[int(height*.25):]=0
    turf=cv2.inRange(hsv,(25,40,35),(95,255,255))
    lines=cv2.HoughLinesP(paint,1,np.pi/1800,100,minLineLength=int(width*.2),maxLineGap=30)
    candidates=[]
    for x1,y1,x2,y2 in (lines.reshape(-1,4) if lines is not None else []):
        if abs(x2-x1)<50 or abs((y2-y1)/(x2-x1))>.2:
            continue
        slope=(y2-y1)/(x2-x1);intercept=y1-slope*x1
        xs=np.arange(30,width-30,5);ys=np.rint(xs*slope+intercept).astype(int)
        below=turf[np.clip(ys+15,0,height-1),xs].mean()/255
        above=turf[np.clip(ys-15,0,height-1),xs].mean()/255
        if below>.65:
            candidates.append((below-above+abs(x2-x1)/width*.3,slope,intercept))
    return max(candidates)[1:] if candidates else None


def image_registration(source,target,source_boxes=(),target_boxes=()):
    import cv2
    def describe(image,boxes):
        hsv=cv2.cvtColor(image,cv2.COLOR_BGR2HSV);mask=cv2.inRange(hsv,(25,20,30),(100,255,255))
        for box in boxes:
            x1,y1,x2,y2=map(int,box);cv2.rectangle(mask,(x1,y1),(x2,y2),0,-1)
        return cv2.ORB_create(nfeatures=5000).detectAndCompute(cv2.cvtColor(image,cv2.COLOR_BGR2GRAY),mask)
    a,da=describe(source,source_boxes);b,db=describe(target,target_boxes)
    if da is None or db is None:
        return None
    matches=cv2.BFMatcher(cv2.NORM_HAMMING).knnMatch(da,db,k=2)
    good=[m for pair in matches if len(pair)==2 for m,n in [pair] if m.distance<.7*n.distance]
    if len(good)<25:
        return None
    h,inliers=cv2.findHomography(np.float32([a[m.queryIdx].pt for m in good]),
        np.float32([b[m.trainIdx].pt for m in good]),cv2.RANSAC,3)
    return h if inliers is not None and inliers.sum()>=25 else None


def calibrate(frames,boxes,*,fps,cache_dir,level=None,log=lambda *a:None):
    """Use later readable labels to calibrate the same camera's earlier paint."""
    from ..jerseys import native_binary
    from ..field import NFL,NCAA,HIGH_SCHOOL
    from .homography import Homography,transform
    from .vision import AutoRegistration,paint_alignment,turf_overlap,sanity_problems,snap_to_paint

    native=native_binary(cache_dir);order=sorted(frames);reference=None;reference_lines=[];row_points=[]
    votes=defaultdict(list);evidence=[];tried=0
    for frame in order[::max(1,round(fps/5))]:
        labels,rows,lines=number_candidates(frames[frame],boxes.get(frame,[]),native)
        tried+=1
        if reference is None:
            if len(rows)<2 or len(lines)<3 or far_sideline(frames[frame]) is None:
                continue
            reference=frame;reference_lines=lines;row_points=list(rows)
        mapping=np.eye(3) if frame==reference else image_registration(frames[frame],frames[reference],
            boxes.get(frame,[]),boxes.get(reference,[]))
        if mapping is None:
            continue
        pitch=float(np.median(np.diff([l.x_at(frames[reference].shape[0]*.25) for l in reference_lines])))
        for label in labels:
            pixel=transform(mapping,[label["pixel"]])[0]
            index=min(range(len(reference_lines)),key=lambda i:abs(reference_lines[i].x_at(pixel[1])-pixel[0]))
            if abs(reference_lines[index].x_at(pixel[1])-pixel[0])>pitch*.3:
                continue
            votes[index].append((label["number"],frame));evidence.append({**label,"frame":frame,"referenceGridIndex":index})
        agreed={}
        for index,observed in votes.items():
            count=Counter(n for n,_ in observed);number,support=count.most_common(1)[0]
            if support/sum(count.values())>=.8:
                agreed[index]=number
        if len(set(agreed.values()))<2 or len({f for seen in votes.values() for _,f in seen})<2:
            continue
        options=[]
        for index,number in agreed.items():
            for x in (10+number,110-number):
                base=x-5*index
                matching=[i for i,n in agreed.items() if abs(50-abs((base+5*i)-60)-n)<.01]
                if len(matching)>=2 and len({agreed[i] for i in matching})>=2:
                    options.append((len(matching),base))
        if not options:
            continue
        _,base=max(options)
        if any(not 10<=base+5*i<=110 for i in range(len(reference_lines))):
            continue
        sideline=far_sideline(frames[reference]);row=np.polyfit(np.array(row_points)[:,0],np.array(row_points)[:,1],1)
        results=[]
        specs=[NFL,NCAA,HIGH_SCHOOL]
        if level in ("nfl","college","high_school"):
            specs=[{"nfl":NFL,"college":NCAA,"high_school":HIGH_SCHOOL}[level]]
        for spec in specs:
            pairs=[]
            for index,line in enumerate(reference_lines):
                slope=(line.x_at_bottom-line.x_at_top)/(line.y_bottom-line.y_top)
                intercept=line.x_at_top-slope*line.y_top
                for (a,b),field_y in ((sideline,spec.width),(row,spec.width-spec.number_inset-1)):
                    x=(slope*b+intercept)/(1-slope*a);y=a*x+b
                    pairs.append(((base+5*index,field_y),(x,y)))
            h=Homography.fit(np.array([a for a,_ in pairs]),np.array([b for _,b in pairs]),threshold_px=8)
            problems=sanity_problems(h,(frames[reference].shape[1],frames[reference].shape[0]),spec,correspondences=pairs)
            if problems:
                continue
            h,score=snap_to_paint(frames[reference],h,spec)
            turf=turf_overlap(frames[reference],h,spec)
            if np.isfinite(score) and score>=1.6 and turf>=.75:
                results.append((score,spec,h,pairs,turf))
        if results:
            score,spec,h,pairs,turf=max(results,key=lambda r:r[0])
            result=AutoRegistration(homography=h,spec=spec,correspondences=pairs,
                reading=SimpleNamespace(level={"NFL":"nfl","NCAA":"college","NFHS":"high_school"}[spec.name]),
                verdict="good",paint=score,turf=turf,
                notes=["automatic temporal numeral OCR and measured paint intersections; no clicked calibration points"])
            result.frame=reference
            result.evidence={"numbers":evidence,"referenceSourceFrame":reference,"framesExamined":tried}
            log(f"    automatic paint calibration: source {reference}, {len(evidence)} numeral readings, paint {score:.2f}")
            return result
        if tried>=45:
            break
    return None
