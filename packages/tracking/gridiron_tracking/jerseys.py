"""Optional offline jersey OCR with temporal consensus and abstention.

Numbers belong to a tracked athlete only after independent source frames agree.
A single confident digit from a folded jersey is not an identity observation.
"""
from __future__ import annotations

from collections import defaultdict
import csv
import io
import shutil
import subprocess
import sys
import json
import base64
import hashlib
from pathlib import Path


def consensus(readings, minimum_frames=3):
    votes=defaultdict(dict)
    for reading in readings:
        number=int(reading["number"]); confidence=float(reading["confidence"])
        if not 0<=number<=99 or confidence<.65:
            continue
        frame=int(reading["frame"])
        votes[number][frame]=max(confidence,votes[number].get(frame,0))
    scores=sorted(((sum(v.values()),n,len(v)) for n,v in votes.items()),reverse=True)
    if not scores:
        return None
    score,number,count=scores[0];other=sum(s for s,_,_ in scores[1:])
    if count<minimum_frames or score/(score+other)<.8:
        return None
    return {"number":number,"confidence":round(score/(score+other),3),"frames":count}


def native_binary(cache_dir):
    if sys.platform!="darwin" or not shutil.which("swiftc"):
        return None
    source=Path(__file__).with_name("ocr")/"JerseyOCR.swift"
    key=hashlib.sha256(source.read_bytes()).hexdigest()[:12]
    cache=Path(cache_dir);cache.mkdir(parents=True,exist_ok=True)
    binary=cache/f"jersey-ocr-{key}"
    if not binary.is_file():
        try:
            result=subprocess.run([shutil.which("swiftc"),"-O","-module-cache-path",str(cache/"swift-modules"),
                str(source),"-o",str(binary)],capture_output=True,timeout=90,check=False)
        except (OSError,subprocess.TimeoutExpired):
            return None
        if result.returncode:
            return None
    return binary


def read_crop(crop, executable=None, native=None):
    import cv2
    import numpy as np

    executable=executable or shutil.which("tesseract")
    if (not executable and not native) or crop.size==0:
        return []
    crop=cv2.resize(crop,None,fx=max(1.,160/crop.shape[0]),fy=max(1.,160/crop.shape[0]))
    gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
    hsv=cv2.cvtColor(crop,cv2.COLOR_BGR2HSV)
    red=((hsv[:,:,0]<14)|(hsv[:,:,0]>166))&(hsv[:,:,1]>90)&(hsv[:,:,2]>65)
    variants=[crop if native else gray]
    if .015<red.mean()<.35:
        variants.append(np.where(red,0,255).astype(np.uint8))
    readings=[]
    for variant in variants:
        if native:
            ok,buffer=cv2.imencode(".png",variant)
            if ok:
                result=subprocess.run([str(native)],input=json.dumps({"png":base64.b64encode(buffer).decode()}).encode(),
                    capture_output=True,timeout=10,check=False)
                if result.returncode==0:
                    try:
                        readings.extend(json.loads(result.stdout))
                    except (ValueError,TypeError):
                        pass
            continue
        image=cv2.copyMakeBorder(variant,15,15,15,15,cv2.BORDER_CONSTANT,value=255)
        ok,buffer=cv2.imencode(".png",image)
        if not ok:
            continue
        result=subprocess.run([executable,"stdin","stdout","--psm","7","-c",
            "tessedit_char_whitelist=0123456789","tsv"],input=buffer.tobytes(),
            capture_output=True,timeout=8,check=False)
        for row in csv.DictReader(io.StringIO(result.stdout.decode(errors="replace")),delimiter="\t"):
            text=(row.get("text") or "").strip()
            if text.isdigit() and 1<=len(text)<=2:
                confidence=float(row["conf"])/100
                if confidence>=.65:
                    readings.append({"number":int(text),"confidence":confidence})
    return readings


def annotate(track,frames,boxes_by_frame,*,fps,max_crops=8,cache_dir=None,log=lambda *a:None):
    """Read clear crops anywhere in the play, then label the complete identity.

    This deliberately uses later frames too. It never changes identity associations
    merely because two OCR guesses happen to share a number.
    """
    import cv2
    import numpy as np
    executable=shutil.which("tesseract")
    native=native_binary(cache_dir) if cache_dir is not None else None
    if not executable and not native:
        track.meta["jerseys"]={"available":False,"reason":"optional tesseract executable unavailable"}
        return {}
    candidates=defaultdict(list)
    for frame,boxes in boxes_by_frame.items():
        frame=int(frame)
        if frame not in frames:
            continue
        image=frames[frame];height,width=image.shape[:2]
        for pid,box in boxes.items():
            x1,y1,x2,y2=map(float,box);w=x2-x1;h=y2-y1
            if h<65 or w<25 or x1<0 or y1<0 or x2>=width or y2>=height:
                continue
            xa,xb=int(x1+.1*w),int(x2-.1*w)
            ya,yb=int(y1+.18*h),int(y1+.6*h)
            crop=image[ya:yb,xa:xb]
            if crop.size==0:
                continue
            gray=cv2.cvtColor(crop,cv2.COLOR_BGR2GRAY)
            sharp=float(cv2.Laplacian(gray,cv2.CV_64F).var())
            candidates[pid].append((sharp*np.sqrt(w*h),frame,crop))
    observations={};accepted={}
    for pid,choices in candidates.items():
        # A sharp but unreadable pre-snap pose must not consume every sample.
        # Reserve crops across the entire observed lifetime, including later backs.
        first=min(f for _,f,_ in choices);last=max(f for _,f,_ in choices)
        bins={}
        for choice in choices:
            bucket=min(max_crops-1,int((choice[1]-first)/max(1,last-first+1)*max_crops))
            if bucket not in bins or choice[0]>bins[bucket][0]:
                bins[bucket]=choice
        diverse=sorted(bins.values(),key=lambda r:r[0],reverse=True)
        remaining=sorted(choices,key=lambda r:r[0],reverse=True)
        selected=[]
        for _,frame,crop in diverse+remaining:
            if any(abs(frame-f)<fps*.35 for f in selected):
                continue
            selected.append(frame)
            try:
                readings=read_crop(crop,executable,native)
            except (OSError,subprocess.TimeoutExpired,ValueError):
                readings=[]
            observations.setdefault(pid,[]).extend({**r,"frame":frame} for r in readings)
            if len(selected)>=max_crops:
                break
        result=consensus(observations.get(pid,[]))
        if result:
            accepted[pid]=result
    # Duplicate same-team claims are evidence of OCR/tracking ambiguity, not an
    # excuse to assign a made-up alternate jersey number to one of the players.
    claims=defaultdict(list)
    for pid,result in accepted.items():
        claims[track.roster[pid].team,result["number"]].append(pid)
    for pids in claims.values():
        if len(pids)>1:
            for pid in pids:
                accepted.pop(pid,None)
    for pid,result in accepted.items():
        track.roster[pid].num=result["number"]
    track.meta["jerseys"]={"available":True,"method":("Apple Vision" if native else "Tesseract")+"; agreement across at least three independent frames",
        "accepted":accepted,"observations":observations,"note":"OCR estimates; conflicting or unreadable jerseys remain unknown."}
    log(f"    jerseys: {len(accepted)}/{len(track.roster)} agreed across multiple frames")
    return accepted
