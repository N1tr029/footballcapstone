"""Check formation crops for players, officials and sideline personnel locally."""
from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path


def classify(image,detections,*,cache_dir,model=None):
    import cv2
    import numpy as np
    from .registration.local import DEFAULT_MODEL,generate_json
    model=(model or DEFAULT_MODEL).removeprefix("mlx:")
    boxes=[[d.x1,d.y1,d.x2,d.y2] for d in detections]
    key=hashlib.sha256(json.dumps({"version":1,"boxes":boxes,"model":model}).encode()+cv2.resize(image,(160,90)).tobytes()).hexdigest()[:16]
    cache=Path(cache_dir);cache.mkdir(parents=True,exist_ok=True);target=cache/f"roles-{key}.json"
    if target.exists():return json.loads(target.read_text())
    width,height=160,224;columns=6;rows=math.ceil(len(boxes)/columns)
    sheet=np.full((rows*height,columns*width,3),35,np.uint8)
    for i,box in enumerate(boxes):
        x1,y1,x2,y2=map(int,box);crop=image[max(0,y1):min(image.shape[0],y2),max(0,x1):min(image.shape[1],x2)]
        if not crop.size:continue
        scale=min((width-8)/crop.shape[1],(height-28)/crop.shape[0]);crop=cv2.resize(crop,None,fx=scale,fy=scale)
        xa=(i%columns)*width+(width-crop.shape[1])//2;ya=(i//columns)*height+24
        sheet[ya:ya+crop.shape[0],xa:xa+crop.shape[1]]=crop
        cv2.putText(sheet,str(i),((i%columns)*width+8,(i//columns)*height+18),cv2.FONT_HERSHEY_SIMPLEX,.6,(255,255,255),1,cv2.LINE_AA)
    cv2.imwrite(str(target.with_suffix('.jpg')),sheet)
    schema={"type":"object","properties":{"roles":{"type":"array","minItems":len(boxes),"maxItems":len(boxes),
        "items":{"type":"object","properties":{"id":{"type":"integer","minimum":0,"maximum":len(boxes)-1},
        "role":{"type":"string","enum":["player","official","other","unclear"]}},"required":["id","role"],"additionalProperties":False}}},"required":["roles"],"additionalProperties":False}
    text,_=generate_json(sheet,"Identify uniforms in American football person crops. Return only the requested JSON.",
        "Each numbered tile shows one detected person. Classify every tile by its number. A player wears a helmet and shoulder pads; "
        "white numbers on a colored jersey still indicate a player. An official wears a black-and-white vertically striped shirt "
        "and a cap, without shoulder pads or a football helmet. Coaches and spectators are other. Use unclear if genuinely hidden. "
        "Do not classify a crouched or partly occluded helmeted player as an official.",schema,model=model,max_tokens=1200)
    answer=json.loads(text);roles={int(r['id']):r['role'] for r in answer['roles']}
    if set(roles)!=set(range(len(boxes))):raise ValueError("formation role reader omitted or duplicated a crop")
    result={"roles":[roles[i] for i in range(len(boxes))],"model":model,"source":"local uniform-role check"}
    target.write_text(json.dumps(result));return result


def refine(image,detections,registration,frame,*,cache_dir,detector_spec=None,model=None,capacity=11,field_cuts=None,log=lambda *a:None):
    """Exclude officials before using a head count as a football roster.

    If people overlap at the selected formation, tile this one frame to look for
    missed athletes. Already-classified crops are reused by box overlap.
    """
    import numpy as np
    from .auto_identity import iou
    log(f"    role check: reading uniforms for {len(detections)} formation crops")
    result=classify(image,detections,cache_dir=cache_dir,model=model)
    roles=result["roles"];initial=list(detections)
    people=[d for d,role in zip(initial,roles) if role in ("player","unclear")]
    unclear=sum(role=="unclear" for role in roles)
    metadata={"available":True,"method":"local uniform roles before roster seeding",
        "excludedOfficials":sum(role=="official" for role in roles),
        "excludedOther":sum(role=="other" for role in roles),"initialPeople":len(initial),"tiled":False}
    if detector_spec is not None and len(people)<2*capacity:
        from .detect.yolo import YoloDetector
        detector=YoloDetector(weights=detector_spec.weights,conf=detector_spec.conf,imgsz=detector_spec.imgsz,
            device=detector_spec.device,tiles=2,tile_overlap=.2,merge_iou=.45)
        candidates=detector.detect(frame,image)
        points=registration.to_field(frame,[d.anchor()[0] for d in candidates]) if candidates else []
        lo,hi=field_cuts if field_cuts is not None else (-1,registration.spec.width+1)
        # The bottom of a truncated torso/head is not a foot position. Projecting
        # those crops at ground height moves near-side spectators onto the turf.
        candidates=[d for d,p in zip(candidates,points) if np.isfinite(p).all() and
            -1<=p[0]<=121 and lo<=p[1]<=hi and d.y2<image.shape[0]-2]
        mapped=[];unread=[]
        for i,d in enumerate(candidates):
            b=[d.x1,d.y1,d.x2,d.y2]
            overlap=[iou(b,[old.x1,old.y1,old.x2,old.y2]) for old in initial]
            if overlap and max(overlap)>=.55:mapped.append(roles[int(np.argmax(overlap))])
            else:mapped.append(None);unread.append(i)
        if unread:
            log(f"    role check: reading {len(unread)} additional on-field crops")
            fresh=classify(image,[candidates[i] for i in unread],cache_dir=cache_dir,model=model)["roles"]
            for i,role in zip(unread,fresh):mapped[i]=role
        # An uncertain new crop cannot establish another athlete. Existing
        # uncertain identities can remain; adding one needs visible player kit.
        verified=[d for d,role in zip(candidates,mapped) if role=="player" or
            (role=="unclear" and d in initial)]
        if len(people)<=len(verified)<=2*capacity:
            people=verified;unclear=sum(role=="unclear" and d in verified for d,role in zip(candidates,mapped))
            metadata.update(excludedOfficials=sum(role=="official" for role in mapped),
                excludedOther=sum(role=="other" for role in mapped))
        metadata.update(tiled=True,tiledPeople=len(candidates),additionalCropsRead=len(unread),fieldCuts=[lo,hi],
            rejectedTiledOverCapacity=max(0,len(verified)-2*capacity))
    metadata["verifiedPlayers"]=len(people)-unclear
    metadata["unclearAthletes"]=unclear
    metadata["overCapacity"]=max(0,len(people)-2*capacity)
    if len(people)>2*capacity:
        people=sorted(people,key=lambda d:d.score,reverse=True)[:2*capacity]
    log(f"    role check: {metadata}")
    return people,metadata
