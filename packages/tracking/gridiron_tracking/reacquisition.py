"""Reacquire one athlete later, then retrack only that athlete backward."""
from __future__ import annotations

from copy import deepcopy
import numpy as np

from .auto_identity import iou


def recovery_windows(accepted, masks, detections, fps, maximum=None):
    """Find a lost identity followed by an independently clear later body.

    A visible SAM channel and a unique detector body must agree on at least three
    later frames. A conflicting accepted box counts as losing that athlete too:
    retaining the ID on somebody else is not successful reacquisition.
    """
    frames=sorted(masks);ids={pid for people in masks.values() for pid in people};windows=[]
    for pid in sorted(ids):
        seam=None;lost=False;support=[];last_support=None;misses=0;reappeared=0
        def finish():
            if seam is not None and len(support)>=3 and support[-1][0]-seam>=fps*.3:
                # Prefer the latest clear sighting: it lets a single reverse pass
                # recover the entire uncertain interval, not just its first gap.
                stop,box,score=support[-1]
                windows.append({"playerId":pid,"start":seam,"stop":stop,
                    "box":box,"laterSupportingFrames":[f for f,_,_ in support],"supportIoU":score})
        for frame in frames:
            if lost and last_support is not None and frame-last_support>fps*.4:
                finish();seam=None;lost=False;support=[];last_support=None;misses=0;reappeared=0
            raw=masks[frame].get(pid);known=accepted.get(frame,{}).get(pid)
            bodies=detections.get(frame,[])
            if raw is None or not bodies:
                if seam is not None:
                    misses+=1
                    if misses>=3:lost=True
                reappeared=0
                continue
            boxes=[[d.x1,d.y1,d.x2,d.y2] for d in bodies]
            overlaps=np.array([iou(raw,box) for box in boxes]);j=int(np.argmax(overlaps))
            competing=max((iou(other,boxes[j]) for other_id,other in masks[frame].items() if other_id!=pid),default=0.)
            clear=overlaps[j]>=.65 and overlaps[j]-competing>=.2 and bodies[j].score>=.5
            agrees=known is not None and iou(raw,known)>=.4
            if agrees and clear and not lost:
                seam=frame;misses=0
            elif seam is not None and not agrees:
                misses+=1
                if misses>=3:lost=True
            elif agrees and not lost:
                misses=0
            if lost and clear:
                support.append((frame,boxes[j],float(overlaps[j])));last_support=frame
            if lost:
                reappeared=reappeared+1 if agrees and clear else 0
                if reappeared>=3:
                    finish();seam=frame;lost=False;support=[];last_support=None;misses=0;reappeared=0
        finish()
    # A player can disappear more than once. Repair each independently supported
    # interval in time order so a second loss is not silently left unexamined.
    windows.sort(key=lambda w:(w["start"],w["playerId"]))
    return windows if maximum is None else windows[:maximum]


def repair_one(accepted, reverse, pid, start, stop, minimum_iou=.35):
    """Require a backward trace to reconnect to this athlete's earlier identity."""
    earlier=accepted.get(start,{}).get(pid);traced=reverse.get(start,{}).get(pid)
    if earlier is None or traced is None or iou(earlier,traced)<minimum_iou:
        return {"accepted":False,"reason":"single-player reverse trace did not reconnect to the earlier identity"}
    own=iou(earlier,traced)
    competing=max((iou(box,traced) for other,box in accepted[start].items() if other!=pid),default=0.)
    if competing>own+.1:
        return {"accepted":False,"reason":"reverse trace fits another earlier identity better"}
    recovered=[];corrected=[];conflicts=[]
    for frame,people in reverse.items():
        if not start<frame<=stop or pid not in people:continue
        old=accepted.setdefault(frame,{}).get(pid)
        if any(iou(box,people[pid])>=.8 for other,box in accepted[frame].items() if other!=pid):
            conflicts.append(frame)
            continue
        if old is None:recovered.append(frame)
        elif iou(old,people[pid])<.3:corrected.append(frame)
        accepted[frame][pid]=deepcopy(people[pid])
    return {"accepted":True,"earlierSeamIoU":round(own,3),
        "recoveredFrames":sorted(recovered),"correctedFrames":sorted(corrected),
        "conflictedFrames":sorted(conflicts)}


def recover(accepted,masks,detections,frames,*,fps,weights,cache_dir,log=lambda *a:None,tracker=None):
    """Each call prompts exactly one player and runs only the affected interval."""
    if tracker is None:
        from .sam_video import track_boxes
        tracker=track_boxes
    results=[]
    for window in recovery_windows(accepted,masks,detections,fps):
        pid=window["playerId"];start=window["start"];stop=window["stop"]
        log(f"    reacquisition: tracing only {pid} backward from {stop} to {start}")
        reverse=tracker(frames,stop,{pid:window["box"]},fps=fps,weights=weights,
            cache_dir=cache_dir,reverse=True,stop=start,log=log)
        result=repair_one(accepted,reverse,pid,start,stop)
        # Once an earlier drift is established, don't resume its conflicting old
        # boxes after the repaired interval. Keep the actor as an explicit display
        # estimate until the identity agrees with a new observation again.
        if result["accepted"]:
            cleared=[]
            for frame in sorted(accepted):
                if frame<=stop or pid not in accepted[frame]:continue
                raw=masks.get(frame,{}).get(pid)
                if raw is not None and iou(raw,accepted[frame][pid])>=.4:break
                accepted[frame].pop(pid,None);cleared.append(frame)
            result["clearedConflictingFrames"]=cleared
        results.append({k:v for k,v in window.items() if k!="box"}|result)
        log(f"    reacquisition {pid}: {'reconnected' if result['accepted'] else result['reason']}")
    return results
