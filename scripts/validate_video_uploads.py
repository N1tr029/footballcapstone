"""Exercise real automatic upload jobs and audit replay/recovery evidence.

Run against a configured local API. No model calls are mocked and no calibration,
snap, player identities or pass annotations are supplied to the service.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import time

import httpx


def replay_checks(doc):
    roster={p["nflId"]:p for p in doc["players"]};expected=set(roster)
    issues=[];observed=estimated=0;last={};largest_step=0.
    if not roster or not doc["frames"]:issues.append("empty tracking output")
    if len(roster)>22 or any(n>11 for club,n in Counter(p["club"] for p in roster.values()).items() if club!="unknown"):
        issues.append("roster exceeds football team capacity")
    role=doc.get("trackingQuality",{}).get("roleCheck")
    if role and (not role.get("available") or role.get("overCapacity")):
        issues.append("athlete/official roster verification failed")
    for frame in doc["frames"]:
        people=frame["players"];ids=[p["nflId"] for p in people]
        if set(ids)!=expected or len(ids)!=len(expected):
            issues.append(f"frame {frame['frameId']}: roster membership changed")
        for p in people:
            pid=p["nflId"]
            if pid not in roster:continue
            if not all(isinstance(p.get(k),(int,float)) and math.isfinite(p[k]) for k in ("x","y")):
                issues.append(f"frame {frame['frameId']}: nonfinite position for {pid}")
                continue
            if p.get("club",roster[pid]["club"])!=roster[pid]["club"]:
                issues.append(f"frame {frame['frameId']}: team changed for {pid}")
            observed+=bool(p.get("observed"));estimated+=not p.get("observed",False)
            if pid in last:
                step=math.hypot(p["x"]-last[pid][0],p["y"]-last[pid][1]);largest_step=max(largest_step,step)
                if step>12/doc["frameRate"]+.04:
                    issues.append(f"frame {frame['frameId']}: display jump for {pid}")
            last[pid]=(p["x"],p["y"])
    checks=doc.get("provenance",{}).get("identity",{}).get("reacquisitionChecks",[])
    return {"rosterSize":len(roster),"perTeam":dict(Counter(p["club"] for p in roster.values())),
        "frames":len(doc["frames"]),"observedSamples":observed,"estimatedSamples":estimated,
        "maximumStepYards":round(largest_step,3),"issues":sorted(set(issues)),
        "individualBackwardChecks":len(checks),"acceptedBackwardChecks":sum(c["accepted"] for c in checks),
        "recoveredSourceSamples":sum(len(c.get("recoveredFrames",[])) for c in checks),
        "correctedSourceSamples":sum(len(c.get("correctedFrames",[])) for c in checks),
        "analysisReady":doc.get("analysisReady"),"analysisBlockers":doc.get("analysisBlockers",[]),
        "quality":doc.get("trackingQuality",{})}


def failed(report):
    return bool(report.get("error") or report.get("state")!="done" or not report.get("plays") or
        any(not p.get("hasTrack") or p.get("issues") for p in report.get("plays",[])))


def run(client,origin,upload,out,*,timeout,poll):
    out.mkdir(parents=True,exist_ok=True);(out/"upload.json").write_text(json.dumps(upload,indent=2))
    started=time.monotonic();last=None
    while True:
        response=client.get(origin+"/video/jobs/"+upload["analysisJobId"]);response.raise_for_status()
        job=response.json();state=(job["state"],job.get("progress"))
        (out/"job.json").write_text(json.dumps(job,indent=2))
        if state!=last:
            print(out.name,state,flush=True);last=state
        if job["state"]!="running":break
        if time.monotonic()-started>timeout:
            return {"error":"validation wait timed out; the server job is still running", "jobId":upload["analysisJobId"]}
        time.sleep(poll)
    report={"upload":upload,"state":job["state"],"error":job.get("error"),"plays":[]}
    for play in (job.get("result") or {}).get("plays",[]):
        row={"clip":play["clip"],"hasTrack":play.get("hasTrack"),"problems":play.get("problems",[])}
        asset=play.get("assets",{}).get("tracking")
        if asset:
            response=client.get(origin+asset);response.raise_for_status();doc=response.json()
            (out/(play["clip"]+".play.json")).write_text(json.dumps(doc,separators=(',',':')))
            row.update(replay_checks(doc))
            for name in ("reconstruction","ballPreview"):
                if url:=play.get("assets",{}).get(name):
                    with client.stream("GET",origin+url) as video:
                        video.raise_for_status()
                        with (out/(play["clip"]+"-"+name+".mp4")).open("wb") as dest:
                            for chunk in video.iter_bytes():dest.write(chunk)
        report["plays"].append(row)
    (out/"report.json").write_text(json.dumps(report,indent=2))
    return report


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("videos",nargs="*",type=Path)
    parser.add_argument("--base-url",default="http://127.0.0.1:8000")
    parser.add_argument("--output",type=Path,default=Path("data/runs/upload-validation/results"))
    parser.add_argument("--resume-upload",type=Path,help="Continue polling a saved upload response")
    parser.add_argument("--timeout",type=float,default=7200)
    parser.add_argument("--poll",type=float,default=10)
    args=parser.parse_args();origin=args.base_url.rstrip("/");reports=[]
    with httpx.Client(timeout=120) as client:
        if args.resume_upload:
            upload=json.loads(args.resume_upload.read_text())
            reports.append(run(client,origin,upload,args.output/Path(upload["filename"]).stem,
                timeout=args.timeout,poll=args.poll))
        for path in args.videos:
            with path.open("rb") as src:
                response=client.post(origin+"/video/uploads",files={"file":(path.name,src,"video/mp4")})
            response.raise_for_status();upload=response.json()
            upload["validationSourceSha256"]=hashlib.sha256(path.read_bytes()).hexdigest()
            reports.append(run(client,origin,upload,args.output/path.stem,timeout=args.timeout,poll=args.poll))
    args.output.mkdir(parents=True,exist_ok=True)
    (args.output/"summary.json").write_text(json.dumps(reports,indent=2))
    return 1 if any(failed(r) for r in reports) else 0


if __name__=="__main__":raise SystemExit(main())
