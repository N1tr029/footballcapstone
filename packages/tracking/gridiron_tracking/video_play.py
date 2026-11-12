"""Video estimates to the same Hindsight play contract as the replay API.

Reviewed annotations are an assisted path, never represented as ball detections.
Unreviewed sparse tracks remain available for replay with explicit model blockers.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
import math

from .contract import Ball, PlayTrack


def apply_review(track: PlayTrack, annotation: dict) -> None:
    """Validate a film-reviewed completed pass and attribute its possession phases.

    Times are original video frame indices. Positions come only from the existing
    player tracks; no missing player is placed at a made-up location. The airborne
    ground path is an interpolation, not a 3D reconstruction of the visible ball.
    """
    if track.meta.get("reviewed_pass") and all(track.meta["reviewed_pass"].get(k) == annotation.get(k)
            for k in ("throwFrame","catchFrame","passerId","receiverId","centerId")):
        return
    fps = float(track.meta.get("fps_in") or 0)
    clip = track.meta.get("clip", {})
    snap = clip.get("snap_source_frame")
    if fps <= 0 or snap is None or not track.frames:
        raise ValueError("review requires source FPS, an identified snap and tracking frames")
    throw, catch = int(annotation["throwFrame"]), int(annotation["catchFrame"])
    if not snap < throw < catch:
        raise ValueError("review requires snapFrame < throwFrame < catchFrame")
    if not clip.get("start_frame", snap) <= throw < catch < clip.get("stop_frame", catch+1):
        raise ValueError("reviewed events must be inside the clip window")
    passer, receiver = annotation["passerId"], annotation["receiverId"]
    center = annotation.get("centerId")
    if passer == receiver or any(pid not in track.roster for pid in (passer, receiver)):
        raise ValueError("passer and receiver must be distinct existing tracking IDs")
    offense = track.roster[passer].team
    if track.roster[receiver].team != offense or (center and (
            center not in track.roster or track.roster[center].team != offense)):
        raise ValueError("reviewed carriers must belong to the same offensive team")
    tf = track.frame_at((throw-snap)/fps)
    cf = track.frame_at((catch-snap)/fps)
    if tf.t >= cf.t or abs(tf.t-(throw-snap)/fps) > .051 or abs(cf.t-(catch-snap)/fps) > .051:
        raise ValueError("reviewed throw/catch must resolve to distinct exported frames")
    if passer not in tf.players or receiver not in cf.players:
        raise ValueError("passer at release and receiver at catch must have observed positions; correct identities first")
    start, end = tf.players[passer], cf.players[receiver]
    raw_ball = deepcopy(track.meta.get("ball", {}))
    # Store original sparse ball positions as evidence before replacing the ground
    # path; a projected airborne pixel is not the ball's point on the ground.
    evidence = raw_ball.get("raw_projected_observations") if track.meta.get("reviewed_pass") else None
    if evidence is None:
        evidence = {str(i+1):f.ball.to_json() for i,f in enumerate(track.frames) if f.ball}
    carriers = {}
    for i,f in enumerate(track.frames):
        f.events = [e for e in f.events if e not in ("pass_forward", "pass_arrived", "pass_outcome_caught")]
        f.ball = None
        if f.t <= tf.t:
            if f.t <= 0 and not center:
                continue
            pid = center if center and f.t <= 0 else passer
            p = f.players.get(pid)
            if p:
                f.ball = Ball(x=p.x,y=p.y,source="carrier")
                carriers[str(i+1)] = pid
        elif f.t >= cf.t:
            p = f.players.get(receiver)
            if p:
                f.ball = Ball(x=p.x,y=p.y,source="carrier")
                carriers[str(i+1)] = receiver
        else:
            alpha = (f.t-tf.t)/(cf.t-tf.t)
            f.ball = Ball(x=start.x+(end.x-start.x)*alpha,
                          y=start.y+(end.y-start.y)*alpha,source="interpolated")
    tf.events.append("pass_forward")
    cf.events.extend(["pass_arrived", "pass_outcome_caught"])
    track.meta["reviewed_pass"] = {**annotation, "source":"operator-reviewed from film",
        "throwTime":tf.t,"catchTime":cf.t,
        "note":"Carrier positions and interpolated airborne ground path are estimates, not ball detections."}
    track.meta["possessionTeam"] = offense
    track.meta["passResult"] = "C"
    sources = Counter(f.ball.source for f in track.frames if f.ball)
    positioned = [f.ball for f in track.frames if f.ball]
    span = math.hypot(max(b.x for b in positioned)-min(b.x for b in positioned),
                      max(b.y for b in positioned)-min(b.y for b in positioned))
    track.meta["ball"] = {**raw_ball,"by_source":dict(sources),"measured_fraction":0.,
        "travelled_yards":span,"coverage":len(positioned)/len(track.frames),
        "raw_projected_observations":evidence,
        "raw_measured_fraction":raw_ball.get("raw_measured_fraction",raw_ball.get("measured_fraction",0)),
        "carriers":carriers,
        "ground_path_source":"reviewed possession and interpolated flight"}


def prepared(raw: PlayTrack, options: dict | None = None, annotation: dict | None = None):
    from .prepare import prepare
    out = deepcopy(raw)
    if not out.frames:
        raise ValueError("no tracking frames; rerun this clip")
    out.meta["bdb"] = {**out.meta.get("bdb", {}), **(options or {})}
    config = out.meta["bdb"]
    if annotation:
        apply_review(out, annotation)
    if config.get("playDirection"):
        out.meta.setdefault("play_direction", config["playDirection"])
    out, report = prepare(out, offense=out.meta.get("possessionTeam"))
    if len(raw.roster) > 22 and raw.meta.get("identity",{}).get("anchorSourceFrame") is None:
        report.blockers.append("fragmented video identities: select a checked fullRosterFrame and rerun before model analysis")
    reviewed = out.meta.get("reviewed_pass")
    if reviewed:
        for key,event in (("passerId","pass_forward"),("receiverId","pass_arrived")):
            pid = reviewed[key]
            t = out.event_time(event)
            f = out.frame_at(t) if t is not None else None
            if pid not in out.roster or f is None or pid not in f.players or f.ball is None:
                report.blockers.append(f"reviewed {key} has no retained pose/ball at {event}; correct identities and rerun")
    report.reviewable = not report.blockers
    out.meta["prepare_report"] = report.to_json()
    clubs = config.get("teamClubs", {})
    for entry in out.roster.values():
        entry.team = clubs.get(entry.team,entry.team)
    for key in ("possessionTeam", "defensiveTeam"):
        if out.meta.get(key):
            out.meta[key] = clubs.get(out.meta[key],out.meta[key])
    # Stable local IDs use the existing BDB allocator and remain explicitly synthetic.
    from .bdb import convert
    _, identifiers = convert(raw, config)
    out.meta["game_id"] = identifiers["gameId"]
    out.meta["play_id"] = identifiers["playId"]
    out.meta["video_identifiers"] = {k:identifiers[k] for k in ("syntheticGameId","syntheticPlayId","playerIdMap")}
    quality = {"rosterSize":len(out.roster),"perTeam":dict(Counter(e.team for e in out.roster.values())),
        "meanObservedPlayers":round(sum(len(f.players) for f in out.frames)/len(out.frames),2),
        "ballFrames":sum(f.ball is not None for f in out.frames),"totalFrames":len(out.frames),
        "ballSources":dict(Counter(f.ball.source or "unverified" for f in out.frames if f.ball)),
        "assisted":bool(out.meta.get("reviewed_pass") or out.meta.get("identity",{}).get("operatorAnchors")),"identity":out.meta.get("identity",{}).get("method"),
        "verification":out.meta.get("verification",{}),
        "notes":["Same-kit crossings may switch identities; roster size does not establish identity accuracy.",
                 "Unobserved players are absent. Separation estimates may miss off-camera defenders.",
                 "Unidentified positions can cause linemen to be treated as possible receivers."]}
    for event,label in (("pass_forward","Throw"),("pass_arrived","Arrival")):
        t = out.event_time(event)
        if t is not None:
            f = out.frame_at(t)
            quality[f"missingAt{label}"] = [pid for pid in out.roster if pid not in f.players]
    out.meta["tracking_quality"] = quality
    from .hindsight import nfl_ids
    out.meta["export_nfl_ids"] = nfl_ids(out)
    return out, report


def play_json(raw: PlayTrack, options: dict | None = None, annotation: dict | None = None) -> dict:
    out, report = prepared(raw, options, annotation)
    return document(out, report)


def document(out, report):
    from .hindsight import to_play_json
    doc = to_play_json(out)
    doc.update(analysisReady=report.reviewable,analysisBlockers=report.blockers,
               trackingQuality=out.meta["tracking_quality"])
    doc["provenance"]["reviewedPass"] = out.meta.get("reviewed_pass")
    doc["provenance"]["identifiers"] = out.meta.get("video_identifiers",{})
    doc["provenance"]["ball"] = out.meta.get("ball",{})
    doc["provenance"]["identity"] = {k:v for k,v in out.meta.get("identity",{}).items() if k != "sourceBoxes"}
    return doc
