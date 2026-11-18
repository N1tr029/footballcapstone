"""Conservative airborne football sightings from colour and temporal motion.

Brown pixels alone are not evidence: skin, helmets and broadcast graphics share
that colour. Candidates must be isolated over turf, outside person boxes, and
form a sustained moving trajectory after camera compensation. Missing hands and
occlusions are deliberately left unknown. This detector targets visible flight,
not possession, and has only been visually checked on the SF/SEA broadcast clip.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy.optimize import linear_sum_assignment


@dataclass
class Sighting:
    frame: int
    pixel: tuple[float, float]
    field: tuple[float, float]
    bbox: tuple[int, int, int, int]
    confidence: float

    def to_json(self):
        return asdict(self)


def candidates(image, boxes, homography, frame):
    import cv2

    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(hsv, np.array([0, 55, 25]), np.array([24, 255, 200]))
    height, width = image.shape[:2]
    mask[:int(height * .1)] = 0
    mask[int(height * .77):] = 0  # score bug and lower-third graphics
    for box in boxes:
        x1, y1, x2, y2 = box
        cv2.rectangle(mask, (max(0, int(x1) - 3), max(0, int(y1) - 3)),
                      (int(x2) + 3, int(y2) + 3), 0, -1)
    _, _, stats, centers = cv2.connectedComponentsWithStats(mask)
    out = []
    scale = width / 1920
    for (x, y, w, h, area), pixel in zip(stats[1:], centers[1:]):
        if not (20 * scale ** 2 <= area <= 650 * scale ** 2
                and .5 <= w / h <= 3 and h >= 5 * scale and w >= 6 * scale):
            continue
        pad = max(3, round(8 * scale))
        ring = hsv[max(0, y-pad):y+h+pad, max(0, x-pad):x+w+pad]
        turf = ((ring[:, :, 0] >= 25) & (ring[:, :, 0] <= 95) & (ring[:, :, 1] >= 40))
        if turf.mean() < .45:
            continue
        # This is a ground-plane projection of a visible airborne point. Camera
        # height inferred from a monocular broadcast homography is unstable during
        # a fast pan; applying a guessed 1-yard correction invents large jumps.
        xy = np.asarray(homography.to_field([pixel])).reshape(-1, 2)[0]
        if not (np.isfinite(xy).all() and 0 <= xy[0] <= 120 and 0 <= xy[1] <= 53.4):
            continue
        out.append(Sighting(frame, tuple(map(float, pixel)), tuple(map(float, xy)),
                            (int(x), int(y), int(x+w), int(y+h)), float(turf.mean())))
    return out


def link(candidates_by_frame, fps, min_sightings=6):
    """Keep sustained flights, rejecting stationary stains and single-frame noise."""
    tracks = []
    for frame, observations in sorted(candidates_by_frame.items()):
        active = [t for t in tracks if frame - t[-1].frame <= 2]
        cost = np.full((len(active), len(observations)), 1e6)
        for i, history in enumerate(active):
            last = history[-1]
            dt = (frame - last.frame) / fps
            velocity = np.zeros(2)
            if len(history) >= 2:
                prev = history[-2]
                velocity = ((np.array(last.field) - prev.field)
                            / ((last.frame - prev.frame) / fps))
            predicted = np.array(last.field) + velocity * dt
            for j, point in enumerate(observations):
                displacement = np.linalg.norm(np.array(point.field) - last.field)
                residual = np.linalg.norm(np.array(point.field) - predicted)
                old_area = (last.bbox[2]-last.bbox[0]) * (last.bbox[3]-last.bbox[1])
                area = (point.bbox[2]-point.bbox[0]) * (point.bbox[3]-point.bbox[1])
                prediction_gate = 40 * dt + .35 if len(history) < 2 else max(.8, 25 * dt)
                if displacement <= 40 * dt + .35 and residual <= prediction_gate and .35 < area/old_area < 2.8:
                    cost[i, j] = residual
        used = set()
        for i, j in zip(*linear_sum_assignment(cost)):
            if cost[i, j] < 1e6:
                active[i].append(observations[j])
                used.add(j)
        tracks.extend([point] for j, point in enumerate(observations) if j not in used)
    flights = []
    for history in tracks:
        if len(history) < min_sightings:
            continue
        xy = np.array([p.field for p in history])
        span = np.linalg.norm(xy[-1] - xy[0])
        duration = (history[-1].frame - history[0].frame) / fps
        if span < 3 or not 10 <= span / duration <= 40:
            continue
        # A sustained direction is necessary; a stain jittering with registration
        # error cannot qualify by accumulating a long zig-zag distance.
        length = np.linalg.norm(np.diff(xy, axis=0), axis=1).sum()
        if span / max(length, 1e-6) < .85:
            continue
        flights.append(history)
    return flights


def detect(frames, boxes_by_frame, registration, fps):
    pool = {}
    for frame, image in frames.items():
        homography = registration.at(frame)
        if homography is not None:
            pool[frame] = candidates(image, boxes_by_frame.get(frame, []), homography, frame)
    return link(pool, fps)


def write_preview(path, frames, observations, registration, fps, slow_factor=.2):
    """A slow-motion source-video proof, with a magnified crop of each sighting."""
    import cv2
    from pathlib import Path
    from .verify import draw_ball_sighting

    path = Path(path)
    if not observations:
        return None
    first = max(min(frames), min(observations) - round(.4 * fps))
    last = min(max(frames), max(observations) + round(.4 * fps))
    writer = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), 15, (1280, 720))
    try:
        for i in range(round((last-first+1) / fps / slow_factor * 15)):
            source = min(last, first + round(i * fps * slow_factor / 15))
            if source not in frames:
                continue
            original = frames[source]
            image = original.copy()
            seen = draw_ball_sighting(image, source, observations, registration.at(source), fps)
            image = cv2.resize(image, (1280, 720))
            cv2.rectangle(image, (0, 0), (1280, 91), (20, 28, 20), -1)
            cv2.putText(image, "BALL TRACKING  /  0.2x speed", (25, 35), 0, .9, (240, 245, 240), 2, cv2.LINE_AA)
            caption = "Visible football: pixel measurement" if seen else "Ball occluded / no direct measurement"
            cv2.putText(image, caption, (25, 69), 0, .65, (90, 255, 140) if seen else (200, 210, 200), 1, cv2.LINE_AA)
            cv2.putText(image, f"Source frame {source}", (1000, 35), 0, .6, (220, 230, 220), 1, cv2.LINE_AA)
            if seen:
                x, y = map(round, observations[source]["pixel"])
                crop = original[max(0, y-65):y+65, max(0, x-100):x+100].copy()
                crop = cv2.resize(crop, (400, 260))
                cv2.rectangle(image, (851, 441), (1256, 706), (90, 255, 140), 2)
                image[444:704, 854:1254] = crop
                cv2.putText(image, "ACTUAL PIXELS / 2x", (862, 470), 0, .55, (90, 255, 140), 1, cv2.LINE_AA)
            writer.write(image)
    finally:
        writer.release()
    return path
