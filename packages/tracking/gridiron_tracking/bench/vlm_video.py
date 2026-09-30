"""Can a video model do the whole job? Measured, in yards.

The question this settles: hand a vision model the clip and ask for every player's
position over time — does what comes back resemble the play? It is worth settling
properly rather than by argument, because if the answer is yes the architecture of this
whole stream changes, and if it is no we learn precisely which jobs to give a model
instead.

The measurement is honest because it runs on **synthetic video rendered from real NGS
tracking**. Every player's true position at every instant is known exactly, so the
model's answer is scored the same way the CV pipeline is: Hungarian assignment per
frame, error in yards. Same scorer, same units, directly comparable numbers.

Two things are asked for, separately, because they fail differently:

*Field coordinates.* The whole job — implicitly registering the field and reading
positions off it. If this works, the CV pipeline is largely redundant.

*Image coordinates.* Only detection and identity, with registration still done by the
existing homography. A model might be hopeless at the first and useful at the second,
and the difference tells you where it belongs in the chain.

The prior going in is that temporal resolution is the binding constraint rather than
accuracy: video models sample frames sparsely, often around one per second, while the
contract is 10 Hz. A model that sees fourteen frames of a fourteen-second play cannot
emit a hundred and forty without inventing a hundred and twenty-six of them. That is
arithmetic, not a judgement about quality — and this harness reports the sampling rate
it actually got back, so the claim is checked rather than assumed.
"""

from __future__ import annotations

import os
import time
from dataclasses import dataclass, field as dc_field
from pathlib import Path
from typing import Any, Literal

import numpy as np

from ..contract import PlayTrack, PlayerFrame, RosterEntry, TrackingFrame
from .score import Score, score

DEFAULT_MODEL = "gemini-flash-latest"


# --------------------------------------------------------------------- schema

try:
    from pydantic import BaseModel, Field

    class PlayerAt(BaseModel):
        player_id: str = Field(description="A label for this player that you keep identical across every timestamp. Jersey number if you can read it, otherwise anything stable like 'offense-1'.")
        team: Literal["offense", "defense", "unknown"] = Field(description="Which side he is on, if you can tell.")
        x: float = Field(description="Position along the field in yards: 0 at the back of the left end zone, 10 at the left goal line, 60 at the 50, 110 at the right goal line, 120 at the back of the right end zone.")
        y: float = Field(description="Position across the field in yards, 0 at one sideline and 53.3 at the other.")

    class FrameAt(BaseModel):
        t: float = Field(description="Seconds from the start of the clip.")
        players: list[PlayerAt] = Field(description="Every player you can place at this instant.")

    class VideoTracking(BaseModel):
        frames: list[FrameAt] = Field(description="One entry per instant you can report. Report as many instants as you can actually distinguish — do not interpolate between them to pad the list.")
        frames_actually_seen: int = Field(description="Roughly how many distinct frames of this video you were able to look at. Be honest; this is used to check whether the timestamps above are real observations or interpolation.")
        notes: str = Field(description="What made this hard, and anything you are reporting with low confidence.")

    HAVE_PYDANTIC = True
except ImportError:  # pragma: no cover
    HAVE_PYDANTIC = False
    PlayerAt = FrameAt = VideoTracking = None  # type: ignore


SYSTEM = """\
You are extracting player tracking data from American football video, to be compared \
against the positions measured by a professional tracking system.

The field is 120 yards long including both end zones and 53.3 yards wide. Positions are \
in yards: x runs 0 to 120 along the field, y runs 0 to 53.3 across it. Work out the \
mapping from what you see to those coordinates from the painted lines — yard lines are \
five yards apart, and the end zones are ten yards deep.

Two things matter more than completeness.

Keep each player's label identical across timestamps. A position series that swaps two \
players halfway through is worse than one that drops a player entirely, because the \
error is invisible in any single frame.

Report only instants you actually looked at. If you sampled this video at one frame per \
second, give one entry per second — do not interpolate between them to produce a denser \
series. Interpolated positions look like data and are not, and the comparison this feeds \
will treat them as real. Say in frames_actually_seen how many distinct frames you saw.\
"""

PROMPT_FIELD = """\
This clip is one American football play. Give me every player's position on the field, \
in yards, at every instant you can distinguish.

Keep the labels consistent across time so each player's path can be followed.\
"""


# ----------------------------------------------------------------------- run


@dataclass
class VlmRun:
    model: str
    track: PlayTrack | None = None
    scored: Score | None = None
    frames_returned: int = 0
    frames_claimed_seen: int = 0
    effective_hz: float = 0.0
    seconds: float = 0.0
    notes: str = ""
    error: str | None = None

    def summary(self) -> str:
        if self.error:
            return f"{self.model}: FAILED — {self.error}"
        s = self.scored
        hz = f"{self.effective_hz:.2f} Hz"
        if s is None:
            return f"{self.model}: {self.frames_returned} frames ({hz}), not scored"
        return (
            f"{self.model}: median {s.median_yards:.2f} yd | p95 {s.p95_yards:.2f} | "
            f"coverage {s.coverage * 100:.0f}% | switches {s.id_switches} | "
            f"{self.frames_returned} frames ({hz}) | {self.seconds:.0f}s"
        )


def _client(api_key: str | None = None):
    try:
        from google import genai
    except ImportError as e:  # pragma: no cover
        raise ImportError("needs the Gemini SDK: pip install google-genai") from e
    key = api_key or os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        raise RuntimeError(
            "no Gemini key — set GEMINI_API_KEY in .env or the environment "
            "(get one at aistudio.google.com/apikey)"
        )
    return genai.Client(api_key=key)


def _upload(client, path: str | Path, timeout: float = 180.0):
    """Upload and wait for the file to finish processing.

    A video is not usable the instant the upload returns; the service transcodes it
    first, and a request against a PROCESSING file fails in a way that looks like a
    model error rather than a timing one.
    """
    f = client.files.upload(file=str(path))
    t0 = time.time()
    while f.state == "PROCESSING":
        if time.time() - t0 > timeout:
            raise TimeoutError(f"{path} was still processing after {timeout:.0f}s")
        time.sleep(2.0)
        f = client.files.get(name=f.name)
    if f.state == "FAILED":
        raise RuntimeError(f"upload failed: {getattr(f, 'error', 'unknown')}")
    return f


def to_play_track(reading, snap_offset: float = 0.0, source: str = "vlm") -> PlayTrack:
    """The model's answer in the shared contract, so the normal scorer can read it."""
    roster: dict[str, RosterEntry] = {}
    frames: list[TrackingFrame] = []
    for fr in sorted(reading.frames, key=lambda f: f.t):
        players: dict[str, PlayerFrame] = {}
        for p in fr.players:
            pid = str(p.player_id)
            roster.setdefault(pid, RosterEntry(id=pid, team=p.team, label=pid))
            players[pid] = PlayerFrame(x=float(p.x), y=float(p.y))
        frames.append(TrackingFrame(t=round(float(fr.t) - snap_offset, 3), players=players))
    return PlayTrack(
        meta={"source": source, "rate": None, "note": "positions as reported by a video model"},
        roster=roster, frames=frames,
    )


def run(
    video: str | Path,
    truth: PlayTrack,
    model: str = DEFAULT_MODEL,
    api_key: str | None = None,
    clip_starts_at: float | None = None,
    max_output_tokens: int = 60000,
) -> VlmRun:
    """Hand the whole clip to a video model and score what comes back.

    ``clip_starts_at`` is the truth-clock time of the video's first frame, used to put
    both series on the same clock. Defaults to the first truth frame's ``t``.
    """
    out = VlmRun(model=model)
    if not HAVE_PYDANTIC:
        out.error = "pydantic is required"
        return out
    try:
        from google.genai import types
        client = _client(api_key)
        f = _upload(client, video)
    except Exception as e:  # noqa: BLE001 — reported, never swallowed
        out.error = f"{type(e).__name__}: {e}"
        return out

    t0 = time.time()
    try:
        resp = client.models.generate_content(
            model=model,
            contents=[
                types.Part.from_uri(file_uri=f.uri, mime_type=f.mime_type),
                types.Part.from_text(text=PROMPT_FIELD),
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,
                response_mime_type="application/json",
                response_schema=VideoTracking,
                temperature=0.0,
                max_output_tokens=max_output_tokens,
            ),
        )
    except Exception as e:  # noqa: BLE001
        out.error = f"{type(e).__name__}: {e}"
        return out
    out.seconds = time.time() - t0

    reading = getattr(resp, "parsed", None)
    if reading is None:
        out.error = "the model returned nothing parseable"
        return out

    out.frames_returned = len(reading.frames)
    out.frames_claimed_seen = int(getattr(reading, "frames_actually_seen", 0) or 0)
    out.notes = reading.notes or ""
    if out.frames_returned >= 2:
        span = max(fr.t for fr in reading.frames) - min(fr.t for fr in reading.frames)
        out.effective_hz = (out.frames_returned - 1) / span if span > 0 else 0.0

    start = clip_starts_at if clip_starts_at is not None else (truth.frames[0].t if truth.frames else 0.0)
    out.track = to_play_track(reading, snap_offset=-start)

    try:
        out.scored = score(out.track, truth, play=f"{model} on video")
    except ValueError as e:
        out.error = f"could not score: {e}"
    return out


def compare(video, truth, models: list[str], **kw) -> list[VlmRun]:
    """Same clip, same truth, several models. The table this exists to produce."""
    return [run(video, truth, model=m, **kw) for m in models]


def table(runs: list[VlmRun]) -> str:
    lines = ["model                       median  p95    cover  switch  frames  Hz     notes",
             "-" * 92]
    for r in runs:
        if r.error:
            lines.append(f"{r.model:<26}  FAILED  {r.error[:50]}")
            continue
        s = r.scored
        lines.append(
            f"{r.model:<26} {s.median_yards:6.2f} {s.p95_yards:6.2f} "
            f"{s.coverage * 100:5.0f}% {s.id_switches:6d} {r.frames_returned:7d} "
            f"{r.effective_hz:5.2f}  {r.notes[:24]}" if s else f"{r.model:<26}  unscored"
        )
    return "\n".join(lines)
