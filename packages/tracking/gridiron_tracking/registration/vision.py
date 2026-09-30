"""Reading the field with a vision model, so registration stops being manual.

Registration has been the blocker for this whole stream, and every purely-algorithmic
attempt has failed on real footage: line detection finds three lines on a frame with
fifteen, and its strongest signal is the offensive line's limbs. The reason is not that
the maths is hard. It is that the hard part is *semantic* — not "where is this white
line", which a computer does to the pixel, but "which yard line is that, and is the
30 to its left or its right", which needs reading painted numbers on worn grass at an
angle. That is a language-and-vision problem.

So the division of labour here is deliberate and it is the whole design:

**The model answers semantic questions.** Where the structural lines are — sidelines,
goal line, hash rows. Which numbers are painted and roughly where, when any are legible.
What the chain crew's down marker says. What level of football this is — which matters
more than it sounds, because the hash marks sit 23.58 yards from the sideline in the
NFL, 20 in college and 17.78 in high school, and using the wrong one puts every player
six yards sideways with a clean-looking reprojection error.

The ordering there is the result of testing rather than design. The first version of
this prompt asked only about painted numbers, and on a real night game there was not a
single legible one in the frame — while the end zone, both sidelines and the hash rows
were unmistakable, and the chain crew's marker read the ball onto the 20 in plain
digits. Structural lines first, numbers if you are lucky.

**Classical CV answers geometric questions.** Given "there is a 30 at roughly 42% across
and 78% down", a local search finds the actual line to the pixel. A vision model asked
for exact coordinates will produce confident numbers that are tens of pixels out, and
four of those fitted a homography that reported no usable perspective and placed
players ninety yards off the field. Asking for fractions of the image, coarsely, is
asking for something it can actually deliver.

**And the loop closes.** The fitted field model is drawn back over the frame and shown
to the model again with one question: do these lines sit on the paint? That is a coarse
visual judgement — exactly what it is good at — and it catches the failure that
reprojection error cannot, which is a fit that is beautifully self-consistent and
labelled one yard line off.

Nothing here is trusted blindly. Every reading is checked against physical sanity
(a camera 3 to 60 yards up, players inside the sidelines) before it is used, and the
module would rather report that it could not register a frame than hand back a
confident wrong answer.
"""

from __future__ import annotations

import base64
import json
import os
from dataclasses import dataclass, field as dc_field
from typing import Any, Literal

import numpy as np

from .. import field as fieldmod

MODEL = "claude-opus-5"
GEMINI_MODEL = "gemini-flash-latest"

# Which backend a model name belongs to. The prompt, the schema and every check are
# shared: a reader is a model plus a transport, and the only thing worth varying while
# tuning a prompt is the model.
def backend_for(model: str) -> str:
    m = (model or "").lower()
    if m.startswith(("gemini", "models/gemini")):
        return "gemini"
    return "anthropic"


# --------------------------------------------------------------------- schema
# These are the shape of the answer. Keeping them narrow is part of the prompt:
# a field the model cannot fill honestly is a field it will fill dishonestly, so
# every one of them has an explicit "I cannot tell" value.

try:
    from pydantic import BaseModel, Field

    class YardNumber(BaseModel):
        """One painted number the model can actually read on the grass."""

        number: int = Field(description="The number painted on the turf: 10, 20, 30, 40 or 50.")
        x_frac: float = Field(description="How far across the image it sits, as a FRACTION between 0.0 (left edge) and 1.0 (right edge). Not pixels: a number halfway across a frame of any size is 0.5. Approximate is fine and expected.")
        y_frac: float = Field(description="How far down the image it sits, as a FRACTION between 0.0 (top edge) and 1.0 (bottom edge). Not pixels: a number two thirds of the way down is 0.67, whatever the frame's height.")
        side: Literal["left_half", "right_half", "unclear"] = Field(description="Which half of the field this number is in. If you are not sure, say 'unclear' — the software works it out from the order the numbers appear in, and a wrong guess here is worse than no guess.")
        confidence: Literal["certain", "probable", "guess"] = Field(description="How sure you are you read this number correctly. Use 'guess' freely; a guess that says so is useful, a guess that claims certainty is not.")

    class Landmark(BaseModel):
        """A structural line on the field, indicated roughly.

        These matter more than the painted numbers do. Testing the first version of
        this prompt on a real night game found not one legible yard number on the whole
        frame — but the end zone, both sidelines and the hash rows were perfectly
        obvious. A prompt that asks only for numbers gets nothing from most real
        footage.
        """

        kind: Literal[
            "goal_line", "end_zone_back", "far_sideline", "near_sideline",
            "far_hash_row", "near_hash_row",
        ] = Field(description="Which line this is. A goal line is where the solid end-zone paint begins; an end zone back is the outer edge of that paint. Hash rows are the two dashed lines running down the middle of the field.")
        p1: list[float] = Field(description="A rough point on this line as [across, down], both FRACTIONS between 0.0 and 1.0 — never pixels. Pick a point near one end of the visible stretch.")
        p2: list[float] = Field(description="A second rough point on the same line, in the same units, near the other end of the visible stretch. Further apart is better.")
        confidence: Literal["certain", "probable", "guess"] = Field(description="How sure you are this line is what you say it is.")

    class FieldReading(BaseModel):
        """Everything the model can say about the field in one frame."""

        landmarks: list[Landmark] = Field(description="Every structural line you can see. This is the most useful thing you can give, especially when the painted numbers are unreadable. Include the far sideline and the goal line whenever they are visible.")
        down_marker_number: int | None = Field(default=None, description="The YARD LINE number shown on the sideline yardage marker, if one is legible — it pins the field position when no painted number can be read. Do NOT report the down here: the orange down marker shows which down it is (1, 2, 3, 4) and the on-screen graphic says things like '2nd & 6', and neither is a yard line. Null unless you can see an actual yard-line number on a sideline marker.")
        yard_numbers: list[YardNumber] = Field(description="Every painted yard number you can actually read, from BOTH rows — the row near the far sideline and the row near the near sideline. A football field has two rows of numbers and reporting only the clearer one is the single most common reason a frame cannot be registered. Empty list if none are legible — that is a valid and useful answer, and on real footage it is a common one.")
        numbers_decrease_toward: Literal["left", "right", "unclear"] = Field(description="Starting at the 50 and walking toward the smaller numbers — 40, then 30, then 20 — which way across the image are you walking? Answer for the numbers you can actually see.")
        end_zone_at: Literal["left", "right", "both", "none_visible"] = Field(description="Where the end zone (solid painted area past the goal line) appears in the frame.")
        far_sideline_at: Literal["top", "bottom", "not_visible"] = Field(description="The sideline further from the camera — usually the one with the crowd or a wall behind it.")
        near_sideline_visible: bool = Field(description="Whether the sideline closest to the camera is actually visible, or hidden behind people or cropped out of frame.")
        camera_view: Literal["sideline_low", "sideline_elevated", "endzone", "corner", "unclear"] = Field(description="Where the camera appears to be. 'sideline_low' means roughly at player height in the stands; 'sideline_elevated' means a press box or high tripod looking down.")
        level: Literal["nfl", "college", "high_school", "unclear"] = Field(description="What level of football. This sets the hash mark spacing, so say 'unclear' rather than guessing — a wrong answer here moves every player sideways.")
        team_colors: list[str] = Field(description="Plain colour names for the two teams' jerseys, e.g. ['white', 'navy']. Empty if you cannot tell them apart.")
        notes: str = Field(description="Anything that would make registering this frame hard: occlusion, glare, worn paint, a camera that is panning, only part of the field visible.")

    class FitCheck(BaseModel):
        """The model's verdict on a drawn field model."""

        verdict: Literal["good", "close", "wrong"] = Field(description="'good' if the drawn lines sit on the real paint; 'close' if they are within about a line's width; 'wrong' if they are clearly off.")
        problem: str = Field(description="If not good, what is wrong in plain terms — 'the drawn lines are one yard line to the left', 'the field is too narrow', 'rotated'. Empty string if good.")
        off_by_yards: float | None = Field(default=None, description="If the drawing looks shifted along the field, roughly how many yards, signed. Null if you cannot tell or it is not a simple shift.")

    HAVE_PYDANTIC = True
except ImportError:  # pragma: no cover - pydantic ships with the core install
    HAVE_PYDANTIC = False
    YardNumber = FieldReading = FitCheck = None  # type: ignore


# --------------------------------------------------------------------- prompts

SYSTEM = """\
You are reading an American football field from a single video frame so that software \
can compute the mapping from image pixels to field coordinates.

You are being asked to do the half of this job that needs judgement — reading painted \
numbers, working out which way the field runs, telling one level of football from \
another. A separate classical computer-vision step handles precision. That division \
matters, so:

Give positions as rough fractions of the image. Do not try to be precise. A yard \
number that you place at 0.4 across when it is really at 0.43 costs nothing, because \
the software searches near where you point. Precision you invent is worse than the \
approximation you are confident in.

Report only what you can see. Every field below has a value meaning "I cannot tell" \
and using it is a correct answer, not a failure. A frame where the numbers are worn \
away, or the field is mostly out of shot, genuinely cannot be read, and saying so lets \
the software fall back to asking a human. Inventing a plausible reading produces a \
registration that is silently wrong by several yards, which is far more expensive than \
admitting the frame is unreadable.

Be careful with two things specifically. Every painted number except the 50 appears \
twice on a field, once in each half, so a number on its own does not say where down the \
field it is. What settles it is the *order*: report every number you can read, and the \
software reads the sequence left to right to see which side of the 50 each one is on. \
So read them all, place them in the right order across the image, and leave "side" as \
"unclear" whenever you are not certain — a confident wrong half is exactly ten to \
eighty yards of error, and an honest "unclear" costs nothing.\

And the level of football sets the hash mark spacing, which differs by nearly six yards \
between the NFL and high school; judge it from the stadium, the crowd, the uniforms and \
the field markings, and say "unclear" if it is genuinely ambiguous.\
"""

READ_PROMPT = """\
This is one frame from a football video. Read the field.

Start with the structural lines, because they are usually the useful part: the far \
sideline, the near sideline if it is not buried behind people, the goal line and the \
outer edge of the end zone if either is in shot, and the two rows of hash marks down \
the middle. Give two rough points along each one, and take the two points from opposite \
ends of the stretch you can actually see rather than from near each other.

Spread matters more than precision everywhere in this. Lines and numbers taken from a \
narrow band across the top of the picture fit perfectly to themselves and then place \
players in the car park, because nothing in the answer says anything about the rest of \
the frame. Whatever you can see low in the image is worth more than a fourth reading \
from the same strip near the top.

Then the painted yard numbers, if any are legible. A field has two rows of them, one \
near each sideline, and both matter: look along the top of the field and again along the \
bottom, and report what you find in each. Numbers from only one row leave the software \
with no evidence about the near half of the picture at all, and a fit made from the top \
strip of an image is the commonest way this fails. Report every one you can read, not \
just the clearest, and place each roughly where it sits in the image — the sequence \
across the frame is what tells the software which half of the field it is looking at, \
and one number on its own tells it almost nothing. Often none are legible — worn paint, \
night games, and camera angles all defeat them, and an empty list is the right answer \
when that is the case. Do not strain to read one.

Give every position as a fraction of the image, between 0 and 1: 0.5 means halfway \
across, or halfway down. Not pixels.

Also look at the sideline for the chain crew's down marker or a yardage marker. If it \
shows a number, that names the yard line the ball is on, which is worth more than a \
guessed painted number.

Finally the level of football, which sets the hash spacing, and anything in the notes \
that would make this frame hard to register.\
"""

CHECK_PROMPT = """\
The green lines drawn over this frame are a computed model of the field — every \
five-yard line, with the goal lines and the back of each end zone in red and the two \
sidelines in magenta.

Do those drawn lines sit on the actual painted lines in the photograph?

Look at whether the drawn yard lines land on real yard lines, whether the drawn \
sidelines land on the real edges of the field, and whether the spacing looks right. A \
fit can be perfectly self-consistent and still be labelled one yard line off, which is \
exactly the error reprojection maths cannot see and you can.\
"""


# ----------------------------------------------------------------------- calls


@dataclass
class VisionResult:
    reading: Any | None
    raw: str
    error: str | None = None


def _encode(img, fmt: str = ".jpg", max_width: int = 1600) -> tuple[str, str]:
    """A frame as base64, downscaled enough to be cheap but not to be unreadable.

    Painted numbers on worn grass are the smallest thing that has to survive, which is
    why this does not shrink further.
    """
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise ImportError("vision registration needs opencv: pip install opencv-python") from e
    h, w = img.shape[:2]
    if w > max_width:
        img = cv2.resize(img, (max_width, int(h * max_width / w)), interpolation=cv2.INTER_AREA)
    ok, buf = cv2.imencode(fmt, img, [cv2.IMWRITE_JPEG_QUALITY, 92])
    if not ok:
        raise RuntimeError("could not encode the frame")
    media = "image/jpeg" if fmt == ".jpg" else "image/png"
    return base64.standard_b64encode(buf.tobytes()).decode("utf-8"), media


def _client(client=None):
    if client is not None:
        return client
    try:
        import anthropic
    except ImportError as e:  # pragma: no cover
        raise ImportError(
            "vision registration needs the Anthropic SDK: pip install anthropic"
        ) from e
    # A bare constructor resolves ANTHROPIC_API_KEY, ANTHROPIC_AUTH_TOKEN, or an
    # `ant auth login` profile — so an unset API key does not mean no credentials.
    return anthropic.Anthropic()


def _ask(client, img, prompt: str, schema, max_tokens: int = 4000,
         model: str | None = None, **extra) -> VisionResult:
    """One structured vision call, to whichever backend the model name names.

    ``extra`` is passed straight through, which is how a caller adds ``betas=[...]``
    and ``fallbacks=...`` without this module guessing at a parameter combination it
    has not verified.
    """
    model = model or MODEL
    if backend_for(model) == "gemini":
        return _ask_gemini(img, prompt, schema, model=model, max_tokens=max_tokens)

    data, media = _encode(img)
    try:
        resp = _client(client).beta.messages.parse(
            model=model,
            max_tokens=max_tokens,
            system=SYSTEM,
            messages=[{
                "role": "user",
                "content": [
                    {"type": "image", "source": {"type": "base64", "media_type": media, "data": data}},
                    {"type": "text", "text": prompt},
                ],
            }],
            output_format=schema,
            **extra,
        )
    except Exception as e:  # noqa: BLE001 - surfaced to the caller, never swallowed
        return VisionResult(None, "", f"{type(e).__name__}: {e}")

    if getattr(resp, "stop_reason", None) == "refusal":
        return VisionResult(None, "", "the model declined this request")
    return VisionResult(resp.parsed_output, str(getattr(resp, "content", "")))


_GEMINI = None


def _ask_gemini(img, prompt: str, schema, model: str, max_tokens: int = 4000) -> VisionResult:
    """The same prompt and the same schema, through Gemini.

    Worth having for a reason beyond which key happens to be in .env: a prompt that only
    works on one model is a prompt that has been fitted to that model's habits. Reading
    the same frame with two backends is the cheapest test there is of whether the
    instructions are actually clear.
    """
    import os

    try:
        from google import genai
        from google.genai import types
    except ImportError as e:  # pragma: no cover
        return VisionResult(None, "", f"needs the Gemini SDK: pip install google-genai ({e})")

    key = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
    if not key:
        return VisionResult(None, "", "GEMINI_API_KEY is not set")

    global _GEMINI
    if _GEMINI is None:
        # Held open deliberately. A client built inline is closed the moment the
        # expression ends, and the request then fails with "the client has been closed"
        # — which reads like an API problem and is not one.
        _GEMINI = genai.Client(api_key=key)

    data, media = _encode(img)
    try:
        resp = _GEMINI.models.generate_content(
            model=model,
            contents=[
                types.Part.from_bytes(data=base64.b64decode(data), mime_type=media),
                prompt,
            ],
            config=types.GenerateContentConfig(
                system_instruction=SYSTEM,
                response_mime_type="application/json",
                response_schema=schema,
                max_output_tokens=max_tokens,
            ),
        )
    except Exception as e:  # noqa: BLE001
        return VisionResult(None, "", f"{type(e).__name__}: {e}")

    parsed = getattr(resp, "parsed", None)
    if parsed is None:
        txt = getattr(resp, "text", "") or ""
        try:
            parsed = schema.model_validate(json.loads(txt))
        except Exception as e:  # noqa: BLE001
            return VisionResult(None, txt, f"could not parse the answer: {type(e).__name__}: {e}")
    return VisionResult(parsed, str(getattr(resp, "text", "")))


def read_field(img, client=None, **extra) -> VisionResult:
    """Ask the model what it can see on this field."""
    if not HAVE_PYDANTIC:
        return VisionResult(None, "", "pydantic is required for structured output")
    return _ask(_client(client), img, READ_PROMPT, FieldReading, **extra)


def check_fit(img_with_overlay, client=None, **extra) -> VisionResult:
    """Show the model the fitted field model and ask whether it lands on the paint."""
    if not HAVE_PYDANTIC:
        return VisionResult(None, "", "pydantic is required for structured output")
    return _ask(_client(client), img_with_overlay, CHECK_PROMPT, FitCheck, **extra)


# ------------------------------------------------------- semantics to geometry


def spec_for(level: str) -> fieldmod.FieldSpec:
    """The reading's level to a field spec. Unclear falls back to NFL, loudly."""
    return {"nfl": fieldmod.NFL, "college": fieldmod.NCAA,
            "high_school": fieldmod.HIGH_SCHOOL}.get(level, fieldmod.NFL)


def infer_halves(reading) -> list[str]:
    """Work out which half of the field each painted number is in, without asking.

    Asking the model this per number does not work, and the failure is not a close call:
    on the reference frame it read 20, 30, 40, 50 running left to right across one half
    and labelled the 40 as belonging to the other one. That single word moves the 40 from
    the 50-yard line to the 70, and the fit built on it is wrong by twenty yards while
    looking entirely healthy.

    Two things settle it from evidence the model is reliable about.

    A visible end zone settles it outright. The software takes the end zone at the right
    of the image as the high-x end of the field, so a frame with an end zone in shot has
    already said which half everything else in it belongs to.

    Failing that, the order does. Numbers on one side of the 50 increase toward it and on
    the other decrease away from it, so sorting what was read by position across the image
    and looking at which way the sequence runs assigns every number at once — from where
    things sit relative to each other, which the model gets right, rather than from a
    judgement it gets wrong.

    The model's own answer is used only when neither applies: one number, or numbers that
    do not form a run, in a frame with no end zone in it.
    """
    nums = sorted([n for n in getattr(reading, "yard_numbers", []) or []],
                  key=lambda n: n.x_frac)
    notes: list[str] = []
    if not nums:
        return notes

    side = None
    ez = getattr(reading, "end_zone_at", "none_visible")
    if ez == "left":
        side, why = "left_half", "the end zone in shot is on the left"
    elif ez == "right":
        side, why = "right_half", "the end zone in shot is on the right"

    if side is None:
        non50 = [n for n in nums if n.number != 50]
        if len(non50) < 2:
            return notes
        rising = sum(1 for a, b in zip(non50, non50[1:]) if b.number > a.number)
        falling = sum(1 for a, b in zip(non50, non50[1:]) if b.number < a.number)
        if rising and falling:
            notes.append("the yard numbers read do not run in one direction; halves left as read")
            return notes
        if not rising and not falling:
            return notes
        # Image-left is the low-x end of the field, the same convention the landmark
        # resolver uses. Numbers rising toward the right are therefore climbing toward a
        # 50 that lies to the right, so everything read is in the low-x half.
        side = "left_half" if rising else "right_half"
        why = f"the sequence {[n.number for n in nums]} runs {'up' if rising else 'down'} to the right"

    changed = [n.number for n in nums if n.number != 50 and n.side != side]
    for n in nums:
        if n.number != 50:
            n.side = side
    notes.append(f"halves set to {side} because {why}"
                 + (f"; overruled {changed} which were read otherwise" if changed else ""))
    return notes


def field_x_for(number: int, side: str, numbers_increase_toward: str = "unclear") -> float | None:
    """Turn a painted number plus which half it is in into a field x.

    The 50 is the only number that appears once. Every other one appears twice, and
    which of the two you are looking at is the single most common way to be exactly
    the wrong distance down the field.
    """
    if number == 50:
        return 60.0
    if side == "left_half":
        return 10.0 + number
    if side == "right_half":
        return 110.0 - number
    return None


def refine_to_line(
    img,
    approx_px: tuple[float, float],
    window: int = 90,
    min_col_support: int = 6,
) -> tuple[float, float] | None:
    """Snap a rough position onto the actual painted line near it.

    This is the half the model is not asked to do. It takes a coarse pointer — "a 30 is
    around here" — and finds the painted line to the pixel.

    The choice that matters is picking the candidate *nearest the pointer*, not the
    strongest one. A window wide enough to tolerate a coarse pointer is wider than the
    gap between yard lines — 90 px against a 96 px spacing on the test frame — so
    "brightest column in the window" would lock onto whichever neighbour has the
    cleanest paint. Nearest peak is what makes a wide window safe.

    Measured on a rendered frame with known geometry, snapping halves the error of a
    coarse pointer: a 40 px guess lands 17.7 px out, a 25 px guess 6.2, a 10 px guess
    3.0, and it found a line on 28 probes out of 28.

    Returns None when nothing in the window looks like a line, because a confident
    wrong point is what breaks a fit.
    """
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise ImportError("vision registration needs opencv") from e

    h, w = img.shape[:2]
    cx, cy = int(approx_px[0]), int(approx_px[1])
    x0, x1 = max(0, cx - window), min(w, cx + window)
    y0, y1 = max(0, cy - window // 2), min(h, cy + window // 2)
    if x1 - x0 < 8 or y1 - y0 < 8:
        return None

    patch = img[y0:y1, x0:x1]
    hsv = cv2.cvtColor(patch, cv2.COLOR_BGR2HSV)
    paint = cv2.inRange(hsv, (0, 0, 140), (180, 80, 255))
    col = paint.sum(axis=0) / 255.0
    if col.max() < min_col_support:
        return None

    # Every column that is a local maximum and carries real support is a candidate
    # line; the pointer decides which one was meant.
    thresh = max(min_col_support, 0.45 * col.max())
    cands = [
        i for i in range(1, len(col) - 1)
        if col[i] >= thresh and col[i] >= col[i - 1] and col[i] >= col[i + 1]
    ]
    if not cands:
        cands = [int(np.argmax(col))]
    target = cx - x0
    best = min(cands, key=lambda i: abs(i - target))

    rows = np.nonzero(paint[:, best])[0]
    if len(rows) < 3:
        return None
    return (float(x0 + best), float(y0 + rows.mean()))


@dataclass
class AutoRegistration:
    """What a vision-assisted attempt produced, and whether to believe it."""

    homography: Any | None = None
    spec: fieldmod.FieldSpec = fieldmod.NFL
    correspondences: list[tuple[tuple[float, float], tuple[float, float]]] = dc_field(default_factory=list)
    reading: Any | None = None
    verdict: str = "not attempted"
    problems: list[str] = dc_field(default_factory=list)
    notes: list[str] = dc_field(default_factory=list)
    turf: float = float("nan")
    paint: float = float("nan")

    @property
    def ok(self) -> bool:
        return self.homography is not None and not self.problems and self.verdict in ("good", "close")

    def summary(self) -> str:
        n = len(self.correspondences)
        s = (f"{self.spec.name} field, {n} correspondence{'' if n == 1 else 's'}, "
             f"turf {self.turf:.2f}, paint {self.paint:.2f}, verdict {self.verdict}")
        if self.notes:
            s += "; " + "; ".join(self.notes)
        return s + ("" if not self.problems else "; " + "; ".join(self.problems))


def sanity_problems(
    h,
    image_size: tuple[int, int],
    spec: fieldmod.FieldSpec,
    correspondences: list | None = None,
) -> list[str]:
    """Physical checks a registration has to survive before anyone uses it.

    These catch what reprojection error cannot. A homography fitted to four points is
    perfect on those four points by construction; the question is whether it implies a
    camera that could exist and a field the right size.

    The correspondence-geometry checks below exist because the camera checks alone are
    not enough: a fit built from points on only two yard lines passed every physical
    test while being a hundred yards wrong down the field. Conditioning is not a
    detail — a homography is only as constrained as the spread of what fitted it.
    """
    from .camera import Camera, DegenerateGeometry

    out: list[str] = []
    try:
        cam = Camera.from_homography(h, image_size=image_size)
        if not (3.0 <= cam.height_yards <= 60.0):
            out.append(f"implies a camera {cam.height_yards:.0f} yards up, which is not where a camera goes")
        if not (200.0 <= cam.focal_px <= 12000.0):
            out.append(f"implies a focal length of {cam.focal_px:.0f} px")
    except DegenerateGeometry:
        out.append("no usable perspective — the points are probably collinear")
    except (np.linalg.LinAlgError, ValueError) as e:
        out.append(f"camera could not be recovered: {e}")

    # The field's own corners must land somewhere sane in the image.
    w, hgt = image_size
    corners = h.to_image([[0, 0], [spec.length, 0], [spec.length, spec.width], [0, spec.width]])
    if not np.isfinite(corners).all():
        out.append("field corners project to infinity")
    elif np.abs(corners).max() > 50 * max(w, hgt):
        out.append("field corners project absurdly far outside the frame")
    else:
        # The four corners must stay in order once projected. A fit that folds the
        # field over on itself is degenerate however well it fits its own points.
        area = 0.0
        for i in range(4):
            x1, y1 = corners[i]
            x2, y2 = corners[(i + 1) % 4]
            area += x1 * y2 - x2 * y1
        if abs(area) < 1e-6:
            out.append("the field projects to zero area")
        else:
            signs = []
            for i in range(4):
                a, b, c = corners[i], corners[(i + 1) % 4], corners[(i + 2) % 4]
                signs.append(np.sign((b[0] - a[0]) * (c[1] - b[1]) - (b[1] - a[1]) * (c[0] - b[0])))
            if len(set(signs)) > 1:
                out.append("the field projects to a self-crossing shape — the fit is folded")

    out.extend(_correspondence_problems(correspondences or []))
    return out


def turf_overlap(img, h, spec: fieldmod.FieldSpec) -> float:
    """What fraction of the projected field actually lands on grass.

    This exists because asking the model to check its own drawing does not work. Shown
    a fit whose sidelines ran through the crowd well above the real field, both
    gemini-3.5-flash and flash-lite answered "good, off by 0.0 yards". A model will
    rubber-stamp its own geometry, so the check has to be something that cannot be
    talked into agreeing.

    Green pixels can. A correct registration puts the playing surface on the playing
    surface; a fit that has drifted into the stands or the stadium roof does not, and
    the turf mask says so without an opinion.
    """
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise ImportError("this check needs opencv") from e
    from .lines import turf_mask

    hgt, wid = img.shape[:2]
    turf = turf_mask(img)

    # Sample the field on a grid, project, and ask how many landed on grass.
    xs = np.linspace(spec.goal_a, spec.goal_b, 24)
    ys = np.linspace(0.0, spec.width, 12)
    pts = np.array([[x, y] for x in xs for y in ys])
    px = h.to_image(pts)
    good = inside = 0
    for u, v in px:
        if not (np.isfinite(u) and np.isfinite(v)):
            continue
        iu, iv = int(round(u)), int(round(v))
        if not (0 <= iu < wid and 0 <= iv < hgt):
            continue
        inside += 1
        if turf[iv, iu] > 0:
            good += 1
    return good / inside if inside else 0.0


def paint_alignment(img, h, spec: fieldmod.FieldSpec, band_px: int = 6) -> float:
    """Do the drawn yard lines land on painted yard lines?

    This check exists because of a fit that was wrong by a factor of two and a half and
    passed everything else. Its reprojection error was small, its camera was at a
    plausible height, the field's corners were in order, 91% of the projected field
    landed on grass, and 90.3% of tracked players reprojected onto a real detected
    person. Every one of those numbers was true and none of them could see the error,
    because all of them are round trips through the same map: a homography that is
    self-consistent agrees with itself no matter how wrong its scale is.

    The paint cannot be talked into agreeing. A correct registration puts its five-yard
    lines on five-yard lines; a fit stretched two and a half times downfield puts them
    on grass, and the ratio of white pixels under the drawn lines to white pixels under
    the same lines nudged sideways says which happened.

    Scored against a shuffled control rather than in absolute terms, because a frame with
    worn paint, a logo, or a wide shot full of numbers has a different amount of white in
    it than a clean one. Above about 1.6 the lines are on the paint; near 1.0 the fit is
    finding no more paint than chance.
    """
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise ImportError("this check needs opencv") from e
    from .lines import paint_mask, turf_mask

    hgt, wid = img.shape[:2]
    paint = paint_mask(img, turf_mask(img))
    paint = cv2.dilate(paint, np.ones((3, 3), np.uint8))

    def hits(offset_yards: float) -> tuple[int, int]:
        on = tot = 0
        for x in np.arange(spec.goal_a, spec.goal_b + 0.01, 5.0):
            ys = np.linspace(2.0, spec.width - 2.0, 40)
            px = h.to_image(np.column_stack([np.full_like(ys, x + offset_yards), ys]))
            for u, v in px:
                if not (np.isfinite(u) and np.isfinite(v)):
                    continue
                iu, iv = int(round(u)), int(round(v))
                if not (band_px <= iu < wid - band_px and band_px <= iv < hgt - band_px):
                    continue
                tot += 1
                win = paint[iv - band_px:iv + band_px + 1, iu - band_px:iu + band_px + 1]
                on += int(win.any())
        return on, tot

    on, tot = hits(0.0)
    if tot < 40:
        return float("nan")
    # The control is the same lines moved two and a half yards — halfway to the next one,
    # which is the emptiest turf there is. Comparing against it removes how much paint
    # the frame happens to contain.
    ctrl_on, ctrl_tot = hits(2.5)
    base = (ctrl_on / ctrl_tot) if ctrl_tot else 0.0
    if base < 0.02:
        # A control that finds no paint at all does not mean the fit is infinitely good,
        # it means the frame has almost no white in it — a night game, worn grass, or a
        # fit that has wandered somewhere with nothing painted on it. Dividing by that
        # produced "inf" on the high-school kickoff and passed a fit that was collinear.
        # There is no ratio to report, and saying so is the answer.
        return float("nan")
    return (on / tot) / base


def normalise_reading_for(reading, image_size: tuple[int, int]) -> list[str]:
    """Fix, or drop, positions the model gave in the wrong units. Reports what it did.

    Every position in the schema is documented as a fraction of the image, and the model
    is told so twice. It still sometimes answers in pixels — on the clip this pipeline was
    built around, every yard number came back with ``x_frac`` correctly between 0 and 1
    and ``y_frac`` as 787, 809, 839. Multiplied by the image height those became points
    half a million pixels below the frame, the fit built on them was stretched two and a
    half times downfield, and nothing downstream noticed.

    So the units are checked here rather than trusted. A value that is plainly a pixel
    coordinate — outside [0, 1] but inside the image — is converted, because the model
    saw the right thing and wrote it down in the wrong unit and that reading is worth
    keeping. A value that is neither is dropped, because it means nothing.

    Mutates the reading and returns a list of what happened, for the caller to record.
    """
    w, h = image_size
    notes: list[str] = []

    def fix(val, extent, what):
        v = float(val)
        if 0.0 <= v <= 1.0:
            return v
        if 1.0 < v <= extent:
            notes.append(f"{what} came back as a pixel ({v:g}); converted to a fraction")
            return v / extent
        notes.append(f"{what} is {v:g}, neither a fraction nor a pixel in a {w}x{h} frame; dropped")
        return None

    keep_numbers = []
    for n in getattr(reading, "yard_numbers", []) or []:
        x = fix(n.x_frac, w, f"the {n.number}'s x")
        y = fix(n.y_frac, h, f"the {n.number}'s y")
        if x is None or y is None:
            continue
        n.x_frac, n.y_frac = x, y
        keep_numbers.append(n)
    reading.yard_numbers = keep_numbers

    keep_lm = []
    for lm in getattr(reading, "landmarks", []) or []:
        p1 = [fix(lm.p1[0], w, f"the {lm.kind}'s first x"), fix(lm.p1[1], h, f"the {lm.kind}'s first y")]
        p2 = [fix(lm.p2[0], w, f"the {lm.kind}'s second x"), fix(lm.p2[1], h, f"the {lm.kind}'s second y")]
        if any(v is None for v in p1 + p2):
            continue
        lm.p1, lm.p2 = p1, p2
        keep_lm.append(lm)
    reading.landmarks = keep_lm
    return notes


def _correspondence_problems(correspondences: list) -> list[str]:
    out: list[str] = []
    if correspondences:
        fx = sorted({round(p[0][0], 2) for p in correspondences})
        fy = sorted({round(p[0][1], 2) for p in correspondences})
        if len(fx) < 3:
            out.append(
                f"points lie on only {len(fx)} yard line(s) — not enough to fix scale "
                "down the field; a fit like this can be a hundred yards wrong and still "
                "look perfect on its own points"
            )
        if len(fy) < 2:
            out.append(f"points lie on only {len(fy)} lane(s) across the field")
        if len(fx) >= 2 and (max(fx) - min(fx)) < 15.0:
            out.append(f"points span only {max(fx) - min(fx):.0f} yards downfield")
        if len(fy) >= 2 and (max(fy) - min(fy)) < 10.0:
            out.append(f"points span only {max(fy) - min(fy):.0f} yards across")
    return out


# ------------------------------------------------------------ reading to a fit

# Where each structural landmark sits in field coordinates. A line is either a fixed
# x (running across the field) or a fixed y (running along it), and which one it is
# decides what a point on it constrains.
_LANDMARK_FIELD: dict[str, tuple[str, Any]] = {
    "far_sideline":  ("y", "width"),
    "near_sideline": ("y", 0.0),
    "far_hash_row":  ("y", "hash_hi"),
    "near_hash_row": ("y", "hash_lo"),
    "goal_line":     ("x", None),      # 10 or 110 — resolved from end_zone_at
    "end_zone_back": ("x", None),      # 0 or 120
}


def _resolve(kind: str, spec: fieldmod.FieldSpec, end_zone_at: str) -> tuple[str, float] | None:
    axis, val = _LANDMARK_FIELD[kind]
    if axis == "y":
        lo, hi = spec.hashes
        return ("y", {"width": spec.width, "hash_hi": hi, "hash_lo": lo}.get(val, val))
    # An end zone on the right of the image is not necessarily the +x end of the field;
    # the caller's coordinate convention decides. Right-of-image is taken as the high-x
    # end, which the fit check will catch if it is backwards.
    if end_zone_at == "left":
        return ("x", 10.0 if kind == "goal_line" else 0.0)
    if end_zone_at == "right":
        return ("x", 110.0 if kind == "goal_line" else 120.0)
    return None


def number_rows(reading, spec: fieldmod.FieldSpec) -> tuple[dict[int, float], list[str]]:
    """Which row of painted numbers each reading belongs to, and its field y.

    A painted number is a *point*, not a line, and its distance from the sideline is
    known to the inch — nine yards in college, twelve in the NFL. The reading already
    carries where each number sits vertically in the image and that was being discarded,
    which is why every correspondence on the reference frame landed in the top 200 pixels
    of a 720-pixel image and the fit came back "no usable perspective". A homography is
    only as constrained as the spread of what fitted it, and there was none.

    The two rows separate cleanly because they are at opposite ends of the frame — a
    number belongs to whichever row centre it is nearer, and which row is the far one is
    the one thing the reading already says outright.

    Returns a map from the index of a yard number to its field y, and notes.
    """
    nums = getattr(reading, "yard_numbers", []) or []
    notes: list[str] = []
    if not nums:
        return {}, notes

    ys = np.array([n.y_frac for n in nums], dtype=float)
    far_is_top = reading.far_sideline_at != "bottom"
    far_y = spec.width - spec.number_inset - 1.0   # centre of the glyph, not its base
    near_y = spec.number_inset + 1.0

    if float(np.ptp(ys)) < 0.12:
        # One row only. Which one it is follows from where it sits in the frame.
        row = far_y if (ys.mean() < 0.5) == far_is_top else near_y
        notes.append(f"all yard numbers are in one row; placed at y={row:.0f}")
        return {i: row for i in range(len(nums))}, notes

    split = (ys.min() + ys.max()) / 2.0
    out: dict[int, float] = {}
    for i, y in enumerate(ys):
        top = y < split
        out[i] = far_y if top == far_is_top else near_y
    n_far = sum(1 for v in out.values() if v == far_y)
    notes.append(f"yard numbers split into {n_far} far and {len(out) - n_far} near")
    return out, notes


def drop_out_of_order(evidence: list[tuple[float, float, str]], far_at_top: bool
                      ) -> tuple[list[tuple[float, float, str]], list[str]]:
    """Throw away y-landmarks whose pixel row contradicts the order of the field.

    A football field's lines have a fixed order across it — far sideline, far hash, near
    hash, near sideline — and a camera cannot reorder them. So a reading that puts the
    far hash row seventeen pixels below the far number row, when the real gap is two
    hundred, has made a mistake that needs no ground truth to detect.

    That is not a hypothetical. On the reference frame the model placed both hash rows
    within a few pixels of the number rows, every time it was asked. Hash marks are the
    least distinctive paint on a field — short dashes, no colour of their own — and they
    are the one landmark it consistently cannot find, while it reads the numbers beside
    them perfectly. Those bad rows were half the vertical evidence in the fit.

    The rows that survive are chosen by preference, not by majority: a row derived from a
    digit the model actually read outranks a line it merely pointed at. Everything that
    breaks the order after that is dropped rather than averaged, because a landmark in
    the wrong place is not a noisy measurement of the right one.
    """
    rank = {"number_row": 0, "sideline": 1, "hash": 2}
    ordered = sorted(evidence, key=lambda e: -e[0] if far_at_top else e[0])
    kept: list[tuple[float, float, str]] = []
    notes: list[str] = []
    for item in ordered:
        fy, py, kind = item
        bad = None
        for k in kept:
            # Going far to near, the pixel row must increase. Ties are as wrong as
            # inversions: two field lines ten yards apart are not the same pixel row.
            if (py - k[1]) <= 1.0:
                bad = k
                break
        if bad is None:
            kept.append(item)
            continue
        if rank[kind] < rank[bad[2]]:
            notes.append(f"dropped the {bad[2]} at y={bad[0]:.0f}: it sits at pixel row "
                         f"{bad[1]:.0f}, on the wrong side of the {kind} at y={fy:.0f}")
            kept.remove(bad)
            kept.append(item)
        else:
            notes.append(f"dropped the {kind} at y={fy:.0f}: pixel row {py:.0f} puts it on "
                         f"the wrong side of the {bad[2]} at y={bad[0]:.0f}")
    return kept, notes


def to_correspondences(
    reading, img, spec: fieldmod.FieldSpec, refine: bool = True,
) -> tuple[list, list[str]]:
    """Turn a reading into field↔pixel pairs, snapping each to real paint.

    A landmark gives a *line*, not a point — "the far sideline runs through here and
    here" fixes one coordinate of each endpoint and says nothing about the other. So
    each endpoint contributes a partial constraint, and only a landmark whose crossing
    with another landmark is known yields a full point. In practice the useful pairs
    come from the yard numbers (which fix x) evaluated at a landmark (which fixes y),
    which is why both halves of the reading matter.
    """
    h, w = img.shape[:2]
    notes: list[str] = []
    lines: dict[str, tuple[str, float, tuple[float, float], tuple[float, float]]] = {}

    for lm in getattr(reading, "landmarks", []) or []:
        if lm.confidence == "guess":
            notes.append(f"ignored a guessed {lm.kind}")
            continue
        res = _resolve(lm.kind, spec, reading.end_zone_at)
        if res is None:
            notes.append(f"could not place {lm.kind}: end zone position unclear")
            continue
        axis, val = res
        p1 = (lm.p1[0] * w, lm.p1[1] * h)
        p2 = (lm.p2[0] * w, lm.p2[1] * h)
        lines[lm.kind] = (axis, val, p1, p2)

    pairs: list[tuple[tuple[float, float], tuple[float, float]]] = []

    rows, row_notes = number_rows(reading, spec)
    notes.extend(row_notes)

    # Everything that claims to know a field y, checked against the one thing that
    # cannot be reordered: the order the lines come in across a field.
    far_at_top = reading.far_sideline_at != "bottom"
    evidence: list[tuple[float, float, str]] = []
    for kind, (axis, val, p1, p2) in lines.items():
        if axis != "y":
            continue
        evidence.append((val, (p1[1] + p2[1]) / 2.0, "hash" if "hash" in kind else "sideline"))
    nums_all = getattr(reading, "yard_numbers", []) or []
    for fy in sorted({v for v in rows.values()}):
        px_rows = [nums_all[i].y_frac * h for i, v in rows.items() if v == fy]
        if px_rows:
            evidence.append((fy, float(np.mean(px_rows)), "number_row"))
    kept, order_notes = drop_out_of_order(evidence, far_at_top)
    notes.extend(order_notes)
    keep_y = {round(e[0], 2) for e in kept}
    for kind in [k for k, v in lines.items() if v[0] == "y" and round(v[1], 2) not in keep_y]:
        lines.pop(kind)

    # A yard number fixes x by which number it is, and y by which row it is painted in.
    # That is a whole point from one reading, and it is the only kind of evidence in a
    # broadcast frame that reaches the bottom of the image.
    y_lines = {k: v for k, v in lines.items() if v[0] == "y"}
    for ni, num in enumerate(getattr(reading, "yard_numbers", []) or []):
        if num.confidence == "guess":
            continue
        fx = field_x_for(num.number, num.side)
        if fx is None:
            notes.append(f"a {num.number} was read but which half is unclear, so it was dropped")
            continue
        px = (num.x_frac * w, num.y_frac * h)
        if refine:
            snapped = refine_to_line(img, px)
            if snapped is None:
                notes.append(f"no paint found near the {num.number}; used the rough position")
            else:
                # The snap moves the point onto the yard line, which is an x correction.
                # Its y is the row the number is painted in, and taking the snapped y
                # instead would put the point wherever the paint search happened to look.
                px = (snapped[0], px[1])
        if ni in rows:
            pairs.append(((fx, rows[ni]), px))
        for kind, (_, fy, q1, q2) in y_lines.items():
            # Where does that yard line cross this lane? Interpolate along the lane to
            # the yard line's own image x.
            if abs(q2[0] - q1[0]) < 1e-6:
                continue
            t = (px[0] - q1[0]) / (q2[0] - q1[0])
            if not (-0.35 <= t <= 1.35):       # far outside the stretch that was seen
                continue
            cross = (px[0], q1[1] + t * (q2[1] - q1[1]))
            pairs.append(((fx, fy), cross))

    # x-landmarks (goal line, end zone back) cross the y-landmarks the same way.
    for kind, (axis, fx, p1, p2) in lines.items():
        if axis != "x":
            continue
        for _, (_unused_axis, fy, q1, q2) in y_lines.items():
            d1 = (p2[0] - p1[0], p2[1] - p1[1])
            d2 = (q2[0] - q1[0], q2[1] - q1[1])
            den = d1[0] * d2[1] - d1[1] * d2[0]
            if abs(den) < 1e-6:
                continue
            t = ((q1[0] - p1[0]) * d2[1] - (q1[1] - p1[1]) * d2[0]) / den
            u = ((q1[0] - p1[0]) * d1[1] - (q1[1] - p1[1]) * d1[0]) / den
            # Both lines were only *seen* between their two given points. An
            # intersection well outside either stretch is an extrapolation of a
            # coarsely-placed line, and its error grows without limit — one such point
            # put a correspondence 800 px outside the frame and dragged the fit 112
            # yards off while every sanity check still passed.
            if not (-0.25 <= t <= 1.25 and -0.25 <= u <= 1.25):
                notes.append(f"{kind} intersection falls outside the stretch that was seen")
                continue
            cross = (p1[0] + t * d1[0], p1[1] + t * d1[1])
            if -0.2 * w <= cross[0] <= 1.2 * w and -0.2 * h <= cross[1] <= 1.2 * h:
                pairs.append(((fx, fy), cross))

    # Two points claiming the same field position help nothing. The key is the whole
    # position: the same yard line read in the far number row and the near one are two
    # different points, and they are the pair that gives the fit its vertical evidence.
    seen: set[tuple[float, float]] = set()
    unique = []
    for fld, px in pairs:
        k = (round(fld[0], 2), round(fld[1], 2))
        if k in seen:
            continue
        seen.add(k)
        unique.append((fld, px))
    return unique, notes


def snap_to_paint(img, h, spec: fieldmod.FieldSpec, rounds: int = 3,
                  search_px: int = 26, min_points: int = 12):
    """Pull a nearly-right fit onto the actual paint, and refuse to make it worse.

    A registration built from a vision reading is as precise as the reading, and a reading
    is coarse on purpose — the model is asked for fractions of the image because that is
    what it can deliver honestly. On the reference frame that leaves the drawn yard lines
    within about a yard and a half of the painted ones: obviously right to look at, and
    still a yard and a half of error on every player.

    The paint knows better. Every drawn yard line has a real one within a few pixels, so
    sampling along each drawn line, searching sideways for the brightest paint, and
    refitting to what was found converges onto the real geometry in two or three passes.
    This is the division of labour the module is built around, applied at the last step:
    the model said which line is the thirty, and the pixels say exactly where it is.

    The search window is deliberately narrower than the gap between yard lines. Wider and
    a line locks onto its neighbour, which fits beautifully and is five yards wrong — the
    same failure as everything else here, arrived at from the opposite direction.

    Returns the refined homography and its paint score, and returns the original when
    refinement does not improve it. A refinement step that can only help is a refinement
    step that has not been measured.
    """
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise ImportError("this needs opencv") from e
    from .homography import Homography
    from .lines import paint_mask, turf_mask

    paint = paint_mask(img, turf_mask(img)).astype(np.float32)
    paint = cv2.GaussianBlur(paint, (5, 5), 0)
    hgt, wid = img.shape[:2]
    best, best_score = h, paint_alignment(img, h, spec)

    for _ in range(rounds):
        fld_pts, img_pts = [], []
        for x in np.arange(spec.goal_a, spec.goal_b + 0.01, 5.0):
            ys = np.linspace(1.0, spec.width - 1.0, 14)
            fl = np.column_stack([np.full_like(ys, x), ys])
            px = best.to_image(fl)
            # Which way is "sideways" for this line in the image: along the field's
            # x direction, so the search crosses the line rather than running down it.
            px2 = best.to_image(np.column_stack([np.full_like(ys, x + 1.0), ys]))
            for f, p, q in zip(fl, px, px2):
                if not (np.isfinite(p).all() and np.isfinite(q).all()):
                    continue
                d = q - p
                n = np.linalg.norm(d)
                if n < 1e-6:
                    continue
                d = d / n
                offs = np.arange(-search_px, search_px + 1, 1.0)
                cand = p[None, :] + offs[:, None] * d[None, :]
                iu = np.round(cand[:, 0]).astype(int)
                iv = np.round(cand[:, 1]).astype(int)
                ok = (iu >= 0) & (iu < wid) & (iv >= 0) & (iv < hgt)
                if ok.sum() < 8:
                    continue
                vals = np.where(ok, paint[np.clip(iv, 0, hgt - 1), np.clip(iu, 0, wid - 1)], 0.0)
                if vals.max() < 40.0:
                    continue
                # Nearest strong peak, not the strongest: a neighbouring line is
                # brighter as often as not, and taking it moves the fit five yards.
                strong = np.flatnonzero(vals >= 0.6 * vals.max())
                pick = strong[np.argmin(np.abs(offs[strong]))]
                fld_pts.append(f)
                img_pts.append(cand[pick])

        if len(fld_pts) < min_points:
            break
        try:
            cand_h = Homography.fit(np.array(fld_pts), np.array(img_pts), threshold_px=6.0)
        except (ValueError, np.linalg.LinAlgError):
            break
        if sanity_problems(cand_h, (wid, hgt), spec):
            break
        score = paint_alignment(img, cand_h, spec)
        if not (score > best_score):
            break
        best, best_score = cand_h, score

    return best, best_score


def auto_register(img, client=None, verify: bool = True, refine_paint: bool = True,
                  **extra) -> AutoRegistration:
    """Read a frame, fit it, and check the fit — the whole loop.

    Returns an :class:`AutoRegistration` whose ``ok`` is only true when the fit both
    survives the physical sanity checks and passes the model's own look at the drawn
    result. Anything else comes back with its problems listed, for a human to take over.
    """
    from .homography import Homography

    out = AutoRegistration()
    res = read_field(img, client=client, **extra)
    if res.reading is None:
        out.problems.append(res.error or "no reading")
        return out
    out.reading = res.reading
    hgt0, wid0 = img.shape[:2]
    out.notes.extend(normalise_reading_for(res.reading, (wid0, hgt0)))
    out.notes.extend(infer_halves(res.reading))
    out.spec = spec_for(res.reading.level)
    if res.reading.level == "unclear":
        out.problems.append(
            "level of football unclear, so hash spacing was assumed NFL — "
            "worth confirming, it moves players sideways by up to six yards"
        )

    pairs, notes = to_correspondences(res.reading, img, out.spec)
    out.correspondences = pairs
    if len(pairs) < 4:
        out.problems.append(
            f"only {len(pairs)} usable correspondence(s) — "
            + ("; ".join(notes) if notes else "the frame did not give enough to work with")
        )
        return out

    fld = np.array([p[0] for p in pairs], dtype=float)
    px = np.array([p[1] for p in pairs], dtype=float)
    h = Homography.fit(fld, px, threshold_px=10.0)
    hgt0b, wid0b = img.shape[:2]
    if refine_paint and not sanity_problems(h, (wid0b, hgt0b), out.spec):
        before = paint_alignment(img, h, out.spec)
        h, after = snap_to_paint(img, h, out.spec)
        if after > before:
            out.notes.append(f"snapped to the paint: alignment {before:.2f} -> {after:.2f}")
    out.homography = h

    hgt, wid = img.shape[:2]
    out.problems.extend(sanity_problems(h, (wid, hgt), out.spec, correspondences=pairs))
    frac = turf_overlap(img, h, out.spec)
    out.turf = frac
    if frac < 0.75:
        out.problems.append(
            f"only {frac * 100:.0f}% of the projected field lands on grass — "
            "the fit has drifted off the playing surface"
        )
    # The scale check. Everything above this line is a round trip through the same map
    # and cannot see a fit that is self-consistent and the wrong size.
    align = paint_alignment(img, h, out.spec)
    out.paint = align
    if align == align and align < 1.6:
        out.problems.append(
            f"the drawn yard lines find only {align:.2f}x as much paint as empty turf — "
            "the fit is the wrong scale or offset, however well it fits its own points"
        )
    if out.problems:
        out.verdict = "failed sanity"
        return out

    if not verify:
        out.verdict = "unverified"
        return out

    from .lines import overlay as _unused  # noqa: F401  (kept for API discoverability)
    drawn = draw_model(img, h, out.spec)
    chk = check_fit(drawn, client=client, **extra)
    if chk.reading is None:
        out.verdict = "unverified"
        out.problems.append(chk.error or "verification call failed")
        return out
    out.verdict = chk.reading.verdict
    if chk.reading.verdict != "good" and chk.reading.problem:
        out.problems.append(f"the model says: {chk.reading.problem}")
    return out


def draw_model(img, h, spec: fieldmod.FieldSpec = fieldmod.NFL):
    """The field model drawn over a frame, for the verification call and for a human."""
    try:
        import cv2
    except ImportError as e:  # pragma: no cover
        raise ImportError("drawing needs opencv") from e
    v = img.copy()

    def ln(a, b, colour, thick):
        p = h.to_image([a, b])
        if np.isfinite(p).all():
            cv2.line(v, tuple(p[0].astype(int)), tuple(p[1].astype(int)), colour, thick, cv2.LINE_AA)

    for x in np.arange(spec.goal_a, spec.goal_b + 0.01, 5.0):
        ln([x, 0], [x, spec.width], (90, 255, 120), 2 if int(x) % 10 == 0 else 1)
    for x in (0, spec.goal_a, spec.goal_b, spec.length):
        ln([x, 0], [x, spec.width], (60, 60, 255), 3)
    for y in (0.0, spec.width):
        ln([0, y], [spec.length, y], (255, 100, 255), 3)
    return v
