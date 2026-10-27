"""The alternative, as a play the replay can draw.

A flag says "he could have thrown to the other man". This turns that sentence into a
``PlayTrack`` — the same shape the real play is in — so the replay engine can put the two
side by side on one clock.

What is real and what is not, because the drawing will not say unless this does:

- **Up to the throw, everything is the real play.** Not a frame is altered.
- **After the throw, the players are still real.** Every man keeps running exactly where
  the tracking says he ran. The alternative receiver really did run that route; nobody is
  moved to make the throw work.
- **Only the ball is invented.** It leaves the passer at the real release and flies to
  where the alternative receiver actually was when it would have arrived.
- **It stops when the ball arrives.** Whether he catches it and what he does next did not
  happen, so it is not drawn. That is a chance and a number, carried in ``meta``.

The one thing this cannot be honest about by construction: defenders would have reacted to
a ball thrown somewhere else, and here they do not. It is logged as an assumption, and it
is the reason the simulation ends at the catch point rather than playing on.
"""

from __future__ import annotations

import copy
import math

from gridiron_tracking.contract import Ball, PlayTrack, TrackingFrame

THROW_EVENTS = ("pass_forward", "pass_shovel")

DEFAULT_BALL_SPEED = 20.0
"""Yards per second along the ground, when the real throw cannot be timed. An NFL pass
travels 45-60 mph, which is 22-29 yd/s through the air and a little less along the turf."""

MAX_BALL_SPEED = 28.0

GRAVITY = 10.73          # yards per second squared
RELEASE_HEIGHT = 2.2     # yards: a quarterback's hand at release
CATCH_HEIGHT = 1.5       # yards: chest height

ASSUMPTIONS = [
    "Players are the real tracking throughout; only the ball's path after the throw is simulated.",
    "Defenders do not react to the different throw. In reality they would, so the receiver may look more open than he would have been.",
    "The ball is released at the real moment of the throw and travels at the speed of the real throw, or 20 yards a second if the real throw was softer than that.",
    "The simulation ends when the ball arrives. The catch and anything after it did not happen and is not drawn.",
]


def throw_time(track: PlayTrack) -> float | None:
    for name in THROW_EVENTS:
        t = track.event_time(name)
        if t is not None:
            return t
    return None


def _ball_xy(frame: TrackingFrame, passer: str) -> tuple[float, float] | None:
    """Where the throw starts: the passer's own position. The ball row at the frame marked
    as the throw is sometimes already a few yards downfield."""
    p = frame.players.get(passer)
    if p is not None:
        return p.x, p.y
    return (frame.ball.x, frame.ball.y) if frame.ball else None


def real_ball_speed(track: PlayTrack) -> float | None:
    """Ground speed of the pass that was actually thrown, timed over its first half second.

    Not release-to-arrival: an incomplete pass has no arrival event, and the ball's speed
    early in its flight is the arm, which is what carries over to a different target.
    """
    t0 = throw_time(track)
    if t0 is None:
        return None
    a, b = track.frame_at(t0 + 0.1), track.frame_at(t0 + 0.5)
    if not (a and b and a.ball and b.ball):
        return None
    return math.hypot(b.ball.x - a.ball.x, b.ball.y - a.ball.y) / (b.t - a.t)


def alternative(track: PlayTrack, passer: str, receiver: str, what_if: dict | None = None) -> PlayTrack:
    """The play as it would have looked with the ball thrown to ``receiver`` instead.

    Returns a new ``PlayTrack`` that is identical to ``track`` up to the throw and ends on
    the frame the ball reaches the receiver. Raises ``ValueError`` if the play has no throw
    or the ball could never have caught up with him.
    """
    t_throw = throw_time(track)
    if t_throw is None:
        raise ValueError("play has no throw event")
    start = next(i for i, f in enumerate(track.frames) if abs(f.t - t_throw) < 1e-6)
    origin = _ball_xy(track.frames[start], passer)
    if origin is None:
        raise ValueError("passer is not in the frame at the throw")

    measured = real_ball_speed(track)
    # A checkdown is lobbed; the same arm throws a deep ball harder. So the real throw's
    # speed is a floor on nothing and a ceiling on nothing — it is used when it was a
    # proper throw, and the default when the real one was soft.
    speed = min(max(measured or 0.0, DEFAULT_BALL_SPEED), MAX_BALL_SPEED)

    # The ball arrives on the first frame where it could have covered the distance to
    # wherever the receiver really was by then.
    arrive = None
    for i in range(start + 1, len(track.frames)):
        p = track.frames[i].players.get(receiver)
        if p is None:
            continue
        if math.hypot(p.x - origin[0], p.y - origin[1]) <= speed * (track.frames[i].t - t_throw):
            arrive = i
            break
    if arrive is None:
        raise ValueError("the ball could not have reached this receiver before the play ended")

    target = track.frames[arrive].players[receiver]
    flight = track.frames[arrive].t - t_throw
    # A thrown ball is a parabola: the vertical speed that gets it from the hand to the
    # chest in exactly `flight` seconds under gravity.
    vz = (CATCH_HEIGHT - RELEASE_HEIGHT + 0.5 * GRAVITY * flight * flight) / flight

    frames = [copy.deepcopy(f) for f in track.frames[: arrive + 1]]
    for i in range(start, arrive + 1):
        f = frames[i]
        dt = f.t - t_throw
        u = dt / flight
        f.ball = Ball(
            x=origin[0] + (target.x - origin[0]) * u,
            y=origin[1] + (target.y - origin[1]) * u,
            z=round(RELEASE_HEIGHT + vz * dt - 0.5 * GRAVITY * dt * dt, 3),
        )
        # The real play's later events (its own arrival, catch, tackle) did not happen here.
        f.events = [e for e in f.events if i == start and e in THROW_EVENTS]
    frames[arrive].events = ["pass_arrived"]

    meta = dict(track.meta)
    meta.update(
        # "simulated" is the repository's word for it: gridiron_tracking.hindsight
        # keys the Unity export's `simulated` flag on it, so the app can never show
        # this as a measured play. `simulated_from_play` names the real play it argues with.
        source="simulated", kind="receiver_what_if", simulated=True, simulated_from_t=t_throw,
        simulated_from_play=track.meta.get("play_id"),
        simulated_passer=passer, simulated_receiver=receiver,
        ball_speed=round(speed, 1), ball_speed_source="real throw" if measured and measured >= DEFAULT_BALL_SPEED else "default",
        flight_seconds=round(flight, 1), assumptions=list(ASSUMPTIONS),
    )
    if what_if is not None:
        meta["what_if"] = what_if
    return PlayTrack(meta=meta, roster=copy.deepcopy(track.roster), frames=frames)
