"""The simulated alternative: what is real, what is invented, and where it stops."""

import math

import pytest

from gridiron_tracking.contract import Ball, PlayerFrame, PlayTrack, RosterEntry, TrackingFrame, validate

from gridiron_counterfactual import simulate


def _track(receiver_speed=5.0, frames=40):
    """QB stands at (30, 26). The real pass goes to WR1; WR2 runs straight downfield."""
    roster = {i: RosterEntry(id=i, team="offense") for i in ("qb", "wr1", "wr2")}
    out = []
    for k in range(frames):
        t = round(k * 0.1, 1)
        players = {
            "qb": PlayerFrame(30.0, 26.0, o_source="measured"),
            "wr1": PlayerFrame(40.0, 10.0, o_source="measured"),
            "wr2": PlayerFrame(40.0 + receiver_speed * t, 40.0, s=receiver_speed, o_source="measured"),
        }
        events = {0: ["ball_snap"], 10: ["pass_forward"], 20: ["pass_arrived"], 30: ["tackle"]}.get(k, [])
        # The real ball: with the QB until the throw, then to WR1 over one second.
        u = min(max((t - 1.0) / 1.0, 0.0), 1.0)
        ball = Ball(30.0 + 10.0 * u, 26.0 - 16.0 * u)
        out.append(TrackingFrame(t=t, players=players, ball=ball, events=events))
    return PlayTrack(meta={"play_id": 1}, roster=roster, frames=out)


def test_nothing_before_the_throw_is_changed():
    real = _track()
    alt = simulate.alternative(real, "qb", "wr2")
    for a, b in zip(real.frames[:10], alt.frames[:10]):
        assert a.to_json() == b.to_json()


def test_players_stay_real_and_only_the_ball_moves():
    real = _track()
    alt = simulate.alternative(real, "qb", "wr2")
    for a, b in zip(real.frames, alt.frames):
        assert {k: v.to_json() for k, v in a.players.items()} == {k: v.to_json() for k, v in b.players.items()}
    assert alt.frames[-1].ball.y == pytest.approx(40.0)          # went to WR2, not WR1 at y = 10


def test_the_ball_arrives_where_the_receiver_really_was_and_the_play_stops_there():
    real = _track()
    alt = simulate.alternative(real, "qb", "wr2")
    last = alt.frames[-1]
    wr = last.players["wr2"]
    assert (last.ball.x, last.ball.y) == pytest.approx((wr.x, wr.y))
    assert last.events == ["pass_arrived"]
    assert len(alt.frames) < len(real.frames)                    # no catch, no tackle
    assert all("tackle" not in f.events for f in alt.frames)


def test_the_ball_flies_at_a_throwable_speed_and_as_a_parabola():
    real = _track()
    alt = simulate.alternative(real, "qb", "wr2")
    assert real.frames[10].ball is not None
    # The real throw was 18.9 yd/s, softer than the default, so the default is used.
    assert alt.meta["ball_speed"] == simulate.DEFAULT_BALL_SPEED and alt.meta["ball_speed_source"] == "default"
    flight = [f for f in alt.frames if f.t >= 1.0]
    heights = [f.ball.z for f in flight]
    assert heights[0] == pytest.approx(simulate.RELEASE_HEIGHT) and heights[-1] == pytest.approx(simulate.CATCH_HEIGHT, abs=0.01)
    assert max(heights) > simulate.RELEASE_HEIGHT                # it goes up before it comes down
    # Constant ground speed, never faster than the throw could carry it.
    steps = [math.hypot(b.ball.x - a.ball.x, b.ball.y - a.ball.y) for a, b in zip(flight, flight[1:])]
    assert max(steps) == pytest.approx(min(steps)) and max(steps) / 0.1 <= alt.meta["ball_speed"] + 1e-6


def test_it_is_labelled_and_still_a_valid_play():
    alt = simulate.alternative(_track(), "qb", "wr2", what_if={"kind": "target_choice"})
    assert alt.meta["simulated"] is True and alt.meta["source"] == "what_if"
    assert alt.meta["assumptions"] and alt.meta["what_if"]["kind"] == "target_choice"
    assert validate(alt).ok


def test_a_receiver_the_ball_can_never_catch_is_refused():
    with pytest.raises(ValueError, match="could not have reached"):
        simulate.alternative(_track(receiver_speed=60.0), "qb", "wr2")


def test_a_play_with_no_throw_is_refused():
    track = _track()
    for f in track.frames:
        f.events = [e for e in f.events if e != "pass_forward"]
    with pytest.raises(ValueError, match="no throw"):
        simulate.alternative(track, "qb", "wr2")
