"""The Python contract and the renderer's JavaScript one must not drift.

Two definitions of one shape is a liability; two definitions that disagree is a bug
that only shows up as a play looking wrong in the viewer. So the same plays are run
through both validators and the verdicts compared — including the wording of the
complaints, because the wording is what a person reads when something fails.
"""

import json
import shutil
import subprocess

import pytest

from conftest import CONTRACT_JS
from gridiron_tracking.contract import (
    Ball, PlayTrack, PlayerFrame, RosterEntry, TrackingFrame, validate,
)

pytestmark = pytest.mark.skipif(
    shutil.which("node") is None or not CONTRACT_JS.exists(),
    reason="needs node and the renderer prototype's contract.js",
)


def _js_validate(track: dict) -> dict:
    script = f"""
      require({str(CONTRACT_JS)!r});
      const track = JSON.parse(process.argv[1]);
      process.stdout.write(JSON.stringify(globalThis.GRID.validate(track)));
    """
    out = subprocess.run(
        ["node", "-e", script, json.dumps(track)],
        capture_output=True, text=True, check=True,
    )
    return json.loads(out.stdout)


def _good(n=12):
    roster = {"A": RosterEntry(id="A", team="offense"), "B": RosterEntry(id="B", team="defense")}
    frames = []
    for k in range(n):
        t = round(-0.5 + 0.1 * k, 3)
        frames.append(TrackingFrame(
            t=t,
            players={
                "A": PlayerFrame(x=50 + k * 0.5, y=26.0, s=5.0, dir=0.0, o=0.0, o_source="from_velocity"),
                "B": PlayerFrame(x=55.0, y=27.0, s=1.0, dir=180.0, o=180.0, o_source="assumed"),
            },
            ball=Ball(50 + k * 0.5, 26.0),
            events=["ball_snap"] if k == 5 else [],
        ))
    return PlayTrack(meta={}, roster=roster, frames=frames)


def _compare(track: PlayTrack):
    py = validate(track)
    js = _js_validate(track.to_json())
    assert py.ok == js["ok"], f"verdict differs: python {py.ok}, js {js['ok']}"
    assert py.problems == js["problems"], f"\npython: {py.problems}\njs:     {js['problems']}"
    assert py.events == js["events"]


def test_a_clean_play_passes_both():
    _compare(_good())


def test_a_play_with_no_snap_fails_both_the_same_way():
    t = _good()
    for f in t.frames:
        f.events = []
    _compare(t)


def test_a_player_off_the_field_is_counted_identically():
    t = _good()
    t.frames[3].players["A"].x = 400.0
    _compare(t)


def test_a_bad_orientation_source_is_caught_by_both():
    t = _good()
    t.frames[2].players["B"].o_source = "vibes"
    _compare(t)


def test_a_missing_player_sample_is_counted_identically():
    t = _good()
    del t.frames[4].players["A"]
    _compare(t)


def test_time_running_backwards_reads_the_same_in_both():
    t = _good()
    t.frames[6].t = t.frames[5].t - 0.1
    _compare(t)


def test_a_play_with_no_ball_fails_both():
    t = _good()
    for f in t.frames:
        f.ball = None
    _compare(t)
