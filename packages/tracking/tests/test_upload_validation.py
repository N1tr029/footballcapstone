"""A completed API job alone does not establish successful tracking."""
import importlib.util
from pathlib import Path
from copy import deepcopy


path=Path(__file__).resolve().parents[3]/"scripts/validate_video_uploads.py"
spec=importlib.util.spec_from_file_location("validate_video_uploads",path)
audit=importlib.util.module_from_spec(spec);spec.loader.exec_module(audit)


def test_a_completed_job_without_tracking_is_a_failed_validation():
    assert audit.failed({"state":"done","plays":[{"hasTrack":False}]})
    assert audit.failed({"state":"done","plays":[]})
    assert not audit.failed({"state":"done","plays":[{"hasTrack":True,"issues":[]}]})


def test_audit_catches_dropped_players_team_swaps_and_teleports():
    base={"players":[{"nflId":1,"club":"A"},{"nflId":2,"club":"B"}],"frameRate":10,
        "frames":[{"frameId":i,"players":[{"nflId":1,"club":"A","x":40+i*.2,"y":20,"observed":True},
            {"nflId":2,"club":"B","x":42,"y":20,"observed":False}]} for i in range(3)]}
    assert audit.replay_checks(base)["issues"]==[]
    broken=deepcopy(base);broken["frames"][0]["players"].pop()
    broken["frames"][1]["players"][0].update(x=80,club="B")
    issues=audit.replay_checks(broken)["issues"]
    assert any("membership" in p for p in issues)
    assert any("team changed" in p for p in issues)
    assert any("display jump" in p for p in issues)
