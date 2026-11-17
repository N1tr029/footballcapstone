"""Unknown kit is uncertainty, not evidence that every later body is an opponent."""
from types import SimpleNamespace
import numpy as np

from gridiron_tracking.contract import RosterEntry
from gridiron_tracking.detect import Detection
from gridiron_tracking.identity_observations import reconcile


def test_unknown_team_can_follow_a_matching_mask_without_becoming_an_opponent():
    bodies={f:[Detection(frame=f,x1=40+f*.1,y1=10,x2=45+f*.1,y2=20,score=.9,team="kitA")]
        for f in (0,3,6)}
    masks={f:{"a":[d[0].x1,10,d[0].x2,20]} for f,d in bodies.items()}
    roster={"a":RosterEntry("a","unknown")}
    registration=SimpleNamespace(to_field=lambda f,p:np.asarray(p))
    observed,report=reconcile(masks,bodies,registration,roster,0,30)
    assert all("a" in observed[f] for f in (0,3,6))
    assert roster["a"].team=="unknown" and not report.get("opponentMasksRejected")
    # A known opposing kit is still rejected; this cannot relax existing locks.
    roster["a"].team="teamB"
    observed,report=reconcile(masks,bodies,registration,roster,0,30)
    assert "a" not in observed[3] and report["opponentMasksRejected"]==2
