"""Paths the tracking suite needs at import time.

Not in conftest.py: ``packages/counterfactual/tests`` has a conftest.py too, and a
test module that says ``from conftest import ...`` gets whichever of the two landed
on sys.path first. Each suite passed alone and the pair failed to collect. A
distinct module name has no such ambiguity.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CONTRACT_JS = REPO / "packages/replay-engine/prototype/engine/contract.js"

# The Unity app's real play files. The app itself lives in EllisJeremy/hindsight-frontend
# and is no longer part of this repository, so its exported plays are committed here
# instead: they are the only real-play fixture the suite has. Read-only — the CLIs that
# write plays default to data/unity-plays so a run cannot edit a committed fixture.
HINDSIGHT_PLAYS = REPO / "tests/fixtures/hindsight-plays"
