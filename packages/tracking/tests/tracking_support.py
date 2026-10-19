"""Paths the tracking suite needs at import time.

Not in conftest.py: ``packages/counterfactual/tests`` has a conftest.py too, and a
test module that says ``from conftest import ...`` gets whichever of the two landed
on sys.path first. Each suite passed alone and the pair failed to collect. A
distinct module name has no such ambiguity.
"""

from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
CONTRACT_JS = REPO / "packages/replay-engine/prototype/engine/contract.js"

# The Unity app's real play files, merged in from hindsight-frontend. The suite
# that uses them skips when the subtree is absent, so a checkout without it still
# runs green rather than erroring.
HINDSIGHT_PLAYS = REPO / "packages/replay-engine/hindsight/Assets/StreamingAssets/plays"
