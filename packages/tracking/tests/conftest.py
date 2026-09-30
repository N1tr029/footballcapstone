import sys
from pathlib import Path

# The package is not installed during development; tests import it from the tree.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

REPO = Path(__file__).resolve().parents[3]
CONTRACT_JS = REPO / "packages/replay-engine/prototype/engine/contract.js"
