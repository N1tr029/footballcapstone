import sys
from pathlib import Path

# Neither package is installed during development; tests import both from the tree.
PACKAGES = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PACKAGES / "counterfactual"))
sys.path.insert(0, str(PACKAGES / "tracking"))
