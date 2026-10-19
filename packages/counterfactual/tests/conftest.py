import sys
from pathlib import Path

# Neither package is installed during development; both are imported from the
# tree, the same way packages/tracking/tests/conftest.py does it. Kept alongside
# pytest.ini's `pythonpath` so this suite still runs on its own.
HERE = Path(__file__).resolve()
sys.path.insert(0, str(HERE.parents[1]))
sys.path.insert(0, str(HERE.parents[3] / "packages/tracking"))

# Shared fixtures and paths live in cf_support.py — see the note there for why.
