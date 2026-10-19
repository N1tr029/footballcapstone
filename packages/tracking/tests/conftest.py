import sys
from pathlib import Path

# The package is not installed during development; tests import it from the tree.
# Kept alongside pytest.ini's `pythonpath` so this suite still runs on its own.
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Paths that tests need at import time live in tracking_support.py — see the note
# there for why they are not in this file.
