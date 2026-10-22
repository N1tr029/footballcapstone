"""
Read-only access to what other streams already write to disk. The API adds no
data of its own, so Unity reads exactly the same plays whether it loads them from
StreamingAssets or from here.

PLAYS_DIR (default: Hindsight's StreamingAssets/plays) is scanned recursively for
  index.json               one per game (Hindsight schemaVersion 1)
  {gameId}_{playId}.json   one per play
To serve several games, export each into its own subfolder; the exporter clears
its output folder on every run.

Simulated plays are play files with "simulated": true and "simulatedFrom": <the
real playId>. The play model's `whatif` command writes them beside the real play
(python -m gridiron_counterfactual whatif ...). The store links each real play to
its simulated version so the API can serve the pair.

DECISIONS_PATH (default: packages/counterfactual/reports/decisions.jsonl) is the
play model's per-play reports, one JSON object per line.
"""
import json
import logging
import os
import re
from pathlib import Path
from typing import Dict, Optional

from pydantic import ValidationError

from .schema import PlayIndex

log = logging.getLogger("api.store")

REPO_ROOT = Path(__file__).resolve().parents[3]
DEFAULT_PLAYS_DIR = REPO_ROOT / "packages/replay-engine/hindsight/Assets/StreamingAssets/plays"
DEFAULT_DECISIONS = REPO_ROOT / "packages/counterfactual/reports/decisions.jsonl"
PLAY_FILE = re.compile(r"^(\d+)_(\d+)\.json$")


class PlayStore:
    def __init__(self, plays_dir: Optional[str] = None, decisions_path: Optional[str] = None):
        self.plays_dir = Path(plays_dir or os.environ.get("PLAYS_DIR") or DEFAULT_PLAYS_DIR)
        self.decisions_path = Path(decisions_path or os.environ.get("DECISIONS_PATH") or DEFAULT_DECISIONS)
        self.games: Dict[int, PlayIndex] = {}
        self.index_paths: Dict[int, Path] = {}
        self.play_files: Dict[str, Path] = {}
        self.simulated_of: Dict[str, str] = {}   # real play id -> simulated play id
        self.reports: Dict[str, dict] = {}
        self.reload()

    def reload(self) -> None:
        games: Dict[int, PlayIndex] = {}
        index_paths: Dict[int, Path] = {}
        files: Dict[str, Path] = {}
        simulated_of: Dict[str, str] = {}

        if self.plays_dir.is_dir():
            for path in sorted(self.plays_dir.rglob("*.json")):
                if path.name == "index.json":
                    try:
                        with open(path, encoding="utf-8") as f:
                            index = PlayIndex.model_validate(json.load(f))
                    except (ValidationError, json.JSONDecodeError) as e:
                        # One bad index must not take every other game offline.
                        log.warning("skipping %s: %s", path, e)
                        continue
                    games[index.gameId] = index
                    index_paths[index.gameId] = path
                elif PLAY_FILE.match(path.name):
                    files[path.stem] = path
                    real = _simulated_from(path)
                    if real is not None:
                        game_id = path.stem.split("_")[0]
                        simulated_of[f"{game_id}_{real}"] = path.stem

        reports: Dict[str, dict] = {}
        if self.decisions_path.is_file():
            with open(self.decisions_path, encoding="utf-8") as f:
                for line in f:
                    if line.strip():
                        r = json.loads(line)
                        if r.get("game_id") is not None and r.get("play_id") is not None:
                            reports[f"{int(r['game_id'])}_{int(r['play_id'])}"] = r

        # Swap in one step so a failed reload leaves the old data serving.
        self.games, self.index_paths = games, index_paths
        self.play_files, self.simulated_of, self.reports = files, simulated_of, reports

    def play_path(self, play_id: str) -> Optional[Path]:
        return self.play_files.get(play_id)

    def optimal_path(self, play_id: str) -> Optional[Path]:
        sim = self.simulated_of.get(play_id)
        return self.play_files.get(sim) if sim else None

    def report(self, play_id: str) -> Optional[dict]:
        return self.reports.get(play_id)


def _simulated_from(path: Path) -> Optional[int]:
    """The real playId a simulated play was made from, or None for a real play."""
    try:
        with open(path, encoding="utf-8") as f:
            doc = json.load(f)
    except (OSError, json.JSONDecodeError) as e:
        log.warning("unreadable play file %s: %s", path, e)
        return None
    if doc.get("simulated") and doc.get("simulatedFrom") is not None:
        return int(doc["simulatedFrom"])
    return None
